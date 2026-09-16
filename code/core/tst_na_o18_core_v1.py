#!/usr/bin/env python3
"""CPU-testable common core for TsT-NativeAdapter-O18 (TsT-NA).

This file contains no dataset reader, VLM loader, runner, optimizer policy, or
validation access.  It freezes three interfaces shared by family backends:

1. source-separated ten-unit physical adapters and O18 prediction heads;
2. a primary dual-source LoRA wrapper for *real native ``nn.Linear`` modules*;
3. native-LM answer-score utilities which derive factor margins and four-state
   composition from canonical answer sequence log-probabilities.

The core never creates a factor classifier.  Camera/object factors, states, and
language results are properties of the family's original LM decoder and head.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import math
from typing import Final, Iterable

import torch
from torch import nn
import torch.nn.functional as F


TEMPORAL_UNITS: Final[int] = 10
CAMERA_TRANSLATION_DIM: Final[int] = 9
CAMERA_ROTATION_DIM: Final[int] = 9
CAMERA18_DIM: Final[int] = 18
OBJECT_TRANSLATION_DIM: Final[int] = 9
OBJECT_ROTATION_DIM: Final[int] = 9
OBJECT18_DIM: Final[int] = 18

CAMERA_TRANSLATION_SCALE_M: Final[float] = 0.5
CAMERA_ROTATION_SCALE_RAD: Final[float] = math.radians(5.0)
OBJECT_TRANSLATION_SCALE_M: Final[float] = 1.0
OBJECT_ROTATION_SCALE_RAD: Final[float] = 1.0122909661567112
SOURCE_GATE_COMPUTE_DTYPE: Final[torch.dtype] = torch.float32

# Canonical native answer sequence score order.  This order is frozen and is
# not the output of a learned four-way head.
FOUR_STATE_ORDER: Final[tuple[str, ...]] = (
    "neither",
    "camera_only",
    "object_only",
    "both",
)

STAGE_P_MODULES: Final[tuple[str, ...]] = (
    "camera_physical_adapter",
    "object_physical_adapter",
    "camera18_head",
    "object18_translation_head",
    "object18_rotation_head",
)

MASKED_ROTATION_RUNNER_INVARIANT: Final[str] = (
    "Every optimizer step must start with optimizer.zero_grad(set_to_none=True); "
    "when object-rotation contributing_cases is zero, every parameter in "
    "object18_rotation_head must retain grad is None before optimizer.step()."
)


@dataclass(frozen=True, slots=True)
class O18PredictionContract:
    schema_version: str = "tst_na_o18_prediction_contract_v1"
    camera_translation_unit: str = "m"
    camera_rotation_unit: str = "rad_rotvec"
    object_translation_unit: str = "m"
    object_rotation_unit: str = "rad_rotvec"
    camera_translation_reference: str = "final_representative_origin_final_camera_axes"
    camera_rotation_reference: str = "final_camera_from_anchor_camera_rotvec"
    object_translation_reference: str = "first_representative_origin_final_camera_axes"
    object_rotation_reference: str = "spatial_first_anchor_relative_final_camera_rotvec"
    camera_translation_scale_m: float = CAMERA_TRANSLATION_SCALE_M
    camera_rotation_scale_rad: float = CAMERA_ROTATION_SCALE_RAD
    object_translation_scale_m: float = OBJECT_TRANSLATION_SCALE_M
    object_rotation_scale_rad: float = OBJECT_ROTATION_SCALE_RAD
    anchor_mask_dtype: str = "torch.bool"
    anchor_mask_shape: str = "[B,3]"
    camera_translation_mask_field: str = "camera_translation_valid"
    camera_rotation_mask_field: str = "camera_rotation_valid"
    object_translation_mask_field: str = "object_translation_valid"
    object_rotation_mask_field: str = "object_rotation_valid"
    mask_semantics: str = (
        "supervision/evaluation validity only; masks are never inference inputs"
    )


O18_PREDICTION_CONTRACT: Final[O18PredictionContract] = O18PredictionContract()


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_finite(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _require_float_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def _require_source_sequence(
    value: torch.Tensor,
    *,
    name: str,
    feature_dim: int,
    temporal_units: int = TEMPORAL_UNITS,
) -> None:
    _require_float_tensor(value, name)
    if value.ndim != 3 or tuple(value.shape[1:]) != (temporal_units, feature_dim):
        raise ValueError(
            f"{name} must have shape [B,{temporal_units},{feature_dim}], "
            f"got {tuple(value.shape)}"
        )
    _require_finite(value, name)


def _require_matrix(value: torch.Tensor, *, name: str, width: int) -> None:
    _require_float_tensor(value, name)
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(
            f"{name} must have shape [B,{width}], got {tuple(value.shape)}"
        )


def _require_anchor_mask(
    value: torch.Tensor,
    *,
    batch: int,
    name: str,
    require_reference_for_later: bool = True,
) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.bool
        or tuple(value.shape) != (batch, 3)
    ):
        raise TypeError(f"{name} must be boolean [B,3]")
    if require_reference_for_later and bool((value[:, 1:] & ~value[:, :1]).any()):
        raise ValueError(
            f"{name}: later valid anchors require a valid reference anchor"
        )


class TemporalPhysicalAdapter(nn.Module):
    """A source-local temporal bottleneck over exactly ten physical units."""

    def __init__(self, input_dim: int, latent_dim: int, rank: int) -> None:
        super().__init__()
        self.input_dim = _positive_int(input_dim, "input_dim")
        self.latent_dim = _positive_int(latent_dim, "latent_dim")
        self.rank = _positive_int(rank, "rank")
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.down = nn.Linear(2 * self.input_dim, self.rank)
        self.up = nn.Linear(self.rank, self.latent_dim)
        self.time_embedding = nn.Parameter(torch.empty(TEMPORAL_UNITS, self.latent_dim))
        self.temporal_mixer = nn.Linear(TEMPORAL_UNITS, TEMPORAL_UNITS)
        self.output_norm = nn.LayerNorm(self.latent_dim)
        self.temporal_logits = nn.Parameter(torch.empty(TEMPORAL_UNITS))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.input_norm.reset_parameters()
        self.down.reset_parameters()
        self.up.reset_parameters()
        self.temporal_mixer.reset_parameters()
        self.output_norm.reset_parameters()
        nn.init.normal_(self.time_embedding, mean=0.0, std=0.02)
        with torch.no_grad():
            self.temporal_logits.copy_(torch.linspace(-0.02, 0.02, TEMPORAL_UNITS))

    def forward(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _require_source_sequence(
            source, name="physical source", feature_dim=self.input_dim
        )
        normalized = self.input_norm(source)
        delta = torch.zeros_like(normalized)
        delta[:, 1:] = normalized[:, 1:] - normalized[:, :-1]
        bottleneck = F.gelu(self.down(torch.cat((normalized, delta), dim=-1)))
        latent = self.up(bottleneck) + self.time_embedding.unsqueeze(0)
        mixed = self.temporal_mixer(latent.transpose(1, 2)).transpose(1, 2)
        sequence = self.output_norm(latent + F.gelu(mixed))
        weights = torch.softmax(self.temporal_logits, dim=0)
        summary = torch.sum(sequence * weights.view(1, -1, 1), dim=1)
        return sequence, summary


class TrajectoryComponentHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 9) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(_positive_int(input_dim, "input_dim"))
        self.hidden = nn.Linear(input_dim, _positive_int(hidden_dim, "hidden_dim"))
        self.output = nn.Linear(hidden_dim, _positive_int(output_dim, "output_dim"))

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        _require_float_tensor(summary, "trajectory summary")
        if summary.ndim != 2:
            raise ValueError("trajectory summary must have shape [B,D]")
        _require_finite(summary, "trajectory summary")
        return self.output(F.silu(self.hidden(self.norm(summary))))


class Camera18Head(nn.Module):
    """Explicit Camera translation9 and rotation9 normalized heads."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.translation9 = TrajectoryComponentHead(input_dim, hidden_dim, 9)
        self.rotation9 = TrajectoryComponentHead(input_dim, hidden_dim, 9)

    def forward(self, summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.translation9(summary), self.rotation9(summary)


@dataclass(frozen=True, slots=True)
class PhysicalOutput:
    camera_source: torch.Tensor
    object_source: torch.Tensor
    camera_summary: torch.Tensor
    object_summary: torch.Tensor
    camera_translation9_normalized: torch.Tensor
    camera_rotation9_normalized: torch.Tensor
    camera18_normalized: torch.Tensor
    camera_translation9_m: torch.Tensor
    camera_rotation9_rad: torch.Tensor
    camera18_physical: torch.Tensor
    object_translation9_normalized: torch.Tensor
    object_rotation9_normalized: torch.Tensor
    object_translation9_m: torch.Tensor
    object_rotation9_rad: torch.Tensor
    object18_physical: torch.Tensor
    prediction_contract: O18PredictionContract


class TsTNativeAdapterO18Core(nn.Module):
    """Only the Stage-P physical core; it owns no Stage-L read/factor heads."""

    def __init__(
        self,
        feature_dim: int,
        *,
        physical_dim: int = 64,
        physical_rank: int = 32,
        head_hidden_dim: int = 64,
        temporal_units: int = TEMPORAL_UNITS,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_int(feature_dim, "feature_dim")
        self.physical_dim = _positive_int(physical_dim, "physical_dim")
        if temporal_units != TEMPORAL_UNITS:
            raise ValueError("TsT-NA V1 requires exactly 10 temporal units")
        self.camera_physical_adapter = TemporalPhysicalAdapter(
            self.feature_dim, self.physical_dim, physical_rank
        )
        self.object_physical_adapter = TemporalPhysicalAdapter(
            self.feature_dim, self.physical_dim, physical_rank
        )
        self.camera18_head = Camera18Head(self.physical_dim, head_hidden_dim)
        self.object18_translation_head = TrajectoryComponentHead(
            self.physical_dim, head_hidden_dim, 9
        )
        self.object18_rotation_head = TrajectoryComponentHead(
            self.physical_dim, head_hidden_dim, 9
        )
        self._active_stage = "P"
        self.configure_stage("P")

    @property
    def active_stage(self) -> str:
        return self._active_stage

    @property
    def prediction_contract(self) -> O18PredictionContract:
        return O18_PREDICTION_CONTRACT

    def configure_stage(self, stage: str) -> tuple[str, ...]:
        """Configure core parameters; Stage-L intentionally freezes all core state."""
        if not isinstance(stage, str):
            raise TypeError("stage must be 'P', 'L', or 'FROZEN'")
        normalized = stage.strip().upper().replace("STAGE-", "")
        if normalized not in {"P", "L", "FROZEN"}:
            raise ValueError("stage must be 'P', 'L', or 'FROZEN'")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if normalized == "P":
            for module_name in STAGE_P_MODULES:
                getattr(self, module_name).requires_grad_(True)
        self._active_stage = normalized
        actual = self.trainable_parameter_names()
        if set(actual) != set(self.expected_trainable_parameter_names(normalized)):
            raise AssertionError("core requires_grad set differs from stage allowlist")
        return actual

    def expected_trainable_parameter_names(self, stage: str) -> tuple[str, ...]:
        normalized = stage.strip().upper().replace("STAGE-", "")
        if normalized == "P":
            prefixes = STAGE_P_MODULES
        elif normalized in {"L", "FROZEN"}:
            prefixes = ()
        else:
            raise ValueError("stage must be 'P', 'L', or 'FROZEN'")
        return tuple(
            name
            for name, _ in self.named_parameters()
            if any(
                name == prefix or name.startswith(prefix + ".") for prefix in prefixes
            )
        )

    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        )

    def physical_forward(
        self,
        target_features: torch.Tensor,
        context_features: torch.Tensor,
    ) -> PhysicalOutput:
        _require_source_sequence(
            target_features, name="target features", feature_dim=self.feature_dim
        )
        _require_source_sequence(
            context_features, name="context features", feature_dim=self.feature_dim
        )
        if target_features.shape[0] != context_features.shape[0]:
            raise ValueError("target and context feature batch sizes differ")
        camera_source, camera_summary = self.camera_physical_adapter(context_features)
        object_source, object_summary = self.object_physical_adapter(target_features)
        camera_translation, camera_rotation = self.camera18_head(camera_summary)
        object_translation = self.object18_translation_head(object_summary)
        object_rotation = self.object18_rotation_head(object_summary)
        camera18_normalized = torch.cat((camera_translation, camera_rotation), dim=-1)
        camera_translation_m = camera_translation * CAMERA_TRANSLATION_SCALE_M
        camera_rotation_rad = camera_rotation * CAMERA_ROTATION_SCALE_RAD
        camera18_physical = torch.cat(
            (camera_translation_m, camera_rotation_rad), dim=-1
        )
        object_translation_m = object_translation * OBJECT_TRANSLATION_SCALE_M
        object_rotation_rad = object_rotation * OBJECT_ROTATION_SCALE_RAD
        object18_physical = torch.cat(
            (object_translation_m, object_rotation_rad), dim=-1
        )
        if camera18_normalized.shape[-1] != CAMERA18_DIM:
            raise AssertionError("Camera18 ABI must be translation9 + rotation9")
        if object18_physical.shape[-1] != OBJECT18_DIM:
            raise AssertionError("Object18 ABI must be translation9 + rotation9")
        return PhysicalOutput(
            camera_source=camera_source,
            object_source=object_source,
            camera_summary=camera_summary,
            object_summary=object_summary,
            camera_translation9_normalized=camera_translation,
            camera_rotation9_normalized=camera_rotation,
            camera18_normalized=camera18_normalized,
            camera_translation9_m=camera_translation_m,
            camera_rotation9_rad=camera_rotation_rad,
            camera18_physical=camera18_physical,
            object_translation9_normalized=object_translation,
            object_rotation9_normalized=object_rotation,
            object_translation9_m=object_translation_m,
            object_rotation9_rad=object_rotation_rad,
            object18_physical=object18_physical,
            prediction_contract=self.prediction_contract,
        )


@dataclass(frozen=True, slots=True)
class DualSourceResidualOutput:
    base_output: torch.Tensor
    camera_residual: torch.Tensor
    object_residual: torch.Tensor
    output: torch.Tensor


class NativeSourceReadLowRankResidual(nn.Module):
    """Engineering-only block-output fallback; this is **not** classic LoRA.

    It may support API diagnosis when a family Linear cannot yet be wrapped.
    It is not instantiated by :class:`TsTNativeAdapterO18Core`, is not included
    in the primary Stage-L allowlist, and outputs only a source residual.
    """

    engineering_fallback_only: Final[bool] = True

    def __init__(self, hidden_dim: int, source_dim: int, rank: int) -> None:
        super().__init__()
        self.hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        self.source_dim = _positive_int(source_dim, "source_dim")
        self.rank = _positive_int(rank, "rank")
        self.hidden_down = nn.Linear(self.hidden_dim, self.rank, bias=False)
        self.source_gate = nn.Linear(self.source_dim, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.hidden_down.weight)
        nn.init.xavier_uniform_(self.source_gate.weight)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.01)

    def forward(self, hidden: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        _require_float_tensor(hidden, "native hidden")
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_dim:
            raise ValueError(f"native hidden must have shape [B,S,{self.hidden_dim}]")
        _require_finite(hidden, "native hidden")
        _require_source_sequence(
            source, name="fallback source", feature_dim=self.source_dim
        )
        if hidden.shape[0] != source.shape[0]:
            raise ValueError("native hidden and fallback source batch sizes differ")
        source_summary = source.detach().mean(dim=1)
        gate = torch.tanh(self.source_gate(source_summary)).unsqueeze(1)
        return self.up(self.hidden_down(hidden) * gate) / math.sqrt(float(self.rank))


def native_read_residuals_engineering_fallback(
    hidden: torch.Tensor,
    camera_source: torch.Tensor,
    object_source: torch.Tensor,
    camera_reader: NativeSourceReadLowRankResidual,
    object_reader: NativeSourceReadLowRankResidual,
) -> DualSourceResidualOutput:
    """Return two inspectable residuals; never factor logits or state scores."""
    camera = camera_reader(hidden, camera_source)
    obj = object_reader(hidden, object_source)
    return DualSourceResidualOutput(
        base_output=hidden,
        camera_residual=camera,
        object_residual=obj,
        output=hidden + camera + obj,
    )


class SourceConditionedDualLoRALinear(nn.Module):
    """Primary Stage-L wrapper around one real native ``nn.Linear``.

    ``W0`` is the exact wrapped Linear and is permanently frozen.  Camera and
    Object each own a true low-rank ``B(A(x))`` update, modulated in rank space
    by a bias-free gate derived from only that source.  An all-zero source gives
    an exactly zero corresponding update for every parameter value.

    Family backends bind detached physical source sequences with
    :meth:`source_context` while the unmodified native block calls ``forward(x)``.
    Suitable targets include declared q/v/attention projection Linear modules.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        source_dim: int,
        rank: int,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be an nn.Linear")
        self.base = base
        self.source_dim = _positive_int(source_dim, "source_dim")
        self.rank = _positive_int(rank, "rank")
        self.alpha = _positive_finite(alpha, "alpha")
        self.scaling = self.alpha / float(self.rank)
        self.camera_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.camera_B = nn.Linear(self.rank, base.out_features, bias=False)
        self.camera_source_gate = nn.Linear(self.source_dim, self.rank, bias=False)
        self.object_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.object_B = nn.Linear(self.rank, base.out_features, bias=False)
        self.object_source_gate = nn.Linear(self.source_dim, self.rank, bias=False)
        self._camera_source: torch.Tensor | None = None
        self._object_source: torch.Tensor | None = None
        self._reset_adapter_parameters()
        self._move_adapters_to_base()
        self.set_adapter_trainable(False)

    def _reset_adapter_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.camera_A.weight, a=math.sqrt(5.0))
        nn.init.kaiming_uniform_(self.object_A.weight, a=math.sqrt(5.0))
        nn.init.normal_(self.camera_B.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.object_B.weight, mean=0.0, std=0.01)
        nn.init.xavier_uniform_(self.camera_source_gate.weight)
        nn.init.xavier_uniform_(self.object_source_gate.weight)

    def _move_adapters_to_base(self) -> None:
        kwargs = {"device": self.base.weight.device, "dtype": self.base.weight.dtype}
        for module in self.adapter_modules():
            module.to(**kwargs)

    def adapter_modules(self) -> tuple[nn.Linear, ...]:
        return (
            self.camera_A,
            self.camera_B,
            self.camera_source_gate,
            self.object_A,
            self.object_B,
            self.object_source_gate,
        )

    def requires_grad_(
        self, requires_grad: bool = True
    ) -> SourceConditionedDualLoRALinear:
        """Honor generic module toggles while making wrapped ``W0`` immutable."""
        if not isinstance(requires_grad, bool):
            raise TypeError("requires_grad must be a bool")
        super().requires_grad_(requires_grad)
        self.base.requires_grad_(False)
        return self

    def adapter_named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        adapter_prefixes = (
            "camera_A.",
            "camera_B.",
            "camera_source_gate.",
            "object_A.",
            "object_B.",
            "object_source_gate.",
        )
        for name, parameter in self.named_parameters():
            if name.startswith(adapter_prefixes):
                yield name, parameter

    def set_adapter_trainable(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        self.base.requires_grad_(False)
        for _, parameter in self.adapter_named_parameters():
            parameter.requires_grad_(enabled)
        self.assert_base_frozen()

    def assert_base_frozen(self) -> None:
        if any(parameter.requires_grad for parameter in self.base.parameters()):
            raise AssertionError("wrapped native base Linear W0 must remain frozen")

    @contextmanager
    def source_context(
        self,
        camera_source: torch.Tensor,
        object_source: torch.Tensor,
    ) -> Iterator[None]:
        _require_source_sequence(
            camera_source, name="Camera LoRA source", feature_dim=self.source_dim
        )
        _require_source_sequence(
            object_source, name="Object LoRA source", feature_dim=self.source_dim
        )
        if camera_source.shape[0] != object_source.shape[0]:
            raise ValueError("Camera/Object LoRA source batch sizes differ")
        camera_weight = self.camera_source_gate.weight
        object_weight = self.object_source_gate.weight
        if (
            camera_weight.device != object_weight.device
            or camera_weight.dtype != object_weight.dtype
        ):
            raise RuntimeError("Camera/Object LoRA source gates differ in device/dtype")
        previous = (self._camera_source, self._object_source)
        # Physical sources are FP32 masters.  Preserve that precision for all
        # families instead of quantizing Qwen/Molmo sources to BF16 and then
        # widening them again inside the explicit FP32 gate precision island.
        # The returned gate alone is cast to the adapter master dtype.
        self._camera_source = camera_source.detach().to(
            device=camera_weight.device,
            dtype=SOURCE_GATE_COMPUTE_DTYPE,
        )
        self._object_source = object_source.detach().to(
            device=object_weight.device,
            dtype=SOURCE_GATE_COMPUTE_DTYPE,
        )
        try:
            yield
        finally:
            self._camera_source, self._object_source = previous

    def _source_gate(self, source: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        # Fixed mean pooling has no unregistered trainable temporal slot.  Keep
        # the reduction, projection, and tanh autograd path in FP32 even when
        # the adapter master is BF16/FP16.  A low-precision tanh can round a
        # finite preactivation near |4| to exact +/-1, which makes its stored
        # backward Jacobian exactly zero.  Casting only the returned gate keeps
        # the native residual dtype unchanged while retaining the finite FP32
        # derivative for the source-gate weight.
        # NVILA runs its native decoder under FP16 autocast, so explicit FP32
        # tensors alone are insufficient: autocast would otherwise select a
        # low-precision Linear kernel again.
        with torch.autocast(device_type=source.device.type, enabled=False):
            source_summary = source.to(dtype=SOURCE_GATE_COMPUTE_DTYPE).mean(dim=1)
            weight = projection.weight.to(dtype=SOURCE_GATE_COMPUTE_DTYPE)
            bias = (
                projection.bias.to(dtype=SOURCE_GATE_COMPUTE_DTYPE)
                if projection.bias is not None
                else None
            )
            gate = torch.tanh(F.linear(source_summary, weight, bias))
        return gate.to(dtype=projection.weight.dtype)

    def forward_with_components(self, hidden: torch.Tensor) -> DualSourceResidualOutput:
        self.assert_base_frozen()
        _require_float_tensor(hidden, "wrapped Linear input")
        if hidden.ndim < 2 or hidden.shape[-1] != self.base.in_features:
            raise ValueError(
                f"wrapped Linear input must end in {self.base.in_features}"
            )
        _require_finite(hidden, "wrapped Linear input")
        if self._camera_source is None or self._object_source is None:
            raise RuntimeError("dual LoRA source_context must be active")
        batch = hidden.shape[0]
        if (
            self._camera_source.shape[0] != batch
            or self._object_source.shape[0] != batch
        ):
            raise ValueError("wrapped Linear input/source batch sizes differ")
        base_output = self.base(hidden)
        broadcast_shape = (batch,) + (1,) * (hidden.ndim - 2) + (self.rank,)
        camera_gate = self._source_gate(
            self._camera_source, self.camera_source_gate
        ).reshape(broadcast_shape)
        object_gate = self._source_gate(
            self._object_source, self.object_source_gate
        ).reshape(broadcast_shape)
        camera_residual = (
            self.camera_B(self.camera_A(hidden) * camera_gate) * self.scaling
        )
        object_residual = (
            self.object_B(self.object_A(hidden) * object_gate) * self.scaling
        )
        return DualSourceResidualOutput(
            base_output=base_output,
            camera_residual=camera_residual,
            object_residual=object_residual,
            output=base_output + camera_residual + object_residual,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.forward_with_components(hidden).output


def _validated_wrapper_mapping(
    wrappers: Mapping[str, SourceConditionedDualLoRALinear],
    *,
    require_nonempty: bool,
) -> tuple[tuple[str, SourceConditionedDualLoRALinear], ...]:
    if not isinstance(wrappers, Mapping):
        raise TypeError(
            "wrappers must be a name -> SourceConditionedDualLoRALinear mapping"
        )
    items = tuple(wrappers.items())
    if require_nonempty and not items:
        raise ValueError("Stage-L requires at least one bound native Linear wrapper")
    if any(not isinstance(name, str) or not name.strip() for name, _ in items):
        raise ValueError("every wrapper binding requires a non-empty name")
    if any(
        not isinstance(wrapper, SourceConditionedDualLoRALinear) for _, wrapper in items
    ):
        raise TypeError("wrapper mapping contains an incompatible module")
    identities = [id(wrapper) for _, wrapper in items]
    if len(identities) != len(set(identities)):
        raise ValueError(
            "one wrapper instance cannot occupy multiple declared bindings"
        )
    return items


def configure_stage_p(
    core: TsTNativeAdapterO18Core,
    wrappers: Mapping[str, SourceConditionedDualLoRALinear] | None = None,
) -> tuple[str, ...]:
    """Enable only physical parameters and disable every supplied native wrapper."""
    if not isinstance(core, TsTNativeAdapterO18Core):
        raise TypeError("core must be TsTNativeAdapterO18Core")
    items = _validated_wrapper_mapping(wrappers or {}, require_nonempty=False)
    for _, wrapper in items:
        wrapper.set_adapter_trainable(False)
    return core.configure_stage("P")


def configure_stage_l(
    core: TsTNativeAdapterO18Core,
    wrappers: Mapping[str, SourceConditionedDualLoRALinear],
) -> tuple[str, ...]:
    """Freeze Stage-P and enable exactly the adapters in actual native wrappers."""
    if not isinstance(core, TsTNativeAdapterO18Core):
        raise TypeError("core must be TsTNativeAdapterO18Core")
    items = _validated_wrapper_mapping(wrappers, require_nonempty=True)
    core.configure_stage("L")
    names: list[str] = []
    for binding, wrapper in items:
        wrapper.set_adapter_trainable(True)
        for local_name, parameter in wrapper.adapter_named_parameters():
            if not parameter.requires_grad:
                raise AssertionError("declared Stage-L adapter parameter is frozen")
            names.append(f"{binding}.{local_name}")
    return tuple(names)


def stage_l_named_parameters(
    wrappers: Mapping[str, SourceConditionedDualLoRALinear],
) -> Iterator[tuple[str, nn.Parameter]]:
    """Yield only actual wrapper adapter parameters; never wrapped W0 tensors."""
    for binding, wrapper in _validated_wrapper_mapping(wrappers, require_nonempty=True):
        wrapper.assert_base_frozen()
        for local_name, parameter in wrapper.adapter_named_parameters():
            if parameter.requires_grad:
                yield f"{binding}.{local_name}", parameter


@contextmanager
def dual_lora_source_context(
    wrappers: Mapping[str, SourceConditionedDualLoRALinear],
    camera_source: torch.Tensor,
    object_source: torch.Tensor,
) -> Iterator[None]:
    """Bind the same frozen physical sources to a declared wrapper set."""
    items = _validated_wrapper_mapping(wrappers, require_nonempty=True)
    with ExitStack() as stack:
        for _, wrapper in items:
            stack.enter_context(wrapper.source_context(camera_source, object_source))
        yield


@dataclass(frozen=True, slots=True)
class NativeLMFactorMargins:
    """Margins derived only from native canonical answer sequence log-probs."""

    camera_moving_minus_stationary: torch.Tensor
    object_moving_minus_stationary: torch.Tensor


def native_lm_factor_margins_from_canonical_sequence_logprobs(
    canonical_answer_sequence_logprobs: torch.Tensor,
) -> NativeLMFactorMargins:
    """Factor four native answer sequence scores into two log-prob margins.

    The last dimension must follow ``FOUR_STATE_ORDER``.  A family runner must
    obtain each value by scoring the complete canonical answer sequence with
    its original decoder and LM head; an auxiliary classifier is not valid.
    """
    _require_float_tensor(
        canonical_answer_sequence_logprobs, "canonical answer sequence log-probs"
    )
    if (
        canonical_answer_sequence_logprobs.ndim != 2
        or canonical_answer_sequence_logprobs.shape[1] != 4
    ):
        raise ValueError("canonical answer sequence log-probs must have shape [B,4]")
    _require_finite(
        canonical_answer_sequence_logprobs, "canonical answer sequence log-probs"
    )
    scores = canonical_answer_sequence_logprobs
    camera_moving = torch.logsumexp(scores[:, (1, 3)], dim=-1)
    camera_stationary = torch.logsumexp(scores[:, (0, 2)], dim=-1)
    object_moving = torch.logsumexp(scores[:, (2, 3)], dim=-1)
    object_stationary = torch.logsumexp(scores[:, (0, 1)], dim=-1)
    return NativeLMFactorMargins(
        camera_moving_minus_stationary=camera_moving - camera_stationary,
        object_moving_minus_stationary=object_moving - object_stationary,
    )


def four_state_probabilities_from_native_lm_margins(
    margins: NativeLMFactorMargins,
) -> torch.Tensor:
    """Compose independent factor probabilities from native-LM margins only."""
    if not isinstance(margins, NativeLMFactorMargins):
        raise TypeError("margins must be NativeLMFactorMargins")
    camera = torch.sigmoid(margins.camera_moving_minus_stationary)
    obj = torch.sigmoid(margins.object_moving_minus_stationary)
    if camera.shape != obj.shape or camera.ndim != 1:
        raise ValueError("native LM factor margins must have matching shape [B]")
    camera_static = 1.0 - camera
    object_static = 1.0 - obj
    return torch.stack(
        (
            camera_static * object_static,
            camera * object_static,
            camera_static * obj,
            camera * obj,
        ),
        dim=-1,
    )


def compose_four_state_from_native_lm_margins(
    margins: NativeLMFactorMargins,
) -> torch.Tensor:
    """Deterministically compose states; a zero margin is conservatively static."""
    probabilities = four_state_probabilities_from_native_lm_margins(margins)
    del probabilities  # validated above; composition uses the signed margins.
    camera_moving = margins.camera_moving_minus_stationary > 0.0
    object_moving = margins.object_moving_minus_stationary > 0.0
    return camera_moving.to(torch.long) + 2 * object_moving.to(torch.long)


def state_labels(state_index: torch.Tensor) -> tuple[str, ...]:
    if not isinstance(state_index, torch.Tensor) or state_index.ndim != 1:
        raise TypeError("state_index must be a one-dimensional torch.Tensor")
    if state_index.is_floating_point() or state_index.dtype == torch.bool:
        raise TypeError("state_index must have an integer dtype")
    values = state_index.detach().cpu().tolist()
    if any(index < 0 or index >= len(FOUR_STATE_ORDER) for index in values):
        raise ValueError("state index is outside the frozen native-answer ABI")
    return tuple(FOUR_STATE_ORDER[index] for index in values)


@dataclass(frozen=True, slots=True)
class StagePLossWeights:
    camera18: float = 0.08
    object_translation: float = 0.10
    object_rotation: float = 0.05

    def __post_init__(self) -> None:
        values = (self.camera18, self.object_translation, self.object_rotation)
        if not all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) >= 0.0
            for value in values
        ):
            raise ValueError("Stage-P loss weights must be finite and non-negative")
        if not any(float(value) > 0.0 for value in values):
            raise ValueError("at least one Stage-P loss weight must be positive")


@dataclass(frozen=True, slots=True)
class StagePLossOutput:
    total: torch.Tensor
    camera18: torch.Tensor
    object_translation: torch.Tensor
    object_rotation: torch.Tensor
    object_translation_contributing_cases: torch.Tensor
    object_rotation_contributing_cases: torch.Tensor


@dataclass(frozen=True, slots=True)
class GradientAudit:
    live: tuple[str, ...]
    missing: tuple[str, ...]
    zero: tuple[str, ...]
    nonfinite: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not (self.missing or self.zero or self.nonfinite)


def masked_three_anchor_smooth_l1(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    valid_anchor_mask: torch.Tensor,
    *,
    beta: float = 1.0,
    require_nonreference_anchor: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    beta = _positive_finite(beta, "SmoothL1 beta")
    _require_matrix(prediction, name="three-anchor prediction", width=9)
    _require_matrix(truth, name="three-anchor truth", width=9)
    if prediction.shape != truth.shape:
        raise ValueError("three-anchor prediction and truth shapes differ")
    batch = prediction.shape[0]
    _require_anchor_mask(valid_anchor_mask, batch=batch, name="valid_anchor_mask")
    _require_finite(prediction, "three-anchor prediction")
    if not isinstance(require_nonreference_anchor, bool):
        raise TypeError("require_nonreference_anchor must be a bool")
    case_valid = valid_anchor_mask.any(dim=1)
    if require_nonreference_anchor:
        case_valid = case_valid & valid_anchor_mask[:, 1:].any(dim=1)
    eligible = valid_anchor_mask & case_valid.unsqueeze(1)
    contributing_cases = case_valid.sum()
    if not bool(eligible.any()):
        return prediction.new_zeros(()), contributing_cases
    prediction3 = prediction.reshape(batch, 3, 3)
    truth3 = truth.reshape(batch, 3, 3)
    if not bool(torch.isfinite(truth3[eligible]).all()):
        raise ValueError("valid three-anchor truth contains non-finite values")
    anchor_loss = F.smooth_l1_loss(
        prediction3[eligible], truth3[eligible], reduction="none", beta=beta
    ).mean(dim=-1)
    case_ids = torch.arange(batch, device=prediction.device).view(-1, 1)
    case_ids = case_ids.expand(batch, 3)[eligible]
    per_case = prediction.new_zeros(batch).index_add(0, case_ids, anchor_loss)
    counts = eligible.sum(dim=1).to(dtype=prediction.dtype).clamp_min(1.0)
    return (per_case / counts)[case_valid].mean(), contributing_cases


def _rotvec_to_quaternion(rotvec: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    vector_scale = 0.5 * torch.sinc(theta / (2.0 * math.pi))
    return torch.cat((torch.cos(theta / 2.0), vector_scale * rotvec), dim=-1)


def masked_three_anchor_geodesic_smooth_l1(
    prediction_normalized: torch.Tensor,
    truth_rad: torch.Tensor,
    valid_anchor_mask: torch.Tensor,
    *,
    rotation_scale_rad: float,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotation_scale_rad = _positive_finite(rotation_scale_rad, "rotation_scale_rad")
    beta = _positive_finite(beta, "SmoothL1 beta")
    _require_matrix(prediction_normalized, name="rotation prediction", width=9)
    _require_matrix(truth_rad, name="rotation truth radians", width=9)
    if prediction_normalized.shape != truth_rad.shape:
        raise ValueError("rotation prediction and truth shapes differ")
    batch = prediction_normalized.shape[0]
    _require_anchor_mask(valid_anchor_mask, batch=batch, name="rotation valid mask")
    _require_finite(prediction_normalized, "rotation prediction")
    case_valid = valid_anchor_mask[:, 1:].any(dim=1)
    eligible = valid_anchor_mask & case_valid.unsqueeze(1)
    contributing_cases = case_valid.sum()
    if not bool(eligible.any()):
        # Disconnected zero is required for AdamW no-drift under an established
        # optimizer state, provided the runner uses the invariant above.
        return prediction_normalized.new_zeros(()), contributing_cases
    prediction_rad = prediction_normalized.reshape(batch, 3, 3) * rotation_scale_rad
    truth3 = truth_rad.reshape(batch, 3, 3)
    if not bool(torch.isfinite(truth3[eligible]).all()):
        raise ValueError("valid rotation truth contains non-finite values")
    prediction_q = _rotvec_to_quaternion(prediction_rad[eligible])
    truth_q = _rotvec_to_quaternion(truth3[eligible])
    prediction_w, prediction_v = prediction_q[:, :1], prediction_q[:, 1:]
    truth_w, truth_v = truth_q[:, :1], truth_q[:, 1:]
    relative_w = prediction_w * truth_w + torch.sum(
        prediction_v * truth_v, dim=-1, keepdim=True
    )
    relative_v = (
        prediction_w * truth_v
        - truth_w * prediction_v
        - torch.linalg.cross(prediction_v, truth_v, dim=-1)
    )
    geodesic_rad = 2.0 * torch.atan2(
        torch.linalg.vector_norm(relative_v, dim=-1),
        relative_w.abs().squeeze(-1),
    )
    normalized_error = geodesic_rad / rotation_scale_rad
    anchor_loss = F.smooth_l1_loss(
        normalized_error,
        torch.zeros_like(normalized_error),
        reduction="none",
        beta=beta,
    )
    case_ids = torch.arange(batch, device=prediction_normalized.device).view(-1, 1)
    case_ids = case_ids.expand(batch, 3)[eligible]
    per_case = prediction_normalized.new_zeros(batch).index_add(
        0, case_ids, anchor_loss
    )
    counts = eligible.sum(dim=1).to(dtype=prediction_normalized.dtype).clamp_min(1.0)
    return (per_case / counts)[case_valid].mean(), contributing_cases


def stage_p_o18_loss(
    core: TsTNativeAdapterO18Core,
    output: PhysicalOutput,
    *,
    camera18_truth_normalized: torch.Tensor,
    object_translation_truth_normalized: torch.Tensor,
    object_rotation_truth_rad: torch.Tensor,
    camera_translation_valid: torch.Tensor,
    camera_rotation_valid: torch.Tensor,
    object_translation_valid: torch.Tensor,
    object_rotation_valid: torch.Tensor,
    weights: StagePLossWeights | None = None,
    smooth_l1_beta: float = 1.0,
) -> StagePLossOutput:
    """Frozen mask-aware Stage-P loss with one authoritative scale contract.

    ``object_rotation_scale_rad`` is intentionally not an argument.  The loss
    asserts that the output and core share the exact module-level prediction
    contract, preventing runner/config scale drift.
    """
    if not isinstance(core, TsTNativeAdapterO18Core):
        raise TypeError("core must be TsTNativeAdapterO18Core")
    if not isinstance(output, PhysicalOutput):
        raise TypeError("output must be PhysicalOutput")
    if output.prediction_contract is not core.prediction_contract:
        raise AssertionError("output/core O18 prediction contracts differ")
    contract = core.prediction_contract
    if (
        contract.camera_translation_scale_m != CAMERA_TRANSLATION_SCALE_M
        or contract.camera_rotation_scale_rad != CAMERA_ROTATION_SCALE_RAD
        or contract.object_translation_scale_m != OBJECT_TRANSLATION_SCALE_M
        or contract.object_rotation_scale_rad != OBJECT_ROTATION_SCALE_RAD
    ):
        raise AssertionError("O18 scale contract drifted from frozen core constants")
    if weights is None:
        weights = StagePLossWeights()
    if not isinstance(weights, StagePLossWeights):
        raise TypeError("weights must be StagePLossWeights")
    beta = _positive_finite(smooth_l1_beta, "SmoothL1 beta")
    _require_matrix(
        camera18_truth_normalized, name="Camera18 truth normalized", width=18
    )
    if camera18_truth_normalized.shape != output.camera18_normalized.shape:
        raise ValueError("Camera18 prediction and truth shapes differ")
    batch = output.camera18_normalized.shape[0]
    _require_anchor_mask(
        camera_translation_valid, batch=batch, name="camera_translation_valid"
    )
    _require_anchor_mask(
        camera_rotation_valid, batch=batch, name="camera_rotation_valid"
    )
    if not bool(camera_translation_valid.all()) or not bool(
        camera_rotation_valid.all()
    ):
        raise ValueError("TsT-NA V1 Camera18 must remain fully supervised")
    _require_finite(camera18_truth_normalized, "Camera18 truth normalized")
    camera_loss = F.smooth_l1_loss(
        output.camera18_normalized, camera18_truth_normalized, beta=beta
    )
    translation_loss, translation_cases = masked_three_anchor_smooth_l1(
        output.object_translation9_normalized,
        object_translation_truth_normalized,
        object_translation_valid,
        beta=beta,
        require_nonreference_anchor=True,
    )
    rotation_loss, rotation_cases = masked_three_anchor_geodesic_smooth_l1(
        output.object_rotation9_normalized,
        object_rotation_truth_rad,
        object_rotation_valid,
        rotation_scale_rad=contract.object_rotation_scale_rad,
        beta=beta,
    )
    terms: list[torch.Tensor] = []
    if float(weights.camera18) > 0.0:
        terms.append(float(weights.camera18) * camera_loss)
    if float(weights.object_translation) > 0.0 and int(translation_cases) > 0:
        terms.append(float(weights.object_translation) * translation_loss)
    if float(weights.object_rotation) > 0.0 and int(rotation_cases) > 0:
        terms.append(float(weights.object_rotation) * rotation_loss)
    if not terms:
        raise ValueError(
            "no positively weighted Stage-P component has valid supervision"
        )
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return StagePLossOutput(
        total=total,
        camera18=camera_loss,
        object_translation=translation_loss,
        object_rotation=rotation_loss,
        object_translation_contributing_cases=translation_cases,
        object_rotation_contributing_cases=rotation_cases,
    )


def audit_named_trainable_gradients(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
) -> GradientAudit:
    live: list[str] = []
    missing: list[str] = []
    zero: list[str] = []
    nonfinite: list[str] = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        elif float(torch.linalg.vector_norm(parameter.grad.detach())) == 0.0:
            zero.append(name)
        else:
            live.append(name)
    return GradientAudit(tuple(live), tuple(missing), tuple(zero), tuple(nonfinite))


def audit_trainable_gradients(module: nn.Module) -> GradientAudit:
    if not isinstance(module, nn.Module):
        raise TypeError("module must be an nn.Module")
    return audit_named_trainable_gradients(module.named_parameters())
