"""Add camera/object source features and bounded residuals to visual tokens."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .contract import (
    BASE_SOURCE_RESIDUAL_SCALE,
    EXPLICIT_BRANCH_MAX_ABS_DELTA,
    EXPLICIT_RESIDUAL_RANK,
    TEMPORAL_UNITS,
)
from .core import MoSDeRMotionSourceCore, MotionSourceOutput
from .routing import FactorRoute, normalize_route


class VisualBridgeContractError(RuntimeError):
    """The raw/source/residual visual contract was violated."""


class BoundedTemporalSourceResidual(nn.Module):
    """Zero-init, bounded residual over a detached source sequence."""

    def __init__(
        self,
        hidden_size: int,
        *,
        rank: int = EXPLICIT_RESIDUAL_RANK,
        max_abs_delta: float = EXPLICIT_BRANCH_MAX_ABS_DELTA,
    ) -> None:
        super().__init__()
        if (
            type(hidden_size) is not int
            or type(rank) is not int
            or hidden_size <= 0
            or not 0 < rank < hidden_size
        ):
            raise ValueError("explicit residual dimensions are invalid")
        if not math.isfinite(max_abs_delta) or not 0 < max_abs_delta <= 0.5:
            raise ValueError("max_abs_delta must be finite in (0,0.5]")
        self.hidden_size = hidden_size
        self.rank = rank
        self.max_abs_delta = float(max_abs_delta)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.gate = nn.Linear(hidden_size, 1, bias=True)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(source, torch.Tensor)
            or not source.is_floating_point()
            or source.ndim != 3
            or tuple(source.shape[1:]) != (TEMPORAL_UNITS, self.hidden_size)
            or not bool(torch.isfinite(source).all())
        ):
            raise VisualBridgeContractError(
                f"explicit source must be finite [B,{TEMPORAL_UNITS},D]"
            )
        parameters = tuple(self.parameters())
        if (
            not parameters
            or any(parameter.dtype != torch.float32 for parameter in parameters)
            or len({parameter.device for parameter in parameters}) != 1
        ):
            raise VisualBridgeContractError(
                "explicit residual must remain one FP32 precision island"
            )
        device = self.down.weight.device
        with torch.autocast(device_type=device.type, enabled=False):
            # Stages G/R may not alter the source/factor representation.
            value = source.detach().to(device=device, dtype=torch.float32)
            normalized = self.norm(value)
            content = self.up(F.gelu(self.down(normalized), approximate="tanh"))
            gate = torch.sigmoid(self.gate(normalized))
            delta = self.max_abs_delta * gate * torch.tanh(content)
        if delta.shape != source.shape or not bool(torch.isfinite(delta).all()):
            raise VisualBridgeContractError("explicit residual output ABI differs")
        if float(delta.detach().abs().max()) > self.max_abs_delta:
            raise VisualBridgeContractError("explicit residual exceeded its bound")
        return delta


class DualSourceExplicitResidual(nn.Module):
    """Independent Camera and Object language-only residual branches."""

    def __init__(
        self,
        hidden_size: int,
        *,
        rank: int = EXPLICIT_RESIDUAL_RANK,
        max_abs_delta: float = EXPLICIT_BRANCH_MAX_ABS_DELTA,
    ) -> None:
        super().__init__()
        self.camera = BoundedTemporalSourceResidual(
            hidden_size, rank=rank, max_abs_delta=max_abs_delta
        )
        self.object = BoundedTemporalSourceResidual(
            hidden_size, rank=rank, max_abs_delta=max_abs_delta
        )

    def zero_init_exact(self) -> bool:
        return bool(
            torch.count_nonzero(self.camera.up.weight) == 0
            and torch.count_nonzero(self.object.up.weight) == 0
        )


@dataclass(frozen=True, slots=True)
class VisualBridgeOutput:
    route: FactorRoute
    raw_native_tokens: torch.Tensor
    output_native_tokens: torch.Tensor
    camera_source: torch.Tensor
    object_source: torch.Tensor
    camera_explicit_residual: torch.Tensor | None
    object_explicit_residual: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class InjectedVisualOutput:
    """Visual tokens after source injection, with optional explicit residuals."""

    route: FactorRoute
    output_native_tokens: torch.Tensor
    camera_explicit_residual: torch.Tensor | None
    object_explicit_residual: torch.Tensor | None


def inject_route_sources(
    *,
    raw_native_tokens: torch.Tensor,
    target_mask: torch.Tensor,
    context_mask: torch.Tensor,
    camera_source: torch.Tensor,
    object_source: torch.Tensor,
    route: FactorRoute | str,
    explicit_residual: DualSourceExplicitResidual,
    source_scale: float = BASE_SOURCE_RESIDUAL_SCALE,
) -> InjectedVisualOutput:
    """Compose native tokens and MoSDeR sources for every integration path.

    The standalone bridge and family backends both call this function to keep
    their route selection and numerical composition consistent.
    """

    normalized = normalize_route(route)
    _validate_native_inputs(raw_native_tokens, target_mask, context_mask)
    if (
        not isinstance(explicit_residual, DualSourceExplicitResidual)
        or not math.isfinite(source_scale)
        or float(source_scale) != BASE_SOURCE_RESIDUAL_SCALE
    ):
        raise VisualBridgeContractError(
            "MoSDeR residual module or fixed source scale is invalid"
        )
    batch, temporal, _, _, hidden = raw_native_tokens.shape
    expected_source = (batch, TEMPORAL_UNITS, hidden)
    if any(
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or tuple(value.shape) != expected_source
        or not bool(torch.isfinite(value).all())
        for value in (camera_source, object_source)
    ):
        raise VisualBridgeContractError(
            f"camera/object sources must be finite {expected_source}"
        )

    camera = _expand_temporal(camera_source, temporal).to(
        device=raw_native_tokens.device, dtype=raw_native_tokens.dtype
    )
    obj = _expand_temporal(object_source, temporal).to(
        device=raw_native_tokens.device, dtype=raw_native_tokens.dtype
    )
    output = raw_native_tokens
    camera_explicit: torch.Tensor | None = None
    object_explicit: torch.Tensor | None = None

    if normalized in {FactorRoute.CAMERA_FACTOR, FactorRoute.FULL_LANGUAGE}:
        output = output + source_scale * _masked_grid(
            camera, context_mask, raw_native_tokens
        )
    if normalized in {FactorRoute.OBJECT_FACTOR, FactorRoute.FULL_LANGUAGE}:
        output = output + source_scale * _masked_grid(
            obj, target_mask, raw_native_tokens
        )
    if normalized is FactorRoute.FULL_LANGUAGE:
        camera_explicit = explicit_residual.camera(camera_source)
        object_explicit = explicit_residual.object(object_source)
        camera_refined = _expand_temporal(camera_explicit, temporal).to(
            device=output.device, dtype=output.dtype
        )
        object_refined = _expand_temporal(object_explicit, temporal).to(
            device=output.device, dtype=output.dtype
        )
        output = (
            output
            + _masked_grid(camera_refined, context_mask, output)
            + _masked_grid(object_refined, target_mask, output)
        )

    if (
        output.shape != raw_native_tokens.shape
        or output.dtype != raw_native_tokens.dtype
        or output.device != raw_native_tokens.device
        or not bool(torch.isfinite(output).all())
    ):
        raise VisualBridgeContractError("refined native-token ABI differs")
    return InjectedVisualOutput(
        route=normalized,
        output_native_tokens=output,
        camera_explicit_residual=camera_explicit,
        object_explicit_residual=object_explicit,
    )


class MoSDeRVisualBridge(nn.Module):
    """Combine source adapters and residuals with native visual tokens.

    The bridge always preserves the raw family-native token path.  It adds the
    selected source stream at scale 0.05; FULL_LANGUAGE additionally
    adds the independent zero-init bounded explicit residuals.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        source_rank: int,
        residual_rank: int = EXPLICIT_RESIDUAL_RANK,
        source_scale: float = BASE_SOURCE_RESIDUAL_SCALE,
    ) -> None:
        super().__init__()
        if not math.isfinite(source_scale) or source_scale != (
            BASE_SOURCE_RESIDUAL_SCALE
        ):
            raise ValueError("MoSDeR source scale must remain exactly 0.05")
        self.hidden_size = hidden_size
        self.source_scale = float(source_scale)
        self.source_core = MoSDeRMotionSourceCore(hidden_size, rank=source_rank)
        self.explicit_residual = DualSourceExplicitResidual(
            hidden_size, rank=residual_rank
        )

    def forward(
        self,
        *,
        raw_native_tokens: torch.Tensor,
        target_features: torch.Tensor,
        context_features: torch.Tensor,
        target_mask: torch.Tensor,
        context_mask: torch.Tensor,
        route: FactorRoute | str,
    ) -> VisualBridgeOutput:
        _validate_native_inputs(raw_native_tokens, target_mask, context_mask)
        sources = self.source_core(target_features, context_features)
        injected = inject_route_sources(
            raw_native_tokens=raw_native_tokens,
            target_mask=target_mask,
            context_mask=context_mask,
            camera_source=sources.camera_source,
            object_source=sources.object_source,
            route=route,
            explicit_residual=self.explicit_residual,
            source_scale=self.source_scale,
        )
        return VisualBridgeOutput(
            route=injected.route,
            raw_native_tokens=raw_native_tokens,
            output_native_tokens=injected.output_native_tokens,
            camera_source=sources.camera_source,
            object_source=sources.object_source,
            camera_explicit_residual=injected.camera_explicit_residual,
            object_explicit_residual=injected.object_explicit_residual,
        )


def _validate_native_inputs(
    raw: torch.Tensor,
    target_mask: torch.Tensor,
    context_mask: torch.Tensor,
) -> None:
    if (
        not isinstance(raw, torch.Tensor)
        or not raw.is_floating_point()
        or raw.ndim != 5
        or raw.shape[1] not in {TEMPORAL_UNITS, 2 * TEMPORAL_UNITS}
        or not bool(torch.isfinite(raw).all())
    ):
        raise VisualBridgeContractError("native tokens must be finite [B,10|20,H,W,D]")
    expected_mask = raw.shape[:-1]
    if (
        not isinstance(target_mask, torch.Tensor)
        or not isinstance(context_mask, torch.Tensor)
        or target_mask.dtype != torch.bool
        or context_mask.dtype != torch.bool
        or target_mask.shape != expected_mask
        or context_mask.shape != expected_mask
        or target_mask.device != raw.device
        or context_mask.device != raw.device
        or bool((target_mask & context_mask).any())
        or not bool((target_mask | context_mask).all())
    ):
        raise VisualBridgeContractError(
            "target/context masks must be complementary boolean native grids"
        )


def _expand_temporal(value: torch.Tensor, temporal: int) -> torch.Tensor:
    if value.ndim != 3 or value.shape[1] != TEMPORAL_UNITS:
        raise VisualBridgeContractError("source sequence must have ten units")
    if temporal == TEMPORAL_UNITS:
        return value
    if temporal == 2 * TEMPORAL_UNITS:
        return value.repeat_interleave(2, dim=1)
    raise VisualBridgeContractError("native temporal axis cannot map to ten units")


def _masked_grid(
    sequence: torch.Tensor,
    mask: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if (
        sequence.shape[:2] != reference.shape[:2]
        or sequence.shape[-1] != reference.shape[-1]
    ):
        raise VisualBridgeContractError("sequence/native grid dimensions differ")
    return sequence[:, :, None, None, :] * mask[..., None].to(reference)


__all__ = [
    "BoundedTemporalSourceResidual",
    "DualSourceExplicitResidual",
    "InjectedVisualOutput",
    "MoSDeRVisualBridge",
    "VisualBridgeContractError",
    "VisualBridgeOutput",
    "inject_route_sources",
]
