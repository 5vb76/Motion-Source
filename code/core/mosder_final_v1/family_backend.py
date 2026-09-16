"""Attach MoSDeR modules to Qwen3-VL, Molmo2, and NVILA backends.

The native backend supplies RGB loading and decoder hooks. This module
registers the camera/object source adapters, visual residuals, TriLoRA
layers, and factor decision biases on the model."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Final, Protocol, runtime_checkable

import torch
from torch import nn


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from family_backends_v1 import (  # noqa: E402
    BaseNativeBackend,
    CoreLoRACallAuditHook,
    MolmoNativeBackend,
    NVILANativeBackend,
    NativeBackendBlocked,
    NativeBackendError,
    NativeForwardResult,
    NativeGeneration,
    NativeVisualBundle,
    ParameterStamp,
    QwenNativeBackend,
    RGB20Request,
    SourceConditionedDualLoRALinear,
    SourceReadAdapterPair,
    TeacherForcedScore,
    TeacherForcedTensorScore,
    binding_for,
    load_local_backend,
    replace_resolved_module,
    resolve_unique_module,
    source_conditioned_lora_target_modules,
    _tensor_digest,
)

from .bridge import (
    DualSourceExplicitResidual,
    VisualBridgeContractError,
    inject_route_sources,
)
from .contract import (
    EXPLICIT_RESIDUAL_RANK,
    FACTORSPAN_LAYERS,
    SOURCE_RANK,
    SUPPORTED_FAMILIES,
    TEMPORAL_UNITS,
    TRILORA_ALPHA,
    TRILORA_DROPOUT,
    TRILORA_RANK,
)
from .core import MoSDeRMotionSourceCore, MotionSourceOutput
from .routing import (
    FactorRoute,
    FactorSpanDecision,
    SourceConditionedTriLoRALinear,
    normalize_route,
)
from .training import (
    StageConfiguration,
    TrainingStage,
    configure_training_stage,
    named_owner_partitions,
    normalize_stage,
)


METHOD_NAME: Final[str] = "MoSDeR"
SCHEMA_VERSION: Final[str] = "mosder_family_backend_v1"


class MoSDeRBackendError(NativeBackendError):
    """A direct MoSDeR family integration invariant failed."""


@dataclass(frozen=True, slots=True)
class BackendStageConfiguration:
    stage: TrainingStage
    owner_configuration: StageConfiguration
    model_trainable_names: tuple[str, ...]


@runtime_checkable
class MoSDeRFamilyBackendProtocol(Protocol):
    model: nn.Module
    source_core: MoSDeRMotionSourceCore
    explicit_residual: DualSourceExplicitResidual
    factor_decision: FactorSpanDecision

    def attach_mosder_plugin(
        self,
        *,
        source_rank: int = SOURCE_RANK,
        residual_rank: int = EXPLICIT_RESIDUAL_RANK,
    ) -> None: ...

    def attach_mosder_decoder(
        self,
        *,
        layer_indices: Sequence[int] | None = None,
        rank: int = TRILORA_RANK,
        alpha: float = TRILORA_ALPHA,
        dropout: float = TRILORA_DROPOUT,
    ) -> Mapping[str, tuple[str, ...]]: ...

    def configure_mosder_stage(
        self, stage: TrainingStage | str
    ) -> BackendStageConfiguration: ...


_ACTIVE_ROUTE: ContextVar[tuple[int, FactorRoute] | None] = ContextVar(
    "mosder_active_factor_route", default=None
)


class MoSDeRBackendMixin:
    """Family-neutral direct registration and routed execution mixin."""

    _allow_test_layer_override: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._mosder_wrappers: dict[str, SourceConditionedTriLoRALinear] = {}
        self._mosder_paths: tuple[str, ...] = ()
        self._mosder_source_core: MoSDeRMotionSourceCore | None = None
        self._mosder_explicit_residual: DualSourceExplicitResidual | None = None
        self._mosder_factor_decision: FactorSpanDecision | None = None
        self._mosder_install_blocked = False
        self._mosder_route_success_counts = {route: 0 for route in FactorRoute}
        self._mosder_explicit_call_counts = {"camera": 0, "object": 0}

    @property
    def source_core(self) -> MoSDeRMotionSourceCore:
        if self._mosder_source_core is None:
            raise MoSDeRBackendError("MoSDeR source core is not attached")
        return self._mosder_source_core

    @property
    def explicit_residual(self) -> DualSourceExplicitResidual:
        if self._mosder_explicit_residual is None:
            raise MoSDeRBackendError("MoSDeR explicit residual is not attached")
        return self._mosder_explicit_residual

    @property
    def factor_decision(self) -> FactorSpanDecision:
        if self._mosder_factor_decision is None:
            raise MoSDeRBackendError("MoSDeR factor decision is not attached")
        return self._mosder_factor_decision

    @property
    def active_factor_route(self) -> FactorRoute:
        active = _ACTIVE_ROUTE.get()
        if active is None or active[0] != id(self):
            raise MoSDeRBackendError("native execution has no MoSDeR route")
        return active[1]

    def _assert_not_poisoned(self) -> None:
        if self._mosder_install_blocked or bool(
            getattr(self.model, "_mosder_install_blocked", False)
        ):
            raise NativeBackendBlocked(
                "MoSDeR backend is poisoned by a partial installation; discard it"
            )

    def _poison(self) -> None:
        self._mosder_install_blocked = True
        setattr(self.model, "_mosder_install_blocked", True)

    def _assert_no_legacy_methods(self) -> None:
        legacy = [
            name
            for name, module in self.model.named_modules()
            if isinstance(
                module,
                (SourceConditionedDualLoRALinear, SourceReadAdapterPair),
            )
        ]
        if self._source_handles or legacy:
            raise NativeBackendBlocked(
                {
                    "reason": "MoSDeR forbids legacy block adapters and dual-LoRA",
                    "source_hook_count": len(self._source_handles),
                    "legacy_modules": legacy,
                }
            )

    def attach_unified_physical_core(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise NativeBackendBlocked(
            "MoSDeR forbids the legacy O18 physical-core attachment API"
        )

    def attach_source_read_adapters(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise NativeBackendBlocked("MoSDeR forbids block-residual adapters")

    def attach_source_conditioned_lora(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise NativeBackendBlocked("MoSDeR forbids legacy dual-LoRA")

    def attach_mosder_plugin(
        self,
        *,
        source_rank: int = SOURCE_RANK,
        residual_rank: int = EXPLICIT_RESIDUAL_RANK,
    ) -> None:
        self._assert_not_poisoned()
        self._assert_no_legacy_methods()
        if (
            self._mosder_source_core is not None
            or self._mosder_explicit_residual is not None
            or self.unified_physical_core is not None
        ):
            raise MoSDeRBackendError("MoSDeR plugin is already attached")
        if not self._allow_test_layer_override and (
            source_rank != SOURCE_RANK or residual_rank != EXPLICIT_RESIDUAL_RANK
        ):
            raise MoSDeRBackendError("production MoSDeR ranks drifted")
        _, owner = resolve_unique_module(self.model, self.binding.post_projector_path)
        forbidden = (
            "_mosder_source_core",
            "_mosder_explicit_residual",
            "_mosder_factor_decision",
        )
        if any(hasattr(owner, name) for name in forbidden):
            raise MoSDeRBackendError("post-projector already owns MoSDeR modules")
        core = MoSDeRMotionSourceCore(self.binding.hidden_size, rank=source_rank).to(
            device=self.device, dtype=torch.float32
        )
        residual = DualSourceExplicitResidual(
            self.binding.hidden_size, rank=residual_rank
        ).to(device=self.device, dtype=torch.float32)
        decision = FactorSpanDecision().to(device=self.device, dtype=torch.float32)
        core.configure_stage("FROZEN")
        residual.requires_grad_(False)
        decision.requires_grad_(False)
        owner.add_module("_mosder_source_core", core)
        owner.add_module("_mosder_explicit_residual", residual)
        owner.add_module("_mosder_factor_decision", decision)
        self._mosder_source_core = core
        self._mosder_explicit_residual = residual
        self._mosder_factor_decision = decision
        # The inherited visual hook consumes only ``physical_forward`` and the
        # two source fields.  This pointer does not create an O18 module.
        self.unified_physical_core = core  # type: ignore[assignment]

    def attach_mosder_decoder(
        self,
        *,
        layer_indices: Sequence[int] | None = None,
        rank: int = TRILORA_RANK,
        alpha: float = TRILORA_ALPHA,
        dropout: float = TRILORA_DROPOUT,
    ) -> Mapping[str, tuple[str, ...]]:
        self._assert_not_poisoned()
        self._assert_no_legacy_methods()
        if self._mosder_source_core is None:
            raise MoSDeRBackendError("attach the source-only plugin first")
        if self._mosder_paths or self._mosder_wrappers:
            raise MoSDeRBackendError("MoSDeR decoder adapters already exist")
        expected = FACTORSPAN_LAYERS.get(self.binding.key)
        if expected is None:
            if not self._allow_test_layer_override or layer_indices is None:
                raise MoSDeRBackendError("family has no frozen MoSDeR layer map")
            selected = tuple(map(int, layer_indices))
        else:
            selected = (
                expected if layer_indices is None else tuple(map(int, layer_indices))
            )
            if selected != expected:
                raise MoSDeRBackendError("production MoSDeR layers drifted")
        if (
            not selected
            or len(selected) != len(set(selected))
            or not math.isfinite(alpha)
            or float(dropout) != 0.0
            or (
                not self._allow_test_layer_override
                and (
                    rank != TRILORA_RANK
                    or float(alpha) != TRILORA_ALPHA
                    or float(dropout) != TRILORA_DROPOUT
                )
            )
        ):
            raise MoSDeRBackendError("MoSDeR decoder hyperparameters drifted")
        targets = source_conditioned_lora_target_modules(
            self.model, self.binding, selected
        )
        installed: list[str] = []
        try:
            for path in targets:
                _, module = resolve_unique_module(self.model, path)
                if not isinstance(module, nn.Linear):
                    raise MoSDeRBackendError(
                        f"MoSDeR target is not native nn.Linear: {path}"
                    )
                wrapper = SourceConditionedTriLoRALinear(
                    module,
                    source_dim=self.binding.hidden_size,
                    rank=rank,
                    alpha=alpha,
                )
                # Any FP16 frozen W0 requires FP32 adapter masters because
                # AdamW(eps=1e-8) underflows in FP16.  Routed execution below
                # supplies ambient FP16 autocast without changing W0.
                if wrapper.base.weight.dtype == torch.float16:
                    wrapper.set_adapter_master_dtype(torch.float32)
                replace_resolved_module(self.model, path, wrapper)
                self._mosder_wrappers[path] = wrapper
                self._source_lora_handles.append(
                    wrapper.register_forward_pre_hook(
                        CoreLoRACallAuditHook(self, path), with_kwargs=True
                    )
                )
                installed.append(path)
        except Exception:
            self._mosder_paths = tuple(installed)
            self._poison()
            raise
        self._mosder_paths = tuple(installed)
        self._source_lora_paths[:] = installed
        self._source_lora_wrappers = self._mosder_wrappers  # type: ignore[assignment]
        self._require_complete()
        self.configure_mosder_stage("FROZEN")
        return self.owner_parameter_names()

    def _require_complete(self) -> None:
        self._assert_not_poisoned()
        self._assert_no_legacy_methods()
        checkpointing_flags = [
            name
            for name, module in self.model.named_modules()
            if bool(getattr(module, "gradient_checkpointing", False))
        ]
        if bool(getattr(self.model, "is_gradient_checkpointing", False)) or (
            checkpointing_flags
        ):
            raise MoSDeRBackendError(
                {
                    "reason": (
                        "MoSDeR-v1 freezes gradient checkpointing OFF because "
                        "native decoder recomputation outlives the route context"
                    ),
                    "modules": checkpointing_flags[:32],
                }
            )
        expected = self._mosder_paths
        if (
            not expected
            or len(expected) != len(set(expected))
            or set(self._mosder_wrappers) != set(expected)
            or tuple(self._source_lora_paths) != expected
            or set(self._source_lora_wrappers) != set(expected)
            or self._mosder_source_core is None
            or self._mosder_explicit_residual is None
            or self._mosder_factor_decision is None
        ):
            raise MoSDeRBackendError("MoSDeR assembly is incomplete")
        live = {
            name: module
            for name, module in self.model.named_modules()
            if isinstance(module, SourceConditionedTriLoRALinear)
        }
        if set(live) != set(expected) or any(
            live[path] is not self._mosder_wrappers[path] for path in expected
        ):
            raise MoSDeRBackendError("live MoSDeR wrapper registry differs")
        if any(
            module.__class__.__name__ in {"Camera18Head", "TrajectoryComponentHead"}
            for module in self.model.modules()
        ):
            raise MoSDeRBackendError("numeric trajectory head is live")
        fp32_islands = (
            tuple(self.source_core.parameters())
            + tuple(self.explicit_residual.parameters())
            + tuple(self.factor_decision.parameters())
        )
        if not fp32_islands or any(
            parameter.dtype != torch.float32 for parameter in fp32_islands
        ):
            raise MoSDeRBackendError(
                "source, decision, and explicit residual must remain FP32 islands"
            )
        for path, wrapper in self._mosder_wrappers.items():
            base = wrapper.base.weight
            expected_dtype = (
                torch.float32 if base.dtype == torch.float16 else base.dtype
            )
            owned = tuple(
                parameter
                for values in wrapper.owner_named_parameters().values()
                for _, parameter in values
            )
            if not owned or any(
                parameter.dtype != expected_dtype or parameter.device != base.device
                for parameter in owned
            ):
                raise MoSDeRBackendError(
                    f"TriLoRA precision/device policy drifted: {path}"
                )

    def owner_parameter_names(self) -> Mapping[str, tuple[str, ...]]:
        self._require_complete()
        return named_owner_partitions(
            source_core=self.source_core,
            wrappers=self._mosder_wrappers,
            decision=self.factor_decision,
            residual=self.explicit_residual,
        ).names()

    def configure_mosder_stage(
        self, stage: TrainingStage | str
    ) -> BackendStageConfiguration:
        self._require_complete()
        normalized = normalize_stage(stage)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        configuration = configure_training_stage(
            normalized,
            installed_model=self.model,
            source_core=self.source_core,
            wrappers=self._mosder_wrappers,
            decision=self.factor_decision,
            residual=self.explicit_residual,
        )
        expected_ids = {
            id(parameter)
            for name, parameter in configuration.owners.all
            if name in configuration.trainable_names
        }
        observed = tuple(
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        )
        observed_ids = {
            id(parameter)
            for parameter in self.model.parameters()
            if parameter.requires_grad
        }
        if observed_ids != expected_ids:
            raise MoSDeRBackendError(
                "live model trainables differ from MoSDeR owner allowlist"
            )
        return BackendStageConfiguration(normalized, configuration, observed)

    @contextmanager
    def factor_route_context(self, route: FactorRoute | str) -> Iterator[FactorRoute]:
        self._assert_not_poisoned()
        normalized = normalize_route(route)
        if _ACTIVE_ROUTE.get() is not None:
            raise MoSDeRBackendError("nested MoSDeR routes are forbidden")
        token = _ACTIVE_ROUTE.set((id(self), normalized))
        try:
            yield normalized
        finally:
            _ACTIVE_ROUTE.reset(token)

    def _run_routed(
        self,
        route: FactorRoute | str,
        operation: Any,
        *,
        purpose: str,
    ) -> Any:
        normalized = normalize_route(route)
        if purpose not in {"native_forward", "teacher_forced", "generate"}:
            raise MoSDeRBackendError("unknown routed operation")
        self._require_complete()
        stage = self.source_core.active_stage
        if stage == "F" and normalized is FactorRoute.FULL_LANGUAGE:
            raise MoSDeRBackendError(
                "Stage-F permits only CAMERA_FACTOR or OBJECT_FACTOR routes"
            )
        if stage in {"G", "R"} and normalized is not FactorRoute.FULL_LANGUAGE:
            raise MoSDeRBackendError(f"Stage-{stage} requires the FULL_LANGUAGE route")
        native_fp16 = any(
            wrapper.base.weight.dtype == torch.float16
            for wrapper in self._mosder_wrappers.values()
        )
        precision = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if native_fp16 and torch.device(self.device).type == "cuda"
            else nullcontext()
        )
        with self.factor_route_context(normalized), precision:
            output = operation()
        self._mosder_route_success_counts[normalized] += 1
        return output

    def native_forward(
        self,
        request: RGB20Request,
        *,
        route: FactorRoute | str,
        capture_layers: Sequence[int] = (),
    ) -> NativeForwardResult:
        return self._run_routed(
            route,
            lambda: super(MoSDeRBackendMixin, self).native_forward(
                request, capture_layers=capture_layers
            ),
            purpose="native_forward",
        )

    def teacher_forced_loss(
        self,
        request: RGB20Request,
        candidate_text: str,
        *,
        route: FactorRoute | str,
    ) -> TeacherForcedTensorScore:
        return self._run_routed(
            route,
            lambda: super(MoSDeRBackendMixin, self).teacher_forced_loss(
                request, candidate_text
            ),
            purpose="teacher_forced",
        )

    def score_tensor(
        self,
        request: RGB20Request,
        candidate_text: str,
        *,
        route: FactorRoute | str,
    ) -> TeacherForcedTensorScore:
        return self.teacher_forced_loss(request, candidate_text, route=route)

    def teacher_forced_score(
        self,
        request: RGB20Request,
        candidate_text: str,
        *,
        route: FactorRoute | str,
    ) -> TeacherForcedScore:
        score = self.teacher_forced_loss(request, candidate_text, route=route)
        selected = score.token_log_probabilities
        return TeacherForcedScore(
            family=self.binding.key,
            text=candidate_text.strip(),
            token_count=int(selected.numel()),
            log_probability_sum=float(selected.sum().detach().cpu()),
            log_probability_mean=float(selected.mean().detach().cpu()),
        )

    def generate(
        self, request: RGB20Request, **generation_kwargs: Any
    ) -> NativeGeneration:
        if "route" in generation_kwargs:
            raise MoSDeRBackendError(
                "MoSDeR generation route is fixed to FULL_LANGUAGE"
            )
        return self._run_routed(
            FactorRoute.FULL_LANGUAGE,
            lambda: super(MoSDeRBackendMixin, self).generate(
                request, **generation_kwargs
            ),
            purpose="generate",
        )

    def route_success_counts(self) -> Mapping[str, int]:
        return {
            route.value: self._mosder_route_success_counts[route]
            for route in FactorRoute
        }

    def explicit_residual_call_counts(self) -> Mapping[str, int]:
        return dict(self._mosder_explicit_call_counts)

    def _validate_physical_output(self, output: Any) -> None:
        if not isinstance(output, MotionSourceOutput):
            raise MoSDeRBackendError("source core returned the wrong output type")
        expected = (1, TEMPORAL_UNITS, self.binding.hidden_size)
        if any(
            not isinstance(value, torch.Tensor)
            or value.shape != expected
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all())
            for value in (output.camera_source, output.object_source)
        ):
            raise MoSDeRBackendError("source-only output ABI differs")

    def _reinject_physical_residuals(
        self,
        bundle: NativeVisualBundle,
        *,
        camera_delta: torch.Tensor,
        object_delta: torch.Tensor,
    ) -> torch.Tensor:
        native = bundle.raw_native_tokens
        try:
            injected = inject_route_sources(
                raw_native_tokens=native.unsqueeze(0),
                target_mask=bundle.native_target_mask.unsqueeze(0),
                context_mask=bundle.native_context_mask.unsqueeze(0),
                camera_source=camera_delta.unsqueeze(0),
                object_source=object_delta.unsqueeze(0),
                route=self.active_factor_route,
                explicit_residual=self.explicit_residual,
            )
        except VisualBridgeContractError as error:
            raise MoSDeRBackendError(
                f"MoSDeR visual reinjection contract failed: {error}"
            ) from error
        if injected.route is FactorRoute.FULL_LANGUAGE:
            if (
                injected.camera_explicit_residual is None
                or injected.object_explicit_residual is None
            ):
                raise MoSDeRBackendError("FULL_LANGUAGE residual was not executed")
            self._mosder_explicit_call_counts["camera"] += 1
            self._mosder_explicit_call_counts["object"] += 1
        return injected.output_native_tokens[0]

    def _bind_source_lora_context(self, state: Any, output: MotionSourceOutput) -> None:
        self._require_complete()
        route = self.active_factor_route
        kwargs: dict[str, torch.Tensor] = {}
        if route in {FactorRoute.CAMERA_FACTOR, FactorRoute.FULL_LANGUAGE}:
            kwargs["camera_source"] = output.camera_source
        if route in {FactorRoute.OBJECT_FACTOR, FactorRoute.FULL_LANGUAGE}:
            kwargs["object_source"] = output.object_source
        for wrapper in self._mosder_wrappers.values():
            state.source_context_stack.enter_context(
                wrapper.route_context(route, **kwargs)
            )


class QwenMoSDeRBackend(MoSDeRBackendMixin, QwenNativeBackend):
    """Qwen3-VL-8B MoSDeR integration."""


class MolmoMoSDeRBackend(MoSDeRBackendMixin, MolmoNativeBackend):
    """Molmo2-O-7B MoSDeR integration."""


class NVILAMoSDeRBackend(MoSDeRBackendMixin, NVILANativeBackend):
    """NVILA-Lite-8B MoSDeR integration."""


MOSDER_BACKEND_TYPES: Final[Mapping[str, type[BaseNativeBackend]]] = {
    "qwen3_vl_8b": QwenMoSDeRBackend,
    "molmo2_o_7b": MolmoMoSDeRBackend,
    "nvila_lite_8b": NVILAMoSDeRBackend,
}


def method_parameter_ids(model: nn.Module) -> set[int]:
    identities: set[int] = set()
    for module in model.modules():
        if isinstance(
            module,
            (
                MoSDeRMotionSourceCore,
                DualSourceExplicitResidual,
                FactorSpanDecision,
            ),
        ):
            identities.update(id(parameter) for parameter in module.parameters())
        elif isinstance(module, SourceConditionedTriLoRALinear):
            identities.update(
                id(parameter)
                for values in module.owner_named_parameters().values()
                for _, parameter in values
            )
    return identities


def snapshot_mosder_frozen_base(
    model: nn.Module, *, mode: str = "sampled"
) -> Mapping[str, ParameterStamp]:
    owned = method_parameter_ids(model)
    if not owned:
        raise MoSDeRBackendError("no MoSDeR parameters were found")
    output: dict[str, ParameterStamp] = {}
    for name, parameter in model.named_parameters():
        if id(parameter) in owned:
            continue
        if parameter.requires_grad:
            raise MoSDeRBackendError(f"original VLM parameter is trainable: {name}")
        output[name] = ParameterStamp(
            shape=tuple(parameter.shape),
            dtype=str(parameter.dtype),
            numel=parameter.numel(),
            tensor_version=int(parameter._version),
            digest=_tensor_digest(parameter, mode),
        )
    if not output:
        raise MoSDeRBackendError("frozen-base snapshot is empty")
    return output


def assert_mosder_frozen_base_unchanged(
    model: nn.Module,
    before: Mapping[str, ParameterStamp],
    *,
    mode: str = "sampled",
) -> Mapping[str, ParameterStamp]:
    after = snapshot_mosder_frozen_base(model, mode=mode)
    if set(after) != set(before):
        raise MoSDeRBackendError("original VLM parameter names changed")
    changed = [name for name in before if before[name] != after[name]]
    if changed:
        raise MoSDeRBackendError(
            {"original_VLM_parameters_changed": changed[:32], "count": len(changed)}
        )
    return after


def assemble_mosder_backend(base: BaseNativeBackend) -> MoSDeRFamilyBackendProtocol:
    if not isinstance(base, BaseNativeBackend):
        raise TypeError("base must be a BaseNativeBackend")
    if base.binding.key not in SUPPORTED_FAMILIES:
        raise MoSDeRBackendError(f"unsupported family: {base.binding.key!r}")
    legacy = [
        name
        for name, module in base.model.named_modules()
        if isinstance(
            module,
            (SourceConditionedDualLoRALinear, SourceReadAdapterPair),
        )
    ]
    if (
        base.unified_physical_core is not None
        or base._source_lora_paths
        or base._source_handles
        or legacy
    ):
        raise MoSDeRBackendError("assembly requires a fresh family backend")
    backend_type = MOSDER_BACKEND_TYPES[base.binding.key]
    values = {
        "model": base.model,
        "processor": base.processor,
        "binding": base.binding,
        "device": base.device,
        "runtime_audit": base.runtime_audit,
        "artifact_audit": base.artifact_audit,
        "video_budget_tier": (
            base.video_budget_tier
            if base.binding.key == "qwen3_vl_8b"
            else "stock_primary"
        ),
        "video_budget_reason": (
            base.video_budget_reason if base.binding.key == "qwen3_vl_8b" else None
        ),
    }
    base.close()
    backend = backend_type(**values)
    try:
        backend.attach_mosder_plugin()
        backend.attach_mosder_decoder()
    except Exception:
        backend.close()
        raise
    return backend


def load_local_mosder_backend(
    family: str,
    *,
    device: str | torch.device = "cuda:0",
    dtype: str | None = None,
    qwen_video_budget_tier: str = "stock_primary",
    qwen_video_budget_reason: str | None = None,
) -> MoSDeRFamilyBackendProtocol:
    binding_for(family)
    if family not in SUPPORTED_FAMILIES:
        raise MoSDeRBackendError(f"unsupported family: {family!r}")
    base = load_local_backend(
        family,
        device=device,
        dtype=dtype,
        qwen_video_budget_tier=qwen_video_budget_tier,
        qwen_video_budget_reason=qwen_video_budget_reason,
    )
    try:
        return assemble_mosder_backend(base)
    except Exception:
        base.close()
        raise


__all__ = [
    "BackendStageConfiguration",
    "METHOD_NAME",
    "MOSDER_BACKEND_TYPES",
    "MoSDeRBackendError",
    "MoSDeRBackendMixin",
    "MoSDeRFamilyBackendProtocol",
    "MolmoMoSDeRBackend",
    "NVILAMoSDeRBackend",
    "QwenMoSDeRBackend",
    "SCHEMA_VERSION",
    "assemble_mosder_backend",
    "assert_mosder_frozen_base_unchanged",
    "load_local_mosder_backend",
    "method_parameter_ids",
    "snapshot_mosder_frozen_base",
]
