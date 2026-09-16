"""Execute stage losses, accumulate gradients, and apply optimizer updates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch
from torch import Tensor, nn

from mosder_final_v1.factor_training import (
    StageFCandidateTexts,
    exact_two_pass_stage_f_backward,
)
from mosder_final_v1.family_backend import (
    assert_mosder_frozen_base_unchanged,
    snapshot_mosder_frozen_base,
)
from mosder_final_v1.language import (
    canonical_answer,
    plan_canonical_language,
    validate_score_against_plan,
)
from mosder_final_v1.objectives import (
    stage_g_language_loss,
    stage_r_language_loss,
)
from mosder_final_v1.optimizer import audit_stage_adamw, build_stage_adamw
from mosder_final_v1.routing import FactorRoute
from mosder_final_v1.contract import STATE_FACTORS

from .protocol import RunProtocol, STAGES, canonical_sha256
from .sampler import FullCoverageStateSampler
from .schedule import WarmupCosineToFloor


class EngineContractError(RuntimeError):
    """A training step, gradient window, or stage transition failed closed."""


STRICT_CLIP_AUDIT_SCHEMA_VERSION = "mosder_strict_gradient_clip_audit_v1"
STRICT_CLIP_ABS_TOLERANCE = 1.0e-6
STRICT_CLIP_MAX_CORRECTION_ATTEMPTS = 2


@dataclass(frozen=True, slots=True)
class StrictClipDTypeAudit:
    dtype: str
    tensor_count: int
    nonzero_tensor_count: int
    gradient_numel: int
    nonzero_element_count: int
    unit_roundoff: float

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "dtype": self.dtype,
            "tensor_count": self.tensor_count,
            "nonzero_tensor_count": self.nonzero_tensor_count,
            "gradient_numel": self.gradient_numel,
            "nonzero_element_count": self.nonzero_element_count,
            "unit_roundoff": self.unit_roundoff,
        }


@dataclass(frozen=True, slots=True)
class StrictClipCorrectionAttempt:
    attempt_one_based: int
    pre_correction_fp64_norm: float
    unit_roundoff_upper_bound: float
    multiplicative_rounding_error_bound: float
    scale_before_nextafter: float
    applied_scale: float
    post_correction_fp64_norm: float | None
    gate_pass: bool
    exception_type: str | None = None
    exception_message: str | None = None

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "attempt_one_based": self.attempt_one_based,
            "pre_correction_fp64_norm": self.pre_correction_fp64_norm,
            "unit_roundoff_upper_bound": self.unit_roundoff_upper_bound,
            "multiplicative_rounding_error_bound": (
                self.multiplicative_rounding_error_bound
            ),
            "scale_before_nextafter": self.scale_before_nextafter,
            "applied_scale": self.applied_scale,
            "post_correction_fp64_norm": self.post_correction_fp64_norm,
            "gate_pass": self.gate_pass,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
        }


@dataclass(frozen=True, slots=True)
class StrictClipAudit:
    schema_version: str
    status: str
    max_norm: float | None
    absolute_tolerance: float
    max_correction_attempts: int
    live_gradient_tensor_count: int
    nonzero_gradient_tensor_count: int
    gradient_numel: int
    dtype_audits: tuple[StrictClipDTypeAudit, ...]
    pre_clip_fp64_norm: float | None
    strict_upper_bound: float | None
    torch_returned_norm: float | None
    torch_returned_norm_dtype: str | None
    torch_returned_norm_device: str | None
    torch_exception_type: str | None
    torch_exception_message: str | None
    post_torch_clip_fp64_norm: float | None
    correction_attempts: tuple[StrictClipCorrectionAttempt, ...]
    final_post_clip_fp64_norm: float | None
    final_gate_pass: bool
    failure_reason: str | None

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "max_norm": self.max_norm,
            "absolute_tolerance": self.absolute_tolerance,
            "max_correction_attempts": self.max_correction_attempts,
            "live_gradient_tensor_count": self.live_gradient_tensor_count,
            "nonzero_gradient_tensor_count": self.nonzero_gradient_tensor_count,
            "gradient_numel": self.gradient_numel,
            "dtype_audits": [value.as_dict() for value in self.dtype_audits],
            "pre_clip_fp64_norm": self.pre_clip_fp64_norm,
            "strict_upper_bound": self.strict_upper_bound,
            "torch_returned_norm": self.torch_returned_norm,
            "torch_returned_norm_dtype": self.torch_returned_norm_dtype,
            "torch_returned_norm_device": self.torch_returned_norm_device,
            "torch_exception_type": self.torch_exception_type,
            "torch_exception_message": self.torch_exception_message,
            "post_torch_clip_fp64_norm": self.post_torch_clip_fp64_norm,
            "correction_attempts": [
                value.as_dict() for value in self.correction_attempts
            ],
            "final_post_clip_fp64_norm": self.final_post_clip_fp64_norm,
            "final_gate_pass": self.final_gate_pass,
            "failure_reason": self.failure_reason,
        }


class StrictClipContractError(EngineContractError):
    """Strict stored-gradient clipping failed with an immutable audit."""

    def __init__(self, reason: str, audit: StrictClipAudit) -> None:
        self.reason = reason
        self.audit = audit
        super().__init__(
            {
                "reason": reason,
                "strict_clip_audit": audit.as_dict(),
            }
        )


@dataclass(frozen=True, slots=True)
class TrainingExample:
    case_id: str
    state: str
    source_dataset: str
    request: Any
    release: Callable[[], None] | None = None

    def validated(self) -> "TrainingExample":
        if (
            not isinstance(self.case_id, str)
            or not self.case_id
            or self.state not in STATE_FACTORS
            or not isinstance(self.source_dataset, str)
            or not self.source_dataset
        ):
            raise EngineContractError("training example identity is invalid")
        return self

    def close(self) -> None:
        if self.release is not None:
            self.release()


@dataclass(slots=True)
class StageRuntime:
    stage: str
    configuration: Any
    optimizer: torch.optim.AdamW
    scheduler: WarmupCosineToFloor
    active_parameters: tuple[tuple[str, nn.Parameter], ...]
    stage_optimizer_step: int = 0
    global_optimizer_step: int = 0
    micro_examples_seen: int = 0


@dataclass(frozen=True, slots=True)
class OptimizerBoundary:
    stage: str
    epoch_zero_based: int
    sampler_cursor: int
    examples_in_update: int
    stage_optimizer_step: int
    global_optimizer_step: int
    micro_examples_seen: int
    gradient_pre_clip_norm: float
    gradient_post_clip_norm: float
    gradient_clip_audit: StrictClipAudit
    scheduler_completed_steps: int
    loss_means: Mapping[str, float]


class EpochBoundaryCallback(Protocol):
    def __call__(
        self,
        runtime: StageRuntime,
        sampler: FullCoverageStateSampler,
        boundaries: Sequence[OptimizerBoundary],
    ) -> None: ...


class OptimizerBoundaryCallback(Protocol):
    def __call__(
        self,
        runtime: StageRuntime,
        sampler: FullCoverageStateSampler,
        boundary: OptimizerBoundary,
    ) -> None: ...


def _active_entries(configuration: Any) -> tuple[tuple[str, nn.Parameter], ...]:
    owner_configuration = configuration.owner_configuration
    allowed = set(owner_configuration.trainable_names)
    entries = tuple(
        (name, parameter)
        for name, parameter in owner_configuration.owners.all
        if name in allowed
    )
    if (
        not entries
        or len(entries) != len({name for name, _ in entries})
        or len(entries) != len({id(parameter) for _, parameter in entries})
        or any(not parameter.requires_grad for _, parameter in entries)
    ):
        raise EngineContractError("active stage parameter inventory is invalid")
    return entries


def all_method_parameters(backend: Any) -> tuple[tuple[str, nn.Parameter], ...]:
    configuration = backend.configure_mosder_stage("FROZEN")
    entries = tuple(configuration.owner_configuration.owners.all)
    if (
        not entries
        or len(entries) != len({name for name, _ in entries})
        or len(entries) != len({id(parameter) for _, parameter in entries})
    ):
        raise EngineContractError("complete MoSDeR owner inventory is invalid")
    return entries


def frozen_base_snapshot_digest(snapshot: Mapping[str, Any]) -> str:
    payload = [
        {
            "name": name,
            "shape": list(stamp.shape),
            "dtype": stamp.dtype,
            "numel": stamp.numel,
            "tensor_version": stamp.tensor_version,
            "digest": stamp.digest,
        }
        for name, stamp in sorted(snapshot.items())
    ]
    return canonical_sha256(payload)


class GradientWindow:
    """Average exact per-example gradients, including a partial final window."""

    def __init__(self, entries: Sequence[tuple[str, nn.Parameter]]) -> None:
        self.entries = tuple(entries)
        if not self.entries:
            raise EngineContractError("gradient window has no active parameters")
        self.buffers = {
            name: torch.zeros_like(parameter, dtype=torch.float32)
            for name, parameter in self.entries
        }
        self.count = 0
        self.ever_nonzero: set[str] = set()

    def capture_and_clear(self) -> None:
        live = 0
        for name, parameter in self.entries:
            gradient = parameter.grad
            if gradient is not None:
                live += 1
                if not bool(torch.isfinite(gradient).all()):
                    raise EngineContractError(f"non-finite gradient: {name}")
                self.buffers[name].add_(gradient.detach().to(torch.float32))
                if bool(torch.count_nonzero(gradient.detach())):
                    self.ever_nonzero.add(name)
            parameter.grad = None
        if live == 0:
            raise EngineContractError("backward produced no active-stage gradients")
        self.count += 1

    def install_mean(self) -> int:
        if self.count <= 0:
            raise EngineContractError("cannot flush an empty gradient window")
        count = self.count
        for name, parameter in self.entries:
            parameter.grad = (self.buffers[name] / float(count)).to(
                device=parameter.device, dtype=parameter.dtype
            )
        return count

    def reset_after_step(self) -> None:
        for name, parameter in self.entries:
            parameter.grad = None
            self.buffers[name].zero_()
        self.count = 0


def _fp64_global_gradient_norm(entries: Sequence[tuple[str, nn.Parameter]]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for _, parameter in entries:
        if parameter.grad is None:
            raise EngineContractError("strict clip gradient inventory is incomplete")
        gradient = parameter.grad.detach().to(torch.float64)
        total += torch.sum(gradient * gradient).cpu()
    return float(torch.sqrt(total))


def _dtype_audits(
    entries: Sequence[tuple[str, nn.Parameter]],
) -> tuple[StrictClipDTypeAudit, ...]:
    grouped: dict[torch.dtype, dict[str, int]] = {}
    for _, parameter in entries:
        gradient = parameter.grad
        if gradient is None:
            continue
        if gradient.is_sparse:
            raise EngineContractError("strict clip does not accept sparse gradients")
        if not gradient.is_floating_point():
            raise EngineContractError("strict clip requires floating gradients")
        nonzero_elements = int(torch.count_nonzero(gradient.detach()).cpu())
        values = grouped.setdefault(
            gradient.dtype,
            {
                "tensor_count": 0,
                "nonzero_tensor_count": 0,
                "gradient_numel": 0,
                "nonzero_element_count": 0,
            },
        )
        values["tensor_count"] += 1
        values["nonzero_tensor_count"] += int(nonzero_elements > 0)
        values["gradient_numel"] += int(gradient.numel())
        values["nonzero_element_count"] += nonzero_elements
    output = []
    for dtype in sorted(grouped, key=str):
        values = grouped[dtype]
        output.append(
            StrictClipDTypeAudit(
                dtype=str(dtype),
                tensor_count=values["tensor_count"],
                nonzero_tensor_count=values["nonzero_tensor_count"],
                gradient_numel=values["gradient_numel"],
                nonzero_element_count=values["nonzero_element_count"],
                unit_roundoff=float(torch.finfo(dtype).eps) / 2.0,
            )
        )
    return tuple(output)


def _make_clip_audit(
    *,
    status: str,
    max_norm: float | None,
    dtype_audits: tuple[StrictClipDTypeAudit, ...] = (),
    pre_clip_fp64_norm: float | None = None,
    strict_upper_bound: float | None = None,
    torch_returned_norm: float | None = None,
    torch_returned_norm_dtype: str | None = None,
    torch_returned_norm_device: str | None = None,
    torch_exception_type: str | None = None,
    torch_exception_message: str | None = None,
    post_torch_clip_fp64_norm: float | None = None,
    correction_attempts: tuple[StrictClipCorrectionAttempt, ...] = (),
    final_post_clip_fp64_norm: float | None = None,
    final_gate_pass: bool = False,
    failure_reason: str | None = None,
) -> StrictClipAudit:
    return StrictClipAudit(
        schema_version=STRICT_CLIP_AUDIT_SCHEMA_VERSION,
        status=status,
        max_norm=max_norm,
        absolute_tolerance=STRICT_CLIP_ABS_TOLERANCE,
        max_correction_attempts=STRICT_CLIP_MAX_CORRECTION_ATTEMPTS,
        live_gradient_tensor_count=sum(value.tensor_count for value in dtype_audits),
        nonzero_gradient_tensor_count=sum(
            value.nonzero_tensor_count for value in dtype_audits
        ),
        gradient_numel=sum(value.gradient_numel for value in dtype_audits),
        dtype_audits=dtype_audits,
        pre_clip_fp64_norm=pre_clip_fp64_norm,
        strict_upper_bound=strict_upper_bound,
        torch_returned_norm=torch_returned_norm,
        torch_returned_norm_dtype=torch_returned_norm_dtype,
        torch_returned_norm_device=torch_returned_norm_device,
        torch_exception_type=torch_exception_type,
        torch_exception_message=torch_exception_message,
        post_torch_clip_fp64_norm=post_torch_clip_fp64_norm,
        correction_attempts=correction_attempts,
        final_post_clip_fp64_norm=final_post_clip_fp64_norm,
        final_gate_pass=final_gate_pass,
        failure_reason=failure_reason,
    )


def _strict_clip_gate(pre_norm: float, post_norm: float, max_norm: float) -> bool:
    return (
        math.isfinite(pre_norm)
        and math.isfinite(post_norm)
        and pre_norm > 0.0
        and post_norm > 0.0
        and post_norm <= min(pre_norm, max_norm) + STRICT_CLIP_ABS_TOLERANCE
    )


def _raise_strict_clip(
    reason: str,
    *,
    max_norm: float | None,
    dtype_audits: tuple[StrictClipDTypeAudit, ...] = (),
    pre_clip_fp64_norm: float | None = None,
    strict_upper_bound: float | None = None,
    torch_returned_norm: float | None = None,
    torch_returned_norm_dtype: str | None = None,
    torch_returned_norm_device: str | None = None,
    torch_exception_type: str | None = None,
    torch_exception_message: str | None = None,
    post_torch_clip_fp64_norm: float | None = None,
    correction_attempts: tuple[StrictClipCorrectionAttempt, ...] = (),
    final_post_clip_fp64_norm: float | None = None,
) -> None:
    audit = _make_clip_audit(
        status="FAIL_STRICT_FP64_GRADIENT_CLIP",
        max_norm=max_norm,
        dtype_audits=dtype_audits,
        pre_clip_fp64_norm=pre_clip_fp64_norm,
        strict_upper_bound=strict_upper_bound,
        torch_returned_norm=torch_returned_norm,
        torch_returned_norm_dtype=torch_returned_norm_dtype,
        torch_returned_norm_device=torch_returned_norm_device,
        torch_exception_type=torch_exception_type,
        torch_exception_message=torch_exception_message,
        post_torch_clip_fp64_norm=post_torch_clip_fp64_norm,
        correction_attempts=correction_attempts,
        final_post_clip_fp64_norm=final_post_clip_fp64_norm,
        final_gate_pass=False,
        failure_reason=reason,
    )
    raise StrictClipContractError(reason, audit)


def strict_clip_grad_norm_(
    entries: Sequence[tuple[str, nn.Parameter]],
    *,
    max_norm: float,
) -> StrictClipAudit:
    """Clip once with PyTorch, then enforce the stored-gradient FP64 gate.

    PyTorch's returned norm is retained only as audit evidence.  If mixed or
    low-precision in-place rounding leaves the actual stored FP64 norm above
    the unchanged gate, at most two deterministic conservative rescalings are
    permitted.  The rescaling bound assumes at most two dtype-level roundings;
    the final independently recomputed FP64 gate remains authoritative.
    """

    materialized = tuple(entries)
    stored_max_norm = (
        float(max_norm)
        if isinstance(max_norm, (int, float)) and math.isfinite(float(max_norm))
        else None
    )
    if isinstance(max_norm, bool) or stored_max_norm is None or stored_max_norm <= 0.0:
        _raise_strict_clip("max_norm must be positive and finite", max_norm=None)
    if (
        not materialized
        or len({name for name, _ in materialized}) != len(materialized)
        or len({id(parameter) for _, parameter in materialized}) != len(materialized)
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(parameter, nn.Parameter)
            for name, parameter in materialized
        )
    ):
        _raise_strict_clip(
            "strict clip parameter inventory is invalid", max_norm=stored_max_norm
        )
    present = tuple(entry for entry in materialized if entry[1].grad is not None)
    try:
        dtype_audits = _dtype_audits(present)
    except Exception as error:
        _raise_strict_clip(
            "strict clip dtype audit raised",
            max_norm=stored_max_norm,
            torch_exception_type=type(error).__name__,
            torch_exception_message=str(error),
        )
    if len(present) != len(materialized):
        _raise_strict_clip(
            "strict clip active gradient inventory is incomplete",
            max_norm=stored_max_norm,
            dtype_audits=dtype_audits,
        )
    try:
        pre_norm = _fp64_global_gradient_norm(materialized)
    except Exception as error:
        _raise_strict_clip(
            "pre-clip FP64 stored-gradient norm computation raised",
            max_norm=stored_max_norm,
            dtype_audits=dtype_audits,
            torch_exception_type=type(error).__name__,
            torch_exception_message=str(error),
        )
    finite_pre = pre_norm if math.isfinite(pre_norm) else None
    if not math.isfinite(pre_norm) or pre_norm <= 0.0:
        _raise_strict_clip(
            "pre-clip FP64 stored-gradient norm must be positive and finite",
            max_norm=stored_max_norm,
            dtype_audits=dtype_audits,
            pre_clip_fp64_norm=finite_pre,
        )
    upper_bound = min(pre_norm, stored_max_norm)
    parameters = [parameter for _, parameter in materialized]
    try:
        returned = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=stored_max_norm,
            error_if_nonfinite=True,
            foreach=None,
        )
    except Exception as error:
        _raise_strict_clip(
            "torch clip_grad_norm_ raised",
            max_norm=stored_max_norm,
            dtype_audits=dtype_audits,
            pre_clip_fp64_norm=pre_norm,
            strict_upper_bound=upper_bound,
            torch_exception_type=type(error).__name__,
            torch_exception_message=str(error),
        )
    if not isinstance(returned, Tensor) or returned.numel() != 1:
        _raise_strict_clip(
            "torch clip_grad_norm_ returned a non-scalar tensor",
            max_norm=stored_max_norm,
            dtype_audits=dtype_audits,
            pre_clip_fp64_norm=pre_norm,
            strict_upper_bound=upper_bound,
            torch_exception_type=type(returned).__name__,
            torch_exception_message="return value is not a scalar Tensor",
        )
    returned_dtype = str(returned.dtype)
    returned_device = str(returned.device)
    try:
        returned_value = float(returned.detach().cpu())
    except Exception as error:
        _raise_strict_clip(
            "torch clip_grad_norm_ return observation raised",
            max_norm=stored_max_norm,
            dtype_audits=dtype_audits,
            pre_clip_fp64_norm=pre_norm,
            strict_upper_bound=upper_bound,
            torch_returned_norm_dtype=returned_dtype,
            torch_returned_norm_device=returned_device,
            torch_exception_type=type(error).__name__,
            torch_exception_message=str(error),
        )
    if not math.isfinite(returned_value) or returned_value < 0.0:
        _raise_strict_clip(
            "torch clip_grad_norm_ returned a non-finite or negative norm",
            max_norm=stored_max_norm,
            dtype_audits=dtype_audits,
            pre_clip_fp64_norm=pre_norm,
            strict_upper_bound=upper_bound,
            torch_returned_norm=(
                returned_value if math.isfinite(returned_value) else None
            ),
            torch_returned_norm_dtype=returned_dtype,
            torch_returned_norm_device=returned_device,
        )

    try:
        post_torch = _fp64_global_gradient_norm(materialized)
    except Exception as error:
        _raise_strict_clip(
            "post-torch FP64 stored-gradient norm computation raised",
            max_norm=stored_max_norm,
            dtype_audits=dtype_audits,
            pre_clip_fp64_norm=pre_norm,
            strict_upper_bound=upper_bound,
            torch_returned_norm=returned_value,
            torch_returned_norm_dtype=returned_dtype,
            torch_returned_norm_device=returned_device,
            torch_exception_type=type(error).__name__,
            torch_exception_message=str(error),
        )
    current_post = post_torch
    attempts: list[StrictClipCorrectionAttempt] = []
    if not math.isfinite(current_post) or current_post <= 0.0:
        _raise_strict_clip(
            "post-torch FP64 stored-gradient norm must be positive and finite",
            max_norm=stored_max_norm,
            dtype_audits=dtype_audits,
            pre_clip_fp64_norm=pre_norm,
            strict_upper_bound=upper_bound,
            torch_returned_norm=returned_value,
            torch_returned_norm_dtype=returned_dtype,
            torch_returned_norm_device=returned_device,
            post_torch_clip_fp64_norm=(
                current_post if math.isfinite(current_post) else None
            ),
            final_post_clip_fp64_norm=(
                current_post if math.isfinite(current_post) else None
            ),
        )

    if not _strict_clip_gate(pre_norm, current_post, stored_max_norm):
        unit_roundoff = max(value.unit_roundoff for value in dtype_audits)
        # Upper multiplier for scalar representation plus stored-gradient
        # rounding.  Any inadequacy in this model is caught by the independent
        # FP64 recomputation and the bounded second attempt.
        rounding_bound = (1.0 + unit_roundoff) ** 2
        for attempt_index in range(1, STRICT_CLIP_MAX_CORRECTION_ATTEMPTS + 1):
            scale_before_nextafter = upper_bound / (current_post * rounding_bound)
            applied_scale = math.nextafter(scale_before_nextafter, 0.0)
            if (
                not math.isfinite(applied_scale)
                or applied_scale <= 0.0
                or applied_scale >= 1.0
            ):
                _raise_strict_clip(
                    "strict clip conservative correction scale is invalid",
                    max_norm=stored_max_norm,
                    dtype_audits=dtype_audits,
                    pre_clip_fp64_norm=pre_norm,
                    strict_upper_bound=upper_bound,
                    torch_returned_norm=returned_value,
                    torch_returned_norm_dtype=returned_dtype,
                    torch_returned_norm_device=returned_device,
                    post_torch_clip_fp64_norm=post_torch,
                    correction_attempts=tuple(attempts),
                    final_post_clip_fp64_norm=current_post,
                )
            try:
                with torch.no_grad():
                    for _, parameter in materialized:
                        if parameter.grad is None:
                            raise EngineContractError(
                                "gradient disappeared during strict correction"
                            )
                        parameter.grad.mul_(applied_scale)
                corrected_post = _fp64_global_gradient_norm(materialized)
            except Exception as error:
                attempts.append(
                    StrictClipCorrectionAttempt(
                        attempt_one_based=attempt_index,
                        pre_correction_fp64_norm=current_post,
                        unit_roundoff_upper_bound=unit_roundoff,
                        multiplicative_rounding_error_bound=rounding_bound,
                        scale_before_nextafter=scale_before_nextafter,
                        applied_scale=applied_scale,
                        post_correction_fp64_norm=None,
                        gate_pass=False,
                        exception_type=type(error).__name__,
                        exception_message=str(error),
                    )
                )
                _raise_strict_clip(
                    "strict clip conservative correction raised",
                    max_norm=stored_max_norm,
                    dtype_audits=dtype_audits,
                    pre_clip_fp64_norm=pre_norm,
                    strict_upper_bound=upper_bound,
                    torch_returned_norm=returned_value,
                    torch_returned_norm_dtype=returned_dtype,
                    torch_returned_norm_device=returned_device,
                    post_torch_clip_fp64_norm=post_torch,
                    correction_attempts=tuple(attempts),
                    final_post_clip_fp64_norm=current_post,
                )
            attempt_pass = _strict_clip_gate(pre_norm, corrected_post, stored_max_norm)
            attempts.append(
                StrictClipCorrectionAttempt(
                    attempt_one_based=attempt_index,
                    pre_correction_fp64_norm=current_post,
                    unit_roundoff_upper_bound=unit_roundoff,
                    multiplicative_rounding_error_bound=rounding_bound,
                    scale_before_nextafter=scale_before_nextafter,
                    applied_scale=applied_scale,
                    post_correction_fp64_norm=(
                        corrected_post if math.isfinite(corrected_post) else None
                    ),
                    gate_pass=attempt_pass,
                )
            )
            current_post = corrected_post
            if attempt_pass:
                break

    final_pass = _strict_clip_gate(pre_norm, current_post, stored_max_norm)
    if not final_pass:
        _raise_strict_clip(
            "strict FP64 stored-gradient clip gate failed after bounded correction",
            max_norm=stored_max_norm,
            dtype_audits=dtype_audits,
            pre_clip_fp64_norm=pre_norm,
            strict_upper_bound=upper_bound,
            torch_returned_norm=returned_value,
            torch_returned_norm_dtype=returned_dtype,
            torch_returned_norm_device=returned_device,
            post_torch_clip_fp64_norm=post_torch,
            correction_attempts=tuple(attempts),
            final_post_clip_fp64_norm=(
                current_post if math.isfinite(current_post) else None
            ),
        )
    return _make_clip_audit(
        status="PASS_STRICT_FP64_GRADIENT_CLIP",
        max_norm=stored_max_norm,
        dtype_audits=dtype_audits,
        pre_clip_fp64_norm=pre_norm,
        strict_upper_bound=upper_bound,
        torch_returned_norm=returned_value,
        torch_returned_norm_dtype=returned_dtype,
        torch_returned_norm_device=returned_device,
        post_torch_clip_fp64_norm=post_torch,
        correction_attempts=tuple(attempts),
        final_post_clip_fp64_norm=current_post,
        final_gate_pass=True,
    )


class MoSDeRExecutionAdapter:
    """Actual family-neutral MoSDeR backward implementation."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        if not isinstance(getattr(backend, "model", None), nn.Module):
            raise EngineContractError("backend has no live model")
        backend.model.eval()
        if backend.model.training:
            raise EngineContractError("MoSDeR requires frozen VLM eval mode")
        self.method_parameters = all_method_parameters(backend)
        self.frozen_base_snapshot = snapshot_mosder_frozen_base(
            backend.model, mode="sampled"
        )
        self.frozen_base_digest = frozen_base_snapshot_digest(self.frozen_base_snapshot)

    def configure_stage(
        self,
        stage: str,
        *,
        total_optimizer_steps: int,
        protocol: RunProtocol,
        global_optimizer_step: int = 0,
    ) -> StageRuntime:
        if stage not in STAGES:
            raise EngineContractError("unknown F/G/R stage")
        configuration = self.backend.configure_mosder_stage(stage)
        optimizer = build_stage_adamw(configuration.owner_configuration)
        scheduler = WarmupCosineToFloor(
            optimizer,
            total_steps=total_optimizer_steps,
            warmup_fraction=protocol.warmup_fraction,
            floor=protocol.cosine_floor,
        )
        optimizer.zero_grad(set_to_none=True)
        return StageRuntime(
            stage=stage,
            configuration=configuration,
            optimizer=optimizer,
            scheduler=scheduler,
            active_parameters=_active_entries(configuration),
            global_optimizer_step=global_optimizer_step,
        )

    def backward_one(
        self,
        runtime: StageRuntime,
        example: TrainingExample,
    ) -> Mapping[str, float]:
        example.validated()
        if runtime.stage == "F":
            camera_moving, object_moving = STATE_FACTORS[example.state]
            result = exact_two_pass_stage_f_backward(
                self.backend,
                example.request,
                configuration=runtime.configuration,
                candidates=StageFCandidateTexts(
                    neither=canonical_answer("neither"),
                    camera_only=canonical_answer("camera_only"),
                    object_only=canonical_answer("object_only"),
                    both=canonical_answer("both"),
                ),
                camera_moving=camera_moving,
                object_moving=object_moving,
            )
            return {
                "total": float(result.losses.total.cpu()),
                "camera": float(result.losses.camera.cpu()),
                "object": float(result.losses.object.cpu()),
            }
        teacher = canonical_answer(example.state)
        plan = plan_canonical_language(self.backend, example.request, example.state)
        score = self.backend.teacher_forced_loss(
            example.request,
            teacher,
            route=FactorRoute.FULL_LANGUAGE,
        )
        validate_score_against_plan(score, plan)
        losses = (
            stage_g_language_loss(score.token_log_probabilities, plan.spans)
            if runtime.stage == "G"
            else stage_r_language_loss(score.token_log_probabilities, plan.spans)
        )
        if losses.total.ndim != 0 or not bool(torch.isfinite(losses.total)):
            raise EngineContractError(f"Stage-{runtime.stage} loss is invalid")
        losses.total.backward()
        return {
            name: float(value.detach().cpu())
            for name, value in losses.as_mapping().items()
        }

    def assert_no_foreign_gradients(self, runtime: StageRuntime) -> None:
        allowed = {id(parameter) for _, parameter in runtime.active_parameters}
        foreign = [
            name
            for name, parameter in self.backend.model.named_parameters()
            if parameter.grad is not None and id(parameter) not in allowed
        ]
        if foreign:
            raise EngineContractError(
                {"outside_stage_allowlist_gradients": foreign[:32]}
            )

    def assert_integrity(self, runtime: StageRuntime) -> None:
        if self.backend.model.training:
            raise EngineContractError("native VLM left eval mode")
        # The sealed optimizer auditor verifies construction-time base LRs,
        # while the explicit scheduler necessarily changes live group LRs.
        # Audit against the base recipe without changing the saved trajectory.
        scheduled_lrs = [float(group["lr"]) for group in runtime.optimizer.param_groups]
        try:
            for group, base_lr in zip(
                runtime.optimizer.param_groups,
                runtime.scheduler.base_lrs,
                strict=True,
            ):
                group["lr"] = base_lr
            audit_stage_adamw(
                runtime.optimizer,
                runtime.configuration.owner_configuration,
                state_mode="initialized"
                if runtime.stage_optimizer_step
                else "pristine",
            )
        finally:
            for group, scheduled_lr in zip(
                runtime.optimizer.param_groups, scheduled_lrs, strict=True
            ):
                group["lr"] = scheduled_lr
        assert_mosder_frozen_base_unchanged(
            self.backend.model, self.frozen_base_snapshot, mode="sampled"
        )
        nonfinite = [
            name
            for name, parameter in self.method_parameters
            if not bool(torch.isfinite(parameter.detach()).all())
        ]
        if nonfinite:
            raise EngineContractError({"nonfinite_method_parameters": nonfinite[:32]})


def stage_total_optimizer_steps(
    row_count: int, *, epochs: int, accumulation: int
) -> int:
    if any(
        type(value) is not int or value <= 0
        for value in (row_count, epochs, accumulation)
    ):
        raise EngineContractError("stage schedule values must be positive integers")
    return math.ceil(row_count / accumulation) * epochs


def run_current_epoch(
    adapter: MoSDeRExecutionAdapter,
    runtime: StageRuntime,
    sampler: FullCoverageStateSampler,
    examples: Sequence[TrainingExample],
    *,
    accumulation: int,
    gradient_clip_max_norm: float,
    on_optimizer_boundary: OptimizerBoundaryCallback | None = None,
    on_epoch_boundary: EpochBoundaryCallback | None = None,
) -> tuple[OptimizerBoundary, ...]:
    """Train from the current cursor through exactly one epoch boundary."""

    if runtime.stage not in STAGES or sampler.exhausted:
        raise EngineContractError("stage/sampler cannot run an epoch")
    if len(examples) != len(sampler.rows):
        raise EngineContractError("examples and sampler membership differ")
    if type(accumulation) is not int or accumulation <= 0:
        raise EngineContractError("accumulation must be positive int")
    if not math.isfinite(gradient_clip_max_norm) or gradient_clip_max_norm <= 0:
        raise EngineContractError("gradient clip max norm is invalid")
    starting_epoch = sampler.epoch
    window = GradientWindow(runtime.active_parameters)
    loss_sums: dict[str, float] = {}
    boundaries: list[OptimizerBoundary] = []
    while not sampler.at_epoch_end:
        index = sampler.next_index()
        if index is None:
            raise EngineContractError("sampler exhausted before epoch boundary")
        example = examples[index]
        try:
            losses = adapter.backward_one(runtime, example)
            adapter.assert_no_foreign_gradients(runtime)
            window.capture_and_clear()
        finally:
            example.close()
        runtime.micro_examples_seen += 1
        for name, value in losses.items():
            if not math.isfinite(value):
                raise EngineContractError(f"non-finite logged loss: {name}")
            loss_sums[name] = loss_sums.get(name, 0.0) + value
        if window.count < accumulation and not sampler.at_epoch_end:
            continue
        examples_in_update = window.install_mean()
        clip_audit = strict_clip_grad_norm_(
            runtime.active_parameters,
            max_norm=gradient_clip_max_norm,
        )
        pre_norm = clip_audit.pre_clip_fp64_norm
        post_norm = clip_audit.final_post_clip_fp64_norm
        if pre_norm is None or post_norm is None or not clip_audit.final_gate_pass:
            raise EngineContractError(
                "strict clip returned an incomplete passing audit"
            )
        runtime.optimizer.step()
        runtime.scheduler.step()
        runtime.stage_optimizer_step += 1
        runtime.global_optimizer_step += 1
        runtime.optimizer.zero_grad(set_to_none=True)
        window.reset_after_step()
        boundary = OptimizerBoundary(
            stage=runtime.stage,
            epoch_zero_based=starting_epoch,
            sampler_cursor=sampler.cursor,
            examples_in_update=examples_in_update,
            stage_optimizer_step=runtime.stage_optimizer_step,
            global_optimizer_step=runtime.global_optimizer_step,
            micro_examples_seen=runtime.micro_examples_seen,
            gradient_pre_clip_norm=pre_norm,
            gradient_post_clip_norm=post_norm,
            gradient_clip_audit=clip_audit,
            scheduler_completed_steps=runtime.scheduler.completed_steps,
            loss_means={
                name: value / float(examples_in_update)
                for name, value in loss_sums.items()
            },
        )
        boundaries.append(boundary)
        loss_sums.clear()
        if on_optimizer_boundary is not None:
            on_optimizer_boundary(runtime, sampler, boundary)
    if window.count != 0 or any(
        parameter.grad is not None for _, parameter in runtime.active_parameters
    ):
        raise EngineContractError("epoch ended with a partial gradient window")
    if not boundaries and sampler.cursor > 0:
        raise EngineContractError("epoch segment committed no optimizer update")
    adapter.assert_integrity(runtime)
    if on_epoch_boundary is not None:
        on_epoch_boundary(runtime, sampler, boundaries)
    if sampler.epoch != starting_epoch or not sampler.at_epoch_end:
        raise EngineContractError("epoch callback changed the training cursor")
    sampler.finish_epoch()
    return tuple(boundaries)


__all__ = [
    "EngineContractError",
    "GradientWindow",
    "MoSDeRExecutionAdapter",
    "OptimizerBoundary",
    "STRICT_CLIP_ABS_TOLERANCE",
    "STRICT_CLIP_AUDIT_SCHEMA_VERSION",
    "STRICT_CLIP_MAX_CORRECTION_ATTEMPTS",
    "StageRuntime",
    "StrictClipAudit",
    "StrictClipContractError",
    "StrictClipCorrectionAttempt",
    "StrictClipDTypeAudit",
    "TrainingExample",
    "all_method_parameters",
    "frozen_base_snapshot_digest",
    "run_current_epoch",
    "stage_total_optimizer_steps",
    "strict_clip_grad_norm_",
]
