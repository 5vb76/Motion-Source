"""Factor classification loss for F and four-line language loss for G/R.

F sums camera and object binary losses. G/R average the token NLL within
each answer field, then give the four fields equal weight."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .routing import FactorLogits


LANGUAGE_SPAN_ORDER = ("camera", "object", "state", "description")
LANGUAGE_SPAN_WEIGHT = 0.25


class ObjectiveContractError(RuntimeError):
    """A factor target, native score, or language span violated its ABI."""


@dataclass(frozen=True, slots=True)
class StageFFactorLoss:
    """Two equally present BCE-logits factors; ``total`` is their sum."""

    camera: Tensor
    object: Tensor
    total: Tensor


@dataclass(frozen=True, slots=True)
class FourLineLanguageLoss:
    """Mean token NLL for each line and their exact equal-weight total."""

    camera: Tensor
    object: Tensor
    state: Tensor
    description: Tensor
    total: Tensor

    def as_mapping(self) -> Mapping[str, Tensor]:
        return {
            "camera": self.camera,
            "object": self.object,
            "state": self.state,
            "description": self.description,
        }


def _require_finite_floating_tensor(value: object, *, name: str) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ObjectiveContractError(f"{name} must be a floating tensor")
    if value.numel() == 0:
        raise ObjectiveContractError(f"{name} must be nonempty")
    if not bool(torch.isfinite(value).all()):
        raise ObjectiveContractError(f"{name} contains non-finite values")
    return value


def _require_binary_target(
    value: object,
    *,
    name: str,
    reference: Tensor,
) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype is not torch.bool:
        raise ObjectiveContractError(f"{name} must be a boolean tensor")
    if value.shape != reference.shape:
        raise ObjectiveContractError(
            f"{name} shape {tuple(value.shape)} differs from logit shape "
            f"{tuple(reference.shape)}"
        )
    if value.device != reference.device:
        raise ObjectiveContractError(f"{name} and its logit must share one device")
    return value


def _require_scalar_finite(value: Tensor, *, name: str) -> Tensor:
    if value.ndim != 0 or not bool(torch.isfinite(value)):
        raise ObjectiveContractError(f"{name} must be one finite scalar tensor")
    return value


def stage_f_factor_loss(
    logits: FactorLogits,
    *,
    camera_moving: Tensor,
    object_moving: Tensor,
) -> StageFFactorLoss:
    """Return the Stage-F Camera/Object softplus losses without detaching.

    ``camera_moving`` and ``object_moving`` are strict boolean tensors with the
    same shape/device as their logits.  For a target sign ``z`` in ``{-1,+1}``,
    each factor is ``mean(softplus(-z*q))``, exactly the stable
    binary-cross-entropy-with-logits objective.  The two factor means are added,
    matching the source-separated Stage-F contract.
    """

    if not isinstance(logits, FactorLogits):
        raise ObjectiveContractError("logits must be a FactorLogits instance")
    camera_logit = _require_finite_floating_tensor(
        logits.camera, name="Camera factor logits"
    )
    object_logit = _require_finite_floating_tensor(
        logits.object, name="Object factor logits"
    )
    if (
        camera_logit.shape != object_logit.shape
        or camera_logit.device != object_logit.device
        or camera_logit.dtype != object_logit.dtype
    ):
        raise ObjectiveContractError(
            "Camera/Object factor logits must share shape, device, and dtype"
        )
    camera_target = _require_binary_target(
        camera_moving,
        name="Camera factor target",
        reference=camera_logit,
    )
    object_target = _require_binary_target(
        object_moving,
        name="Object factor target",
        reference=object_logit,
    )

    camera_sign = camera_target.to(dtype=camera_logit.dtype).mul(2).sub(1)
    object_sign = object_target.to(dtype=object_logit.dtype).mul(2).sub(1)
    camera = F.softplus(-camera_sign * camera_logit).mean()
    obj = F.softplus(-object_sign * object_logit).mean()
    total = camera + obj
    return StageFFactorLoss(
        camera=_require_scalar_finite(camera, name="Camera factor loss"),
        object=_require_scalar_finite(obj, name="Object factor loss"),
        total=_require_scalar_finite(total, name="Stage-F total loss"),
    )


def _validated_language_spans(
    token_log_probabilities: Tensor,
    spans: Mapping[str, tuple[int, int]],
) -> Mapping[str, tuple[int, int]]:
    if not isinstance(spans, Mapping):
        raise ObjectiveContractError("language spans must be a mapping")
    if tuple(spans) != LANGUAGE_SPAN_ORDER:
        raise ObjectiveContractError(
            "language spans must appear exactly as camera/object/state/description"
        )
    cursor = 0
    validated: dict[str, tuple[int, int]] = {}
    for name in LANGUAGE_SPAN_ORDER:
        bounds = spans[name]
        if (
            not isinstance(bounds, tuple)
            or len(bounds) != 2
            or type(bounds[0]) is not int
            or type(bounds[1]) is not int
        ):
            raise ObjectiveContractError(
                f"{name} language span must be one integer (start,end) tuple"
            )
        start, end = bounds
        if start != cursor or end <= start or end > token_log_probabilities.numel():
            raise ObjectiveContractError(
                "language spans must be one ordered, contiguous, nonempty partition"
            )
        validated[name] = (start, end)
        cursor = end
    if cursor != token_log_probabilities.numel():
        raise ObjectiveContractError(
            "language spans must cover every answer token exactly once"
        )
    return validated


def four_line_language_span_loss(
    token_log_probabilities: Tensor,
    spans: Mapping[str, tuple[int, int]],
) -> FourLineLanguageLoss:
    """Return equal Camera/Object/State/Description mean-span NLL.

    Each line contributes exactly ``0.25`` regardless of its token length.
    The input must be the differentiable one-dimensional native answer-token
    log-probability tensor, and the four spans must exactly partition it.
    """

    values = _require_finite_floating_tensor(
        token_log_probabilities,
        name="native answer token log probabilities",
    )
    if values.ndim != 1:
        raise ObjectiveContractError(
            "native answer token log probabilities must be one-dimensional"
        )
    if bool((values > 0).any()):
        raise ObjectiveContractError("a token log probability is greater than zero")
    # Native logits may be BF16/FP16; perform every reduction in FP32 while
    # preserving the differentiable path to the routed adapters.
    values = values.to(dtype=torch.float32)
    validated = _validated_language_spans(values, spans)
    line_losses: dict[str, Tensor] = {}
    for name in LANGUAGE_SPAN_ORDER:
        start, end = validated[name]
        line_losses[name] = _require_scalar_finite(
            -values[start:end].mean(),
            name=f"{name} language-span NLL",
        )
    total = sum(
        (LANGUAGE_SPAN_WEIGHT * line_losses[name] for name in LANGUAGE_SPAN_ORDER),
        values.new_zeros(()),
    )
    return FourLineLanguageLoss(
        camera=line_losses["camera"],
        object=line_losses["object"],
        state=line_losses["state"],
        description=line_losses["description"],
        total=_require_scalar_finite(total, name="four-line language loss"),
    )


def stage_g_language_loss(
    token_log_probabilities: Tensor,
    spans: Mapping[str, tuple[int, int]],
) -> FourLineLanguageLoss:
    """Stage-G Shared-TriLoRA objective."""

    return four_line_language_span_loss(token_log_probabilities, spans)


def stage_r_language_loss(
    token_log_probabilities: Tensor,
    spans: Mapping[str, tuple[int, int]],
) -> FourLineLanguageLoss:
    """Stage-R explicit-residual objective; numerically identical to Stage G."""

    return four_line_language_span_loss(token_log_probabilities, spans)


__all__ = [
    "FourLineLanguageLoss",
    "LANGUAGE_SPAN_ORDER",
    "LANGUAGE_SPAN_WEIGHT",
    "ObjectiveContractError",
    "StageFFactorLoss",
    "four_line_language_span_loss",
    "stage_f_factor_loss",
    "stage_g_language_loss",
    "stage_r_language_loss",
]
