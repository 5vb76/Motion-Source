"""Exact low-memory Stage-F backward for the six frozen factor-span calls.

The ordinary Stage-F expression keeps two Camera-route and four Object-route
native VLM graphs alive at once.  This module computes the same gradient with a
deterministic two-pass procedure:

1. evaluate all six scalar mean answer-span scores under ``torch.no_grad`` and
   derive the two logits, the real softplus objective, and analytic dL/dscore;
2. replay one candidate at a time, verify its complete native score record,
   and immediately backpropagate ``dL/dscore * score`` before releasing it.

The learned intercepts are handled by backpropagating the real Stage-F
objective through ``FactorSpanDecision`` with detached margins.  The routine
never constructs an optimizer and never performs a parameter update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .family_backend import BackendStageConfiguration
from .language import canonical_answer
from .objectives import StageFFactorLoss, stage_f_factor_loss
from .routing import (
    FactorLogits,
    FactorMargins,
    FactorRoute,
    FactorSpanDecision,
    FactorSpanRouteScores,
    factor_span_margins,
)
from .training import TrainingContractError, TrainingStage


class FactorTrainingContractError(TrainingContractError):
    """The exact two-pass Stage-F execution contract was violated."""


@dataclass(frozen=True, slots=True)
class StageFCandidateTexts:
    """The four fixed state candidates used by the 2+4 FactorSpan calls."""

    neither: str
    camera_only: str
    object_only: str
    both: str

    def validated(self) -> "StageFCandidateTexts":
        values = (self.neither, self.camera_only, self.object_only, self.both)
        if any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in values
        ):
            raise FactorTrainingContractError(
                "Stage-F candidates must be nonempty, already-stripped strings"
            )
        if len(set(values)) != len(values):
            raise FactorTrainingContractError(
                "the four Stage-F state candidates must be distinct"
            )
        expected = tuple(
            canonical_answer(state)
            for state in ("neither", "camera_only", "object_only", "both")
        )
        if values != expected:
            raise FactorTrainingContractError(
                "Stage-F candidates must exactly match the canonical state mapping"
            )
        return self


@dataclass(frozen=True, slots=True)
class StageFScoreCoefficients:
    """Analytic derivatives of total Stage-F loss with respect to six scores."""

    camera_neither: Tensor
    camera_camera_only: Tensor
    object_neither: Tensor
    object_camera_only: Tensor
    object_object_only: Tensor
    object_both: Tensor


@dataclass(frozen=True, slots=True)
class ExactStageFTwoPassResult:
    """Detached audit record left after exact gradient accumulation."""

    scores: FactorSpanRouteScores
    logits: FactorLogits
    losses: StageFFactorLoss
    coefficients: StageFScoreCoefficients
    route_call_delta: Mapping[str, int]
    maximum_replay_score_error: float
    parameter_versions_unchanged: bool


@dataclass(frozen=True, slots=True)
class _CandidateCall:
    field: str
    text: str
    route: FactorRoute


@dataclass(frozen=True, slots=True)
class _ScoreObservation:
    scalar: Tensor
    answer_token_ids: Tensor
    token_log_probabilities: Tensor
    text: str


def _candidate_calls(candidates: StageFCandidateTexts) -> tuple[_CandidateCall, ...]:
    return (
        _CandidateCall("camera_neither", candidates.neither, FactorRoute.CAMERA_FACTOR),
        _CandidateCall(
            "camera_camera_only",
            candidates.camera_only,
            FactorRoute.CAMERA_FACTOR,
        ),
        _CandidateCall("object_neither", candidates.neither, FactorRoute.OBJECT_FACTOR),
        _CandidateCall(
            "object_camera_only",
            candidates.camera_only,
            FactorRoute.OBJECT_FACTOR,
        ),
        _CandidateCall(
            "object_object_only",
            candidates.object_only,
            FactorRoute.OBJECT_FACTOR,
        ),
        _CandidateCall("object_both", candidates.both, FactorRoute.OBJECT_FACTOR),
    )


def _route_counts(backend: Any) -> dict[str, int]:
    getter = getattr(backend, "route_success_counts", None)
    if not callable(getter):
        raise FactorTrainingContractError(
            "Stage-F backend must expose route_success_counts()"
        )
    values = getter()
    expected = tuple(route.value for route in FactorRoute)
    if (
        not isinstance(values, Mapping)
        or set(values) != set(expected)
        or any(type(values[name]) is not int or values[name] < 0 for name in expected)
    ):
        raise FactorTrainingContractError("Stage-F route counters are invalid")
    return {name: values[name] for name in expected}


def _observe_score(
    backend: Any,
    request: Any,
    call: _CandidateCall,
    *,
    require_grad: bool,
) -> _ScoreObservation:
    evaluator = getattr(backend, "teacher_forced_loss", None)
    if not callable(evaluator):
        raise FactorTrainingContractError(
            "Stage-F backend must expose teacher_forced_loss()"
        )
    value = evaluator(request, call.text, route=call.route)
    text = getattr(value, "text", None)
    token_ids = getattr(value, "answer_token_ids", None)
    token_scores = getattr(value, "token_log_probabilities", None)
    if text != call.text:
        raise FactorTrainingContractError(
            f"native evaluator changed candidate text for {call.field}"
        )
    if (
        not isinstance(token_ids, Tensor)
        or token_ids.ndim != 1
        or token_ids.numel() == 0
        or token_ids.dtype != torch.long
    ):
        raise FactorTrainingContractError(
            f"native answer-token IDs are invalid for {call.field}"
        )
    if (
        not isinstance(token_scores, Tensor)
        or not token_scores.is_floating_point()
        or token_scores.ndim != 1
        or token_scores.shape != token_ids.shape
        or token_scores.device != token_ids.device
        or not bool(torch.isfinite(token_scores).all())
        or bool((token_scores > 0).any())
    ):
        raise FactorTrainingContractError(
            f"native answer-token scores are invalid for {call.field}"
        )
    scalar = token_scores.to(dtype=torch.float32).mean()
    if scalar.ndim != 0 or not bool(torch.isfinite(scalar)):
        raise FactorTrainingContractError(
            f"native mean answer-span score is invalid for {call.field}"
        )
    if require_grad and (not scalar.requires_grad or scalar.grad_fn is None):
        raise FactorTrainingContractError(
            f"replayed native score has no Stage-F graph for {call.field}"
        )
    if not require_grad and scalar.requires_grad:
        raise FactorTrainingContractError(
            f"pass-1 score unexpectedly retained a graph for {call.field}"
        )
    return _ScoreObservation(
        scalar=scalar,
        answer_token_ids=token_ids.detach().clone(),
        token_log_probabilities=token_scores.detach().to(torch.float32).clone(),
        text=text,
    )


def _factor_scores(
    observations: Mapping[str, _ScoreObservation],
) -> FactorSpanRouteScores:
    expected = {
        "camera_neither",
        "camera_camera_only",
        "object_neither",
        "object_camera_only",
        "object_object_only",
        "object_both",
    }
    if set(observations) != expected:
        raise FactorTrainingContractError("Stage-F score registry is incomplete")
    return FactorSpanRouteScores(
        camera_neither=observations["camera_neither"].scalar,
        camera_camera_only=observations["camera_camera_only"].scalar,
        object_neither=observations["object_neither"].scalar,
        object_camera_only=observations["object_camera_only"].scalar,
        object_object_only=observations["object_object_only"].scalar,
        object_both=observations["object_both"].scalar,
    )


def _target(value: bool, reference: Tensor) -> Tensor:
    if type(value) is not bool:
        raise FactorTrainingContractError("Stage-F moving targets must be bool")
    return torch.tensor(value, dtype=torch.bool, device=reference.device)


def _analytic_coefficients(
    logits: FactorLogits,
    *,
    camera_moving: bool,
    object_moving: bool,
) -> StageFScoreCoefficients:
    if logits.camera.ndim != 0 or logits.object.ndim != 0:
        raise FactorTrainingContractError(
            "low-memory Stage-F requires one scalar logit per RGB20 request"
        )
    camera_sign = 1.0 if camera_moving else -1.0
    object_sign = 1.0 if object_moving else -1.0
    d_camera = -camera_sign * torch.sigmoid(-camera_sign * logits.camera.detach())
    d_object = -object_sign * torch.sigmoid(-object_sign * logits.object.detach())
    output = StageFScoreCoefficients(
        camera_neither=-d_camera,
        camera_camera_only=d_camera,
        object_neither=-0.5 * d_object,
        object_camera_only=-0.5 * d_object,
        object_object_only=0.5 * d_object,
        object_both=0.5 * d_object,
    )
    if any(
        value.ndim != 0 or not bool(torch.isfinite(value))
        for value in (
            output.camera_neither,
            output.camera_camera_only,
            output.object_neither,
            output.object_camera_only,
            output.object_object_only,
            output.object_both,
        )
    ):
        raise FactorTrainingContractError("Stage-F analytic score gradient is invalid")
    return output


def _configuration_parameters(
    backend: Any,
    configuration: BackendStageConfiguration,
) -> tuple[tuple[nn.Parameter, ...], set[int]]:
    if not isinstance(configuration, BackendStageConfiguration):
        raise FactorTrainingContractError(
            "two-pass Stage-F requires a BackendStageConfiguration"
        )
    owner_configuration = configuration.owner_configuration
    if (
        configuration.stage is not TrainingStage.F
        or owner_configuration.stage is not TrainingStage.F
        or getattr(getattr(backend, "source_core", None), "active_stage", None) != "F"
    ):
        raise FactorTrainingContractError("backend is not configured for Stage-F")
    model = getattr(backend, "model", None)
    decision = getattr(backend, "factor_decision", None)
    if not isinstance(model, nn.Module) or not isinstance(decision, FactorSpanDecision):
        raise FactorTrainingContractError("Stage-F backend graph is incomplete")
    parameters = tuple(model.parameters())
    if not parameters:
        raise FactorTrainingContractError("Stage-F model has no parameters")
    allowed = {
        id(parameter)
        for name, parameter in owner_configuration.owners.all
        if name in owner_configuration.trainable_names
    }
    observed = {id(parameter) for parameter in parameters if parameter.requires_grad}
    if not allowed or observed != allowed:
        raise FactorTrainingContractError(
            "live Stage-F trainables differ from the owner allowlist"
        )
    if any(parameter.grad is not None for parameter in parameters):
        raise FactorTrainingContractError(
            "exact Stage-F backward requires every starting gradient to be None"
        )
    if model.training or any(module.training for module in model.modules()):
        raise FactorTrainingContractError(
            "exact two-pass Stage-F requires model.eval() determinism"
        )
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise FactorTrainingContractError(
            "exact Stage-F backward requires autograd enabled outside inference mode"
        )
    return parameters, allowed


def _parameter_state(
    parameters: tuple[nn.Parameter, ...],
) -> tuple[tuple[int, bool], ...]:
    return tuple(
        (int(parameter._version), parameter.requires_grad) for parameter in parameters
    )


def _assert_parameter_state(
    parameters: tuple[nn.Parameter, ...],
    expected: tuple[tuple[int, bool], ...],
) -> None:
    observed = _parameter_state(parameters)
    if observed != expected:
        raise FactorTrainingContractError(
            "a parameter version or requires_grad flag changed during Stage-F backward"
        )


def _clear_gradients(parameters: tuple[nn.Parameter, ...]) -> None:
    for parameter in parameters:
        parameter.grad = None


def _detached_scores(scores: FactorSpanRouteScores) -> FactorSpanRouteScores:
    return FactorSpanRouteScores(
        **{
            field: getattr(scores, field).detach().clone()
            for field in (
                "camera_neither",
                "camera_camera_only",
                "object_neither",
                "object_camera_only",
                "object_object_only",
                "object_both",
            )
        }
    )


def _detached_logits(logits: FactorLogits) -> FactorLogits:
    return FactorLogits(
        camera=logits.camera.detach().clone(),
        object=logits.object.detach().clone(),
    )


def _detached_losses(losses: StageFFactorLoss) -> StageFFactorLoss:
    return StageFFactorLoss(
        camera=losses.camera.detach().clone(),
        object=losses.object.detach().clone(),
        total=losses.total.detach().clone(),
    )


def _detached_coefficients(
    values: StageFScoreCoefficients,
) -> StageFScoreCoefficients:
    return StageFScoreCoefficients(
        **{
            field: getattr(values, field).detach().clone()
            for field in (
                "camera_neither",
                "camera_camera_only",
                "object_neither",
                "object_camera_only",
                "object_object_only",
                "object_both",
            )
        }
    )


def exact_two_pass_stage_f_backward(
    backend: Any,
    request: Any,
    *,
    configuration: BackendStageConfiguration,
    candidates: StageFCandidateTexts,
    camera_moving: bool,
    object_moving: bool,
) -> ExactStageFTwoPassResult:
    """Accumulate the exact six-call Stage-F gradient with one live graph.

    On success, only current Stage-F owner gradients are populated and the
    caller may audit/clip/step them.  This function itself cannot update model
    parameters.  Any failure after execution begins clears all partial grads so
    a caller cannot accidentally step an incomplete factor objective.
    """

    if not isinstance(candidates, StageFCandidateTexts):
        raise FactorTrainingContractError(
            "candidates must be a StageFCandidateTexts instance"
        )
    candidates = candidates.validated()
    # Validate booleans before any native call.
    if type(camera_moving) is not bool or type(object_moving) is not bool:
        raise FactorTrainingContractError("Stage-F moving targets must be bool")
    parameters, allowed_parameter_ids = _configuration_parameters(
        backend, configuration
    )
    initial_parameter_state = _parameter_state(parameters)
    initial_route_counts = _route_counts(backend)
    calls = _candidate_calls(candidates)

    try:
        pass1: dict[str, _ScoreObservation] = {}
        with torch.no_grad():
            for call in calls:
                pass1[call.field] = _observe_score(
                    backend, request, call, require_grad=False
                )
                _assert_parameter_state(parameters, initial_parameter_state)
            pass1_scores = _factor_scores(pass1)
            margins = factor_span_margins(pass1_scores)
            logits = backend.factor_decision(margins)
            camera_target = _target(camera_moving, logits.camera)
            object_target = _target(object_moving, logits.object)
            losses = stage_f_factor_loss(
                logits,
                camera_moving=camera_target,
                object_moving=object_target,
            )
            coefficients = _analytic_coefficients(
                logits,
                camera_moving=camera_moving,
                object_moving=object_moving,
            )

        maximum_replay_error = 0.0
        for call in calls:
            replay = _observe_score(backend, request, call, require_grad=True)
            reference = pass1[call.field]
            if (
                replay.text != reference.text
                or not torch.equal(
                    replay.answer_token_ids.cpu(), reference.answer_token_ids.cpu()
                )
                or replay.token_log_probabilities.shape
                != reference.token_log_probabilities.shape
            ):
                raise FactorTrainingContractError(
                    f"native evaluator replay identity drifted for {call.field}"
                )
            difference = (
                replay.token_log_probabilities
                - reference.token_log_probabilities.to(
                    device=replay.token_log_probabilities.device
                )
            ).abs()
            replay_error = float(difference.max().detach().cpu())
            maximum_replay_error = max(maximum_replay_error, replay_error)
            if not torch.equal(
                replay.token_log_probabilities,
                reference.token_log_probabilities.to(
                    device=replay.token_log_probabilities.device
                ),
            ):
                raise FactorTrainingContractError(
                    {
                        "reason": "native evaluator score replay drifted",
                        "candidate": call.field,
                        "maximum_absolute_error": replay_error,
                    }
                )
            coefficient = getattr(coefficients, call.field).to(
                device=replay.scalar.device, dtype=replay.scalar.dtype
            )
            (coefficient * replay.scalar).backward()
            # Drop the final Python references before constructing the next
            # native graph; backward has already released its saved tensors.
            del replay, coefficient
            _assert_parameter_state(parameters, initial_parameter_state)

        # The six replayed scores deliberately do not include b_c/b_o.  Use the
        # real objective with detached margins to accumulate exactly their two
        # gradients, without reconnecting any score graph.
        detached_margins = FactorMargins(
            camera=margins.camera.detach(), object=margins.object.detach()
        )
        bias_logits = backend.factor_decision(detached_margins)
        bias_losses = stage_f_factor_loss(
            bias_logits,
            camera_moving=camera_target,
            object_moving=object_target,
        )
        if not torch.equal(bias_losses.total, losses.total):
            raise FactorTrainingContractError(
                "detached-margin intercept objective differs from pass 1"
            )
        bias_losses.total.backward()
        _assert_parameter_state(parameters, initial_parameter_state)

        for parameter in parameters:
            if id(parameter) in allowed_parameter_ids:
                if parameter.grad is None or not bool(
                    torch.isfinite(parameter.grad).all()
                ):
                    raise FactorTrainingContractError(
                        "a Stage-F owner gradient is missing or non-finite"
                    )
            elif parameter.grad is not None:
                raise FactorTrainingContractError(
                    "a non-Stage-F parameter received a gradient"
                )

        final_route_counts = _route_counts(backend)
        route_delta = {
            name: final_route_counts[name] - initial_route_counts[name]
            for name in initial_route_counts
        }
        expected_route_delta = {
            FactorRoute.CAMERA_FACTOR.value: 4,
            FactorRoute.OBJECT_FACTOR.value: 8,
            FactorRoute.FULL_LANGUAGE.value: 0,
        }
        if route_delta != expected_route_delta:
            raise FactorTrainingContractError(
                {
                    "reason": "exact Stage-F route count differs from 4/8/0",
                    "observed": route_delta,
                }
            )
        _assert_parameter_state(parameters, initial_parameter_state)
        return ExactStageFTwoPassResult(
            scores=_detached_scores(pass1_scores),
            logits=_detached_logits(logits),
            losses=_detached_losses(losses),
            coefficients=_detached_coefficients(coefficients),
            route_call_delta=dict(route_delta),
            maximum_replay_score_error=maximum_replay_error,
            parameter_versions_unchanged=True,
        )
    except Exception:
        _clear_gradients(parameters)
        raise


__all__ = [
    "ExactStageFTwoPassResult",
    "FactorTrainingContractError",
    "StageFCandidateTexts",
    "StageFScoreCoefficients",
    "exact_two_pass_stage_f_backward",
]
