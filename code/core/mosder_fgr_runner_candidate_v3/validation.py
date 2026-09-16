"""Collect factor and language metrics for each F/G/R stage."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Final, Mapping, Sequence

import torch

from mosder_final_v1.family_backend import assert_mosder_frozen_base_unchanged
from mosder_final_v1.language import (
    canonical_answer,
    parse_four_line_language,
    plan_canonical_language,
    validate_score_against_plan,
)
from mosder_final_v1.objectives import (
    stage_f_factor_loss,
    stage_g_language_loss,
    stage_r_language_loss,
)
from mosder_final_v1.routing import (
    FactorRoute,
    FactorSpanRouteScores,
    four_state_from_logits,
)
from mosder_final_v1.contract import STATE_FACTORS

from .checkpoint import (
    capture_rng_state,
    method_parameter_state,
    restore_rng_state,
    tensor_state_sha256,
)
from .engine import MoSDeRExecutionAdapter, TrainingExample
from .sampler import STATE_ORDER


SCHEMA_VERSION: Final[str] = "mosder_validation_metrics_candidate_v1"
ADT_SOURCE: Final[str] = "ADT-LiteOffice"


class ValidationContractError(RuntimeError):
    """Validation input, generation, or metric aggregation is invalid."""


def _mean(values: Sequence[float], name: str) -> float:
    if not values:
        raise ValidationContractError(f"{name} has no observations")
    output = sum(values) / float(len(values))
    if not math.isfinite(output):
        raise ValidationContractError(f"{name} is non-finite")
    return output


def _macro_recall(
    truth: Sequence[str], predicted: Sequence[str]
) -> tuple[float, Mapping[str, float]]:
    if len(truth) != len(predicted) or not truth:
        raise ValidationContractError("macro recall inputs differ/are empty")
    recalls: dict[str, float] = {}
    for state in STATE_ORDER:
        members = [index for index, value in enumerate(truth) if value == state]
        if not members:
            raise ValidationContractError(f"macro recall state is empty: {state}")
        recalls[state] = sum(predicted[index] == state for index in members) / float(
            len(members)
        )
    return _mean(tuple(recalls.values()), "four-state macro recall"), recalls


def _factor_scores(backend: Any, request: Any) -> FactorSpanRouteScores:
    calls = {
        "camera_neither": ("neither", FactorRoute.CAMERA_FACTOR),
        "camera_camera_only": ("camera_only", FactorRoute.CAMERA_FACTOR),
        "object_neither": ("neither", FactorRoute.OBJECT_FACTOR),
        "object_camera_only": ("camera_only", FactorRoute.OBJECT_FACTOR),
        "object_object_only": ("object_only", FactorRoute.OBJECT_FACTOR),
        "object_both": ("both", FactorRoute.OBJECT_FACTOR),
    }
    values: dict[str, torch.Tensor] = {}
    for name, (state, route) in calls.items():
        score = backend.teacher_forced_loss(
            request, canonical_answer(state), route=route
        )
        values[name] = score.token_log_probabilities.to(torch.float32).mean()
    return FactorSpanRouteScores(**values)


@torch.inference_mode()
def evaluate_stage_f(
    adapter: MoSDeRExecutionAdapter,
    dataset: Sequence[TrainingExample],
) -> Mapping[str, Any]:
    adapter.backend.configure_mosder_stage("FROZEN")
    truth: list[str] = []
    predicted: list[str] = []
    adt_truth: list[str] = []
    adt_predicted: list[str] = []
    adt_losses: list[float] = []
    source_state: Counter[tuple[str, str]] = Counter()
    with torch.no_grad():
        for index in range(len(dataset)):
            example = dataset[index]
            try:
                scores = _factor_scores(adapter.backend, example.request)
                logits = adapter.backend.factor_decision.from_scores(scores)
                prediction = four_state_from_logits(logits)
                if len(prediction) != 1:
                    raise ValidationContractError(
                        "factor evaluator returned non-scalar state"
                    )
                observed = prediction[0]
                truth.append(example.state)
                predicted.append(observed)
                source_state[(example.source_dataset, example.state)] += 1
                if example.source_dataset == ADT_SOURCE:
                    adt_truth.append(example.state)
                    adt_predicted.append(observed)
                    camera, obj = STATE_FACTORS[example.state]
                    loss = stage_f_factor_loss(
                        logits,
                        camera_moving=torch.tensor(camera, device=logits.camera.device),
                        object_moving=torch.tensor(obj, device=logits.object.device),
                    )
                    adt_losses.append(float(loss.total.cpu()))
            finally:
                example.close()
    adt_macro, adt_recalls = _macro_recall(adt_truth, adt_predicted)
    all_macro, all_recalls = _macro_recall(truth, predicted)
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "F",
        "selection_metrics": {
            "adt_four_state_macro_recall": adt_macro,
            "factor_loss": _mean(adt_losses, "ADT factor loss"),
        },
        "diagnostics": {
            "all_four_state_macro_recall": all_macro,
            "adt_recall_by_state": adt_recalls,
            "all_recall_by_state": all_recalls,
            "rows": len(truth),
            "adt_rows": len(adt_truth),
            "source_state_exposure": {
                f"{source}::{state}": count
                for (source, state), count in sorted(source_state.items())
            },
        },
    }


@torch.inference_mode()
def evaluate_stage_language(
    adapter: MoSDeRExecutionAdapter,
    dataset: Sequence[TrainingExample],
    *,
    stage: str,
) -> Mapping[str, Any]:
    if stage not in {"G", "R"}:
        raise ValidationContractError("language validation stage must be G or R")
    adapter.backend.configure_mosder_stage("FROZEN")
    adt_truth: list[str] = []
    adt_prediction: list[str] = []
    adt_exact: list[float] = []
    adt_factor: list[float] = []
    adt_nll: list[float] = []
    parse_failures = 0
    source_rows: Counter[str] = Counter()
    for index in range(len(dataset)):
        example = dataset[index]
        try:
            generation = adapter.backend.generate(
                example.request, max_new_tokens=96, do_sample=False
            )
            text = generation.text
            predicted_state = "__invalid__"
            factor_score = 0.0
            try:
                parsed = parse_four_line_language(text)
                predicted_state = parsed.state
                truth_factors = STATE_FACTORS[example.state]
                factor_score = 0.5 * (
                    float(parsed.camera_moving == truth_factors[0])
                    + float(parsed.object_moving == truth_factors[1])
                )
            except Exception:
                parse_failures += 1
            source_rows[example.source_dataset] += 1
            if example.source_dataset == ADT_SOURCE:
                teacher = canonical_answer(example.state)
                with torch.no_grad():
                    plan = plan_canonical_language(
                        adapter.backend, example.request, example.state
                    )
                    score = adapter.backend.teacher_forced_loss(
                        example.request,
                        teacher,
                        route=FactorRoute.FULL_LANGUAGE,
                    )
                    validate_score_against_plan(score, plan)
                    losses = (
                        stage_g_language_loss(score.token_log_probabilities, plan.spans)
                        if stage == "G"
                        else stage_r_language_loss(
                            score.token_log_probabilities, plan.spans
                        )
                    )
                adt_truth.append(example.state)
                adt_prediction.append(predicted_state)
                adt_exact.append(float(text == teacher))
                adt_factor.append(factor_score)
                adt_nll.append(float(losses.total.cpu()))
        finally:
            example.close()
    macro, recalls = _macro_recall(adt_truth, adt_prediction)
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "selection_metrics": {
            "strict_four_line_exact": _mean(adt_exact, "ADT strict exact"),
            "minimum_state_recall": min(recalls.values()),
            "factor_consistency": _mean(adt_factor, "ADT factor accuracy"),
            "span_nll": _mean(adt_nll, "ADT span NLL"),
        },
        "diagnostics": {
            "adt_four_state_macro_recall": macro,
            "adt_recall_by_state": recalls,
            "parse_failures_all_sources": parse_failures,
            "source_rows": dict(sorted(source_rows.items())),
            "adt_rows": len(adt_truth),
            "decoding": "greedy_do_sample_false",
        },
    }


def evaluate_with_rng_isolation(
    adapter: MoSDeRExecutionAdapter,
    dataset: Sequence[TrainingExample],
    *,
    stage: str,
) -> Mapping[str, Any]:
    if any(
        parameter.grad is not None for parameter in adapter.backend.model.parameters()
    ):
        raise ValidationContractError("Validation entered with residual gradients")
    method_before = tensor_state_sha256(
        method_parameter_state(adapter.method_parameters)
    )
    state = capture_rng_state()
    try:
        if stage == "F":
            return evaluate_stage_f(adapter, dataset)
        return evaluate_stage_language(adapter, dataset, stage=stage)
    finally:
        try:
            method_after = tensor_state_sha256(
                method_parameter_state(adapter.method_parameters)
            )
            if method_after != method_before:
                raise ValidationContractError(
                    "Validation changed MoSDeR method parameters"
                )
            if any(
                parameter.grad is not None
                for parameter in adapter.backend.model.parameters()
            ):
                raise ValidationContractError("Validation left residual gradients")
            assert_mosder_frozen_base_unchanged(
                adapter.backend.model,
                adapter.frozen_base_snapshot,
                mode="sampled",
            )
        finally:
            restore_rng_state(state)


__all__ = [
    "ADT_SOURCE",
    "SCHEMA_VERSION",
    "ValidationContractError",
    "evaluate_stage_f",
    "evaluate_stage_language",
    "evaluate_with_rng_isolation",
]
