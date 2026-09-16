"""Sequential, crash-resumable inference orchestration."""

from __future__ import annotations

from time import perf_counter_ns
from typing import Any, Callable, Mapping, Sequence

from .contract import RESULT_SCHEMA_VERSION, PredictedState
from .evaluator import score_question
from .manifest import QuestionAsset, materialize_target_visible
from .persistence import ResultStore


ProgressCallback = Callable[[int, int, str, str], None]


def _seconds(started_ns: int) -> float:
    return (perf_counter_ns() - started_ns) / 1_000_000_000.0


def run_assets(
    backend: Any,
    *,
    assets: Sequence[QuestionAsset],
    store: ResultStore,
    arm: str,
    scoring_strategy: str,
    predicted_states: Mapping[str, PredictedState] | None = None,
    retry_errors: bool = False,
    progress: ProgressCallback | None = None,
) -> Mapping[str, int]:
    """Score and atomically persist every label-free question."""

    values = tuple(assets)
    if not values:
        raise ValueError("assets cannot be empty")
    state_map = {} if predicted_states is None else dict(predicted_states)
    question_ids = {asset.question.question_id for asset in values}
    if arm == "raw_plus_predicted_state":
        if set(state_map) != question_ids:
            raise ValueError("predicted-state IDs must exactly match question IDs")
    elif state_map:
        raise ValueError("predicted states are accepted only by their dedicated arm")
    counts = {"success": 0, "error": 0, "skipped": 0}
    total = len(values)
    for index, asset in enumerate(values, start=1):
        question = asset.question
        if store.should_skip(question.question_id, retry_errors=retry_errors):
            counts["skipped"] += 1
            if progress is not None:
                progress(index, total, question.question_id, "skipped")
            continue
        whole_started = perf_counter_ns()
        materialization_seconds = 0.0
        try:
            materialization_started = perf_counter_ns()
            visual = materialize_target_visible(asset)
            materialization_seconds = _seconds(materialization_started)
            result = score_question(
                backend,
                question=question,
                asset_sha256=asset.identity_sha256,
                visual=visual,
                arm=arm,
                scoring_strategy=scoring_strategy,
                predicted_state=state_map.get(question.question_id),
            ).to_dict()
            result["timing"] = {
                **dict(result["timing"]),
                "materialization_seconds": materialization_seconds,
                "total_question_seconds": _seconds(whole_started),
            }
            store.write(question.question_id, result)
            counts["success"] += 1
            status = "success"
        except Exception as error:
            message = str(error).replace("\x00", "").strip()
            record = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": "error",
                "question_id": question.question_id,
                "question_sha256": question.identity_sha256,
                "asset_sha256": asset.identity_sha256,
                "data_role": question.data_role,
                "review_status": question.review_status,
                "source_dataset": question.source_dataset,
                "sequence_id": question.sequence_id,
                "physical_event_id": question.physical_event_id,
                "target_id": question.target_id,
                "family": getattr(getattr(backend, "binding", None), "key", "unknown"),
                "arm": arm,
                "scoring_strategy": scoring_strategy,
                "error": {
                    "type": type(error).__name__,
                    "message": message[:2000],
                },
                "timing": {
                    "materialization_seconds": materialization_seconds,
                    "total_question_seconds": _seconds(whole_started),
                },
                "label_fields_present": False,
            }
            store.write(question.question_id, record)
            counts["error"] += 1
            status = "error"
        if progress is not None:
            progress(index, total, question.question_id, status)
    return counts


__all__ = ["ProgressCallback", "run_assets"]
