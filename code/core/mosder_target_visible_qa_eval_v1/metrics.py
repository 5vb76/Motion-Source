"""Compute motion-state metrics from single-letter QA predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Final

import numpy as np

from .contract import (
    DECISION_SCHEMA_VERSION,
    LETTERS,
    STATE_FACTORS,
    STATE_ORDER,
    LetterDecision,
    QAContractError,
    choose_from_letter_logits,
    option_mapping,
)


METRICS_SCHEMA_VERSION: Final[str] = "mosder_target_visible_qa_metrics_v1"
RECORD_SCHEMA_VERSION: Final[str] = "mosder_target_visible_qa_scored_record_v1"
ADT_PRIMARY_SOURCE: Final[str] = "ADT-LiteOffice"


class MetricContractError(RuntimeError):
    """A scored row or aggregate metric contract was invalid."""


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """One validated scored case with its derived semantic prediction."""

    case_id: str
    source_dataset: str
    truth_state: str
    option_mapping: tuple[tuple[str, str], ...]
    letter_logits: tuple[tuple[str, float], ...]
    predicted_letter: str
    predicted_state: str
    tie_letters: tuple[str, ...]
    tie_states: tuple[str, ...]
    margin: float

    @property
    def is_tie(self) -> bool:
        return len(self.tie_letters) > 1

    @property
    def truth_letter(self) -> str:
        matches = tuple(
            letter for letter, state in self.option_mapping if state == self.truth_state
        )
        if len(matches) != 1:
            raise MetricContractError("validated record lost its unique truth letter")
        return matches[0]


def _required_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MetricContractError(f"{label} must be a nonempty, already-trimmed string")
    return value


def _as_tuple_of_strings(value: object, *, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise MetricContractError(f"{label} must be a sequence of strings")
    try:
        result = tuple(value)  # type: ignore[arg-type]
    except Exception as error:
        raise MetricContractError(f"{label} must be a sequence") from error
    if any(not isinstance(item, str) for item in result):
        raise MetricContractError(f"{label} must contain only strings")
    return result


def record_from_mapping(row: Mapping[str, Any]) -> EvaluationRecord:
    """Validate one runner row and derive its state from letter logits.

    Required fields are ``case_id``, ``source_dataset``, ``truth_state``,
    ``option_mapping``, and ``letter_logits``.  Extra provenance fields are
    allowed.  If a runner also records a derived prediction/tie, it is checked
    against the authoritative CPU decision rather than trusted.
    """

    if not isinstance(row, Mapping):
        raise MetricContractError("evaluation row must be a mapping")
    required = {
        "case_id",
        "source_dataset",
        "truth_state",
        "option_mapping",
        "letter_logits",
    }
    if not required.issubset(row):
        missing = sorted(required.difference(row))
        raise MetricContractError(f"evaluation row is missing fields: {missing}")
    case_id = _required_string(row["case_id"], label="case_id")
    source = _required_string(row["source_dataset"], label="source_dataset")
    truth = row["truth_state"]
    if truth not in STATE_ORDER:
        raise MetricContractError("truth_state is not one of the four frozen states")
    supplied_mapping = row["option_mapping"]
    if not isinstance(supplied_mapping, Mapping):
        raise MetricContractError("option_mapping must be a mapping")
    expected_mapping = option_mapping(case_id)
    if set(supplied_mapping) != set(LETTERS) or any(
        supplied_mapping.get(letter) != expected_mapping[letter] for letter in LETTERS
    ):
        raise MetricContractError(
            "option_mapping differs from the case-derived mapping"
        )
    logits = row["letter_logits"]
    if not isinstance(logits, Mapping):
        raise MetricContractError("letter_logits must be a mapping")
    try:
        decision = choose_from_letter_logits(expected_mapping, logits)
    except QAContractError as error:
        raise MetricContractError("letter-logit decision contract failed") from error

    optional_exact = {
        "predicted_letter": decision.predicted_letter,
        "predicted_state": decision.predicted_state,
    }
    for field, expected in optional_exact.items():
        if field in row and row[field] != expected:
            raise MetricContractError(f"recorded {field} differs from CPU derivation")
    optional_sequences = {
        "tie_letters": decision.tie_letters,
        "tie_states": decision.tie_states,
    }
    for field, expected in optional_sequences.items():
        if field in row and _as_tuple_of_strings(row[field], label=field) != expected:
            raise MetricContractError(f"recorded {field} differs from CPU derivation")
    if "margin" in row:
        margin = row["margin"]
        if isinstance(margin, (bool, np.bool_)) or not isinstance(margin, Real):
            raise MetricContractError("recorded margin must be finite numeric")
        observed_margin = float(margin)
        if not math.isfinite(observed_margin) or observed_margin != decision.margin:
            raise MetricContractError("recorded margin differs from CPU derivation")

    return _evaluation_record(case_id, source, str(truth), expected_mapping, decision)


def _evaluation_record(
    case_id: str,
    source: str,
    truth: str,
    mapping: Mapping[str, str],
    decision: LetterDecision,
) -> EvaluationRecord:
    return EvaluationRecord(
        case_id=case_id,
        source_dataset=source,
        truth_state=truth,
        option_mapping=tuple((letter, mapping[letter]) for letter in LETTERS),
        letter_logits=decision.ordered_letter_logits,
        predicted_letter=decision.predicted_letter,
        predicted_state=decision.predicted_state,
        tie_letters=decision.tie_letters,
        tie_states=decision.tie_states,
        margin=decision.margin,
    )


def _validate_evaluation_record(record: EvaluationRecord) -> EvaluationRecord:
    """Revalidate dataclass input through the same mapping/logit path."""

    if not isinstance(record, EvaluationRecord):
        raise MetricContractError("metric input must be mapping or EvaluationRecord")
    return record_from_mapping(
        {
            "case_id": record.case_id,
            "source_dataset": record.source_dataset,
            "truth_state": record.truth_state,
            "option_mapping": dict(record.option_mapping),
            "letter_logits": dict(record.letter_logits),
            "predicted_letter": record.predicted_letter,
            "predicted_state": record.predicted_state,
            "tie_letters": record.tie_letters,
            "tie_states": record.tie_states,
            "margin": record.margin,
        }
    )


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    result = numerator / float(denominator)
    if not math.isfinite(result):
        raise MetricContractError("computed ratio is non-finite")
    return result


def _empty_confusion() -> dict[str, dict[str, int]]:
    return {truth: {predicted: 0 for predicted in STATE_ORDER} for truth in STATE_ORDER}


def _empty_letter_counts() -> dict[str, int]:
    return {letter: 0 for letter in LETTERS}


def _empty_binary_confusion() -> dict[str, dict[str, int]]:
    return {
        truth: {predicted: 0 for predicted in ("static", "moving")}
        for truth in ("static", "moving")
    }


def _binary_factor_metrics(
    confusion: Mapping[str, Mapping[str, int]],
) -> Mapping[str, Any]:
    true_negative = confusion["static"]["static"]
    false_positive = confusion["static"]["moving"]
    false_negative = confusion["moving"]["static"]
    true_positive = confusion["moving"]["moving"]
    negative_support = true_negative + false_positive
    positive_support = true_positive + false_negative
    total = negative_support + positive_support
    true_positive_rate = _safe_ratio(true_positive, positive_support)
    true_negative_rate = _safe_ratio(true_negative, negative_support)
    balanced_accuracy = (
        (true_positive_rate + true_negative_rate) / 2.0
        if true_positive_rate is not None and true_negative_rate is not None
        else None
    )
    f1_denominator = 2 * true_positive + false_positive + false_negative
    f1 = _safe_ratio(2 * true_positive, f1_denominator)
    return {
        "positive_class": "moving",
        "confusion": {
            truth: {
                predicted: int(confusion[truth][predicted])
                for predicted in ("static", "moving")
            }
            for truth in ("static", "moving")
        },
        "accuracy": _safe_ratio(true_positive + true_negative, total),
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "true_positive_rate": true_positive_rate,
        "true_negative_rate": true_negative_rate,
        "positive_support": positive_support,
        "negative_support": negative_support,
    }


def _subset_metrics(records: Sequence[EvaluationRecord]) -> Mapping[str, Any]:
    confusion = _empty_confusion()
    camera_confusion = _empty_binary_confusion()
    object_confusion = _empty_binary_confusion()
    truth_letter_counts = _empty_letter_counts()
    predicted_letter_counts = _empty_letter_counts()
    truth_letter_correct = _empty_letter_counts()
    tie_count = 0
    margin_total = 0.0
    camera_correct = 0
    object_correct = 0

    for record in records:
        confusion[record.truth_state][record.predicted_state] += 1
        truth_letter_counts[record.truth_letter] += 1
        predicted_letter_counts[record.predicted_letter] += 1
        if record.truth_state == record.predicted_state:
            truth_letter_correct[record.truth_letter] += 1
        tie_count += int(record.is_tie)
        margin_total += record.margin
        truth_camera, truth_object = STATE_FACTORS[record.truth_state]
        predicted_camera, predicted_object = STATE_FACTORS[record.predicted_state]
        camera_correct += int(truth_camera == predicted_camera)
        object_correct += int(truth_object == predicted_object)
        camera_confusion["moving" if truth_camera else "static"][
            "moving" if predicted_camera else "static"
        ] += 1
        object_confusion["moving" if truth_object else "static"][
            "moving" if predicted_object else "static"
        ] += 1

    row_count = len(records)
    correct = sum(confusion[state][state] for state in STATE_ORDER)
    per_state: dict[str, Mapping[str, Any]] = {}
    recalls: list[float] = []
    for state in STATE_ORDER:
        support = sum(confusion[state].values())
        predicted_count = sum(confusion[truth][state] for truth in STATE_ORDER)
        state_correct = confusion[state][state]
        recall = _safe_ratio(state_correct, support)
        if recall is not None:
            recalls.append(recall)
        per_state[state] = {
            "support": support,
            "predicted_count": predicted_count,
            "correct": state_correct,
            "recall": recall,
        }

    macro_recall = sum(recalls) / len(recalls) if recalls else None
    minimum_recall = min(recalls) if recalls else None
    factor_denominator = 2 * row_count
    factor_correct = camera_correct + object_correct
    camera_metrics = _binary_factor_metrics(camera_confusion)
    object_metrics = _binary_factor_metrics(object_confusion)
    by_truth_letter = {
        letter: {
            "support": truth_letter_counts[letter],
            "correct": truth_letter_correct[letter],
            "accuracy": _safe_ratio(
                truth_letter_correct[letter], truth_letter_counts[letter]
            ),
        }
        for letter in LETTERS
    }
    return {
        "row_count": row_count,
        "correct": correct,
        "four_state_accuracy": _safe_ratio(correct, row_count),
        "supported_state_count": len(recalls),
        "four_state_macro_recall": macro_recall,
        "minimum_state_recall": minimum_recall,
        "camera_factor_confusion": camera_metrics["confusion"],
        "camera_factor_accuracy": camera_metrics["accuracy"],
        "camera_factor_balanced_accuracy": camera_metrics["balanced_accuracy"],
        "camera_factor_f1": camera_metrics["f1"],
        "camera_factor_true_positive_rate": camera_metrics["true_positive_rate"],
        "camera_factor_true_negative_rate": camera_metrics["true_negative_rate"],
        "object_factor_confusion": object_metrics["confusion"],
        "object_factor_accuracy": object_metrics["accuracy"],
        "object_factor_balanced_accuracy": object_metrics["balanced_accuracy"],
        "object_factor_f1": object_metrics["f1"],
        "object_factor_true_positive_rate": object_metrics["true_positive_rate"],
        "object_factor_true_negative_rate": object_metrics["true_negative_rate"],
        "factor_hamming_accuracy": _safe_ratio(factor_correct, factor_denominator),
        "factor_hamming_error": _safe_ratio(
            factor_denominator - factor_correct, factor_denominator
        ),
        "tie_count": tie_count,
        "tie_rate": _safe_ratio(tie_count, row_count),
        "mean_winner_margin": (margin_total / row_count if row_count else None),
        "confusion": confusion,
        "per_state": per_state,
        "letter_diagnostics": {
            "truth_letter_counts": truth_letter_counts,
            "predicted_letter_counts": predicted_letter_counts,
            "by_truth_letter": by_truth_letter,
        },
    }


def summarize_records(
    rows: Sequence[Mapping[str, Any] | EvaluationRecord],
) -> Mapping[str, Any]:
    """Derive semantic states and summarize overall, ADT, and source metrics.

    Macro and minimum recall are computed across states with nonzero support in
    the relevant subset.  ``supported_state_count`` makes this denominator
    explicit; all four states are still present in the confusion/per-state
    objects with ``None`` recall when absent.
    """

    if isinstance(rows, (str, bytes, bytearray)):
        raise MetricContractError("metric rows must be a finite sequence")
    try:
        raw_rows = tuple(rows)
    except Exception as error:
        raise MetricContractError("metric rows must be a finite sequence") from error
    if not raw_rows:
        raise MetricContractError("at least one scored row is required")
    records: list[EvaluationRecord] = []
    identities: set[tuple[str, str]] = set()
    for raw in raw_rows:
        record = (
            _validate_evaluation_record(raw)
            if isinstance(raw, EvaluationRecord)
            else record_from_mapping(raw)
        )
        identity = (record.source_dataset, record.case_id)
        if identity in identities:
            raise MetricContractError("duplicate source_dataset/case_id identity")
        identities.add(identity)
        records.append(record)

    sources = sorted({record.source_dataset for record in records})
    by_source = {
        source: _subset_metrics(
            tuple(record for record in records if record.source_dataset == source)
        )
        for source in sources
    }
    adt_records = tuple(
        record for record in records if record.source_dataset == ADT_PRIMARY_SOURCE
    )
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "decision_schema_version": DECISION_SCHEMA_VERSION,
        "state_order": list(STATE_ORDER),
        "letter_order": list(LETTERS),
        "semantic_prediction_rule": (
            "exact argmax over A/B/C/D logits; exact ties select first letter "
            "and remain explicitly counted"
        ),
        "recall_aggregation_rule": "unweighted mean/minimum over supported states",
        "binary_factor_rule": {
            "positive_class": "moving",
            "confusion_order": ["static", "moving"],
            "balanced_accuracy": (
                "mean of moving recall and static recall; null unless both truth "
                "classes have support"
            ),
            "f1": (
                "moving-positive 2TP/(2TP+FP+FN); null when its denominator is zero"
            ),
        },
        "primary_source": ADT_PRIMARY_SOURCE,
        "overall": _subset_metrics(tuple(records)),
        "adt_primary": _subset_metrics(adt_records),
        "by_source": by_source,
    }


compute_evaluation_metrics = summarize_records


__all__ = [
    "ADT_PRIMARY_SOURCE",
    "EvaluationRecord",
    "METRICS_SCHEMA_VERSION",
    "MetricContractError",
    "RECORD_SCHEMA_VERSION",
    "compute_evaluation_metrics",
    "record_from_mapping",
    "summarize_records",
]
