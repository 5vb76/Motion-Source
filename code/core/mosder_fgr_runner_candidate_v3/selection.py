"""Rank existing validation metrics to select a checkpoint per stage."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Final, Mapping

from .protocol import STAGES, canonical_sha256


SCHEMA_VERSION: Final[str] = "mosder_best_validation_tracker_candidate_v1"


class SelectionContractError(RuntimeError):
    """A metric record or checkpoint-selection transition is invalid."""


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionContractError(f"{name} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise SelectionContractError(f"{name} must be finite")
    return output


@dataclass(frozen=True, slots=True)
class ValidationCandidate:
    stage: str
    epoch_one_based: int
    optimizer_step: int
    checkpoint_name: str
    checkpoint_sha256: str
    receipt_sha256: str
    metrics: Mapping[str, float]
    kind: str = "trained"

    def validated(self) -> "ValidationCandidate":
        if self.stage not in STAGES:
            raise SelectionContractError("candidate stage is invalid")
        if (
            type(self.epoch_one_based) is not int
            or self.epoch_one_based < 0
            or type(self.optimizer_step) is not int
            or self.optimizer_step < 0
        ):
            raise SelectionContractError("candidate epoch/step is invalid")
        if (
            not self.checkpoint_name
            or "/" in self.checkpoint_name
            or "\\" in self.checkpoint_name
        ):
            raise SelectionContractError("candidate checkpoint name is invalid")
        for name, value in (
            ("checkpoint_sha256", self.checkpoint_sha256),
            ("receipt_sha256", self.receipt_sha256),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise SelectionContractError(f"{name} is invalid")
        if self.kind not in {"trained", "r_step0_parent_g"}:
            raise SelectionContractError("candidate kind is invalid")
        if self.kind == "r_step0_parent_g" and not (
            self.stage == "R" and self.epoch_one_based == 0 and self.optimizer_step == 0
        ):
            raise SelectionContractError("R step-0 parent candidate is malformed")
        required = (
            {"adt_four_state_macro_recall", "factor_loss"}
            if self.stage == "F"
            else {
                "strict_four_line_exact",
                "minimum_state_recall",
                "factor_consistency",
                "span_nll",
            }
        )
        if set(self.metrics) != required:
            raise SelectionContractError("candidate metric fields differ")
        for name, value in self.metrics.items():
            score = _finite(value, name)
            if name != "span_nll" and name != "factor_loss" and not 0.0 <= score <= 1.0:
                raise SelectionContractError(f"{name} must lie in [0,1]")
            if name in {"span_nll", "factor_loss"} and score < 0.0:
                raise SelectionContractError(f"{name} must be nonnegative")
        return self

    def rank_key(self) -> tuple[float, ...]:
        self.validated()
        if self.stage == "F":
            return (
                float(self.metrics["adt_four_state_macro_recall"]),
                -float(self.metrics["factor_loss"]),
                -float(self.optimizer_step),
            )
        return (
            float(self.metrics["strict_four_line_exact"]),
            float(self.metrics["minimum_state_recall"]),
            float(self.metrics["factor_consistency"]),
            -float(self.metrics["span_nll"]),
            -float(self.optimizer_step),
        )

    def as_dict(self) -> Mapping[str, Any]:
        self.validated()
        return {
            "stage": self.stage,
            "epoch_one_based": self.epoch_one_based,
            "optimizer_step": self.optimizer_step,
            "checkpoint_name": self.checkpoint_name,
            "checkpoint_sha256": self.checkpoint_sha256,
            "receipt_sha256": self.receipt_sha256,
            "metrics": dict(self.metrics),
            "kind": self.kind,
            "rank_key": list(self.rank_key()),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ValidationCandidate":
        candidate = cls(
            stage=str(value.get("stage")),
            epoch_one_based=int(value.get("epoch_one_based", -1)),
            optimizer_step=int(value.get("optimizer_step", -1)),
            checkpoint_name=str(value.get("checkpoint_name", "")),
            checkpoint_sha256=str(value.get("checkpoint_sha256", "")),
            receipt_sha256=str(value.get("receipt_sha256", "")),
            metrics=dict(value.get("metrics", {})),
            kind=str(value.get("kind", "")),
        ).validated()
        if value.get("rank_key") != list(candidate.rank_key()):
            raise SelectionContractError("recorded candidate rank key differs")
        return candidate


class BestValidationTracker:
    def __init__(self, stage: str) -> None:
        if stage not in STAGES:
            raise SelectionContractError("tracker stage is invalid")
        self.stage = stage
        self.best: ValidationCandidate | None = None
        self.history: list[ValidationCandidate] = []

    def consider(self, candidate: ValidationCandidate) -> bool:
        candidate.validated()
        if candidate.stage != self.stage:
            raise SelectionContractError("candidate/tracker stage differs")
        if any(
            previous.checkpoint_sha256 == candidate.checkpoint_sha256
            for previous in self.history
        ):
            raise SelectionContractError("candidate checkpoint was already considered")
        self.history.append(candidate)
        if self.best is None or candidate.rank_key() > self.best.rank_key():
            self.best = candidate
            return True
        return False

    def state_dict(self) -> Mapping[str, Any]:
        history = [candidate.as_dict() for candidate in self.history]
        return {
            "schema_version": SCHEMA_VERSION,
            "stage": self.stage,
            "best": None if self.best is None else self.best.as_dict(),
            "history": history,
            "history_sha256": canonical_sha256(history),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if (
            value.get("schema_version") != SCHEMA_VERSION
            or value.get("stage") != self.stage
        ):
            raise SelectionContractError("tracker checkpoint identity differs")
        history_value = value.get("history")
        if not isinstance(history_value, list) or value.get(
            "history_sha256"
        ) != canonical_sha256(history_value):
            raise SelectionContractError("tracker history digest differs")
        rebuilt = BestValidationTracker(self.stage)
        for item in history_value:
            if not isinstance(item, Mapping):
                raise SelectionContractError("tracker history item is invalid")
            rebuilt.consider(ValidationCandidate.from_dict(item))
        expected_best = None if rebuilt.best is None else rebuilt.best.as_dict()
        if value.get("best") != expected_best:
            raise SelectionContractError("tracker best candidate differs")
        self.history = rebuilt.history
        self.best = rebuilt.best


__all__ = [
    "BestValidationTracker",
    "SCHEMA_VERSION",
    "SelectionContractError",
    "ValidationCandidate",
]
