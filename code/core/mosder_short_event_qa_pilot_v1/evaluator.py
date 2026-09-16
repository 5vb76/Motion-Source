"""Score two- to five-choice questions across three frozen inference modes."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import math
from time import perf_counter_ns
from typing import Any, Final, Mapping

from .contract import (
    ARMS,
    RESULT_SCHEMA_VERSION,
    SCORING_STRATEGIES,
    PredictedState,
    QuestionSpec,
    ShortEventQAContractError,
    candidate_answer_texts,
    render_prompt,
)
from .manifest import TargetVisibleRGB20


FULL_LANGUAGE_ROUTE: Final[str] = "FULL_LANGUAGE"


class ShortEventEvaluationError(ShortEventQAContractError):
    """The arm/backend binding or candidate scoring contract failed."""


def _backend_family(backend: object) -> str:
    family = getattr(getattr(backend, "binding", None), "key", None)
    if not isinstance(family, str) or not family:
        raise ShortEventEvaluationError("backend family binding is absent")
    return family


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ShortEventEvaluationError(f"{label} must be finite")
    try:
        output = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ShortEventEvaluationError(f"{label} must be finite") from error
    if not math.isfinite(output):
        raise ShortEventEvaluationError(f"{label} must be finite")
    return output


@dataclass(frozen=True, slots=True)
class CandidateScore:
    option_index: int
    option_letter: str
    answer_text: str
    token_count: int
    log_probability_sum: float
    log_probability_mean: float
    scoring_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MCQDecision:
    selected_option_index: int
    selected_option_letter: str
    selected_option_text: str
    tied_option_indices: tuple[int, ...]
    exact_tie: bool

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["tied_option_indices"] = list(self.tied_option_indices)
        return value


@dataclass(frozen=True, slots=True)
class SuccessfulQuestionResult:
    question_id: str
    question_sha256: str
    asset_sha256: str
    visual_sha256: str
    data_role: str
    review_status: str
    source_dataset: str
    sequence_id: str
    physical_event_id: str
    target_id: str
    family: str
    arm: str
    scoring_strategy: str
    prompt_sha256: str
    predicted_state: Mapping[str, object] | None
    candidate_scores: tuple[CandidateScore, ...]
    decision: MCQDecision
    scoring_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "success",
            "question_id": self.question_id,
            "question_sha256": self.question_sha256,
            "asset_sha256": self.asset_sha256,
            "visual_sha256": self.visual_sha256,
            "data_role": self.data_role,
            "review_status": self.review_status,
            "source_dataset": self.source_dataset,
            "sequence_id": self.sequence_id,
            "physical_event_id": self.physical_event_id,
            "target_id": self.target_id,
            "family": self.family,
            "arm": self.arm,
            "scoring_strategy": self.scoring_strategy,
            "prompt_sha256": self.prompt_sha256,
            "predicted_state": (
                None if self.predicted_state is None else dict(self.predicted_state)
            ),
            "candidate_scores": [value.to_dict() for value in self.candidate_scores],
            "decision": self.decision.to_dict(),
            "timing": {"candidate_scoring_seconds": self.scoring_seconds},
            "label_fields_present": False,
        }


def _validate_arm_backend(backend: object, arm: str) -> str:
    if arm not in ARMS:
        raise ShortEventEvaluationError("arm is invalid")
    routed = callable(getattr(backend, "_run_routed", None))
    if arm == "mosder_full_language" and not routed:
        raise ShortEventEvaluationError("FULL_LANGUAGE arm requires MoSDeR backend")
    if arm != "mosder_full_language" and routed:
        raise ShortEventEvaluationError("raw arm received a routed MoSDeR backend")
    if not callable(getattr(backend, "teacher_forced_score", None)):
        raise ShortEventEvaluationError("backend lacks teacher_forced_score")
    model = getattr(backend, "model", None)
    if model is not None and bool(getattr(model, "training", False)):
        raise ShortEventEvaluationError("backend model must be in eval mode")
    return _backend_family(backend)


def _score_candidate(
    backend: Any,
    request: Any,
    answer_text: str,
    *,
    arm: str,
    option_index: int,
    option_letter: str,
    strategy: str,
) -> CandidateScore:
    started = perf_counter_ns()
    try:
        import torch

        inference_context = torch.inference_mode()
    except ImportError:  # Keeps pure contract tests usable without Torch.
        inference_context = nullcontext()
    with inference_context:
        if arm == "mosder_full_language":
            score = backend.teacher_forced_score(
                request, answer_text, route=FULL_LANGUAGE_ROUTE
            )
        else:
            score = backend.teacher_forced_score(request, answer_text)
    elapsed = (perf_counter_ns() - started) / 1_000_000_000.0
    token_count = getattr(score, "token_count", None)
    if type(token_count) is not int or token_count <= 0:
        raise ShortEventEvaluationError("candidate score has invalid token count")
    if strategy == "exact_option_letter" and token_count != 1:
        raise ShortEventEvaluationError(
            "exact-option-letter strategy requires one scored native token"
        )
    returned_text = getattr(score, "text", None)
    if returned_text != answer_text.strip():
        raise ShortEventEvaluationError("candidate score text binding differs")
    return CandidateScore(
        option_index=option_index,
        option_letter=option_letter,
        answer_text=answer_text,
        token_count=token_count,
        log_probability_sum=_finite(
            getattr(score, "log_probability_sum", None), "log_probability_sum"
        ),
        log_probability_mean=_finite(
            getattr(score, "log_probability_mean", None), "log_probability_mean"
        ),
        scoring_seconds=elapsed,
    )


def score_question(
    backend: Any,
    *,
    question: QuestionSpec,
    asset_sha256: str,
    visual: TargetVisibleRGB20,
    arm: str,
    scoring_strategy: str,
    predicted_state: PredictedState | None = None,
) -> SuccessfulQuestionResult:
    """Score one label-free question; no correctness field can be produced."""

    if not isinstance(question, QuestionSpec):
        raise ShortEventEvaluationError("question must be QuestionSpec")
    if not isinstance(visual, TargetVisibleRGB20):
        raise ShortEventEvaluationError("visual must be TargetVisibleRGB20")
    if visual.question_id != question.question_id:
        raise ShortEventEvaluationError("visual/question identity differs")
    if scoring_strategy not in SCORING_STRATEGIES:
        raise ShortEventEvaluationError("scoring strategy is invalid")
    family = _validate_arm_backend(backend, arm)
    if arm == "raw_plus_predicted_state":
        if predicted_state is None:
            raise ShortEventEvaluationError("predicted-state arm lacks its evidence")
        if predicted_state.visual_sha256 != visual.visual_sha256:
            raise ShortEventEvaluationError("predicted state binds another visual")
    elif predicted_state is not None:
        raise ShortEventEvaluationError("only the predicted-state arm accepts a hint")

    prompt = render_prompt(
        question,
        scoring_strategy=scoring_strategy,
        predicted_state=predicted_state,
    )
    request = visual.request(prompt)
    answers = candidate_answer_texts(question, scoring_strategy)
    started = perf_counter_ns()
    scores = tuple(
        _score_candidate(
            backend,
            request,
            answer,
            arm=arm,
            option_index=index,
            option_letter=question.option_letters[index],
            strategy=scoring_strategy,
        )
        for index, answer in enumerate(answers)
    )
    elapsed = (perf_counter_ns() - started) / 1_000_000_000.0
    maximum = max(value.log_probability_mean for value in scores)
    tied = tuple(
        value.option_index for value in scores if value.log_probability_mean == maximum
    )
    selected = scores[tied[0]]
    decision = MCQDecision(
        selected_option_index=selected.option_index,
        selected_option_letter=selected.option_letter,
        selected_option_text=question.options[selected.option_index],
        tied_option_indices=tied,
        exact_tie=len(tied) > 1,
    )
    state_record = None if predicted_state is None else predicted_state.to_dict()
    return SuccessfulQuestionResult(
        question_id=question.question_id,
        question_sha256=question.identity_sha256,
        asset_sha256=asset_sha256,
        visual_sha256=visual.visual_sha256,
        data_role=question.data_role,
        review_status=question.review_status,
        source_dataset=question.source_dataset,
        sequence_id=question.sequence_id,
        physical_event_id=question.physical_event_id,
        target_id=question.target_id,
        family=family,
        arm=arm,
        scoring_strategy=scoring_strategy,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        predicted_state=state_record,
        candidate_scores=scores,
        decision=decision,
        scoring_seconds=elapsed,
    )


__all__ = [
    "CandidateScore",
    "FULL_LANGUAGE_ROUTE",
    "MCQDecision",
    "ShortEventEvaluationError",
    "SuccessfulQuestionResult",
    "score_question",
]
