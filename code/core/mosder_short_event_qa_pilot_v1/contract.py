"""Separate schemas for multiple-choice question inputs and answer labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

from mosder_target_visible_qa_eval_v1.contract import (
    OVERLAY_CONTRACT_SHA256 as TARGET_VISIBLE_OVERLAY_CONTRACT_SHA256,
)


QUESTION_SCHEMA_VERSION: Final[str] = "mosder_short_event_qa_question_v1"
LABEL_SCHEMA_VERSION: Final[str] = "mosder_short_event_qa_independent_label_v1"
STATE_SCHEMA_VERSION: Final[str] = "mosder_short_event_qa_predicted_state_v1"
RESULT_SCHEMA_VERSION: Final[str] = "mosder_short_event_qa_result_v1"

ALLOWED_DATA_ROLES: Final[tuple[str, ...]] = (
    "train_only_nonoptimizer_technical_pilot",
    "public_external_nonauthorizing_pilot",
)
ARMS: Final[tuple[str, ...]] = (
    "raw_vlm",
    "mosder_full_language",
    "raw_plus_predicted_state",
)
SCORING_STRATEGIES: Final[tuple[str, ...]] = (
    "exact_option_letter",
    "full_option_span_mean_logprob",
)
STATES: Final[tuple[str, ...]] = (
    "neither",
    "camera_only",
    "object_only",
    "both",
)
STATE_FACTORS: Final[Mapping[str, tuple[bool, bool]]] = MappingProxyType(
    {
        "neither": (False, False),
        "camera_only": (True, False),
        "object_only": (False, True),
        "both": (True, True),
    }
)
LETTERS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E")
FACTOR_PROMPT: Final[str] = (
    "Analyze the motion source in this 20-frame video for the queried target "
    "specified by the oracle box track. Distinguish camera motion from "
    "target-object motion. Respond using exactly four lines: Camera, Object, "
    "State, Description."
)
SIX_SPAN_PREDICTOR_CONTRACT: Final[Mapping[str, object]] = MappingProxyType(
    {
        "schema_version": "mosder_frozen_six_canonical_span_predictor_v1",
        "prompt": FACTOR_PROMPT,
        "camera_route_candidates": ["neither", "camera_only"],
        "object_route_candidates": [
            "neither",
            "camera_only",
            "object_only",
            "both",
        ],
        "score": "native_answer_token_mean_log_probability",
        "camera_margin": "camera_camera_only-camera_neither",
        "object_margin": (
            "0.5*((object_object_only-object_neither)+(object_both-object_camera_only))"
        ),
        "logits": "q_camera=camera_margin+b_c;q_object=object_margin+b_o",
        "decision": "strict_greater_than_zero;exact_zero_is_static",
        "four_choice_state_classifier_forbidden": True,
        "target_visible_overlay_contract_sha256": (
            TARGET_VISIBLE_OVERLAY_CONTRACT_SHA256
        ),
    }
)

BASE_PROMPT_TEMPLATE: Final[str] = (
    "You are given a 20-frame video clip. A red outline marks the "
    "queried target in every frame. Use the entire observed clip and answer "
    "the multiple-choice question.\nQuestion: {question}\n{options}\n{answer_rule}"
)
STATE_HINT_TEMPLATE: Final[str] = (
    "A frozen motion-source model supplied this fallible diagnostic for the "
    "same marked target: camera_moving={camera}; target_moving={object}; "
    "state={state}. Use it only as evidence.\n"
)
ANSWER_RULES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "exact_option_letter": (
            "Reply with exactly one uppercase option letter and no other text."
        ),
        "full_option_span_mean_logprob": (
            "Reply with exactly the full text of the selected option and no other text."
        ),
    }
)


class ShortEventQAContractError(ValueError):
    """A question, label, state hint, score, or decision is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ShortEventQAContractError("value is outside canonical JSON") from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


SIX_SPAN_PREDICTOR_CONTRACT_SHA256: Final[str] = canonical_sha256(
    dict(SIX_SPAN_PREDICTOR_CONTRACT)
)
MCQ_SCORING_CONTRACT: Final[Mapping[str, object]] = MappingProxyType(
    {
        "schema_version": "mosder_short_event_qa_mcq_scoring_contract_v1",
        "candidate_count": "2_to_5_in_manifest_order",
        "base_prompt_template": BASE_PROMPT_TEMPLATE,
        "state_hint_template": STATE_HINT_TEMPLATE,
        "answer_rules": dict(ANSWER_RULES),
        "candidate_value": "native_answer_token_mean_log_probability",
        "full_language_route": "FULL_LANGUAGE",
        "raw_route": None,
        "decision": "argmax_candidate_value",
        "exact_tie_policy": "lowest_zero_based_option_index",
        "target_visible_overlay_contract_sha256": (
            TARGET_VISIBLE_OVERLAY_CONTRACT_SHA256
        ),
    }
)
MCQ_SCORING_CONTRACT_SHA256: Final[str] = canonical_sha256(dict(MCQ_SCORING_CONTRACT))


def _identifier(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character in value for character in "\x00\r\n")
    ):
        raise ShortEventQAContractError(
            f"{label} must be a nonempty, already-trimmed identifier"
        )
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ShortEventQAContractError(f"{label} must be nonempty text")
    return value.strip()


def _sha256(value: object, label: str) -> str:
    output = _identifier(value, label)
    if len(output) != 64 or any(c not in "0123456789abcdef" for c in output):
        raise ShortEventQAContractError(f"{label} must be lowercase SHA-256")
    return output


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ShortEventQAContractError(f"{label} must be a finite real")
    output = float(value)
    if not math.isfinite(output):
        raise ShortEventQAContractError(f"{label} must be a finite real")
    return output


def _exact_keys(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], label: str
) -> None:
    if set(value) != set(expected):
        raise ShortEventQAContractError(
            f"{label} fields differ: missing={sorted(set(expected) - set(value))}, "
            f"extra={sorted(set(value) - set(expected))}"
        )


@dataclass(frozen=True, slots=True)
class QuestionSpec:
    """Label-free semantic portion of one question."""

    question_id: str
    data_role: str
    review_status: str
    optimizer_membership: str
    source_dataset: str
    sequence_id: str
    physical_event_id: str
    target_id: str
    question: str
    options: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "question_id",
            "source_dataset",
            "sequence_id",
            "physical_event_id",
            "target_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if self.data_role not in ALLOWED_DATA_ROLES:
            raise ShortEventQAContractError("question data_role is not pilot-safe")
        if self.review_status not in {"adjudicated", "pending_review"}:
            raise ShortEventQAContractError("question review_status is invalid")
        expected_membership = {
            "train_only_nonoptimizer_technical_pilot": "confirmed_nonoptimizer_reserve",
            "public_external_nonauthorizing_pilot": "public_external_not_applicable",
        }[self.data_role]
        if self.optimizer_membership != expected_membership:
            raise ShortEventQAContractError(
                "question optimizer-membership claim differs from its role"
            )
        object.__setattr__(self, "question", _text(self.question, "question"))
        if not isinstance(self.options, tuple):
            object.__setattr__(self, "options", tuple(self.options))
        if not 2 <= len(self.options) <= 5:
            raise ShortEventQAContractError("a question requires 2--5 options")
        normalized = tuple(
            _text(option, f"option[{index}]")
            for index, option in enumerate(self.options)
        )
        if len(set(normalized)) != len(normalized):
            raise ShortEventQAContractError("question options must be distinct")
        object.__setattr__(self, "options", normalized)

    @property
    def option_letters(self) -> tuple[str, ...]:
        return LETTERS[: len(self.options)]

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": QUESTION_SCHEMA_VERSION,
            "question_id": self.question_id,
            "data_role": self.data_role,
            "review_status": self.review_status,
            "optimizer_membership": self.optimizer_membership,
            "source_dataset": self.source_dataset,
            "sequence_id": self.sequence_id,
            "physical_event_id": self.physical_event_id,
            "target_id": self.target_id,
            "question": self.question,
            "options": list(self.options),
        }


QUESTION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "question_id",
        "data_role",
        "review_status",
        "optimizer_membership",
        "source_dataset",
        "sequence_id",
        "physical_event_id",
        "target_id",
        "question",
        "options",
    }
)


def question_from_mapping(value: Mapping[str, Any]) -> QuestionSpec:
    if not isinstance(value, Mapping):
        raise ShortEventQAContractError("question row must be a mapping")
    _exact_keys(value, QUESTION_FIELDS, "question row")
    if value.get("schema_version") != QUESTION_SCHEMA_VERSION:
        raise ShortEventQAContractError("question schema_version differs")
    options = value.get("options")
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        raise ShortEventQAContractError("question options must be a sequence")
    return QuestionSpec(
        question_id=value.get("question_id"),
        data_role=value.get("data_role"),
        review_status=value.get("review_status"),
        optimizer_membership=value.get("optimizer_membership"),
        source_dataset=value.get("source_dataset"),
        sequence_id=value.get("sequence_id"),
        physical_event_id=value.get("physical_event_id"),
        target_id=value.get("target_id"),
        question=value.get("question"),
        options=tuple(options),
    )


@dataclass(frozen=True, slots=True)
class IndependentLabel:
    """Loaded only by post-hoc metrics, never by request construction."""

    question_id: str
    data_role: str
    physical_event_id: str
    correct_option_index: int
    annotation_source: str
    independent_of_mosder_training_labels: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "question_id", _identifier(self.question_id, "question_id")
        )
        if self.data_role not in ALLOWED_DATA_ROLES:
            raise ShortEventQAContractError("label data_role is not pilot-safe")
        object.__setattr__(
            self,
            "physical_event_id",
            _identifier(self.physical_event_id, "physical_event_id"),
        )
        if (
            type(self.correct_option_index) is not int
            or not 0 <= self.correct_option_index < 5
        ):
            raise ShortEventQAContractError("correct_option_index is outside [0,5)")
        object.__setattr__(
            self,
            "annotation_source",
            _text(self.annotation_source, "annotation_source"),
        )
        if self.independent_of_mosder_training_labels is not True:
            raise ShortEventQAContractError(
                "pilot labels must be independent of MoSDeR training labels"
            )

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": LABEL_SCHEMA_VERSION, **asdict(self)}


LABEL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "question_id",
        "data_role",
        "physical_event_id",
        "correct_option_index",
        "annotation_source",
        "independent_of_mosder_training_labels",
    }
)


def label_from_mapping(value: Mapping[str, Any]) -> IndependentLabel:
    if not isinstance(value, Mapping):
        raise ShortEventQAContractError("label row must be a mapping")
    _exact_keys(value, LABEL_FIELDS, "label row")
    if value.get("schema_version") != LABEL_SCHEMA_VERSION:
        raise ShortEventQAContractError("label schema_version differs")
    return IndependentLabel(
        question_id=value.get("question_id"),
        data_role=value.get("data_role"),
        physical_event_id=value.get("physical_event_id"),
        correct_option_index=value.get("correct_option_index"),
        annotation_source=value.get("annotation_source"),
        independent_of_mosder_training_labels=value.get(
            "independent_of_mosder_training_labels"
        ),
    )


@dataclass(frozen=True, slots=True)
class PredictedState:
    question_id: str
    visual_sha256: str
    state: str
    q_camera: float
    q_object: float
    predictor_checkpoint_sha256: str
    predictor_contract_sha256: str
    data_role: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "question_id", _identifier(self.question_id, "question_id")
        )
        object.__setattr__(
            self, "visual_sha256", _sha256(self.visual_sha256, "visual_sha256")
        )
        if self.state not in STATES:
            raise ShortEventQAContractError("predicted state is invalid")
        object.__setattr__(self, "q_camera", _finite(self.q_camera, "q_camera"))
        object.__setattr__(self, "q_object", _finite(self.q_object, "q_object"))
        object.__setattr__(
            self,
            "predictor_checkpoint_sha256",
            _sha256(self.predictor_checkpoint_sha256, "predictor_checkpoint_sha256"),
        )
        object.__setattr__(
            self,
            "predictor_contract_sha256",
            _sha256(self.predictor_contract_sha256, "predictor_contract_sha256"),
        )
        if self.predictor_contract_sha256 != SIX_SPAN_PREDICTOR_CONTRACT_SHA256:
            raise ShortEventQAContractError(
                "predicted state was not produced by frozen six-span readout"
            )
        if self.data_role not in ALLOWED_DATA_ROLES:
            raise ShortEventQAContractError("predicted-state role is not pilot-safe")
        expected = STATE_FACTORS[self.state]
        if expected != (self.q_camera > 0.0, self.q_object > 0.0):
            raise ShortEventQAContractError("predicted state disagrees with q logits")

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": STATE_SCHEMA_VERSION, **asdict(self)}


STATE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "question_id",
        "visual_sha256",
        "state",
        "q_camera",
        "q_object",
        "predictor_checkpoint_sha256",
        "predictor_contract_sha256",
        "data_role",
    }
)


def predicted_state_from_mapping(value: Mapping[str, Any]) -> PredictedState:
    if not isinstance(value, Mapping):
        raise ShortEventQAContractError("predicted-state row must be a mapping")
    _exact_keys(value, STATE_FIELDS, "predicted-state row")
    if value.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ShortEventQAContractError("predicted-state schema_version differs")
    return PredictedState(
        question_id=value.get("question_id"),
        visual_sha256=value.get("visual_sha256"),
        state=value.get("state"),
        q_camera=value.get("q_camera"),
        q_object=value.get("q_object"),
        predictor_checkpoint_sha256=value.get("predictor_checkpoint_sha256"),
        predictor_contract_sha256=value.get("predictor_contract_sha256"),
        data_role=value.get("data_role"),
    )


def render_prompt(
    question: QuestionSpec,
    *,
    scoring_strategy: str,
    predicted_state: PredictedState | None = None,
) -> str:
    if not isinstance(question, QuestionSpec):
        raise ShortEventQAContractError("question must be QuestionSpec")
    if scoring_strategy not in SCORING_STRATEGIES:
        raise ShortEventQAContractError("scoring strategy is invalid")
    if predicted_state is not None:
        if (
            not isinstance(predicted_state, PredictedState)
            or predicted_state.question_id != question.question_id
            or predicted_state.data_role != question.data_role
        ):
            raise ShortEventQAContractError("predicted state/question binding differs")
        camera, obj = STATE_FACTORS[predicted_state.state]
        prefix = STATE_HINT_TEMPLATE.format(
            camera=str(camera).lower(),
            object=str(obj).lower(),
            state=predicted_state.state,
        )
    else:
        prefix = ""
    options = "\n".join(
        f"{letter}: {text}"
        for letter, text in zip(question.option_letters, question.options, strict=True)
    )
    return prefix + BASE_PROMPT_TEMPLATE.format(
        question=question.question,
        options=options,
        answer_rule=ANSWER_RULES[scoring_strategy],
    )


def candidate_answer_texts(
    question: QuestionSpec, scoring_strategy: str
) -> tuple[str, ...]:
    if scoring_strategy == "exact_option_letter":
        return question.option_letters
    if scoring_strategy == "full_option_span_mean_logprob":
        return question.options
    raise ShortEventQAContractError("scoring strategy is invalid")


__all__ = [
    "ALLOWED_DATA_ROLES",
    "ARMS",
    "IndependentLabel",
    "LABEL_SCHEMA_VERSION",
    "MCQ_SCORING_CONTRACT",
    "MCQ_SCORING_CONTRACT_SHA256",
    "PredictedState",
    "QUESTION_SCHEMA_VERSION",
    "QuestionSpec",
    "RESULT_SCHEMA_VERSION",
    "SCORING_STRATEGIES",
    "SIX_SPAN_PREDICTOR_CONTRACT",
    "SIX_SPAN_PREDICTOR_CONTRACT_SHA256",
    "STATE_SCHEMA_VERSION",
    "STATES",
    "TARGET_VISIBLE_OVERLAY_CONTRACT_SHA256",
    "ShortEventQAContractError",
    "candidate_answer_texts",
    "canonical_json_bytes",
    "canonical_sha256",
    "label_from_mapping",
    "predicted_state_from_mapping",
    "question_from_mapping",
    "render_prompt",
]
