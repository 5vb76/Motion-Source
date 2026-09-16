"""Four-line motion answers, parser, and native tokenizer span alignment.

Span plans map Camera, Object, State, and Description to answer-token ranges.
Training and scoring validate against the same plan."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import hmac
import json
from types import MappingProxyType
from typing import Final, Protocol

import torch

from family_backends_v1 import (
    FamilyBinding,
    PreparedNativeInputs,
    RGB20Request,
    TeacherForcedTensorScore,
)

from .contract import LANGUAGE_SCHEMA_LINES, STATE_FACTORS, STATE_ORDER


LANGUAGE_PLAN_SCHEMA_VERSION: Final[str] = "mosder_canonical_language_plan_v1"
LANGUAGE_SPAN_ORDER: Final[tuple[str, ...]] = tuple(
    prefix.removesuffix(":").lower() for prefix in LANGUAGE_SCHEMA_LINES
)
_MOTION_VALUES: Final[frozenset[str]] = frozenset(("static", "moving"))
_LOWER_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")

# These strings are package-owned and intentionally do not import an archived
# smoke script.  The first three fields are the strict public schema; the
# fourth is the one canonical constrained-language target for each state.
CANONICAL_ANSWERS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "neither": (
            "Camera: static\n"
            "Object: static\n"
            "State: neither\n"
            "Description: the camera and queried target remain stationary."
        ),
        "camera_only": (
            "Camera: moving\n"
            "Object: static\n"
            "State: camera_only\n"
            "Description: the camera moves while the queried target remains stationary."
        ),
        "object_only": (
            "Camera: static\n"
            "Object: moving\n"
            "State: object_only\n"
            "Description: the queried target moves while the camera remains stationary."
        ),
        "both": (
            "Camera: moving\n"
            "Object: moving\n"
            "State: both\n"
            "Description: both the camera and queried target move."
        ),
    }
)


class LanguagePlanContractError(RuntimeError):
    """A language schema, native token plan, or score contract failed."""


class NativeLanguagePreprocessor(Protocol):
    """The deliberately small backend surface used by this module."""

    binding: FamilyBinding

    def prepare_rgb20(
        self,
        request: RGB20Request,
        *,
        answer_text: str | None = None,
    ) -> PreparedNativeInputs: ...


@dataclass(frozen=True, slots=True)
class ParsedMotionLanguage:
    """One schema-valid, factor-consistent four-line generated answer."""

    text: str
    camera: str
    object: str
    state: str
    description: str

    @property
    def camera_moving(self) -> bool:
        return self.camera == "moving"

    @property
    def object_moving(self) -> bool:
        return self.object == "moving"

    @property
    def factors(self) -> tuple[bool, bool]:
        return self.camera_moving, self.object_moving


@dataclass(frozen=True, slots=True)
class CanonicalLanguagePlan:
    """Immutable identity of one family-native canonical answer token plan."""

    schema_version: str
    family: str
    state: str
    text: str
    answer_token_ids: tuple[int, ...]
    spans: Mapping[str, tuple[int, int]]
    sha256: str

    def as_dict(self) -> dict[str, object]:
        """Return the exact JSON-compatible values bound by ``sha256``."""

        return {
            **_plan_payload(
                schema_version=self.schema_version,
                family=self.family,
                state=self.state,
                text=self.text,
                answer_token_ids=self.answer_token_ids,
                spans=self.spans,
            ),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class _NativeTokenization:
    prompt_token_ids: tuple[int, ...]
    answer_token_ids: tuple[int, ...]


def _reject_embedded_line_controls(text: str) -> None:
    # ``str.split("\n")`` alone would not catch CR, Unicode line separators,
    # or other C0 controls that render as an extra line in downstream tools.
    forbidden = {"\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"}
    if any(character in forbidden for character in text):
        raise LanguagePlanContractError(
            "language text contains a forbidden non-LF line/control separator"
        )
    if any(ord(character) < 32 and character != "\n" for character in text):
        raise LanguagePlanContractError(
            "language text contains a forbidden control character"
        )


def parse_four_line_language(text: str) -> ParsedMotionLanguage:
    """Parse exactly four fixed-prefix lines and enforce factor/state meaning.

    ``Camera`` and ``Object`` accept only the public values ``static`` and
    ``moving``.  ``State`` accepts exactly the four frozen states and must
    agree with those two factors.  ``Description`` may be constrained natural
    language, but it must be one nonempty, whitespace-trimmed line.
    """

    if not isinstance(text, str):
        raise LanguagePlanContractError("language text must be a string")
    _reject_embedded_line_controls(text)
    lines = text.split("\n")
    if len(lines) != len(LANGUAGE_SCHEMA_LINES):
        raise LanguagePlanContractError(
            "language output must contain exactly four LF-separated lines"
        )

    values: list[str] = []
    for line, prefix in zip(lines, LANGUAGE_SCHEMA_LINES, strict=True):
        exact_prefix = f"{prefix} "
        if not line.startswith(exact_prefix):
            raise LanguagePlanContractError(
                f"language line must start exactly with {exact_prefix!r}"
            )
        value = line[len(exact_prefix) :]
        if not value or value != value.strip():
            raise LanguagePlanContractError(
                f"{prefix.removesuffix(':')} value must be nonempty and trimmed"
            )
        values.append(value)

    camera, obj, state, description = values
    if camera not in _MOTION_VALUES:
        raise LanguagePlanContractError(
            "Camera value must be exactly 'static' or 'moving'"
        )
    if obj not in _MOTION_VALUES:
        raise LanguagePlanContractError(
            "Object value must be exactly 'static' or 'moving'"
        )
    if state not in STATE_ORDER:
        raise LanguagePlanContractError(
            "State value must be neither/camera_only/object_only/both"
        )
    observed_factors = (camera == "moving", obj == "moving")
    if observed_factors != STATE_FACTORS[state]:
        raise LanguagePlanContractError(
            "Camera/Object factors are inconsistent with State"
        )
    # The prefix/value loop already establishes a nonempty, trimmed value;
    # retain this explicit branch as the semantic Description contract.
    if not description:
        raise LanguagePlanContractError("Description must be nonempty")

    return ParsedMotionLanguage(
        text=text,
        camera=camera,
        object=obj,
        state=state,
        description=description,
    )


def canonical_answer(state: str) -> str:
    """Return the one exact package-owned answer for a frozen state."""

    if not isinstance(state, str) or state not in STATE_ORDER:
        raise LanguagePlanContractError(f"unsupported canonical state: {state!r}")
    try:
        text = CANONICAL_ANSWERS[state]
    except KeyError as error:  # fail closed if the mapping and state ABI drift
        raise LanguagePlanContractError(
            f"canonical answer is absent for state {state!r}"
        ) from error
    parsed = parse_four_line_language(text)
    if parsed.state != state or parsed.factors != STATE_FACTORS[state]:
        raise LanguagePlanContractError(
            f"canonical answer semantics drifted for state {state!r}"
        )
    return text


def _backend_family(backend: object) -> str:
    binding = getattr(backend, "binding", None)
    family = getattr(binding, "key", None)
    if (
        not isinstance(binding, FamilyBinding)
        or not isinstance(family, str)
        or not family
        or family != family.strip()
    ):
        raise LanguagePlanContractError(
            "language preprocessing requires a bound nonempty native family key"
        )
    if not callable(getattr(backend, "prepare_rgb20", None)):
        raise LanguagePlanContractError(
            "language preprocessing requires backend.prepare_rgb20"
        )
    return family


def _prepared_tokenization(
    prepared: object,
    *,
    expected_text: str,
) -> _NativeTokenization:
    if not isinstance(prepared, PreparedNativeInputs):
        raise LanguagePlanContractError(
            "backend.prepare_rgb20 returned the wrong prepared-input type"
        )
    if prepared.answer_text != expected_text:
        raise LanguagePlanContractError(
            "prepared answer text differs from the requested cumulative text"
        )
    input_ids = prepared.input_ids
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.dtype != torch.long
        or input_ids.ndim != 2
        or input_ids.shape[0] != 1
    ):
        raise LanguagePlanContractError(
            "prepared input_ids must be one batch of native torch.long IDs"
        )
    prompt_count = prepared.prompt_token_count
    if (
        type(prompt_count) is not int
        or prompt_count <= 0
        or prompt_count >= input_ids.shape[1]
    ):
        raise LanguagePlanContractError(
            "prepared prompt/answer token boundary is invalid or empty"
        )
    if not isinstance(prepared.model_kwargs, Mapping):
        raise LanguagePlanContractError("prepared model_kwargs must be a mapping")
    keyword_ids = prepared.model_kwargs.get("input_ids")
    if (
        not isinstance(keyword_ids, torch.Tensor)
        or keyword_ids.dtype != torch.long
        or keyword_ids.shape != input_ids.shape
        or not torch.equal(keyword_ids.detach().cpu(), input_ids.detach().cpu())
    ):
        raise LanguagePlanContractError(
            "prepared model_kwargs/input_ids identity drifted"
        )

    prompt_ids = tuple(
        int(value) for value in input_ids[0, :prompt_count].detach().cpu().tolist()
    )
    answer_ids = tuple(
        int(value) for value in input_ids[0, prompt_count:].detach().cpu().tolist()
    )
    if not prompt_ids or not answer_ids:
        raise LanguagePlanContractError(
            "prepared native prompt and answer token sequences must be nonempty"
        )
    if any(token_id < 0 for token_id in answer_ids):
        raise LanguagePlanContractError(
            "prepared native answer contains a negative token ID"
        )
    return _NativeTokenization(
        prompt_token_ids=prompt_ids,
        answer_token_ids=answer_ids,
    )


def _prepare_answer_tokens(
    backend: NativeLanguagePreprocessor,
    request: RGB20Request,
    text: str,
) -> _NativeTokenization:
    try:
        prepared = backend.prepare_rgb20(request, answer_text=text)
    except Exception as error:
        raise LanguagePlanContractError(
            "backend.prepare_rgb20 failed during language token planning"
        ) from error
    return _prepared_tokenization(prepared, expected_text=text)


def _longest_common_prefix(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> int:
    count = 0
    for left_id, right_id in zip(left, right):
        if left_id != right_id:
            break
        count += 1
    return count


def _validated_spans(
    spans: object,
    *,
    token_count: int,
) -> dict[str, tuple[int, int]]:
    if not isinstance(spans, Mapping):
        raise LanguagePlanContractError("language spans must be a mapping")
    if tuple(spans) != LANGUAGE_SPAN_ORDER:
        raise LanguagePlanContractError(
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
            raise LanguagePlanContractError(
                f"{name} span must be one integer (start,end) tuple"
            )
        start, end = bounds
        if start != cursor or end <= start or end > token_count:
            raise LanguagePlanContractError(
                "language spans must be an ordered contiguous nonempty partition"
            )
        validated[name] = (start, end)
        cursor = end
    if cursor != token_count:
        raise LanguagePlanContractError(
            "language spans must cover every native answer token exactly once"
        )
    return validated


def _plan_payload(
    *,
    schema_version: str,
    family: str,
    state: str,
    text: str,
    answer_token_ids: tuple[int, ...],
    spans: Mapping[str, tuple[int, int]],
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "family": family,
        "state": state,
        "text": text,
        "answer_token_ids": list(answer_token_ids),
        "spans": [
            [name, spans[name][0], spans[name][1]] for name in LANGUAGE_SPAN_ORDER
        ],
    }


def _payload_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def plan_canonical_language(
    backend: NativeLanguagePreprocessor,
    request: RGB20Request,
    state: str,
) -> CanonicalLanguagePlan:
    """Plan four nonempty native-token spans without running the model.

    The complete answer is tokenized once.  The first three line-terminated
    cumulative prefixes are tokenized independently, and their longest common
    prefixes with the complete native answer define the first three span
    boundaries.  Any prompt drift, retokenization that cannot form four
    strictly increasing boundaries, or malformed backend result fails closed.
    """

    if not isinstance(request, RGB20Request):
        raise LanguagePlanContractError("request must be an RGB20Request")
    family = _backend_family(backend)
    text = canonical_answer(state)
    full = _prepare_answer_tokens(backend, request, text)

    lines = text.split("\n")
    if len(lines) != len(LANGUAGE_SPAN_ORDER):
        raise LanguagePlanContractError("canonical four-line text drifted")
    cumulative_prefixes = tuple(
        "\n".join(lines[:line_count]) + "\n" for line_count in range(1, len(lines))
    )
    boundaries: list[int] = []
    for prefix_text in cumulative_prefixes:
        prefix = _prepare_answer_tokens(backend, request, prefix_text)
        if prefix.prompt_token_ids != full.prompt_token_ids:
            raise LanguagePlanContractError(
                "native prompt token IDs drift across cumulative answers"
            )
        boundaries.append(
            _longest_common_prefix(
                prefix.answer_token_ids,
                full.answer_token_ids,
            )
        )

    if _backend_family(backend) != family:
        raise LanguagePlanContractError(
            "native backend family changed during language token planning"
        )
    boundaries.append(len(full.answer_token_ids))
    if any(
        right <= left
        for left, right in zip((0, *boundaries[:-1]), boundaries, strict=True)
    ):
        raise LanguagePlanContractError(
            "cumulative native LCPs do not define four consecutive nonempty spans"
        )

    cursor = 0
    span_dict: dict[str, tuple[int, int]] = {}
    for name, boundary in zip(LANGUAGE_SPAN_ORDER, boundaries, strict=True):
        span_dict[name] = (cursor, boundary)
        cursor = boundary
    validated_spans = _validated_spans(
        span_dict,
        token_count=len(full.answer_token_ids),
    )
    immutable_spans = MappingProxyType(validated_spans)
    payload = _plan_payload(
        schema_version=LANGUAGE_PLAN_SCHEMA_VERSION,
        family=family,
        state=state,
        text=text,
        answer_token_ids=full.answer_token_ids,
        spans=immutable_spans,
    )
    plan = CanonicalLanguagePlan(
        schema_version=LANGUAGE_PLAN_SCHEMA_VERSION,
        family=family,
        state=state,
        text=text,
        answer_token_ids=full.answer_token_ids,
        spans=immutable_spans,
        sha256=_payload_sha256(payload),
    )
    return validate_language_plan(plan)


def validate_language_plan(
    plan: CanonicalLanguagePlan,
) -> CanonicalLanguagePlan:
    """Revalidate every semantic, partition, and digest field of a plan."""

    if not isinstance(plan, CanonicalLanguagePlan):
        raise LanguagePlanContractError("language plan must be a CanonicalLanguagePlan")
    if plan.schema_version != LANGUAGE_PLAN_SCHEMA_VERSION:
        raise LanguagePlanContractError("language plan schema version drifted")
    if (
        not isinstance(plan.family, str)
        or not plan.family
        or plan.family != plan.family.strip()
    ):
        raise LanguagePlanContractError("language plan family is invalid")
    expected_text = canonical_answer(plan.state)
    if plan.text != expected_text:
        raise LanguagePlanContractError("language plan canonical text drifted")
    if (
        not isinstance(plan.answer_token_ids, tuple)
        or not plan.answer_token_ids
        or any(
            type(token_id) is not int or token_id < 0
            for token_id in plan.answer_token_ids
        )
    ):
        raise LanguagePlanContractError(
            "language plan token IDs must be one nonempty tuple of native IDs"
        )
    spans = _validated_spans(
        plan.spans,
        token_count=len(plan.answer_token_ids),
    )
    if (
        not isinstance(plan.sha256, str)
        or len(plan.sha256) != 64
        or any(character not in _LOWER_HEX for character in plan.sha256)
    ):
        raise LanguagePlanContractError(
            "language plan sha256 must be 64 lowercase hexadecimal characters"
        )
    expected_sha256 = _payload_sha256(
        _plan_payload(
            schema_version=plan.schema_version,
            family=plan.family,
            state=plan.state,
            text=plan.text,
            answer_token_ids=plan.answer_token_ids,
            spans=spans,
        )
    )
    if not hmac.compare_digest(plan.sha256, expected_sha256):
        raise LanguagePlanContractError("language plan sha256 mismatch")
    return plan


def validate_score_against_plan(
    score: TeacherForcedTensorScore,
    plan: CanonicalLanguagePlan,
) -> TeacherForcedTensorScore:
    """Require a differentiable native score to match one plan exactly."""

    validate_language_plan(plan)
    if not isinstance(score, TeacherForcedTensorScore):
        raise LanguagePlanContractError(
            "native score must be a TeacherForcedTensorScore"
        )
    try:
        score.validate()
    except Exception as error:
        raise LanguagePlanContractError("native score validation failed") from error
    if score.family != plan.family:
        raise LanguagePlanContractError("native score family differs from plan")
    if score.text != plan.text:
        raise LanguagePlanContractError("native score text differs from plan")

    answer_ids = score.answer_token_ids
    if (
        not isinstance(answer_ids, torch.Tensor)
        or answer_ids.dtype != torch.long
        or answer_ids.ndim != 1
    ):
        raise LanguagePlanContractError(
            "native score answer token IDs must be one torch.long vector"
        )
    observed_ids = tuple(int(value) for value in answer_ids.detach().cpu().tolist())
    if observed_ids != plan.answer_token_ids:
        raise LanguagePlanContractError(
            "native score answer token IDs differ from plan"
        )

    log_probabilities = score.token_log_probabilities
    nll = score.negative_log_likelihood
    if (
        not isinstance(log_probabilities, torch.Tensor)
        or not log_probabilities.is_floating_point()
        or log_probabilities.ndim != 1
        or log_probabilities.shape != answer_ids.shape
        or log_probabilities.device != answer_ids.device
        or not bool(torch.isfinite(log_probabilities).all())
        or bool((log_probabilities > 0).any())
        or not log_probabilities.requires_grad
        or log_probabilities.grad_fn is None
    ):
        raise LanguagePlanContractError(
            "native score token log probabilities violate the plan ABI"
        )
    if (
        not isinstance(nll, torch.Tensor)
        or not nll.is_floating_point()
        or nll.ndim != 0
        or nll.device != log_probabilities.device
        or nll.dtype != log_probabilities.dtype
        or not bool(torch.isfinite(nll))
        or not nll.requires_grad
        or nll.grad_fn is None
    ):
        raise LanguagePlanContractError(
            "native score negative log likelihood violates the plan ABI"
        )
    if not torch.equal(nll, -log_probabilities.mean()):
        raise LanguagePlanContractError(
            "native score negative log likelihood differs from token scores"
        )
    return score


__all__ = [
    "CANONICAL_ANSWERS",
    "CanonicalLanguagePlan",
    "LANGUAGE_PLAN_SCHEMA_VERSION",
    "LANGUAGE_SPAN_ORDER",
    "LanguagePlanContractError",
    "NativeLanguagePreprocessor",
    "ParsedMotionLanguage",
    "canonical_answer",
    "parse_four_line_language",
    "plan_canonical_language",
    "validate_language_plan",
    "validate_score_against_plan",
]
