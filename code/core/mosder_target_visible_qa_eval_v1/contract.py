"""Deterministically assign the four motion states to A/B/C/D per example.

Scoring uses the four single-token answer labels. State permutations depend
on the case ID, making them reproducible across evaluation runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from types import MappingProxyType
from typing import Final

import numpy as np


STATE_ORDER: Final[tuple[str, ...]] = (
    "neither",
    "camera_only",
    "object_only",
    "both",
)
LETTERS: Final[tuple[str, ...]] = ("A", "B", "C", "D")
SEED: Final[int] = 20260902

STATE_FACTORS: Final[Mapping[str, tuple[bool, bool]]] = MappingProxyType(
    {
        "neither": (False, False),
        "camera_only": (True, False),
        "object_only": (False, True),
        "both": (True, True),
    }
)
STATE_OPTION_TEXT: Final[Mapping[str, str]] = MappingProxyType(
    {
        "neither": "neither (camera static, object static)",
        "camera_only": "camera_only (camera moving, object static)",
        "object_only": "object_only (camera static, object moving)",
        "both": "both (camera moving, object moving)",
    }
)

# The placeholders contain state descriptions, not letters.  This exact
# template plus the case-specific option mapping is the complete model query.
QA_PROMPT_TEMPLATE: Final[str] = (
    "You are given a 20-frame egocentric video. A red outline marks the queried "
    "target in every frame. Judge motion across the entire 20-frame clip. Camera "
    "motion means that the viewpoint/camera moves. Object motion means that the "
    "queried target moves independently in the scene; do not count apparent "
    "image motion caused only by camera motion. For this case, the options are:\n"
    "A: {A}\n"
    "B: {B}\n"
    "C: {C}\n"
    "D: {D}\n"
    "Reply with exactly one uppercase letter: A, B, C, or D."
)
QA_PROMPT_TEMPLATE_SHA256: Final[str] = hashlib.sha256(
    QA_PROMPT_TEMPLATE.encode("utf-8")
).hexdigest()

FRAME_COUNT: Final[int] = 20
RED_RGB: Final[tuple[int, int, int]] = (255, 0, 0)
OPTION_MAPPING_SCHEMA_VERSION: Final[str] = (
    "mosder_case_permuted_letter_state_mapping_v1"
)
OVERLAY_SCHEMA_VERSION: Final[str] = "mosder_target_visible_red_outline_v1"
DECISION_SCHEMA_VERSION: Final[str] = "mosder_one_token_letter_argmax_v1"
OVERLAY_THICKNESS_RULE: Final[str] = (
    "base=max(2,ceil(min(frame_height,frame_width)/256)); "
    "per_box=min(base,max(1,floor((min(raster_width,raster_height)-1)/2)))"
)
OVERLAY_RASTERIZATION_RULE: Final[str] = (
    "continuous half-open xyxy -> inclusive (floor(x0),floor(y0),ceil(x1)-1,ceil(y1)-1)"
)


def _contract_sha256(value: Mapping[str, object]) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# These hashes bind behavior, not merely implementation filenames.  They are
# designed to be embedded verbatim in later release/protocol receipts.
PROMPT_CONTRACT_SHA256: Final[str] = _contract_sha256(
    {
        "schema_version": OPTION_MAPPING_SCHEMA_VERSION,
        "prompt_template": QA_PROMPT_TEMPLATE,
        "prompt_template_sha256": QA_PROMPT_TEMPLATE_SHA256,
        "seed": SEED,
        "letters": LETTERS,
        "state_order": STATE_ORDER,
        "state_option_text": [
            [state, STATE_OPTION_TEXT[state]] for state in STATE_ORDER
        ],
        "per_case_digest_utf8": "{seed}|{case_id}|{state}",
        "per_case_digest": "sha256 lowercase hexadecimal",
        "permutation": (
            "sort states by (digest,state_order_index) ascending, then zip to A/B/C/D"
        ),
        "truth_is_not_an_input": True,
    }
)
OVERLAY_CONTRACT_SHA256: Final[str] = _contract_sha256(
    {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "frame_count": FRAME_COUNT,
        "input": "uint8 HWC RGB, identical dimensions, finite half-open [20,4] xyxy",
        "output": "fresh C-contiguous uint8 RGB copies",
        "color_rgb": RED_RGB,
        "opacity": "fully opaque",
        "fill": False,
        "border_location": "inside raster box",
        "rasterization_rule": OVERLAY_RASTERIZATION_RULE,
        "thickness_rule": OVERLAY_THICKNESS_RULE,
        "minimum_raster_extent": 3,
        "invalid_input_policy": "fail closed; never clip or repair",
    }
)
SCORER_SEMANTIC_SHA256: Final[str] = _contract_sha256(
    {
        "schema_version": DECISION_SCHEMA_VERSION,
        "letters": LETTERS,
        "state_order": STATE_ORDER,
        "input": "one finite conditional logit for each exact label A/B/C/D",
        "tokenization_precondition": (
            "each exact uppercase label must add exactly one scored token after the "
            "rendered prompt; otherwise fail closed"
        ),
        "selection": "maximum logit",
        "tie_definition": "exact equality after conversion to finite Python float",
        "tie_output": "record all tied letters in A/B/C/D order and select the first",
        "semantic_output": "map selected letter through the case option mapping",
        "near_tie_rounding": False,
    }
)


class QAContractError(RuntimeError):
    """The QA mapping, target overlay, or one-letter decision is invalid."""


def _validated_case_id(case_id: object) -> str:
    if not isinstance(case_id, str) or not case_id or case_id != case_id.strip():
        raise QAContractError("case_id must be a nonempty, already-trimmed string")
    return case_id


def option_mapping(case_id: str) -> dict[str, str]:
    """Return the deterministic letter-to-state permutation for one case.

    States are sorted by ascending SHA-256 hex digest of the exact UTF-8 string
    ``20260902|{case_id}|{state}``, then zipped to A/B/C/D.  No truth label is an
    input, so the permutation is independent of ground truth.
    """

    identity = _validated_case_id(case_id)
    keyed_states = []
    for state in STATE_ORDER:
        payload = f"{SEED}|{identity}|{state}".encode("utf-8")
        keyed_states.append((hashlib.sha256(payload).hexdigest(), state))
    keyed_states.sort(key=lambda item: (item[0], STATE_ORDER.index(item[1])))
    return {letter: keyed_states[index][1] for index, letter in enumerate(LETTERS)}


def _validated_option_mapping(mapping: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(mapping, Mapping) or set(mapping) != set(LETTERS):
        raise QAContractError("option mapping must have exactly A/B/C/D keys")
    ordered = {letter: mapping[letter] for letter in LETTERS}
    if (
        any(not isinstance(state, str) for state in ordered.values())
        or set(ordered.values()) != set(STATE_ORDER)
        or len(set(ordered.values())) != len(STATE_ORDER)
    ):
        raise QAContractError("option mapping must be a permutation of four states")
    return ordered


def render_prompt(case_id: str) -> str:
    """Render the exact per-case QA prompt with its frozen option permutation."""

    mapping = option_mapping(case_id)
    fields = {letter: STATE_OPTION_TEXT[mapping[letter]] for letter in LETTERS}
    prompt = QA_PROMPT_TEMPLATE.format(**fields)
    if not prompt.endswith("Reply with exactly one uppercase letter: A, B, C, or D."):
        raise QAContractError("rendered QA prompt lost its exact-answer instruction")
    return prompt


def _validated_frames(frames: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    try:
        values = tuple(frames)
    except Exception as error:
        raise QAContractError("frames must be a finite sequence") from error
    if len(values) != FRAME_COUNT:
        raise QAContractError(f"exactly {FRAME_COUNT} RGB frames are required")
    shape: tuple[int, int, int] | None = None
    validated: list[np.ndarray] = []
    for frame in values:
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
        ):
            raise QAContractError("every frame must be a uint8 HWC RGB ndarray")
        if frame.shape[0] < 3 or frame.shape[1] < 3:
            raise QAContractError("RGB frames are too small for a non-filling outline")
        if shape is None:
            shape = frame.shape
        elif frame.shape != shape:
            raise QAContractError("all 20 RGB frames must have identical dimensions")
        validated.append(frame)
    return tuple(validated)


def _validated_boxes(
    boxes_xyxy: Sequence[Sequence[float]], *, height: int, width: int
) -> np.ndarray:
    try:
        boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    except Exception as error:
        raise QAContractError("oracle boxes must be numeric [20,4] xyxy") from error
    if boxes.shape != (FRAME_COUNT, 4) or not np.isfinite(boxes).all():
        raise QAContractError("oracle boxes must be finite [20,4] half-open xyxy")
    if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
        raise QAContractError("oracle boxes must be nonempty half-open xyxy")
    if (
        np.any(boxes[:, 0] < 0.0)
        or np.any(boxes[:, 1] < 0.0)
        or np.any(boxes[:, 2] > float(width))
        or np.any(boxes[:, 3] > float(height))
    ):
        raise QAContractError("oracle boxes fall outside their RGB frame")
    return np.ascontiguousarray(boxes)


def _raster_box(
    box: np.ndarray, *, height: int, width: int
) -> tuple[int, int, int, int]:
    left = int(math.floor(float(box[0])))
    top = int(math.floor(float(box[1])))
    right = int(math.ceil(float(box[2]))) - 1
    bottom = int(math.ceil(float(box[3]))) - 1
    if not (0 <= left <= right < width and 0 <= top <= bottom < height):
        raise QAContractError("oracle box cannot be rasterized inside the frame")
    if right - left + 1 < 3 or bottom - top + 1 < 3:
        raise QAContractError(
            "oracle box is raster-degenerate for a non-filling outline"
        )
    return left, top, right, bottom


def _paint_red_border(
    frame: np.ndarray,
    rectangle: tuple[int, int, int, int],
    *,
    thickness: int,
) -> None:
    left, top, right, bottom = rectangle
    if type(thickness) is not int or thickness <= 0:
        raise QAContractError("outline thickness must be a positive integer")
    color = np.asarray(RED_RGB, dtype=np.uint8)
    frame[top : top + thickness, left : right + 1] = color
    frame[bottom - thickness + 1 : bottom + 1, left : right + 1] = color
    frame[top : bottom + 1, left : left + thickness] = color
    frame[top : bottom + 1, right - thickness + 1 : right + 1] = color


@dataclass(frozen=True, slots=True)
class FrameOverlayAudit:
    frame_index: int
    continuous_box_xyxy: tuple[float, float, float, float]
    raster_box_ltrb_inclusive: tuple[int, int, int, int]
    thickness_pixels: int


@dataclass(frozen=True, slots=True)
class OverlayAudit:
    schema_version: str
    frame_count: int
    height: int
    width: int
    color_rgb: tuple[int, int, int]
    fill: bool
    rasterization_rule: str
    thickness_rule: str
    frames: tuple[FrameOverlayAudit, ...]


def overlay_frames(
    frames: Sequence[np.ndarray],
    boxes_xyxy: Sequence[Sequence[float]],
) -> tuple[tuple[np.ndarray, ...], OverlayAudit]:
    """Copy 20 frames and draw a deterministic opaque red inside outline.

    The continuous half-open box is conservatively rasterized with
    floor/ceil-minus-one.  Border thickness starts at
    ``max(2, ceil(min(H,W)/256))`` and is capped per box so at least one center
    pixel remains along its shorter dimension.  Boxes below three raster
    pixels in either dimension are refused.  Inputs are never mutated.
    """

    validated_frames = _validated_frames(frames)
    height, width = validated_frames[0].shape[:2]
    boxes = _validated_boxes(boxes_xyxy, height=height, width=width)
    base_thickness = max(2, math.ceil(min(height, width) / 256))

    rendered: list[np.ndarray] = []
    frame_audits: list[FrameOverlayAudit] = []
    for frame_index, (frame, box) in enumerate(
        zip(validated_frames, boxes, strict=True)
    ):
        rectangle = _raster_box(box, height=height, width=width)
        left, top, right, bottom = rectangle
        minimum_extent = min(right - left + 1, bottom - top + 1)
        thickness = min(base_thickness, max(1, (minimum_extent - 1) // 2))
        if 2 * thickness >= minimum_extent:
            raise QAContractError("outline thickness would fill the oracle box")
        output = np.array(frame, dtype=np.uint8, copy=True, order="C")
        _paint_red_border(output, rectangle, thickness=thickness)
        if np.shares_memory(output, frame):
            raise QAContractError("overlay output unexpectedly aliases its input")
        rendered.append(output)
        frame_audits.append(
            FrameOverlayAudit(
                frame_index=frame_index,
                continuous_box_xyxy=tuple(float(value) for value in box),
                raster_box_ltrb_inclusive=rectangle,
                thickness_pixels=thickness,
            )
        )
    audit = OverlayAudit(
        schema_version=OVERLAY_SCHEMA_VERSION,
        frame_count=FRAME_COUNT,
        height=height,
        width=width,
        color_rgb=RED_RGB,
        fill=False,
        rasterization_rule=OVERLAY_RASTERIZATION_RULE,
        thickness_rule=OVERLAY_THICKNESS_RULE,
        frames=tuple(frame_audits),
    )
    return tuple(rendered), audit


overlay_oracle_track = overlay_frames


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise QAContractError(f"{label} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise QAContractError(f"{label} must be a finite real number")
    return result


@dataclass(frozen=True, slots=True)
class LetterDecision:
    """Deterministic one-letter argmax and its semantic state."""

    predicted_letter: str
    predicted_state: str
    ordered_letter_logits: tuple[tuple[str, float], ...]
    tie_letters: tuple[str, ...]
    tie_states: tuple[str, ...]
    winner_logit: float
    runner_up_logit: float
    margin: float

    @property
    def is_tie(self) -> bool:
        return len(self.tie_letters) > 1

    def letter_logits(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self.ordered_letter_logits))


def choose_from_letter_logits(
    mapping: Mapping[str, str],
    logits: Mapping[str, float],
) -> LetterDecision:
    """Argmax exact A/B/C/D logits and map the winner to semantic state.

    Exact finite-float ties are recorded in A/B/C/D order, and the first tied
    letter is the deterministic output.  Near-equal logits are not rounded.
    """

    options = _validated_option_mapping(mapping)
    if not isinstance(logits, Mapping) or set(logits) != set(LETTERS):
        raise QAContractError("letter logits must have exactly A/B/C/D keys")
    ordered_logits = tuple(
        (letter, _finite_number(logits[letter], label=f"{letter} logit"))
        for letter in LETTERS
    )
    maximum = max(value for _, value in ordered_logits)
    tie_letters = tuple(letter for letter, value in ordered_logits if value == maximum)
    predicted_letter = tie_letters[0]
    descending = sorted((value for _, value in ordered_logits), reverse=True)
    winner_logit = descending[0]
    runner_up_logit = descending[1]
    margin = winner_logit - runner_up_logit
    if margin < 0.0 or not math.isfinite(margin):
        raise QAContractError("letter-logit margin is invalid")
    return LetterDecision(
        predicted_letter=predicted_letter,
        predicted_state=options[predicted_letter],
        ordered_letter_logits=ordered_logits,
        tie_letters=tie_letters,
        tie_states=tuple(options[letter] for letter in tie_letters),
        winner_logit=winner_logit,
        runner_up_logit=runner_up_logit,
        margin=margin,
    )


__all__ = [
    "DECISION_SCHEMA_VERSION",
    "FRAME_COUNT",
    "FrameOverlayAudit",
    "LETTERS",
    "LetterDecision",
    "OPTION_MAPPING_SCHEMA_VERSION",
    "OVERLAY_RASTERIZATION_RULE",
    "OVERLAY_CONTRACT_SHA256",
    "OVERLAY_SCHEMA_VERSION",
    "OVERLAY_THICKNESS_RULE",
    "OverlayAudit",
    "QAContractError",
    "QA_PROMPT_TEMPLATE",
    "QA_PROMPT_TEMPLATE_SHA256",
    "PROMPT_CONTRACT_SHA256",
    "RED_RGB",
    "SEED",
    "SCORER_SEMANTIC_SHA256",
    "STATE_FACTORS",
    "STATE_OPTION_TEXT",
    "STATE_ORDER",
    "choose_from_letter_logits",
    "option_mapping",
    "overlay_frames",
    "overlay_oracle_track",
    "render_prompt",
]
