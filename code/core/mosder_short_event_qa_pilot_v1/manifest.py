"""Build question manifests and load target-visible 20-frame video inputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Final, Mapping, Sequence

import numpy as np

from family_backends_v1 import RGB20Request
from mosder_target_visible_qa_eval_v1.contract import overlay_frames

from .contract import (
    ALLOWED_DATA_ROLES,
    QuestionSpec,
    ShortEventQAContractError,
    canonical_json_bytes,
    question_from_mapping,
)


ASSET_SCHEMA_VERSION: Final[str] = "mosder_short_event_qa_asset_v1"
ASSET_FIELDS: Final[frozenset[str]] = frozenset(
    {"schema_version", "question", "rgb20_npz_path", "rgb20_npz_sha256"}
)
NPZ_KEYS: Final[frozenset[str]] = frozenset({"frames", "timestamps_ns", "boxes_xyxy"})
FRAME_COUNT: Final[int] = 20
_FORBIDDEN_ROLE_TOKENS: Final[frozenset[str]] = frozenset(
    {"val", "validation", "test", "held", "confirmation", "finalb"}
)


class ShortEventManifestError(ShortEventQAContractError):
    """A manifest, scope, asset path, or RGB20 tensor is invalid."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file(path_value: object, *, label: str, suffix: str | None = None) -> Path:
    # LOCAL_PATH: Request manifests and their rgb20_npz_path fields require existing
    # absolute paths on the target machine; relative paths and symlinks are rejected.
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise ShortEventManifestError(f"{label} must be an absolute path")
    path = Path(path_value)
    if path.is_symlink():
        raise ShortEventManifestError(f"{label} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ShortEventManifestError(f"{label} cannot be resolved") from error
    if not resolved.is_file():
        raise ShortEventManifestError(f"{label} must be a regular non-symlink file")
    tokens = {
        token
        for part in resolved.parts
        for token in re.split(r"[^a-z0-9]+", part.lower())
        if token
    }
    if tokens & _FORBIDDEN_ROLE_TOKENS:
        raise ShortEventManifestError(f"{label} enters a forbidden data role")
    lowered = str(resolved).lower()
    if (
        "confirmation-a" in lowered
        or "confirmation_a" in lowered
        or "final-b" in lowered
    ):
        raise ShortEventManifestError(f"{label} enters a forbidden held role")
    if suffix is not None and resolved.suffix.lower() != suffix:
        raise ShortEventManifestError(f"{label} must end in {suffix}")
    return resolved


def _sha(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ShortEventManifestError(f"{label} must be lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class QuestionAsset:
    question: QuestionSpec
    rgb20_npz_path: Path
    rgb20_npz_sha256: str

    @property
    def identity_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "schema_version": ASSET_SCHEMA_VERSION,
                    "question_sha256": self.question.identity_sha256,
                    "rgb20_npz_path": str(self.rgb20_npz_path),
                    "rgb20_npz_sha256": self.rgb20_npz_sha256,
                }
            )
        ).hexdigest()


def asset_from_mapping(
    value: Mapping[str, Any], *, expected_role: str
) -> QuestionAsset:
    if expected_role not in ALLOWED_DATA_ROLES:
        raise ShortEventManifestError("expected role is not pilot-safe")
    if not isinstance(value, Mapping) or set(value) != set(ASSET_FIELDS):
        raise ShortEventManifestError("asset row fields differ")
    if value.get("schema_version") != ASSET_SCHEMA_VERSION:
        raise ShortEventManifestError("asset schema_version differs")
    question = question_from_mapping(value.get("question"))
    if question.data_role != expected_role:
        raise ShortEventManifestError("question role differs from explicit CLI scope")
    path = _safe_file(value.get("rgb20_npz_path"), label="RGB20 NPZ", suffix=".npz")
    digest = _sha(value.get("rgb20_npz_sha256"), "RGB20 NPZ SHA-256")
    if file_sha256(path) != digest:
        raise ShortEventManifestError("RGB20 NPZ SHA-256 differs")
    return QuestionAsset(
        question=question, rgb20_npz_path=path, rgb20_npz_sha256=digest
    )


def _load_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ShortEventManifestError(f"blank JSONL line {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ShortEventManifestError(
                    f"invalid JSONL line {line_number}"
                ) from error
            if not isinstance(value, Mapping):
                raise ShortEventManifestError(
                    f"JSONL line {line_number} is not an object"
                )
            rows.append(value)
    if not rows:
        raise ShortEventManifestError("JSONL input is empty")
    return tuple(rows)


def load_question_assets(
    manifest_path: str | Path, *, expected_role: str
) -> tuple[QuestionAsset, ...]:
    path = _safe_file(str(manifest_path), label="request manifest", suffix=".jsonl")
    assets = tuple(
        asset_from_mapping(row, expected_role=expected_role)
        for row in _load_jsonl(path)
    )
    ids = tuple(asset.question.question_id for asset in assets)
    if len(ids) != len(set(ids)):
        raise ShortEventManifestError(
            "request manifest contains duplicate question IDs"
        )
    return assets


def _visual_sha256(
    frames: np.ndarray, timestamps: np.ndarray, boxes: np.ndarray
) -> str:
    digest = hashlib.sha256()
    for name, value in (
        ("frames", np.ascontiguousarray(frames)),
        ("timestamps_ns", np.ascontiguousarray(timestamps)),
        ("boxes_xyxy", np.ascontiguousarray(boxes)),
    ):
        header = canonical_json_bytes(
            {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        )
        raw = value.tobytes(order="C")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TargetVisibleRGB20:
    question_id: str
    frames: tuple[np.ndarray, ...]
    timestamps_ns: tuple[int, ...]
    boxes_xyxy: tuple[tuple[float, float, float, float], ...]
    visual_sha256: str
    changed_pixels_per_frame: tuple[int, ...]

    def request(self, prompt: str) -> RGB20Request:
        return RGB20Request(
            frames=self.frames,
            timestamps_ns=self.timestamps_ns,
            prompt=prompt,
            oracle_boxes_xyxy=self.boxes_xyxy,
            request_id=self.question_id,
        )


def materialize_target_visible(asset: QuestionAsset) -> TargetVisibleRGB20:
    if not isinstance(asset, QuestionAsset):
        raise ShortEventManifestError("asset must be QuestionAsset")
    if file_sha256(asset.rgb20_npz_path) != asset.rgb20_npz_sha256:
        raise ShortEventManifestError("RGB20 NPZ changed after manifest loading")
    try:
        with np.load(asset.rgb20_npz_path, allow_pickle=False) as archive:
            if set(archive.files) != set(NPZ_KEYS):
                raise ShortEventManifestError("RGB20 NPZ key inventory differs")
            frames = np.ascontiguousarray(archive["frames"])
            timestamps = np.ascontiguousarray(archive["timestamps_ns"])
            boxes = np.ascontiguousarray(archive["boxes_xyxy"])
    except ShortEventManifestError:
        raise
    except Exception as error:
        raise ShortEventManifestError("RGB20 NPZ cannot be decoded safely") from error
    if (
        frames.dtype != np.uint8
        or frames.ndim != 4
        or frames.shape[0] != FRAME_COUNT
        or frames.shape[-1] != 3
        or frames.shape[1] < 3
        or frames.shape[2] < 3
    ):
        raise ShortEventManifestError("frames must be uint8 [20,H,W,3]")
    if (
        timestamps.dtype != np.int64
        or timestamps.shape != (FRAME_COUNT,)
        or np.any(timestamps[1:] <= timestamps[:-1])
    ):
        raise ShortEventManifestError("timestamps_ns must be increasing int64 [20]")
    if boxes.shape != (FRAME_COUNT, 4) or not np.issubdtype(boxes.dtype, np.floating):
        raise ShortEventManifestError("boxes_xyxy must be floating [20,4]")
    boxes = np.ascontiguousarray(boxes, dtype=np.float64)
    if not np.isfinite(boxes).all():
        raise ShortEventManifestError("boxes_xyxy contains non-finite values")
    source_frames = tuple(np.ascontiguousarray(frame) for frame in frames)
    source_hash = _visual_sha256(frames, timestamps, boxes)
    rendered, _audit = overlay_frames(source_frames, boxes)
    changed = tuple(
        int(np.count_nonzero(np.any(before != after, axis=2)))
        for before, after in zip(source_frames, rendered, strict=True)
    )
    if len(changed) != FRAME_COUNT or any(value <= 0 for value in changed):
        raise ShortEventManifestError("target overlay is not visible in every frame")
    return TargetVisibleRGB20(
        question_id=asset.question.question_id,
        frames=rendered,
        timestamps_ns=tuple(int(value) for value in timestamps.tolist()),
        boxes_xyxy=tuple(tuple(float(item) for item in row) for row in boxes.tolist()),
        visual_sha256=source_hash,
        changed_pixels_per_frame=changed,
    )


__all__ = [
    "ASSET_SCHEMA_VERSION",
    "QuestionAsset",
    "ShortEventManifestError",
    "TargetVisibleRGB20",
    "asset_from_mapping",
    "file_sha256",
    "load_question_assets",
    "materialize_target_visible",
]
