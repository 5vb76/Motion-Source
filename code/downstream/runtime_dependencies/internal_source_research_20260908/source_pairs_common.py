"""Validate and load TRAIN-origin source-pair diagnostic inputs.

The manifest contains RGB identity and query metadata; source labels stay in a
separate gold document and are used only by the scoring code.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROLE = (
    "TRAIN-origin exposed source diagnostic; source references retain their numerical HOLD; "
    "not independent generalization, four-state coverage, event ground truth, or the downstream main metric"
)
REQUIRED = {
    "candidate_id",
    "pair_id",
    "sequence_id",
    "official_instance_id",
    "query_target",
    "rgb20_npz_path",
    "rgb20_npz_sha256",
    "rgb_sha256",
    "timestamps_ns",
}
OPTIONAL = {"ordinal", "frame_rgb_sha256", "rgb_shape"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def source_prompts(noun):
    # Deliberately identical to evaluate_source19.py; no state or event wording.
    return {
        "camera": f"Target: the {noun}. Does the camera change its physical position or orientation during this video? Answer only yes or no.",
        "object": f"Does the {noun} change its physical position or orientation relative to the scene during this video? Answer only yes or no.",
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_manifest(document):
    require(document.get("schema_version") == "source_pairs_v1", "wrong input schema")
    require(
        document.get("source_split") == "TRAIN",
        "only authorized TRAIN-origin inputs are supported",
    )
    require(
        document.get("qualification_status") == "FROZEN_VISUALLY_QUALIFIED",
        "visual qualification is not frozen",
    )
    require(
        document.get("oracle_states_or_boxes_in_model_input") is False,
        "input boundary not declared",
    )
    rows = document["items"]
    require(
        document["count"] == len(rows) and len(rows) > 0,
        "input count mismatch or empty diagnostic",
    )
    require(
        len({r["candidate_id"] for r in rows}) == len(rows), "duplicate candidate IDs"
    )
    pairs = defaultdict(list)
    for row in rows:
        require(
            REQUIRED <= set(row) <= REQUIRED | OPTIONAL,
            "input item has missing fields or nonallowlisted metadata (labels/boxes must be separate)",
        )
        for key in (
            "candidate_id",
            "pair_id",
            "sequence_id",
            "official_instance_id",
            "query_target",
        ):
            require(
                isinstance(row[key], str) and row[key].strip() == row[key] and row[key],
                f"invalid {key}",
            )
        for key in ("rgb20_npz_sha256", "rgb_sha256"):
            value = row[key]
            require(
                isinstance(value, str)
                and len(value) == 64
                and set(value) <= set("0123456789abcdef"),
                f"invalid {key}",
            )
        times = row["timestamps_ns"]
        require(
            len(times) == 20 and all(type(x) is int for x in times),
            "expected 20 integer timestamps",
        )
        require(
            all(x < y for x, y in zip(times, times[1:])),
            "timestamps must be strictly increasing",
        )
        pairs[row["pair_id"]].append(row)
    for pair_id, pair in pairs.items():
        require(len(pair) == 2, f"incomplete pair: {pair_id}")
        for key in ("sequence_id", "official_instance_id", "query_target"):
            require(pair[0][key] == pair[1][key], f"pair differs in {key}: {pair_id}")
        require(
            pair[0]["rgb_sha256"] != pair[1]["rgb_sha256"],
            f"identical RGB pair: {pair_id}",
        )
        earlier, later = sorted(pair, key=lambda r: r["timestamps_ns"][0])
        require(
            earlier["timestamps_ns"][-1] < later["timestamps_ns"][0],
            f"overlapping pair windows: {pair_id}",
        )
    return rows


def read_manifest(path, expected_sha):
    """Verify the frozen manifest hash before checking its input contract."""
    require(sha(path) == expected_sha, "qualified input manifest hash mismatch")
    return validate_manifest(json.loads(Path(path).read_text()))


def load_rgb(row):
    """Load twenty unmarked RGB frames and verify the recorded content hashes."""
    # LOCAL_PATH: rgb20_npz_path must resolve to a local archive with the recorded hash.
    require(sha(row["rgb20_npz_path"]) == row["rgb20_npz_sha256"], "NPZ hash mismatch")
    with np.load(row["rgb20_npz_path"], allow_pickle=False) as archive:
        require(
            set(archive.files) == {"rgb", "timestamps_ns"},
            "NPZ contains unexpected fields",
        )
        rgb = archive["rgb"].copy()
        times = archive["timestamps_ns"].tolist()
    require(
        rgb.dtype == np.uint8
        and rgb.ndim == 4
        and rgb.shape[0] == 20
        and rgb.shape[-1] == 3,
        "expected unmarked uint8 RGB20",
    )
    require(times == row["timestamps_ns"], "timestamp content mismatch")
    require(
        hashlib.sha256(rgb.tobytes()).hexdigest() == row["rgb_sha256"],
        "RGB content hash mismatch",
    )
    if "rgb_shape" in row:
        require(list(rgb.shape) == row["rgb_shape"], "RGB shape mismatch")
    if "frame_rgb_sha256" in row:
        require(
            [hashlib.sha256(frame.tobytes()).hexdigest() for frame in rgb]
            == row["frame_rgb_sha256"],
            "frame content hash mismatch",
        )
    return rgb, times


def binding(row):
    """Record the input identity and exact prompts alongside predictions."""
    return {
        key: row[key]
        for key in (
            "candidate_id",
            "pair_id",
            "sequence_id",
            "official_instance_id",
            "query_target",
            "rgb20_npz_sha256",
            "rgb_sha256",
            "timestamps_ns",
        )
    } | {"binary_prompts": source_prompts(row["query_target"])}


def validate_gold(document, rows):
    """Check source labels against the ordered, qualified input pairs."""
    gold = document["items"]
    require(
        document.get("numeric_HOLD_unchanged") is True,
        "source-reference HOLD declaration missing",
    )
    require(document["count"] == len(gold) == len(rows), "gold/input count mismatch")
    require(
        [r["candidate_id"] for r in gold] == [r["candidate_id"] for r in rows],
        "gold order or membership mismatch",
    )
    pairs = defaultdict(list)
    for row, target in zip(rows, gold):
        require(
            target["camera_moving"] is True,
            "this diagnostic requires camera-moving pairs only",
        )
        require(
            type(target["object_moving"]) is bool,
            "object source reference must be boolean",
        )
        require(
            target["state"] == ("both" if target["object_moving"] else "camera_only"),
            "state/reference disagreement",
        )
        require(
            target.get("source_labels_copied_unchanged") is True,
            "source reference was not preserved",
        )
        require(
            target.get("numeric_HOLD_unchanged") is True,
            "item source-reference HOLD declaration missing",
        )
        pairs[row["pair_id"]].append(target["object_moving"])
    require(
        all(sorted(v) == [False, True] for v in pairs.values()),
        "each pair needs one static and one moving target",
    )
    return gold
