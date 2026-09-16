#!/usr/bin/env python3
"""Extract model input fields and motion labels from Strict-V3 manifests.

The projection keeps RGB locators, target boxes, and factor/state/language
supervision. Dense trajectory targets and source evidence stay in the
upstream records. Release checks apply to these files."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence


METHOD_PUBLIC_NAME = "MoSDeR"
METHOD_ARCHITECTURE_VERSION = "MoSDeR-v1"
IMPLEMENTATION_VARIANT = "headless_explicit_residual"
HOLD_STATUS = "HOLD_GATE2_V3"
OPERATIONAL_STATUS = "CLOSED_NONAUTHORIZING"

SOURCE_ROW_SCHEMA = "tst_plugin_o18_strict_v3_operational_case_v2"
PROJECTION_ROW_SCHEMA = "mosder_v1_sanitized_consumed_field_projection_case_v2"
RECEIPT_SCHEMA = "mosder_v1_sanitized_consumed_field_projection_receipt_v2"
STATUS = (
    "PASS_MOSDER_V1_SANITIZED_CONSUMED_FIELD_PROJECTION_V2_NONAUTHORIZING_HOLD_GATE2_V3"
)

ALLOWED_ROLES = ("train", "validation")
ALLOWED_STATES = ("neither", "camera_only", "object_only", "both")
ALLOWED_SOURCES = ("ADT-LiteOffice", "HOT3D", "TACO-V1-allocentric")
LOCAL_SOURCES = frozenset(("ADT-LiteOffice", "HOT3D"))
TACO_SOURCE = "TACO-V1-allocentric"
BOX_ANCHOR_INDICES = (0, 9, 19)
BOX_TO_MASK_CONTRACT = (
    "interpolate_cxcy_logwh_then_floor_min_ceil_max_clip_union_adjacent_frame_pairs_v1"
)

QUERY_TEXT = (
    "Analyze the motion source in this 20-frame video for the queried target "
    "specified by the oracle box track. Distinguish camera motion from target-object "
    "motion. Respond using exactly four lines: Camera, Object, State, Description."
)

STATE_FACTORS: Mapping[str, tuple[bool, bool]] = {
    "neither": (False, False),
    "camera_only": (True, False),
    "object_only": (False, True),
    "both": (True, True),
}

CANONICAL_ANSWERS: Mapping[str, str] = {
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

# LOCAL_PATH: External source checkout containing the loader, language module,
# and quality policy referenced below; these paths do not use this checkout.
REPOSITORY_ROOT = Path("/root/story2_camera_object_motion")
# LOCAL_PATH: Sanitized train/validation manifests and their validation records.
SANITIZED_ROOT = Path("/root/autodl-tmp/tst_na_v3_trainval_ready_candidate_v1")
QUALITY_POLICY = (
    REPOSITORY_ROOT / "experiments/tst_na_v3_train8000_ready_candidate_v1/"
    "TRAINONLY_RGB20_STATE_O18_QUALITY_POLICY_V1.json"
)
QUALITY_POLICY_SHA256 = (
    "9335865cb090265166e2bc7498c03c83c42b11c0db9a199ebab9feddcf381717"
)
PRACTICAL_READY_POSTHOC_SEAL = (
    SANITIZED_ROOT / "EXECUTION_EVIDENCE_V1/PRACTICAL_READY_POSTHOC_SEAL.json"
)
PRACTICAL_READY_POSTHOC_SEAL_SHA256 = (
    "efcc02caeb56902bec8719e5a4a62036ea5e09309ea064213363f77a7b68c651"
)
REQUIRED_EXCLUDED_TRAIN_CASE = "hot3d_train_002379_w000030_n030_target_27"
FORMAL_RGB20_LOADER = REPOSITORY_ROOT / "scripts/formal_rgb20_input_loader_v1.py"
FORMAL_RGB20_LOADER_SHA256 = (
    "9cd729cb9df992f4c46dd00a1cc50a06b57a5d1b76fe893d3caa4e07b78fa212"
)
MOSDER_LANGUAGE = (
    REPOSITORY_ROOT
    / "experiments/tst_native_adapter_o18_sandbox_v1/mosder_final_v1/language.py"
)
MOSDER_LANGUAGE_SHA256 = (
    "d6e8bba0b013cc8dd44483462471b50d6be4f592150d52b228e80dcd6b0ef189"
)

# LOCAL_PATH: Destination for projected manifests; readers must use the same root.
DEFAULT_OUTPUT_ROOT = Path(
    "/root/autodl-tmp/tst_native_adapter_o18_sandbox_v1/"
    "MOSDER_V1_SANITIZED_CONSUMED_FIELD_PROJECTION_V2_NONAUTHORIZING"
)

OUTPUT_MANIFEST_NAMES = {
    "train": "train_mosder_v1_sanitized_consumed_fields_v2.jsonl",
    "validation": "validation_mosder_v1_sanitized_consumed_fields_v2.jsonl",
}
RECEIPT_NAME = "MOSDER_V1_SANITIZED_CONSUMED_FIELD_PROJECTION_RECEIPT_V2.json"
LEDGER_NAME = "SHA256SUMS"
MARKERS = (
    "NONAUTHORIZING",
    "NOT_AUTHORIZED_FOR_TRAINING",
    "HOLD_GATE2_V3_UNCHANGED",
    "PROJECTION_EXECUTION_HELD_PAYLOAD_ACCESS_FALSE",
    "CAMERA18_OBJECT18_NOT_CONSUMED",
    "QUALITY_SANITIZED_7959_1991",
    "SUPERSEDES_8000_2000_DATA_CHAIN",
)

SOURCE_TOP_LEVEL_KEYS = frozenset(
    (
        "anchors",
        "authority",
        "candidate_method_name",
        "identity",
        "method_name",
        "record_payload_sha256",
        "rgb20",
        "runtime_label_source_match_attested_upstream",
        "schema_version",
        "selection_row_binding",
        "selector_row_canonical_sha256",
        "source_adapter",
        "source_evidence",
        "source_supervision_contract_complete",
        "supervision",
    )
)
SOURCE_IDENTITY_KEYS = frozenset(
    (
        "case_id",
        "physical_window_id",
        "rig_id",
        "sequence_id",
        "source_dataset",
        "split_role",
        "state",
        "target_id",
    )
)
SOURCE_AUTHORITY = {
    "execution_path_sealed_payload_accessed": False,
    "formal_gate2_pass": False,
    "formal_release_authorized": False,
    "formal_ten_gate_pass": False,
    "ready_to_train": False,
    "sealed_role_payload_accessed": False,
    "training_authorized": False,
    "underlying_gate2_status": HOLD_STATUS,
}

FORBIDDEN_EXACT_KEYS = frozenset(
    (
        "camera18",
        "object18",
        "values18",
        "translation9",
        "rotation9",
        "loss_scalar_mask18",
        "loss_scalar_mask9",
        "valid_vector_mask3",
        "target_payload_binding",
        "source_evidence",
        "trajectory_anchors_are_box_anchors",
    )
)


class ProjectionHold(RuntimeError):
    """A projection, source binding, role, or authority contract failed."""


@dataclass(frozen=True, slots=True)
class SourceSpec:
    role: str
    path: Path
    sha256: str
    rows: int
    directory: Path
    audit_sha256: str
    completion_sha256: str
    ledger_sha256: str
    retained_membership_sha256: str
    quarantine_sha256: str
    failure_provenance_sha256: str
    input_path: Path
    input_manifest_sha256: str
    input_rows: int
    quarantined_rows: int


# LOCAL_PATH: Each role needs its sanitized manifest and the external input_path
# manifest below; configure these together with SANITIZED_ROOT.
SOURCE_SPECS: Mapping[str, SourceSpec] = {
    "train": SourceSpec(
        role="train",
        directory=SANITIZED_ROOT / "TRAIN_SANITIZED_V1",
        path=(
            SANITIZED_ROOT
            / "TRAIN_SANITIZED_V1"
            / "train_o18_strict_v3_quality_retained_v1.jsonl"
        ),
        sha256="69f04339f384709fe1871f0f760f32083df27cd0726f51d98ca2450c32416ba9",
        rows=7959,
        audit_sha256="d8e3bc55216f8446a2f7ddd9ba7d70d15da8940daf6952932deddcb911207577",
        completion_sha256="158f8122fe24d52d714789de06e4468b34dab840f384ce49952253f8c979f920",
        ledger_sha256="16de985454409e45f83c1b9b09f3e2378810bbd4a8011fa0e50cdc24dc9f0da4",
        retained_membership_sha256="fdafb9e9f9f4653635483dc0553107a84a003dd6c68138162bd75651a8340193",
        quarantine_sha256="5ff5e6644d15e85410a383fd936928128393b3986eefcf0ff014e0a33faf4608",
        failure_provenance_sha256="2899232b53ac64b8e7ed922a33063dfcec6d69ba9eeba910e4eff26902bd0ffb",
        input_path=(
            Path("/root/autodl-tmp/tst_plugin_formal_preconfirmation_v2")
            / "TST_PLUGIN_O18_STRICT_V3_OPERATIONAL_DATA_CONTRACT_V2_NONAUTHORIZING"
            / "train_o18_strict_v3_operational_v2.jsonl"
        ),
        input_manifest_sha256="2595242cec6b2e652180d5c9b0107b2a2900fb78b919cc1f5808a0b098c075d6",
        input_rows=8000,
        quarantined_rows=41,
    ),
    "validation": SourceSpec(
        role="validation",
        directory=SANITIZED_ROOT / "VALIDATION_SANITIZED_V1",
        path=(
            SANITIZED_ROOT
            / "VALIDATION_SANITIZED_V1"
            / "validation_o18_strict_v3_quality_retained_v1.jsonl"
        ),
        sha256="681a96b3a4e520ed08c78318d59a033f6cfd8b319c582bd1ca0812cc2a9338df",
        rows=1991,
        audit_sha256="bb4ce5ac8a73cf2a0cf6274907fde8e5cd207a65ec2f0ba4e9fe7967eec9378a",
        completion_sha256="5f374960129448960f20a8d1182b486edc156b307c972bdfb6c41137e1d952fb",
        ledger_sha256="b406494dcf8188641737c9e52e58903dccfe65ae737f4a26b5e7bf5bbc6a6a67",
        retained_membership_sha256="24b562398f8261b058daf5ecea0e206448dba204a5954ebc638e270e12a2cb4e",
        quarantine_sha256="25a74bb27bff66e784505b488808278c035cce1a6b28e7fc69da1746204a2a9e",
        failure_provenance_sha256="2899232b53ac64b8e7ed922a33063dfcec6d69ba9eeba910e4eff26902bd0ffb",
        input_path=(
            Path("/root/autodl-tmp/tst_plugin_formal_preconfirmation_v2")
            / "TST_PLUGIN_O18_STRICT_V3_OPERATIONAL_DATA_CONTRACT_V2_NONAUTHORIZING"
            / "validation_o18_strict_v3_operational_v2.jsonl"
        ),
        input_manifest_sha256="f428b2d38c38804b7eec6bc59ed8e2f0587bd8a2a8c5368ec4fab7c96e733ee6",
        input_rows=2000,
        quarantined_rows=9,
    ),
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_json_line(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProjectionHold(f"{label} must be a lowercase SHA-256")
    return value


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ProjectionHold(f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def strict_json_bytes(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload, object_pairs_hook=_strict_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProjectionHold(f"{label} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ProjectionHold(f"{label} must contain one JSON object")
    if canonical_json_bytes(value) != payload:
        raise ProjectionHold(f"{label} is not canonical JSON")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: Iterable[str], label: str
) -> None:
    observed = frozenset(value)
    wanted = frozenset(expected)
    if observed != wanted:
        raise ProjectionHold(
            f"{label} keys drifted: missing={sorted(wanted - observed)}, "
            f"extra={sorted(observed - wanted)}"
        )


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ProjectionHold(f"{label} must be a nonempty trimmed string")
    return value


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ProjectionHold(f"{label} must be an integer >= {minimum}")
    return value


def _require_bool(value: Any, expected: bool, label: str) -> None:
    if value is not expected:
        raise ProjectionHold(f"{label} must be {expected}")


def _require_numeric_box(
    value: Any, width: int, height: int, label: str
) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ProjectionHold(f"{label} must be one xyxy list")
    if any(type(item) not in (int, float) for item in value):
        raise ProjectionHold(f"{label} coordinates must be numeric and not bool")
    box = [float(item) for item in value]
    if not all(math.isfinite(item) for item in box):
        raise ProjectionHold(f"{label} contains a non-finite coordinate")
    x0, y0, x1, y1 = box
    if not (0.0 <= x0 < x1 <= width and 0.0 <= y0 < y1 <= height):
        raise ProjectionHold(f"{label} is empty or outside its source-native grid")
    return box


def _assert_allowed_manifest_path(spec: SourceSpec) -> Path:
    expected_directory = SANITIZED_ROOT / (
        "TRAIN_SANITIZED_V1" if spec.role == "train" else "VALIDATION_SANITIZED_V1"
    )
    expected = expected_directory / spec.path.name
    resolved = spec.path.resolve(strict=True)
    if resolved != expected.resolve(strict=True):
        raise ProjectionHold(
            f"{spec.role} source path is not the exact sanitized retained manifest"
        )
    if spec.role not in ALLOWED_ROLES:
        raise ProjectionHold(f"unsupported role: {spec.role!r}")
    if resolved.name != (
        "train_o18_strict_v3_quality_retained_v1.jsonl"
        if spec.role == "train"
        else "validation_o18_strict_v3_quality_retained_v1.jsonl"
    ):
        raise ProjectionHold(f"{spec.role} source filename drifted")
    return resolved


def _iter_strict_source_rows(
    spec: SourceSpec,
) -> Iterator[tuple[int, bytes, Mapping[str, Any]]]:
    path = _assert_allowed_manifest_path(spec)
    if sha256_file(path) != spec.sha256:
        raise ProjectionHold(f"{spec.role} source manifest SHA-256 drifted")
    original_path = spec.input_path.resolve(strict=True)
    if sha256_file(original_path) != spec.input_manifest_sha256:
        raise ProjectionHold(f"{spec.role} original operational manifest drifted")
    rows = 0
    previous_original_index = -1
    original_cursor = -1
    with path.open("rb") as handle, original_path.open("rb") as original_handle:
        for index, raw in enumerate(handle):
            if not raw.endswith(b"\n") or raw == b"\n":
                raise ProjectionHold(f"{spec.role}[{index}] lacks one nonempty LF line")
            payload = raw[:-1]
            row = strict_json_bytes(payload, f"{spec.role}[{index}]")
            selection = row.get("selection_row_binding")
            original_index = (
                selection.get("row_index_zero_based")
                if isinstance(selection, Mapping)
                else None
            )
            if (
                type(original_index) is not int
                or original_index <= previous_original_index
                or original_index >= spec.input_rows
            ):
                raise ProjectionHold(
                    f"{spec.role}[{index}] retained/original row ordering drifted"
                )
            original_raw = b""
            while original_cursor < original_index:
                original_raw = original_handle.readline()
                original_cursor += 1
                if not original_raw:
                    raise ProjectionHold(
                        f"{spec.role}[{index}] original operational row is absent"
                    )
            if original_raw != raw:
                raise ProjectionHold(
                    f"{spec.role}[{index}] is not the original operational row byte stream"
                )
            if (
                spec.role == "train"
                and row.get("identity", {}).get("case_id")
                == REQUIRED_EXCLUDED_TRAIN_CASE
            ):
                raise ProjectionHold(
                    "required both #04 quarantine case survived retention"
                )
            previous_original_index = original_index
            yield index, payload, row
            rows += 1
    if rows != spec.rows:
        raise ProjectionHold(
            f"{spec.role} row count drifted: observed={rows}, expected={spec.rows}"
        )


def _validate_source_envelope(
    row: Mapping[str, Any], *, role: str, row_index: int
) -> Mapping[str, Any]:
    label = f"{role}[{row_index}]"
    _require_exact_keys(row, SOURCE_TOP_LEVEL_KEYS, label)
    if row.get("schema_version") != SOURCE_ROW_SCHEMA:
        raise ProjectionHold(f"{label} source schema drifted")
    if (
        row.get("method_name") != "TsT-plugin"
        or row.get("candidate_method_name") != "TsT-plugin-O18"
    ):
        raise ProjectionHold(f"{label} source method identity drifted")
    _require_bool(
        row.get("source_supervision_contract_complete"),
        True,
        f"{label}.source_supervision_contract_complete",
    )
    if row.get("authority") != SOURCE_AUTHORITY:
        raise ProjectionHold(f"{label} authority/HOLD boundary drifted")
    identity = row.get("identity")
    if not isinstance(identity, dict):
        raise ProjectionHold(f"{label}.identity must be an object")
    _require_exact_keys(identity, SOURCE_IDENTITY_KEYS, f"{label}.identity")
    if identity.get("split_role") != role:
        raise ProjectionHold(f"{label} role mismatch")
    if identity.get("source_dataset") not in ALLOWED_SOURCES:
        raise ProjectionHold(f"{label} source is unsupported")
    expected_adapter = (
        "hash_bound_local_native_rgb20_v1"
        if identity.get("source_dataset") in LOCAL_SOURCES
        else "taco_formal_marker_removed_rgb20_v1"
    )
    if row.get("source_adapter") != expected_adapter:
        raise ProjectionHold(f"{label} source adapter drifted")
    expected_runtime_attestation = identity.get("source_dataset") != TACO_SOURCE
    _require_bool(
        row.get("runtime_label_source_match_attested_upstream"),
        expected_runtime_attestation,
        f"{label}.runtime_label_source_match_attested_upstream",
    )
    if identity.get("state") not in ALLOWED_STATES:
        raise ProjectionHold(f"{label} state is unsupported")
    for key in SOURCE_IDENTITY_KEYS - {"state", "split_role"}:
        _require_text(identity.get(key), f"{label}.identity.{key}")
    declared = _sha(row.get("record_payload_sha256"), f"{label}.record_payload_sha256")
    payload = dict(row)
    payload.pop("record_payload_sha256")
    if canonical_sha256(payload) != declared:
        raise ProjectionHold(f"{label} record payload hash drifted")
    return identity


def _local_visual_projection(
    rgb20: Mapping[str, Any], box_record: Mapping[str, Any], *, role: str, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_rgb_keys = {
        "frame_count",
        "frames",
        "historical_pixel_decode_evidence_bound",
        "locator_kind",
        "physical_pixels_redecoded_by_this_report",
        "raw_rgb_container",
        "selection_algorithm",
        "selection_sha256",
    }
    _require_exact_keys(rgb20, expected_rgb_keys, f"{label}.rgb20")
    if (
        rgb20.get("frame_count") != 20
        or rgb20.get("locator_kind") != "native_frame_locator_bundle"
    ):
        raise ProjectionHold(f"{label} local RGB20 identity drifted")
    _require_bool(
        rgb20.get("historical_pixel_decode_evidence_bound"),
        True,
        f"{label}.historical_pixel_decode_evidence_bound",
    )
    _require_bool(
        rgb20.get("physical_pixels_redecoded_by_this_report"),
        False,
        f"{label}.physical_pixels_redecoded_by_this_report",
    )
    _sha(rgb20.get("selection_sha256"), f"{label}.rgb20.selection_sha256")
    frames = rgb20.get("frames")
    if not isinstance(frames, list) or len(frames) != 20:
        raise ProjectionHold(f"{label} local RGB20 must contain 20 frame locators")
    frame_keys = {
        "height",
        "locator",
        "model_frame_index",
        "native_frame_id",
        "native_ordinal",
        "pixel_rgb_sha256",
        "timestamp_ns",
        "width",
    }
    normalized_frames: list[dict[str, Any]] = []
    timestamps: list[int] = []
    grid: tuple[int, int] | None = None
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ProjectionHold(f"{label}.rgb20.frames[{index}] must be an object")
        _require_exact_keys(frame, frame_keys, f"{label}.rgb20.frames[{index}]")
        if frame.get("model_frame_index") != index:
            raise ProjectionHold(f"{label} local model frame order drifted")
        width = _require_int(frame.get("width"), f"{label}.frame.width", minimum=1)
        height = _require_int(frame.get("height"), f"{label}.frame.height", minimum=1)
        if grid is None:
            grid = (width, height)
        if grid != (width, height):
            raise ProjectionHold(f"{label} local frame grid is inconsistent")
        timestamp = _require_int(
            frame.get("timestamp_ns"), f"{label}.frame.timestamp_ns", minimum=0
        )
        timestamps.append(timestamp)
        locator = _require_text(frame.get("locator"), f"{label}.frame.locator")
        storage_role = "train" if role == "train" else "val"
        if f"/{storage_role}/" not in locator:
            raise ProjectionHold(f"{label} local locator crosses its role boundary")
        _sha(frame.get("pixel_rgb_sha256"), f"{label}.frame.pixel_rgb_sha256")
        normalized_frames.append(dict(frame))
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ProjectionHold(
            f"{label} local frame timestamps are not strictly increasing"
        )
    assert grid is not None
    raw_container = rgb20.get("raw_rgb_container")
    if not isinstance(raw_container, dict):
        raise ProjectionHold(f"{label}.raw_rgb_container must be an object")
    _require_exact_keys(
        raw_container,
        {"base_path", "path_kind", "stable_stat_fingerprint_sha256"},
        f"{label}.raw_rgb_container",
    )
    if raw_container.get("path_kind") != "regular_file":
        raise ProjectionHold(f"{label} local raw container kind drifted")
    storage_role = "train" if role == "train" else "val"
    if f"/{storage_role}/" not in _require_text(
        raw_container.get("base_path"), f"{label}.raw_rgb_container.base_path"
    ):
        raise ProjectionHold(f"{label} local raw container crosses its role boundary")
    _sha(
        raw_container.get("stable_stat_fingerprint_sha256"),
        f"{label}.raw_rgb_container.stable_stat_fingerprint_sha256",
    )

    anchors = box_record.get("anchors")
    if not isinstance(anchors, list) or len(anchors) != 3:
        raise ProjectionHold(f"{label} local box anchors must contain three records")
    normalized_anchors: list[dict[str, Any]] = []
    for slot, anchor in zip(BOX_ANCHOR_INDICES, anchors, strict=True):
        if not isinstance(anchor, dict):
            raise ProjectionHold(f"{label} local anchor must be an object")
        if anchor.get("model_frame_index") != slot:
            raise ProjectionHold(f"{label} local anchor slot drifted")
        if anchor.get("box_to_mask_contract") != BOX_TO_MASK_CONTRACT:
            raise ProjectionHold(f"{label} local box contract drifted")
        if anchor.get("timestamp_ns") != timestamps[slot]:
            raise ProjectionHold(f"{label} local anchor timestamp drifted")
        box = _require_numeric_box(
            anchor.get("box_xyxy"), grid[0], grid[1], f"{label}.anchor[{slot}].box"
        )
        normalized_anchors.append(
            {
                "model_frame_index": slot,
                "box_xyxy_half_open_float": box,
                "temporal_reference": {
                    "kind": "timestamp_ns",
                    "value": timestamps[slot],
                },
            }
        )
    if box_record.get("model_frame_indices") != list(BOX_ANCHOR_INDICES):
        raise ProjectionHold(f"{label} local box anchor index list drifted")
    _sha(box_record.get("geometry_sha256"), f"{label}.box.geometry_sha256")

    visual = {
        "kind": "native_frame_locator_bundle",
        "frame_count": 20,
        "selection_algorithm": rgb20["selection_algorithm"],
        "selection_sha256": rgb20["selection_sha256"],
        "source_native_grid_wh": [grid[0], grid[1]],
        "raw_rgb_container": dict(raw_container),
        "frames": normalized_frames,
        "temporal_order": {"kind": "timestamp_ns", "values": timestamps},
    }
    boxes = {
        "runtime_shape": [20, 4],
        "coordinate_space": "source_native_decoded_RGB_pixels_xyxy_half_open",
        "source_native_grid_wh": [grid[0], grid[1]],
        "anchor_model_frame_indices": list(BOX_ANCHOR_INDICES),
        "anchors": normalized_anchors,
        "interpolation_and_mask_contract": BOX_TO_MASK_CONTRACT,
        "runtime_resolution": "deterministic_from_bound_timestamps_and_three_anchors",
    }
    return visual, boxes


def _taco_visual_projection(
    rgb20: Mapping[str, Any], box_record: Mapping[str, Any], *, role: str, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_rgb_keys = {
        "exact_twenty_actual_RGB_frames_decoded_upstream",
        "formal_RGB_container",
        "frame_count",
        "full_container_content_sha256_revalidated_by_this_report",
        "locator_kind",
        "physical_pixels_redecoded_by_this_report",
        "selection_algorithm",
        "source_frame_ordinals",
    }
    _require_exact_keys(rgb20, expected_rgb_keys, f"{label}.rgb20")
    if (
        rgb20.get("frame_count") != 20
        or rgb20.get("locator_kind") != "formal_video_path_plus_exact_source_ordinals"
    ):
        raise ProjectionHold(f"{label} TACO RGB20 identity drifted")
    _require_bool(
        rgb20.get("exact_twenty_actual_RGB_frames_decoded_upstream"),
        True,
        f"{label}.exact_twenty_actual_RGB_frames_decoded_upstream",
    )
    _require_bool(
        rgb20.get("full_container_content_sha256_revalidated_by_this_report"),
        True,
        f"{label}.full_container_content_sha256_revalidated_by_this_report",
    )
    _require_bool(
        rgb20.get("physical_pixels_redecoded_by_this_report"),
        False,
        f"{label}.physical_pixels_redecoded_by_this_report",
    )
    ordinals = rgb20.get("source_frame_ordinals")
    if (
        not isinstance(ordinals, list)
        or len(ordinals) != 20
        or any(type(value) is not int or value < 0 for value in ordinals)
        or any(right <= left for left, right in zip(ordinals, ordinals[1:]))
    ):
        raise ProjectionHold(f"{label} TACO source ordinals are invalid")
    container = rgb20.get("formal_RGB_container")
    if not isinstance(container, dict):
        raise ProjectionHold(f"{label}.formal_RGB_container must be an object")
    _require_exact_keys(
        container,
        {"bytes", "grid_wh", "is_raw_or_original", "path", "sha256", "variant"},
        f"{label}.formal_RGB_container",
    )
    _require_int(container.get("bytes"), f"{label}.container.bytes", minimum=1)
    _require_bool(
        container.get("is_raw_or_original"), False, f"{label}.is_raw_or_original"
    )
    if container.get("variant") != "OFFICIAL_MARKER_REMOVED_DERIVED_PIXEL_RGB":
        raise ProjectionHold(f"{label} TACO RGB variant drifted")
    path = _require_text(container.get("path"), f"{label}.container.path")
    if f"/{role}/" not in path:
        raise ProjectionHold(f"{label} TACO container crosses its role boundary")
    _sha(container.get("sha256"), f"{label}.container.sha256")
    grid_value = container.get("grid_wh")
    if (
        not isinstance(grid_value, list)
        or len(grid_value) != 2
        or any(type(item) is not int or item <= 0 for item in grid_value)
    ):
        raise ProjectionHold(f"{label} TACO grid_wh is invalid")
    width, height = grid_value

    anchors = box_record.get("anchors")
    if not isinstance(anchors, list) or len(anchors) != 3:
        raise ProjectionHold(f"{label} TACO box anchors must contain three records")
    normalized_anchors: list[dict[str, Any]] = []
    expected_anchor_keys = {
        "box_xyxy_half_open_float",
        "coordinate_space",
        "model_frame_index",
        "source_frame_ordinal",
    }
    for slot, anchor in zip(BOX_ANCHOR_INDICES, anchors, strict=True):
        if not isinstance(anchor, dict):
            raise ProjectionHold(f"{label} TACO anchor must be an object")
        _require_exact_keys(anchor, expected_anchor_keys, f"{label}.anchor[{slot}]")
        if anchor.get("model_frame_index") != slot:
            raise ProjectionHold(f"{label} TACO anchor slot drifted")
        if anchor.get("coordinate_space") != {"width": width, "height": height}:
            raise ProjectionHold(f"{label} TACO anchor coordinate space drifted")
        if anchor.get("source_frame_ordinal") != ordinals[slot]:
            raise ProjectionHold(f"{label} TACO anchor ordinal drifted")
        box = _require_numeric_box(
            anchor.get("box_xyxy_half_open_float"),
            width,
            height,
            f"{label}.anchor[{slot}].box",
        )
        normalized_anchors.append(
            {
                "model_frame_index": slot,
                "box_xyxy_half_open_float": box,
                "temporal_reference": {
                    "kind": "source_frame_ordinal",
                    "value": ordinals[slot],
                },
            }
        )
    if box_record.get("model_frame_indices") != list(BOX_ANCHOR_INDICES):
        raise ProjectionHold(f"{label} TACO box anchor index list drifted")
    _sha(box_record.get("geometry_sha256"), f"{label}.box.geometry_sha256")

    visual = {
        "kind": "formal_video_path_plus_exact_source_ordinals",
        "frame_count": 20,
        "selection_algorithm": rgb20["selection_algorithm"],
        "source_native_grid_wh": [width, height],
        "formal_RGB_container": dict(container),
        "source_frame_ordinals": list(ordinals),
        "temporal_order": {
            "kind": "decoded_timestamp_ns_from_bound_container_and_ordinals",
            "source_frame_ordinals": list(ordinals),
            "decoded_timestamps_required_at_runtime": True,
        },
    }
    boxes = {
        "runtime_shape": [20, 4],
        "coordinate_space": "source_native_decoded_RGB_pixels_xyxy_half_open",
        "source_native_grid_wh": [width, height],
        "anchor_model_frame_indices": list(BOX_ANCHOR_INDICES),
        "anchors": normalized_anchors,
        "interpolation_and_mask_contract": BOX_TO_MASK_CONTRACT,
        "runtime_resolution": "deterministic_after_bound_frame_timestamp_decode",
    }
    return visual, boxes


def _assert_forbidden_dense_fields_absent(value: Any, path: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_EXACT_KEYS:
                raise ProjectionHold(
                    f"forbidden dense/trajectory field survived projection: {path}.{key}"
                )
            _assert_forbidden_dense_fields_absent(
                item, f"{path}.{key}" if path else key
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_forbidden_dense_fields_absent(item, f"{path}[{index}]")


def project_source_row(
    row: Mapping[str, Any], *, role: str, row_index: int, source_manifest_sha256: str
) -> dict[str, Any]:
    """Project one exact sanitized operational-format row to the MoSDeR ABI."""

    if role not in ALLOWED_ROLES:
        raise ProjectionHold(f"unsupported projection role: {role!r}")
    identity = _validate_source_envelope(row, role=role, row_index=row_index)
    label = f"{role}[{row_index}]"
    rgb20 = row.get("rgb20")
    anchors = row.get("anchors")
    if not isinstance(rgb20, dict) or not isinstance(anchors, dict):
        raise ProjectionHold(f"{label} RGB20/anchors must be objects")
    box_record = anchors.get("box")
    if not isinstance(box_record, dict):
        raise ProjectionHold(f"{label}.anchors.box must be an object")
    source = str(identity["source_dataset"])
    if source in LOCAL_SOURCES:
        visual, box_track = _local_visual_projection(
            rgb20, box_record, role=role, label=label
        )
    elif source == TACO_SOURCE:
        visual, box_track = _taco_visual_projection(
            rgb20, box_record, role=role, label=label
        )
    else:  # guarded above, retained fail closed for future source changes
        raise ProjectionHold(f"{label} unsupported source")

    state = str(identity["state"])
    camera_moving, object_moving = STATE_FACTORS[state]
    selection_binding = row.get("selection_row_binding")
    original_row_index = (
        selection_binding.get("row_index_zero_based")
        if isinstance(selection_binding, Mapping)
        else None
    )
    spec = SOURCE_SPECS[role]
    if (
        type(original_row_index) is not int
        or original_row_index < 0
        or original_row_index >= spec.input_rows
    ):
        raise ProjectionHold(f"{label} original operational row index drifted")
    projected: dict[str, Any] = {
        "schema_version": PROJECTION_ROW_SCHEMA,
        "method_identity": {
            "public_name": METHOD_PUBLIC_NAME,
            "architecture_version": METHOD_ARCHITECTURE_VERSION,
            "implementation_variant": IMPLEMENTATION_VARIANT,
        },
        "audit_identity": dict(identity),
        "loader_request": {
            "split_role": role,
            "source_dataset": source,
            "case_id": identity["case_id"],
            "resolver": "formal_rgb20_input_loader_v1",
        },
        "model_input": {
            "rgb20": visual,
            "timestamps": {
                "consumed_for_deterministic_order_validation": True,
                "learned_model_feature": False,
                "source": visual["temporal_order"],
            },
            "target_query": QUERY_TEXT,
            "oracle_box_track": box_track,
        },
        "supervision": {
            "camera_moving": camera_moving,
            "object_moving": object_moving,
            "state": state,
            "canonical_answer": CANONICAL_ANSWERS[state],
            "dense_numeric_trajectory_target_consumed": False,
            "trajectory_derived_state_label": True,
        },
        "source_binding": {
            "sanitized_manifest_sha256": _sha(
                source_manifest_sha256, f"{label}.source_manifest_sha256"
            ),
            "sanitized_row_index_zero_based": row_index,
            "original_operational_manifest_sha256": spec.input_manifest_sha256,
            "original_operational_row_index_zero_based": original_row_index,
            "source_record_payload_sha256": row["record_payload_sha256"],
            "source_rgb20_canonical_sha256": canonical_sha256(rgb20),
            "source_box_record_canonical_sha256": canonical_sha256(box_record),
            "selector_row_canonical_sha256": row["selector_row_canonical_sha256"],
        },
        "authority": {
            "operational_data_status": OPERATIONAL_STATUS,
            "formal_data_role_authority_complete": False,
            "formal_data_release": False,
            "formal_training_authorized": False,
            "formal_validation_authorized": False,
            "projection_is_authorization": False,
            "underlying_gate2_status": HOLD_STATUS,
            "hold_changed": False,
            "held_role_payload_accessed_by_projection": False,
        },
    }
    _assert_forbidden_dense_fields_absent(projected)
    projected["projected_payload_sha256"] = canonical_sha256(projected)
    return projected


def validate_projected_row(
    row: Mapping[str, Any], *, role: str, row_index: int
) -> None:
    expected_keys = {
        "audit_identity",
        "authority",
        "loader_request",
        "method_identity",
        "model_input",
        "projected_payload_sha256",
        "schema_version",
        "source_binding",
        "supervision",
    }
    _require_exact_keys(row, expected_keys, f"projected {role}[{row_index}]")
    if row.get("schema_version") != PROJECTION_ROW_SCHEMA:
        raise ProjectionHold("projected row schema drifted")
    payload = dict(row)
    declared = _sha(
        payload.pop("projected_payload_sha256"),
        f"projected {role}[{row_index}].projected_payload_sha256",
    )
    if canonical_sha256(payload) != declared:
        raise ProjectionHold(f"projected {role}[{row_index}] payload hash drifted")
    identity = row.get("audit_identity")
    supervision = row.get("supervision")
    authority = row.get("authority")
    model_input = row.get("model_input")
    source_binding = row.get("source_binding")
    if not all(
        isinstance(value, dict)
        for value in (identity, supervision, authority, model_input, source_binding)
    ):
        raise ProjectionHold(f"projected {role}[{row_index}] nested contract drifted")
    assert isinstance(identity, dict)
    assert isinstance(supervision, dict)
    assert isinstance(authority, dict)
    assert isinstance(model_input, dict)
    assert isinstance(source_binding, dict)
    if identity.get("split_role") != role:
        raise ProjectionHold(f"projected {role}[{row_index}] role drifted")
    state = identity.get("state")
    if state not in STATE_FACTORS:
        raise ProjectionHold(f"projected {role}[{row_index}] state drifted")
    expected_camera, expected_object = STATE_FACTORS[str(state)]
    if not (
        supervision.get("state") == state
        and supervision.get("camera_moving") is expected_camera
        and supervision.get("object_moving") is expected_object
        and supervision.get("canonical_answer") == CANONICAL_ANSWERS[str(state)]
        and supervision.get("dense_numeric_trajectory_target_consumed") is False
        and supervision.get("trajectory_derived_state_label") is True
    ):
        raise ProjectionHold(f"projected {role}[{row_index}] supervision drifted")
    if authority != {
        "operational_data_status": OPERATIONAL_STATUS,
        "formal_data_role_authority_complete": False,
        "formal_data_release": False,
        "formal_training_authorized": False,
        "formal_validation_authorized": False,
        "projection_is_authorization": False,
        "underlying_gate2_status": HOLD_STATUS,
        "hold_changed": False,
        "held_role_payload_accessed_by_projection": False,
    }:
        raise ProjectionHold(f"projected {role}[{row_index}] authority drifted")
    if model_input.get("target_query") != QUERY_TEXT:
        raise ProjectionHold(f"projected {role}[{row_index}] query drifted")
    required_source_binding_keys = {
        "sanitized_manifest_sha256",
        "sanitized_row_index_zero_based",
        "original_operational_manifest_sha256",
        "original_operational_row_index_zero_based",
        "source_record_payload_sha256",
        "source_rgb20_canonical_sha256",
        "source_box_record_canonical_sha256",
        "selector_row_canonical_sha256",
    }
    if set(source_binding) not in (
        required_source_binding_keys,
        required_source_binding_keys | {"source_canonical_line_sha256"},
    ):
        raise ProjectionHold(
            f"projected {role}[{row_index}] source binding keys drifted"
        )
    spec = SOURCE_SPECS[role]
    if not (
        source_binding.get("sanitized_manifest_sha256") == spec.sha256
        and source_binding.get("sanitized_row_index_zero_based") == row_index
        and source_binding.get("original_operational_manifest_sha256")
        == spec.input_manifest_sha256
        and type(source_binding.get("original_operational_row_index_zero_based")) is int
        and 0
        <= source_binding["original_operational_row_index_zero_based"]
        < spec.input_rows
    ):
        raise ProjectionHold(
            f"projected {role}[{row_index}] dual manifest binding drifted"
        )
    for key in required_source_binding_keys - {
        "sanitized_row_index_zero_based",
        "original_operational_row_index_zero_based",
    }:
        _sha(source_binding.get(key), f"projected {role}[{row_index}].{key}")
    if "source_canonical_line_sha256" in source_binding:
        _sha(
            source_binding["source_canonical_line_sha256"],
            f"projected {role}[{row_index}].source_canonical_line_sha256",
        )
    _assert_forbidden_dense_fields_absent(row)


def _write_projected_manifest(
    spec: SourceSpec, output: Path
) -> tuple[dict[str, Any], dict[str, set[Any]]]:
    digest = hashlib.sha256()
    counts_state: Counter[str] = Counter()
    counts_source: Counter[str] = Counter()
    counts_source_state: Counter[tuple[str, str]] = Counter()
    cases: set[str] = set()
    windows: set[str] = set()
    sequences: set[tuple[str, str]] = set()
    rows = 0
    with output.open("wb") as handle:
        for index, source_payload, source_row in _iter_strict_source_rows(spec):
            projected = project_source_row(
                source_row,
                role=spec.role,
                row_index=index,
                source_manifest_sha256=spec.sha256,
            )
            validate_projected_row(projected, role=spec.role, row_index=index)
            projected["source_binding"]["source_canonical_line_sha256"] = (
                hashlib.sha256(source_payload).hexdigest()
            )
            # The binding insertion changes the projected payload identity.
            projected.pop("projected_payload_sha256")
            projected["projected_payload_sha256"] = canonical_sha256(projected)
            validate_projected_row(projected, role=spec.role, row_index=index)
            encoded = canonical_json_line(projected)
            handle.write(encoded)
            digest.update(encoded)
            identity = projected["audit_identity"]
            case_id = str(identity["case_id"])
            if case_id in cases:
                raise ProjectionHold(f"duplicate {spec.role} case_id: {case_id}")
            cases.add(case_id)
            windows.add(str(identity["physical_window_id"]))
            source = str(identity["source_dataset"])
            state = str(identity["state"])
            sequences.add((source, str(identity["sequence_id"])))
            counts_state[state] += 1
            counts_source[source] += 1
            counts_source_state[(source, state)] += 1
            rows += 1
    if rows != spec.rows:
        raise ProjectionHold(f"{spec.role} projected row count drifted")
    stat = output.stat()
    summary = {
        "filename": output.name,
        "rows": rows,
        "bytes": stat.st_size,
        "sha256": digest.hexdigest(),
        "by_state": dict(sorted(counts_state.items())),
        "by_source": dict(sorted(counts_source.items())),
        "by_source_state": {
            f"{source}::{state}": count
            for (source, state), count in sorted(counts_source_state.items())
        },
        "unique_case_ids": len(cases),
        "unique_physical_windows": len(windows),
        "unique_source_sequences": len(sequences),
    }
    memberships = {"cases": cases, "windows": windows, "sequences": sequences}
    return summary, memberships


def _binding(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _read_bound_json(path: Path, expected_sha256: str, label: str) -> Mapping[str, Any]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ProjectionHold(f"{label} path/hash binding drifted")
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=_strict_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProjectionHold(f"{label} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ProjectionHold(f"{label} must contain one JSON object")
    return value


def _verify_quality_ledger(spec: SourceSpec) -> Mapping[str, Any]:
    ledger_path = spec.directory / "SHA256SUMS"
    if sha256_file(ledger_path) != spec.ledger_sha256:
        raise ProjectionHold(f"{spec.role} quality ledger hash drifted")
    entries: dict[str, str] = {}
    for line_index, line in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines()
    ):
        parts = line.split("  ")
        if len(parts) != 2:
            raise ProjectionHold(
                f"{spec.role} quality ledger line {line_index} drifted"
            )
        digest, name = parts
        _sha(digest, f"{spec.role}.quality_ledger[{line_index}]")
        if not name or "/" in name or name in entries:
            raise ProjectionHold(f"{spec.role} quality ledger member drifted")
        member = spec.directory / name
        if not member.is_file() or sha256_file(member) != digest:
            raise ProjectionHold(
                f"{spec.role} quality ledger member hash drifted: {name}"
            )
        entries[name] = digest
    retained_name = spec.path.name
    expected_entries = {
        "AUDIT.json": spec.audit_sha256,
        "FAILURE_PROVENANCE.json": spec.failure_provenance_sha256,
        "quarantine.jsonl": spec.quarantine_sha256,
        "retained_membership.jsonl": spec.retained_membership_sha256,
        retained_name: spec.sha256,
    }
    for name, digest in expected_entries.items():
        if entries.get(name) != digest:
            raise ProjectionHold(
                f"{spec.role} quality ledger lacks exact {name} binding"
            )
    return {
        "path": str(ledger_path),
        "sha256": spec.ledger_sha256,
        "entries_verified": len(entries),
    }


def _sanitization_lineage() -> Mapping[str, Any]:
    if sha256_file(QUALITY_POLICY) != QUALITY_POLICY_SHA256:
        raise ProjectionHold("quality policy drifted")
    posthoc = _read_bound_json(
        PRACTICAL_READY_POSTHOC_SEAL,
        PRACTICAL_READY_POSTHOC_SEAL_SHA256,
        "practical-ready posthoc seal",
    )
    expected_both04 = {
        "case_id": REQUIRED_EXCLUDED_TRAIN_CASE,
        "decision": "QUARANTINE_NO_RELABEL",
        "human_state": "neither",
        "original_state": "both",
        "present_in_retained": False,
    }
    decision = posthoc.get("decision")
    pair_seal = posthoc.get("posthoc_train_validation_cross_package_seal")
    if not (
        posthoc.get("both04") == expected_both04
        and posthoc.get("data", {}).get("labels_rewritten") is False
        and posthoc.get("data", {}).get("retained_rows_rewritten") is False
        and isinstance(decision, Mapping)
        and decision.get("training_authorized") is False
        and decision.get("formal_ready_to_train") is False
        and decision.get("underlying_gate2_status") == HOLD_STATUS
        and isinstance(pair_seal, Mapping)
        and pair_seal.get("status") == "PASS_CURRENT_PAIR_HASH_BOUND"
    ):
        raise ProjectionHold("sanitized pair/posthoc exclusion semantics drifted")

    roles: dict[str, Any] = {}
    for role, spec in SOURCE_SPECS.items():
        audit_path = spec.directory / "AUDIT.json"
        completion_path = spec.directory / "COMPLETION.json"
        membership_path = spec.directory / "retained_membership.jsonl"
        quarantine_path = spec.directory / "quarantine.jsonl"
        failure_path = spec.directory / "FAILURE_PROVENANCE.json"
        audit = _read_bound_json(audit_path, spec.audit_sha256, f"{role} quality audit")
        completion = _read_bound_json(
            completion_path, spec.completion_sha256, f"{role} quality completion"
        )
        counts = audit.get("counts")
        invariants = audit.get("invariants")
        authority = audit.get("authority")
        if not (
            audit.get("role") == role
            and isinstance(counts, Mapping)
            and counts.get("retained") == spec.rows
            and counts.get("quarantined") == spec.quarantined_rows
            and isinstance(invariants, Mapping)
            and invariants.get("labels_rewritten") is False
            and invariants.get("original_manifest_modified") is False
            and invariants.get("retained_rows_are_original_bytes_in_original_order")
            is True
            and invariants.get("partition_closes") is True
            and invariants.get("policy_recalibrated_on_role") is False
            and invariants.get("strict_zero_open_isolation_attestation") is False
            and isinstance(authority, Mapping)
            and authority.get("training_authorized") is False
            and authority.get("formal_release_authorized") is False
            and authority.get("underlying_gate2_status") == HOLD_STATUS
            and audit.get("input", {}).get("sha256_after") == spec.input_manifest_sha256
            and audit.get("outputs", {}).get(spec.path.name) == spec.sha256
            and audit.get("policy", {}).get("sha256") == QUALITY_POLICY_SHA256
        ):
            raise ProjectionHold(f"{role} quality audit semantics drifted")
        if not (
            completion.get("schema_version")
            == "tst_na_v3_rgb20_state_o18_quality_completion_v1"
            and completion.get("publication_complete") is True
            and completion.get("role") == role
            and completion.get("retained_rows") == spec.rows
            and completion.get("quarantined_rows") == spec.quarantined_rows
            and completion.get("input_manifest_sha256") == spec.input_manifest_sha256
            and completion.get("policy_sha256") == QUALITY_POLICY_SHA256
            and completion.get("audit_sha256") == spec.audit_sha256
            and completion.get("sha256sums_sha256") == spec.ledger_sha256
            and completion.get("strict_zero_open_isolation_attestation") is False
            and completion.get("training_authorized") is False
            and completion.get("underlying_gate2_status") == HOLD_STATUS
        ):
            raise ProjectionHold(f"{role} quality completion semantics drifted")
        if pair_seal.get(f"{role}_completion_sha256") != spec.completion_sha256:
            raise ProjectionHold(f"posthoc pair seal lacks exact {role} completion")
        ledger = _verify_quality_ledger(spec)
        roles[role] = {
            "original_operational_manifest": _binding(
                spec.input_path, rows=spec.input_rows
            ),
            "sanitized_retained_manifest": _binding(spec.path, rows=spec.rows),
            "excluded_rows": spec.quarantined_rows,
            "audit": _binding(audit_path),
            "completion": _binding(completion_path),
            "ledger": ledger,
            "retained_membership": _binding(membership_path),
            "quarantine": _binding(quarantine_path),
            "failure_provenance": _binding(failure_path),
            "labels_rewritten": False,
            "retained_rows_are_original_bytes_in_original_order": True,
            "strict_zero_open_isolation_attestation": False,
        }
    return {
        "quality_policy": _binding(QUALITY_POLICY),
        "posthoc_pair_seal": _binding(PRACTICAL_READY_POSTHOC_SEAL),
        "roles": roles,
        "required_exclusion": expected_both04,
        "membership_identity": "sanitized_retained_7959_train_1991_validation",
    }


def _code_bindings() -> dict[str, Any]:
    here = Path(__file__).resolve()
    tests = here.with_name("test_projection.py")
    readme = here.with_name("README.md")
    init = here.with_name("__init__.py")
    bindings = {
        "projection_implementation": _binding(here),
        "projection_tests": _binding(tests),
        "projection_readme": _binding(readme),
        "package_init": _binding(init),
        "formal_rgb20_loader": _binding(FORMAL_RGB20_LOADER),
        "mosder_canonical_language": _binding(MOSDER_LANGUAGE),
    }
    if bindings["formal_rgb20_loader"]["sha256"] != FORMAL_RGB20_LOADER_SHA256:
        raise ProjectionHold("bound formal RGB20 loader drifted")
    if bindings["mosder_canonical_language"]["sha256"] != MOSDER_LANGUAGE_SHA256:
        raise ProjectionHold("bound MoSDeR language module drifted")
    return bindings


def _write_canonical(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_line(value))


def _write_ledger(root: Path) -> None:
    members = sorted(path.name for path in root.iterdir() if path.name != LEDGER_NAME)
    lines = [f"{sha256_file(root / name)}  {name}\n" for name in members]
    (root / LEDGER_NAME).write_text("".join(lines), encoding="utf-8", newline="\n")


def build_bundle(output_root: Path = DEFAULT_OUTPUT_ROOT) -> Mapping[str, Any]:
    """Build one immutable-by-convention projection directory; never overwrite."""

    output_root = output_root.resolve(strict=False)
    if output_root.exists():
        raise ProjectionHold(
            f"refusing to overwrite existing projection root: {output_root}"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp.", dir=output_root.parent)
    )
    try:
        sanitization_lineage = _sanitization_lineage()
        summaries: dict[str, Any] = {}
        memberships: dict[str, dict[str, set[Any]]] = {}
        for role in ALLOWED_ROLES:
            summary, membership = _write_projected_manifest(
                SOURCE_SPECS[role], temporary / OUTPUT_MANIFEST_NAMES[role]
            )
            summaries[role] = summary
            memberships[role] = membership
        isolation = {
            "train_validation_case_id_overlap": len(
                memberships["train"]["cases"] & memberships["validation"]["cases"]
            ),
            "train_validation_physical_window_overlap": len(
                memberships["train"]["windows"] & memberships["validation"]["windows"]
            ),
            "train_validation_source_sequence_overlap": len(
                memberships["train"]["sequences"]
                & memberships["validation"]["sequences"]
            ),
        }
        if any(isolation.values()):
            raise ProjectionHold(f"Train/Validation isolation drifted: {isolation}")
        for marker in MARKERS:
            (temporary / marker).touch(exist_ok=False)
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "status": STATUS,
            "method_identity": {
                "public_name": METHOD_PUBLIC_NAME,
                "architecture_version": METHOD_ARCHITECTURE_VERSION,
                "implementation_variant": IMPLEMENTATION_VARIANT,
            },
            "scope": {
                "roles_read": list(ALLOWED_ROLES),
                "held_roles_read": [],
                "held_role_payload_accessed": False,
                "model_loaded": False,
                "gpu_used": False,
                "training_started": False,
                "validation_scoring_started": False,
                "projection_is_authorization": False,
            },
            "authority": {
                "strict_data_operational_abi": OPERATIONAL_STATUS,
                "underlying_gate2_status": HOLD_STATUS,
                "hold_changed": False,
                "formal_data_role_authority_complete": False,
                "formal_data_release": False,
                "formal_training_authorized": False,
                "formal_validation_authorized": False,
                "ready_to_train": False,
            },
            "source_manifests": {
                role: _binding(spec.path, rows=spec.rows)
                for role, spec in SOURCE_SPECS.items()
            },
            "sanitization_lineage": sanitization_lineage,
            "outputs": summaries,
            "isolation": isolation,
            "consumed_field_contract": {
                "model_inputs": [
                    "20 chronological real source-native RGB frames",
                    "timestamps for deterministic sampling/order validation only",
                    "fixed target query",
                    "oracle target box track [20,4] resolved from bound three-anchor ABI",
                ],
                "supervision": [
                    "camera_moving boolean",
                    "object_moving boolean",
                    "four-state label",
                    "package-owned canonical four-line answer",
                ],
                "not_consumed": [
                    "Camera18",
                    "Object18",
                    "numeric trajectory values or masks",
                    "trajectory anchors",
                    "dataset/source/identity metadata as learned model input",
                    "ground-truth state or answer as model input",
                ],
                "trajectory_derived_state_supervision_remains": True,
                "box_to_mask_contract": BOX_TO_MASK_CONTRACT,
                "query_text": QUERY_TEXT,
                "query_sha256": hashlib.sha256(QUERY_TEXT.encode("utf-8")).hexdigest(),
                "canonical_answers_sha256": canonical_sha256(CANONICAL_ANSWERS),
                "formal_prompt_protocol_sealed_by_this_artifact": False,
            },
            "code_bindings": _code_bindings(),
            "negative_authority": {
                "numeric_camera18_object18_heads_present": False,
                "numeric_trajectory_output_claimed": False,
                "future_forecasting_claimed": False,
                "all_trajectory_derived_supervision_removed": False,
                "formal_runner_complete": False,
                "formal_protocol_and_seed_seal_complete": False,
            },
        }
        _write_canonical(temporary / RECEIPT_NAME, receipt)
        _write_ledger(temporary)
        temporary.rename(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return verify_bundle(output_root, replay_sources=True)


def _parse_ledger(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        parts = line.split("  ")
        if len(parts) != 2:
            raise ProjectionHold(f"ledger line {index} is malformed")
        digest, name = parts
        _sha(digest, f"ledger[{index}].sha256")
        if not name or "/" in name or name in output:
            raise ProjectionHold(f"ledger line {index} filename is invalid")
        output[name] = digest
    return output


def verify_bundle(
    output_root: Path = DEFAULT_OUTPUT_ROOT, *, replay_sources: bool = True
) -> Mapping[str, Any]:
    """Verify exact tree/ledger and optionally replay every source projection."""

    root = output_root.resolve(strict=True)
    expected_members = (
        set(OUTPUT_MANIFEST_NAMES.values()) | set(MARKERS) | {RECEIPT_NAME, LEDGER_NAME}
    )
    observed = {path.name for path in root.iterdir()}
    if observed != expected_members or any(
        not path.is_file() for path in root.iterdir()
    ):
        raise ProjectionHold("projection output tree is not exactly closed")
    ledger = _parse_ledger(root / LEDGER_NAME)
    if set(ledger) != expected_members - {LEDGER_NAME}:
        raise ProjectionHold("projection ledger membership drifted")
    for name, digest in ledger.items():
        if sha256_file(root / name) != digest:
            raise ProjectionHold(f"projection ledger hash drifted: {name}")
    receipt_payload = (root / RECEIPT_NAME).read_bytes()
    if not receipt_payload.endswith(b"\n"):
        raise ProjectionHold("projection receipt lacks terminal LF")
    receipt = strict_json_bytes(receipt_payload[:-1], "projection receipt")
    if not (
        receipt.get("schema_version") == RECEIPT_SCHEMA
        and receipt.get("status") == STATUS
        and receipt.get("authority", {}).get("underlying_gate2_status") == HOLD_STATUS
        and receipt.get("authority", {}).get("hold_changed") is False
        and receipt.get("authority", {}).get("formal_training_authorized") is False
        and receipt.get("scope", {}).get("held_roles_read") == []
        and receipt.get("scope", {}).get("held_role_payload_accessed") is False
    ):
        raise ProjectionHold("projection receipt identity/authority drifted")
    if receipt.get("code_bindings") != _code_bindings():
        raise ProjectionHold("projection code bindings drifted")
    if receipt.get("sanitization_lineage") != _sanitization_lineage():
        raise ProjectionHold("projection sanitization lineage drifted")

    source_iters: dict[str, Iterator[tuple[int, bytes, Mapping[str, Any]]]] = {}
    if replay_sources:
        source_iters = {
            role: iter(_iter_strict_source_rows(spec))
            for role, spec in SOURCE_SPECS.items()
        }
    observed_memberships: dict[str, dict[str, set[Any]]] = {}
    for role in ALLOWED_ROLES:
        output = root / OUTPUT_MANIFEST_NAMES[role]
        summary = receipt.get("outputs", {}).get(role)
        if not isinstance(summary, dict):
            raise ProjectionHold(f"receipt lacks {role} output summary")
        if not (
            summary.get("filename") == output.name
            and summary.get("sha256") == sha256_file(output)
            and summary.get("bytes") == output.stat().st_size
        ):
            raise ProjectionHold(f"{role} output binding drifted")
        cases: set[str] = set()
        windows: set[str] = set()
        sequences: set[tuple[str, str]] = set()
        rows = 0
        with output.open("rb") as handle:
            for index, raw in enumerate(handle):
                if not raw.endswith(b"\n") or raw == b"\n":
                    raise ProjectionHold(
                        f"projected {role}[{index}] line framing drifted"
                    )
                projected = strict_json_bytes(raw[:-1], f"projected {role}[{index}]")
                validate_projected_row(projected, role=role, row_index=index)
                if replay_sources:
                    try:
                        source_index, source_payload, source_row = next(
                            source_iters[role]
                        )
                    except StopIteration as error:
                        raise ProjectionHold(
                            f"{role} projection has extra rows"
                        ) from error
                    if source_index != index:
                        raise ProjectionHold(f"{role} source cursor drifted")
                    expected = project_source_row(
                        source_row,
                        role=role,
                        row_index=index,
                        source_manifest_sha256=SOURCE_SPECS[role].sha256,
                    )
                    expected["source_binding"]["source_canonical_line_sha256"] = (
                        hashlib.sha256(source_payload).hexdigest()
                    )
                    expected.pop("projected_payload_sha256")
                    expected["projected_payload_sha256"] = canonical_sha256(expected)
                    if canonical_json_bytes(projected) != canonical_json_bytes(
                        expected
                    ):
                        raise ProjectionHold(
                            f"projected {role}[{index}] differs from source replay"
                        )
                identity = projected["audit_identity"]
                case_id = str(identity["case_id"])
                if case_id in cases:
                    raise ProjectionHold(f"duplicate projected {role} case_id")
                cases.add(case_id)
                windows.add(str(identity["physical_window_id"]))
                sequences.add(
                    (str(identity["source_dataset"]), str(identity["sequence_id"]))
                )
                rows += 1
        if rows != summary.get("rows") or rows != SOURCE_SPECS[role].rows:
            raise ProjectionHold(f"projected {role} row count drifted")
        if replay_sources:
            try:
                next(source_iters[role])
            except StopIteration:
                pass
            else:
                raise ProjectionHold(f"{role} source has unprojected rows")
        observed_memberships[role] = {
            "cases": cases,
            "windows": windows,
            "sequences": sequences,
        }
    observed_isolation = {
        "train_validation_case_id_overlap": len(
            observed_memberships["train"]["cases"]
            & observed_memberships["validation"]["cases"]
        ),
        "train_validation_physical_window_overlap": len(
            observed_memberships["train"]["windows"]
            & observed_memberships["validation"]["windows"]
        ),
        "train_validation_source_sequence_overlap": len(
            observed_memberships["train"]["sequences"]
            & observed_memberships["validation"]["sequences"]
        ),
    }
    if observed_isolation != receipt.get("isolation") or any(
        observed_isolation.values()
    ):
        raise ProjectionHold("projection Train/Validation isolation drifted")
    return receipt


def audit_source_manifests() -> Mapping[str, Any]:
    """Read-only full source audit without materializing a projection."""

    output: dict[str, Any] = {}
    memberships: dict[str, dict[str, set[Any]]] = {}
    for role, spec in SOURCE_SPECS.items():
        counts: Counter[tuple[str, str]] = Counter()
        cases: set[str] = set()
        windows: set[str] = set()
        sequences: set[tuple[str, str]] = set()
        rows = 0
        for index, _, row in _iter_strict_source_rows(spec):
            projected = project_source_row(
                row, role=role, row_index=index, source_manifest_sha256=spec.sha256
            )
            validate_projected_row(projected, role=role, row_index=index)
            identity = projected["audit_identity"]
            case_id = str(identity["case_id"])
            if case_id in cases:
                raise ProjectionHold(f"duplicate source {role} case_id")
            cases.add(case_id)
            windows.add(str(identity["physical_window_id"]))
            source = str(identity["source_dataset"])
            state = str(identity["state"])
            sequences.add((source, str(identity["sequence_id"])))
            counts[(source, state)] += 1
            rows += 1
        output[role] = {
            "rows": rows,
            "sha256": spec.sha256,
            "source_state": {
                f"{source}::{state}": count
                for (source, state), count in sorted(counts.items())
            },
        }
        memberships[role] = {"cases": cases, "windows": windows, "sequences": sequences}
    output["isolation"] = {
        "case_id_overlap": len(
            memberships["train"]["cases"] & memberships["validation"]["cases"]
        ),
        "physical_window_overlap": len(
            memberships["train"]["windows"] & memberships["validation"]["windows"]
        ),
        "source_sequence_overlap": len(
            memberships["train"]["sequences"] & memberships["validation"]["sequences"]
        ),
    }
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build", help="build the nonauthorizing projection bundle"
    )
    build.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    build.add_argument("--quiet", action="store_true")
    verify = subparsers.add_parser(
        "verify", help="verify an existing projection bundle"
    )
    verify.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    verify.add_argument("--no-source-replay", action="store_true")
    verify.add_argument("--quiet", action="store_true")
    subparsers.add_parser(
        "audit-sources", help="read-only audit of the two exact source manifests"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        result = build_bundle(args.output_root)
    elif args.command == "verify":
        result = verify_bundle(
            args.output_root, replay_sources=not args.no_source_replay
        )
    else:
        result = audit_source_manifests()
    if getattr(args, "quiet", False):
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "train_rows": result["outputs"]["train"]["rows"],
                    "validation_rows": result["outputs"]["validation"]["rows"],
                    "output_root": str(args.output_root.resolve()),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
