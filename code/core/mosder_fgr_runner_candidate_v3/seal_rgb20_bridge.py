#!/root/miniconda3/bin/python
# LOCAL_PATH: Direct execution requires the Python interpreter in this shebang.
"""Record RGB loader metadata and file hashes.

Metadata is resolved for all training and validation records. Pixel checks
cover three fixed training examples: ADT, HOT3D, and TACO."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Final, Mapping

import numpy as np

RUNNER_ROOT = Path(__file__).resolve().parent
if str(RUNNER_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNNER_ROOT))

from strict_v3_rgb20_bridge import (
    HOLD_STATUS,
    RoleScopedStrictV3RGB20Loader,
    _canonical_json_bytes,
    _canonical_json_line,
    _sha256_file,
    bridge_identity,
    dependency_binding,
    expected_metadata_closure,
)


SCHEMA_VERSION: Final[str] = "mosder_strict_v3_rgb20_bridge_seal_receipt_v2"
STATUS: Final[str] = (
    "PASS_MOSDER_STRICT_V3_RGB20_BRIDGE_SEAL_V2_SANITIZED_7959_1991_"
    "NONAUTHORIZING_HOLD_GATE2_V3"
)
# LOCAL_PATH: RGB decoding interpreter; keep aligned with the shebang and data.py.
BOUND_INTERPRETER: Final[Path] = Path("/root/miniconda3/bin/python")
# LOCAL_PATH: RGB20 loader script from the external source checkout.
FORMAL_LOADER: Final[Path] = Path(
    "/root/story2_camera_object_motion/scripts/formal_rgb20_input_loader_v1.py"
)
FORMAL_LOADER_SHA256: Final[str] = (
    "9cd729cb9df992f4c46dd00a1cc50a06b57a5d1b76fe893d3caa4e07b78fa212"
)
# LOCAL_PATH: External pixel policy shared with the RGB20 decoding worker.
PIXEL_POLICY: Final[Path] = Path(
    "/root/autodl-tmp/tst_plugin_formal_preconfirmation_v2/"
    "GATE6_POLICY_FREEZES_V6/"
    "gate6_pixel_v4.m5_da7c4431d873e802f02099d6a168ac47ae55d4c5c91d8a50c4f672a96c4ddd68.frozen.json"
)
PIXEL_POLICY_SHA256: Final[str] = (
    "73cdda361c20fca300e071719694e32f1108a369f165f45079d833699be0370c"
)
# LOCAL_PATH: Receipt destination; runner.BRIDGE_SEAL_RECEIPT must point here.
OUTPUT_ROOT: Final[Path] = Path(
    "/root/autodl-tmp/tst_native_adapter_o18_sandbox_v1/"
    "MOSDER_STRICT_V3_RGB20_BRIDGE_SEAL_V2_SANITIZED_NONAUTHORIZING_HOLD_GATE2_V3"
)
RECEIPT_NAME: Final[str] = "MOSDER_STRICT_V3_RGB20_BRIDGE_SEAL_RECEIPT_V2.json"
REPRESENTATIVE_CASES: Final[tuple[Mapping[str, Any], ...]] = (
    {
        "role": "train",
        "source_dataset": "ADT-LiteOffice",
        "case_id": "adtq_501631747f489cefa575a6d7",
        "sanitized_row_index_zero_based": 6150,
        "original_operational_row_index_zero_based": 6191,
        "selection_reason": "source_coverage_and_predecessor_index_missing",
        "predecessor_index_must_be_absent": True,
    },
    {
        "role": "train",
        "source_dataset": "HOT3D",
        "case_id": "hot3d_train_002490_w000045_n060_target_18",
        "sanitized_row_index_zero_based": 21,
        "original_operational_row_index_zero_based": 21,
        "selection_reason": "source_coverage",
        "predecessor_index_must_be_absent": False,
    },
    {
        "role": "train",
        "source_dataset": "TACO-V1-allocentric",
        "case_id": (
            "TACO-ALLOC-22070938:(dust, roller, pan)/20231019_140:000015:target_084"
        ),
        "sanitized_row_index_zero_based": 2343,
        "original_operational_row_index_zero_based": 2343,
        "selection_reason": "source_coverage_and_predecessor_index_missing",
        "predecessor_index_must_be_absent": True,
    },
)


class BridgeSealError(RuntimeError):
    """The independent bridge seal cannot be built exactly."""


def _file_binding(path: Path, expected_sha256: str | None = None) -> Mapping[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise BridgeSealError(f"dependency is not a regular file: {resolved}")
    digest = _sha256_file(resolved)
    if expected_sha256 is not None and digest != expected_sha256:
        raise BridgeSealError(f"dependency SHA256 drifted: {resolved}")
    return {
        "path": str(path),
        "realpath": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": digest,
    }


def _validate_box_track(
    boxes: Any, width: int, height: int
) -> tuple[list[list[float]], str]:
    if not isinstance(boxes, list) or len(boxes) != 20:
        raise BridgeSealError("representative interpolation did not return 20 boxes")
    normalized: list[list[float]] = []
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4:
            raise BridgeSealError("representative box shape differs")
        values = [float(value) for value in box]
        x1, y1, x2, y2 = values
        if not (
            bool(np.isfinite(values).all())
            and 0.0 <= x1 < x2 <= float(width)
            and 0.0 <= y1 < y2 <= float(height)
        ):
            raise BridgeSealError("representative box is nonfinite or out of bounds")
        normalized.append(values)
    return normalized, hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest()


def _representative_pixel_smoke(
    loader: RoleScopedStrictV3RGB20Loader,
) -> tuple[list[Mapping[str, Any]], Mapping[str, bool]]:
    old_data_side = loader.formal_module.verify_data_side_snapshot()
    try:
        old_train_index = old_data_side.get("indexes", {}).get("train")
        if not isinstance(old_train_index, Mapping):
            raise BridgeSealError("predecessor Train membership index is unavailable")
        predecessor_absence = {
            str(specification["case_id"]): (
                (str(specification["source_dataset"]), str(specification["case_id"]))
                not in old_train_index
            )
            for specification in REPRESENTATIVE_CASES
            if specification["predecessor_index_must_be_absent"] is True
        }
    finally:
        del old_data_side
        gc.collect()
    if not all(predecessor_absence.values()):
        raise BridgeSealError(
            "a fixed predecessor-index-missing representative is no longer absent"
        )

    receipts: list[Mapping[str, Any]] = []
    for specification in REPRESENTATIVE_CASES:
        role = str(specification["role"])
        source = str(specification["source_dataset"])
        case_id = str(specification["case_id"])
        identity = (source, role, case_id)
        if identity not in loader.rows:
            raise BridgeSealError("representative escaped role-complete membership")
        binding = loader.source_bindings[identity]
        if (
            binding.get("original_operational_row_index_zero_based")
            != specification["original_operational_row_index_zero_based"]
            or binding.get("sanitized_row_index_zero_based")
            != specification["sanitized_row_index_zero_based"]
        ):
            raise BridgeSealError("representative dual row index binding drifted")
        loaded = loader.load_case(role, source, case_id, binding)
        try:
            frames = tuple(np.asarray(frame) for frame in loaded.frames)
            if not (
                len(frames) == 20
                and all(
                    frame.dtype == np.uint8
                    and frame.ndim == 3
                    and frame.shape[-1] == 3
                    and frame.shape == frames[0].shape
                    for frame in frames
                )
            ):
                raise BridgeSealError("representative RGB20 shape/dtype differs")
            rgb_digest = hashlib.sha256()
            for frame in frames:
                rgb_digest.update(frame)
            timestamps_raw = loaded.interpolation.get("frame_timestamps")
            if not (
                isinstance(timestamps_raw, list)
                and len(timestamps_raw) == 20
                and all(type(value) is int for value in timestamps_raw)
                and all(
                    right > left
                    for left, right in zip(
                        timestamps_raw, timestamps_raw[1:], strict=False
                    )
                )
            ):
                raise BridgeSealError(
                    "representative timestamps are not a strictly ordered RGB20"
                )
            timestamps = list(timestamps_raw)
            height, width, _ = frames[0].shape
            boxes, box_sha256 = _validate_box_track(
                loaded.interpolation.get("interpolated_boxes_xyxy_half_open_float"),
                width,
                height,
            )
            receipts.append(
                {
                    **dict(specification),
                    "predecessor_membership_index_absent": (
                        predecessor_absence.get(case_id)
                        if specification["predecessor_index_must_be_absent"] is True
                        else None
                    ),
                    "source_binding": dict(binding),
                    "rgb20": {
                        "shape": [20, height, width, 3],
                        "dtype": "uint8",
                        "sha256": rgb_digest.hexdigest(),
                        "all_frames_same_shape": True,
                    },
                    "timestamps": {
                        "count": 20,
                        "strictly_increasing": True,
                        "first": timestamps[0],
                        "last": timestamps[-1],
                        "canonical_sha256": hashlib.sha256(
                            _canonical_json_bytes(timestamps)
                        ).hexdigest(),
                    },
                    "oracle_box_track": {
                        "shape": [20, 4],
                        "all_finite": True,
                        "all_half_open_boxes_in_source_native_bounds": True,
                        "canonical_sha256": box_sha256,
                    },
                    "interpolation_contract_executed": True,
                    "pixel_smoke_pass": True,
                }
            )
            del boxes, frames
        finally:
            loaded.close()
            del loaded
            gc.collect()
    return receipts, predecessor_absence


def build_receipt(
    output_root: Path = OUTPUT_ROOT,
) -> tuple[Path, str, Mapping[str, Any]]:
    expected_interpreter = BOUND_INTERPRETER.resolve(strict=True)
    observed_interpreter = Path(sys.executable).resolve(strict=True)
    if observed_interpreter != expected_interpreter:
        raise BridgeSealError("bridge seal must run under /root/miniconda3/bin/python")

    train_loader = RoleScopedStrictV3RGB20Loader("train")
    try:
        train_closure = dict(train_loader.metadata_closure)
        if train_closure != dict(expected_metadata_closure("train")):
            raise BridgeSealError("Train metadata closure receipt differs")
        pixel_smoke, predecessor_absence = _representative_pixel_smoke(train_loader)
        formal_module_path = Path(train_loader.formal_module.__file__).resolve()
        pixel_policy_path = Path(
            train_loader.formal_module.FROZEN_PIXEL_POLICY
        ).resolve()
    finally:
        del train_loader
        gc.collect()

    validation_loader = RoleScopedStrictV3RGB20Loader("validation")
    try:
        validation_closure = dict(validation_loader.metadata_closure)
        if validation_closure != dict(expected_metadata_closure("validation")):
            raise BridgeSealError("Validation metadata closure receipt differs")
        if (
            Path(validation_loader.formal_module.__file__).resolve()
            != formal_module_path
        ):
            raise BridgeSealError("role-scoped formal loader modules differ")
    finally:
        del validation_loader
        gc.collect()

    import av
    from PIL import Image

    runner_root = Path(__file__).resolve().parent
    dependencies = {
        "seal_builder": _file_binding(Path(__file__)),
        "bridge_source": _file_binding(runner_root / "strict_v3_rgb20_bridge.py"),
        "ipc_worker": _file_binding(runner_root / "rgb20_loader_worker.py"),
        "parent_dataset_bridge": _file_binding(runner_root / "data.py"),
        "bound_interpreter": _file_binding(BOUND_INTERPRETER),
        "formal_rgb20_loader": _file_binding(formal_module_path, FORMAL_LOADER_SHA256),
        "frozen_pixel_policy": _file_binding(pixel_policy_path, PIXEL_POLICY_SHA256),
        "reader_modules": {
            "pyav": _file_binding(Path(av.__file__).resolve()),
            "pillow_image": _file_binding(Path(Image.__file__).resolve()),
        },
        "strict_v3_resolver_dependencies": dict(dependency_binding()),
    }
    if (
        formal_module_path != FORMAL_LOADER.resolve()
        or pixel_policy_path != PIXEL_POLICY.resolve()
    ):
        raise BridgeSealError("formal loader/pixel policy realpath differs")

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "method_identity": {
            "public_name": "MoSDeR",
            "architecture_version": "MoSDeR-v1",
            "implementation_variant": "headless_explicit_residual",
        },
        "bridge_identity": dict(bridge_identity()),
        "role_metadata_closure": {
            "train": train_closure,
            "validation": validation_closure,
        },
        "representative_pixel_smoke": {
            "status": "PASS_THREE_SOURCE_REPRESENTATIVE_PIXEL_SMOKE",
            "cases": pixel_smoke,
            "sources_covered": [
                "ADT-LiteOffice",
                "HOT3D",
                "TACO-V1-allocentric",
            ],
            "predecessor_index_missing_cases_verified": predecessor_absence,
            "pixel_windows_decoded_during_seal": 3,
        },
        "coverage_limitations": {
            "metadata_case_rows_resolved_and_bound": 9950,
            "metadata_unique_physical_windows_bound": 6167,
            "pixel_windows_exhaustively_decoded": False,
            "pixel_windows_decoded_during_seal": 3,
            "pixel_windows_not_decoded_during_seal": 6164,
            "claim": (
                "Representative source/path smoke only; this seal does not prove "
                "physical pixel decode success for all 6,167 unique windows."
            ),
        },
        "dependency_tree": dependencies,
        "dependency_tree_sha256": hashlib.sha256(
            _canonical_json_bytes(dependencies)
        ).hexdigest(),
        "scope": {
            "roles_opened": ["train", "validation"],
            "held_roles_opened": [],
            "held_role_payload_accessed": False,
            "model_loaded": False,
            "training_started": False,
            "validation_scoring_started": False,
            "pixel_windows_decoded_at_worker_startup": 0,
            "representative_pixel_smoke_executed": True,
        },
        "authority": {
            "seal_is_training_authorization": False,
            "formal_training_authorized": False,
            "formal_validation_authorized": False,
            "ready_to_train": False,
            "underlying_gate2_status": HOLD_STATUS,
            "hold_changed": False,
        },
    }
    receipt["receipt_payload_sha256"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    encoded = _canonical_json_line(receipt)
    output_root.mkdir(parents=True, exist_ok=True)
    receipt_path = output_root / RECEIPT_NAME
    try:
        descriptor = os.open(
            receipt_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
    except FileExistsError:
        if receipt_path.read_bytes() != encoded:
            raise BridgeSealError("existing bridge seal differs; refusing overwrite")
    else:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    return receipt_path, _sha256_file(receipt_path), receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    arguments = parser.parse_args()
    path, digest, receipt = build_receipt(arguments.output_root)
    print(
        _canonical_json_bytes(
            {
                "path": str(path),
                "sha256": digest,
                "status": receipt["status"],
            }
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
