"""Materialize V2 from immutable V1 records and source-native physical test reserves."""

import collections
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

# LOCAL_PATH: parents[1] resolves to code/ here; set the benchmark data root
# before using the release/, candidates/, or raw/ subdirectories.
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
# LOCAL_PATH: Requires this sibling data directory and its scripts/ modules.
PREVIOUS_RELEASE_ROOT = BENCHMARK_ROOT.with_name("general_motion_benchmark_v1")
sys.path.insert(0, str(PREVIOUS_RELEASE_ROOT / "scripts"))
from build_release import (
    sha as file_sha256,
)
from build_release import (
    standardize as standardize_record,
)
from build_release import (
    summarize as summarize_split,
)
from build_release import (
    weights as sampling_distribution,
)
from build_release import (
    write as write_json,
)
from build_release import (
    write_rows as write_jsonl,
)
from export_old_world_poses import at_times, series, unpack_pose
from select_release import read as read_jsonl

sys.path.remove(str(PREVIOUS_RELEASE_ROOT / "scripts"))
source_registry = {}
pixel_cache = {}
archive_handles = {}
media_receipts = {}


def source_reference(path):
    """Deduplicate source paths and record their immutable content hashes."""
    path = Path(path)
    key = str(path)
    if key not in source_registry:
        source_registry[key] = {
            "path": key,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return source_registry[key]


def build_native_record(row, adt_index, adt_pixel_hashes):
    """Export ADT or HOT3D inputs, physical trajectories, and source provenance."""
    source = row["source"]
    timestamps = row["timestamps_ns"]
    state = row["state"]
    record = {
        k: row[k]
        for k in ["case_id", "source", "state", "sequence", "physical_window_id"]
    }
    if source == "ADT-LiteOffice":
        sequence_root = Path(row["sequence_path"])
        object_uid = str(row["object_uid"])
        camera_pose_path = sequence_root / "aria_trajectory.csv"
        object_pose_path = sequence_root / "scene_objects.csv"
        vrs = row["raw_vrs"]
        device_positions, device_rotations, _ = at_times(
            series(str(camera_pose_path), "camera"), timestamps
        )
        object_positions, object_rotations, alignment = at_times(
            series(str(object_pose_path), "object", object_uid), timestamps
        )
        camera_extrinsic = np.array(adt_index[vrs]["device_from_camera"])
        camera_positions = (
            device_rotations.apply(np.repeat(camera_extrinsic[None, :3, 3], 20, axis=0))
            + device_positions
        )
        camera_rotations = device_rotations * Rotation.from_matrix(
            camera_extrinsic[:3, :3]
        )
        frames = [adt_pixel_hashes[vrs][str(t)] for t in timestamps]
        grid = [1408, 1408]
        record["split_group"] = "ADT-LiteOffice:" + row["sequence"]
        record["target_id"] = row["sequence"] + ":" + object_uid
        instance = json.loads((sequence_root / "instances.json").read_text())[
            object_uid
        ]
        assert instance["instance_type"] == "object" and instance["rigidity"] == "rigid"
        record["target_qualification"] = {
            "instance_type": "object",
            "rigidity": "rigid",
            "category": instance["category"],
            "source": source_reference(sequence_root / "instances.json"),
        }
        record["physical_annotation"] = {
            "camera_pose_source": source_reference(camera_pose_path),
            "object_world_pose_source": source_reference(object_pose_path),
            "device_from_camera": camera_extrinsic.tolist(),
            "calibration_source": source_reference(
                BENCHMARK_ROOT / "candidates/adt_native_index.json"
            ),
            "rgb_source": source_reference(vrs),
            "bbox_source": source_reference(sequence_root / "2d_bounding_box.csv"),
        }
        # LOCAL_PATH: Source-specific physical label rules, hashed into provenance.
        rule = Path(
            "/root/story2_camera_object_motion/configs/adt_confirmation_a_pose_scan_v1.json"
        )
        record["label_derivation"] = {
            "source_rules": source_reference(rule),
            "method": "full native pose window extent; original ADT thresholds and gray rejection",
            "window_start_ns": row["window_start_ns"],
            "window_end_ns": row["window_end_ns"],
            "metrics": {
                k: v
                for k, v in row.items()
                if ("extent" in k or "pose_" in k or "explicit_static" in k)
            },
            "camera_label_reference": "world device trajectory, matching legacy source rule; exported RGB camera pose additionally includes sensor extrinsic",
        }
        record["bbox_alignment"] = {
            "method": "nearest native annotation, maximum 0.5 ms",
            "annotation_timestamps_ns": row["bbox_annotation_timestamps_ns"],
            "max_absolute_offset_ns": max(
                abs(frame_timestamp - annotation_timestamp)
                for frame_timestamp, annotation_timestamp in zip(
                    timestamps, row["bbox_annotation_timestamps_ns"]
                )
            ),
        }
        annotation_paths = [str(camera_pose_path), str(object_pose_path)]
        coordinate_frame = "ADT source world; calibrated RGB camera pose"
    else:
        path = row["raw_tar"]
        if path not in archive_handles:
            archive_handles[path] = tarfile.open(path)
        archive = archive_handles[path]
        frames = []
        grid = row["grid_wh"]
        for i, t in zip(row["native_ordinals"], timestamps):
            member = f"{i:06d}.image_214-1.jpg"
            key = (path, member)
            if key not in pixel_cache:
                with Image.open(
                    io.BytesIO(archive.extractfile(member).read())
                ) as image:
                    rgb_array = np.asarray(image.convert("RGB"))
                assert list(rgb_array.shape) == [grid[1], grid[0], 3]
                pixel_cache[key] = {
                    "locator": path + "#" + member,
                    "timestamp_ns": t,
                    "native_ordinal": i,
                    "width": grid[0],
                    "height": grid[1],
                    "pixel_rgb_sha256": hashlib.sha256(rgb_array.tobytes()).hexdigest(),
                }
            assert pixel_cache[key]["timestamp_ns"] == t
            frames.append(pixel_cache[key])
        camera_positions, camera_quaternions = zip(
            *(unpack_pose(p) for p in row["camera_poses"])
        )
        object_positions, object_quaternions = zip(
            *(unpack_pose(p) for p in row["object_poses"])
        )
        camera_positions = np.array(camera_positions)
        object_positions = np.array(object_positions)
        camera_rotations = Rotation.from_quat(camera_quaternions)
        object_rotations = Rotation.from_quat(object_quaternions)
        alignment = "exact source RGB timestamp pose"
        record["split_group"] = row["split_group"].removeprefix("HOT3D_participant:")
        record["target_id"] = row["target_id"]
        record["target_qualification"] = {
            "rigid_object_pose_annotation": True,
            "human_animal_targets_excluded": True,
        }
        record["physical_annotation"] = {
            "camera_pose_source": source_reference(path),
            "object_world_pose_source": source_reference(path),
            "native_ordinals": row["native_ordinals"],
        }
        # LOCAL_PATH: Source-specific physical label rules, hashed into provenance.
        rule = Path(
            "/root/story2_camera_object_motion/configs/hot3d_motion_scan_validation_role_v2.json"
        )
        record["label_derivation"] = {
            "source_rules": source_reference(rule),
            "method": "full native window path / actual duration, original HOT3D thresholds and gray rejection",
            "full_window_ordinals": row["full_window_ordinals"],
            "metrics": row["motion_metrics"],
            "quality": row["quality"],
            "authorization": "User authorized reuse of old test reserves for new benchmark; numerical rules reused, historical role gate not changed.",
        }
        annotation_paths = [path]
        coordinate_frame = "HOT3D source world"
    record["model_input"] = {
        "native_rgb20_v2": True,
        "rgb20": {"frames": frames},
        "target_boxes_xyxy": row["boxes_xyxy"],
        "grid_wh": grid,
    }
    record["trajectory_gt"] = {
        "timestamps_ns": timestamps,
        "coordinate_frame": coordinate_frame,
        "camera_position_world_m": camera_positions.tolist(),
        "camera_orientation_world_quat_xyzw": camera_rotations.as_quat().tolist(),
        "object_position_world_m": object_positions.tolist(),
        "object_orientation_world_quat_xyzw": object_rotations.as_quat().tolist(),
        "alignment": alignment,
        "raw_annotation_paths": annotation_paths,
        "dense_orientation_training_certified": False,
        "label_policy": "Derived from full native physical window; do not rethreshold only these 20 subsamples.",
    }
    record["supervision"] = {
        "camera_moving": state in ["camera_only", "both"],
        "object_moving": state in ["object_only", "both"],
        "state": state,
        "origin": "physical_trajectory_derived_cached_label",
        "gt_derivation_location": "dataset_preprocessing",
        "model_forward_must_not_receive_gt": True,
    }
    record.update(
        domain="human_egocentric",
        schema_version="general_motion_physical_rgb20_v2",
        historical_exposure={
            "original_role": "testA_raw_reserve",
            "opened_for_current_user_authorized_construction": True,
            "prior_model_exposure_not_independently_certified": True,
        },
        fresh_independent_test=False,
        selection_used_model_output=False,
    )
    return record


def main():
    for name in ["media_receipt", "extra_media_receipt"]:
        for row in json.loads(
            (PREVIOUS_RELEASE_ROOT / f"av2_rgb20/{name}.json").read_text()
        )["files"]:
            media_receipts[row["path"]] = row

    adt_index = json.loads(
        (BENCHMARK_ROOT / "candidates/adt_native_index.json").read_text()
    )
    adt_pixel_hashes = json.loads(
        (BENCHMARK_ROOT / "candidates/adt_pixel_hashes.json").read_text()
    )
    assert (
        json.loads((BENCHMARK_ROOT / "candidates/adt_pixel_receipt.json").read_text())[
            "status"
        ]
        == "PASS"
    )
    release_dir = BENCHMARK_ROOT / "release"
    release_dir.mkdir(exist_ok=True)
    splits = {}
    for role in ["train", "val", "test"]:
        result = []
        for row in read_jsonl(BENCHMARK_ROOT / f"candidates/{role}_selected.jsonl"):
            if "raw_vrs" in row or "raw_tar" in row:
                row = build_native_record(row, adt_index, adt_pixel_hashes)
            elif "trajectory_gt" not in row:
                row = standardize_record(row, {}, media_receipts, {})
            row["previous_role"] = row.get("role")
            row["role"] = role
            row["benchmark_version"] = "general_motion_benchmark_v2"
            row["release_exposure"] = {
                "intended_use": role,
                "requires_retraining_with_new_split": role != "train",
                "test_origin": (
                    "old_testA_raw_reserve"
                    if row["case_id"].startswith("v2:")
                    else "historical_training_candidate_group_holdout"
                )
                if role == "test"
                else None,
                "fresh_independent_test_certified": False,
            }
            result.append(row)
        assert len(result) == (5000 if role == "train" else 1200)
        splits[role] = result
        write_jsonl(release_dir / f"{role}.jsonl", result)
        print(role, len(result), "built", flush=True)
    sampling_weights, policy = sampling_distribution(splits["train"])
    write_jsonl(release_dir / "train_sampling_weights.jsonl", sampling_weights)
    write_json(release_dir / "sampling_policy.json", policy)
    write_json(release_dir / "native_source_registry.json", source_registry)
    write_json(
        release_dir / "native_pixel_receipt.json",
        {
            "status": "PASS",
            "ADT": json.loads(
                (BENCHMARK_ROOT / "candidates/adt_pixel_receipt.json").read_text()
            ),
            "HOT3D_unique_decoded_frames": len(pixel_cache),
            "HOT3D_cases": 66,
        },
    )
    write_json(
        release_dir / "inventory.json",
        {
            "status": "BUILT_PENDING_VALIDATION",
            "manifest_sha256": {
                f: file_sha256(release_dir / f)
                for f in [
                    "train.jsonl",
                    "val.jsonl",
                    "test.jsonl",
                    "train_sampling_weights.jsonl",
                ]
            },
            "splits": {k: summarize_split(v) for k, v in splits.items()},
            "test_origin_counts": dict(
                collections.Counter(
                    row["release_exposure"]["test_origin"] for row in splits["test"]
                )
            ),
            "fresh_independent_test_certified": False,
        },
    )
    for archive in archive_handles.values():
        archive.close()


if __name__ == "__main__":
    main()
