"""Validate all metadata and a stratified set of real RGB20 inputs."""

import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from benchmark_loader import interpolate_boxes, load_input
from build_v2 import BENCHMARK_ROOT, file_sha256, read_jsonl, write_json

STATES = ["neither", "camera_only", "object_only", "both"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--samples-per-cell", type=int, default=2)
    args = parser.parse_args()
    release = BENCHMARK_ROOT / "release"
    inventory = json.loads((release / "inventory.json").read_text())
    report = {"failures": [], "metadata_cases": 0, "decode_samples": []}
    # LOCAL_PATH: Requires the sibling inventory directory and its referenced files.
    legacy_root = BENCHMARK_ROOT.parent / "old_own_benchmark"
    legacy_inventory = json.loads((legacy_root / "inventory.json").read_text())
    for split, info in legacy_inventory["splits"].items():
        assert file_sha256(legacy_root / info["snapshot"]) == info["sha256"], (
            "Old snapshot modified " + split
        )
    report["old_snapshots_sha256_unchanged"] = True
    rows = []
    content_signatures = {}
    for name in ["train", "val", "test"]:
        manifest_path = release / (name + ".jsonl")
        assert (
            file_sha256(manifest_path)
            == inventory["manifest_sha256"][manifest_path.name]
        )
        split_rows = read_jsonl(manifest_path)
        rows += split_rows
        assert len({row["case_id"] for row in split_rows}) == len(split_rows)
        for row in split_rows:
            trajectory = row["trajectory_gt"]
            timestamps = trajectory["timestamps_ns"]
            supervision = row["supervision"]
            assert len(timestamps) == 20 and all(
                earlier < later for earlier, later in zip(timestamps, timestamps[1:])
            )
            native = trajectory.get("native_sensor_timestamps_ns")
            if native is not None:
                assert len(native) == 20 and all(
                    earlier < later for earlier, later in zip(native, native[1:])
                )
            assert (
                supervision["state"]
                == row["state"]
                == STATES[
                    int(supervision["camera_moving"])
                    + 2 * int(supervision["object_moving"])
                ]
            )
            for kind in ["camera", "object"]:
                positions = np.asarray(trajectory[kind + "_position_world_m"])
                quaternions = np.asarray(
                    trajectory[kind + "_orientation_world_quat_xyzw"]
                )
                assert (
                    positions.shape == (20, 3)
                    and quaternions.shape == (20, 4)
                    and np.isfinite(positions).all()
                    and np.isfinite(quaternions).all()
                )
                assert np.max(np.abs(np.linalg.norm(quaternions, axis=1) - 1)) < 1e-6
            model_input = row["model_input"]
            assert not any(
                k in model_input
                for k in [
                    "supervision",
                    "state",
                    "camera_moving",
                    "object_moving",
                    "trajectory_gt",
                    "physical_annotation",
                ]
            )
            if row["source"] == "AV2":
                frame_records = model_input["rgb20_frames"]
                assert (
                    len(frame_records) == 20
                    and len({f["path"] for f in frame_records}) == 20
                )
                assert [f["timestamp_ns"] for f in frame_records] == timestamps
                if args.decode:
                    assert all(
                        "sha256" in f and Path(f["path"]).is_file()
                        for f in frame_records
                    )
                boxes = model_input["target_boxes_xyxy"]
                width, height = model_input["grid_wh"]
            else:
                rgb = model_input["rgb20"]
                if row["source"] == "TACO-V1-allocentric":
                    assert len(set(rgb["source_frame_ordinals"])) == 20
                    assert Path(rgb["formal_RGB_container"]["path"]).is_file()
                else:
                    assert len({f["locator"] for f in rgb["frames"]}) == 20
                    assert [f["timestamp_ns"] for f in rgb["frames"]] == timestamps
                    assert all(
                        Path(f["locator"].split("#")[0]).is_file()
                        for f in rgb["frames"]
                    )
                boxes = (
                    model_input["target_boxes_xyxy"]
                    if model_input.get("native_rgb20_v2")
                    else interpolate_boxes(model_input["oracle_box_track"], timestamps)
                )
                width, height = (
                    model_input["grid_wh"]
                    if model_input.get("native_rgb20_v2")
                    else model_input["oracle_box_track"]["source_native_grid_wh"]
                )
                for key in ["camera_pose_source", "object_world_pose_source"]:
                    assert Path(row["physical_annotation"][key]["path"]).is_file()
            for x1, y1, x2, y2 in boxes:
                assert (
                    np.isfinite([x1, y1, x2, y2]).all()
                    and 0 <= x1 < x2 <= width
                    and 0 <= y1 < y2 <= height
                )
            if row["source"] == "AV2":
                fingerprint = [
                    f.get("sha256", f["path"]) for f in model_input["rgb20_frames"]
                ]
            elif row["source"] == "TACO-V1-allocentric":
                fingerprint = [
                    model_input["rgb20"]["formal_RGB_container"]["sha256"],
                    model_input["rgb20"]["source_frame_ordinals"],
                ]
            else:
                fingerprint = [
                    f["pixel_rgb_sha256"] for f in model_input["rgb20"]["frames"]
                ]
            signature = hashlib.sha256(json.dumps(fingerprint).encode()).hexdigest()
            if signature in content_signatures:
                assert content_signatures[signature] == row["role"], (
                    "Exact content fingerprint crosses splits"
                )
            content_signatures[signature] = row["role"]
            report["metadata_cases"] += 1
    report["cross_split_exact_content_fingerprint_overlap"] = 0
    report["content_fingerprint_scope"] = (
        "20 pixel hashes for ADT/HOT3D; 20 encoded JPEG hashes for AV2; container hash plus frame ordinals for TACO. Not a perceptual duplicate search."
    )
    weights = read_jsonl(release / "train_sampling_weights.jsonl")
    by_id = {row["case_id"]: row for row in rows if row["role"] == "train"}
    assert len(weights) == len(by_id) and len(
        {row["case_id"] for row in weights}
    ) == len(weights)
    assert all(row["sampling_probability"] > 0 for row in weights)
    assert abs(sum(row["sampling_probability"] for row in weights) - 1) < 1e-9
    state = collections.Counter()
    source = collections.Counter()
    for weight in weights:
        row = by_id[weight["case_id"]]
        state[row["state"]] += weight["sampling_probability"]
        source[row["source"]] += weight["sampling_probability"]
    assert all(abs(v - 0.25) < 1e-9 for v in state.values())
    expected = json.loads((release / "sampling_policy.json").read_text())[
        "source_probabilities"
    ]
    assert all(
        abs(source[source_name] - probability) < 1e-9
        for source_name, probability in expected.items()
    )
    report["verified_sampler_marginals"] = {
        "source": dict(source),
        "state": dict(state),
    }
    if args.decode:
        assert inventory["status"] != "DRAFT_PENDING_MEDIA"
        cells = collections.defaultdict(list)
        for row in rows:
            cells[(row["role"], row["source"], row["state"])].append(row)
        for key, cell_rows in sorted(cells.items()):
            chosen = sorted(
                cell_rows,
                key=lambda row: hashlib.sha256(row["case_id"].encode()).hexdigest(),
            )[: args.samples_per_cell]
            for row in chosen:
                start = time.monotonic()
                try:
                    model_input = load_input(row)
                    assert set(model_input) == {
                        "frames",
                        "target_boxes_xyxy",
                        "timestamps_ns",
                        "prompt",
                    }
                    # TACO frame time conversion may differ by one ns due rational rounding.
                    assert (
                        np.max(
                            np.abs(
                                np.array(model_input["timestamps_ns"], dtype=np.int64)
                                - np.array(
                                    row["trajectory_gt"]["timestamps_ns"],
                                    dtype=np.int64,
                                )
                            )
                        )
                        <= 1
                    )
                    report["decode_samples"].append(
                        {
                            "case_id": row["case_id"],
                            "role": row["role"],
                            "source": row["source"],
                            "state": row["state"],
                            "frames": len(model_input["frames"]),
                            "shape": list(model_input["frames"][0].shape),
                            "seconds": round(time.monotonic() - start, 3),
                        }
                    )
                    del model_input
                except Exception as error:
                    report["failures"].append(
                        {"case_id": row["case_id"], "error": str(error)}
                    )
                print(
                    "decode",
                    len(report["decode_samples"]),
                    "failed",
                    len(report["failures"]),
                    row["case_id"],
                    flush=True,
                )
        report["decode_scope"] = (
            "two deterministic cases per nonempty role/source/state cell; not exhaustive visual or label review"
        )
    report["status"] = "PASS" if not report["failures"] else "FAIL"
    report["validation_scope"] = (
        "full_metadata_and_stratified_decode" if args.decode else "metadata_only"
    )
    report["manifest_sha256"] = inventory["manifest_sha256"]
    write_json(
        release
        / (
            "validation_receipt.json"
            if args.decode
            else "metadata_validation_receipt.json"
        ),
        report,
    )
    if args.decode:
        write_json(
            release / "metadata_validation_receipt.json",
            {
                k: v
                for k, v in report.items()
                if k not in ["decode_samples", "decode_scope"]
            },
        )
    if report["failures"]:
        raise RuntimeError(report["failures"])
    if args.decode:
        inventory["status"] = "READY_FOR_RETRAINING_WITH_HISTORICAL_EXPOSURE"
        write_json(release / "inventory.json", inventory)
    print(
        json.dumps({k: v for k, v in report.items() if k != "decode_samples"}, indent=2)
    )


if __name__ == "__main__":
    main()
