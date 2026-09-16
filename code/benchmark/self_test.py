"""Behavioral tests using deterministic synthetic predictions and real inputs."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
from benchmark_loader import interpolate_boxes, load_input
from build_v2 import BENCHMARK_ROOT, read_jsonl, write_json
from evaluate import STATES, evaluate


def main():
    rows = read_jsonl(BENCHMARK_ROOT / "release/val.jsonl")
    perfect = [{"case_id": row["case_id"], "state": row["state"]} for row in rows]
    result = evaluate(rows, perfect, 10)
    assert result["joint_state_accuracy"] == 1 and result[
        "joint_accuracy_group_bootstrap_95pct"
    ] == [1, 1]
    inverted = [
        {"case_id": row["case_id"], "state": STATES[3 - STATES.index(row["state"])]}
        for row in rows
    ]
    result = evaluate(rows, inverted, 0)
    assert (
        result["joint_state_accuracy"]
        == result["camera_accuracy"]
        == result["object_accuracy"]
        == 0
    )
    for bad in [
        perfect[:-1],
        perfect + [perfect[0]],
        perfect[:-1] + [{"case_id": perfect[-1]["case_id"], "state": "bad"}],
    ]:
        try:
            evaluate(rows, bad, 0)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid predictions accepted")
    # LOCAL_PATH: External RGB20 loader used to check interpolation consistency.
    sys.path.insert(0, "/root/story2_camera_object_motion")
    from scripts.formal_rgb20_input_loader_v1 import interpolate_and_mask_native_rgb20

    parity = []
    for source in ["ADT-LiteOffice", "HOT3D", "TACO-V1-allocentric"]:
        row = next(row for row in rows if row["source"] == source)
        track = row["model_input"]["oracle_box_track"]
        timestamps = row["trajectory_gt"]["timestamps_ns"]
        width, height = track["source_native_grid_wh"]
        legacy_interpolation = interpolate_and_mask_native_rgb20(
            anchor_boxes=[
                original_input["box_xyxy_half_open_float"]
                for original_input in track["anchors"]
            ],
            anchor_timestamps=[timestamps[i] for i in [0, 9, 19]],
            frame_timestamps=timestamps,
            width=width,
            height=height,
        )
        delta = float(
            np.max(
                np.abs(
                    np.array(interpolate_boxes(track, timestamps))
                    - np.array(
                        legacy_interpolation["interpolated_boxes_xyxy_half_open_float"]
                    )
                )
            )
        )
        assert delta < 1e-8
        parity.append({"source": source, "max_box_difference_pixels": delta})
    train = read_jsonl(BENCHMARK_ROOT / "release/train.jsonl")
    row = next(
        row
        for row in train
        if row["source"] == "AV2"
        and all(Path(f["path"]).is_file() for f in row["model_input"]["rgb20_frames"])
    )
    altered = copy.deepcopy(row)
    altered["supervision"] = {"arbitrary": "DO NOT LEAK"}
    altered["trajectory_gt"] = {"arbitrary": "DO NOT LEAK"}
    altered["state"] = "invalid_gt"
    original_input = load_input(row)
    altered_input = load_input(altered)
    assert (
        set(original_input)
        == set(altered_input)
        == {"frames", "target_boxes_xyxy", "timestamps_ns", "prompt"}
    )
    assert all(
        np.array_equal(x, y)
        for x, y in zip(original_input["frames"], altered_input["frames"])
    )
    for k in ["target_boxes_xyxy", "timestamps_ns", "prompt"]:
        assert original_input[k] == altered_input[k]
    del original_input, altered_input
    corrupted = copy.deepcopy(row)
    corrupted["model_input"]["rgb20_frames"][0]["sha256"] = "0" * 64
    try:
        load_input(corrupted)
    except ValueError as error:
        assert "hash mismatch" in str(error).lower()
    else:
        raise AssertionError("Incorrect image hash accepted")
    for source in ["ADT-LiteOffice", "HOT3D"]:
        row = next(
            row
            for row in read_jsonl(BENCHMARK_ROOT / "release/test.jsonl")
            if row["source"] == source
        )
        altered = copy.deepcopy(row)
        altered["supervision"] = {}
        altered["trajectory_gt"] = {}
        altered["state"] = "invalid"
        original_input = load_input(row)
        altered_input = load_input(altered)
        assert (
            set(original_input)
            == set(altered_input)
            == {"frames", "target_boxes_xyxy", "timestamps_ns", "prompt"}
        )
        assert all(
            np.array_equal(x, y)
            for x, y in zip(original_input["frames"], altered_input["frames"])
        )
        for k in ["target_boxes_xyxy", "timestamps_ns", "prompt"]:
            assert original_input[k] == altered_input[k]
        assert (
            original_input["target_boxes_xyxy"]
            == row["model_input"]["target_boxes_xyxy"]
        )
        corrupted = copy.deepcopy(row)
        corrupted["model_input"]["rgb20"]["frames"][0]["pixel_rgb_sha256"] = "0" * 64
        try:
            load_input(corrupted)
        except (ValueError, RuntimeError) as error:
            assert "hash mismatch" in str(error).lower()
        else:
            raise AssertionError("Incorrect native pixel hash accepted")
        del original_input, altered_input
    receipt = {
        "status": "PASS",
        "tests": [
            "perfect and inverted synthetic metric outputs",
            "duplicate/missing/invalid prediction rejection",
            "old box interpolation parity",
            "GT mutation does not alter model input",
            "incorrect image hash rejected",
            "new ADT and HOT3D native loader GT isolation, exact boxes and corrupted pixel rejection",
        ],
        "interpolation": parity,
        "note": "Functional tests only; no model performance measured.",
    }
    write_json(BENCHMARK_ROOT / "release/functional_test_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
