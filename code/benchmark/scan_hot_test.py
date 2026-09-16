"""Scan native HOT3D windows using the recorded motion and visibility thresholds."""

import collections
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

# LOCAL_PATH: External source directory providing the HOT3D scan functions.
sys.path.insert(0, "/root/story2_camera_object_motion/scripts")
from scan_hot3d_role_motion_inventory_v2 import (
    _bbox_audit,
    factor_camera,
    factor_target,
    path_length,
    rotation_path_degrees,
    state_label,
)

# LOCAL_PATH: parents[1] resolves to code/ here; set the benchmark data root
# before using the release/, candidates/, or raw/ subdirectories.
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]


def main():
    # LOCAL_PATH: Physical motion thresholds and annotation-quality rules.
    config = json.loads(
        Path(
            "/root/story2_camera_object_motion/configs/hot3d_motion_scan_validation_role_v2.json"
        ).read_text()
    )
    thresholds = config["motion_thresholds_per_second"]
    quality = config["annotation_quality"]
    candidates = []
    # LOCAL_PATH: HOT3D source archives containing RGB frames and annotations.
    for path in sorted(
        Path("/root/autodl-tmp/tst_plugin_formal_data_v1/testA/HOT3D/raw_clips").glob(
            "*/*.tar"
        )
    ):
        with tarfile.open(path) as archive:
            names = archive.getnames()
            frame_indices = sorted(
                int(window_length.split(".")[0])
                for window_length in names
                if window_length.endswith(".cameras.json")
            )

            def read_annotation(i, k):
                return json.load(archive.extractfile(f"{i:06d}.{k}.json"))

            cameras = {i: read_annotation(i, "cameras")["214-1"] for i in frame_indices}
            objects = {
                i: {
                    str(o["object_uid"]): o
                    for group in read_annotation(i, "objects").values()
                    for o in group
                }
                for i in frame_indices
            }
            infos = {i: read_annotation(i, "info") for i in frame_indices}
            for window_length in [30, 60]:
                for start in range(0, len(frame_indices) - window_length + 1, 15):
                    window = frame_indices[start : start + window_length]
                    assert window == list(range(window[0], window[0] + window_length))
                    timestamps = [
                        infos[i]["image_timestamps_ns"]["214-1"] for i in window
                    ]
                    duration_seconds = (timestamps[-1] - timestamps[0]) / 1e9
                    if duration_seconds <= 0 or max(np.diff(timestamps)) / 1e9 > 0.05:
                        continue
                    camera_poses = [cameras[i]["T_world_from_camera"] for i in window]
                    camera_speed = path_length(camera_poses) / duration_seconds
                    camera_angular_speed = (
                        rotation_path_degrees(camera_poses) / duration_seconds
                    )
                    camera_state = factor_camera(
                        camera_speed, camera_angular_speed, thresholds
                    )
                    if camera_state == "gray":
                        continue
                    targets = set.intersection(*(set(objects[i]) for i in window))
                    sample_indices = np.linspace(0, window_length - 1, 20).astype(int)
                    chosen = [window[i] for i in sample_indices]
                    for object_uid in sorted(targets):
                        annotations = [objects[i][object_uid] for i in window]
                        box = _bbox_audit(
                            annotations, [cameras[i] for i in window], quality, "214-1"
                        )
                        if not box or not box["pass"]:
                            continue
                        object_poses = [o["T_world_from_object"] for o in annotations]
                        object_speed = path_length(object_poses) / duration_seconds
                        object_state = factor_target(object_speed, thresholds)
                        if object_state == "gray":
                            continue
                        state = state_label(camera_state, object_state)
                        sequence = infos[window[0]]["sequence_id"]
                        participant = infos[window[0]]["participant_id"]
                        width, height = box["width"], box["height"]
                        sampled_boxes = np.array(box["boxes_xyxy"])[sample_indices]
                        sampled_boxes[:, [0, 2]] = np.clip(
                            sampled_boxes[:, [0, 2]], 0, width
                        )
                        sampled_boxes[:, [1, 3]] = np.clip(
                            sampled_boxes[:, [1, 3]], 0, height
                        )
                        candidates.append(
                            {
                                "case_id": f"v2:HOT:testA:{path.stem}:{window[0]}:{window_length}:{object_uid}",
                                "source": "HOT3D",
                                "state": state,
                                "sequence": sequence,
                                "split_group": "HOT3D_participant:" + participant,
                                "physical_window_id": f"HOT:{path.stem}:{window[0]}:{window_length}",
                                "target_id": object_uid,
                                "raw_tar": str(path),
                                "native_ordinals": chosen,
                                "timestamps_ns": [
                                    timestamps[i] for i in sample_indices
                                ],
                                "camera_poses": [
                                    camera_poses[i] for i in sample_indices
                                ],
                                "object_poses": [
                                    object_poses[i] for i in sample_indices
                                ],
                                "boxes_xyxy": sampled_boxes.tolist(),
                                "grid_wh": [width, height],
                                "full_window_ordinals": window,
                                "motion_metrics": {
                                    "camera_translation_path_m_per_s": camera_speed,
                                    "camera_rotation_path_deg_per_s": camera_angular_speed,
                                    "object_translation_path_m_per_s": object_speed,
                                },
                                "quality": {
                                    k: v for k, v in box.items() if k != "boxes_xyxy"
                                },
                            }
                        )
        print(
            path.name,
            dict(
                collections.Counter(
                    row["state"] for row in candidates if row["raw_tar"] == str(path)
                )
            ),
            flush=True,
        )
    (BENCHMARK_ROOT / "candidates/hot_testA_rgb20_eligible.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in candidates)
    )
    print(
        "TOTAL",
        len(candidates),
        dict(collections.Counter(row["state"] for row in candidates)),
        flush=True,
    )


if __name__ == "__main__":
    main()
