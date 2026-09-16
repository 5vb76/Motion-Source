"""Load RGB20 inputs without ground-truth labels.

File verification uses content hashes so datasets can be relocated.
"""

import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

import av
import numpy as np
from PIL import Image

# LOCAL_PATH: parents[1] resolves to code/ here; set the benchmark data root
# before using the release/, candidates/, or raw/ subdirectories.
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
# LOCAL_PATH: Python environment with the Aria decoding dependencies installed.
ARIA_PYTHON = "/root/autodl-tmp/envs/adt_pose_scan_py310/bin/python3.10"
_VERIFIED_FILES = {}
MOTION_QUERY = "Analyze the motion source in this 20-frame video for the queried target specified by the oracle box track. Distinguish camera motion from target-object motion. Respond using exactly four lines: Camera, Object, State, Description."


def verify_file(path, expected):
    """Verify SHA256 once per file identity, size, and modification time."""
    file_path = Path(path)
    file_stat = file_path.stat()
    key = (
        str(file_path),
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )
    if _VERIFIED_FILES.get(key) == expected:
        return
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    if hasher.hexdigest() != expected:
        raise ValueError("File hash mismatch: " + str(file_path))
    _VERIFIED_FILES[key] = expected


def interpolate_boxes(track, timestamps):
    """Interpolate box centers and log sizes, retaining exact anchor boxes."""
    anchors = track["anchors"]
    indices = track["anchor_model_frame_indices"]
    boxes = np.array(
        [anchor["box_xyxy_half_open_float"] for anchor in anchors],
        dtype=float,
    )
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    size = boxes[:, 2:] - boxes[:, :2]
    if np.any(size <= 0):
        raise ValueError("Invalid anchor box")
    values = np.concatenate([centers, np.log(size)], axis=1)
    times = np.array(timestamps, dtype=np.int64)
    seconds = (times - times[0]) / 1e9
    interpolated = np.stack(
        [np.interp(seconds, seconds[indices], values[:, axis]) for axis in range(4)],
        axis=1,
    )
    half = np.exp(interpolated[:, 2:]) / 2
    result = np.concatenate(
        [interpolated[:, :2] - half, interpolated[:, :2] + half], axis=1
    )
    result[indices] = boxes
    width, height = track["source_native_grid_wh"]
    result[:, [0, 2]] = np.clip(result[:, [0, 2]], 0, width)
    result[:, [1, 3]] = np.clip(result[:, [1, 3]], 0, height)
    return result.tolist()


def load_input(row):
    """Decode and validate twenty RGB frames without exposing labels or poses."""
    # LOCAL_PATH: Manifest paths/locators must resolve to local media; no remapping occurs.
    model_input = row["model_input"]
    frames = []
    if row["source"] == "AV2":
        records = model_input["rgb20_frames"]
        for record in records:
            if "sha256" in record:
                verify_file(record["path"], record["sha256"])
            with Image.open(record["path"]) as image:
                frames.append(np.array(image.convert("RGB")))
        timestamps = [record["timestamp_ns"] for record in records]
        boxes = model_input["target_boxes_xyxy"]
    else:
        rgb = model_input["rgb20"]
        if row["source"] == "ADT-LiteOffice":
            records = rgb["frames"]
            path = records[0]["locator"].split("#")[0]
            with tempfile.TemporaryDirectory(
                prefix="general_benchmark_adt_"
            ) as temporary_dir:
                request_path = Path(temporary_dir) / "request.json"
                frames_path = Path(temporary_dir) / "frames.npy"
                request_path.write_text(json.dumps({"path": path, "frames": records}))
                # LOCAL_PATH: Requires the ADT decoder in this sibling scripts/ directory.
                decoder = subprocess.run(
                    [
                        ARIA_PYTHON,
                        str(
                            BENCHMARK_ROOT.with_name("general_motion_benchmark_v1")
                            / "scripts/adt_decode.py"
                        ),
                        str(request_path),
                        str(frames_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                if decoder.returncode:
                    raise RuntimeError(decoder.stderr[-4000:])
                frames = list(np.load(frames_path, allow_pickle=False))
            timestamps = [record["timestamp_ns"] for record in records]
        elif row["source"] == "HOT3D":
            records = rgb["frames"]
            path = records[0]["locator"].split("#")[0]
            with tarfile.open(path, "r:*") as archive:
                for record in records:
                    member = record["locator"].split("#", 1)[1]
                    with Image.open(
                        io.BytesIO(archive.extractfile(member).read())
                    ) as image:
                        array = np.array(image.convert("RGB"))
                    if (
                        hashlib.sha256(array.tobytes()).hexdigest()
                        != record["pixel_rgb_sha256"]
                    ):
                        raise ValueError("HOT3D pixel hash mismatch")
                    frames.append(array)
            timestamps = [record["timestamp_ns"] for record in records]
        elif row["source"] == "TACO-V1-allocentric":
            container = rgb["formal_RGB_container"]
            verify_file(container["path"], container["sha256"])
            selected = rgb["source_frame_ordinals"]
            wanted = set(selected)
            decoded = {}
            frame_timestamps = {}
            with av.open(container["path"]) as video:
                for frame_index, frame in enumerate(video.decode(video=0)):
                    if frame_index in wanted:
                        decoded[frame_index] = frame.to_ndarray(format="rgb24")
                        if frame.pts is None or frame.time_base is None:
                            raise ValueError("TACO frame timestamp missing")
                        frame_timestamps[frame_index] = int(
                            frame.pts * frame.time_base * 1000000000
                        )
                    if frame_index >= selected[-1]:
                        break
            if set(decoded) != wanted:
                raise ValueError("TACO selected frames missing")
            frames = [decoded[frame_index] for frame_index in selected]
            timestamps = [frame_timestamps[frame_index] for frame_index in selected]
        else:
            raise ValueError("Unsupported source")
        boxes = (
            model_input["target_boxes_xyxy"]
            if model_input.get("native_rgb20_v2")
            else interpolate_boxes(model_input["oracle_box_track"], timestamps)
        )
    if len(frames) != 20 or len(timestamps) != 20 or len(boxes) != 20:
        raise ValueError("Expected RGB20")
    if not all(
        left_timestamp < right_timestamp
        for left_timestamp, right_timestamp in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("Nonchronological frames")
    expected_grid = (
        model_input["grid_wh"]
        if row["source"] == "AV2" or model_input.get("native_rgb20_v2")
        else model_input["oracle_box_track"]["source_native_grid_wh"]
    )
    for image, box in zip(frames, boxes):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Invalid RGB frame")
        if [image.shape[1], image.shape[0]] != list(expected_grid):
            raise ValueError("Decoded dimensions differ from target-box grid")
        x1, y1, x2, y2 = box
        if not (
            np.isfinite(box).all()
            and 0 <= x1 < x2 <= image.shape[1]
            and 0 <= y1 < y2 <= image.shape[0]
        ):
            raise ValueError("Invalid target box")
    # Only this dictionary is passed to a model. Label/pose/source metadata stay
    # outside the forward input, including when the entire release row has GT.
    return {
        "frames": frames,
        "target_boxes_xyxy": boxes,
        "timestamps_ns": timestamps,
        "prompt": MOTION_QUERY,
    }
