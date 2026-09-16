"""Select ADT windows with twenty visible boxes aligned to native RGB frames."""

import collections
import csv
import json
from pathlib import Path

import numpy as np

# LOCAL_PATH: parents[1] resolves to code/ here; set the benchmark data root
# before using the release/, candidates/, or raw/ subdirectories.
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = [
        json.loads(line)
        for line in (
            BENCHMARK_ROOT / "candidates/adt_testA_physical_candidates.jsonl"
        ).open()
    ]
    index = json.loads(
        (BENCHMARK_ROOT / "candidates/adt_native_index.json").read_text()
    )
    eligible_rows = []
    rejections = collections.Counter()
    for sequence in sorted({row["sequence"] for row in rows}):
        sequence_rows = [row for row in rows if row["sequence"] == sequence]
        boxes = collections.defaultdict(dict)
        with (
            Path(sequence_rows[0]["sequence_path"]) / "2d_bounding_box.csv"
        ).open() as handle:
            for row in csv.DictReader(handle):
                if row["stream_id"] != "214-1":
                    continue
                visibility = float(row["visibility_ratio[%]"])
                visibility = visibility / 100 if visibility > 1 else visibility
                box = [
                    float(row[k])
                    for k in [
                        "x_min[pixel]",
                        "y_min[pixel]",
                        "x_max[pixel]",
                        "y_max[pixel]",
                    ]
                ]
                if (
                    visibility >= 0.5
                    and 0 <= box[0] < box[2] <= 1408
                    and 0 <= box[1] < box[3] <= 1408
                ):
                    boxes[row["object_uid"]][int(row["timestamp[ns]"])] = box
        times = np.array(
            index[sequence_rows[0]["raw_vrs"]]["timestamps_ns"], dtype=np.int64
        )
        aligned = collections.defaultdict(dict)
        for object_uid, object_boxes in boxes.items():
            for timestamp, box in object_boxes.items():
                nearest_index = int(np.abs(times - timestamp).argmin())
                native_timestamp = int(times[nearest_index])
                if abs(native_timestamp - timestamp) <= 500000:
                    aligned[object_uid][native_timestamp] = (box, timestamp)
        boxes = aligned
        for row in sequence_rows:
            window_timestamps = times[
                (times >= row["window_start_ns"]) & (times <= row["window_end_ns"])
            ]
            if len(window_timestamps) < 20:
                rejections["few_rgb"] += 1
                continue
            chosen = window_timestamps[
                np.linspace(0, len(window_timestamps) - 1, 20).astype(int)
            ].tolist()
            lookup = boxes[str(row["object_uid"])]
            # Native frames; nearest annotation within 0.5 ms, recorded explicitly.
            if any(timestamp not in lookup for timestamp in chosen):
                rejections["selected_frame_visibility_or_box"] += 1
                continue
            row["timestamps_ns"] = chosen
            row["boxes_xyxy"] = [lookup[timestamp][0] for timestamp in chosen]
            row["bbox_annotation_timestamps_ns"] = [
                lookup[timestamp][1] for timestamp in chosen
            ]
            row["state"] = row["joint_state"].replace("-", "_")
            row["case_id"] = (
                "v2:ADT:testA:"
                + sequence
                + ":"
                + str(row["object_uid"])
                + ":"
                + str(row["window_start_ns"])
            )
            row["physical_window_id"] = (
                "ADT:" + sequence + ":" + str(row["window_start_ns"])
            )
            eligible_rows.append(row)
        print(
            sequence,
            dict(
                collections.Counter(
                    row["state"] for row in eligible_rows if row["sequence"] == sequence
                )
            ),
            flush=True,
        )
    (BENCHMARK_ROOT / "candidates/adt_testA_rgb20_eligible.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in eligible_rows)
    )
    print(
        "TOTAL",
        len(eligible_rows),
        dict(collections.Counter(row["state"] for row in eligible_rows)),
        dict(rejections),
        flush=True,
    )


if __name__ == "__main__":
    main()
