"""Decode selected native ADT frames and record hashes of their RGB pixels."""

import hashlib
import json
from pathlib import Path

import numpy as np
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId

# LOCAL_PATH: parents[1] resolves to code/ here; set the benchmark data root
# before using the release/, candidates/, or raw/ subdirectories.
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]


def main():
    selected_rows = [
        json.loads(line)
        for line in (BENCHMARK_ROOT / "candidates/test_selected.jsonl").open()
    ]
    pixel_hashes = {}
    stream = StreamId("214-1")
    for path in sorted(
        {row["raw_vrs"] for row in selected_rows if row["source"] == "ADT-LiteOffice"}
    ):
        provider = data_provider.create_vrs_data_provider(path)
        pixel_hashes[path] = {}
        selected_timestamps = sorted(
            {
                t
                for row in selected_rows
                if row.get("raw_vrs") == path
                for t in row["timestamps_ns"]
            }
        )
        for timestamp in selected_timestamps:
            data, record = provider.get_image_data_by_time_ns(
                stream, timestamp, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST
            )
            assert int(record.capture_timestamp_ns) == timestamp, (
                timestamp,
                record.capture_timestamp_ns,
            )
            pixels = np.asarray(data.to_numpy_array())
            assert pixels.dtype == np.uint8 and pixels.shape == (1408, 1408, 3)
            pixel_hashes[path][str(timestamp)] = {
                "timestamp_ns": timestamp,
                "width": 1408,
                "height": 1408,
                "pixel_rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                "locator": path + "#214-1@" + str(timestamp),
            }
        print(
            Path(path).name, len(selected_timestamps), "decoded and hashed", flush=True
        )
        (BENCHMARK_ROOT / "candidates/adt_pixel_hashes.json").write_text(
            json.dumps(pixel_hashes)
        )
    (BENCHMARK_ROOT / "candidates/adt_pixel_receipt.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "cases": sum(
                    row["source"] == "ADT-LiteOffice" for row in selected_rows
                ),
                "unique_decoded_frames": sum(
                    len(frames_by_timestamp)
                    for frames_by_timestamp in pixel_hashes.values()
                ),
                "native_frame_alignment": "exact capture timestamp",
                "selection_sha256": hashlib.sha256(
                    (BENCHMARK_ROOT / "candidates/test_selected.jsonl").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
