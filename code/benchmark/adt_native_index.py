"""Index native ADT RGB timestamps and the device-to-camera calibration."""

import json
from pathlib import Path

from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.stream_id import StreamId

# LOCAL_PATH: parents[1] resolves to code/ here; set the benchmark data root
# before using the release/, candidates/, or raw/ subdirectories.
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]


def main():
    candidates = [
        json.loads(line)
        for line in (
            BENCHMARK_ROOT / "candidates/adt_testA_physical_candidates.jsonl"
        ).open()
    ]
    native_index = {}
    for vrs_path in sorted({row["raw_vrs"] for row in candidates}):
        provider = data_provider.create_vrs_data_provider(vrs_path)
        native_index[vrs_path] = {
            "timestamps_ns": list(
                provider.get_timestamps_ns(StreamId("214-1"), TimeDomain.DEVICE_TIME)
            ),
            "device_from_camera": provider.get_device_calibration()
            .get_transform_device_sensor("camera-rgb")
            .to_matrix()
            .tolist(),
        }
        print(
            Path(vrs_path).name,
            len(native_index[vrs_path]["timestamps_ns"]),
            flush=True,
        )
    (BENCHMARK_ROOT / "candidates/adt_native_index.json").write_text(
        json.dumps(native_index)
    )


if __name__ == "__main__":
    main()
