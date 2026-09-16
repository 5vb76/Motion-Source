"""Decode benchmark inputs in the worker environment through a pickle pipe.

Standard output is reserved for responses to ``BenchmarkWorker``; diagnostics go
through standard error to the parent's ``data_worker.log``.
"""

import pickle
import sys
import traceback
from pathlib import Path

# LOCAL_PATH: Requires benchmark data and scripts/benchmark_loader.py outside this repository.
BENCHMARK_DIR = Path(
    "/root/story2_camera_object_motion/data/benchmarks/general_motion_benchmark_v2"
)
# LOCAL_PATH: External benchmark scripts take precedence over repository modules.
sys.path.insert(0, str(BENCHMARK_DIR / "scripts"))
from benchmark_loader import load_input


def main():
    """Read requests until the parent closes the pipe or sends the None sentinel."""
    while True:
        try:
            request = pickle.load(sys.stdin.buffer)
        except EOFError:
            break
        if request is None:
            break
        try:
            assert set(request) == {"source", "model_input"}
            model_input = load_input(request)
            assert set(model_input) == {
                "frames",
                "target_boxes_xyxy",
                "timestamps_ns",
                "prompt",
            }
            response = {"ok": True, "data": model_input}
        except Exception:
            response = {"ok": False, "error": traceback.format_exc()}
        pickle.dump(response, sys.stdout.buffer, protocol=5)
        sys.stdout.buffer.flush()
        del request, response


if __name__ == "__main__":
    main()
