"""Data loading, durable run records, and the fixed training-order sampler.

See ../README.md for the benchmark and backend dependencies.
"""

import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
# LOCAL_PATH: Set the benchmark, external backend source, and worker interpreter for this machine.
BENCHMARK_DIR = Path(
    "/root/story2_camera_object_motion/data/benchmarks/general_motion_benchmark_v2"
)
BACKEND_SOURCE_DIR = Path(
    "/root/story2_camera_object_motion/experiments/tst_native_adapter_o18_sandbox_v1"
)
WORKER_PYTHON = "/root/miniconda3/bin/python"

# LOCAL_PATH: Prepending this directory can load external modules before repository code.
if str(BACKEND_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_SOURCE_DIR))


def read_jsonl(path):
    """Read JSON records, ignoring blank lines."""
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def file_sha256(path):
    """Hash a file without loading checkpoints into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """Replace a JSON record atomically after flushing it to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)


def append_jsonl(path, value):
    """Append and flush one record so a resumed run can recover its progress."""
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class BenchmarkWorker:
    """Decode RGB inputs in the benchmark's separate Python environment."""

    def __init__(self):
        environment = os.environ.copy()
        for name in ["PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "VIRTUAL_ENV"]:
            environment.pop(name, None)
        self.log = (RUN_DIR / "data_worker.log").open("ab")
        self.proc = subprocess.Popen(
            [WORKER_PYTHON, str(RUN_DIR / "data_worker.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            env=environment,
        )

    def load(self, row):
        """Send only benchmark input fields; labels stay in the training process."""
        pickle.dump(
            {"source": row["source"], "model_input": row["model_input"]},
            self.proc.stdin,
            protocol=5,
        )
        self.proc.stdin.flush()
        try:
            response = pickle.load(self.proc.stdout)
        except EOFError:
            raise RuntimeError("RGB worker exited; see data_worker.log")
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["data"]

    def close(self):
        """Ask the worker to exit, terminating it if graceful shutdown fails."""
        if self.proc.poll() is None:
            try:
                pickle.dump(None, self.proc.stdin)
                self.proc.stdin.flush()
                self.proc.wait(timeout=15)
            except Exception:
                self.proc.terminate()
        self.log.close()


class MotionDataset:
    """Load one benchmark row as the backend's validated training example."""

    def __init__(self, rows, worker):
        self.rows = rows
        self.worker = worker

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from family_backends_v1 import RGB20Request
        from mosder_fgr_runner_candidate_v3.engine import TrainingExample
        from mosder_consumed_field_projection_v2.projection import QUERY_TEXT

        row = self.rows[index]
        model_input = self.worker.load(row)
        assert model_input["prompt"] == QUERY_TEXT
        request = RGB20Request(
            frames=tuple(model_input["frames"]),
            timestamps_ns=tuple(model_input["timestamps_ns"]),
            prompt=model_input["prompt"],
            oracle_boxes_xyxy=model_input["target_boxes_xyxy"],
            request_id=row["case_id"],
        )
        request.validated()
        return TrainingExample(
            case_id=row["case_id"],
            state=row["state"],
            source_dataset=row["source"],
            request=request,
        )


class PlannedEpochSampler:
    """Consume the plan's precomputed sample order, with a resumable cursor.

    Sampling weights are already reflected in ``order``. This class performs no
    random draws and exposes the interface expected by ``run_current_epoch``.
    """

    def __init__(self, rows, order, cursor=0):
        self.rows = rows
        self.order = order
        self.cursor = cursor
        self.epoch = 0

    @property
    def at_epoch_end(self):
        return self.cursor == len(self.order)

    @property
    def exhausted(self):
        return self.epoch > 0

    def next_index(self):
        if self.at_epoch_end:
            return None
        index = self.order[self.cursor]
        self.cursor += 1
        return index

    def finish_epoch(self):
        assert self.at_epoch_end
        self.epoch += 1


ROOT = RUN_DIR
B = BENCHMARK_DIR
EX = BACKEND_SOURCE_DIR
Worker = BenchmarkWorker
Dataset = MotionDataset
WeightedEpoch = PlannedEpochSampler


def read(p):
    return read_jsonl(p)


def sha(p):
    return file_sha256(p)


def write(p, v):
    return write_json(p, v)


def append(p, v):
    return append_jsonl(p, v)
