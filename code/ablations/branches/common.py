"""JSON receipts and RGB decoding used by the inference interventions."""

import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

# LOCAL_PATH: External source directory providing the backend and runner packages.
BACKEND_ROOT = Path(
    "/root/story2_camera_object_motion/experiments/tst_native_adapter_o18_sandbox_v1"
)
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def read_jsonl(path):
    """Read nonempty JSONL records in their original order."""
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path):
    """Hash a file without loading it into memory."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(data)
    return hasher.hexdigest()


def write_json(path, value):
    """Atomically replace a JSON receipt after flushing it to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def append_jsonl(path, value):
    """Append and flush one completed prediction for resumable evaluation."""
    with Path(path).open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class RGB20Worker:
    """Decode RGB frames in the separate data environment."""

    def __init__(self, arm_dir):
        arm_dir = Path(arm_dir)
        env = os.environ.copy()
        for variable in ["PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "VIRTUAL_ENV"]:
            env.pop(variable, None)
        self.log_file = (arm_dir / "data_worker.log").open("ab")
        # LOCAL_PATH: Requires this Python environment and data_worker.py in each arm.
        # prepare.py creates the worker symlink; its external target must remain available.
        self.process = subprocess.Popen(
            ["/root/miniconda3/bin/python", str(arm_dir / "data_worker.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log_file,
            env=env,
        )

    def load(self, row):
        pickle.dump(
            {"source": row["source"], "model_input": row["model_input"]},
            self.process.stdin,
            protocol=5,
        )
        self.process.stdin.flush()
        try:
            response = pickle.load(self.process.stdout)
        except EOFError as error:
            raise RuntimeError("RGB worker exited; see data_worker.log") from error
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["data"]

    def close(self):
        if self.process.poll() is None:
            try:
                pickle.dump(None, self.process.stdin)
                self.process.stdin.flush()
                self.process.wait(timeout=15)
            except Exception:
                self.process.terminate()
        self.log_file.close()


class MotionDataset:
    """Keep labels outside RGB20 requests and wrap them in training examples."""

    def __init__(self, rows, worker):
        self.rows = rows
        self.worker = worker

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from family_backends_v1 import RGB20Request
        from mosder_consumed_field_projection_v2.projection import QUERY_TEXT
        from mosder_fgr_runner_candidate_v3.engine import TrainingExample

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


class OrderedEpoch:
    """Consume a precomputed sample order once, supporting a resume cursor."""

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
