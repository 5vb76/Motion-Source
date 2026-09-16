"""Shared input, checkpoint, and file helpers for grounding experiments.

JSON result creation is exclusive to prevent overwriting existing results.
"""

from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import sys
import numpy as np

# LOCAL_PATH: Configure external runtime/data sources and the Python interpreter for this machine.
ROOT = Path("/root/autodl-tmp/mosder_reference_research_20260906")
MODULE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = Path(
    "/root/story2_camera_object_motion/experiments/tst_native_adapter_o18_sandbox_v1"
)
GPU_UUID = "GPU-d4f2be0f-222f-82e3-450e-90eadde46613"
PYTHON_EXECUTABLE = "/root/autodl-tmp/envs/molmo2/bin/python3.12"
# LOCAL_PATH: These import directories take precedence over other repository modules.
for import_path in (EXPERIMENT_DIR, MODULE_DIR / "model"):
    sys.path.insert(0, str(import_path))


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8388608), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    """Create a JSON result; fail if a result already exists at this path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def module(path, name):
    """Load a path under a distinct module name to avoid import collisions."""
    spec = importlib.util.spec_from_file_location(name, path)
    loaded_module = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded_module
    spec.loader.exec_module(loaded_module)
    return loaded_module


# LOCAL_PATH: Requires nvila_native_qa_comparison_v1/run_comparison.py under the external ROOT.
def reference():
    return module(
        ROOT / "nvila_native_qa_comparison_v1/run_comparison.py",
        "identity_reference_only",
    )


def train_prompt(row):
    return (
        row["question"]
        + "\n"
        + "\n".join(f"{chr(65 + i)}. {x}" for i, x in enumerate(row["options"]))
        + "\n\nOnly reply with the best option."
    )


def request(row, *, kind="qa", prompt=None):
    """Load verified RGB frames and construct a box-free video request."""
    from soft_query_backend import VideoQuery20Request

    # LOCAL_PATH: The input manifest must point to local RGB NPZ files with matching hashes.
    archive_path = row["rgb20_npz_path"]
    assert sha(archive_path) == row["rgb20_npz_sha256"]
    with np.load(archive_path, allow_pickle=False) as archive:
        rgb = archive["rgb"].copy()
        timestamps = archive["timestamps_ns"].copy()
    expected = row.get("timestamps_ns", row.get("RGB20_timestamps_ns"))
    assert (
        rgb.shape[0] == 20 and rgb.dtype == np.uint8 and timestamps.tolist() == expected
    )
    if prompt is None:
        prompt = row.get("prompt") or train_prompt(row)
    stem = (row.get("question_stem") or row.get("question")) if kind == "qa" else prompt
    video_request = VideoQuery20Request(
        tuple(rgb),
        list(map(int, timestamps)),
        prompt,
        str(row.get("qa_id", row.get("question_id", row.get("pair_id", "new")))),
        stem,
    )
    video_request.validated()
    return video_request


def raw_boxfree_backend(base, router):
    """Attach a uniform router to frozen Molmo without MoSDeR modules."""
    from family_backends_v1 import MolmoNativeBackend
    from soft_query_backend import SoftQueryBackendMixin

    class RawSoftQueryMolmoBackend(SoftQueryBackendMixin, MolmoNativeBackend):
        pass

    args = {
        k: getattr(base, k)
        for k in (
            "model",
            "processor",
            "binding",
            "device",
            "runtime_audit",
            "artifact_audit",
        )
    }
    base.close()
    backend = RawSoftQueryMolmoBackend(**args)
    backend.attach_query_router(router, mode="uniform")
    backend.model.eval()
    backend.freeze_base()
    return backend


def method_state(backend):
    """Return named trainable-method tensors, excluding backbone parameters."""
    from mosder_fgr_runner_candidate_v3.engine import all_method_parameters

    return tuple(all_method_parameters(backend))


def parent_on_raw(base, router):
    """Rebuild the MoSDeR backend and load the recorded parent checkpoint."""
    import torch
    from soft_query_backend import assemble_soft_query_backend
    from mosder_fgr_runner_candidate_v3.checkpoint import load_method_parameter_state

    reference_module = reference()
    # LOCAL_PATH: The external reference module supplies the local parent checkpoint path.
    _, identity = reference_module.parent_paths("molmo2_o_7b")
    backend = assemble_soft_query_backend(base, router)
    payload = torch.load(
        identity["checkpoint_path"], map_location="cpu", weights_only=False
    )
    load_method_parameter_state(
        method_state(backend), payload["method_parameter_state"]
    )
    backend.configure_mosder_stage("FROZEN")
    backend.model.eval()
    return backend, identity


def tensor_hashes(named):
    import torch

    return {
        name: hashlib.sha256(
            parameter.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        ).hexdigest()
        for name, parameter in named
    }


def stamp(model, exclude=()):
    """Capture parameter identity, version, and gradient flags for freeze checks."""
    excluded_parameter_ids = set(exclude)
    return tuple(
        (
            name,
            id(parameter),
            parameter._version,
            parameter.requires_grad,
            parameter.grad is None,
        )
        for name, parameter in model.named_parameters()
        if id(parameter) not in excluded_parameter_ids
    )


HERE = MODULE_DIR
WORK = EXPERIMENT_DIR
GPU = GPU_UUID
PYTHON = PYTHON_EXECUTABLE
