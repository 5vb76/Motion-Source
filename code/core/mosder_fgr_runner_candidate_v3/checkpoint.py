"""Atomic checkpoints, parameter snapshots, and reproducible resume state."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Final, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .protocol import (
    HOLD_STATUS,
    RunProtocol,
    STAGES,
    canonical_json_bytes,
    canonical_sha256,
)
from .sampler import FullCoverageStateSampler
from .schedule import WarmupCosineToFloor


SCHEMA_VERSION: Final[str] = "mosder_fgr_checkpoint_candidate_v1"


class CheckpointContractError(RuntimeError):
    """Checkpoint identity, tensor inventory, or resume state is invalid."""


NamedParameters = Sequence[tuple[str, nn.Parameter]]


def numeric_environment_snapshot() -> Mapping[str, Any]:
    """Return every Torch switch that can change the resumed CUDA update.

    This function is intentionally side-effect free.  The snapshot is bound
    into the runtime manifest, while ``configure_reproducible_numeric_environment``
    must run before the first CUDA query/allocation in a training process.
    """

    cuda_matmul = torch.backends.cuda.matmul
    return {
        "schema_version": "mosder_reproducible_numeric_environment_v1",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": bool(cuda_matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_fp16_reduced_precision_reduction": bool(
            cuda_matmul.allow_fp16_reduced_precision_reduction
        ),
        "matmul_allow_bf16_reduced_precision_reduction": bool(
            cuda_matmul.allow_bf16_reduced_precision_reduction
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "deterministic_debug_mode": int(torch.get_deterministic_debug_mode()),
        "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "mem_efficient_sdp_enabled": bool(
            torch.backends.cuda.mem_efficient_sdp_enabled()
        ),
        "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
    }


def configure_reproducible_numeric_environment() -> Mapping[str, Any]:
    """Fail closed onto the numeric environment used by formal GPU runs."""

    if torch.cuda.is_initialized():
        raise CheckpointContractError(
            "numeric environment must be installed before CUDA initialization"
        )
    requested_workspace = ":4096:8"
    existing_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_workspace not in {None, requested_workspace}:
        raise CheckpointContractError(
            "CUBLAS_WORKSPACE_CONFIG differs from the sealed numeric policy"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = requested_workspace
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    observed = numeric_environment_snapshot()
    expected = {
        "cublas_workspace_config": requested_workspace,
        "float32_matmul_precision": "highest",
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "matmul_allow_fp16_reduced_precision_reduction": False,
        "matmul_allow_bf16_reduced_precision_reduction": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
    }
    if any(observed.get(name) != value for name, value in expected.items()):
        raise CheckpointContractError(
            "failed to install the sealed reproducible numeric environment"
        )
    return observed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_bytes(value: Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    # ``Tensor.view(dtype)`` rejects a zero-dimensional scalar when element
    # sizes differ.  Flatten first so scalars, BF16 tensors, and ordinary
    # matrices all use the same exact raw-byte representation.
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")


def tensor_state_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, Tensor):
            raise CheckpointContractError(f"non-tensor state value: {name}")
        header = {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        encoded = str(canonical_sha256(header)).encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        raw = _tensor_bytes(value)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _validated_named_parameters(
    values: NamedParameters,
) -> tuple[tuple[str, nn.Parameter], ...]:
    entries = tuple(values)
    names = tuple(name for name, _ in entries)
    identities = tuple(id(parameter) for _, parameter in entries)
    if (
        not entries
        or any(not isinstance(name, str) or not name for name in names)
        or any(not isinstance(parameter, nn.Parameter) for _, parameter in entries)
        or len(names) != len(set(names))
        or len(identities) != len(set(identities))
    ):
        raise CheckpointContractError("method parameter inventory is invalid")
    return entries


def method_parameter_state(values: NamedParameters) -> Mapping[str, Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in _validated_named_parameters(values)
    }


def method_parameter_specs(values: NamedParameters) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "numel": parameter.numel(),
        }
        for name, parameter in _validated_named_parameters(values)
    )


def load_method_parameter_state(
    values: NamedParameters,
    state: Mapping[str, Tensor],
) -> None:
    entries = _validated_named_parameters(values)
    if set(state) != {name for name, _ in entries}:
        raise CheckpointContractError("checkpoint method parameter names differ")
    with torch.no_grad():
        for name, parameter in entries:
            saved = state[name]
            if (
                not isinstance(saved, Tensor)
                or tuple(saved.shape) != tuple(parameter.shape)
                or saved.dtype != parameter.dtype
                or not bool(torch.isfinite(saved).all())
            ):
                raise CheckpointContractError(
                    f"checkpoint tensor spec/non-finite value differs: {name}"
                )
            parameter.copy_(saved.to(device=parameter.device))


def capture_rng_state() -> Mapping[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "schema_version": "mosder_rng_state_v1",
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "keys_uint32": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [value.clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
        "cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
    }


def restore_rng_state(value: Mapping[str, Any]) -> None:
    if (
        set(value)
        != {
            "schema_version",
            "python",
            "numpy",
            "torch_cpu",
            "torch_cuda",
            "cuda_device_count",
        }
        or value.get("schema_version") != "mosder_rng_state_v1"
    ):
        raise CheckpointContractError("checkpoint RNG fields differ")
    numpy_state = value.get("numpy")
    if not isinstance(numpy_state, Mapping) or set(numpy_state) != {
        "bit_generator",
        "keys_uint32",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise CheckpointContractError("checkpoint NumPy RNG state differs")
    keys = numpy_state["keys_uint32"]
    if not isinstance(keys, Tensor) or keys.dtype != torch.uint32 or keys.ndim != 1:
        raise CheckpointContractError("checkpoint NumPy RNG keys are invalid")
    expected_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if value.get("cuda_device_count") != expected_cuda:
        raise CheckpointContractError("checkpoint CUDA RNG device count differs")
    cuda_state = value.get("torch_cuda")
    if not isinstance(cuda_state, list) or len(cuda_state) != expected_cuda:
        raise CheckpointContractError("checkpoint CUDA RNG state count differs")
    random.setstate(value["python"])
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            keys.cpu().numpy().copy(),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    cpu_state = value.get("torch_cpu")
    if not isinstance(cpu_state, Tensor) or cpu_state.dtype != torch.uint8:
        raise CheckpointContractError("checkpoint Torch CPU RNG state is invalid")
    torch.set_rng_state(cpu_state.cpu())
    if expected_cuda:
        if any(
            not isinstance(item, Tensor) or item.dtype != torch.uint8
            for item in cuda_state
        ):
            raise CheckpointContractError("checkpoint Torch CUDA RNG state is invalid")
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_state])


def set_global_seed(seed: int) -> None:
    if type(seed) is not int or seed < 0:
        raise CheckpointContractError("seed must be a nonnegative int")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_checkpoint(
    *,
    protocol: RunProtocol,
    stage: str,
    stage_complete: bool,
    stage_optimizer_step: int,
    global_optimizer_step: int,
    micro_examples_seen: int,
    method_parameters: NamedParameters,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineToFloor,
    sampler: FullCoverageStateSampler,
    best_validation: Mapping[str, Any],
    frozen_base_digest: str,
    parent_lineage: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    protocol.validated()
    if stage not in STAGES or type(stage_complete) is not bool:
        raise CheckpointContractError("checkpoint stage is invalid")
    for name, value in (
        ("stage_optimizer_step", stage_optimizer_step),
        ("global_optimizer_step", global_optimizer_step),
        ("micro_examples_seen", micro_examples_seen),
    ):
        if type(value) is not int or value < 0:
            raise CheckpointContractError(f"{name} is invalid")
    entries = _validated_named_parameters(method_parameters)
    if any(parameter.grad is not None for _, parameter in entries):
        raise CheckpointContractError(
            "checkpoint is allowed only at a zero-gradient optimizer boundary"
        )
    if not isinstance(frozen_base_digest, str) or len(frozen_base_digest) != 64:
        raise CheckpointContractError("frozen base digest is invalid")
    state = method_parameter_state(entries)
    specs = method_parameter_specs(entries)
    state_sha = tensor_state_sha256(state)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": dict(protocol.as_dict()),
        "protocol_sha256": protocol.identity_sha256,
        "stage": stage,
        "stage_complete": stage_complete,
        "stage_optimizer_step": stage_optimizer_step,
        "global_optimizer_step": global_optimizer_step,
        "micro_examples_seen": micro_examples_seen,
        "method_parameter_specs": list(specs),
        "method_parameter_state": state,
        "method_parameter_state_sha256": state_sha,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": dict(scheduler.state_dict()),
        "sampler_state": dict(sampler.state_dict()),
        "rng_state": capture_rng_state(),
        "best_validation": dict(best_validation),
        "parent_lineage": dict(parent_lineage or {}),
        "frozen_base_digest": frozen_base_digest,
        "accumulation_partial_count": 0,
        "authority": {
            "formal_training_authorized": False,
            "hold_status": HOLD_STATUS,
            "confirmation_a_opened": False,
            "final_b_opened": False,
            "held_roles_opened": False,
        },
    }


def validate_checkpoint(
    value: Mapping[str, Any],
    *,
    protocol: RunProtocol,
    expected_stage: str | None = None,
) -> None:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointContractError("checkpoint schema differs")
    if (
        value.get("protocol") != dict(protocol.as_dict())
        or value.get("protocol_sha256") != protocol.identity_sha256
    ):
        raise CheckpointContractError("checkpoint protocol identity differs")
    stage = value.get("stage")
    if stage not in STAGES or (expected_stage is not None and stage != expected_stage):
        raise CheckpointContractError("checkpoint stage differs")
    authority = value.get("authority")
    if authority != {
        "formal_training_authorized": False,
        "hold_status": HOLD_STATUS,
        "confirmation_a_opened": False,
        "final_b_opened": False,
        "held_roles_opened": False,
    }:
        raise CheckpointContractError("checkpoint negative authority differs")
    if value.get("accumulation_partial_count") != 0:
        raise CheckpointContractError("checkpoint contains a partial accumulation")
    if type(value.get("stage_complete")) is not bool:
        raise CheckpointContractError("checkpoint stage_complete is invalid")
    for name in (
        "stage_optimizer_step",
        "global_optimizer_step",
        "micro_examples_seen",
    ):
        if type(value.get(name)) is not int or value[name] < 0:
            raise CheckpointContractError(f"checkpoint {name} is invalid")
    if not isinstance(value.get("best_validation"), Mapping):
        raise CheckpointContractError("checkpoint best-validation state is invalid")
    if not isinstance(value.get("parent_lineage"), Mapping):
        raise CheckpointContractError("checkpoint parent lineage is invalid")
    state = value.get("method_parameter_state")
    if not isinstance(state, Mapping) or value.get(
        "method_parameter_state_sha256"
    ) != tensor_state_sha256(state):
        raise CheckpointContractError("checkpoint method state digest differs")


def restore_checkpoint(
    value: Mapping[str, Any],
    *,
    protocol: RunProtocol,
    expected_stage: str,
    method_parameters: NamedParameters,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineToFloor,
    sampler: FullCoverageStateSampler,
    frozen_base_digest: str,
    restore_rng: bool = True,
) -> Mapping[str, Any]:
    validate_checkpoint(value, protocol=protocol, expected_stage=expected_stage)
    if value.get("frozen_base_digest") != frozen_base_digest:
        raise CheckpointContractError("frozen base digest differs on resume")
    specs = list(method_parameter_specs(method_parameters))
    if value.get("method_parameter_specs") != specs:
        raise CheckpointContractError("method parameter inventory differs on resume")
    state = value["method_parameter_state"]
    load_method_parameter_state(method_parameters, state)
    if tensor_state_sha256(method_parameter_state(method_parameters)) != value.get(
        "method_parameter_state_sha256"
    ):
        raise CheckpointContractError("loaded method state digest differs")
    optimizer.load_state_dict(value["optimizer_state"])
    scheduler.load_state_dict(value["scheduler_state"])
    if scheduler.completed_steps != value.get("stage_optimizer_step"):
        raise CheckpointContractError("scheduler/stage step differs on resume")
    sampler.load_state_dict(value["sampler_state"])
    if any(parameter.grad is not None for _, parameter in method_parameters):
        raise CheckpointContractError("resume restored a partial gradient")
    if restore_rng:
        restore_rng_state(value["rng_state"])
    return {
        "stage": value["stage"],
        "stage_complete": value["stage_complete"],
        "stage_optimizer_step": value["stage_optimizer_step"],
        "global_optimizer_step": value["global_optimizer_step"],
        "micro_examples_seen": value["micro_examples_seen"],
        "best_validation": dict(value["best_validation"]),
        "parent_lineage": dict(value["parent_lineage"]),
        "method_parameter_state_sha256": value["method_parameter_state_sha256"],
    }


def atomic_save_checkpoint(path: Path, value: Mapping[str, Any]) -> str:
    """Atomically replace one explicitly scoped checkpoint path."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(value), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_once_checkpoint(path: Path, value: Mapping[str, Any]) -> str:
    """Create one immutable checkpoint; an existing name is never replaced."""

    path = path.resolve()
    if path.suffix != ".pt" or path.name in {".pt", "..pt"}:
        raise CheckpointContractError("checkpoint filename must end in .pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise CheckpointContractError("immutable checkpoint path already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(value), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise CheckpointContractError(
                "immutable checkpoint path raced with another writer"
            ) from error
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def commit_pointer(
    directory: Path,
    *,
    pointer_name: str,
    checkpoint_name: str,
    checkpoint_sha256: str,
    protocol: RunProtocol,
    stage: str,
    stage_optimizer_step: int,
    global_optimizer_step: int,
) -> Mapping[str, Any]:
    """Atomically publish the sole authority for latest/best recovery."""

    if (
        pointer_name not in {"LATEST.json", "BEST.json"}
        or not checkpoint_name.endswith(".pt")
        or Path(checkpoint_name).name != checkpoint_name
        or len(checkpoint_sha256) != 64
        or stage not in STAGES
        or type(stage_optimizer_step) is not int
        or stage_optimizer_step < 0
        or type(global_optimizer_step) is not int
        or global_optimizer_step < 0
    ):
        raise CheckpointContractError("checkpoint pointer arguments are invalid")
    directory = directory.resolve()
    target = directory / checkpoint_name
    if not target.is_file() or target.is_symlink():
        raise CheckpointContractError("pointer target is not an immutable regular file")
    if sha256_file(target) != checkpoint_sha256:
        raise CheckpointContractError("pointer target SHA256 differs")
    payload = {
        "schema_version": "mosder_checkpoint_pointer_v1",
        "pointer_role": pointer_name.removesuffix(".json").lower(),
        "checkpoint_name": checkpoint_name,
        "checkpoint_sha256": checkpoint_sha256,
        "protocol_sha256": protocol.identity_sha256,
        "stage": stage,
        "stage_optimizer_step": stage_optimizer_step,
        "global_optimizer_step": global_optimizer_step,
    }
    pointer = directory / pointer_name
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{pointer.name}.", suffix=".tmp", dir=str(directory)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, pointer)
        _fsync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def load_committed_checkpoint(
    directory: Path,
    *,
    pointer_name: str,
    protocol: RunProtocol,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Verify pointer/path/SHA before safe deserialization."""

    if pointer_name not in {"LATEST.json", "BEST.json"}:
        raise CheckpointContractError("unknown checkpoint pointer role")
    directory = directory.resolve()
    pointer = directory / pointer_name
    if not pointer.is_file() or pointer.is_symlink():
        raise CheckpointContractError("checkpoint pointer is not a regular file")
    raw_pointer = pointer.read_bytes()
    try:
        metadata = json.loads(raw_pointer)
    except Exception as error:
        raise CheckpointContractError("checkpoint pointer JSON is invalid") from error
    if (
        not isinstance(metadata, Mapping)
        or raw_pointer != canonical_json_bytes(metadata) + b"\n"
        or metadata.get("schema_version") != "mosder_checkpoint_pointer_v1"
        or metadata.get("pointer_role") != pointer_name.removesuffix(".json").lower()
        or metadata.get("protocol_sha256") != protocol.identity_sha256
    ):
        raise CheckpointContractError("checkpoint pointer identity differs")
    name = metadata.get("checkpoint_name")
    digest = metadata.get("checkpoint_sha256")
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or not name.endswith(".pt")
        or not isinstance(digest, str)
        or len(digest) != 64
    ):
        raise CheckpointContractError("checkpoint pointer target fields are invalid")
    target = directory / name
    if not target.is_file() or target.is_symlink():
        raise CheckpointContractError("committed checkpoint is not a regular file")
    payload = target.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise CheckpointContractError("committed checkpoint SHA256 differs")
    try:
        value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as error:
        raise CheckpointContractError(
            "safe committed checkpoint load failed"
        ) from error
    if not isinstance(value, Mapping):
        raise CheckpointContractError("committed checkpoint root is invalid")
    validate_checkpoint(value, protocol=protocol, expected_stage=str(metadata["stage"]))
    if value.get("stage_optimizer_step") != metadata.get(
        "stage_optimizer_step"
    ) or value.get("global_optimizer_step") != metadata.get("global_optimizer_step"):
        raise CheckpointContractError("pointer/checkpoint progress differs")
    return value, metadata


def load_checkpoint(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CheckpointContractError("checkpoint must be a regular local file")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CheckpointContractError("safe checkpoint load failed") from error
    if not isinstance(value, Mapping):
        raise CheckpointContractError("checkpoint root is not a mapping")
    return value


__all__ = [
    "CheckpointContractError",
    "SCHEMA_VERSION",
    "atomic_save_checkpoint",
    "build_checkpoint",
    "capture_rng_state",
    "commit_pointer",
    "configure_reproducible_numeric_environment",
    "create_once_checkpoint",
    "load_committed_checkpoint",
    "load_checkpoint",
    "load_method_parameter_state",
    "method_parameter_specs",
    "method_parameter_state",
    "numeric_environment_snapshot",
    "restore_checkpoint",
    "restore_rng_state",
    "set_global_seed",
    "sha256_file",
    "tensor_state_sha256",
    "validate_checkpoint",
]
