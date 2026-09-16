"""Record parameter names, model files, code hashes, and runtime versions."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
from typing import Any, Final, Mapping, Sequence

import torch
from torch import nn

from .checkpoint import (
    method_parameter_state,
    numeric_environment_snapshot,
    tensor_state_sha256,
)
from .engine import all_method_parameters
from .protocol import HOLD_STATUS, canonical_sha256


SCHEMA_VERSION: Final[str] = "mosder_runtime_and_parameter_manifest_candidate_v1"


class ManifestContractError(RuntimeError):
    """A file, parameter, owner, or environment binding is incomplete."""


def sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ManifestContractError(f"manifest path is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files_under(root: Path) -> tuple[Path, ...]:
    if not root.is_dir() or root.is_symlink():
        raise ManifestContractError(f"manifest root is not a regular directory: {root}")
    output: list[Path] = []
    for directory, names, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        if any((directory_path / name).is_symlink() for name in names):
            raise ManifestContractError(
                f"symlinked directory in manifest root: {directory}"
            )
        for name in files:
            path = directory_path / name
            if not path.is_symlink() and not path.is_file():
                raise ManifestContractError(f"non-regular manifest file: {path}")
            output.append(path)
    return tuple(sorted(output))


def file_tree_manifest(root: Path) -> Mapping[str, Any]:
    root = root.resolve()
    files: list[Mapping[str, Any]] = []
    for path in _files_under(root):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            link_target = os.readlink(path)
            resolved_target = (path.parent / link_target).resolve(strict=False)
            target_exists = resolved_target.exists()
            if target_exists and (
                not resolved_target.is_file() or resolved_target.is_symlink()
            ):
                raise ManifestContractError(
                    f"symlink target is not a regular file: {path}"
                )
            entry: dict[str, Any] = {
                "relative_path": relative,
                "kind": "symbolic_link",
                "link_target": link_target,
                "link_target_sha256": hashlib.sha256(
                    link_target.encode("utf-8")
                ).hexdigest(),
                "resolved_target": str(resolved_target),
                "target_exists": target_exists,
            }
            if target_exists:
                entry.update(
                    {
                        "target_bytes": resolved_target.stat().st_size,
                        "target_sha256": sha256_file(resolved_target),
                    }
                )
            files.append(entry)
        else:
            files.append(
                {
                    "relative_path": relative,
                    "kind": "regular_file",
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "root": str(root),
        "files": files,
        "file_count": len(files),
        "regular_file_count": sum(item["kind"] == "regular_file" for item in files),
        "symbolic_link_count": sum(item["kind"] == "symbolic_link" for item in files),
        "broken_symbolic_link_count": sum(
            item["kind"] == "symbolic_link" and item["target_exists"] is False
            for item in files
        ),
        "total_regular_file_bytes": sum(
            int(item["bytes"]) for item in files if item["kind"] == "regular_file"
        ),
        "tree_sha256": canonical_sha256(files),
    }


def code_manifest(paths: Sequence[Path]) -> Mapping[str, Any]:
    resolved = tuple(path.resolve() for path in paths)
    if not resolved or len(resolved) != len(set(resolved)):
        raise ManifestContractError("code manifest paths are empty/duplicated")
    files = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(resolved)
    ]
    return {
        "files": files,
        "file_count": len(files),
        "manifest_sha256": canonical_sha256(files),
    }


def _initialization_rule(owner: str, name: str) -> str:
    if owner == "frozen_base":
        return "load_exact_local_pretrained_artifact"
    if name.endswith("time_embedding"):
        return "normal_mean0_std0p02"
    if name.endswith("decision.b_c") or name.endswith("decision.b_o"):
        return "zeros"
    if owner in {"camera", "object", "shared"} and any(
        marker in name for marker in ("_A.", ".down.weight")
    ):
        return "torch_linear_default_kaiming_uniform"
    if owner in {"camera", "object", "shared"} and "_B." in name:
        return "normal_mean0_std0p01"
    if owner in {"camera", "object"} and "source_gate" in name:
        return "xavier_uniform"
    if owner == "residual" and any(
        name.endswith(suffix) for suffix in ("up.weight", "gate.weight", "gate.bias")
    ):
        return "zeros"
    if owner == "residual" and name.endswith("down.weight"):
        return "kaiming_uniform_a_sqrt5"
    if name.startswith("source_core."):
        return "pytorch_module_reset_parameters"
    raise ManifestContractError(f"no initialization rule for {owner}:{name}")


def parameter_manifest(backend: Any) -> Mapping[str, Any]:
    if not isinstance(getattr(backend, "model", None), nn.Module):
        raise ManifestContractError("backend has no live model")
    method = all_method_parameters(backend)
    logical_by_id = {
        id(parameter): (name, owner)
        for owner, entries in (
            (
                "camera",
                backend.configure_mosder_stage(
                    "FROZEN"
                ).owner_configuration.owners.camera,
            ),
            (
                "object",
                backend.configure_mosder_stage(
                    "FROZEN"
                ).owner_configuration.owners.object,
            ),
            (
                "shared",
                backend.configure_mosder_stage(
                    "FROZEN"
                ).owner_configuration.owners.shared,
            ),
            (
                "residual",
                backend.configure_mosder_stage(
                    "FROZEN"
                ).owner_configuration.owners.residual,
            ),
        )
        for name, parameter in entries
    }
    if set(logical_by_id) != {id(parameter) for _, parameter in method}:
        raise ManifestContractError("method owner partition is incomplete")
    active_by_stage: dict[str, set[int]] = {}
    for stage in ("F", "G", "R"):
        configuration = backend.configure_mosder_stage(stage)
        active_by_stage[stage] = {
            id(parameter)
            for name, parameter in configuration.owner_configuration.owners.all
            if name in configuration.owner_configuration.trainable_names
        }
    backend.configure_mosder_stage("FROZEN")
    rows: list[Mapping[str, Any]] = []
    seen_method: set[int] = set()
    for model_name, parameter in backend.model.named_parameters():
        identity = id(parameter)
        if identity in logical_by_id:
            logical_name, owner = logical_by_id[identity]
            seen_method.add(identity)
        else:
            logical_name, owner = model_name, "frozen_base"
        rows.append(
            {
                "model_name": model_name,
                "logical_name": logical_name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "numel": parameter.numel(),
                "owner": owner,
                "initialization_rule": _initialization_rule(owner, logical_name),
                "trainable_stages": [
                    stage
                    for stage in ("F", "G", "R")
                    if identity in active_by_stage[stage]
                ],
            }
        )
    if seen_method != set(logical_by_id):
        raise ManifestContractError(
            "one or more method parameters are not in live model"
        )
    owner_counts = {
        owner: {
            "tensor_count": sum(row["owner"] == owner for row in rows),
            "parameter_count": sum(
                int(row["numel"]) for row in rows if row["owner"] == owner
            ),
        }
        for owner in ("camera", "object", "shared", "residual", "frozen_base")
    }
    method_state = method_parameter_state(method)
    payload = {
        "parameters": rows,
        "owner_counts": owner_counts,
        "total_tensor_count": len(rows),
        "total_parameter_count": sum(int(row["numel"]) for row in rows),
        "method_initial_state_sha256": tensor_state_sha256(method_state),
    }
    return {**payload, "parameter_manifest_sha256": canonical_sha256(payload)}


def runtime_environment_manifest() -> Mapping[str, Any]:
    packages: dict[str, str | None] = {}
    for name in (
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "numpy",
        "safetensors",
        "pillow",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory": properties.total_memory,
                    "major": properties.major,
                    "minor": properties.minor,
                    "multi_processor_count": properties.multi_processor_count,
                }
            )
    payload = {
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "packages": packages,
        "torch": {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": cuda_devices,
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "numeric_environment": numeric_environment_snapshot(),
        },
    }
    return {**payload, "environment_sha256": canonical_sha256(payload)}


def build_complete_manifest(
    backend: Any,
    *,
    code_paths: Sequence[Path],
) -> Mapping[str, Any]:
    binding = getattr(backend, "binding", None)
    model_root = getattr(binding, "model_root", None)
    if not isinstance(model_root, Path):
        raise ManifestContractError("backend model root is unavailable")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "family": str(getattr(binding, "key", "")),
        "parameters": parameter_manifest(backend),
        "model_artifacts": file_tree_manifest(model_root),
        "code": code_manifest(code_paths),
        "environment": runtime_environment_manifest(),
        "authority": {
            "formal_training_authorized": False,
            "hold_status": HOLD_STATUS,
            "confirmation_a_opened": False,
            "final_b_opened": False,
            "held_roles_opened": False,
        },
    }
    reference_root = getattr(binding, "reference_code_root", None)
    payload["reference_code"] = (
        file_tree_manifest(reference_root) if isinstance(reference_root, Path) else None
    )
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


__all__ = [
    "ManifestContractError",
    "SCHEMA_VERSION",
    "build_complete_manifest",
    "code_manifest",
    "file_tree_manifest",
    "parameter_manifest",
    "runtime_environment_manifest",
    "sha256_file",
]
