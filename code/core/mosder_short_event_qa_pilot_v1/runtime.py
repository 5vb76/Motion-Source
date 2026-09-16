"""Load raw or MoSDeR inference backends from training checkpoints.

Model loading is deferred until ``load_runtime`` is called, so importing
this module does not initialize CUDA."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contract import ARMS


FAMILY_DTYPES = {
    "qwen3_vl_8b": "bfloat16",
    "molmo2_o_7b": "bfloat16",
    "nvila_lite_8b": "float16",
}
FORMAL_COMPLETION_SCHEMA = "mosder_compute65_f1_g1_r1_orchestrator_v1"
FORMAL_COMPLETION_STATUS = "PASS_COMPUTE65_F1_G1_R1_TRAINING_AND_R_SELECTION"
RECOVERY_COMPLETION_SCHEMA = "mosder_compute65_validation_recovery_completion_v1"
RECOVERY_COMPLETION_STATUS = "PASS_COMPUTE65_SOURCE_F1_G1_PLUS_RECOVERED_R1_VALIDATION_SELECTION_AND_COMPLETE_STEP_METRICS"
EXPECTED_R_CALLS = (
    {"stage": "R", "epoch_one_based": 0, "kind": "r_step0_parent_g"},
    {"stage": "R", "epoch_one_based": 1, "kind": "trained"},
)


class ShortEventRuntimeError(RuntimeError):
    """A local model or frozen training checkpoint failed validation."""


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    family: str
    arm: str
    dtype: str
    device: str
    checkpoint_sha256: str | None
    training_protocol_sha256: str | None
    training_completion_file_sha256: str | None
    qwen_video_budget_tier: str
    qwen_video_budget_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "arm": self.arm,
            "dtype": self.dtype,
            "device": self.device,
            "checkpoint_sha256": self.checkpoint_sha256,
            "training_protocol_sha256": self.training_protocol_sha256,
            "training_completion_file_sha256": self.training_completion_file_sha256,
            "qwen_video_budget_tier": self.qwen_video_budget_tier,
            "qwen_video_budget_reason": self.qwen_video_budget_reason,
        }


@dataclass(slots=True)
class LoadedRuntime:
    backend: Any
    identity: RuntimeIdentity

    def close(self) -> None:
        closer = getattr(self.backend, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> "LoadedRuntime":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ShortEventRuntimeError(f"{label} must be an absolute regular file")
    try:
        value = json.loads(path.read_bytes())
    except Exception as error:
        raise ShortEventRuntimeError(f"{label} JSON is invalid") from error
    if not isinstance(value, Mapping):
        raise ShortEventRuntimeError(f"{label} root is invalid")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_completion(
    value: Mapping[str, Any],
    *,
    family: str,
    protocol_sha256: str,
    checkpoint_sha256: str,
) -> None:
    common = (
        value.get("family") == family
        and value.get("protocol_sha256") == protocol_sha256
        and value.get("selected_final_checkpoint_sha256") == checkpoint_sha256
        and value.get("confirmation_a_opened") is False
        and value.get("final_b_opened") is False
        and value.get("held_roles_opened") is False
    )
    if not common:
        raise ShortEventRuntimeError(
            "training completion does not bind the selected final checkpoint"
        )
    schema = value.get("schema_version")
    status = value.get("status")
    if schema == FORMAL_COMPLETION_SCHEMA:
        if status != FORMAL_COMPLETION_STATUS:
            raise ShortEventRuntimeError("formal completion status differs")
        if tuple(value.get("formal_validation_call_sequence", ())) != EXPECTED_R_CALLS:
            raise ShortEventRuntimeError("formal Stage-R validation sequence differs")
        stages = value.get("stages")
        stage_r = stages.get("R") if isinstance(stages, Mapping) else None
        pointer = stage_r.get("best_pointer") if isinstance(stage_r, Mapping) else None
        selected = stage_r.get("selected") if isinstance(stage_r, Mapping) else None
        if not (
            isinstance(pointer, Mapping)
            and isinstance(selected, Mapping)
            and pointer.get("checkpoint_sha256") == checkpoint_sha256
            and selected.get("checkpoint_sha256") == checkpoint_sha256
        ):
            raise ShortEventRuntimeError("formal completion Stage-R selection differs")
    elif schema == RECOVERY_COMPLETION_SCHEMA:
        if not (
            status == RECOVERY_COMPLETION_STATUS
            and value.get("validation_execution_repair_only") is True
            and value.get("training_objective_changed") is False
            and value.get("legacy_gate2_result_retained") is True
            and value.get("legacy_gate2_status") == "HOLD_GATE2_V3"
        ):
            raise ShortEventRuntimeError("recovery completion contract differs")
    else:
        raise ShortEventRuntimeError("training completion schema is not allowlisted")


def load_runtime(
    *,
    family: str,
    arm: str,
    device: str,
    dtype: str | None,
    training_protocol_json: str | None = None,
    checkpoint_directory: str | None = None,
    training_completion_json: str | None = None,
    pointer_name: str = "BEST.json",
) -> LoadedRuntime:
    if arm not in ARMS:
        raise ShortEventRuntimeError("arm is invalid")
    if family not in FAMILY_DTYPES:
        raise ShortEventRuntimeError("family is invalid")
    expected_dtype = FAMILY_DTYPES[family]
    if dtype is not None and dtype != expected_dtype:
        raise ShortEventRuntimeError("dtype differs from the family-native dtype")
    dtype = expected_dtype
    qwen_budget_tier = "memory_fallback" if family == "qwen3_vl_8b" else "stock_primary"
    qwen_budget_reason = "preflight_oom" if family == "qwen3_vl_8b" else None
    model_kwargs = {
        "qwen_video_budget_tier": qwen_budget_tier,
        "qwen_video_budget_reason": qwen_budget_reason,
    }
    if pointer_name != "BEST.json":
        raise ShortEventRuntimeError("short-event pilot accepts only BEST.json")

    from family_backends_v1 import load_local_backend

    backend: Any | None = None
    checkpoint_sha: str | None = None
    protocol_sha: str | None = None
    completion_file_sha: str | None = None
    method_parameters: tuple[tuple[str, Any], ...] = ()
    try:
        if arm in {"raw_vlm", "raw_plus_predicted_state"}:
            if any(
                value is not None
                for value in (
                    training_protocol_json,
                    checkpoint_directory,
                    training_completion_json,
                )
            ):
                raise ShortEventRuntimeError("raw arms must not receive a checkpoint")
            backend = load_local_backend(
                family, device=device, dtype=dtype, **model_kwargs
            )
            backend.freeze_base()
        else:
            if (
                training_protocol_json is None
                or checkpoint_directory is None
                or training_completion_json is None
            ):
                raise ShortEventRuntimeError(
                    "MoSDeR requires protocol, checkpoint directory, and training completion"
                )
            # LOCAL_PATH: Supply matching protocol/completion JSON files and an
            # absolute checkpoint directory containing BEST.json and its weights.
            protocol_path = Path(training_protocol_json)
            directory = Path(checkpoint_directory)
            completion_path = Path(training_completion_json)
            if (
                not directory.is_absolute()
                or not directory.is_dir()
                or directory.is_symlink()
            ):
                raise ShortEventRuntimeError("checkpoint directory is invalid")
            from mosder_fgr_runner_compute65_v1.protocol import (
                run_protocol_from_mapping,
            )
            from mosder_fgr_runner_candidate_v3.checkpoint import (
                load_committed_checkpoint,
                load_method_parameter_state,
                set_global_seed,
            )
            from mosder_fgr_runner_candidate_v3.engine import all_method_parameters
            from mosder_final_v1.family_backend import load_local_mosder_backend

            protocol = run_protocol_from_mapping(
                _load_json(protocol_path, label="training protocol")
            )
            if getattr(protocol, "family", None) != family:
                raise ShortEventRuntimeError("training protocol family differs")
            protocol_sha = str(protocol.identity_sha256)
            checkpoint, pointer = load_committed_checkpoint(
                directory, pointer_name=pointer_name, protocol=protocol
            )
            if pointer.get("stage") != "R":
                raise ShortEventRuntimeError(
                    "FULL_LANGUAGE requires selected Stage-R checkpoint"
                )
            checkpoint_sha = str(pointer["checkpoint_sha256"])
            completion = _load_json(completion_path, label="training completion")
            completion_file_sha = _file_sha256(completion_path)
            _validate_completion(
                completion,
                family=family,
                protocol_sha256=protocol_sha,
                checkpoint_sha256=checkpoint_sha,
            )
            set_global_seed(int(protocol.seed))
            backend = load_local_mosder_backend(
                family, device=device, dtype=dtype, **model_kwargs
            )
            method_parameters = all_method_parameters(backend)
            load_method_parameter_state(
                method_parameters, checkpoint["method_parameter_state"]
            )
            backend.configure_mosder_stage("FROZEN")
        backend.model.eval()
        parameters = tuple(backend.model.parameters())
        if any(
            parameter.requires_grad or parameter.grad is not None
            for parameter in parameters
        ):
            raise ShortEventRuntimeError("loaded runtime is not frozen evaluation")
        if any(
            parameter.requires_grad or parameter.grad is not None
            for _name, parameter in method_parameters
        ):
            raise ShortEventRuntimeError("MoSDeR method parameters are not frozen")
        return LoadedRuntime(
            backend=backend,
            identity=RuntimeIdentity(
                family=family,
                arm=arm,
                dtype=dtype,
                device=device,
                checkpoint_sha256=checkpoint_sha,
                training_protocol_sha256=protocol_sha,
                training_completion_file_sha256=completion_file_sha,
                qwen_video_budget_tier=qwen_budget_tier,
                qwen_video_budget_reason=qwen_budget_reason,
            ),
        )
    except Exception:
        if backend is not None:
            closer = getattr(backend, "close", None)
            if callable(closer):
                closer()
        raise


__all__ = [
    "LoadedRuntime",
    "RuntimeIdentity",
    "ShortEventRuntimeError",
    "FAMILY_DTYPES",
    "load_runtime",
]
