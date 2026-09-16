#!/usr/bin/env python3
"""Command-line preflight and training for the F/G/R protocol."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from mosder_final_v1.family_backend import load_local_mosder_backend  # noqa: E402
from mosder_final_v1.verify_architecture_selection_seal import (
    verify as verify_architecture,
)  # noqa: E402
from family_backends_v1 import discover_local_family  # noqa: E402
from mosder_consumed_field_projection_v2 import projection  # noqa: E402

from .authority import load_release_authority  # noqa: E402
from .checkpoint import (  # noqa: E402
    configure_reproducible_numeric_environment,
    set_global_seed,
)
from .data import ProjectedMoSDeRDataset  # noqa: E402
from .manifest import (  # noqa: E402
    build_complete_manifest,
    code_manifest,
    runtime_environment_manifest,
    sha256_file as manifest_sha256_file,
)
from .orchestrator import FamilyRun  # noqa: E402
from .protocol import (  # noqa: E402
    HOLD_STATUS,
    RunProtocol,
    canonical_json_bytes,
    canonical_sha256,
    run_protocol_from_mapping,
)
from .sampler import FullCoverageStateSampler  # noqa: E402


DTYPES = {
    "qwen3_vl_8b": "bfloat16",
    "molmo2_o_7b": "bfloat16",
    "nvila_lite_8b": "float16",
}
RUNNER_CODE = tuple(sorted(HERE.glob("*.py")))
# LOCAL_PATH: Supply the architecture seal JSON beside the mosder_final_v1 sources.
ARCHITECTURE_SEAL = EXPERIMENT_ROOT / "mosder_final_v1/ARCHITECTURE_SELECTION_SEAL.json"
# LOCAL_PATH: Projection receipt under the external root configured in projection.py.
PROJECTION_RECEIPT = projection.DEFAULT_OUTPUT_ROOT / projection.RECEIPT_NAME
# LOCAL_PATH: RGB20 bridge receipt produced at seal_rgb20_bridge.OUTPUT_ROOT.
BRIDGE_SEAL_RECEIPT = Path(
    "/root/autodl-tmp/tst_native_adapter_o18_sandbox_v1/"
    "MOSDER_STRICT_V3_RGB20_BRIDGE_SEAL_V2_SANITIZED_NONAUTHORIZING_HOLD_GATE2_V3/"
    "MOSDER_STRICT_V3_RGB20_BRIDGE_SEAL_RECEIPT_V2.json"
)


class RunnerContractError(RuntimeError):
    """CLI preflight or independently authorized launch failed closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    temporary.replace(path)


def cpu_preflight() -> Mapping[str, Any]:
    architecture = verify_architecture()
    train = ProjectedMoSDeRDataset("train")
    validation = ProjectedMoSDeRDataset("validation", allow_validation=True)
    samplers: dict[str, Any] = {}
    for stage, epochs in (("F", 4), ("G", 4), ("R", 2)):
        sampler = FullCoverageStateSampler(
            train.refs, seed=20260902 + ord(stage), epochs=epochs
        )
        samplers[stage] = {
            "epochs": epochs,
            "rows_per_epoch": len(train),
            "first_epoch_order_sha256": sampler.order_sha256,
            "full_coverage_without_replacement": True,
            "unequal_state_counts_supported": True,
        }
    code = code_manifest(RUNNER_CODE)
    environment = runtime_environment_manifest()
    return {
        "schema_version": "mosder_fgr_cpu_preflight_candidate_v2",
        "status": "PASS_CPU_PREFLIGHT_NONAUTHORIZING_GPU_SMOKES_STILL_REQUIRED",
        "completed_utc": _utc_now(),
        "architecture": architecture,
        "data": {
            "train": {
                "rows": len(train),
                "sha256": train.manifest_sha256,
                "states": dict(
                    sorted(Counter(ref.state for ref in train.refs).items())
                ),
                "sources": dict(
                    sorted(Counter(ref.source_dataset for ref in train.refs).items())
                ),
            },
            "validation": {
                "rows": len(validation),
                "sha256": validation.manifest_sha256,
                "scoring_started": False,
            },
        },
        "samplers": samplers,
        "runner_code": code,
        "environment": environment,
        "implementation": {
            "stages": ["F", "G", "R"],
            "stage_f_exact_two_pass_six_score": True,
            "full_coverage_state_stratified_sampler": True,
            "partial_final_accumulation_scaled_by_actual_count": True,
            "immutable_checkpoints_and_atomic_latest_best_pointers": True,
            "python_numpy_torch_cpu_cuda_rng_resume": True,
            "optimizer_scheduler_sampler_resume": True,
            "best_validation_selection": True,
            "r_step0_no_refine_candidate": True,
            "independent_release_authority_required": True,
        },
        "remaining": {
            "three_family_parameter_environment_manifests": True,
            "each_family_each_stage_real_save_reload_resume_smokes": True,
            "train_only_tiny_overfit_and_canary": True,
            "protocol_seed_seal": True,
            "independent_release_authority": True,
        },
        "authority": {
            "formal_training_authorized": False,
            "formal_validation_authorized": False,
            "hold_status": HOLD_STATUS,
            "confirmation_a_opened": False,
            "final_b_opened": False,
            "held_roles_opened": False,
        },
    }


def _load_protocol(path: Path) -> RunProtocol:
    if not path.is_file() or path.is_symlink():
        raise RunnerContractError("protocol file is absent")
    try:
        value = json.loads(path.read_bytes())
    except Exception as error:
        raise RunnerContractError("protocol JSON is invalid") from error
    return run_protocol_from_mapping(value)


def _verify_protocol_static_bindings(
    protocol: RunProtocol, discovery: Mapping[str, Any]
) -> None:
    verify_architecture()
    if (
        manifest_sha256_file(ARCHITECTURE_SEAL) != protocol.architecture_seal_sha256
        or manifest_sha256_file(PROJECTION_RECEIPT)
        != protocol.consumed_field_seal_sha256
        or manifest_sha256_file(BRIDGE_SEAL_RECEIPT)
        != protocol.rgb20_bridge_seal_sha256
        or discovery.get("artifact_control_sha256")
        != protocol.tokenizer_manifest_sha256
    ):
        raise RunnerContractError("protocol static artifact identity differs")
    try:
        receipt = json.loads(PROJECTION_RECEIPT.read_bytes())
        prompt_sha = receipt["consumed_field_contract"]["query_sha256"]
        bridge = json.loads(BRIDGE_SEAL_RECEIPT.read_bytes())
    except Exception as error:
        raise RunnerContractError("consumed-field receipt is invalid") from error
    if prompt_sha != protocol.prompt_sha256:
        raise RunnerContractError("protocol prompt identity differs")
    if not (
        bridge.get("schema_version") == "mosder_strict_v3_rgb20_bridge_seal_receipt_v2"
        and bridge.get("status")
        == (
            "PASS_MOSDER_STRICT_V3_RGB20_BRIDGE_SEAL_V2_SANITIZED_7959_1991_"
            "NONAUTHORIZING_HOLD_GATE2_V3"
        )
        and bridge.get("coverage_limitations", {}).get(
            "metadata_case_rows_resolved_and_bound"
        )
        == 9950
        and bridge.get("representative_pixel_smoke", {}).get("status")
        == "PASS_THREE_SOURCE_REPRESENTATIVE_PIXEL_SMOKE"
        and bridge.get("authority", {}).get("hold_changed") is False
        and bridge.get("authority", {}).get("underlying_gate2_status") == HOLD_STATUS
    ):
        raise RunnerContractError("RGB20 bridge seal semantics differ")


def _ensure_run_start_manifest(
    *,
    output_dir: Path,
    protocol: RunProtocol,
    authority: Mapping[str, Any],
    backend: Any,
) -> str:
    complete = build_complete_manifest(backend, code_paths=RUNNER_CODE)
    payload = {
        "schema_version": "mosder_formal_run_start_manifest_v2",
        "protocol": dict(protocol.as_dict()),
        "protocol_sha256": protocol.identity_sha256,
        "complete_runtime_model_manifest": complete,
        "complete_runtime_model_manifest_sha256": complete["manifest_sha256"],
        "release_authority": dict(authority),
        "confirmation_a_opened": False,
        "final_b_opened": False,
        "held_roles_opened": False,
    }
    encoded = canonical_json_bytes(payload) + b"\n"
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not output_dir.is_dir() or output_dir.is_symlink():
            raise RunnerContractError("formal output path is not a regular directory")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "RUN_START_MANIFEST.json"
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != encoded:
            raise RunnerContractError("existing run-start manifest differs")
    else:
        if any(output_dir.iterdir()):
            raise RunnerContractError("unsealed formal output directory is not empty")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    return manifest_sha256_file(path)


def formal_train(
    *,
    protocol_path: Path,
    authority_path: Path,
    output_dir: Path,
) -> Mapping[str, Any]:
    protocol = _load_protocol(protocol_path)
    # Reject absent/incorrect external authority before creating a CUDA
    # context, a backend, an output directory, or any data reader.
    authority = load_release_authority(authority_path, protocol)
    discovery = discover_local_family(protocol.family)
    _verify_protocol_static_bindings(protocol, discovery)
    # This must precede even ``is_available`` so cuBLAS sees its workspace
    # contract before a CUDA context can be initialized.
    configure_reproducible_numeric_environment()
    environment = runtime_environment_manifest()
    observed_runtime_sha = canonical_sha256(
        {
            "runner_code": code_manifest(RUNNER_CODE)["manifest_sha256"],
            "environment": environment["environment_sha256"],
            "static_family": discovery["artifact_control_sha256"],
        }
    )
    if protocol.runtime_manifest_sha256 != observed_runtime_sha:
        raise RunnerContractError("protocol/runtime manifest identity differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RunnerContractError(
            "one family run requires exactly one visible CUDA GPU"
        )
    # MoSDeR parameters are attached while constructing the backend, so the
    # protocol seed must be installed before that construction.
    set_global_seed(protocol.seed)
    backend = load_local_mosder_backend(
        protocol.family,
        device="cuda:0",
        dtype=DTYPES[protocol.family],
        qwen_video_budget_tier=(
            "memory_fallback" if protocol.family == "qwen3_vl_8b" else "stock_primary"
        ),
        qwen_video_budget_reason=(
            "preflight_oom" if protocol.family == "qwen3_vl_8b" else None
        ),
    )
    run: FamilyRun | None = None
    try:
        run_start_sha256 = _ensure_run_start_manifest(
            output_dir=output_dir,
            protocol=protocol,
            authority=authority,
            backend=backend,
        )
        run = FamilyRun(
            backend,
            protocol,
            output_dir=output_dir,
            release_authority_path=authority_path,
            run_start_manifest_sha256=run_start_sha256,
        )
        return run.run()
    finally:
        if run is not None:
            run.close()
        backend.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("cpu-preflight")
    preflight.add_argument("--output", type=Path)
    train = commands.add_parser("formal-train")
    train.add_argument("--protocol", type=Path, required=True)
    train.add_argument("--authority", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "cpu-preflight":
        result = cpu_preflight()
        if args.output is not None:
            _write_json(args.output, result)
    else:
        result = formal_train(
            protocol_path=args.protocol,
            authority_path=args.authority,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
