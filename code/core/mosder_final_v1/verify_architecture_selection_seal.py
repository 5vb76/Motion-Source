"""Read-only verifier for the MoSDeR-v1 architecture-selection seal.

The verifier reads only explicitly listed architecture/evidence files and the
three Train-only engineering receipts. It does not load a VLM, run training,
or discover/open any held role.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
# LOCAL_PATH: Seal entries, including experiments/... files, resolve from this root.
REPOSITORY_ROOT = PACKAGE_DIR.parents[2]
# LOCAL_PATH: Supply both metadata JSON files beside this module before verification.
SEAL_PATH = PACKAGE_DIR / "ARCHITECTURE_SELECTION_SEAL.json"
AGGREGATE_PATH = PACKAGE_DIR / "REAL_FGR_CONNECTIVITY_SMOKE_AGGREGATE_V2.json"
PASS_STATUS = "PASS_TRAIN_ONLY_REAL_FGR_CONNECTIVITY_NOT_ACCURACY_NOT_FORMAL"
EXPECTED_FAMILIES = ("qwen3_vl_8b", "molmo2_o_7b", "nvila_lite_8b")
EXPECTED_ROUTES = {
    "CAMERA_FACTOR": 4,
    "OBJECT_FACTOR": 8,
    "FULL_LANGUAGE": 3,
}
# LOCAL_PATH: Requires this external loader layout under REPOSITORY_ROOT.
TRAIN_SAMPLE_LOADER_KEY = (
    "experiments/tst_plugin_20f_small_v1/tst_plugin_20f_video_input_v1.py"
)


class SealVerificationError(RuntimeError):
    """The selected architecture or its scoped evidence drifted."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise SealVerificationError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise SealVerificationError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except Exception as error:
        raise SealVerificationError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SealVerificationError(message)


def _gradient_pair(stage: dict[str, Any]) -> list[int]:
    audit = stage.get("gradient_audit")
    _require(isinstance(audit, dict), "stage gradient audit is absent")
    _require(
        audit.get("finite_gradient_tensors") == audit.get("allowed_parameter_tensors"),
        "an allowed stage gradient is non-finite or absent",
    )
    _require(
        audit.get("foreign_gradient_tensors") == 0,
        "legacy outside-allowlist gradient count is nonzero",
    )
    _require(
        audit.get("outside_stage_allowlist_gradient_tensors") == 0,
        "outside-stage-allowlist gradient count is nonzero",
    )
    return [
        audit.get("nonzero_gradient_tensors"),
        audit.get("allowed_parameter_tensors"),
    ]


def _verify_receipt(
    summary: dict[str, Any],
    *,
    expected_backend_sha256: str,
    expected_smoke_sha256: str,
    sealed_hashes: dict[str, str],
) -> tuple[str, str]:
    # LOCAL_PATH: Aggregate records must point to readable local receipt files.
    path = Path(summary["receipt_path"])
    _require(
        _sha256(path) == summary["receipt_sha256"], f"receipt SHA-256 drifted: {path}"
    )
    receipt = _load_json(path)
    family = summary["family"]
    _require(receipt.get("family") == family, f"receipt family drifted: {family}")
    _require(receipt.get("status") == PASS_STATUS, f"receipt did not PASS: {family}")
    _require(
        receipt.get("schema_version") == "mosder_real_fgr_connectivity_smoke_v2",
        f"receipt schema drifted: {family}",
    )
    _require(
        summary.get("receipt_status") == PASS_STATUS,
        f"aggregate receipt status drifted: {family}",
    )
    code = receipt.get("code", {})
    _require(
        code.get("backend_sha256") == expected_backend_sha256,
        f"receipt backend binding drifted: {family}",
    )
    _require(
        code.get("smoke_sha256") == expected_smoke_sha256,
        f"receipt smoke binding drifted: {family}",
    )

    scope = receipt.get("scope", {})
    for key in (
        "accuracy_computed",
        "formal_training_started",
        "formal_validation_opened",
        "confirmation_a_opened",
        "final_b_opened",
        "held_roles_opened",
    ):
        _require(scope.get(key) is False, f"receipt isolation drifted: {family}/{key}")
    _require(
        scope.get("hold_status") == "HOLD_GATE2_V3", f"receipt HOLD drifted: {family}"
    )
    sample = receipt.get("sample", {})
    _require(
        sample.get("scope") == "train_only_engineering_sample",
        f"receipt sample is not Train-only engineering: {family}",
    )
    _require(
        sample.get("sealed_roles_opened") == [],
        f"receipt opened a sealed role: {family}",
    )
    loader_path = Path(sample.get("old_20f_loader", "")).resolve()
    expected_loader_path = (REPOSITORY_ROOT / TRAIN_SAMPLE_LOADER_KEY).resolve()
    _require(
        loader_path == expected_loader_path,
        f"Train-only sample loader path drifted: {family}",
    )
    _require(
        sample.get("old_20f_loader_sha256")
        == sealed_hashes.get(TRAIN_SAMPLE_LOADER_KEY)
        == _sha256(expected_loader_path),
        f"Train-only sample loader hash drifted: {family}",
    )

    runtime = receipt.get("runtime_dependency_manifest")
    _require(isinstance(runtime, dict), f"runtime manifest is absent: {family}")
    runtime_sha256 = runtime.get("sha256")
    runtime_payload = {key: value for key, value in runtime.items() if key != "sha256"}
    canonical_runtime = json.dumps(
        runtime_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    _require(
        hashlib.sha256(canonical_runtime).hexdigest() == runtime_sha256,
        f"runtime manifest payload hash drifted: {family}",
    )
    runtime_files = runtime.get("files")
    _require(
        isinstance(runtime_files, list) and len(runtime_files) == 16,
        f"runtime manifest file count drifted: {family}",
    )
    runtime_seen: set[str] = set()
    # LOCAL_PATH: Receipts also bind local runtime files beneath REPOSITORY_ROOT.
    for runtime_record in runtime_files:
        runtime_path = Path(runtime_record.get("path", "")).resolve()
        _require(
            runtime_path.is_relative_to(REPOSITORY_ROOT.resolve()),
            f"runtime dependency escapes repository: {family}/{runtime_path}",
        )
        relative = str(runtime_path.relative_to(REPOSITORY_ROOT.resolve()))
        _require(
            relative not in runtime_seen and relative in sealed_hashes,
            f"runtime dependency is duplicate or unsealed: {family}/{relative}",
        )
        runtime_seen.add(relative)
        bound_hash = runtime_record.get("sha256")
        _require(
            bound_hash == sealed_hashes[relative] == _sha256(runtime_path),
            f"runtime dependency differs from current seal: {family}/{relative}",
        )

    _require(
        receipt.get("route_success_counts") == EXPECTED_ROUTES,
        f"route counts drifted: {family}",
    )
    _require(
        receipt.get("route_success_counts_after_owner_probe")
        == {
            "CAMERA_FACTOR": 5,
            "OBJECT_FACTOR": 9,
            "FULL_LANGUAGE": 3,
        },
        f"post-probe route counts drifted: {family}",
    )
    _require(
        receipt.get("frozen_base_unchanged") is True,
        f"frozen native base changed: {family}",
    )
    precision = receipt.get("precision", {})
    _require(
        precision.get("gradient_checkpointing") is False,
        f"gradient checkpointing was enabled: {family}",
    )
    _require(
        precision.get("source_decision_residual_master_dtype") == "float32",
        f"FP32 method island drifted: {family}",
    )

    stages = receipt.get("stages", {})
    _require(
        stages.get("F", {}).get("maximum_replay_score_error") == 0.0,
        f"Stage-F replay was not exact: {family}",
    )
    observed_pairs = {
        "F_nonzero_allowed": _gradient_pair(stages["F"]),
        "G_nonzero_allowed": _gradient_pair(stages["G"]),
        "R_first_nonzero_allowed": _gradient_pair(stages["R_first"]),
        "R_second_nonzero_allowed": _gradient_pair(stages["R_second"]),
    }
    _require(
        observed_pairs == summary.get("owner_gradient_tensors"),
        f"owner gradient summary drifted: {family}",
    )
    _require(
        summary.get("outside_stage_allowlist_gradient_tensors") == 0,
        f"aggregate outside-allowlist gradient summary drifted: {family}",
    )
    _require(
        summary.get("maximum_stage_f_replay_score_error") == 0.0,
        f"aggregate Stage-F replay summary drifted: {family}",
    )

    probe = receipt.get("factor_route_owner_probe", {})
    probe_summary = summary.get("factor_route_owner_probe", {})
    expected_probe_delta = {
        "CAMERA_FACTOR": 1,
        "OBJECT_FACTOR": 1,
        "FULL_LANGUAGE": 0,
    }
    _require(
        probe.get("route_call_delta") == expected_probe_delta,
        f"factor route-owner probe counts drifted: {family}",
    )
    _require(
        probe_summary.get("route_call_delta") == expected_probe_delta,
        f"aggregate route-owner probe counts drifted: {family}",
    )
    for owner in ("camera", "object"):
        route_probe = probe.get("routes", {}).get(owner, {})
        _require(
            route_probe.get("opposite_owner_gradient_tensors") == 0,
            f"opposite-owner route gradient observed: {family}/{owner}",
        )
        _require(
            route_probe.get("outside_same_owner_gradient_tensors") == 0,
            f"outside-owner route gradient observed: {family}/{owner}",
        )
        _require(
            route_probe.get("same_owner_nonzero_gradient_tensors")
            == probe_summary.get(f"{owner}_same_owner_nonzero_parameter_tensors"),
            f"same-owner route gradient summary drifted: {family}/{owner}",
        )
        _require(
            probe_summary.get(f"{owner}_opposite_owner_gradient_tensors") == 0,
            f"aggregate opposite-owner route gradient drifted: {family}/{owner}",
        )

    plan = receipt.get("language_plan", {})
    spans = plan.get("spans")
    _require(
        isinstance(spans, list) and len(spans) == 4,
        f"language span plan is incomplete: {family}",
    )
    _require(
        [span[0] for span in spans] == ["camera", "object", "state", "description"],
        f"language span order drifted: {family}",
    )
    _require(
        spans[0][1] == 0
        and all(left[2] == right[1] for left, right in zip(spans, spans[1:])),
        f"language spans are not contiguous: {family}",
    )
    _require(
        spans[-1][2] == len(plan.get("answer_token_ids", [])),
        f"language spans do not cover the answer: {family}",
    )
    _require(
        spans == summary.get("language_spans")
        and plan.get("sha256") == summary.get("language_plan_sha256"),
        f"language plan summary drifted: {family}",
    )

    _require(
        receipt.get("gpu_peak_allocated_gib") == summary.get("gpu_peak_allocated_gib"),
        f"peak-memory summary drifted: {family}",
    )
    _require(
        receipt.get("elapsed_seconds") == summary.get("elapsed_seconds"),
        f"elapsed-time summary drifted: {family}",
    )
    return code.get("method_spec_sha256", ""), runtime_sha256


def verify() -> dict[str, Any]:
    seal = _load_json(SEAL_PATH)
    _require(
        seal.get("schema_version") == "mosder_architecture_selection_seal_v1",
        "seal schema drifted",
    )
    _require(seal.get("public_method_name") == "MoSDeR", "public method name drifted")
    _require(
        seal.get("architecture_version") == "MoSDeR-v1", "architecture version drifted"
    )
    _require(
        seal.get("status")
        == "CURRENT_FINAL_ARCHITECTURE_SELECTED_HASH_SEALED_NOT_EMPIRICALLY_FINAL",
        "architecture-selection status drifted",
    )
    selection = seal.get("selection_semantics", {})
    _require(
        selection.get("final_means_current_architecture_selection") is True,
        "final-architecture selection semantics drifted",
    )
    _require(
        selection.get("paper_final_empirical_method") is False,
        "seal claimed a paper-final empirical method",
    )
    _require(
        selection.get("historical_candidate_retroactively_renamed") is False,
        "seal retroactively renamed the historical candidate",
    )
    _require(
        selection.get("historical_candidate_promoted") is False,
        "seal promoted the historical NonPromoted candidate",
    )
    scope = seal.get("scope", {})
    _require(
        scope.get("formal_training_authorized") is False,
        "seal authorized formal training",
    )
    _require(
        scope.get("formal_validation_authorized") is False,
        "seal authorized formal validation",
    )
    _require(
        scope.get("formal_runner_sealed") is False,
        "seal falsely closed the formal runner",
    )
    _require(
        scope.get("formal_protocol_and_seed_list_sealed") is False,
        "seal falsely closed the formal protocol/seed list",
    )
    _require(
        scope.get("formal_data_roles_sealed_or_released") is False,
        "seal falsely released formal data roles",
    )
    _require(
        scope.get("train_only_engineering_sample_loader_sealed") is True,
        "seal omitted the Train-only engineering sample loader",
    )
    _require(
        scope.get("family_runtime_implementation_trees_and_python_environments_sealed")
        is False,
        "seal falsely claimed external family runtimes/environments",
    )
    _require(scope.get("confirmation_a_opened") is False, "seal opened Confirmation-A")
    _require(scope.get("final_b_opened") is False, "seal opened Final-B")
    _require(scope.get("held_roles_opened") is False, "seal opened held roles")
    _require(scope.get("hold_status") == "HOLD_GATE2_V3", "seal changed HOLD")

    files = seal.get("architecture_files")
    _require(
        isinstance(files, list) and files, "sealed architecture file list is empty"
    )
    seen: set[str] = set()
    sealed_hashes: dict[str, str] = {}
    for record in files:
        relative = record.get("path")
        expected = record.get("sha256")
        _require(
            isinstance(relative, str) and relative not in seen,
            "sealed architecture path is invalid or duplicated",
        )
        seen.add(relative)
        _require(
            isinstance(expected, str) and len(expected) == 64,
            f"sealed architecture hash is invalid: {relative}",
        )
        sealed_hashes[relative] = expected
        path = REPOSITORY_ROOT / relative
        _require(
            path.resolve().is_relative_to(REPOSITORY_ROOT.resolve()),
            f"sealed architecture path escapes repository: {relative}",
        )
        _require(
            _sha256(path) == expected, f"sealed architecture file drifted: {relative}"
        )

    aggregate_record = seal.get("real_smoke_aggregate", {})
    _require(
        Path(aggregate_record.get("path", "")).resolve() == AGGREGATE_PATH.resolve(),
        "aggregate path drifted",
    )
    _require(
        _sha256(AGGREGATE_PATH) == aggregate_record.get("sha256"),
        "aggregate SHA-256 drifted",
    )
    aggregate = _load_json(AGGREGATE_PATH)
    _require(
        aggregate.get("status")
        == "PASS_THREE_FAMILY_TRAIN_ONLY_REAL_FGR_CONNECTIVITY_NOT_ACCURACY_NOT_FORMAL",
        "three-family aggregate did not PASS",
    )
    aggregate_scope = aggregate.get("scope", {})
    for key in (
        "accuracy_computed",
        "natural_language_usability_established",
        "source_disentanglement_established",
        "formal_training_started",
        "formal_validation_opened",
        "confirmation_a_opened",
        "final_b_opened",
        "held_roles_opened",
    ):
        _require(
            aggregate_scope.get(key) is False,
            f"aggregate negative scope drifted: {key}",
        )
    families = aggregate.get("families")
    _require(
        isinstance(families, list)
        and tuple(item.get("family") for item in families) == EXPECTED_FAMILIES,
        "aggregate family order/content drifted",
    )
    # LOCAL_PATH: These verification inputs require the stated layout under the root.
    backend_key = (
        "experiments/tst_native_adapter_o18_sandbox_v1/"
        "mosder_final_v1/family_backend.py"
    )
    smoke_key = "experiments/tst_native_adapter_o18_sandbox_v1/smoke_mosder_real_v1.py"
    _require(
        backend_key in sealed_hashes and smoke_key in sealed_hashes,
        "sealed real-smoke code dependencies are absent",
    )
    receipt_evidence = {
        _verify_receipt(
            item,
            expected_backend_sha256=sealed_hashes[backend_key],
            expected_smoke_sha256=sealed_hashes[smoke_key],
            sealed_hashes=sealed_hashes,
        )
        for item in families
    }
    method_spec_hashes = {value[0] for value in receipt_evidence}
    runtime_manifest_hashes = {value[1] for value in receipt_evidence}
    _require(
        method_spec_hashes == {seal.get("real_smoke_run_method_spec_sha256")},
        "real-smoke run-time METHOD_SPEC binding drifted",
    )
    aggregate_runtime = aggregate.get("runtime_dependency_manifest", {})
    _require(
        runtime_manifest_hashes
        == {seal.get("real_smoke_runtime_manifest_sha256")}
        == {aggregate_runtime.get("sha256")},
        "real-smoke exact-current runtime manifest drifted",
    )
    _require(
        aggregate_runtime.get("identical_across_all_three_receipts") is True,
        "aggregate did not bind one common runtime manifest",
    )

    failure_summary = aggregate.get("preserved_failure", {})
    failure_path = Path(failure_summary.get("receipt_path", ""))
    _require(
        _sha256(failure_path) == failure_summary.get("receipt_sha256"),
        "preserved Qwen OOM receipt drifted",
    )
    failure = _load_json(failure_path)
    _require(
        failure.get("status")
        == "FAIL_EXPECTED_PRESERVED_QWEN_STOCK_PRIMARY_BACKWARD_OOM",
        "preserved Qwen stock-primary failure status drifted",
    )
    _require(
        failure.get("scope", {}).get("held_roles_opened") is False,
        "preserved Qwen failure opened held roles",
    )

    return {
        "status": "PASS_MOSDER_V1_ARCHITECTURE_SELECTION_SEAL",
        "public_method_name": "MoSDeR",
        "architecture_version": "MoSDeR-v1",
        "architecture_files_verified": len(files),
        "real_family_receipts_verified": len(families),
        "exact_current_runtime_manifests_verified": len(families),
        "preserved_failures_verified": 1,
        "formal_training_authorized": False,
        "hold_status": "HOLD_GATE2_V3",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
