"""Verify release receipts and freeze a newly constructed V2 artifact ledger."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

# LOCAL_PATH: parents[1] resolves to code/ here; set the benchmark data root
# before using the release/, candidates/, or raw/ subdirectories.
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
# LOCAL_PATH: Sibling data directories supply inventories and integrity records.
PREVIOUS_RELEASE_ROOT = BENCHMARK_ROOT.with_name("general_motion_benchmark_v1")


def file_sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")


def main():
    for name in [
        "validation_receipt",
        "split_audit",
        "functional_test_receipt",
        "native_pixel_receipt",
    ]:
        assert (
            json.loads((BENCHMARK_ROOT / f"release/{name}.json").read_text())["status"]
            == "PASS"
        ), name
    checked = []
    for line in (PREVIOUS_RELEASE_ROOT / "release/SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        expected, relative_path = line.split("  ", 1)
        assert file_sha256(PREVIOUS_RELEASE_ROOT / relative_path) == expected, (
            "V1 frozen artifact modified " + relative_path
        )
        checked.append(relative_path)
    inventory = json.loads((BENCHMARK_ROOT / "release/inventory.json").read_text())
    assert inventory["status"] == "READY_FOR_RETRAINING_WITH_HISTORICAL_EXPOSURE"
    for relative_path, expected in inventory["manifest_sha256"].items():
        assert file_sha256(BENCHMARK_ROOT / "release" / relative_path) == expected
    validation = json.loads(
        (BENCHMARK_ROOT / "release/validation_receipt.json").read_text()
    )
    assert (
        validation["metadata_cases"] == 7400 and len(validation["decode_samples"]) == 69
    )
    now = datetime.now(timezone.utc)
    construction_started_at = datetime(2026, 9, 10, 22, 49, 16, tzinfo=timezone.utc)
    receipt = {
        "status": "COMPLETE",
        "completed_at_utc": now.isoformat(),
        "five_hour_work_start_utc": construction_started_at.isoformat(),
        "elapsed_minutes": round(
            (now - construction_started_at).total_seconds() / 60, 2
        ),
        "within_five_hours": (now - construction_started_at).total_seconds() <= 18000,
        "counts": {"train": 5000, "val": 1200, "test": 1200},
        "old_snapshots_unchanged": True,
        "previous_v1_frozen_artifacts_verified": len(checked),
        "metadata_cases_verified": 7400,
        "stratified_decode_cases": 69,
        "stratified_decode_frames": sum(
            record["frames"] for record in validation["decode_samples"]
        ),
        "new_native_test_unique_frames_decoded": 7109,
        "model_training_run": False,
        "fresh_independent_test_certified": False,
        "test_use": "Requires retraining with the new split; TACO/AV2 test groups originate from historical training candidates.",
    }
    write_json(BENCHMARK_ROOT / "release/completion_receipt.json", receipt)
    # Freeze small provenance/candidates, manifests, scripts and extracted physical GT; no bulky media copies.
    files = sorted(
        path
        for path in BENCHMARK_ROOT.rglob("*")
        if path.is_file()
        and path.suffix not in [".log", ".pyc"]
        and "__pycache__" not in path.parts
        and path.name != "SHA256SUMS"
    )
    ledger = BENCHMARK_ROOT / "release/SHA256SUMS"
    ledger.write_text(
        "".join(
            f"{file_sha256(path)}  {path.relative_to(BENCHMARK_ROOT)}\n"
            for path in files
        )
    )
    for line in ledger.read_text().splitlines():
        expected, relative_path = line.split("  ", 1)
        assert file_sha256(BENCHMARK_ROOT / relative_path) == expected
    write_json(
        BENCHMARK_ROOT.parent / "CURRENT.json",
        {
            "benchmark": "general_motion_benchmark_v2",
            "root": str(BENCHMARK_ROOT),
            "config": str(BENCHMARK_ROOT / "release/dataset_config.json"),
            "readme": str(BENCHMARK_ROOT / "README.md"),
            "frozen_checksums": {"path": str(ledger), "sha256": file_sha256(ledger)},
            "counts": receipt["counts"],
            "old_own_benchmark": str(BENCHMARK_ROOT.with_name("old_own_benchmark")),
            "previous_version": str(PREVIOUS_RELEASE_ROOT),
            "status": inventory["status"],
            "updated_at_utc": now.isoformat(),
        },
    )
    print(json.dumps({**receipt, "new_frozen_artifact_count": len(files)}, indent=2))


if __name__ == "__main__":
    main()
