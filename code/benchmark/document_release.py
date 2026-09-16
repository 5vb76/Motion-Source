"""Write release configuration, source-prior diagnostics, and provenance receipts."""

import collections
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

# LOCAL_PATH: parents[1] resolves to code/ here; set the benchmark data root
# before using the release/, candidates/, or raw/ subdirectories.
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
# LOCAL_PATH: Sibling data directories supply inventories and integrity records.
LEGACY_ROOT = BENCHMARK_ROOT.with_name("old_own_benchmark")
PREVIOUS_RELEASE_ROOT = BENCHMARK_ROOT.with_name("general_motion_benchmark_v1")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def file_sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    # LOCAL_PATH: Source media archives used to populate the inventory.
    raw_reserve_root = Path("/root/autodl-tmp/tst_plugin_formal_data_v1")
    source_registry = json.loads(
        (BENCHMARK_ROOT / "release/native_source_registry.json").read_text()
    )
    # Fail before writing provenance if the scan receipt cannot be read.
    json.loads((BENCHMARK_ROOT / "candidates/adt_testA_scan_receipt.json").read_text())
    files = []
    for role in ["testA", "testB"]:
        for source, pattern in [
            ("ADT-LiteOffice", "raw_sealed/*/*"),
            ("HOT3D", "raw_clips/*/*.tar"),
        ]:
            for path in sorted((raw_reserve_root / role / source).glob(pattern)):
                if not path.is_file():
                    continue
                entry = {
                    "original_role": role,
                    "source": source,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "ready_legacy_RGB20_manifest": False,
                }
                if str(path) in source_registry:
                    entry["sha256"] = source_registry[str(path)]["sha256"]
                if source == "HOT3D":
                    entry["native_image_stream"] = (
                        "214-1 RGB" if role == "testA" else "1201-1/1201-2 monochrome"
                    )
                files.append(entry)
    write_json(
        LEGACY_ROOT / "legacy_test_reserves.json",
        {
            "added_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Supplement the original snapshot inventory with previously unarchived raw test reserves; original train/validation distributions and snapshots remain unchanged.",
            "original_testA_ADT_sequences": 11,
            "original_testA_HOT_RGB_clips": 19,
            "original_testB_HOT_monochrome_clips": 40,
            "legacy_ready_test_manifest_found": False,
            "other_reserves": "HOI4D raw archives/status files exist but are not qualified for this RGB20 physical release.",
            "current_access": "User explicitly authorized reuse of original validation/test data. ADT testA groundtruth archives and HOT3D testA annotations/images were opened for V2 construction; HOT3D testB stream availability was inspected. Historical SEALED_STATUS files are preserved as historical records, not current unopened claims.",
            "ADT_archive_hash_receipt": str(
                BENCHMARK_ROOT / "candidates/adt_testA_scan_receipt.json"
            ),
            "files": files,
        },
    )
    rows = {
        split_name: [
            json.loads(line)
            for line in (BENCHMARK_ROOT / f"release/{split_name}.jsonl").open()
        ]
        for split_name in ["train", "val", "test"]
    }
    majority = {
        source: collections.Counter(
            row["state"] for row in rows["train"] if row["source"] == source
        ).most_common(1)[0][0]
        for source in {row["source"] for row in rows["train"]}
    }
    write_json(
        BENCHMARK_ROOT / "release/source_prior_diagnostic.json",
        {
            "method": "Fit majority state per source on new train labels; evaluate fixed rule on val/test. No trained video model.",
            "source_prediction": majority,
            "accuracy": {
                split_name: sum(
                    row["state"] == majority[row["source"]] for row in rows[split_name]
                )
                / len(rows[split_name])
                for split_name in ["val", "test"]
            },
            "chance_for_balanced_four_states": 0.25,
            "interpretation": "Source/state confounding remains despite marginal balancing; always report per-source and source-macro results.",
        },
    )
    write_json(
        BENCHMARK_ROOT / "release/dataset_config.json",
        {
            "benchmark": "general_motion_benchmark_v2",
            "train_manifest": str(BENCHMARK_ROOT / "release/train.jsonl"),
            "val_manifest": str(BENCHMARK_ROOT / "release/val.jsonl"),
            "test_manifest": str(BENCHMARK_ROOT / "release/test.jsonl"),
            "train_sampling_weights": str(
                BENCHMARK_ROOT / "release/train_sampling_weights.jsonl"
            ),
            # LOCAL_PATH: Requires scripts/ under the data root; code lives in code/benchmark/ here.
            "loader": str(BENCHMARK_ROOT / "scripts/benchmark_loader.py"),
            "loader_function": "load_input",
            "model_input_keys": [
                "frames",
                "target_boxes_xyxy",
                "timestamps_ns",
                "prompt",
            ],
            "supervision_keys": ["camera_moving", "object_moving", "state"],
            "trajectory_gt_for_loss_or_audit_only": True,
            "derive_motion_labels_in_model_forward": False,
            "test_requires_new_split_retraining": True,
        },
    )
    write_json(
        BENCHMARK_ROOT / "release/physical_provenance.json",
        {
            "inherited_v1_provenance": {
                "path": str(PREVIOUS_RELEASE_ROOT / "release/physical_provenance.json"),
                "sha256": file_sha256(
                    PREVIOUS_RELEASE_ROOT / "release/physical_provenance.json"
                ),
            },
            "native_test_source_registry": {
                "path": str(BENCHMARK_ROOT / "release/native_source_registry.json"),
                "sha256": file_sha256(
                    BENCHMARK_ROOT / "release/native_source_registry.json"
                ),
            },
            "ADT_full_window_scan": {
                "path": str(BENCHMARK_ROOT / "candidates/adt_testA_scan_receipt.json"),
                "sha256": file_sha256(
                    BENCHMARK_ROOT / "candidates/adt_testA_scan_receipt.json"
                ),
            },
            "selection_protocol": {
                "path": str(BENCHMARK_ROOT / "candidates/selection_protocol.json"),
                "sha256": file_sha256(
                    BENCHMARK_ROOT / "candidates/selection_protocol.json"
                ),
            },
            "label_derivation": "Cached in dataset preprocessing, using full native physical trajectory windows and source-specific historical thresholds; 20-frame trajectories are aligned exports, not a new label thresholding basis.",
            "model_gt_isolation": "load_input returns video, target boxes, timestamps, prompt only. GT and cached labels stay outside forward.",
            "human_animal_policy": "Only rigid object / AV2 vehicle targets included. Nonrigid human/animal limb/head-only motion targets excluded.",
        },
    )


if __name__ == "__main__":
    main()
