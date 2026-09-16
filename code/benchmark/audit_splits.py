"""Audit split overlap, source-media hashes, and snapshot integrity."""

import hashlib
import itertools
import json
import re
from pathlib import Path

# LOCAL_PATH: parents[1] resolves to code/ here; set the benchmark data root
# before using the release/, candidates/, or raw/ subdirectories.
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    splits = {
        split_name: read_jsonl(BENCHMARK_ROOT / f"release/{split_name}.jsonl")
        for split_name in ["train", "val", "test"]
    }
    report = {"pairs": {}, "failures": []}

    def frame_fingerprints(row):
        model_input = row["model_input"]
        if row["source"] == "AV2":
            return [("JPEG", frame["sha256"]) for frame in model_input["rgb20_frames"]]
        if row["source"] == "TACO-V1-allocentric":
            return [
                ("TACO", model_input["rgb20"]["formal_RGB_container"]["sha256"], i)
                for i in model_input["rgb20"]["source_frame_ordinals"]
            ]
        return [
            ("RGB", frame["pixel_rgb_sha256"])
            for frame in model_input["rgb20"]["frames"]
        ]

    for left_split, right_split in itertools.combinations(splits, 2):
        pair_report = {}
        for key in [
            "case_id",
            "physical_window_id",
            "sequence",
            "split_group",
            "target_id",
        ]:
            overlap = {(row["source"], row[key]) for row in splits[left_split]} & {
                (row["source"], row[key]) for row in splits[right_split]
            }
            pair_report[key] = {
                "count": len(overlap),
                "examples": [list(x) for x in sorted(overlap)[:25]],
            }
            if overlap and key != "target_id":
                report["failures"].append(f"{left_split}/{right_split} {key}")
        left_frames = {
            frame for row in splits[left_split] for frame in frame_fingerprints(row)
        }
        right_frames = {
            frame for row in splits[right_split] for frame in frame_fingerprints(row)
        }
        overlap = left_frames & right_frames
        pair_report["any_identical_frame_content"] = {
            "count": len(overlap),
            "scope": "native RGB pixel hash ADT/HOT; encoded JPEG hash AV2; video hash plus ordinal TACO",
        }
        if overlap:
            report["failures"].append(f"{left_split}/{right_split} frame content")

        def adt_activity_families(rows):
            return {
                re.sub(r"_seq\d+_\d+$", "", row["sequence"])
                for row in rows
                if row["source"] == "ADT-LiteOffice"
            }

        overlap = adt_activity_families(splits[left_split]) & adt_activity_families(
            splits[right_split]
        )
        pair_report["ADT_activity_family"] = {
            "count": len(overlap),
            "examples": sorted(overlap),
        }
        if overlap:
            report["failures"].append(f"{left_split}/{right_split} ADT family")
        report["pairs"][left_split + "__" + right_split] = pair_report
    protocol = json.loads(
        (BENCHMARK_ROOT / "candidates/selection_protocol.json").read_text()
    )
    assert not {
        row["split_group"] for row in splits["train"] if row["source"] == "AV2"
    } & set(protocol["held_out_av2_logs"])
    assert not {
        row["split_group"]
        for row in splits["train"]
        if row["source"] == "TACO-V1-allocentric"
    } & set(protocol["held_out_taco_capture_days"])
    # Verify every AV2 media file against existing verified media receipts, including expanded targets.
    av2_media = {
        frame["path"]: frame["sha256"]
        for split_rows in splits.values()
        for row in split_rows
        if row["source"] == "AV2"
        for frame in row["model_input"]["rgb20_frames"]
    }
    for path, expected in av2_media.items():
        assert file_sha256(Path(path)) == expected, path
    report["AV2_unique_media_files_full_hash_verified"] = len(av2_media)
    report["native_source_registry_full_hash_verified"] = 0
    for path, ref in json.loads(
        (BENCHMARK_ROOT / "release/native_source_registry.json").read_text()
    ).items():
        assert file_sha256(Path(path)) == ref["sha256"], path
        report["native_source_registry_full_hash_verified"] += 1
    # LOCAL_PATH: Requires the sibling inventory directory and its referenced files.
    legacy_root = BENCHMARK_ROOT.with_name("old_own_benchmark")
    legacy_inventory = json.loads((legacy_root / "inventory.json").read_text())
    for split_name, row in legacy_inventory["splits"].items():
        assert file_sha256(legacy_root / row["snapshot"]) == row["sha256"]
    report["old_snapshots_unchanged"] = True
    report["target_identity_note"] = (
        "Shared HOT3D/TACO object identities are allowed; split is by capture group, not unseen object identity."
    )
    report["status"] = "PASS" if not report["failures"] else "FAIL"
    (BENCHMARK_ROOT / "release/split_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    assert not report["failures"]


if __name__ == "__main__":
    main()
