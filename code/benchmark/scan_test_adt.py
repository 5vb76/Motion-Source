"""Extract the ADT test reserve annotations and scan physical motion candidates."""

import collections
import sys
import zipfile
from pathlib import Path

# LOCAL_PATH: parents[1] resolves to code/ here; set the benchmark data root
# before using the release/, candidates/, or raw/ subdirectories.
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
# LOCAL_PATH: Requires external scan sources and the sibling scripts/ directory.
sys.path.insert(0, "/root/story2_camera_object_motion")
sys.path.insert(0, str(BENCHMARK_ROOT.parent / "general_motion_benchmark_v1/scripts"))
from build_release import (
    sha as file_sha256,
)
from build_release import (
    write as write_json,
)
from build_release import (
    write_rows as write_jsonl,
)
from scripts.scan_adt_pose_inventory_v1 import ScanConfig, scan_sequence

if __name__ == "__main__":
    # LOCAL_PATH: Provide ADT VRS/annotation archives and the label-rule JSON below.
    root = Path(
        "/root/autodl-tmp/tst_plugin_formal_data_v1/testA/ADT-LiteOffice/raw_sealed"
    )
    config_path = Path(
        "/root/story2_camera_object_motion/configs/adt_confirmation_a_pose_scan_v1.json"
    )
    config = ScanConfig.from_json(config_path)
    candidate_dir = BENCHMARK_ROOT / "candidates"
    candidate_dir.mkdir(exist_ok=True)
    rows = []
    audits = []
    sources = []
    required = [
        "aria_trajectory.csv",
        "scene_objects.csv",
        "instances.json",
        "2d_bounding_box.csv",
        "metadata.json",
    ]
    for folder in sorted(root.iterdir()):
        archive = next(folder.glob("*groundtruth.zip"))
        vrs = next(folder.glob("*.vrs"))
        extraction_dir = BENCHMARK_ROOT / "raw/ADT" / folder.name
        extraction_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as archive_file:
            for name in required:
                matches = [n for n in archive_file.namelist() if Path(n).name == name]
                if len(matches) != 1:
                    raise ValueError((name, matches))
                (extraction_dir / name).write_bytes(archive_file.read(matches[0]))
        sequence_rows, receipt = scan_sequence(extraction_dir, config)
        for row in sequence_rows:
            row["raw_vrs"] = str(vrs)
        rows += sequence_rows
        audits.append(receipt)
        sources.append(
            {
                "sequence": folder.name,
                "vrs": str(vrs),
                "groundtruth_zip": str(archive),
                "groundtruth_zip_sha256": file_sha256(archive),
                "extracted_files_sha256": {
                    str(extraction_dir / n): file_sha256(extraction_dir / n)
                    for n in required
                },
            }
        )
        print(
            folder.name,
            len(sequence_rows),
            dict(collections.Counter(row["joint_state"] for row in sequence_rows)),
            flush=True,
        )
    write_jsonl(candidate_dir / "adt_testA_physical_candidates.jsonl", rows)
    write_json(
        candidate_dir / "adt_testA_scan_receipt.json",
        {
            "candidates": len(rows),
            "states": dict(collections.Counter(row["joint_state"] for row in rows)),
            "source_files": sources,
            "audits": audits,
            "rules_path": str(config_path),
            "rules_sha256": file_sha256(config_path),
            "user_authorization": "User explicitly authorized reuse of old val/test for new benchmark on 2026-09-11. Original archives and historical SEALED_STATUS files remain unchanged.",
        },
    )
