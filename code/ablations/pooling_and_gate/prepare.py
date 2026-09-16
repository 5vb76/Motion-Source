"""Prepare pooling/gate intervention data and plans without regenerating code."""

import hashlib
import json
import shutil
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
# LOCAL_PATH: External input directory for validation data and intervention settings.
PREVIOUS_EXPERIMENT_ROOT = Path("/root/autodl-tmp/mosder_core_ablation_fast_v1")
# LOCAL_PATH: Supplies model weights, data_worker.py, and validation predictions/identity.
TRAINING_ROOT = Path("/root/autodl-tmp/mosder_general_v2_molmo_20260911")


def main():
    """Copy reference data and retain the original checkpoint and sampling plan."""
    shutil.copy2(
        PREVIOUS_EXPERIMENT_ROOT / "val1200.jsonl", EXPERIMENT_ROOT / "val1200.jsonl"
    )
    for arm in ["full", "global_evidence", "constant_gate"]:
        arm_dir = EXPERIMENT_ROOT / arm
        arm_dir.mkdir(exist_ok=True)
        worker_path = arm_dir / "data_worker.py"
        if not worker_path.exists():
            worker_path.symlink_to(TRAINING_ROOT / "data_worker.py")
    rows = [
        json.loads(line)
        for line in (EXPERIMENT_ROOT / "val1200.jsonl").read_text().splitlines()
    ]
    records = [
        json.loads(line)
        for line in (TRAINING_ROOT / "validation/R_endpoint/predictions.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [record["case_id"] for record in rows] == [
        record["case_id"] for record in records
    ]
    assert sum(record["correct"] for record in records) == 1009
    for record, row in zip(records, rows):
        record.update(
            split_group=row["split_group"], physical_window_id=row["physical_window_id"]
        )
    baseline_dir = EXPERIMENT_ROOT / "full/val1200"
    baseline_dir.mkdir(exist_ok=True)
    (baseline_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    plan = dict(
        arms=["full", "global_evidence", "constant_gate"],
        checkpoint=str(TRAINING_ROOT / "checkpoints/R_endpoint.pt"),
        checkpoint_sha256=hashlib.sha256(
            (TRAINING_ROOT / "checkpoints/R_endpoint.pt").read_bytes()
        ).hexdigest(),
        baseline_identity=str(TRAINING_ROOT / "validation/R_endpoint/identity.json"),
        baseline_predictions_sha256=hashlib.sha256(
            (TRAINING_ROOT / "validation/R_endpoint/predictions.jsonl").read_bytes()
        ).hexdigest(),
        n=1200,
        training=False,
        test=False,
        max_new_tokens=96,
        do_sample=False,
        readout="original four-field generation and strict parser",
        claim="inference-time component dependency, includes intervention distribution shift; not retrained structure ablation",
        baseline_reuse_gate="matching weights and sample order plus exact replay of first example in each state",
        interventions=json.loads((PREVIOUS_EXPERIMENT_ROOT / "PLAN.json").read_text())[
            "interventions"
        ],
    )
    (EXPERIMENT_ROOT / "PLAN.json").write_text(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
