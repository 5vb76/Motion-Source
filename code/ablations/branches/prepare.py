"""Prepare branch-knockout data and plans without overwriting repository code."""

import json
import shutil
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
# LOCAL_PATH: External input directory for validation data and intervention settings.
PREVIOUS_EXPERIMENT_ROOT = Path("/root/autodl-tmp/mosder_inference_interventions_v1")


def main():
    """Copy reference data and retain the original checkpoint and sampling plan."""
    shutil.copy2(
        PREVIOUS_EXPERIMENT_ROOT / "val1200.jsonl", EXPERIMENT_ROOT / "val1200.jsonl"
    )
    plan = json.loads((PREVIOUS_EXPERIMENT_ROOT / "PLAN.json").read_text())
    plan.update(
        arms=["full", "no_camera", "no_object"],
        interventions={
            "no_camera": "zero camera source adapter output, camera explicit residual output and each camera LoRA B output; preserve original raw tokens, Object and Shared",
            "no_object": "zero object source adapter output, object explicit residual output and each object LoRA B output; preserve original raw tokens, Camera and Shared",
        },
        claim="source-branch dependence of the trained model under inference-time knockout; not retrained single-branch comparison",
    )
    (EXPERIMENT_ROOT / "PLAN.json").write_text(json.dumps(plan, indent=2))
    for arm in plan["arms"]:
        arm_dir = EXPERIMENT_ROOT / arm
        arm_dir.mkdir(exist_ok=True)
        worker_path = arm_dir / "data_worker.py"
        if not worker_path.exists():
            # LOCAL_PATH: Shared data worker source; the symlink retains this absolute target.
            worker_path.symlink_to(
                Path("/root/autodl-tmp/mosder_general_v2_molmo_20260911/data_worker.py")
            )
    (EXPERIMENT_ROOT / "full/val1200").mkdir(exist_ok=True)
    shutil.copy2(
        PREVIOUS_EXPERIMENT_ROOT / "full/val1200/predictions.jsonl",
        EXPERIMENT_ROOT / "full/val1200/predictions.jsonl",
    )


if __name__ == "__main__":
    main()
