"""Run frozen-checkpoint inference interventions and replay the baseline."""

import fcntl
import json
import os
import time
import traceback
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
import torch
from common import MotionDataset, RGB20Worker, file_sha256, read_jsonl, write_json
from interventions import apply_intervention
from mosder_fgr_runner_candidate_v3.checkpoint import (
    configure_reproducible_numeric_environment,
    load_method_parameter_state,
    method_parameter_state,
    set_global_seed,
    tensor_state_sha256,
)
from mosder_fgr_runner_candidate_v3.engine import MoSDeRExecutionAdapter
from mosder_final_v1.family_backend import load_local_mosder_backend
from summarize_function import summarize
from validate_function import validate


def main():
    """Check baseline identity, evaluate each arm, then write the comparison."""
    lock = (EXPERIMENT_ROOT / "RUN.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # LOCAL_PATH: prepare.py must supply PLAN.json, val1200.jsonl, and baseline
    # predictions beside this script; checkpoint/baseline_identity paths must exist.
    plan = json.loads((EXPERIMENT_ROOT / "PLAN.json").read_text())
    assert file_sha256(plan["checkpoint"]) == plan["checkpoint_sha256"]
    configure_reproducible_numeric_environment()
    set_global_seed(20260913)
    torch.set_num_threads(4)
    rows = read_jsonl(EXPERIMENT_ROOT / "val1200.jsonl")
    baseline_predictions = {
        x["case_id"]: x
        for x in read_jsonl(EXPERIMENT_ROOT / "full/val1200/predictions.jsonl")
    }
    for arm in plan["arms"]:
        arm_dir = EXPERIMENT_ROOT / arm
        if (arm_dir / "COMPLETE.json").exists():
            continue
        write_json(
            EXPERIMENT_ROOT / "STATUS.json",
            dict(state="running", arm=arm, pid=os.getpid(), time=time.time()),
        )

        def write_status(state, **details):
            write_json(
                arm_dir / "STATUS.json", dict(state=state, time=time.time(), **details)
            )

        write_status("loading")
        backend = load_local_mosder_backend(
            "molmo2_o_7b", device="cuda:0", dtype="bfloat16"
        )
        adapter = MoSDeRExecutionAdapter(backend)
        checkpoint = torch.load(
            plan["checkpoint"], map_location="cpu", weights_only=False
        )
        load_method_parameter_state(
            adapter.method_parameters, checkpoint["method_parameter_state"]
        )
        del checkpoint
        identity = json.loads(Path(plan["baseline_identity"]).read_text())
        assert (
            tensor_state_sha256(method_parameter_state(adapter.method_parameters))
            == identity["method_sha256"]
        )
        assert identity["case_ids"] == [x["case_id"] for x in rows]
        worker = RGB20Worker(arm_dir)
        try:
            if arm == "full":
                backend.configure_mosder_stage("R")
                replay = []
                for state in ["neither", "camera_only", "object_only", "both"]:
                    row = next(x for x in rows if x["state"] == state)
                    example = MotionDataset([row], worker)[0]
                    try:
                        with torch.no_grad():
                            text = backend.generate(
                                example.request, max_new_tokens=96, do_sample=False
                            ).text
                        replay.append(
                            dict(
                                case_id=example.case_id,
                                text=text,
                                matches_original=text
                                == baseline_predictions[example.case_id]["text"],
                            )
                        )
                    finally:
                        example.close()
                write_json(
                    EXPERIMENT_ROOT / "BASELINE_REPLAY.json",
                    dict(records=replay, weights_match=True),
                )
                assert all(x["matches_original"] for x in replay), (
                    "Baseline replay mismatch: do not reuse scores"
                )
            apply_intervention(backend, arm)
            metrics = validate(adapter, worker, rows, arm_dir, write_status)
            write_json(
                arm_dir / "COMPLETE.json",
                dict(metrics=metrics, checkpoint_sha256=plan["checkpoint_sha256"]),
            )
            write_status("complete", metrics=metrics)
        finally:
            worker.close()
            backend.close()
            del backend, adapter
            torch.cuda.empty_cache()
    summarize(plan, EXPERIMENT_ROOT)
    write_json(
        EXPERIMENT_ROOT / "STATUS.json", dict(state="complete", time=time.time())
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write_json(
            EXPERIMENT_ROOT / "FAILURE.json",
            dict(error=str(error), traceback=traceback.format_exc()),
        )
        write_json(
            EXPERIMENT_ROOT / "STATUS.json",
            dict(state="failed", error=str(error), time=time.time()),
        )
        raise
