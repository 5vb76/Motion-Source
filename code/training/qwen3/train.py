"""Train the plan-selected model through F/G/R and report the R endpoint."""

import argparse
import fcntl
import json
import os
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

from common import (
    BACKEND_SOURCE_DIR,
    BENCHMARK_DIR,
    RUN_DIR,
    BenchmarkWorker,
    MotionDataset,
    PlannedEpochSampler,
    append_jsonl,
    file_sha256,
    read_jsonl,
    write_json,
)
from mosder_final_v1.family_backend import load_local_mosder_backend
from mosder_fgr_runner_candidate_v3.engine import (
    MoSDeRExecutionAdapter,
    run_current_epoch,
    GradientWindow,
)
from mosder_fgr_runner_candidate_v3.checkpoint import (
    configure_reproducible_numeric_environment,
    set_global_seed,
    capture_rng_state,
    restore_rng_state,
    method_parameter_state,
    load_method_parameter_state,
    tensor_state_sha256,
)
from evaluation import evaluate


def write_checkpoint(path, payload):
    """Write a checkpoint atomically, then save its SHA256 sidecar."""
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)
    write_json(path.with_suffix(".sha256.json"), {"sha256": file_sha256(path)})


def save_checkpoint(adapter, runtime, sampler, stage, plan_digest, tag):
    """Save model, optimizer, sampler, and RNG state at an update boundary."""
    assert all(p.grad is None for _, p in adapter.method_parameters)
    path = RUN_DIR / "checkpoints" / f"{stage}_{tag}.pt"
    path.parent.mkdir(exist_ok=True)
    payload = {
        "plan_sha256": plan_digest,
        "stage": stage,
        "stage_optimizer_step": runtime.stage_optimizer_step,
        "global_optimizer_step": runtime.global_optimizer_step,
        "micro_examples_seen": runtime.micro_examples_seen,
        "cursor": sampler.cursor,
        "method_parameter_state": method_parameter_state(adapter.method_parameters),
        "optimizer_state": runtime.optimizer.state_dict(),
        "scheduler_state": runtime.scheduler.state_dict(),
        "rng_state": capture_rng_state(),
        "initialization": "fresh raw pretrained base, no old task checkpoint",
    }
    write_checkpoint(path, payload)
    write_json(
        RUN_DIR / "LATEST.json",
        {
            "path": str(path),
            "sha256": file_sha256(path),
            "stage": stage,
            "step": runtime.stage_optimizer_step,
            "plan_sha256": plan_digest,
        },
    )
    return path


def run_preflight(adapter, training_rows, worker, protocol):
    """Check gradients on four training examples, then restore weights and RNG."""
    rng_state = capture_rng_state()
    initial_parameters = method_parameter_state(adapter.method_parameters)
    results = []
    # Four sources and all four states, with no validation/test access.
    coverage_cases = [
        ("ADT-LiteOffice", "neither"),
        ("HOT3D", "camera_only"),
        ("TACO-V1-allocentric", "object_only"),
        ("AV2", "both"),
    ]
    preflight_rows = [
        next(r for r in training_rows if r["source"] == s and r["state"] == st)
        for s, st in coverage_cases
    ]
    dataset = MotionDataset(preflight_rows, worker)
    for stage in "FGR":
        runtime = adapter.configure_stage(
            stage, total_optimizer_steps=625, protocol=protocol
        )
        for i in range(len(dataset)):
            example = dataset[i]
            try:
                losses = adapter.backward_one(runtime, example)
                adapter.assert_no_foreign_gradients(runtime)
                window = GradientWindow(runtime.active_parameters)
                window.capture_and_clear()
                assert window.ever_nonzero
                results.append(
                    {
                        "stage": stage,
                        "source": example.source_dataset,
                        "state": example.state,
                        "losses": losses,
                        "finite_nonzero_gradients": True,
                    }
                )
                print("SMOKE", stage, example.source_dataset, losses, flush=True)
                del window
            finally:
                example.close()
        adapter.assert_integrity(runtime)
        del runtime
    metrics = evaluate(
        adapter, dataset, RUN_DIR / "preflight/eval_train4", "preflight_train4_only"
    )
    load_method_parameter_state(adapter.method_parameters, initial_parameters)
    restore_rng_state(rng_state)
    assert tensor_state_sha256(initial_parameters) == tensor_state_sha256(
        method_parameter_state(adapter.method_parameters)
    )
    write_json(
        RUN_DIR / "preflight/receipt.json",
        {
            "status": "PASS",
            "checks": results,
            "evaluation_n": metrics["n"],
            "training_updates_committed": 0,
            "method_and_rng_restored": True,
            "val_test_used": False,
        },
    )


def main():
    """Run or resume the fixed experiment plan and write validation summaries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    lock = (RUN_DIR / "RUN.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Refuse to resume against changed data, code, or experiment settings.
    # LOCAL_PATH: Supply PLAN.json beside this script; its hash maps contain local dependency paths.
    plan = json.loads((RUN_DIR / "PLAN.json").read_text())
    plan_digest = file_sha256(RUN_DIR / "PLAN.json")
    for path, digest in {
        **plan["data_hashes"],
        **plan["code_hashes"],
        **plan["inherited_code_hashes"],
    }.items():
        assert file_sha256(path) == digest, "Changed dependency: " + path
    assert (
        file_sha256(BACKEND_SOURCE_DIR / "family_backends_v1.py")
        == plan["baseline_backend_sha256"]
    )
    assert plan["other_models_auto_start"] is False
    configure_reproducible_numeric_environment()
    torch.set_num_threads(4)
    set_global_seed(plan["seed"])
    torch.cuda.set_device(0)
    # LOCAL_PATH: These manifests and the files referenced by model_input must exist locally.
    training_rows = read_jsonl(BENCHMARK_DIR / "release/train.jsonl")
    validation_rows = read_jsonl(BENCHMARK_DIR / "release/val.jsonl")
    worker = BenchmarkWorker()
    backend = None
    protocol = SimpleNamespace(warmup_fraction=0.05, cosine_floor=0.1)
    try:
        write_json(
            RUN_DIR / "STATUS.json",
            {
                "state": "loading_fresh_pretrained_backbone",
                "pid": os.getpid(),
                "updated_at_unix": time.time(),
            },
        )
        backend = load_local_mosder_backend(
            plan["family"], device="cuda:0", dtype=plan["dtype"]
        )
        adapter = MoSDeRExecutionAdapter(backend)
        if not (RUN_DIR / "INITIALIZATION.json").exists():
            write_json(
                RUN_DIR / "INITIALIZATION.json",
                {
                    "status": "FRESH",
                    "old_task_checkpoint_loaded": False,
                    "method_state_sha256": tensor_state_sha256(
                        method_parameter_state(adapter.method_parameters)
                    ),
                    "method_parameter_count": sum(
                        p.numel() for _, p in adapter.method_parameters
                    ),
                    "gpu": torch.cuda.get_device_name(0),
                    "runtime_audit": getattr(backend, "runtime_audit", None),
                    "artifact_audit": getattr(backend, "artifact_audit", None),
                    "plan_sha256": plan_digest,
                },
            )
        if not args.resume:
            if (RUN_DIR / "LATEST.json").exists():
                raise RuntimeError("Existing run: use --resume")
            run_preflight(adapter, training_rows, worker, protocol)
        else:
            assert (
                json.loads((RUN_DIR / "preflight/receipt.json").read_text())["status"]
                == "PASS"
            )
        resumed = None
        if args.resume:
            latest = json.loads((RUN_DIR / "LATEST.json").read_text())
            assert (
                latest["plan_sha256"] == plan_digest
                and file_sha256(latest["path"]) == latest["sha256"]
            )
            resumed = torch.load(latest["path"], map_location="cpu", weights_only=False)
            load_method_parameter_state(
                adapter.method_parameters, resumed["method_parameter_state"]
            )
        # Each stage follows the stored order: 5,000 examples / 8 = 625 updates.
        for stage_index, stage in enumerate("FGR"):
            completion_path = RUN_DIR / f"{stage}_COMPLETE.json"
            if completion_path.exists():
                info = json.loads(completion_path.read_text())
                assert (
                    info["plan_sha256"] == plan_digest
                    and file_sha256(info["checkpoint"]) == info["sha256"]
                )
                if resumed is not None and stage_index < list("FGR").index(
                    resumed["stage"]
                ):
                    continue
                state = torch.load(
                    info["checkpoint"], map_location="cpu", weights_only=False
                )
                load_method_parameter_state(
                    adapter.method_parameters, state["method_parameter_state"]
                )
                restore_rng_state(state["rng_state"])
                del state
                continue
            if stage == "R":
                g_metrics_path = RUN_DIR / "validation/G_endpoint/metrics.json"
                if not g_metrics_path.exists():
                    evaluate(
                        adapter,
                        MotionDataset(validation_rows, worker),
                        RUN_DIR / "validation/G_endpoint",
                        "G_endpoint_before_R",
                    )
            runtime = adapter.configure_stage(
                stage,
                total_optimizer_steps=625,
                protocol=protocol,
                global_optimizer_step=stage_index * 625,
            )
            sampler = PlannedEpochSampler(training_rows, plan["orders"][stage])
            dataset = MotionDataset(training_rows, worker)
            if resumed is not None and resumed["stage"] == stage:
                runtime.optimizer.load_state_dict(resumed["optimizer_state"])
                runtime.scheduler.load_state_dict(resumed["scheduler_state"])
                runtime.stage_optimizer_step = resumed["stage_optimizer_step"]
                runtime.global_optimizer_step = resumed["global_optimizer_step"]
                runtime.micro_examples_seen = resumed["micro_examples_seen"]
                sampler.cursor = resumed["cursor"]
                restore_rng_state(resumed["rng_state"])
                resumed = None
            started = time.monotonic()
            last_update_time = [started]
            update_durations = []
            attempt_id = time.time_ns()
            metric_path = RUN_DIR / f"{stage}_metrics_{attempt_id}.jsonl"

            def record_optimizer_step(current_runtime, current_sampler, batch_stats):
                """Persist update metrics and periodic resumable checkpoints."""
                now = time.monotonic()
                seconds = now - last_update_time[0]
                update_durations.append(seconds)
                step_metrics = {
                    "event": "optimizer_step",
                    "stage": stage,
                    "stage_step": current_runtime.stage_optimizer_step,
                    "stage_total": 625,
                    "global_step": current_runtime.global_optimizer_step,
                    "total_steps": 1875,
                    "cursor": current_sampler.cursor,
                    "loss": dict(batch_stats.loss_means),
                    "gradient_pre_clip": batch_stats.gradient_pre_clip_norm,
                    "gradient_post_clip": batch_stats.gradient_post_clip_norm,
                    "clip_pass": batch_stats.gradient_clip_audit.final_gate_pass,
                    "learning_rates_next": [
                        g["lr"] for g in current_runtime.optimizer.param_groups
                    ],
                    "seconds_per_update": seconds,
                    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                    "updated_at_unix": time.time(),
                    "sample_case_ids": [
                        training_rows[i]["case_id"]
                        for i in current_sampler.order[
                            current_sampler.cursor - 8 : current_sampler.cursor
                        ]
                    ],
                    "plan_sha256": plan_digest,
                }
                import csv

                curve_row = {
                    k: step_metrics[k]
                    for k in [
                        "stage",
                        "stage_step",
                        "global_step",
                        "cursor",
                        "seconds_per_update",
                        "peak_cuda_memory_bytes",
                        "updated_at_unix",
                    ]
                }
                curve_row.update(
                    {f"loss_{k}": v for k, v in step_metrics["loss"].items()}
                )
                curve_row["learning_rates_next"] = json.dumps(
                    step_metrics["learning_rates_next"]
                )
                curve_path = RUN_DIR / f"{stage}_training_curve.csv"
                exists = curve_path.exists()
                with curve_path.open("a") as curve_stream:
                    writer = csv.DictWriter(curve_stream, fieldnames=list(curve_row))
                    if not exists:
                        writer.writeheader()
                    writer.writerow(curve_row)
                append_jsonl(metric_path, step_metrics)
                write_json(
                    RUN_DIR / "STATUS.json", {"state": "training", **step_metrics}
                )
                print(json.dumps(step_metrics), flush=True)
                if (
                    current_runtime.stage_optimizer_step in [1, 5, 10]
                    or current_runtime.stage_optimizer_step % 25 == 0
                ):
                    save_checkpoint(
                        adapter,
                        current_runtime,
                        current_sampler,
                        stage,
                        plan_digest,
                        f"step_{current_runtime.stage_optimizer_step:04d}",
                    )
                if current_runtime.stage_optimizer_step in [10, 20]:
                    write_json(
                        RUN_DIR / "TIMING.json",
                        {
                            "stage": stage,
                            "measured_updates": len(update_durations),
                            "median_seconds_per_update": float(
                                np.median(update_durations[-10:])
                            ),
                            "estimated_remaining_current_stage_hours": float(
                                np.median(update_durations[-10:])
                                * (625 - current_runtime.stage_optimizer_step)
                                / 3600
                            ),
                            "note": "Other-stage speeds remain historical estimates until measured.",
                        },
                    )
                # Checkpoint I/O is excluded from the next update duration.
                last_update_time[0] = time.monotonic()

            if not sampler.at_epoch_end:
                run_current_epoch(
                    adapter,
                    runtime,
                    sampler,
                    dataset,
                    accumulation=8,
                    gradient_clip_max_norm=1.0,
                    on_optimizer_boundary=record_optimizer_step,
                )
            assert (
                runtime.stage_optimizer_step == 625
                and runtime.global_optimizer_step == (stage_index + 1) * 625
            )
            path = save_checkpoint(
                adapter, runtime, sampler, stage, plan_digest, "endpoint"
            )
            write_json(
                completion_path,
                {
                    "status": "COMPLETE",
                    "checkpoint": str(path),
                    "sha256": file_sha256(path),
                    "plan_sha256": plan_digest,
                    "updates": 625,
                },
            )
            del runtime, dataset
        r_metrics_path = RUN_DIR / "validation/R_endpoint/metrics.json"
        if not r_metrics_path.exists():
            evaluate(
                adapter,
                MotionDataset(validation_rows, worker),
                RUN_DIR / "validation/R_endpoint",
                "R_endpoint",
            )
        from qa_readout import write_qa_readout

        for endpoint in ["G_endpoint", "R_endpoint"]:
            write_qa_readout(RUN_DIR / "validation" / endpoint)
        r_metrics = json.loads(r_metrics_path.read_text())
        selected_path = RUN_DIR / "checkpoints/R_endpoint.pt"
        write_json(
            RUN_DIR / "SELECTED.json",
            {
                "stage": "R",
                "checkpoint": str(selected_path),
                "sha256": file_sha256(selected_path),
                "metrics": r_metrics,
                "selection": "User-locked original Shared F/G/R recipe; R endpoint. G reported as stage ablation.",
            },
        )
        write_json(
            RUN_DIR / "COMPLETION.json",
            {
                "status": "COMPLETE",
                "selected": str(selected_path),
                "plan_sha256": plan_digest,
                "test_used": False,
                "qa_readout": str(RUN_DIR / "validation/R_endpoint/qa_metrics.json"),
            },
        )
        write_json(
            RUN_DIR / "STATUS.json",
            {"state": "complete", "updated_at_unix": time.time()},
        )
    finally:
        worker.close()
        if backend is not None:
            backend.close()


atomic_checkpoint = write_checkpoint


def checkpoint(adapter, runtime, sampler, stage, plan_sha, tag):
    return save_checkpoint(adapter, runtime, sampler, stage, plan_sha, tag)


def smoke(adapter, train, worker, protocol):
    return run_preflight(adapter, train, worker, protocol)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if not isinstance(error, KeyboardInterrupt):
            write_json(
                RUN_DIR / "FAILURE.json",
                {
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "updated_at_unix": time.time(),
                },
            )
            write_json(
                RUN_DIR / "STATUS.json",
                {
                    "state": "failed",
                    "error": str(error),
                    "updated_at_unix": time.time(),
                },
            )
        raise
