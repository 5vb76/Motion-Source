"""Continue NVILA training on the remaining 3,653 QA examples with source rehearsal.

PLAN.json supplies the parent checkpoint, grounder, seed, and learning rate.
Only method parameters are updated; the backbone and query router stay frozen.
"""

import sys
import os
import json
import time
import traceback
import contextlib
import importlib.util
import fcntl
import argparse
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
# LOCAL_PATH: External run_pilot and its imports take precedence over repository modules.
sys.path.insert(0, "/root/autodl-tmp/mosder_two_hour_20260911")
import run_pilot as pilot
import torch
import numpy as np
from mosder_fgr_runner_candidate_v3.checkpoint import (
    load_method_parameter_state,
    capture_rng_state,
    restore_rng_state,
)
from nvila_soft_backend import assemble_soft_query_backend, VideoQuery20Request
from soft_query_router import (
    SoftQueryRouter,
    box_grid_occupancy,
    pool_soft_sources,
    SoftRoutingOutput,
)
from score_candidate import frozen_functions

parse_yn, parse_mc, _, _ = frozen_functions()


def native_loss(backend, request, answer):
    """Evaluate native answer loss with NVILA float16 autocast."""
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        return training_runtime.native_loss(backend, request, answer)


# LOCAL_PATH: This helper is loaded directly from external source, including its worker configuration.
utils_spec = importlib.util.spec_from_file_location(
    "general_input_utils", "/root/autodl-tmp/mosder_general_v2_molmo_20260911/common.py"
)
data_utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(data_utils)


def read_json(path):
    """Read an experiment plan or input manifest."""
    return json.loads(Path(path).read_text())


write_json = data_utils.write
sha256_file = data_utils.sha
training_runtime = pilot.rt
runtime_adapter = training_runtime.a
freeze_backend = pilot.frozen
seed_everything = pilot.seed


def write_status(state, **fields):
    """Replace the status snapshot consumed by run monitors."""
    write_json(
        RUN_DIR / "STATUS.json",
        dict(
            state=state, pid=os.getpid(), **{**fields, "updated_at_unix": time.time()}
        ),
    )


def save_checkpoint(path, backend, method_parameters, optimizer, step, plan_sha256):
    """Save method, optimizer, router, and RNG state for exact continuation."""
    payload = dict(
        method_state={
            parameter_name: parameter.detach().cpu().clone()
            for parameter_name, parameter in method_parameters
        },
        router_state=backend.query_router.state_dict(),
        optimizer=optimizer.state_dict(),
        updates=step,
        next_qa_position=min(step * 8, 3653),
        cumulative_qa_seen=2048 + min(step * 8, 3653),
        rng=capture_rng_state(),
        plan_sha256=plan_sha256,
    )
    temporary_path = path.with_suffix(".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)
    write_json(
        RUN_DIR / "LATEST.json",
        dict(path=str(path), updates=step, sha256=sha256_file(path)),
    )


@contextlib.contextmanager
def source_training_request(backend, row, source_worker):
    """Use training boxes for source pooling, then restore the learned router."""
    source_input = source_worker.load(row)
    rgb = source_input["frames"]
    height, width = rgb[0].shape[:2]
    mask = torch.stack(
        [
            box_grid_occupancy(
                np.array(box) / [width, height, width, height], 11, 11, device="cuda"
            )
            for box in source_input["target_boxes_xyxy"]
        ]
    )
    original_router_forward = backend.query_router.forward

    def pool_oracle_boxes(tokens, query_embedding, *, mode="query"):
        target, context, units, unit_weights = pool_soft_sources(tokens, mask)
        return SoftRoutingOutput(
            torch.zeros_like(mask), mask, 1 - mask, target, context, units, unit_weights
        )

    backend.query_router.forward = pool_oracle_boxes
    try:
        yield VideoQuery20Request(
            tuple(rgb),
            source_input["timestamps_ns"],
            source_input["prompt"],
            row["case_id"],
            source_input["prompt"],
        )
    finally:
        backend.query_router.forward = original_router_forward


def evaluate(backend, rows, gold_by_id, name):
    """Resume a fixed evaluation panel and append predictions in manifest order."""
    output_dir = RUN_DIR / name
    output_dir.mkdir(exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    records = data_utils.read(predictions_path) if predictions_path.exists() else []
    assert [x["qa_id"] for x in records] == [x["qa_id"] for x in rows[: len(records)]]
    freeze_backend(backend)
    pixel_cache = {}
    start = time.monotonic()
    for row in rows[len(records) :]:
        item = runtime_adapter.OriginalInput.from_row(row)
        if item.video_id not in pixel_cache:
            if len(pixel_cache) >= 8:
                pixel_cache.pop(next(iter(pixel_cache)))
            pixel_cache[item.video_id] = runtime_adapter.load_pixels(item)
        request = runtime_adapter.native_request(
            item, pixel_cache[item.video_id], item.prompt
        )
        with torch.inference_mode():
            text = backend.generate(request, **runtime_adapter.GENERATION).text
        parser = parse_yn if row["question_type"] == "ynqa" else parse_mc
        parsed = parser(text)
        target = gold_by_id[row["qa_id"]]
        truth = (
            target["yn_answer"]
            if row["question_type"] == "ynqa"
            else target["mc_answer"]
        )
        record = dict(
            qa_id=row["qa_id"],
            video_id=row["video_id"],
            question_type=row["question_type"],
            text=text,
            parsed=parsed,
            gold=truth,
            correct=parsed == truth,
        )
        records.append(record)
        data_utils.append(predictions_path, record)
        write_status(
            "evaluating",
            evaluation=name,
            rows=len(records),
            total_rows=len(rows),
            seconds=time.monotonic() - start,
        )
    metrics = dict(
        n=len(records),
        correct=sum(x["correct"] for x in records),
        accuracy=sum(x["correct"] for x in records) / len(records),
        parse_failures=sum(x["parsed"] is None for x in records),
        seconds=time.monotonic() - start,
    )
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def main():
    """Validate the plan, resume if requested, train, and evaluate the fixed endpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    lock = (RUN_DIR / "RUN.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # LOCAL_PATH: Supply sibling PLAN.json; parent, grounder, and hash-map paths must resolve locally.
    plan = read_json(RUN_DIR / "PLAN.json")
    plan_sha256 = sha256_file(RUN_DIR / "PLAN.json")
    for path, expected_hash in {
        **plan["input_hashes"],
        **plan.get("code_hashes", {}),
    }.items():
        assert sha256_file(path) == expected_hash, path
    assert sha256_file(plan["parent"]) == plan["parent_sha256"]
    assert sha256_file(plan["grounder"]) == plan["grounder_sha256"]
    if (RUN_DIR / "LATEST.json").exists() and not args.resume:
        raise RuntimeError("Use --resume")
    seed_everything(plan["seed"])
    torch.set_num_threads(4)
    source_worker = data_utils.Worker()
    # LOCAL_PATH: Supply sibling QA/source manifests; their RGB and source-data paths must exist locally.
    qa_rows = read_json(RUN_DIR / "qa_train.json")
    source_rows = read_json(RUN_DIR / "source512.json")
    # LOCAL_PATH: Training gold is read from the external pilot data directory.
    gold_by_id = {
        x["qa_id"]: x
        for x in read_json(pilot.ROOT / "data/OFFICIAL_TRAIN_GOLD.json")["items"]
    }
    from mosder_final_v1.language import canonical_answer

    try:
        write_status("loading")
        with pilot.load_runtime(
            family="nvila_lite_8b", arm="raw_vlm", device="cuda:0", dtype=None
        ) as loaded:
            router = SoftQueryRouter(3584, rank=64)
            router.load_state_dict(
                torch.load(plan["grounder"], map_location="cpu", weights_only=True)
            )
            backend = assemble_soft_query_backend(loaded.backend, router)
            loaded.backend = backend
            payload = torch.load(plan["parent"], map_location="cpu", weights_only=False)
            method_parameters = training_runtime.method(backend)
            load_method_parameter_state(method_parameters, payload["method_state"])
            backend.configure_mosder_stage("FROZEN")
            freeze_backend(backend)
            trainable_parameters = [
                (parameter_name, parameter)
                for parameter_name, parameter in method_parameters
                if not parameter_name.startswith("decision.")
            ]
            for parameter_name, parameter in trainable_parameters:
                parameter.requires_grad_(True)
            trainable_parameter_ids = {
                id(parameter) for parameter_name, parameter in trainable_parameters
            }
            assert {
                id(parameter)
                for parameter in backend.model.parameters()
                if parameter.requires_grad
            } == trainable_parameter_ids
            assert not any(
                parameter.requires_grad
                for parameter in backend.query_router.parameters()
            )
            frozen_parameter_stamp = tuple(
                (parameter_name, id(parameter), parameter._version)
                for parameter_name, parameter in backend.model.named_parameters()
                if id(parameter) not in trainable_parameter_ids
            )
            optimizer = torch.optim.AdamW(
                [parameter for parameter_name, parameter in trainable_parameters],
                lr=plan["lr"],
                weight_decay=0.01,
            )
            optimizer.zero_grad(set_to_none=True)
            step = 0
            if not args.resume:
                assert payload["updates"] == 256
                optimizer.load_state_dict(payload["optimizer"])
                restore_rng_state(payload["rng"])
                for group in optimizer.param_groups:
                    group["lr"] = plan["lr"]
            del payload
            if args.resume:
                latest = read_json(RUN_DIR / "LATEST.json")
                assert sha256_file(latest["path"]) == latest["sha256"]
                resume_payload = torch.load(
                    latest["path"], map_location="cpu", weights_only=False
                )
                assert resume_payload["plan_sha256"] == plan_sha256
                load_method_parameter_state(
                    method_parameters, resume_payload["method_state"]
                )
                backend.query_router.load_state_dict(resume_payload["router_state"])
                optimizer.load_state_dict(resume_payload["optimizer"])
                step = resume_payload["updates"]
                restore_rng_state(resume_payload["rng"])
                del resume_payload
                frozen_parameter_stamp = tuple(
                    (parameter_name, id(parameter), parameter._version)
                    for parameter_name, parameter in backend.model.named_parameters()
                    if id(parameter) not in trainable_parameter_ids
                )
            write_json(
                RUN_DIR / "ASSEMBLY.json",
                dict(
                    parent=plan["parent"],
                    parent_sha256=plan["parent_sha256"],
                    trainable_count=sum(
                        parameter.numel()
                        for parameter_name, parameter in trainable_parameters
                    ),
                    trainable_names=[
                        parameter_name
                        for parameter_name, parameter in trainable_parameters
                    ],
                    grounder_frozen=True,
                    old_task_parent_loaded=False,
                ),
            )
            start = time.monotonic()
            pixel_cache = {}
            qa_losses = []
            source_losses = []
            update_start = time.monotonic()
            for position in range(step * 8, 3653):
                row = qa_rows[position]
                item = runtime_adapter.OriginalInput.from_row(row)
                if item.video_id not in pixel_cache:
                    if len(pixel_cache) >= 8:
                        pixel_cache.pop(next(iter(pixel_cache)))
                    pixel_cache[item.video_id] = runtime_adapter.load_pixels(item)
                request = runtime_adapter.native_request(
                    item, pixel_cache[item.video_id], item.prompt
                )
                target = gold_by_id[row["qa_id"]]
                answer = (
                    ("是" if target["yn_answer"] == "yes" else "否")
                    if row["question_type"] == "ynqa"
                    else target["mc_answer"]
                )
                loss = native_loss(backend, request, answer)
                assert torch.isfinite(loss)
                qa_losses.append(float(loss.detach()))
                (loss / min(8, 3653 - (position // 8) * 8)).backward()
                del loss
                if position % 4 == 0:
                    source_row = source_rows[position // 4]
                    with source_training_request(
                        backend, source_row, source_worker
                    ) as source_request:
                        loss = native_loss(
                            backend,
                            source_request,
                            canonical_answer(source_row["state"]),
                        )
                    assert torch.isfinite(loss)
                    source_losses.append(float(loss.detach()))
                    (0.125 * loss).backward()
                    del loss
                if position % 8 == 7 or position == 3652:
                    for group in optimizer.param_groups:
                        group["lr"] = plan["lr"]
                    norm = torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for parameter_name, parameter in trainable_parameters
                        ],
                        1.0,
                    )
                    assert torch.isfinite(norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    assert (
                        tuple(
                            (parameter_name, id(parameter), parameter._version)
                            for parameter_name, parameter in backend.model.named_parameters()
                            if id(parameter) not in trainable_parameter_ids
                        )
                        == frozen_parameter_stamp
                    )
                    record = dict(
                        update=step,
                        total_updates=457,
                        qa_seen=position + 1,
                        source_seen=position // 4 + 1,
                        qa_loss=sum(qa_losses) / len(qa_losses),
                        source_loss=sum(source_losses) / len(source_losses),
                        lr=optimizer.param_groups[0]["lr"],
                        grad_norm=float(norm),
                        seconds_per_update=time.monotonic() - update_start,
                        elapsed_seconds=time.monotonic() - start,
                        updated_at_unix=time.time(),
                    )
                    data_utils.append(RUN_DIR / "TRAIN.jsonl", record)
                    write_status("training", **record)
                    print(json.dumps(record), flush=True)
                    qa_losses = []
                    source_losses = []
                    if step in [1, 5] or step % 32 == 0:
                        save_checkpoint(
                            RUN_DIR / f"checkpoint_{step:04d}.pt",
                            backend,
                            method_parameters,
                            optimizer,
                            step,
                            plan_sha256,
                        )
                    update_start = time.monotonic()
            assert step == 457
            save_checkpoint(
                RUN_DIR / "FINAL.pt",
                backend,
                method_parameters,
                optimizer,
                step,
                plan_sha256,
            )
            write_json(
                RUN_DIR / "TRAIN_COMPLETE.json",
                dict(
                    updates=457,
                    parent_sha256=plan["parent_sha256"],
                    checkpoint_sha256=sha256_file(RUN_DIR / "FINAL.pt"),
                ),
            )
            del optimizer, pixel_cache
            metrics = {}
            # Evaluate the validation panel at the fixed continuation endpoint.
            # LOCAL_PATH: Requires external validation gold and sibling qa_val914.json.
            validation_gold = {
                x["qa_id"]: x
                for x in read_json(
                    "/root/autodl-tmp/mosder_reference_research_20260906/omnivchall_val_eval_v1/data/OFFICIAL_VAL_GOLD.json"
                )["items"]
            }
            metrics["val914"] = evaluate(
                backend,
                read_json(RUN_DIR / "qa_val914.json"),
                validation_gold,
                "val914",
            )
            write_json(
                RUN_DIR / "COMPLETE.json",
                dict(
                    metrics=metrics,
                    checkpoint_sha256=sha256_file(RUN_DIR / "FINAL.pt"),
                    ordinary_lora_started=False,
                ),
            )
            write_status("complete", metrics=metrics)
    finally:
        source_worker.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write_json(
            RUN_DIR / "FAILURE.json",
            dict(error=str(error), traceback=traceback.format_exc()),
        )
        write_status("failed", error=str(error))
        raise
