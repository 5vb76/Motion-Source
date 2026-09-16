"""Train QA residuals on a frozen source-pretrained LoRA with source rehearsal."""

import csv
import fcntl
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
# LOCAL_PATH: External source directories supply training utilities and QA input code.
sys.path.insert(0, "/root/autodl-tmp/molmo_fourfield_lora_qa_v1")
import run as base

training_utils = base.u
sys.path.insert(0, "/root/autodl-tmp/mosder_two_hour_20260911")
import run_pilot as pilot
import torch
from family_backends_v1 import RGB20Request, load_local_backend
from mosder_fgr_runner_candidate_v3.checkpoint import (
    capture_rng_state,
    configure_reproducible_numeric_environment,
    restore_rng_state,
    set_global_seed,
)
from qa_update_controls import OrdinaryUpdateResidual
from score_candidate import frozen_functions

parse_yn, parse_mc, _, _ = frozen_functions()


def read_json(path):
    """Read a plan, split, or result document."""
    return json.loads(Path(path).read_text())


# LOCAL_PATH: Supply PLAN.json and qa_train/source512/qa_dev650/qa_val914 JSON files
# beside this script; PLAN["parent"] must reference the source-pretrained weights.
PLAN = read_json(EXPERIMENT_ROOT / "PLAN.json")


def write_status(state, **kw):
    training_utils.write(
        EXPERIMENT_ROOT / "STATUS.json",
        dict(state=state, pid=os.getpid(), updated_at_unix=time.time(), **kw),
    )


def build_request(row, pixels):
    """Build the original QA request with its fixed central target box."""
    item = pilot.rt.a.OriginalInput.from_row(row)
    height, width = pixels.shape[1:3]
    return RGB20Request(
        frames=tuple(pixels),
        timestamps_ns=item.timestamps_ns,
        prompt=item.prompt,
        oracle_boxes_xyxy=[(0.25 * width, 0.25 * height, 0.75 * width, 0.75 * height)]
        * 20,
        request_id=item.qa_id,
    )


def load_rgb(row, cache):
    """Reuse decoded clips in an eight-entry insertion-order cache."""
    key = row["rgb20_npz_path"]
    if key not in cache:
        if len(cache) >= 8:
            cache.pop(next(iter(cache)))
        cache[key] = pilot.rt.a.load_pixels(pilot.rt.a.OriginalInput.from_row(row))
    return cache[key]


def answer(row, gold):
    """Use the source protocol's yes/no text or multiple-choice answer."""
    target = gold[row["qa_id"]]
    return (
        ("是" if target["yn_answer"] == "yes" else "否")
        if row["question_type"] == "ynqa"
        else target["mc_answer"]
    )


def save_checkpoint(adapter_parameters, optimizer, step, path):
    """Save the QA residual and resume state without changing the parent adapter."""
    tmp = path.with_suffix(".tmp")
    torch.save(
        dict(
            adapter={n: p.detach().cpu().clone() for n, p in adapter_parameters},
            optimizer=optimizer.state_dict(),
            step=step,
            rng=capture_rng_state(),
            parent_sha256=PLAN["parent_sha256"],
            plan_sha256=training_utils.sha(EXPERIMENT_ROOT / "PLAN.json"),
        ),
        tmp,
    )
    os.replace(tmp, path)
    training_utils.write(
        EXPERIMENT_ROOT / "LATEST.json",
        dict(path=str(path), step=step, sha256=training_utils.sha(path)),
    )


def evaluate(backend, rows, gold, name):
    """Resume QA generation and check that evaluation leaves weights unchanged."""
    dest = EXPERIMENT_ROOT / name
    dest.mkdir(exist_ok=True)
    predictions_path = dest / "predictions.jsonl"
    if (dest / "metrics.json").exists():
        return read_json(dest / "metrics.json")
    records = training_utils.read(predictions_path) if predictions_path.exists() else []
    initial = len(records)
    assert [x["qa_id"] for x in records] == [x["qa_id"] for x in rows[:initial]]
    start = time.monotonic()
    cache = {}
    stamp = tuple((n, id(p), p._version) for n, p in backend.model.named_parameters())
    for row in rows[initial:]:
        request = build_request(row, load_rgb(row, cache))
        with torch.no_grad():
            text = backend.generate(request, max_new_tokens=32, do_sample=False).text
        parsed = (parse_yn if row["question_type"] == "ynqa" else parse_mc)(text)
        target = gold[row["qa_id"]]
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
        training_utils.append(predictions_path, record)
        write_status(
            "evaluating",
            evaluation=name,
            rows=len(records),
            total_rows=len(rows),
            seconds=time.monotonic() - start,
            measured_rows=len(records) - initial,
        )
    assert stamp == tuple(
        (n, id(p), p._version) for n, p in backend.model.named_parameters()
    )
    metrics = dict(
        n=len(records),
        correct=sum(x["correct"] for x in records),
        accuracy=sum(x["correct"] for x in records) / len(records),
        parse_failures=sum(x["parsed"] is None for x in records),
        seconds=time.monotonic() - start,
    )
    training_utils.write(dest / "metrics.json", metrics)
    return metrics


def main():
    """Train QA residuals with one source rehearsal example per four QA examples."""
    lock = (EXPERIMENT_ROOT / "RUN.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    configure_reproducible_numeric_environment()
    set_global_seed(PLAN["seed"])
    torch.set_num_threads(4)
    for path, expected_hash in PLAN["input_hashes"].items():
        assert training_utils.sha(path) == expected_hash
    assert training_utils.sha(PLAN["parent"]) == PLAN["parent_sha256"]
    write_status("loading")
    backend = load_local_backend("molmo2_o_7b", device="cuda:0", dtype="bfloat16")
    backend.model.eval()
    backend.model.requires_grad_(False)
    assert backend.unified_physical_core is None and not backend._source_handles
    checkpoint = torch.load(PLAN["parent"], map_location="cpu", weights_only=False)
    assert checkpoint["seen"] == 5000
    parent_adapters = []
    for layer in [27, 28, 29, 30]:
        for leaf in ["self_attn.att_proj", "self_attn.attn_out"]:
            path = f"model.transformer.blocks.{layer}.{leaf}"
            parent, attr = path.rsplit(".", 1)
            owner = backend.model.get_submodule(parent)
            parent_adapter = base.LoRA(getattr(owner, attr))
            setattr(owner, attr, parent_adapter)
            with torch.no_grad():
                parent_adapter.A.copy_(checkpoint["adapter"][path + ".A"])
                parent_adapter.B.copy_(checkpoint["adapter"][path + ".B"])
            parent_adapter.requires_grad_(False)
            parent_adapters.append((path, parent_adapter))
    del checkpoint
    adapter_parameters = []
    handles = []
    for i, (path, parent_adapter) in enumerate(parent_adapters):
        rank = 30 if path.endswith("att_proj") else [33, 32, 32, 31][i // 2]
        update = OrdinaryUpdateResidual(
            parent_adapter.base.in_features,
            parent_adapter.base.out_features,
            rank,
            seed=PLAN["seed"] + i,
        ).to(device=parent_adapter.base.weight.device)
        parent_adapter.add_module("qa_update", update)

        def hook(module, args, out):
            return out + module.qa_update(args[0]).to(out)

        handles.append(parent_adapter.register_forward_hook(hook))
        adapter_parameters.extend(
            (path + ".qa_update." + n, p) for n, p in update.named_parameters()
        )
    assert sum(p.numel() for n, p in adapter_parameters) == PLAN["trainable_count"]
    assert {id(p) for p in backend.model.parameters() if p.requires_grad} == {
        id(p) for n, p in adapter_parameters
    }
    optimizer = torch.optim.AdamW(
        [p for n, p in adapter_parameters],
        lr=PLAN["lr"],
        weight_decay=PLAN["weight_decay"],
    )
    step = 0
    if (EXPERIMENT_ROOT / "LATEST.json").exists():
        latest = read_json(EXPERIMENT_ROOT / "LATEST.json")
        assert training_utils.sha(latest["path"]) == latest["sha256"]
        checkpoint = torch.load(latest["path"], map_location="cpu", weights_only=False)
        assert checkpoint["plan_sha256"] == training_utils.sha(
            EXPERIMENT_ROOT / "PLAN.json"
        )
        with torch.no_grad():
            for n, p in adapter_parameters:
                p.copy_(checkpoint["adapter"][n])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        restore_rng_state(checkpoint["rng"])
        del checkpoint
    trainable_ids = {id(p) for n, p in adapter_parameters}
    stamp = tuple(
        (n, id(p), p._version)
        for n, p in backend.model.named_parameters()
        if id(p) not in trainable_ids
    )
    training_utils.write(
        EXPERIMENT_ROOT / "ASSEMBLY.json",
        dict(
            parent=PLAN["parent"],
            parent_sha256=PLAN["parent_sha256"],
            parent_adapter_frozen=True,
            parent_adapter_active=True,
            base_frozen=True,
            no_mosder_components=True,
            trainable_count=sum(p.numel() for n, p in adapter_parameters),
            trainable_names=[n for n, p in adapter_parameters],
        ),
    )
    qa_rows = read_json(EXPERIMENT_ROOT / "qa_train.json")
    source_rows = read_json(EXPERIMENT_ROOT / "source512.json")
    # LOCAL_PATH: Training QA labels are read from the imported pilot module's data root.
    gold = {
        x["qa_id"]: x
        for x in read_json(pilot.ROOT / "data/OFFICIAL_TRAIN_GOLD.json")["items"]
    }
    # LOCAL_PATH: The imported utilities choose the data worker and its Python environment.
    worker = training_utils.Worker()
    source_dataset = training_utils.Dataset(source_rows, worker)
    cache = {}
    qa_losses = []
    source_losses = []
    optimizer.zero_grad(set_to_none=True)
    start = begin = time.monotonic()
    try:
        for position in range(step * 8, 2048):
            row = qa_rows[position]
            request = build_request(row, load_rgb(row, cache))
            score = backend.teacher_forced_loss(request, answer(row, gold))
            loss = score.negative_log_likelihood
            assert torch.isfinite(loss)
            qa_losses.append(float(loss.detach()))
            (loss / 8).backward()
            del loss, score
            if position % 4 == 0:
                example = source_dataset[position // 4]
                try:
                    score = backend.teacher_forced_loss(
                        base.request(example, True),
                        base.canonical_answer(example.state),
                    )
                    loss = score.negative_log_likelihood
                    assert torch.isfinite(loss)
                    source_losses.append(float(loss.detach()))
                    (0.125 * loss).backward()
                    del loss, score
                finally:
                    example.close()
            if position % 8 == 7:
                progress = step / 255
                warmup_fraction = 0.05
                scale = (
                    (step + 1) / math.ceil(256 * warmup_fraction)
                    if progress < warmup_fraction
                    else 0.1
                    + 0.9
                    * 0.5
                    * (
                        1
                        + math.cos(
                            math.pi
                            * (progress - warmup_fraction)
                            / (1 - warmup_fraction)
                        )
                    )
                )
                for group in optimizer.param_groups:
                    group["lr"] = PLAN["lr"] * scale
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    [p for n, p in adapter_parameters], 1.0
                )
                assert torch.isfinite(gradient_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                assert stamp == tuple(
                    (n, id(p), p._version)
                    for n, p in backend.model.named_parameters()
                    if id(p) not in trainable_ids
                )
                record = dict(
                    update=step,
                    total_updates=256,
                    qa_seen=position + 1,
                    source_seen=(position + 1) // 4,
                    qa_loss=sum(qa_losses) / len(qa_losses),
                    source_loss=sum(source_losses) / len(source_losses),
                    lr=optimizer.param_groups[0]["lr"],
                    grad_norm=float(gradient_norm),
                    seconds_per_update=time.monotonic() - begin,
                    elapsed_seconds=time.monotonic() - start,
                    updated_at_unix=time.time(),
                )
                training_utils.append(EXPERIMENT_ROOT / "TRAIN.jsonl", record)
                curve_path = EXPERIMENT_ROOT / "training_curve.csv"
                exists = curve_path.exists()
                with curve_path.open("a") as f:
                    writer = csv.DictWriter(f, fieldnames=list(record))
                    if not exists:
                        writer.writeheader()
                    writer.writerow(record)
                write_status(
                    "training",
                    **{k: v for k, v in record.items() if k != "updated_at_unix"},
                )
                print(json.dumps(record), flush=True)
                if step in [1, 5] or step % 32 == 0:
                    save_checkpoint(
                        adapter_parameters,
                        optimizer,
                        step,
                        EXPERIMENT_ROOT / f"checkpoint_{step:04d}.pt",
                    )
                qa_losses = []
                source_losses = []
                begin = time.monotonic()
        save_checkpoint(
            adapter_parameters, optimizer, step, EXPERIMENT_ROOT / "FINAL.pt"
        )
        training_utils.write(
            EXPERIMENT_ROOT / "TRAIN_COMPLETE.json",
            dict(updates=step, qa_seen=2048, source_seen=512, parent_frozen=True),
        )
        del optimizer
        cache.clear()
        results = {}
        results["train64"] = evaluate(backend, qa_rows[:64], gold, "train64")
        results["dev650"] = evaluate(
            backend, read_json(EXPERIMENT_ROOT / "qa_dev650.json"), gold, "dev650"
        )
        # LOCAL_PATH: External validation QA labels; provide this JSON on the target machine.
        validation_gold = {
            x["qa_id"]: x
            for x in read_json(
                "/root/autodl-tmp/mosder_reference_research_20260906/omnivchall_val_eval_v1/data/OFFICIAL_VAL_GOLD.json"
            )["items"]
        }
        results["val914"] = evaluate(
            backend,
            read_json(EXPERIMENT_ROOT / "qa_val914.json"),
            validation_gold,
            "val914",
        )
        training_utils.write(
            EXPERIMENT_ROOT / "COMPLETE.json",
            dict(
                metrics=results,
                checkpoint_sha256=training_utils.sha(EXPERIMENT_ROOT / "FINAL.pt"),
                parent_sha256=PLAN["parent_sha256"],
            ),
        )
        write_status("complete", metrics=results)
    finally:
        worker.close()
        backend.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        training_utils.write(
            EXPERIMENT_ROOT / "FAILURE.json",
            dict(error=str(error), traceback=traceback.format_exc()),
        )
        write_status("failed", error=str(error))
        raise
