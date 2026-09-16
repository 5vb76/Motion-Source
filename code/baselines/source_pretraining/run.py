"""Train ordinary LoRA on source motion labels and compare its QA readout."""

import csv
import fcntl
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
# LOCAL_PATH: Load training helpers and QA protocol code from external source directories.
spec = importlib.util.spec_from_file_location(
    "fourfield_utils", "/root/autodl-tmp/mosder_general_v2_molmo_20260911/common.py"
)
training_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training_utils)
sys.path.insert(0, "/root/autodl-tmp/molmo_raw_motion_qa_v3")
import qa_protocol as qa
import torch
from family_backends_v1 import load_local_backend
from mosder_fgr_runner_candidate_v3.checkpoint import (
    capture_rng_state,
    configure_reproducible_numeric_environment,
    load_method_parameter_state,
    restore_rng_state,
    set_global_seed,
)
from mosder_fgr_runner_candidate_v3.engine import all_method_parameters
from mosder_final_v1.family_backend import load_local_mosder_backend
from mosder_final_v1.language import canonical_answer, parse_four_line_language


def read_json(path):
    """Read a plan, split, or result document."""
    return json.loads(Path(path).read_text())


# LOCAL_PATH: Supply PLAN.json, TRAIN.json, VAL.json, and QA.json beside this script;
# PLAN["checkpoint"] must reference the MoSDeR weights used for comparison.
PLAN = read_json(EXPERIMENT_ROOT / "PLAN.json")


def write_status(task, done, total, seconds=0, **extra):
    training_utils.write(
        EXPERIMENT_ROOT / "STATUS.json",
        dict(
            task=task,
            done=done,
            total=total,
            seconds=seconds,
            updated_at_unix=time.time(),
            pid=os.getpid(),
            **extra,
        ),
    )


class LowRankResidual(torch.nn.Module):
    """Rank-32 residual with zero-initialized output; keep A/B checkpoint keys."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.A = torch.nn.Parameter(
            torch.empty(
                32, base.in_features, device=base.weight.device, dtype=torch.float32
            )
        )
        self.B = torch.nn.Parameter(
            torch.zeros(
                base.out_features, 32, device=base.weight.device, dtype=torch.float32
            )
        )
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        with torch.autocast(device_type="cuda", enabled=False):
            residual = torch.nn.functional.linear(
                torch.nn.functional.linear(x.float(), self.A), self.B
            )
        return self.base(x) + residual.to(x.dtype)


def request(example, ordinary):
    """Expose normalized boxes in text for the ordinary LoRA baseline."""
    if not ordinary:
        return example.request
    validated = example.request.validated()
    normalized_boxes = [
        [
            round(
                float(box[j]) / (validated.width if j % 2 == 0 else validated.height),
                4,
            )
            for j in range(4)
        ]
        for box in validated.boxes_xyxy
    ]
    return replace(
        example.request,
        prompt=example.request.prompt
        + "\nOracle target box track, normalized [x_min,y_min,x_max,y_max], frames 1–20: "
        + json.dumps(normalized_boxes, separators=(",", ":")),
    )


def metrics(records):
    """Aggregate QA correctness and parsed motion-state accuracy."""

    def stats(group_records):
        return dict(
            n=len(group_records),
            correct=sum(x["correct"] for x in group_records),
            accuracy=sum(x["correct"] for x in group_records) / len(group_records),
        )

    result = stats(records)
    result["four_state_accuracy"] = sum(
        x["predicted_state"] == x["state"] for x in records
    ) / len(records)
    result["parse_failures"] = sum(x["predicted_state"] is None for x in records)
    result["per_kind"] = {
        k: stats([x for x in records if x["kind"] == k])
        for k in ["camera", "object", "joint"]
    }
    result["per_source"] = {
        k: stats([x for x in records if x["source"] == k])
        for k in sorted({x["source"] for x in records})
    }
    result["source_macro_accuracy"] = sum(
        x["accuracy"] for x in result["per_source"].values()
    ) / len(result["per_source"])
    return result


def evaluate(backend, worker, rows, name, ordinary):
    """Resume evaluation in manifest order and verify weights stay frozen."""
    dest = EXPERIMENT_ROOT / name
    dest.mkdir(exist_ok=True)
    if (dest / "metrics.json").exists():
        return
    specs = {x["case_id"]: x["spec"] for x in read_json(EXPERIMENT_ROOT / "QA.json")}
    dataset = training_utils.Dataset(rows, worker)
    predictions_path = dest / "predictions.jsonl"
    group_records = (
        training_utils.read(predictions_path) if predictions_path.exists() else []
    )
    initial = len(group_records)
    assert [x["case_id"] for x in group_records] == [
        x["case_id"] for x in rows[:initial]
    ]
    backend.model.eval()
    start = time.monotonic()
    stamp = tuple((n, id(p), p._version) for n, p in backend.model.named_parameters())
    for i in range(initial, len(rows)):
        example = dataset[i]
        try:
            with torch.no_grad():
                text = backend.generate(
                    request(example, ordinary), max_new_tokens=96, do_sample=False
                ).text
            try:
                predicted_state = parse_four_line_language(text).state
            except (ValueError, RuntimeError):
                predicted_state = None
            question = specs[example.case_id]
            mapped = (
                qa.answer(question, predicted_state)
                if predicted_state in qa.STATES
                else None
            )
            gold = qa.answer(question, example.state)
            record = dict(
                case_id=example.case_id,
                source=example.source_dataset,
                state=example.state,
                text=text,
                predicted_state=predicted_state,
                display_state=predicted_state.replace("_", " ")
                if predicted_state
                else None,
                kind=question["kind"],
                question=question["question"],
                options=question["options"],
                option_texts=[x.replace("_", " ") for x in question["options"]]
                if question["options"]
                else None,
                predicted_answer=mapped,
                gold=gold,
                correct=mapped == gold,
            )
            group_records.append(record)
            training_utils.append(predictions_path, record)
        finally:
            example.close()
        write_status(
            name,
            i + 1,
            len(rows),
            time.monotonic() - start,
            measured_units=i + 1 - initial,
        )
        if (i + 1) % 20 == 0:
            print(json.dumps(dict(task=name, done=i + 1, total=len(rows))), flush=True)
    assert stamp == tuple(
        (n, id(p), p._version) for n, p in backend.model.named_parameters()
    )
    result = metrics(group_records)
    result.update(seconds=time.monotonic() - start, weights_unchanged=True)
    training_utils.write(dest / "metrics.json", result)
    print(json.dumps(dict(task=name, metrics=result)), flush=True)


def save_checkpoint(adapter_parameters, optimizer, step, examples_seen):
    """Save adapter, optimizer, and RNG state for an exact resume."""
    path = EXPERIMENT_ROOT / f"checkpoint_{step:04d}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save(
        dict(
            adapter={n: p.detach().cpu() for n, p in adapter_parameters},
            optimizer=optimizer.state_dict(),
            step=step,
            seen=examples_seen,
            rng=capture_rng_state(),
            plan_sha256=training_utils.sha(EXPERIMENT_ROOT / "PLAN.json"),
        ),
        tmp,
    )
    os.replace(tmp, path)
    training_utils.write(
        EXPERIMENT_ROOT / "LATEST.json",
        dict(
            path=str(path),
            step=step,
            seen=examples_seen,
            sha256=training_utils.sha(path),
        ),
    )


def main():
    """Compare MoSDeR with two consecutive halves of source-only LoRA training."""
    lock = (EXPERIMENT_ROOT / "RUN.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    configure_reproducible_numeric_environment()
    set_global_seed(PLAN["seed"])
    torch.set_num_threads(4)
    assert training_utils.sha(PLAN["checkpoint"]) == PLAN["checkpoint_sha256"]
    # LOCAL_PATH: The imported utilities choose the data worker and its Python environment.
    worker = training_utils.Worker()
    validation_rows = read_json(EXPERIMENT_ROOT / "VAL.json")
    try:
        if not (EXPERIMENT_ROOT / "mosder_val1200/metrics.json").exists():
            write_status("mosder_val1200", 0, 1200, loading=True)
            backend = load_local_mosder_backend(
                "molmo2_o_7b", device="cuda:0", dtype="bfloat16"
            )
            checkpoint = torch.load(
                PLAN["checkpoint"], map_location="cpu", weights_only=False
            )
            load_method_parameter_state(
                all_method_parameters(backend), checkpoint["method_parameter_state"]
            )
            del checkpoint
            backend.configure_mosder_stage("FROZEN")
            backend.model.requires_grad_(False)
            evaluate(backend, worker, validation_rows, "mosder_val1200", False)
            backend.close()
            del backend
            torch.cuda.empty_cache()
        write_status("train_half1", 0, 2500, loading=True)
        backend = load_local_backend("molmo2_o_7b", device="cuda:0", dtype="bfloat16")
        backend.model.eval()
        backend.model.requires_grad_(False)
        assert backend.unified_physical_core is None and not backend._source_handles
        adapter_parameters = []
        for layer in PLAN["layers"]:
            for leaf in PLAN["leaves"]:
                path = f"model.transformer.blocks.{layer}.{leaf}"
                parent, attr = path.rsplit(".", 1)
                owner = backend.model.get_submodule(parent)
                base = getattr(owner, attr)
                adapter_layer = LowRankResidual(base)
                setattr(owner, attr, adapter_layer)
                adapter_parameters.extend(
                    [(path + ".A", adapter_layer.A), (path + ".B", adapter_layer.B)]
                )
        assert {id(p) for p in backend.model.parameters() if p.requires_grad} == {
            id(p) for n, p in adapter_parameters
        }
        training_utils.write(
            EXPERIMENT_ROOT / "ASSEMBLY.json",
            dict(
                trainable_parameters=sum(p.numel() for n, p in adapter_parameters),
                names=[n for n, p in adapter_parameters],
                initialization="raw pretrained Molmo; zero-output B; no MoSDeR weights",
                rank=32,
                alpha=32,
            ),
        )
        optimizer = torch.optim.AdamW(
            [p for n, p in adapter_parameters], lr=PLAN["lr"], weight_decay=0.01
        )
        examples_seen = step = 0
        if (EXPERIMENT_ROOT / "LATEST.json").exists():
            latest = read_json(EXPERIMENT_ROOT / "LATEST.json")
            assert training_utils.sha(latest["path"]) == latest["sha256"]
            checkpoint = torch.load(
                latest["path"], map_location="cpu", weights_only=False
            )
            assert checkpoint["plan_sha256"] == training_utils.sha(
                EXPERIMENT_ROOT / "PLAN.json"
            )
            with torch.no_grad():
                for n, p in adapter_parameters:
                    p.copy_(checkpoint["adapter"][n])
            optimizer.load_state_dict(checkpoint["optimizer"])
            step = checkpoint["step"]
            examples_seen = checkpoint["seen"]
            restore_rng_state(checkpoint["rng"])
            del checkpoint
        frozen = tuple(
            (n, id(p), p._version)
            for n, p in backend.model.named_parameters()
            if not p.requires_grad
        )
        train = read_json(EXPERIMENT_ROOT / "TRAIN.json")
        dataset = training_utils.Dataset(train, worker)
        optimizer.zero_grad(set_to_none=True)
        for half in [1, 2]:
            half_start = (half - 1) * 2500
            half_end = half * 2500
            start = time.monotonic()
            initial = max(examples_seen, half_start)
            for batch_start in range(initial, half_end, 8):
                begin = time.monotonic()
                batch = list(range(batch_start, min(batch_start + 8, half_end)))
                losses = []
                for idx in batch:
                    example = dataset[idx]
                    try:
                        score = backend.teacher_forced_loss(
                            request(example, True), canonical_answer(example.state)
                        )
                        loss = score.negative_log_likelihood
                        assert torch.isfinite(loss)
                        losses.append(float(loss.detach()))
                        (loss / len(batch)).backward()
                        del loss, score
                    finally:
                        example.close()
                scale = (
                    (step + 1) / 32
                    if step < 32
                    else 0.1 + 0.45 * (1 + math.cos(math.pi * (step - 32) / (625 - 32)))
                )
                for group in optimizer.param_groups:
                    group["lr"] = PLAN["lr"] * scale
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for n, p in adapter_parameters], 1.0
                )
                assert torch.isfinite(norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                examples_seen = batch[-1] + 1
                assert frozen == tuple(
                    (n, id(p), p._version)
                    for n, p in backend.model.named_parameters()
                    if not p.requires_grad
                )
                record = dict(
                    step=step,
                    half=half,
                    seen=examples_seen,
                    loss=sum(losses) / len(losses),
                    lr=optimizer.param_groups[0]["lr"],
                    grad_norm=float(norm),
                    batch_size=len(batch),
                    seconds_per_update=time.monotonic() - begin,
                    time_unix=time.time(),
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
                    f"train_half{half}",
                    examples_seen - half_start,
                    2500,
                    time.monotonic() - start,
                    measured_units=examples_seen - initial,
                    step=step,
                    loss=record["loss"],
                )
                print(json.dumps(record), flush=True)
                if step == 1 or step % 32 == 0 or examples_seen == half_end:
                    save_checkpoint(adapter_parameters, optimizer, step, examples_seen)
            training_utils.write(
                EXPERIMENT_ROOT / f"train_half{half}_COMPLETE.json",
                dict(seen=half_end, step=313 * half),
            )
            rows = (
                [r for r in validation_rows if r["case_id"] in set(PLAN["val400_ids"])]
                if half == 1
                else validation_rows
            )
            evaluate(
                backend,
                worker,
                rows,
                "lora_val400" if half == 1 else "lora_val1200",
                True,
            )
        backend.close()
        training_utils.write(
            EXPERIMENT_ROOT / "COMPLETE.json",
            {
                name: read_json(EXPERIMENT_ROOT / name / "metrics.json")
                for name in ["mosder_val1200", "lora_val400", "lora_val1200"]
            },
        )
        write_status("complete", 1, 1)
    finally:
        worker.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        training_utils.write(
            EXPERIMENT_ROOT / "FAILURE.json",
            dict(error=str(error), traceback=traceback.format_exc()),
        )
        write_status("failed", 0, 1, error=str(error))
        raise
