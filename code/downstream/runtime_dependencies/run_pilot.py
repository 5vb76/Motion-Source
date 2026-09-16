"""Run the fixed three-arm, three-seed QA continuation pilot.

Each run updates the same parent for 32 steps, then evaluates QA and source
panels. The helpers are also imported by the full downstream training scripts.
"""

import json
import sys
import time
import math
import random
import traceback
from pathlib import Path
import numpy as np
import torch

OUTPUT_DIR = Path(__file__).resolve().parent
# LOCAL_PATH: External training helpers/data are required; their imports take precedence here.
ROOT = Path(
    "/root/autodl-tmp/mosder_reference_research_20260906/internal_source_research_20260908"
)
sys.path.insert(0, str(ROOT))
import research_train as research_training
from source_pairs_common import load_rgb, source_prompts, binding, sha, save
from soft_query_backend import VideoQuery20Request
from mosder_short_event_qa_pilot_v1.runtime import load_runtime
from qa_update_controls import install_same_parent_ordinary_update
from mosder_fgr_runner_candidate_v3.checkpoint import load_method_parameter_state

runtime_adapter = research_training.a


def read_json(path):
    return json.loads(Path(path).read_text())


def seed_everything(seed_value):
    """Seed Python, NumPy, and all PyTorch devices before each pilot arm."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def freeze_backend(backend):
    """Disable model gradients and select evaluation mode."""
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    backend.model.eval()
    backend.query_router.eval()


def evaluate(backend, output_dir, qa_rows, panels):
    """Write QA and source-panel predictions while checking frozen parameters."""
    freeze_backend(backend)
    start = time.monotonic()
    frozen_parameter_stamp = runtime_adapter.parameter_stamp(backend.model)
    predictions = []
    pixel_cache = {}
    for row in qa_rows:
        item = runtime_adapter.OriginalInput.from_row(row)
        if item.video_id not in pixel_cache:
            pixel_cache[item.video_id] = runtime_adapter.load_pixels(item)
        request = runtime_adapter.native_request(
            item, pixel_cache[item.video_id], item.prompt
        )
        with torch.inference_mode():
            answer = backend.generate(request, **runtime_adapter.GENERATION)
        prediction = {
            "qa_id": row["qa_id"],
            "video_id": row["video_id"],
            "question_type": row["question_type"],
            "text": answer.text,
            "generated_token_ids": list(answer.generated_token_ids),
            "rgb_sha256": row["rgb_sha256"],
        }
        predictions.append(prediction)
        with (output_dir / "QA_PREDICTIONS.jsonl").open("a") as f:
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        if len(predictions) % 32 == 0:
            print(
                json.dumps(
                    {
                        "run": output_dir.name,
                        "eval_qa": len(predictions),
                        "seconds": time.monotonic() - start,
                    }
                ),
                flush=True,
            )
    save(output_dir / "QA_PREDICTIONS.json", {"items": predictions})
    del pixel_cache
    for panel, rows in panels.items():
        predictions = []
        for row in rows:
            rgb, timestamps = load_rgb(row) if panel == "source56" else load_target(row)
            record = {
                "candidate_id": row["candidate_id"],
                "pair_id": row["pair_id"],
                "sequence_id": row["sequence_id"],
                "rgb_sha256": row["rgb_sha256"],
                "query_target": row["query_target"],
                "binary_native": {},
            }
            if panel == "source56":
                record["input_binding"] = binding(row)
            for factor, prompt in source_prompts(row["query_target"]).items():
                request = VideoQuery20Request(
                    tuple(rgb),
                    timestamps,
                    prompt,
                    row["candidate_id"] + "/" + factor,
                    prompt,
                )
                with torch.inference_mode():
                    answer = backend.generate(request, **runtime_adapter.GENERATION)
                record["binary_native"][factor] = {
                    "prompt": prompt,
                    "success": True,
                    "text": answer.text,
                    "generated_token_ids": list(answer.generated_token_ids),
                }
            predictions.append(record)
            with (output_dir / (panel + "_PREDICTIONS.jsonl")).open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        save(output_dir / (panel + "_PREDICTIONS.json"), {"items": predictions})
        print(
            json.dumps(
                {
                    "run": output_dir.name,
                    "panel": panel,
                    "count": len(predictions),
                    "seconds": time.monotonic() - start,
                }
            ),
            flush=True,
        )
    assert runtime_adapter.parameter_stamp(backend.model) == frozen_parameter_stamp
    save(
        output_dir / "EVAL_COMPLETE.json",
        {
            "seconds": time.monotonic() - start,
            "qa_count": len(qa_rows),
            "source56": len(panels.get("source56", [])),
            "target26": len(panels.get("target26", [])),
            "inference_reads_gold": False,
            "prediction_sha256": {
                prediction.name: sha(prediction)
                for prediction in output_dir.glob("*PREDICTIONS.json")
            },
        },
    )


def load_target(row):
    """Load a target-switch clip and verify its archive and RGB hashes."""
    # LOCAL_PATH: Each input row must point to an available local RGB NPZ archive.
    assert sha(row["rgb20_npz_path"]) == row["rgb20_npz_sha256"]
    with np.load(row["rgb20_npz_path"], allow_pickle=False) as archive:
        rgb = archive["rgb"].copy()
        timestamps = archive["timestamps_ns"].tolist()
    assert timestamps == row["timestamps_ns"] and len(rgb) == 20
    import hashlib

    assert hashlib.sha256(rgb.tobytes()).hexdigest() == row["rgb_sha256"]
    return rgb, timestamps


def main():
    """Train and evaluate the nine runs declared in PILOT_PLAN.json."""
    torch.set_num_threads(4)
    # LOCAL_PATH: Supply PILOT_PLAN.json and QA/TARGET_SWITCH input manifests beside this script.
    plan = read_json(OUTPUT_DIR / "PILOT_PLAN.json")
    assert all(sha(k) == v for k, v in plan["input_sha256"].items())
    train = read_json(OUTPUT_DIR / "QA_TRAIN_INPUTS.json")["items"]
    dev = read_json(OUTPUT_DIR / "QA_DEV_INPUTS.json")["items"]
    trainable_parameter_ids = {r["qa_id"] for r in train}
    # LOCAL_PATH: Training gold and source manifests below are read from external ROOT/data.
    gold = {
        r["qa_id"]: r
        for r in read_json(ROOT / "data/OFFICIAL_TRAIN_GOLD.json")["items"]
        if r["qa_id"] in trainable_parameter_ids
    }
    sources = read_json(ROOT / "data/source_rehearsal128/PREPARED.json")["items"]
    panels = {
        "source56": read_json(
            ROOT
            / "data/identity_balanced_source_candidate/qualified56_v1/INPUTS_WITHOUT_LABELS.json"
        )["items"],
        "target26": read_json(OUTPUT_DIR / "TARGET_SWITCH_INPUTS.json")["items"],
    }
    started = time.monotonic()
    runs_dir = OUTPUT_DIR / "pilot"
    runs_dir.mkdir(exist_ok=False)
    save(
        runs_dir / "EXECUTION.json",
        {
            "plan_sha256": sha(OUTPUT_DIR / "PILOT_PLAN.json"),
            "code_sha256": sha(__file__),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
            "parent": str(runtime_adapter.PARENT_PATH),
            "parent_sha256": runtime_adapter.PARENT_SHA,
            "grounder_sha256": runtime_adapter.GROUND_SHA,
        },
    )
    with load_runtime(
        family="molmo2_o_7b", arm="raw_vlm", device="cuda:0", dtype=None
    ) as loaded:
        backend, identity = runtime_adapter.open_frozen_backend(loaded, internal=True)
        all_method_parameters = research_training.method(backend)
        initial = {
            parameter_name: parameter.detach().cpu().clone()
            for parameter_name, parameter in all_method_parameters
        }
        for seed_value in plan["seeds"]:
            for arm in plan["arms"]:
                output_dir = runs_dir / f"{arm}_seed{seed_value}"
                output_dir.mkdir(exist_ok=False)
                seed_everything(seed_value)
                freeze_backend(backend)
                load_method_parameter_state(all_method_parameters, initial)
                ordinary_update = None
                if arm.startswith("ordinary"):
                    ordinary_update = install_same_parent_ordinary_update(
                        backend, seed=seed_value
                    )
                    trainable_parameters = list(ordinary_update.named_parameters())
                else:
                    trainable_parameters = [
                        (parameter_name, parameter)
                        for parameter_name, parameter in all_method_parameters
                        if not parameter_name.startswith("decision.")
                    ]
                    for parameter_name, parameter in trainable_parameters:
                        parameter.requires_grad_(True)
                backend.model.eval()
                backend.query_router.requires_grad_(False)
                trainable_parameter_ids = {
                    id(parameter) for parameter_name, parameter in trainable_parameters
                }
                assert {
                    id(parameter)
                    for parameter in backend.model.parameters()
                    if parameter.requires_grad
                } == trainable_parameter_ids
                frozen_parameter_stamp = tuple(
                    (parameter_name, id(parameter), parameter._version)
                    for parameter_name, parameter in backend.model.named_parameters()
                    if id(parameter) not in trainable_parameter_ids
                )
                order = torch.randperm(
                    len(train), generator=torch.Generator().manual_seed(seed_value)
                ).tolist()
                source_order = research_training.balanced_source_order(
                    sources, seed_value
                )
                save(
                    output_dir / "START.json",
                    {
                        "arm": arm,
                        "seed": seed_value,
                        "plan_sha256": sha(OUTPUT_DIR / "PILOT_PLAN.json"),
                        "parameters": sum(
                            parameter.numel()
                            for parameter_name, parameter in trainable_parameters
                        ),
                        "parameter_names": [
                            parameter_name
                            for parameter_name, parameter in trainable_parameters
                        ],
                        "qa_order": [train[i]["qa_id"] for i in order],
                        "source_order": [
                            sources[i]["case_id"] for i in source_order[:64]
                        ]
                        if arm != "ordinary_qa"
                        else [],
                        "existing_source_path": "retained and frozen for ordinary arms; updated for typed arm",
                        "no_checkpoint_selection": True,
                    },
                )
                optimizer = torch.optim.AdamW(
                    [parameter for parameter_name, parameter in trainable_parameters],
                    lr=plan["learning_rate"],
                    weight_decay=0.01,
                )
                optimizer.zero_grad(set_to_none=True)
                arm_started = time.monotonic()
                updates = 0
                pixel_cache = {}
                first = True
                for position, index in enumerate(order):
                    row = train[index]
                    item = runtime_adapter.OriginalInput.from_row(row)
                    if item.video_id not in pixel_cache:
                        if len(pixel_cache) >= 8:
                            pixel_cache.pop(next(iter(pixel_cache)))
                        pixel_cache[item.video_id] = runtime_adapter.load_pixels(item)
                    request = runtime_adapter.native_request(
                        item, pixel_cache[item.video_id], item.prompt
                    )
                    target = gold[row["qa_id"]]
                    answer = (
                        ("是" if target["yn_answer"] == "yes" else "否")
                        if row["question_type"] == "ynqa"
                        else target["mc_answer"]
                    )
                    loss = research_training.native_loss(backend, request, answer)
                    assert torch.isfinite(loss)
                    log = {
                        "position": position,
                        "qa_id": row["qa_id"],
                        "qa_nll": float(loss.detach()),
                    }
                    (loss / 8).backward()
                    del loss
                    if arm != "ordinary_qa" and position % 4 == 0:
                        source_row = sources[source_order[(position // 4) % 128]]
                        with research_training.source_request(
                            backend, source_row
                        ) as source_request:
                            loss = research_training.native_loss(
                                backend, source_request, source_row["canonical_answer"]
                            )
                        assert torch.isfinite(loss)
                        log.update(
                            source_case=source_row["case_id"],
                            source_nll=float(loss.detach()),
                        )
                        (0.125 * loss).backward()
                        del loss
                    if first:
                        gradient_norms = {
                            parameter_name: float(parameter.grad.norm())
                            if parameter.grad is not None
                            else None
                            for parameter_name, parameter in trainable_parameters
                        }
                        assert all(
                            v is None or math.isfinite(v)
                            for v in gradient_norms.values()
                        ) and any(
                            v is not None and v > 0 for v in gradient_norms.values()
                        )
                        save(output_dir / "FIRST_GRADIENT.json", gradient_norms)
                        first = False
                    if position % 8 == 7:
                        progress_fraction = updates / 31
                        warmup_fraction = 0.05
                        scale = (
                            (updates + 1) / 2
                            if progress_fraction < warmup_fraction
                            else 0.1
                            + 0.9
                            * 0.5
                            * (
                                1
                                + math.cos(
                                    math.pi
                                    * (progress_fraction - warmup_fraction)
                                    / (1 - warmup_fraction)
                                )
                            )
                        )
                        for group in optimizer.param_groups:
                            group["lr"] = plan["learning_rate"] * scale
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
                        updates += 1
                        assert (
                            tuple(
                                (parameter_name, id(parameter), parameter._version)
                                for parameter_name, parameter in backend.model.named_parameters()
                                if id(parameter) not in trainable_parameter_ids
                            )
                            == frozen_parameter_stamp
                        )
                        log.update(
                            update=updates,
                            grad_norm=float(norm),
                            lr=optimizer.param_groups[0]["lr"],
                        )
                    log["seconds"] = time.monotonic() - arm_started
                    with (output_dir / "TRAIN.jsonl").open("a") as f:
                        f.write(json.dumps(log) + "\n")
                    if (position + 1) % 32 == 0:
                        print(
                            json.dumps(
                                {
                                    "run": output_dir.name,
                                    "train_completed": position + 1,
                                    "seconds": time.monotonic() - arm_started,
                                }
                            ),
                            flush=True,
                        )
                assert updates == 32
                payload = {
                    "arm": arm,
                    "seed": seed_value,
                    "method_state": {
                        parameter_name: parameter.detach().cpu().clone()
                        for parameter_name, parameter in all_method_parameters
                    },
                    "ordinary_state": ordinary_update.state_dict()
                    if ordinary_update
                    else None,
                    "updates": updates,
                    "qa_presentations": 256,
                    "source_presentations": 64 if arm != "ordinary_qa" else 0,
                    "parent_sha256": runtime_adapter.PARENT_SHA,
                    "router_sha256": runtime_adapter.GROUND_SHA,
                    "short_pilot_complete": True,
                    "not_full5701_endpoint": True,
                    "plan_sha256": sha(OUTPUT_DIR / "PILOT_PLAN.json"),
                }
                torch.save(payload, output_dir / "FINAL.pt")
                save(
                    output_dir / "TRAIN_COMPLETE.json",
                    {
                        "updates": updates,
                        "seconds": time.monotonic() - arm_started,
                        "checkpoint_sha256": sha(output_dir / "FINAL.pt"),
                        "frozen_base_unchanged": True,
                    },
                )
                del optimizer, pixel_cache
                evaluate(backend, output_dir, dev, panels)
                if ordinary_update:
                    ordinary_update.close()
                save(
                    output_dir / "COMPLETE.json",
                    {
                        "seconds": time.monotonic() - arm_started,
                        "scope": plan["claim_scope"],
                    },
                )
                torch.cuda.empty_cache()
    save(
        runs_dir / "COMPLETE.json",
        {
            "runs": 9,
            "seconds": time.monotonic() - started,
            "all_fixed_short_endpoints": True,
        },
    )


rt = research_training
OUT = OUTPUT_DIR
read = read_json
seed = seed_everything
frozen = freeze_backend


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        save(
            OUTPUT_DIR / f"PILOT_FAILURE_{time.time_ns()}.json",
            {"traceback": traceback.format_exc()},
        )
        raise
