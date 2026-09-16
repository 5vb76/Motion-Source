"""Matched native QA/source training of original vs contextual differences.

Only public TRAIN-origin QA and authorized source TRAIN are read. Downstream
val/test questions, answers, predictions and per-item rules are not read.
"""

from __future__ import annotations
import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch

RESEARCH_DIR = Path(__file__).resolve().parent
sys.path.insert(
    0, str(RESEARCH_DIR.parent / "omnivchall_source_event_diagnostic_v1/runtime")
)
import adapter as runtime_adapter

runtime_adapter.ensure_import_paths()
from contextual_difference import install_contextual_difference


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temp.replace(path)


def collect_method_parameters(backend):
    """Return named MoSDeR method tensors for training or checkpoint loading."""
    from mosder_fgr_runner_candidate_v3.engine import all_method_parameters

    return tuple(all_method_parameters(backend))


def balanced_source_order(sources, seed):
    """Fixed four-state blocks without sorted-state training runs."""
    rng = random.Random(seed + 1)
    buckets = {
        state: [i for i, r in enumerate(sources) if r["state"] == state]
        for state in ("neither", "camera_only", "object_only", "both")
    }
    assert all(len(v) == 32 for v in buckets.values())
    for values in buckets.values():
        rng.shuffle(values)
    order = []
    for j in range(32):
        states = list(buckets)
        rng.shuffle(states)
        order.extend(buckets[state][j] for state in states)
    assert len(order) == len(set(order)) == 128
    return order


def native_loss(backend, request, text):
    """Compute answer loss through the full-language factor route."""
    from family_backends_v1 import BaseNativeBackend
    from mosder_final_v1.routing import FactorRoute

    with backend.factor_route_context(FactorRoute.FULL_LANGUAGE):
        result = BaseNativeBackend.teacher_forced_loss(backend, request, text)
    backend._mosder_route_success_counts[FactorRoute.FULL_LANGUAGE] += 1
    return result.negative_log_likelihood


@contextlib.contextmanager
def source_request(backend, row):
    """Temporarily pool source examples with TRAIN-only oracle box weights."""
    from soft_query_backend import VideoQuery20Request
    from soft_query_router import (
        box_grid_occupancy,
        pool_soft_sources,
        SoftRoutingOutput,
    )

    assert row["split"] == "TRAIN" and row["training_oracle_boxes_only"]
    # LOCAL_PATH: Prepared source rows must point to local RGB/box NPZ archives.
    path = Path(row["rgb20_npz_path"])
    assert runtime_adapter.sha(path) == row["rgb20_npz_sha256"]
    with np.load(path, allow_pickle=False) as archive:
        rgb = archive["rgb"].copy()
        times = archive["timestamps_ns"].tolist()
        boxes = archive["oracle_boxes_xyxy"].copy()
    height, width = rgb.shape[1:3]
    target_weights = torch.stack(
        [
            box_grid_occupancy(
                box / np.array([width, height, width, height]), 9, 9, device="cuda"
            )
            for box in boxes
        ]
    )
    previous = backend.query_router.forward

    def oracle_pool(tokens, query_embedding, *, mode="query"):
        target, context, units, unit_weights = pool_soft_sources(tokens, target_weights)
        return SoftRoutingOutput(
            torch.zeros_like(target_weights),
            target_weights,
            1 - target_weights,
            target,
            context,
            units,
            unit_weights,
        )

    backend.query_router.forward = oracle_pool
    prompt = row["prompt"]
    request = VideoQuery20Request(
        tuple(rgb), times, prompt, "source_train/" + row["case_id"], prompt
    )
    try:
        yield request
    finally:
        backend.query_router.forward = previous


def load_candidate(backend, path, *, for_training_resume=False, method_parameters=None):
    """Check checkpoint provenance and restore its declared difference mode."""
    from mosder_fgr_runner_candidate_v3.checkpoint import load_method_parameter_state

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert (
        payload["parent_sha256"] == runtime_adapter.PARENT_SHA
        and payload["router_sha256"] == runtime_adapter.GROUND_SHA
    )
    assert payload["mechanism"] in ("original", "relative")
    if not for_training_resume:
        assert (
            payload.get("training_complete") is True
            and payload.get("smoke_only") is False
        ), "evaluation requires the declared full-training endpoint"
    install_contextual_difference(backend, enabled=payload["mechanism"] == "relative")
    values = (
        collect_method_parameters(backend)
        if method_parameters is None
        else tuple(method_parameters)
    )
    load_method_parameter_state(values, payload["method_state"])
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism", choices=["original", "relative"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assert (
        os.environ.get("CUDA_VISIBLE_DEVICES")
        == "GPU-d4f2be0f-222f-82e3-450e-90eadde46613"
    )
    # LOCAL_PATH: Supply ROUND1_PROTOCOL.json and the data/ manifests below beside this module.
    config = json.loads((RESEARCH_DIR / "ROUND1_PROTOCOL.json").read_text())
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    prepared = json.loads(
        (RESEARCH_DIR / "data/PREPARED_TRAIN_WITHOUT_ANSWERS.json").read_text()
    )["items"]
    partitions = json.loads(
        (RESEARCH_DIR / "data/TRAIN_ONLY_PARTITIONS.json").read_text()
    )
    # This field is fixed by the independently prepared TRAIN-origin split.
    train_video_ids = set(partitions["partitions"]["train"]["video_ids"])
    rows = [r for r in prepared if r["video_id"] in train_video_ids]
    gold = json.loads((RESEARCH_DIR / "data/OFFICIAL_TRAIN_GOLD.json").read_text())[
        "items"
    ]
    gold_by_id = {r["qa_id"]: r for r in gold}
    assert len(rows) == 5702 and len({r["qa_id"] for r in rows}) == 5702
    assert all(r["qa_id"] in gold_by_id for r in rows)
    malformed = [
        r
        for r in rows
        if (
            gold_by_id[r["qa_id"]].get("yn_answer") not in ("yes", "no")
            if r["question_type"] == "ynqa"
            else gold_by_id[r["qa_id"]].get("mc_answer") not in ("A", "B", "C")
        )
    ]
    assert [r["qa_id"] for r in malformed] == ["m_ynqa_id:420"]
    # Preserve malformed official 'ye' in source gold. It has no valid target
    # for supervised CE; it is never coerced to 'no' or corrected from val.
    rows = [r for r in rows if r["qa_id"] != "m_ynqa_id:420"]
    assert len(rows) == 5701
    source_data = json.loads(
        (RESEARCH_DIR / "data/source_rehearsal128/PREPARED.json").read_text()
    )
    sources = source_data["items"]
    assert len(sources) == 128 and all(
        r["prepared"] and r["split"] == "TRAIN" for r in sources
    )
    source_order = balanced_source_order(sources, seed)
    order = torch.randperm(
        len(rows), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    if args.smoke:
        order = order[:2]
    output_dir = RESEARCH_DIR / "training" / args.name
    if not args.resume:
        output_dir.mkdir(parents=True, exist_ok=False)
    else:
        assert output_dir.is_dir()
    plan = {
        "config": config,
        "mechanism": args.mechanism,
        "smoke_only": args.smoke,
        "qa_presentations": len(order),
        "excluded_answer_loss_only": [
            {
                "qa_id": "official_train/m_ynqa_id:420",
                "official_target": "ye",
                "reason": "malformed binary TRAIN target; original record preserved",
            }
        ],
        "source_presentations": math.ceil(len(order) / config["source_every_qa"]),
        "source_order_case_ids": [sources[i]["case_id"] for i in source_order],
        "source_presentation_state_counts": dict(
            Counter(
                sources[source_order[j % 128]]["state"]
                for j in range(math.ceil(len(order) / config["source_every_qa"]))
            )
        ),
        "input_hashes": {
            str(p): runtime_adapter.sha(p)
            for p in [
                RESEARCH_DIR / "data/PREPARED_TRAIN_WITHOUT_ANSWERS.json",
                RESEARCH_DIR / "data/OFFICIAL_TRAIN_GOLD.json",
                RESEARCH_DIR / "data/TRAIN_ONLY_PARTITIONS.json",
                RESEARCH_DIR / "data/source_rehearsal128/PREPARED.json",
            ]
        },
        "code_hashes": {
            str(p): runtime_adapter.sha(p)
            for p in [Path(__file__), RESEARCH_DIR / "contextual_difference.py"]
        },
        "vision_execution": "original native forward; no feature cache",
        "native_backbone_frozen": True,
        "oracle_input_at_inference": False,
        "val_test_labels_read": False,
        "order_qa_ids": [rows[i]["qa_id"] for i in order],
    }
    if not args.resume:
        save(output_dir / "PLAN.json", plan)
    else:
        assert json.loads((output_dir / "PLAN.json").read_text()) == plan, (
            "resume protocol drift"
        )
    from mosder_short_event_qa_pilot_v1.runtime import load_runtime

    start = time.monotonic()
    completed = 0
    try:
        with load_runtime(
            family="molmo2_o_7b", arm="raw_vlm", device="cuda:0", dtype=None
        ) as loaded:
            backend, identity = runtime_adapter.open_frozen_backend(
                loaded, internal=True
            )
            install_contextual_difference(backend, enabled=args.mechanism == "relative")
            all_method_parameters = collect_method_parameters(backend)
            trainable_parameters = tuple(
                (n, p)
                for n, p in all_method_parameters
                if not n.startswith("decision.")
            )
            assert (
                len(trainable_parameters) == 62
                and sum(p.numel() for _, p in trainable_parameters) == 3014942
            )
            for _, p in trainable_parameters:
                p.requires_grad_(True)
            backend.query_router.requires_grad_(False)
            backend.model.eval()
            trainable_parameter_ids = {id(p) for _, p in trainable_parameters}
            assert {
                id(p) for p in backend.model.parameters() if p.requires_grad
            } == trainable_parameter_ids
            optimizer = torch.optim.AdamW(
                [p for _, p in trainable_parameters],
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
            )
            cursor = 0
            updates = 0
            if args.resume:
                previous = load_candidate(
                    backend,
                    args.resume,
                    for_training_resume=True,
                    method_parameters=all_method_parameters,
                )
                assert previous["mechanism"] == args.mechanism
                assert previous["plan_sha256"] == runtime_adapter.sha(
                    output_dir / "PLAN.json"
                ), "resume checkpoint belongs to another plan"
                optimizer.load_state_dict(previous["optimizer"])
                cursor = previous["next_cursor"]
                updates = previous["updates"]
                for state in optimizer.state.values():
                    for key, value in tuple(state.items()):
                        if isinstance(value, torch.Tensor):
                            state[key] = value.to(trainable_parameters[0][1].device)
                assert cursor % config["accumulation"] == 0 or cursor == len(order)
                assert updates == math.ceil(cursor / config["accumulation"])
                random.setstate(previous["rng_python"])
                np.random.set_state(previous["rng_numpy"])
                torch.set_rng_state(previous["rng_torch"])
                torch.cuda.set_rng_state_all(previous["rng_cuda"])
            frozen_parameter_stamp = tuple(
                (n, id(p), p._version)
                for n, p in backend.model.named_parameters()
                if id(p) not in trainable_parameter_ids
            )
            optimizer.zero_grad(set_to_none=True)
            total_updates = math.ceil(len(order) / config["accumulation"])

            def checkpoint(label, next_cursor):
                payload = {
                    "method_state": {
                        n: p.detach().cpu().clone() for n, p in all_method_parameters
                    },
                    "mechanism": args.mechanism,
                    "parent_sha256": runtime_adapter.PARENT_SHA,
                    "router_sha256": runtime_adapter.GROUND_SHA,
                    "smoke_only": args.smoke,
                    "training_complete": next_cursor == len(order),
                    "optimizer": optimizer.state_dict(),
                    "next_cursor": next_cursor,
                    "updates": updates,
                    "config": config,
                    "plan_sha256": runtime_adapter.sha(output_dir / "PLAN.json"),
                    "rng_python": random.getstate(),
                    "rng_numpy": np.random.get_state(),
                    "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all(),
                }
                path = output_dir / (label + ".pt")
                temp = path.with_suffix(".tmp")
                torch.save(payload, temp)
                temp.replace(path)
                save(
                    output_dir / "LATEST.json",
                    {
                        "checkpoint": str(path),
                        "sha256": runtime_adapter.sha(path),
                        "next_cursor": next_cursor,
                        "updates": updates,
                    },
                )
                return path

            for position in range(cursor, len(order)):
                row = rows[order[position]]
                target = gold_by_id[row["qa_id"]]
                answer = (
                    ("是" if target["yn_answer"] == "yes" else "否")
                    if row["question_type"] == "ynqa"
                    else target["mc_answer"]
                )
                item = runtime_adapter.OriginalInput.from_row(row)
                rgb = runtime_adapter.load_pixels(item)
                request = runtime_adapter.native_request(item, rgb, item.prompt)
                group_start = (position // config["accumulation"]) * config[
                    "accumulation"
                ]
                divisor = min(config["accumulation"], len(order) - group_start)
                group_end = group_start + divisor
                source_divisor = sum(
                    i % config["source_every_qa"] == 0
                    for i in range(group_start, group_end)
                )
                # Check that actual QA loss reaches both source branches through
                # the native forward before the first optimizer update.
                if position == 0 and not args.resume:
                    loss = native_loss(backend, request, answer)
                    assert bool(torch.isfinite(loss))
                    loss.backward()
                    gradients = {
                        n: float(p.grad.norm()) if p.grad is not None else None
                        for n, p in trainable_parameters
                    }
                    assert all(
                        v is None or math.isfinite(v) for v in gradients.values()
                    )
                    assert any(
                        v and v > 0
                        for n, v in gradients.items()
                        if n.startswith("source_core.camera")
                    )
                    assert any(
                        v and v > 0
                        for n, v in gradients.items()
                        if n.startswith("source_core.object")
                    )
                    save(
                        output_dir / "NATIVE_GRADIENT_CHECK.json",
                        {
                            "qa_id": row["qa_id"],
                            "loss": float(loss.detach()),
                            "gradient_norms": gradients,
                            "before_any_optimizer_update": True,
                            "vision_execution": "original native forward; no feature cache",
                        },
                    )
                    del loss
                    optimizer.zero_grad(set_to_none=True)
                loss = native_loss(backend, request, answer)
                assert bool(torch.isfinite(loss))
                value = float(loss.detach())
                (loss / divisor).backward()
                del loss
                log = {"position": position, "qa_id": row["qa_id"], "qa_nll": value}
                if position % config["source_every_qa"] == 0:
                    source_index = source_order[
                        (position // config["source_every_qa"]) % len(sources)
                    ]
                    source_row = sources[source_index]
                    with source_request(backend, source_row) as source_example_request:
                        loss = native_loss(
                            backend,
                            source_example_request,
                            source_row["canonical_answer"],
                        )
                    assert bool(torch.isfinite(loss))
                    log.update(
                        source_case=source_row["case_id"],
                        source_nll=float(loss.detach()),
                    )
                    assert source_divisor > 0
                    (config["source_weight"] * loss / source_divisor).backward()
                    del loss
                if (position + 1) % config["accumulation"] == 0 or position + 1 == len(
                    order
                ):
                    fraction = updates / max(1, total_updates - 1)
                    warmup = config["warmup_fraction"]
                    scale = (
                        (updates + 1) / max(1, math.ceil(total_updates * warmup))
                        if fraction < warmup
                        else 0.1
                        + 0.9
                        * 0.5
                        * (1 + math.cos(math.pi * (fraction - warmup) / (1 - warmup)))
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = config["learning_rate"] * scale
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        [p for _, p in trainable_parameters], config["clip_grad_norm"]
                    )
                    assert bool(torch.isfinite(gradient_norm))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    log.update(
                        update=updates,
                        grad_norm=float(gradient_norm),
                        lr=optimizer.param_groups[0]["lr"],
                    )
                    assert (
                        tuple(
                            (n, id(p), p._version)
                            for n, p in backend.model.named_parameters()
                            if id(p) not in trainable_parameter_ids
                        )
                        == frozen_parameter_stamp
                    )
                    if updates % config["checkpoint_every_updates"] == 0:
                        checkpoint(f"step_{updates:06d}", position + 1)
                completed = position + 1
                log["seconds"] = time.monotonic() - start
                with (output_dir / "TRAIN_LOG.jsonl").open("a") as f:
                    f.write(json.dumps(log, ensure_ascii=False) + "\n")
                if completed % 25 == 0:
                    progress = {
                        "completed": completed,
                        "total": len(order),
                        "updates": updates,
                        "seconds": time.monotonic() - start,
                        "last_qa_nll": value,
                    }
                    save(output_dir / "PROGRESS.json", progress)
                    print(json.dumps(progress), flush=True)
            final = checkpoint("FINAL", len(order))
            save(
                output_dir / "COMPLETE.json",
                {
                    "presentations": len(order),
                    "updates": updates,
                    "seconds": time.monotonic() - start,
                    "checkpoint": str(final),
                    "checkpoint_sha256": runtime_adapter.sha(final),
                    "smoke_only": args.smoke,
                    "frozen_base_identity_version_unchanged": True,
                    "no_val_test_labels_used": True,
                },
            )
    except BaseException:
        save(
            output_dir / ("FAILURE_" + str(time.time_ns()) + ".json"),
            {
                "completed": completed,
                "traceback": traceback.format_exc(),
                "seconds": time.monotonic() - start,
            },
        )
        raise


a = runtime_adapter
HERE = RESEARCH_DIR
method = collect_method_parameters


if __name__ == "__main__":
    main()
