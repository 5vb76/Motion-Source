"""Fit the NVILA query router on sparse box anchors.

Native features are cached once. The 37 fitting videos train the router; the
eight held-out videos are reported only after the fixed 200-step endpoint.
"""

from pathlib import Path
import json
import time
import hashlib
import gc

RUN_DIR = Path(__file__).resolve().parent
from nvila_soft_backend import (
    assemble_soft_query_backend,
    VideoQuery20Request,
    SoftQueryRouter,
)
from soft_query_router import box_grid_occupancy, sparse_box_anchor_loss
from family_backends_v1 import load_local_backend
import torch
import numpy as np


def write_json(path, value):
    """Replace a grounding status or result file atomically."""
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(value, indent=2))
    temporary_path.replace(path)


def main():
    """Cache frozen NVILA features, train the router, and report anchor losses."""
    torch.set_num_threads(4)
    torch.manual_seed(20260907)
    # LOCAL_PATH: Supply sibling GROUNDING_ROWS.json and the external grounding protocol below.
    rows = json.loads((RUN_DIR / "GROUNDING_ROWS.json").read_text())
    grounder_config = json.loads(
        Path(
            "/root/autodl-tmp/mosder_reference_research_20260906/internal_grounding_physics_qa_v1/RUN_PROTOCOL_BEFORE_TRAINING.json"
        ).read_text()
    )["grounder"]
    grounding_dir = RUN_DIR / "grounder"
    grounding_dir.mkdir(exist_ok=True)
    cache = grounding_dir / "cache"
    cache.mkdir(exist_ok=True)
    if (grounding_dir / "COMPLETE.json").exists():
        return
    base = load_local_backend("nvila_lite_8b", device="cuda:0", dtype="float16")
    backend = assemble_soft_query_backend(
        base, SoftQueryRouter(3584, rank=64), mode="uniform", raw=True
    )
    backend.model.eval()
    backend.freeze_base()
    try:
        for i, row in enumerate(rows):
            cache_path = cache / f"{row['ordinal']:03d}.pt"
            if cache_path.exists():
                continue
            # LOCAL_PATH: Each restored_rgb20 field points to a required local NPZ file.
            assert (
                hashlib.sha256(Path(row["restored_rgb20"]).read_bytes()).hexdigest()
                == row["restored_sha256"]
            )
            with np.load(row["restored_rgb20"]) as archive:
                rgb = archive["rgb"]
                timestamps = archive["timestamps_ns"].tolist()
            request = VideoQuery20Request(
                tuple(rgb), timestamps, row["query"], str(row["ordinal"]), row["query"]
            )
            with torch.inference_mode():
                output = backend.native_forward(request)
                tokens = output.visual.raw_native_tokens
                assert torch.equal(tokens, output.visual.modified_native_tokens)
                query = backend._grounding_embedding(request.validated())
            assert tuple(tokens.shape) == (20, 11, 11, 3584)
            torch.save(
                dict(native_tokens=tokens.cpu(), query_embedding=query.cpu()),
                cache_path,
            )
            write_json(
                RUN_DIR / "STATUS.json",
                dict(state="grounder_features", done=i + 1, total=45, time=time.time()),
            )
            del output, tokens, query, rgb
    finally:
        backend.close()
        del backend, base
        gc.collect()
        torch.cuda.empty_cache()
    data = {}
    for row in rows:
        features = torch.load(
            cache / f"{row['ordinal']:03d}.pt", map_location="cuda", weights_only=True
        )
        truth = torch.zeros(20, 11, 11, device="cuda")
        for t, box in enumerate(row["boxes_xyxy_normalized"]):
            if box is not None:
                truth[t] = box_grid_occupancy(box, 11, 11, device="cuda")
        data[row["ordinal"]] = dict(
            x=features,
            truth=truth,
            fit=torch.tensor(
                row["grounder_fit_loss_mask"], device="cuda", dtype=torch.bool
            ),
            hold=torch.tensor(
                row["grounder_holdout_eval_mask"], device="cuda", dtype=torch.bool
            ),
        )
    fit_ids = [i for i, example in data.items() if bool(example["fit"].any())]
    holdout_ids = [i for i, example in data.items() if bool(example["hold"].any())]
    assert len(fit_ids) == 37 and len(holdout_ids) == 8
    torch.manual_seed(20260907)
    router = SoftQueryRouter(3584, rank=64).cuda()
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=grounder_config["lr"],
        weight_decay=grounder_config["weight_decay"],
    )
    order_generator = torch.Generator().manual_seed(grounder_config["seed"])
    order = []
    while len(order) < 1600:
        order += [
            fit_ids[k]
            for k in torch.randperm(len(fit_ids), generator=order_generator).tolist()
        ]
    logs = []
    for step in range(200):
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0
        for i in order[step * 8 : (step + 1) * 8]:
            example = data[i]
            routing = router(
                example["x"]["native_tokens"], example["x"]["query_embedding"]
            )
            loss = (
                sparse_box_anchor_loss(
                    routing.membership_logits, example["truth"], example["fit"]
                )
                / 8
            )
            assert torch.isfinite(loss)
            loss.backward()
            batch_loss += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        assert torch.isfinite(norm)
        optimizer.step()
        record = dict(
            step=step + 1, loss=batch_loss, grad_norm=float(norm), time=time.time()
        )
        logs.append(record)
        with (grounding_dir / "TRAIN.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        write_json(
            RUN_DIR / "STATUS.json",
            dict(state="grounder_training", done=step + 1, total=200, time=time.time()),
        )
    torch.save(router.state_dict(), grounding_dir / "FINAL.pt")
    metrics = []
    router.eval()
    with torch.no_grad():
        for i, example in data.items():
            routing = router(
                example["x"]["native_tokens"], example["x"]["query_embedding"]
            )
            for part in ["fit", "hold"]:
                if example[part].any():
                    metrics.append(
                        dict(
                            ordinal=i,
                            partition=part,
                            loss=float(
                                sparse_box_anchor_loss(
                                    routing.membership_logits,
                                    example["truth"],
                                    example[part],
                                )
                            ),
                        )
                    )
    write_json(grounding_dir / "METRICS.json", dict(items=metrics))
    write_json(
        grounding_dir / "COMPLETE.json",
        dict(
            steps=200,
            fit_videos=37,
            holdout_videos=8,
            fit_anchors=sum(int(data[i]["fit"].sum()) for i in fit_ids),
            holdout_anchors=sum(int(data[i]["hold"].sum()) for i in holdout_ids),
            rank=64,
            hidden_size=3584,
            grid=[20, 11, 11],
            checkpoint_sha256=hashlib.sha256(
                (grounding_dir / "FINAL.pt").read_bytes()
            ).hexdigest(),
            holdout_selection=False,
            qa_answers_used=False,
        ),
    )


if __name__ == "__main__":
    main()
