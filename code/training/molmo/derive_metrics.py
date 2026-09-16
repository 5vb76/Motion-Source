"""Refresh the training CSV from the authoritative, flushed JSONL step records."""

import csv
import json
import math
import os
import time
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent


def learning_rate_multiplier(step):
    """Compute the 625-update warmup and cosine schedule."""
    if step < 32:
        return (step + 1) / 32
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, (step - 32) / 592)))


def collect_training_records():
    """Keep the latest record for each update across resumed training attempts."""
    records = []
    for metrics_path in RUN_DIR.glob("*_metrics_*.jsonl"):
        for line in metrics_path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A writer may still be appending the last line.
                continue
    records.sort(key=lambda record: record["updated_at_unix"])
    return {record["global_step"]: record for record in records}


def build_curve_rows(latest_records):
    """Recover the learning rates used for each update and flatten its metrics."""
    rows = []
    for global_step, record in sorted(latest_records.items()):
        loss = record["loss"]
        learning_rates_used = [
            rate
            * learning_rate_multiplier(record["stage_step"] - 1)
            / learning_rate_multiplier(record["stage_step"])
            for rate in record["learning_rates_next"]
        ]
        rows.append(
            {
                "global_step": global_step,
                "stage": record["stage"],
                "stage_step": record["stage_step"],
                "loss_total": loss["total"]
                if "total" in loss
                else sum(
                    loss[key] for key in ["camera", "object", "state", "description"]
                )
                / 4,
                "loss_camera": loss.get("camera"),
                "loss_object": loss.get("object"),
                "loss_state": loss.get("state"),
                "loss_description": loss.get("description"),
                "gradient_pre_clip": record["gradient_pre_clip"],
                "gradient_post_clip": record["gradient_post_clip"],
                "learning_rates_used": json.dumps(learning_rates_used),
                "seconds_per_update": record["seconds_per_update"],
                "peak_gpu_GiB": record["peak_cuda_memory_bytes"] / 2**30,
            }
        )
    return rows


def refresh_training_curve():
    """Replace the derived CSV when at least one optimizer update is available."""
    rows = build_curve_rows(collect_training_records())
    if rows:
        temporary_path = RUN_DIR / "training_curve.tmp"
        with temporary_path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, RUN_DIR / "training_curve.csv")


def main():
    """Refresh every 20 seconds until the experiment launcher writes EXIT.json."""
    while True:
        refresh_training_curve()
        if (RUN_DIR / "EXIT.json").exists():
            break
        time.sleep(20)


R = RUN_DIR
mult = learning_rate_multiplier


if __name__ == "__main__":
    main()
