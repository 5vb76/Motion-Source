"""Validation metrics and resumable prediction collection for all data sources."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from common import RUN_DIR, append_jsonl, read_jsonl, write_json
from mosder_final_v1.contract import STATE_FACTORS
from mosder_final_v1.language import (
    canonical_answer,
    parse_four_line_language,
    plan_canonical_language,
    validate_score_against_plan,
)
from mosder_final_v1.objectives import stage_r_language_loss
from mosder_final_v1.routing import FactorRoute
from mosder_fgr_runner_candidate_v3.checkpoint import (
    capture_rng_state,
    restore_rng_state,
    method_parameter_state,
    tensor_state_sha256,
)

STATES = ["neither", "camera_only", "object_only", "both"]


def summarize(records):
    """Aggregate state predictions by source and by ground-truth motion state."""

    def mean_field(rows, field):
        return sum(row[field] for row in rows) / len(rows)

    per_source = {}
    for source in sorted({row["source"] for row in records}):
        source_rows = [row for row in records if row["source"] == source]
        per_source[source] = {
            "n": len(source_rows),
            "joint_state_accuracy": mean_field(source_rows, "correct"),
            "strict_four_line_exact": mean_field(source_rows, "exact"),
        }

    recall_by_state = {}
    for state in STATES:
        state_rows = [row for row in records if row["state"] == state]
        if state_rows:
            recall_by_state[state] = mean_field(state_rows, "correct")

    return {
        "n": len(records),
        "joint_state_accuracy": mean_field(records, "correct"),
        "source_macro_accuracy": sum(
            metrics["joint_state_accuracy"] for metrics in per_source.values()
        )
        / len(per_source),
        "state_macro_recall": sum(recall_by_state.values()) / len(recall_by_state),
        "minimum_state_recall": min(recall_by_state.values()),
        "strict_four_line_exact": mean_field(records, "exact"),
        "camera_accuracy": mean_field(records, "camera_correct"),
        "object_accuracy": mean_field(records, "object_correct"),
        "span_nll": mean_field(records, "span_nll"),
        "per_source": per_source,
        "recall_by_state": recall_by_state,
        "parse_failures": sum(
            row["predicted_state"] == "__invalid__" for row in records
        ),
    }


def checkpoint_rank(metrics):
    """Order validation endpoints; ties are resolved by the caller."""
    return (
        metrics["joint_state_accuracy"],
        metrics["source_macro_accuracy"],
        metrics["minimum_state_recall"],
        metrics["strict_four_line_exact"],
        -metrics["span_nll"],
    )


def rank(m):
    return checkpoint_rank(m)


def evaluate(adapter, dataset, dest, name):
    """Resume validation records and check that evaluation leaves weights unchanged.

    The saved identity binds partial predictions to both the model parameters and
    dataset order. RNG state is restored before returning to training.
    """
    output_dir = Path(dest)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    rng_state = capture_rng_state()
    parameter_digest = tensor_state_sha256(
        method_parameter_state(adapter.method_parameters)
    )
    records = []
    started_at = time.monotonic()
    assert all(p.grad is None for _, p in adapter.method_parameters)
    identity = {
        "method_sha256": parameter_digest,
        "case_ids": [r["case_id"] for r in dataset.rows],
        "name": name,
    }
    identity_path = output_dir / "identity.json"
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity
    else:
        write_json(identity_path, identity)
    if predictions_path.exists():
        records = read_jsonl(predictions_path)
        assert [r["case_id"] for r in records] == identity["case_ids"][: len(records)]
    try:
        adapter.backend.configure_mosder_stage("R")
        for i in range(len(records), len(dataset)):
            example = dataset[i]
            try:
                with torch.no_grad():
                    text = adapter.backend.generate(
                        example.request, max_new_tokens=96, do_sample=False
                    ).text
                parsed = None
                try:
                    parsed = parse_four_line_language(text)
                except (ValueError, RuntimeError):
                    # Invalid four-line answers count as prediction failures.
                    pass
                expected_factors = STATE_FACTORS[example.state]
                expected_answer = canonical_answer(example.state)
                plan = plan_canonical_language(
                    adapter.backend, example.request, example.state
                )
                with torch.enable_grad():
                    score = adapter.backend.teacher_forced_loss(
                        example.request,
                        expected_answer,
                        route=FactorRoute.FULL_LANGUAGE,
                    )
                    validate_score_against_plan(score, plan)
                    losses = stage_r_language_loss(
                        score.token_log_probabilities, plan.spans
                    )
                    span_nll = float(losses.total.detach().cpu())
                del score, losses
                assert np.isfinite(span_nll)
                record = {
                    "case_id": example.case_id,
                    "source": example.source_dataset,
                    "state": example.state,
                    "predicted_state": parsed.state if parsed else "__invalid__",
                    "text": text,
                    "correct": bool(parsed and parsed.state == example.state),
                    "camera_correct": bool(
                        parsed and parsed.camera_moving == expected_factors[0]
                    ),
                    "object_correct": bool(
                        parsed and parsed.object_moving == expected_factors[1]
                    ),
                    "exact": text == expected_answer,
                    "span_nll": span_nll,
                }
                records.append(record)
                append_jsonl(predictions_path, record)
            finally:
                example.close()
            write_json(
                RUN_DIR / "STATUS.json",
                {
                    "state": "evaluating",
                    "evaluation": name,
                    "rows": i + 1,
                    "total_rows": len(dataset),
                    "seconds": time.monotonic() - started_at,
                    "updated_at_unix": time.time(),
                },
            )
            if (i + 1) % 20 == 0:
                print(
                    json.dumps(
                        {
                            "event": "evaluation_progress",
                            "evaluation": name,
                            "row": i + 1,
                            "total": len(dataset),
                            "seconds": time.monotonic() - started_at,
                        }
                    ),
                    flush=True,
                )
        metrics = summarize(records)
        metrics.update(evaluation=name, seconds=time.monotonic() - started_at)
        assert parameter_digest == tensor_state_sha256(
            method_parameter_state(adapter.method_parameters)
        )
        assert all(p.grad is None for _, p in adapter.method_parameters)
        write_json(output_dir / "metrics.json", metrics)
        return metrics
    finally:
        adapter.backend.configure_mosder_stage("FROZEN")
        restore_rng_state(rng_state)
