"""Validation and metric aggregation for a frozen intervention arm."""

import time

import torch
from common import MotionDataset, append_jsonl, read_jsonl, write_json
from mosder_fgr_runner_candidate_v3.checkpoint import (
    method_parameter_state,
    tensor_state_sha256,
)
from mosder_final_v1.contract import STATE_FACTORS
from mosder_final_v1.language import canonical_answer, parse_four_line_language


def validate(adapter, worker, rows, output_dir, write_status):
    """Resume an arm in manifest order and score strict four-line responses."""
    write_status("evaluating", done=0, total=len(rows))
    evaluation_dir = output_dir / "val1200"
    evaluation_dir.mkdir(exist_ok=True)
    predictions_path = evaluation_dir / "predictions.jsonl"
    records = read_jsonl(predictions_path) if predictions_path.exists() else []
    assert [x["case_id"] for x in records] == [
        r["case_id"] for r in rows[: len(records)]
    ]
    adapter.backend.configure_mosder_stage("R")
    before = tensor_state_sha256(method_parameter_state(adapter.method_parameters))
    dataset = MotionDataset(rows, worker)
    started = time.monotonic()
    initial = len(records)
    for i in range(initial, len(rows)):
        example = dataset[i]
        try:
            with torch.no_grad():
                text = adapter.backend.generate(
                    example.request, max_new_tokens=96, do_sample=False
                ).text
            try:
                parsed = parse_four_line_language(text)
            except (ValueError, RuntimeError):
                parsed = None
            gold = STATE_FACTORS[example.state]
            record = dict(
                case_id=example.case_id,
                source=example.source_dataset,
                state=example.state,
                predicted_state=parsed.state if parsed else "__invalid__",
                text=text,
                correct=bool(parsed and parsed.state == example.state),
                camera_correct=bool(parsed and parsed.camera_moving == gold[0]),
                object_correct=bool(parsed and parsed.object_moving == gold[1]),
                exact=text == canonical_answer(example.state),
                split_group=rows[i]["split_group"],
                physical_window_id=rows[i]["physical_window_id"],
            )
            append_jsonl(predictions_path, record)
            records.append(record)
        finally:
            example.close()
        elapsed_seconds = time.monotonic() - started
        write_status(
            "evaluating",
            done=i + 1,
            total=len(rows),
            seconds=elapsed_seconds,
            seconds_per_example=elapsed_seconds / (i + 1 - initial),
            remaining_seconds=elapsed_seconds / (i + 1 - initial) * (len(rows) - i - 1),
        )
    states = list(STATE_FACTORS)
    confusion = {s: {q: 0 for q in states + ["__invalid__"]} for s in states}
    for x in records:
        confusion[x["state"]][x["predicted_state"]] += 1
    per_state = {}
    for s in states:
        true_positive = confusion[s][s]
        n = sum(confusion[s].values())
        predicted_count = sum(confusion[t][s] for t in states)
        per_state[s] = dict(
            support=n,
            recall=true_positive / n,
            precision=true_positive / predicted_count if predicted_count else 0,
            f1=2 * true_positive / (n + predicted_count) if n + predicted_count else 0,
        )
    per_source = {
        s: dict(
            n=sum(x["source"] == s for x in records),
            accuracy=sum(x["correct"] for x in records if x["source"] == s)
            / sum(x["source"] == s for x in records),
        )
        for s in sorted({x["source"] for x in records})
    }
    metrics = dict(
        n=len(records),
        correct=sum(x["correct"] for x in records),
        accuracy=sum(x["correct"] for x in records) / len(records),
        camera_accuracy=sum(x["camera_correct"] for x in records) / len(records),
        object_accuracy=sum(x["object_correct"] for x in records) / len(records),
        macro_f1=sum(x["f1"] for x in per_state.values()) / 4,
        balanced_accuracy=sum(x["recall"] for x in per_state.values()) / 4,
        parse_failures=sum(x["predicted_state"] == "__invalid__" for x in records),
        per_source=per_source,
        source_macro_accuracy=sum(x["accuracy"] for x in per_source.values())
        / len(per_source),
        per_state=per_state,
        confusion=confusion,
        weights_unchanged=before
        == tensor_state_sha256(method_parameter_state(adapter.method_parameters)),
        protocol="four-field generate then strict parse; fixed R endpoint",
        seconds_this_attempt=time.monotonic() - started,
    )
    assert metrics["weights_unchanged"]
    write_json(evaluation_dir / "metrics.json", metrics)
    return metrics
