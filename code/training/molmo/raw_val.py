"""Queued raw pretrained Molmo validation; never reads test or task checkpoints."""

import fcntl
import json
import os
import subprocess
import time
import traceback

from common import (
    BENCHMARK_DIR,
    RUN_DIR,
    BenchmarkWorker,
    MotionDataset,
    append_jsonl,
    file_sha256,
    read_jsonl,
    write_json,
)

OUTPUT_DIR = RUN_DIR / "validation/raw_molmo"


def write_status(state, **details):
    """Record queue or evaluation progress for the raw-model baseline."""
    write_json(
        OUTPUT_DIR / "status.json",
        dict(state=state, pid=os.getpid(), updated_at_unix=time.time(), **details),
    )


def main():
    """Wait for upstream evaluation and GPU release, then evaluate raw Molmo."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT_DIR / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUTPUT_DIR / "metrics.json").exists():
        return
    write_status(
        "queued", waiting_for="new R validation, old MoSDeR validation, GPU release"
    )
    # LOCAL_PATH: Requires the upstream run records, including EXIT.json, beside train.py.
    while True:
        if (RUN_DIR / "FAILURE.json").exists():
            write_status(
                "blocked", reason="Upstream pipeline failed; waiting for recovery"
            )
            time.sleep(20)
            continue
        if (RUN_DIR / "COMPLETION.json").exists() and (RUN_DIR / "EXIT.json").exists():
            if json.loads((RUN_DIR / "EXIT.json").read_text())["exit_code"] != 0:
                raise RuntimeError("Upstream nonzero exit")
            gpu_processes = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                text=True,
            ).strip()
            if not gpu_processes:
                break
        time.sleep(20)
    write_status("loading")
    import torch
    from dataclasses import replace
    from family_backends_v1 import load_local_backend
    from mosder_final_v1.language import parse_four_line_language, canonical_answer
    from mosder_final_v1.contract import STATE_FACTORS
    from evaluation import summarize
    from mosder_fgr_runner_candidate_v3.checkpoint import (
        configure_reproducible_numeric_environment,
        set_global_seed,
    )

    # LOCAL_PATH: Supply validation/raw_molmo/plan.json and every local file in its hashes map.
    plan = json.loads((OUTPUT_DIR / "plan.json").read_text())
    for path, digest in plan["hashes"].items():
        assert file_sha256(path) == digest, path
    configure_reproducible_numeric_environment()
    set_global_seed(20260911)
    torch.set_num_threads(4)
    rows = read_jsonl(BENCHMARK_DIR / "release/val.jsonl")
    assert len(rows) == 1200
    backend = load_local_backend("molmo2_o_7b", device="cuda:0", dtype="bfloat16")
    assert backend.unified_physical_core is None
    assert not backend._source_handles and not backend._source_lora_paths
    backend.model.eval()
    backend.model.requires_grad_(False)
    worker = BenchmarkWorker()
    dataset = MotionDataset(rows, worker)
    predictions_path = OUTPUT_DIR / "predictions.jsonl"
    records = read_jsonl(predictions_path) if predictions_path.exists() else []
    assert [r["case_id"] for r in records] == [
        r["case_id"] for r in rows[: len(records)]
    ]
    started = time.monotonic()
    completed_rows = len(records)
    try:
        for i in range(completed_rows, len(rows)):
            example = dataset[i]
            row_started_at = time.monotonic()
            try:
                # Native raw decoder has no MoSDeR target route: expose only existing oracle boxes.
                original_request = example.request
                validated = original_request.validated()
                frame_width, frame_height = validated.width, validated.height
                boxes = original_request.oracle_boxes_xyxy
                target_prompt = (
                    "\nOracle target box track: coordinates are normalized [x_min, y_min, x_max, y_max], in frame order 1–20:\n"
                    + json.dumps(
                        [
                            [
                                round(float(box[0]) / frame_width, 4),
                                round(float(box[1]) / frame_height, 4),
                                round(float(box[2]) / frame_width, 4),
                                round(float(box[3]) / frame_height, 4),
                            ]
                            for box in boxes
                        ]
                    )
                )
                request = replace(
                    original_request, prompt=original_request.prompt + target_prompt
                )
                with torch.inference_mode():
                    generated_text = backend.generate(
                        request, max_new_tokens=96, do_sample=False
                    ).text
                try:
                    parsed = parse_four_line_language(generated_text)
                except (ValueError, RuntimeError):
                    parsed = None
                expected_factors = STATE_FACTORS[example.state]
                record = dict(
                    case_id=example.case_id,
                    source=example.source_dataset,
                    state=example.state,
                    predicted_state=parsed.state if parsed else "__invalid__",
                    text=generated_text,
                    correct=bool(parsed and parsed.state == example.state),
                    camera_correct=bool(
                        parsed and parsed.camera_moving == expected_factors[0]
                    ),
                    object_correct=bool(
                        parsed and parsed.object_moving == expected_factors[1]
                    ),
                    exact=generated_text == canonical_answer(example.state),
                    seconds=time.monotonic() - row_started_at,
                )
                append_jsonl(predictions_path, record)
                records.append(record)
            finally:
                example.close()
            elapsed = time.monotonic() - started
            seconds_per_row = elapsed / (i + 1 - completed_rows)
            write_status(
                "evaluating",
                rows=i + 1,
                total_rows=len(rows),
                seconds=elapsed,
                seconds_per_row=seconds_per_row,
                eta_seconds=(len(rows) - i - 1) * seconds_per_row,
            )
            if (i + 1) % 20 == 0:
                print(
                    json.dumps(dict(row=i + 1, seconds_per_row=seconds_per_row)),
                    flush=True,
                )
        # Reuse identical classification metric definitions; no teacher-forced pass in raw inference.
        metrics = summarize([dict(r, span_nll=0.0) for r in records])
        metrics.pop("span_nll")
        metrics.update(
            evaluation="raw_molmo",
            protocol=plan,
            seconds=time.monotonic() - started,
            teacher_forced_nll_computed=False,
        )
        write_json(OUTPUT_DIR / "metrics.json", metrics)
        write_status("complete", rows=len(rows), total_rows=len(rows))
    finally:
        worker.close()
        backend.close()


DEST = OUTPUT_DIR
status = write_status


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write_status("failed", error=str(error))
        write_json(
            OUTPUT_DIR / "failure.json",
            dict(error=str(error), traceback=traceback.format_exc()),
        )
        raise
