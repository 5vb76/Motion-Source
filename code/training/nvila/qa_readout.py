"""Map saved four-field predictions to the fixed benchmark QA questions."""

import json
import sys

from common import RUN_DIR, read_jsonl, write_json

# LOCAL_PATH: External QA helpers take precedence over repository modules.
sys.path.insert(0, "/root/autodl-tmp/molmo_raw_motion_qa_v3")
import qa_protocol as qa


def write_qa_readout(output_dir):
    """Write per-question answers and accuracy without additional model calls."""
    # LOCAL_PATH: Supply QA.json beside this script; it is not included in this directory.
    specs = {
        r["case_id"]: r["spec"] for r in json.loads((RUN_DIR / "QA.json").read_text())
    }
    rows = read_jsonl(output_dir / "predictions.jsonl")
    records = []
    for row in rows:
        spec = specs[row["case_id"]]
        predicted_answer = (
            qa.answer(spec, row["predicted_state"])
            if row["predicted_state"] in qa.STATES
            else None
        )
        gold_answer = qa.answer(spec, row["state"])
        records.append(
            dict(
                case_id=row["case_id"],
                source=row["source"],
                state=row["state"],
                predicted_state=row["predicted_state"],
                kind=spec["kind"],
                question=spec["question"],
                options=spec["options"],
                predicted_answer=predicted_answer,
                gold=gold_answer,
                correct=predicted_answer == gold_answer,
            )
        )
    write_json(output_dir / "qa_predictions.json", records)

    def answer_metrics(question_records):
        return dict(
            n=len(question_records),
            correct=sum(x["correct"] for x in question_records),
            accuracy=sum(x["correct"] for x in question_records)
            / len(question_records),
        )

    result = answer_metrics(records)
    result["per_kind"] = {
        k: answer_metrics([r for r in records if r["kind"] == k])
        for k in ["camera", "object", "joint"]
    }
    result["readout"] = (
        "original four-field predictions mapped by fixed code; no further VLM inference"
    )
    write_json(output_dir / "qa_metrics.json", result)


def convert(dest):
    return write_qa_readout(dest)
