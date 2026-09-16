"""Evaluate JSONL {case_id, state}; strict full manifest coverage is required."""

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


STATES = ["neither", "camera_only", "object_only", "both"]


def evaluate(rows, predictions, bootstrap=1000):
    """Score complete predictions, with source-stratified collection bootstraps."""
    case_ids = [row["case_id"] for row in rows]
    prediction_ids = [row["case_id"] for row in predictions]
    if len(set(case_ids)) != len(case_ids) or len(set(prediction_ids)) != len(
        prediction_ids
    ):
        raise ValueError("Duplicate case ID")
    if set(case_ids) != set(prediction_ids):
        raise ValueError("Predictions must cover exactly the requested manifest")
    predictions_by_id = {row["case_id"]: row["state"] for row in predictions}
    if any(state not in STATES for state in predictions_by_id.values()):
        raise ValueError("Unknown predicted state")
    true_states = np.array([STATES.index(row["state"]) for row in rows])
    predicted_states = np.array(
        [STATES.index(predictions_by_id[row["case_id"]]) for row in rows]
    )
    correct = true_states == predicted_states

    def group_accuracy(key):
        groups = collections.defaultdict(list)
        for i, row in enumerate(rows):
            groups[row[key]].append(i)
        return {
            k: {"n": len(indices), "accuracy": float(correct[indices].mean())}
            for k, indices in groups.items()
        }

    sources = group_accuracy("source")
    domains = group_accuracy("domain")
    states = group_accuracy("state")
    windows = group_accuracy("physical_window_id")
    confusion = np.zeros((4, 4), dtype=int)
    for true_state, predicted_state in zip(true_states, predicted_states):
        confusion[true_state, predicted_state] += 1
    result = {
        "n": len(rows),
        "joint_state_accuracy": float(correct.mean()),
        "camera_accuracy": float(((true_states % 2) == (predicted_states % 2)).mean()),
        "object_accuracy": float(
            ((true_states // 2) == (predicted_states // 2)).mean()
        ),
        "source_macro_accuracy": float(
            np.mean([v["accuracy"] for v in sources.values()])
        ),
        "domain_macro_accuracy": float(
            np.mean([v["accuracy"] for v in domains.values()])
        ),
        "state_macro_recall": float(np.mean([v["accuracy"] for v in states.values()])),
        "physical_window_macro_accuracy": float(
            np.mean([v["accuracy"] for v in windows.values()])
        ),
        "per_source": sources,
        "per_domain": domains,
        "per_state": states,
        "confusion_matrix_true_rows_pred_columns": confusion.tolist(),
        "state_order": STATES,
        "evaluation_role": sorted({row["role"] for row in rows}),
        "exposure_note": "Requires new-split retraining; historical exposure is recorded in manifest.",
    }
    if bootstrap:
        groups = collections.defaultdict(list)
        for i, row in enumerate(rows):
            groups[(row["source"], row["split_group"])].append(i)
        strata = collections.defaultdict(list)
        for (source, split_group), indices in groups.items():
            strata[source].append((int(correct[indices].sum()), len(indices)))
        rng = np.random.default_rng(20260911)
        scores = []
        for _ in range(bootstrap):
            correct_count = sample_count = 0
            for values in strata.values():
                group_counts = np.array(values)
                sample = group_counts[
                    rng.integers(0, len(group_counts), len(group_counts))
                ]
                total = sample.sum(axis=0)
                correct_count += total[0]
                sample_count += total[1]
            scores.append(correct_count / sample_count)
        result["joint_accuracy_group_bootstrap_95pct"] = np.quantile(
            scores, [0.025, 0.975]
        ).tolist()
        result["bootstrap"] = {
            "replicates": bootstrap,
            "seed": 20260911,
            "unit": "source-specific split_group, stratified by source",
            "limitation": "Few groups and repeated target identities limit generalization; interval is conditional on covered sources.",
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # LOCAL_PATH: The default resolves to code/release/test.jsonl here; pass --manifest
    # with the actual test manifest, such as data/benchmark/splits/test.jsonl.
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "release/test.jsonl",
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    result = evaluate(
        read_jsonl(args.manifest), read_jsonl(args.predictions), args.bootstrap
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
