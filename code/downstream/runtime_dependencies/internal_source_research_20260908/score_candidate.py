"""Exact frozen-parser scoring and video-cluster uncertainty for candidate runs.

No import-time data reads or writes. Full scoring waits for COMPLETE.json;
--terminal-failure explicitly permits missing/failed rows, counted as incorrect
in all 914 denominators. Partial replay compares generated tokens without gold.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
RESEARCH_DIR = Path(__file__).resolve().parent
# LOCAL_PATH: The sibling validation directory supplies scorer code and data;
# the scorer file must match FROZEN_SCORER_SHA256.
VALIDATION_DIR = RESEARCH_DIR.parent / "omnivchall_val_eval_v1"
SCORER_PATH = VALIDATION_DIR / "score_val.py"
FROZEN_SCORER_SHA256 = (
    "458741466083055fae74408ee6f7da5956b6ec1d6ad1632fc24f87736a14f1a3"
)
EXPECTED_QA_COUNT = 914
SEED = 20260908
BOOTSTRAP_SAMPLES = 10000
LEGACY_MINIMAL_BINDING_SHA = (
    "7e3899cbb5339469649410e987213b17e7c75018341d42648a428254bcdef300"
)
SPLITS = ("s_ynqa", "s_mcqa", "m_ynqa", "m_mcqa")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def load_frozen_scoring_functions():
    """Load scoring functions only if their source hash matches."""
    if sha(SCORER_PATH) != FROZEN_SCORER_SHA256:
        raise ValueError(
            "Frozen scorer SHA-256 mismatch. This entry point requires the original "
            "score_val.py bytes; see code/downstream/README.md before using a "
            "formatted or edited scorer."
        )
    spec = importlib.util.spec_from_file_location(
        "_frozen_candidate_evalscore", SCORER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_yn, module.parse_mc, module.counts_for, module.by_type


def unique(rows, name):
    ids = [r["qa_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate {name} QA IDs")
    return dict(zip(ids, rows))


def expected_binding(row):
    text_sha = lambda x: hashlib.sha256(x.encode()).hexdigest()
    return {
        key: row[key]
        for key in ("ordinal", "qa_id", "video_id", "rgb20_npz_sha256", "rgb_sha256")
    } | {
        "question_stem_sha256": text_sha(row["question_stem"]),
        "original_prompt_sha256": text_sha(row["prompt"]),
    }


def validate_predictions(
    rows, inputs, *, permit_missing=False, require_full_binding=True
):
    """Check input identity and count permitted missing or failed rows as wrong."""
    observed = unique(rows, "prediction")
    expected = unique(inputs, "input")
    if set(observed) - set(expected):
        raise ValueError("prediction has IDs outside fixed manifest")
    missing = sorted(set(expected) - set(observed))
    if missing and not permit_missing:
        raise ValueError(
            "missing prediction rows; use explicit terminal-failure mode, never reduce denominator"
        )
    normalized, explicit_failed = [], []
    for qid, item in expected.items():
        if qid not in observed:
            normalized.append(
                {
                    "qa_id": qid,
                    "text": None,
                    "success": False,
                    "failure_kind": "missing_prediction",
                    "input_binding": expected_binding(item),
                }
            )
            continue
        row = observed[qid]
        for key in ("ordinal", "qa_id", "video_id"):
            if row.get(key) != item[key]:
                raise ValueError(f"prediction identity drift: {qid}/{key}")
        if "question_type" in row and row["question_type"] != item["question_type"]:
            raise ValueError(f"prediction kind drift: {qid}")
        binding = row.get("input_binding", {})
        full = expected_binding(item)
        mandatory = (
            set(full)
            if require_full_binding
            else {"rgb_sha256", "original_prompt_sha256"}
        )
        if not mandatory <= set(binding) or any(
            binding[k] != full[k] for k in binding if k in full
        ):
            raise ValueError(f"prediction input binding drift: {qid}")
        if type(row.get("success")) is not bool:
            raise ValueError(f"explicit boolean success required: {qid}")
        if row["success"] is False:
            explicit_failed.append(qid)
        # Failed calls cannot score via a stale partial answer in their text field.
        normalized.append(row | {"text": row.get("text") if row["success"] else None})
    if set(unique(normalized, "normalized prediction")) != set(expected):
        raise ValueError("normalization did not preserve exact membership")
    return normalized, {
        "observed_count": len(rows),
        "normalized_count": len(normalized),
        "missing_ids": missing,
        "explicit_failure_ids": explicit_failed,
    }


def answer(row):
    return row["yn_answer"] if row["question_type"] == "ynqa" else row["mc_answer"]


def classification(gold, parsed, labels):
    values = {}
    for label in labels:
        support = sum(answer(r) == label for r in gold)
        tp = sum(answer(r) == label and parsed[r["qa_id"]] == label for r in gold)
        fp = sum(answer(r) != label and parsed[r["qa_id"]] == label for r in gold)
        fn = support - tp  # Includes invalids/failed/missing calls.
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        values[label] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn_including_abstentions": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    supported = [v for v in values.values() if v["support"]]
    return {
        "per_class": values,
        **{
            f"macro_{key}": sum(v[key] for v in supported) / len(supported)
            if supported
            else 0.0
            for key in ("precision", "recall", "f1")
        },
        "recall_denominator": "all gold support, including invalid/missing/failed predictions",
    }


def compact(gold, parsed):
    n = len(gold)
    valid = sum(parsed[r["qa_id"]] is not None for r in gold)
    correct = sum(parsed[r["qa_id"]] == answer(r) for r in gold)
    return {
        "total": n,
        "valid": valid,
        "invalid": n - valid,
        "correct": correct,
        "fixed_accuracy": correct / n if n else 0.0,
        "valid_accuracy": correct / valid if valid else 0.0,
        "parse_coverage": valid / n if n else 0.0,
    }


def condition_metrics(items, gold, video_source, functions):
    parse_yn, parse_mc, counts_for, by_type = functions
    item_by = unique(items, "condition")
    parsed = {
        r["qa_id"]: (parse_yn if r["question_type"] == "ynqa" else parse_mc)(
            item_by[r["qa_id"]].get("text")
        )
        for r in gold
    }
    result = {
        "n_predictions": len(items),
        "success_count": sum(r["success"] for r in items),
        "subsets": {},
        "by_type_id": {},
        "video_splits": {},
    }
    groups = {}

    def metrics(gold_rows, kind):
        prediction_rows = [item_by[r["qa_id"]] for r in gold_rows]
        if kind is None:
            return compact(gold_rows, parsed)
        value, parsed_rows, truth = counts_for(prediction_rows, gold_rows, kind)
        assert set(parsed_rows) == {r["qa_id"] for r in gold_rows}
        value["classification"] = classification(
            gold_rows, parsed_rows, ("yes", "no") if kind == "ynqa" else "ABC"
        )
        return value

    for split in SPLITS:
        kind = "ynqa" if split.endswith("ynqa") else "mcqa"
        gold_rows = [r for r in gold if r["qa_id"].startswith(split + "_id:")]
        result["subsets"][split] = metrics(gold_rows, kind)
        prediction_rows = [item_by[r["qa_id"]] for r in gold_rows]
        _, parsed_rows, truth = counts_for(prediction_rows, gold_rows, kind)
        result["by_type_id"][split] = by_type(prediction_rows, truth, parsed_rows, kind)
        groups["subsets/" + split] = (gold_rows, kind)
        for type_id in range(1, 9):
            types = lambda r: (
                r["type_id"] if isinstance(r["type_id"], list) else [r["type_id"]]
            )
            group = [r for r in gold_rows if type_id in types(r)]
            groups[f"by_type_id/{split}/{type_id}"] = (group, kind)
    for label, kind in (("all_ynqa", "ynqa"), ("all_mcqa", "mcqa"), ("all_qa", None)):
        gold_rows = [r for r in gold if kind is None or r["question_type"] == kind]
        result["subsets"][label] = metrics(gold_rows, kind)
        groups["subsets/" + label] = (gold_rows, kind)
        for source in ("video_real", "video_generated"):
            video_gold = [r for r in gold_rows if video_source[r["video_id"]] == source]
            result["video_splits"].setdefault(source, {})[label] = metrics(
                video_gold, kind
            )
            groups[f"video_splits/{source}/{label}"] = (video_gold, kind)
    return result, parsed, groups


def cluster_intervals(groups, parsed_maps):
    """Compute paired bootstrap intervals by resampling whole video clusters."""
    videos = sorted(
        {r["video_id"] for gold_rows, _ in groups.values() for r in gold_rows}
    )
    video_indices = {v: i for i, v in enumerate(videos)}
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(videos), size=(BOOTSTRAP_SAMPLES, len(videos)))
    weights = np.zeros((BOOTSTRAP_SAMPLES, len(videos)), dtype=np.int32)
    np.add.at(weights, (np.arange(BOOTSTRAP_SAMPLES)[:, None], samples), 1)
    out = {}

    def quantiles(values):
        values = values[np.isfinite(values)]
        return (
            {
                "low": float(np.quantile(values, 0.025)),
                "high": float(np.quantile(values, 0.975)),
                "valid_resamples": int(len(values)),
            }
            if len(values)
            else {"low": None, "high": None, "valid_resamples": 0}
        )

    for name, (gold, kind) in groups.items():
        if not gold:
            out[name] = {"total": 0, "video_clusters": 0}
            continue
        question_counts = np.zeros(len(videos), dtype=np.float64)
        for row in gold:
            question_counts[video_indices[row["video_id"]]] += 1
        denominator = weights @ question_counts
        valid = denominator > 0
        accuracies, condition_cis = {}, {}
        for condition, parsed in parsed_maps.items():
            correct = np.zeros_like(question_counts)
            for row in gold:
                correct[video_indices[row["video_id"]]] += parsed[
                    row["qa_id"]
                ] == answer(row)
            distribution = np.full(BOOTSTRAP_SAMPLES, np.nan)
            distribution[valid] = (weights @ correct)[valid] / denominator[valid]
            accuracies[condition] = distribution
            condition_cis[condition] = {"fixed_accuracy": quantiles(distribution)}
            if kind is not None:
                labels = ("yes", "no") if kind == "ynqa" else "ABC"
                tp = np.zeros((len(videos), len(labels)))
                fp = np.zeros_like(tp)
                support = np.zeros_like(tp)
                for row in gold:
                    i, truth, prediction = (
                        video_indices[row["video_id"]],
                        answer(row),
                        parsed[row["qa_id"]],
                    )
                    support[i, labels.index(truth)] += 1
                    if prediction == truth:
                        tp[i, labels.index(truth)] += 1
                    elif prediction is not None:
                        fp[i, labels.index(prediction)] += 1
                sampled_true_positives, sampled_false_positives, sampled_support = (
                    weights @ tp,
                    weights @ fp,
                    weights @ support,
                )
                precision = np.divide(
                    sampled_true_positives,
                    sampled_true_positives + sampled_false_positives,
                    out=np.zeros_like(sampled_true_positives),
                    where=sampled_true_positives + sampled_false_positives > 0,
                )
                recall = np.divide(
                    sampled_true_positives,
                    sampled_support,
                    out=np.zeros_like(sampled_true_positives),
                    where=sampled_support > 0,
                )
                f1 = np.divide(
                    2 * precision * recall,
                    precision + recall,
                    out=np.zeros_like(sampled_true_positives),
                    where=precision + recall > 0,
                )
                present = sampled_support > 0
                for metric, value in (
                    ("macro_precision", precision),
                    ("macro_recall", recall),
                    ("macro_f1", f1),
                ):
                    distribution = np.full(BOOTSTRAP_SAMPLES, np.nan)
                    distribution[valid] = (value * present).sum(1)[valid] / present.sum(
                        1
                    )[valid]
                    condition_cis[condition][metric] = quantiles(distribution)
        out[name] = {
            "total": len(gold),
            "video_clusters": int((question_counts > 0).sum()),
            "conditions": condition_cis,
            "paired_delta_accuracy": {
                baseline: quantiles(accuracies["candidate"] - accuracies[baseline])
                for baseline in ("raw", "internal_only")
            },
        }
    return {
        "seed": SEED,
        "resamples": BOOTSTRAP_SAMPLES,
        "level": 0.95,
        "method": f"paired video-cluster percentile bootstrap; resample {len(videos)} videos with replacement, retain all their questions; QA-weighted ratio",
        "video_clusters": len(videos),
        "groups": out,
        "claim_limit": "Exposed development evidence; intervals are descriptive, not corrected for model selection or multiple comparisons, and do not establish independent confirmation.",
    }


def paired(gold, candidate, baseline):
    fixes = [
        r["qa_id"]
        for r in gold
        if candidate[r["qa_id"]] == answer(r) and baseline[r["qa_id"]] != answer(r)
    ]
    breaks = [
        r["qa_id"]
        for r in gold
        if baseline[r["qa_id"]] == answer(r) and candidate[r["qa_id"]] != answer(r)
    ]
    return {
        "total": len(gold),
        "fix_count": len(fixes),
        "break_count": len(breaks),
        "net": len(fixes) - len(breaks),
        "delta_accuracy": (len(fixes) - len(breaks)) / len(gold) if gold else 0.0,
        "fix_ids": fixes,
        "break_ids": breaks,
    }


def load_pinned(path, freeze):
    if freeze["artifact_sha256"].get(str(path)) != sha(path):
        raise ValueError(f"baseline artifact is not pinned: {path}")
    return load(path)


def score_run(run_dir, *, terminal_failure=False, replay=False):
    """Score a complete fixed evaluation run, or explicitly report terminal failure."""
    run_dir = Path(run_dir).resolve()
    if run_dir.parent != RESEARCH_DIR / "runs":
        raise ValueError("run directory must be a direct child of new research runs/")
    # LOCAL_PATH: Requires run inputs, BASELINE_FREEZE.json, and sibling validation data/results.
    plan = load(run_dir / "PLAN.json")
    freeze = load(RESEARCH_DIR / "BASELINE_FREEZE.json")
    if plan["baseline_freeze_sha256"] != sha(RESEARCH_DIR / "BASELINE_FREEZE.json"):
        raise ValueError("baseline freeze changed since inference")
    manifest_path = VALIDATION_DIR / "data/PREPARED_VAL_WITHOUT_ANSWERS.json"
    inputs = load_pinned(manifest_path, freeze)["items"]
    if (
        len(inputs) != EXPECTED_QA_COUNT
        or len(unique(inputs, "input")) != EXPECTED_QA_COUNT
        or len({r["video_id"] for r in inputs}) != 82
    ):
        raise ValueError("fixed914/82 input membership changed")
    if (
        plan["input_manifest_sha256"] != sha(manifest_path)
        or plan["generation"] != freeze["generation"]
    ):
        raise ValueError("inference input/generation contract changed")
    if terminal_failure:
        if not (run_dir / "FAILURE.json").is_file():
            raise ValueError("terminal failure mode requires FAILURE.json")
    elif not (run_dir / "COMPLETE.json").is_file():
        raise ValueError("wait until predictions are frozen complete before scoring")
    pred_path = run_dir / "PREDICTIONS.json"
    if pred_path.exists():
        candidate = load(pred_path)["items"]
        if (run_dir / "COMPLETE.json").exists():
            complete = load(run_dir / "COMPLETE.json")
            if complete["predictions_sha256"] != sha(pred_path) or complete[
                "count"
            ] != len(candidate):
                raise ValueError("prediction completion seal mismatch")
    elif terminal_failure:
        pred_path = run_dir / "PREDICTIONS.jsonl"
        candidate = (
            [
                json.loads(line)
                for line in pred_path.read_text().splitlines()
                if line.strip()
            ]
            if pred_path.exists()
            else []
        )
    else:
        raise ValueError("complete prediction bundle missing")
    baseline_path = VALIDATION_DIR / "runs/INTERNAL_HYBRID_PREDICTIONS.json"
    internal = load_pinned(baseline_path, freeze)["items"]
    if replay:
        if plan["mode"] != "replay" or plan["count"] != 8 or plan["full_evaluation"]:
            raise ValueError("this replay contract is exactly the first eight inputs")
        checked, audit = validate_predictions(
            candidate, inputs[:8], require_full_binding=False
        )
        old = unique(internal, "internal")
        mismatches = [
            {
                "qa_id": r["qa_id"],
                "token_ids_match": r.get("generated_token_ids")
                == old[r["qa_id"]]["generated_token_ids"],
                "text_match": r.get("text") == old[r["qa_id"]]["text"],
            }
            for r in checked
        ]
        return "REPLAY_PARITY.json", {
            "schema": "candidate_limit8_native_replay_parity_v1",
            "gold_read": False,
            "rows": mismatches,
            "all_token_ids_match": all(r["token_ids_match"] for r in mismatches),
            "all_texts_match": all(r["text_match"] for r in mismatches),
            "binding_audit": audit,
        }
    if plan["count"] != EXPECTED_QA_COUNT or not plan["full_evaluation"]:
        raise ValueError("no accuracy scoring of partial candidate runs")
    full_binding = plan["code_sha256"] != LEGACY_MINIMAL_BINDING_SHA
    checked, candidate_audit = validate_predictions(
        candidate,
        inputs,
        permit_missing=terminal_failure,
        require_full_binding=full_binding,
    )
    candidate_audit["observed_binding_contract"] = (
        "all seven original binding fields"
        if full_binding
        else "pinned initial runner: RGB byte hash + original prompt hash in each prediction; remaining input identity comes from row IDs and frozen whole-manifest hash"
    )
    if len(checked) != EXPECTED_QA_COUNT:
        raise ValueError("normalized prediction denominator must be 914")
    raw_path = VALIDATION_DIR / "runs/RAW_PREDICTIONS.json"
    raw, raw_audit = validate_predictions(
        load_pinned(raw_path, freeze)["items"], inputs
    )
    internal, internal_audit = validate_predictions(internal, inputs)
    gold_path = VALIDATION_DIR / "data/OFFICIAL_VAL_GOLD.json"
    gold = load_pinned(gold_path, freeze)["items"]
    gold_by_id = unique(gold, "gold")
    if len(gold) != EXPECTED_QA_COUNT or set(gold_by_id) != {
        r["qa_id"] for r in inputs
    }:
        raise ValueError("gold membership differs from fixed914")
    for item in inputs:
        truth = gold_by_id[item["qa_id"]]
        if any(truth[k] != item[k] for k in ("ordinal", "video_id", "question_type")):
            raise ValueError("gold/input identity mismatch")
        if answer(truth) not in (
            ("yes", "no") if truth["question_type"] == "ynqa" else "ABC"
        ):
            raise ValueError("invalid frozen gold label")
    video_source = {}
    # LOCAL_PATH: Set the OmniVCHall media root containing both video subsets.
    for video in sorted({r["video_id"] for r in inputs}):
        # Stat only these 82 known VAL assets; never enumerate/read Omni test.
        sources = [
            source
            for source in ("video_real", "video_generated")
            if (
                Path("/root/autodl-tmp/OmniVCHall")
                / source
                / "all_video"
                / (video + ".mp4")
            ).is_file()
        ]
        if len(sources) != 1:
            raise ValueError("VAL real/generated membership ambiguous or absent")
        video_source[video] = sources[0]
    functions = load_frozen_scoring_functions()
    conditions, parsed = {}, {}
    for name, rows in (
        ("raw", raw),
        ("internal_only", internal),
        ("candidate", checked),
    ):
        conditions[name], parsed[name], groups = condition_metrics(
            rows, gold, video_source, functions
        )
    for name in ("raw", "internal_only"):
        if (
            conditions[name]["subsets"]["all_qa"]["correct"]
            != freeze["metrics"][name]["correct"]
        ):
            raise ValueError("frozen baseline arithmetic failed reproduction")
    comparisons = {
        baseline: {
            name: paired(gold_rows, parsed["candidate"], parsed[baseline])
            for name, (gold_rows, _) in groups.items()
        }
        for baseline in ("raw", "internal_only")
    }
    result = {
        "schema": "candidate_fixed914_metrics_v1",
        "n_gold": EXPECTED_QA_COUNT,
        "conditions": conditions,
        "comparisons": comparisons,
        "bootstrap": cluster_intervals(groups, parsed),
        "input_binding_audit": {
            "candidate": candidate_audit,
            "raw": raw_audit,
            "internal_only": internal_audit,
        },
        "video_source_counts": dict(Counter(video_source.values())),
        "denominator_rule": "all914; invalid, explicit failed and explicitly permitted terminal missing rows are incorrect",
        "claim_limit": "Exposed development result only. No best-run selection, independent confirmation, population gate calibration or deployment claim is made by this scorer.",
        "provenance_sha256": {
            str(p): sha(p)
            for p in (
                Path(__file__),
                SCORER_PATH,
                RESEARCH_DIR / "BASELINE_FREEZE.json",
                manifest_path,
                gold_path,
                raw_path,
                baseline_path,
                run_dir / "PLAN.json",
            )
        },
        "prediction_file": str(pred_path),
        "predictions_sha256": sha(pred_path) if pred_path.exists() else None,
    }
    return "CANDIDATE_METRICS.json", result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        required=True,
        help="name or absolute path of run under this research runs/",
    )
    parser.add_argument("--terminal-failure", action="store_true")
    parser.add_argument(
        "--replay",
        action="store_true",
        help="first8 generated token/text parity; no gold scoring",
    )
    args = parser.parse_args()
    path = Path(args.run)
    run = path if path.is_absolute() else RESEARCH_DIR / "runs" / path
    filename, result = score_run(
        run, terminal_failure=args.terminal_failure, replay=args.replay
    )
    output = run / filename
    with output.open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"output": str(output), "sha256": sha(output)}))


frozen_functions = load_frozen_scoring_functions


if __name__ == "__main__":
    main()
