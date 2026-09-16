"""Parse OmniVCHall answers and report fixed-denominator validation metrics."""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path

# LOCAL_PATH: Official gold and prediction bundles are read from external data/ and runs/ directories.
ROOT = Path(
    "/root/autodl-tmp/mosder_reference_research_20260906/omnivchall_val_eval_v1"
)
RUNS = ROOT / "runs"


def parse_yn(text):
    """Accept only a bare Chinese or English yes/no answer after edge trimming."""
    if not isinstance(text, str):
        return None
    normalized = unicodedata.normalize("NFKC", text).casefold()

    def is_boundary_character(character):
        return character.isspace() or unicodedata.category(character).startswith("P")

    start, end = 0, len(normalized)
    while start < end and is_boundary_character(normalized[start]):
        start += 1
    while end > start and is_boundary_character(normalized[end - 1]):
        end -= 1
    return {"是": "yes", "yes": "yes", "否": "no", "no": "no"}.get(
        normalized[start:end]
    )


def parse_mc(text):
    """Accept one option letter A, B, or C after Unicode and edge normalization."""
    if not isinstance(text, str):
        return None
    normalized = unicodedata.normalize("NFKC", text).strip().upper()
    while normalized and (
        normalized[0].isspace() or unicodedata.category(normalized[0]).startswith("P")
    ):
        normalized = normalized[1:]
    while normalized and (
        normalized[-1].isspace() or unicodedata.category(normalized[-1]).startswith("P")
    ):
        normalized = normalized[:-1]
    return normalized if normalized in {"A", "B", "C"} else None


def load(path):
    return json.loads(path.read_text())


def counts_for(items, answers, qa_kind):
    """Count every selected question, including unparseable answers."""
    truth = {x["qa_id"]: x for x in answers if x["question_type"] == qa_kind}
    selected = [x for x in items if x["qa_id"] in truth]
    parsed = {}
    for x in selected:
        parsed[x["qa_id"]] = (
            parse_yn(x.get("text")) if qa_kind == "ynqa" else parse_mc(x.get("text"))
        )
    total = len(selected)
    valid = sum(v is not None for v in parsed.values())
    correct = sum(
        parsed[q]
        == (truth[q]["yn_answer"] if qa_kind == "ynqa" else truth[q]["mc_answer"])
        for q in parsed
    )
    result = {
        "total": total,
        "valid": valid,
        "parse_coverage": valid / total if total else 0.0,
        "correct": correct,
        "fixed_accuracy": correct / total if total else 0.0,
        "valid_accuracy": correct / valid if valid else 0.0,
        "invalid": total - valid,
    }
    if qa_kind == "ynqa":
        positive_ids = [q for q in parsed if truth[q]["yn_answer"] == "yes"]
        negative_ids = [q for q in parsed if truth[q]["yn_answer"] == "no"]
        tp = sum(parsed[q] == "yes" for q in positive_ids)
        tn = sum(parsed[q] == "no" for q in negative_ids)
        fp = sum(parsed[q] == "yes" for q in negative_ids)
        fn = sum(parsed[q] == "no" for q in positive_ids)
        result.update(
            {
                "yes_total": len(positive_ids),
                "no_total": len(negative_ids),
                "yes_correct": tp,
                "no_correct": tn,
                "yes_accuracy": tp / len(positive_ids) if positive_ids else 0.0,
                "no_accuracy": tn / len(negative_ids) if negative_ids else 0.0,
                "false_positive_rate": fp / len(negative_ids) if negative_ids else 0.0,
                "false_negative_rate": fn / len(positive_ids) if positive_ids else 0.0,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "answer_distribution": dict(
                    Counter(v for v in parsed.values() if v is not None)
                ),
            }
        )
    else:
        support = Counter(truth[q]["mc_answer"] for q in parsed)
        tp = Counter()
        fp = Counter()
        fn = Counter()
        for q, pred in parsed.items():
            gold = truth[q]["mc_answer"]
            if pred == gold:
                tp[gold] += 1
            elif pred is not None:
                fp[pred] += 1
                fn[gold] += 1
            else:
                fn[gold] += 1
        metrics = {}
        precisions, recalls, f1_scores = [], [], []
        for label in "ABC":
            precision = (
                tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
            )
            recall = (
                tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            metrics[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support[label],
                "tp": tp[label],
                "fp": fp[label],
                "fn": fn[label],
            }
            if support[label]:
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
        result.update(
            {
                "macro_precision": sum(precisions) / len(precisions)
                if precisions
                else 0.0,
                "macro_recall": sum(recalls) / len(recalls) if recalls else 0.0,
                "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
                "option_metrics": metrics,
                "answer_distribution": dict(
                    Counter(v for v in parsed.values() if v is not None)
                ),
            }
        )
    return result, parsed, truth


def by_type(items, truth, parsed, qa_kind):
    """Summarize each question type using the same fixed denominator."""
    result = {}
    for t in range(1, 9):
        keys = [
            q
            for q in parsed
            if t
            in (
                truth[q].get("type_id")
                if isinstance(truth[q].get("type_id"), list)
                else [truth[q].get("type_id")]
            )
        ]
        sub = [
            {
                "qa_id": q,
                "text": ("是" if parsed[q] == "yes" else "否")
                if qa_kind == "ynqa"
                else parsed[q],
            }
            for q in keys
        ]
        # Compute a compact fixed-denominator type result directly, preserving invalids.
        total = len(keys)
        correct = sum(
            parsed[q]
            == (truth[q]["yn_answer"] if qa_kind == "ynqa" else truth[q]["mc_answer"])
            for q in keys
        )
        valid = sum(parsed[q] is not None for q in keys)
        result[str(t)] = {
            "total": total,
            "valid": valid,
            "correct": correct,
            "fixed_accuracy": correct / total if total else 0.0,
            "valid_accuracy": correct / valid if valid else 0.0,
            "parse_coverage": valid / total if total else 0.0,
        }
    return result


def main():
    gold = load(ROOT / "data" / "OFFICIAL_VAL_GOLD.json")["items"]
    output_specs = {
        "raw": RUNS / "RAW_PREDICTIONS.json",
        "internal_only": RUNS / "INTERNAL_HYBRID_PREDICTIONS.json",
        "predicted_both_only": RUNS / "PREDICTED_BOTH_PREDICTIONS.json",
    }
    internal_bundle = load(output_specs["internal_only"])["items"]
    output_specs["hybrid"] = None
    pred_maps = {
        "raw": load(output_specs["raw"])["items"],
        "internal_only": internal_bundle,
        "predicted_both_only": load(output_specs["predicted_both_only"])["items"],
        "hybrid": [
            {
                **x,
                "text": x["hybrid"]["text"],
                "condition": "hybrid",
                "generated_token_ids": x["hybrid"].get("generated_token_ids", []),
            }
            for x in internal_bundle
        ],
    }
    results = {
        "schema": "omnivchall_val_metrics_v1",
        "n_gold": len(gold),
        "conditions": {},
        "comparisons_to_raw": {},
    }
    # LOCAL_PATH: Both video directories below must exist to classify real/generated subsets.
    real_ids = {
        p.stem
        for p in (Path("/root/autodl-tmp/OmniVCHall/video_real/all_video")).glob(
            "*.mp4"
        )
    }
    generated_ids = {
        p.stem
        for p in (Path("/root/autodl-tmp/OmniVCHall/video_generated/all_video")).glob(
            "*.mp4"
        )
    }
    parsed_by_condition = {}
    for name, items in pred_maps.items():
        item_by = {x["qa_id"]: x for x in items}
        condition_result = {
            "n_predictions": len(items),
            "success_count": sum(bool(x.get("success")) for x in items),
            "subsets": {},
            "by_type_id": {},
            "video_splits": {},
        }
        all_parsed = {}
        for split in ("s_ynqa", "s_mcqa", "m_ynqa", "m_mcqa"):
            kind = "ynqa" if split.endswith("ynqa") else "mcqa"
            split_gold = [x for x in gold if x["qa_id"].startswith(split + "_id:")]
            split_preds = [
                item_by[x["qa_id"]] for x in split_gold if x["qa_id"] in item_by
            ]
            m, parsed, truth = counts_for(split_preds, split_gold, kind)
            condition_result["subsets"][split] = m
            all_parsed.update(parsed)
            condition_result["by_type_id"][split] = by_type(
                split_preds, truth, parsed, kind
            )
        # OmniVCHall's paper tables also separate real and generated videos.
        for video_split, video_ids in (
            ("video_real", real_ids),
            ("video_generated", generated_ids),
        ):
            condition_result["video_splits"][video_split] = {}
            for label, kind in (
                ("all_ynqa", "ynqa"),
                ("all_mcqa", "mcqa"),
                ("all_qa", None),
            ):
                gs = [
                    x
                    for x in gold
                    if x["video_id"] in video_ids
                    and (kind is None or x["question_type"] == kind)
                ]
                ps = [item_by[x["qa_id"]] for x in gs if x["qa_id"] in item_by]
                if kind is None:
                    vals = [(q, all_parsed.get(q["qa_id"])) for q in gs]
                    total = len(vals)
                    valid = sum(v is not None for _, v in vals)
                    correct = sum(
                        v == (q.get("yn_answer") or q.get("mc_answer")) for q, v in vals
                    )
                    condition_result["video_splits"][video_split][label] = {
                        "total": total,
                        "valid": valid,
                        "correct": correct,
                        "fixed_accuracy": correct / total if total else 0.0,
                        "valid_accuracy": correct / valid if valid else 0.0,
                        "parse_coverage": valid / total if total else 0.0,
                    }
                else:
                    m, _, _ = counts_for(ps, gs, kind)
                    condition_result["video_splits"][video_split][label] = m
        # Aggregate YN, MC, and all QA with fixed and valid denominators.
        for label, kind in (
            ("all_ynqa", "ynqa"),
            ("all_mcqa", "mcqa"),
            ("all_qa", None),
        ):
            gs = [x for x in gold if (kind is None or x["question_type"] == kind)]
            ps = [item_by[x["qa_id"]] for x in gs if x["qa_id"] in item_by]
            if kind is None:
                vals = [(q, all_parsed.get(q["qa_id"])) for q in gs]
                total = len(vals)
                valid = sum(v is not None for _, v in vals)
                correct = sum(
                    v == (q.get("yn_answer") or q.get("mc_answer")) for q, v in vals
                )
                condition_result["subsets"][label] = {
                    "total": total,
                    "valid": valid,
                    "invalid": total - valid,
                    "correct": correct,
                    "fixed_accuracy": correct / total if total else 0.0,
                    "valid_accuracy": correct / valid if valid else 0.0,
                    "parse_coverage": valid / total if total else 0.0,
                }
            else:
                m, _, _ = counts_for(ps, gs, kind)
                condition_result["subsets"][label] = m
        results["conditions"][name] = condition_result
        parsed_by_condition[name] = all_parsed
    raw_correct = {
        q["qa_id"]: parsed_by_condition["raw"].get(q["qa_id"])
        == (q.get("yn_answer") or q.get("mc_answer"))
        for q in gold
    }
    for name, parsed in parsed_by_condition.items():
        if name == "raw":
            continue
        fixed_correct = {
            q["qa_id"]: parsed.get(q["qa_id"])
            == (q.get("yn_answer") or q.get("mc_answer"))
            for q in gold
        }
        fixes = [
            q["qa_id"]
            for q in gold
            if fixed_correct[q["qa_id"]] and not raw_correct[q["qa_id"]]
        ]
        breaks = [
            q["qa_id"]
            for q in gold
            if raw_correct[q["qa_id"]] and not fixed_correct[q["qa_id"]]
        ]
        results["comparisons_to_raw"][name] = {
            "fix_count": len(fixes),
            "break_count": len(breaks),
            "net": len(fixes) - len(breaks),
            "fix_rate_all": len(fixes) / len(gold),
            "break_rate_all": len(breaks) / len(gold),
            "fix_ids": fixes,
            "break_ids": breaks,
        }
    # Source-state counts are diagnostic only; they do not enter answer scoring.
    src = load(RUNS / "SOURCE_PREDICTIONS.json")["items"]
    results["source_prediction_distribution"] = dict(
        Counter(x["source"]["state"] for x in src)
    )
    results["runtime"] = load(RUNS / "RUNTIME_COMPLETE.json")
    (RUNS / "VAL_METRICS.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    )
    for name in ("raw", "internal_only", "predicted_both_only", "hybrid"):
        s = results["conditions"][name]["subsets"]["all_qa"]
        print(
            name,
            f"fixed={s['fixed_accuracy']:.4f} ({s['correct']}/{s['total']})",
            f"valid={s['valid_accuracy']:.4f}",
            f"coverage={s['parse_coverage']:.4f}",
        )
    print(json.dumps(results["comparisons_to_raw"], ensure_ascii=False))


if __name__ == "__main__":
    main()
