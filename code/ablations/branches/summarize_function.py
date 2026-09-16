"""Compare intervention arms without changing their saved predictions."""

import json

import numpy as np
from common import read_jsonl, write_json


def summarize(plan, output_dir):
    """Report all planned arms and paired, source-stratified group intervals."""
    metrics_by_arm = {
        arm: json.loads((output_dir / arm / "val1200/metrics.json").read_text())
        for arm in plan["arms"]
    }
    table = [
        "# Molmo完整checkpoint推理时干预",
        "",
        "同一原始84.08% R checkpoint，权重冻结，仅改变推理计算；完整val1200。此实验检验组件依赖性，不是重新训练的结构消融。",
        "",
        "| 配置 | 正确数 | 四状态Acc | macro-F1 | Camera Acc | Object Acc | 解析失败 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    names = {
        "full": "完整MoSDeR",
        "no_camera": "关闭Camera分支",
        "no_object": "关闭Object分支",
    }
    for arm, metrics in metrics_by_arm.items():
        table.append(
            f"| {names[arm]} | {metrics['correct']}/1200 | {metrics['accuracy']:.2%} | {metrics['macro_f1']:.2%} | {metrics['camera_accuracy']:.2%} | {metrics['object_accuracy']:.2%} | {metrics['parse_failures']} |"
        )
    # Paired cluster bootstrap over collection groups, stratified by source.

    predictions_by_arm = {
        arm: read_jsonl(output_dir / arm / "val1200/predictions.jsonl")
        for arm in plan["arms"]
    }
    case_ids = [x["case_id"] for x in predictions_by_arm["full"]]
    assert all(
        [x["case_id"] for x in v] == case_ids for v in predictions_by_arm.values()
    )
    strata = {}
    for i, x in enumerate(predictions_by_arm["full"]):
        strata.setdefault(x["source"], {}).setdefault(x["split_group"], []).append(i)
    contrasts = {}
    for arm in plan["arms"][1:]:
        delta = np.array(
            [
                int(x["correct"]) - int(y["correct"])
                for x, y in zip(predictions_by_arm["full"], predictions_by_arm[arm])
            ]
        )
        draws = []
        rng = np.random.default_rng(20260914)
        for _ in range(2000):
            value = 0
            for source, groups in strata.items():
                group_indices = list(groups.values())
                selected_groups = rng.integers(
                    0, len(group_indices), size=len(group_indices)
                )
                indices = np.concatenate([group_indices[j] for j in selected_groups])
                source_weight = sum(len(v) for v in group_indices) / 1200
                value += source_weight * float(delta[indices].mean())
            draws.append(value)
        contrasts[arm] = dict(
            full_minus_ablation=float(delta.mean()),
            group_bootstrap_95pct=np.quantile(draws, [0.025, 0.975]).tolist(),
            resamples=2000,
            note="source-stratified collection-group bootstrap; single training seed, not training variance",
        )
        lower, upper = contrasts[arm]["group_bootstrap_95pct"]
        table += [
            "",
            f"完整模型减去{names[arm]}：{delta.mean() * 100:+.2f}个百分点；采集组bootstrap区间[{lower * 100:+.2f}, {upper * 100:+.2f}]。",
        ]
    table += [
        "",
        "按所有预定配置报告；接近零或方向相反也保留。性能变化包括推理干预导致的分布变化，不据此断言删除模块后重新训练的能力。",
    ]
    write_json(
        output_dir / "RESULTS.json",
        dict(metrics=metrics_by_arm, paired_contrasts=contrasts),
    )
    (output_dir / "RESULTS.md").write_text("\n".join(table) + "\n")
