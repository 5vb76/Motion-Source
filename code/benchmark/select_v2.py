"""Choose V2 cases while reserving complete AV2 logs and TACO capture days."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

# LOCAL_PATH: parents[1] resolves to code/ here; set the benchmark data root
# before using the release/, candidates/, or raw/ subdirectories.
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
# LOCAL_PATH: Requires this sibling data directory and its scripts/ modules.
PREVIOUS_RELEASE_ROOT = BENCHMARK_ROOT.with_name("general_motion_benchmark_v1")
sys.path.insert(0, str(PREVIOUS_RELEASE_ROOT / "scripts"))
from build_release import write as write_json
from build_release import write_rows as write_jsonl
from select_release import STATES
from select_release import read as read_jsonl
from select_release import select as select_cases


def main():
    previous_train = read_jsonl(PREVIOUS_RELEASE_ROOT / "release/train.jsonl")
    av2_candidates = [
        row
        for row in read_jsonl(
            PREVIOUS_RELEASE_ROOT / "av2_rgb20/source_train_eligible.jsonl"
        )
        if all(Path(f["path"]).is_file() for f in row["model_input"]["rgb20_frames"])
    ]
    groups = sorted({row["split_group"] for row in av2_candidates})
    group_state_counts = np.array(
        [
            [
                sum(
                    row["split_group"] == g and row["state"] == s
                    for row in av2_candidates
                )
                for g in groups
            ]
            for s in STATES
        ],
        float,
    )
    # Reserve complete logs, while retaining at least 35 neither and 80 object-only training candidates.
    av2_test_quota = np.array([70, 185, 40, 185])
    state_totals = group_state_counts.sum(axis=1)
    objective = group_state_counts.sum(axis=0) + np.arange(len(groups)) * 1e-6
    solution = milp(
        objective,
        integrality=np.ones(len(groups)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(
            group_state_counts,
            av2_test_quota,
            state_totals - np.array([35, 150, 80, 150]),
        ),
        options={"time_limit": 60},
    )
    if solution.x is None:
        raise RuntimeError(solution.message)
    heldout_logs = {g for g, x in zip(groups, solution.x) if x > 0.5}
    heldout_days = {"TACO_capture_day:20231020", "TACO_capture_day:20231026"}
    adt_candidates = read_jsonl(
        BENCHMARK_ROOT / "candidates/adt_testA_rgb20_eligible.jsonl"
    )
    for row in adt_candidates:
        row["source"] = "ADT-LiteOffice"
    hot_candidates = read_jsonl(
        BENCHMARK_ROOT / "candidates/hot_testA_rgb20_eligible.jsonl"
    )
    quotas = {
        "ADT-LiteOffice": [219, 85, 85, 91],
        "HOT3D": [11, 30, 1, 24],
        "TACO-V1-allocentric": [0, 0, 174, 0],
        "AV2": av2_test_quota.tolist(),
    }
    test_pool = (
        adt_candidates
        + hot_candidates
        + [
            row
            for row in previous_train
            if row["source"] == "TACO-V1-allocentric"
            and row["split_group"] in heldout_days
        ]
        + [row for row in av2_candidates if row["split_group"] in heldout_logs]
    )
    test = select_cases(test_pool, quotas)
    train_pool = [
        row
        for row in previous_train
        if row["source"] != "AV2"
        and not (
            row["source"] == "TACO-V1-allocentric"
            and row["split_group"] in heldout_days
        )
    ] + [row for row in av2_candidates if row["split_group"] not in heldout_logs]
    # Keep non-egocentric coverage and 1,250 cases/state; ADT fills remaining cells.
    train_quotas = {s: [0] * 4 for s in quotas}
    for state_index, state in enumerate(STATES):
        remaining = 1250
        for source in ["AV2", "HOT3D", "TACO-V1-allocentric"]:
            n = min(
                sum(
                    row["source"] == source and row["state"] == state
                    for row in train_pool
                ),
                remaining,
                900 if source == "TACO-V1-allocentric" else 5000,
            )
            train_quotas[source][state_index] = n
            remaining -= n
        train_quotas["ADT-LiteOffice"][state_index] = remaining
    train = select_cases(train_pool, train_quotas)
    val = read_jsonl(PREVIOUS_RELEASE_ROOT / "release/evaluation_reused.jsonl")
    write_jsonl(BENCHMARK_ROOT / "candidates/test_selected.jsonl", test)
    write_jsonl(BENCHMARK_ROOT / "candidates/train_selected.jsonl", train)
    write_jsonl(BENCHMARK_ROOT / "candidates/val_selected.jsonl", val)
    write_json(
        BENCHMARK_ROOT / "candidates/selection_protocol.json",
        {
            "counts": {"train": 5000, "val": 1200, "test": 1200},
            "states_order": STATES,
            "train_source_state_quotas": train_quotas,
            "test_source_state_quotas": quotas,
            "held_out_av2_logs": sorted(heldout_logs),
            "held_out_taco_capture_days": sorted(heldout_days),
            "heldout_av2_capacity": dict(
                zip(
                    STATES,
                    (
                        group_state_counts[:, [g in heldout_logs for g in groups]].sum(
                            axis=1
                        )
                    )
                    .astype(int)
                    .tolist(),
                )
            ),
            "selection_uses_model_predictions": False,
            "selection": "deterministic SHA256 tie break, favors unique windows and sequence coverage",
            "historical_exposure": "val inherits reused development; test ADT/HOT from old testA reserves, TACO/AV2 held out from historical train candidate groups. Only evaluate newly trained models that exclude these groups; no fully fresh-test claim.",
        },
    )
    print(
        json.dumps(
            {
                "train_quotas": train_quotas,
                "test_quotas": quotas,
                "AV2_groups": len(heldout_logs),
                "heldout_av2_capacity": group_state_counts[
                    :, [g in heldout_logs for g in groups]
                ]
                .sum(axis=1)
                .tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
