"""Model dimensions, training constants, and experiment constraints."""

from __future__ import annotations

from typing import Final, Mapping


# No acronym expansion has been frozen.  The exact public string is the only
# authoritative paper name.
PUBLIC_METHOD_NAME: Final[str] = "MoSDeR"
METHOD_SPEC_ID: Final[str] = "mosder_method_spec_v1"
ARCHITECTURE_VERSION: Final[str] = "MoSDeR-v1"
IMPLEMENTATION_VARIANT: Final[str] = "headless_explicit_residual"

HISTORICAL_STRUCTURAL_CANDIDATE: Final[str] = (
    "MoSDeR-Headless-ExplicitResidual-StructuralCandidate-NonPromoted-v1"
)
HISTORICAL_STRUCTURAL_EVIDENCE_STATUS: Final[str] = (
    "PASS_EXPOSED_CAL80_HEADLESS_EXACT_NONPROMOTED"
)
HISTORICAL_PROVENANCE: Final[tuple[Mapping[str, str], ...]] = (
    {
        "method_name": ("MoSDeR-NoO18-V3-FrozenPhysicalInit-TrainOnlyAblation-v1"),
        "status": "COMPLETE_MOLMO_TRAIN400_NO_O18_P0_L2_DEVELOPMENT_ONLY",
        "relation": "development_ancestor_not_mosder_v1",
    },
    {
        "method_name": "MoSDeR-NoO18-V3-StageG-G2-TrainOnlyAblation-v1",
        "status": ("COMPLETE_MOLMO_TRAIN400_NO_O18_STAGE_G_G2_DEVELOPMENT_ONLY"),
        "relation": "development_ancestor_not_mosder_v1",
    },
    {
        "method_name": ("MoSDeR-NoO18-StageG-ExplicitResidual-TrainOnlyCandidate-v1"),
        "status": (
            "COMPLETE_MOLMO_TRAIN400_NO_O18_EXPLICIT_RESIDUAL_R1_DEVELOPMENT_ONLY"
        ),
        "relation": "development_ancestor_not_mosder_v1",
    },
    {
        "method_name": HISTORICAL_STRUCTURAL_CANDIDATE,
        "status": HISTORICAL_STRUCTURAL_EVIDENCE_STATUS,
        "relation": "structural_evidence_artifact_not_mosder_v1",
    },
)
HISTORICAL_IDENTITIES_NOT_ALIASES: Final[tuple[str, ...]] = (
    "T-SCMA",
    "T-SCMA A+B",
    "T-SCMA-Traj",
    "GeoT",
    "TsT-plugin",
    "TsT-plugin-O18",
    "TsT-NA-O18",
    "TsT-NA-O18-FCR-V2",
    "TsT-NA-O18-FCR-V3-FactorSpan",
    "TsT-NA-O18-FCR-V3-LG",
    "TsT-NA-O18-FCR-V3-LG2-ContinuousEvidence",
    "ARUQ-R4.1",
)

HISTORICAL_CAL80_DEVELOPMENT_EVIDENCE: Final[Mapping[str, object]] = {
    "role": "TRAIN_ONLY_EXPOSED_CAL80_DEVELOPMENT_DIAGNOSTIC",
    "rows": 80,
    "factor_joint_correct": 66,
    "language_state_correct": 76,
    "canonical_description_correct": 74,
    "strict_natural_language_gate_passed": False,
    "factor_records_ieee754_generation_ids_and_text_exact": True,
    "formal_or_generalization_evidence": False,
}
HISTORICAL_O18_DEVELOPMENT_EVIDENCE: Final[Mapping[str, object]] = {
    "time_semantics": "observed_window_three_anchor_reconstruction_not_future",
    "camera_translation_ade_m": 0.05887,
    "camera_rotation_error_deg": 4.7357,
    "object_translation_ade_m": 0.04300,
    "object_rotation_error_deg": 15.1228,
    "v3_stage_h_stage_t_all_four_beat_zero_and_constant": False,
    "trajectory_contribution_established": False,
}

TARGET_FAMILIES: Final[tuple[str, ...]] = (
    "qwen3_vl_8b",
    "molmo2_o_7b",
    "nvila_lite_8b",
)
# Backend family allowlist.
SUPPORTED_FAMILIES: Final[tuple[str, ...]] = TARGET_FAMILIES
DISPLAY_NAMES: Final[Mapping[str, str]] = {
    "qwen3_vl_8b": "Qwen3-VL-8B",
    "molmo2_o_7b": "Molmo2-O-7B",
    "nvila_lite_8b": "NVILA-Lite-8B",
}
FACTORSPAN_LAYERS: Final[Mapping[str, tuple[int, ...]]] = {
    "qwen3_vl_8b": (31, 32, 33, 34),
    "molmo2_o_7b": (27, 28, 29, 30),
    "nvila_lite_8b": (23, 24, 25, 26),
}

RGB_FRAME_COUNT: Final[int] = 20
TEMPORAL_UNITS: Final[int] = 10
SOURCE_RANK: Final[int] = 32
TRILORA_RANK: Final[int] = 8
TRILORA_ALPHA: Final[float] = 16.0
TRILORA_DROPOUT: Final[float] = 0.0
BASE_SOURCE_RESIDUAL_SCALE: Final[float] = 0.05
EXPLICIT_RESIDUAL_RANK: Final[int] = 16
EXPLICIT_BRANCH_MAX_ABS_DELTA: Final[float] = 0.10

STATE_ORDER: Final[tuple[str, ...]] = (
    "neither",
    "camera_only",
    "object_only",
    "both",
)
STATE_FACTORS: Final[Mapping[str, tuple[bool, bool]]] = {
    "neither": (False, False),
    "camera_only": (True, False),
    "object_only": (False, True),
    "both": (True, True),
}
DECISION_RULE: Final[str] = "q_c>0_and_q_o>0_zero_is_static"

LANGUAGE_SCHEMA_LINES: Final[tuple[str, ...]] = (
    "Camera:",
    "Object:",
    "State:",
    "Description:",
)

HOLD_STATUS: Final[str] = "HOLD_GATE2_V3"
FORMAL_TRAINING_AUTHORIZED: Final[bool] = False
CONFIRMATION_A_OPENED: Final[bool] = False
FINAL_B_OPENED: Final[bool] = False
HELD_ROLES_OPENED: Final[bool] = False

IMPLEMENTATION_EVIDENCE: Final[Mapping[str, object]] = {
    "architecture_selection_hash_seal_complete": True,
    "cpu_contract_tests_passed": 85,
    "three_family_real_fgr_connectivity_smoke_complete": True,
    "three_family_real_fgr_connectivity_smoke_is_accuracy": False,
    "formal_runner_complete": False,
    "formal_checkpoint_resume_smoke_complete": False,
    "formal_protocol_and_seed_seal_complete": False,
}

NEGATIVE_AUTHORITY: Final[Mapping[str, object]] = {
    "historical_candidate_retroactively_renamed": False,
    "historical_candidate_promoted": False,
    "architecture_selection_is_empirical_result": False,
    "architecture_hash_seal_complete": True,
    "formal_training_authorized": FORMAL_TRAINING_AUTHORIZED,
    "formal_validation_authorized": False,
    "formal_data_release": False,
    "formal_release": False,
    "paper_final_empirical_method": False,
    "three_family_real_runner_smoke_complete": True,
    "three_family_empirical_closure": False,
    "generalization_evidence": False,
    "natural_language_usable_established": False,
    "strict_natural_language_gate_passed": False,
    "historical_strict_nl_canonical_correct": 74,
    "historical_strict_nl_canonical_total": 80,
    "open_ended_intent_language_claim": False,
    "cal80_is_exposed_development": True,
    "cal80_formal_evidence": False,
    "fresh80_formal_evidence": False,
    "numeric_trajectory_output": False,
    "numeric_trajectory_quality_established": False,
    "future_trajectory_claim": False,
    "historical_o18_was_future_forecasting": False,
    "all_trajectory_derived_supervision_removed": False,
    "gates_3_to_10_formal_pass": False,
    "confirmation_a_opened": CONFIRMATION_A_OPENED,
    "final_b_opened": FINAL_B_OPENED,
    "held_roles_opened": HELD_ROLES_OPENED,
    "hold_status": HOLD_STATUS,
    "hold_status_changed": False,
}


def architecture_contract() -> Mapping[str, object]:
    """Return the selected, family-neutral method topology."""

    return {
        "public_method_name": PUBLIC_METHOD_NAME,
        "method_spec_id": METHOD_SPEC_ID,
        "architecture_version": ARCHITECTURE_VERSION,
        "architecture_status": (
            "current_final_architecture_selected_and_hash_sealed_not_empirically_final"
        ),
        "implementation_variant": IMPLEMENTATION_VARIANT,
        "target_families": list(TARGET_FAMILIES),
        "families": list(TARGET_FAMILIES),
        "implementation_status": {
            "direct_family_backend_module_present": True,
            "fgr_objective_module_present": True,
            "cpu_toy_integration_tests_present": True,
            "three_family_real_runner_smoke_complete": True,
            "formal_runner_complete": False,
            "architecture_hash_seal_complete": True,
        },
        "input": {
            "rgb_frames": RGB_FRAME_COUNT,
            "temporal_units": TEMPORAL_UNITS,
            "query_required": True,
            "oracle_box_track_required": True,
            "timestamps_role": "sampling_and_order_validation_not_model_feature",
        },
        "source_core": {
            "camera_input": "context_features",
            "object_input": "target_features",
            "output_fields": ["camera_source", "object_source"],
            "numeric_18d_heads": False,
            "numeric_18d_outputs": False,
            "summary_outputs": False,
            "stage_p": False,
            "both_sources_may_be_computed_per_native_forward": True,
            "opposite_owner_source_consumed_by_factor_route": False,
        },
        "decoder": {
            "family_native_decoder_and_lm_head": True,
            "frozen_base": True,
            "routes": [
                "CAMERA_FACTOR",
                "OBJECT_FACTOR",
                "FULL_LANGUAGE",
            ],
            "route_local_camera_object_trilora": True,
            "shared_language_trilora": True,
        },
        "decision": {
            "camera_margin": "camera_C-camera_N",
            "object_margin": "0.5*((object_O-object_N)+(object_B-object_C))",
            "learned_train_only_intercepts": ["b_c", "b_o"],
            "rule": DECISION_RULE,
            "state_order": list(STATE_ORDER),
        },
        "explicit_residual": {
            "full_language_only": True,
            "raw_native_path_preserved": True,
            "camera_object_parameters_independent": True,
            "zero_output_initialized": True,
            "bounded": True,
            "source_stop_gradient": True,
        },
        "training": {
            "stages": ["F", "G", "R"],
            "full_native_vlm_frozen_in_all_stages": True,
            "stage_f_exact_two_pass_one_live_graph": True,
            "stage_f_score_replay_bitwise_exact": True,
            "stage_f_candidates_canonical_exact": True,
            "stage_route_binding_fail_closed": True,
            "stage_g_owner": "shared_trilora_only",
            "stage_r_owner": "eight_explicit_residual_tensors_only",
            "language_score_differentiability_required": True,
            "gradient_checkpointing_allowed": False,
            "source_decision_residual_master_dtype": "float32",
            "fp16_native_w0_uses_fp32_trilora_master": True,
        },
        "supervision": {
            "dense_camera18_object18_reconstruction_heads": False,
            "dense_camera18_object18_reconstruction_loss": False,
            # Dense numeric auxiliary regression excludes state labels
            # derived from trajectory thresholds.
            "dense_camera18_object18_regression": False,
            "four_state_labels_trajectory_threshold_derived": True,
        },
        "outputs": {
            "four_state": True,
            "four_line_motion_description_target": True,
            "natural_language_usable_established": False,
            "numeric_or_future_trajectory": False,
        },
    }


def validate_contract() -> Mapping[str, object]:
    value = architecture_contract()
    required_false = (
        "historical_candidate_retroactively_renamed",
        "historical_candidate_promoted",
        "architecture_selection_is_empirical_result",
        "formal_training_authorized",
        "formal_validation_authorized",
        "formal_data_release",
        "formal_release",
        "paper_final_empirical_method",
        "three_family_empirical_closure",
        "generalization_evidence",
        "natural_language_usable_established",
        "strict_natural_language_gate_passed",
        "open_ended_intent_language_claim",
        "cal80_formal_evidence",
        "fresh80_formal_evidence",
        "numeric_trajectory_output",
        "numeric_trajectory_quality_established",
        "future_trajectory_claim",
        "historical_o18_was_future_forecasting",
        "all_trajectory_derived_supervision_removed",
        "gates_3_to_10_formal_pass",
        "confirmation_a_opened",
        "final_b_opened",
        "held_roles_opened",
        "hold_status_changed",
    )
    if (
        tuple(value["target_families"]) != TARGET_FAMILIES
        or tuple(value["families"]) != TARGET_FAMILIES
        or value["source_core"]["output_fields"] != ["camera_source", "object_source"]
        or value["source_core"]["numeric_18d_heads"] is not False
        or value["source_core"]["stage_p"] is not False
        or value["explicit_residual"]["raw_native_path_preserved"] is not True
        or value["explicit_residual"]["zero_output_initialized"] is not True
        or value["supervision"]["dense_camera18_object18_reconstruction_loss"]
        is not False
        or value["outputs"]["natural_language_usable_established"] is not False
        or value["implementation_status"]["architecture_hash_seal_complete"] is not True
        or value["implementation_status"]["three_family_real_runner_smoke_complete"]
        is not True
        or NEGATIVE_AUTHORITY["architecture_hash_seal_complete"] is not True
        or NEGATIVE_AUTHORITY["three_family_real_runner_smoke_complete"] is not True
        or IMPLEMENTATION_EVIDENCE["architecture_selection_hash_seal_complete"]
        is not True
        or IMPLEMENTATION_EVIDENCE["three_family_real_fgr_connectivity_smoke_complete"]
        is not True
        or IMPLEMENTATION_EVIDENCE["cpu_contract_tests_passed"] != 85
        or set(required_false) - set(NEGATIVE_AUTHORITY)
        or any(NEGATIVE_AUTHORITY[key] is not False for key in required_false)
        or NEGATIVE_AUTHORITY["cal80_is_exposed_development"] is not True
        or NEGATIVE_AUTHORITY["historical_strict_nl_canonical_correct"] != 74
        or NEGATIVE_AUTHORITY["historical_strict_nl_canonical_total"] != 80
        or NEGATIVE_AUTHORITY["hold_status"] != HOLD_STATUS
        or FORMAL_TRAINING_AUTHORIZED is not False
    ):
        raise AssertionError("MoSDeR-v1 selected architecture contract drifted")
    return value


__all__ = [
    "ARCHITECTURE_VERSION",
    "BASE_SOURCE_RESIDUAL_SCALE",
    "CONFIRMATION_A_OPENED",
    "DECISION_RULE",
    "DISPLAY_NAMES",
    "EXPLICIT_BRANCH_MAX_ABS_DELTA",
    "EXPLICIT_RESIDUAL_RANK",
    "FACTORSPAN_LAYERS",
    "FINAL_B_OPENED",
    "FORMAL_TRAINING_AUTHORIZED",
    "HELD_ROLES_OPENED",
    "HISTORICAL_CAL80_DEVELOPMENT_EVIDENCE",
    "HISTORICAL_IDENTITIES_NOT_ALIASES",
    "HISTORICAL_O18_DEVELOPMENT_EVIDENCE",
    "HISTORICAL_PROVENANCE",
    "HISTORICAL_STRUCTURAL_CANDIDATE",
    "HISTORICAL_STRUCTURAL_EVIDENCE_STATUS",
    "HOLD_STATUS",
    "IMPLEMENTATION_VARIANT",
    "IMPLEMENTATION_EVIDENCE",
    "LANGUAGE_SCHEMA_LINES",
    "METHOD_SPEC_ID",
    "NEGATIVE_AUTHORITY",
    "PUBLIC_METHOD_NAME",
    "RGB_FRAME_COUNT",
    "SOURCE_RANK",
    "STATE_FACTORS",
    "STATE_ORDER",
    "SUPPORTED_FAMILIES",
    "TARGET_FAMILIES",
    "TEMPORAL_UNITS",
    "TRILORA_ALPHA",
    "TRILORA_DROPOUT",
    "TRILORA_RANK",
    "architecture_contract",
    "validate_contract",
]
