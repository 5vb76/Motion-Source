"""Build and validate AdamW parameter groups for each F/G/R stage."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
from torch import nn

from .training import (
    PREFREEZE_OPTIMIZER_RECIPE,
    StageConfiguration,
    TrainingContractError,
    TrainingStage,
)


GROUP_NAME_KEY = "mosder_group"
GROUP_STAGE_KEY = "mosder_stage"
GROUP_PARAMETER_NAMES_KEY = "mosder_parameter_names"


class OptimizerContractError(TrainingContractError):
    """The optimizer partition, hyperparameters, or state violated its ABI."""


NamedParameters = tuple[tuple[str, nn.Parameter], ...]


def _finite_float(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OptimizerContractError(f"{name} must be numeric")
    output = float(value)
    if not math.isfinite(output) or (positive and output <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise OptimizerContractError(f"{name} must be {qualifier}")
    return output


def _common_recipe() -> tuple[tuple[float, float], float]:
    if PREFREEZE_OPTIMIZER_RECIPE.get("optimizer") != "AdamW":
        raise OptimizerContractError("pre-freeze optimizer must remain AdamW")
    betas = PREFREEZE_OPTIMIZER_RECIPE.get("betas")
    if (
        not isinstance(betas, (tuple, list))
        or len(betas) != 2
        or isinstance(betas[0], bool)
        or isinstance(betas[1], bool)
    ):
        raise OptimizerContractError("AdamW betas recipe is invalid")
    beta_values = (
        _finite_float(betas[0], name="AdamW beta1"),
        _finite_float(betas[1], name="AdamW beta2"),
    )
    if not 0.0 <= beta_values[0] < 1.0 or not 0.0 <= beta_values[1] < 1.0:
        raise OptimizerContractError("AdamW betas must lie in [0,1)")
    eps = _finite_float(
        PREFREEZE_OPTIMIZER_RECIPE.get("eps"),
        name="AdamW eps",
        positive=True,
    )
    return beta_values, eps


def _stage_recipe(stage: TrainingStage) -> Mapping[str, object]:
    value = PREFREEZE_OPTIMIZER_RECIPE.get(stage.value)
    if not isinstance(value, Mapping):
        raise OptimizerContractError(
            f"pre-freeze optimizer recipe has no Stage-{stage.value} mapping"
        )
    return value


def _owner_entries(configuration: StageConfiguration) -> Mapping[str, NamedParameters]:
    if not isinstance(configuration, StageConfiguration):
        raise OptimizerContractError(
            "optimizer construction requires a StageConfiguration"
        )
    if not isinstance(configuration.stage, TrainingStage):
        raise OptimizerContractError("StageConfiguration has an invalid stage")
    configuration.owners.validate()
    owners = {
        "camera": tuple(configuration.owners.camera),
        "object": tuple(configuration.owners.object),
        "shared": tuple(configuration.owners.shared),
        "residual": tuple(configuration.owners.residual),
    }
    all_entries = tuple(value for owner in owners.values() for value in owner)
    if any(
        not isinstance(name, str) or not name or not isinstance(parameter, nn.Parameter)
        for name, parameter in all_entries
    ):
        raise OptimizerContractError("an owner entry is not a named Parameter")
    return owners


def _validate_owner_name(owner: str, name: str) -> None:
    if owner == "camera":
        valid = (
            name.startswith("source_core.camera_source_adapter.")
            or (
                name.startswith("wrappers.")
                and any(
                    marker in name
                    for marker in (
                        ".camera_A.",
                        ".camera_B.",
                        ".camera_source_gate.",
                    )
                )
            )
            or name == "decision.b_c"
        )
    elif owner == "object":
        valid = (
            name.startswith("source_core.object_source_adapter.")
            or (
                name.startswith("wrappers.")
                and any(
                    marker in name
                    for marker in (
                        ".object_A.",
                        ".object_B.",
                        ".object_source_gate.",
                    )
                )
            )
            or name == "decision.b_o"
        )
    elif owner == "shared":
        valid = name.startswith("wrappers.") and any(
            marker in name for marker in (".shared_A.", ".shared_B.")
        )
    elif owner == "residual":
        valid = name.startswith("explicit_residual.")
    else:
        raise OptimizerContractError(f"unknown parameter owner: {owner!r}")
    if not valid or ".base." in name:
        raise OptimizerContractError(
            f"parameter name is outside its {owner} owner ABI: {name}"
        )


def _current_stage_entries(configuration: StageConfiguration) -> NamedParameters:
    owners = _owner_entries(configuration)
    for owner, values in owners.items():
        for name, _ in values:
            _validate_owner_name(owner, name)
    if configuration.stage is TrainingStage.F:
        selected_owners = ("camera", "object")
    elif configuration.stage is TrainingStage.G:
        selected_owners = ("shared",)
    elif configuration.stage is TrainingStage.R:
        selected_owners = ("residual",)
    else:
        raise OptimizerContractError("FROZEN has no optimizer")
    selected = tuple(entry for owner in selected_owners for entry in owners[owner])
    expected_names = tuple(name for name, _ in selected)
    if (
        not isinstance(configuration.trainable_names, tuple)
        or configuration.trainable_names != expected_names
    ):
        raise OptimizerContractError(
            "StageConfiguration trainable_names differs from its ordered allowlist"
        )
    selected_ids = {id(parameter) for _, parameter in selected}
    if len(selected_ids) != len(selected):
        raise OptimizerContractError("current-stage parameters are duplicated")
    fp16_names = [
        name for name, parameter in selected if parameter.dtype == torch.float16
    ]
    if fp16_names:
        raise OptimizerContractError(
            {
                "reason": "AdamW eps=1e-8 forbids FP16 trainable masters",
                "parameters": fp16_names,
            }
        )
    for owner, values in owners.items():
        for name, parameter in values:
            expected_trainable = owner in selected_owners
            if parameter.requires_grad is not expected_trainable:
                state = "trainable" if expected_trainable else "frozen"
                raise OptimizerContractError(
                    f"{name} must be {state} for Stage-{configuration.stage.value}"
                )
            if not expected_trainable and id(parameter) in selected_ids:
                raise OptimizerContractError(
                    "a frozen owner entered the stage allowlist"
                )
    if not selected:
        raise OptimizerContractError("current stage has no trainable parameters")
    return selected


def _append_group(
    output: list[dict[str, object]],
    *,
    stage: TrainingStage,
    name: str,
    values: Sequence[tuple[str, nn.Parameter]],
    lr: float,
    weight_decay: float,
) -> None:
    entries = tuple(values)
    if not entries:
        raise OptimizerContractError(f"optimizer group {name} is empty")
    names = tuple(parameter_name for parameter_name, _ in entries)
    parameters = tuple(parameter for _, parameter in entries)
    if (
        len(names) != len(set(names))
        or len(parameters) != len({id(parameter) for parameter in parameters})
        or any(not parameter.requires_grad for parameter in parameters)
    ):
        raise OptimizerContractError(
            f"optimizer group {name} is duplicated or contains a frozen parameter"
        )
    output.append(
        {
            "params": parameters,
            "lr": lr,
            "weight_decay": weight_decay,
            GROUP_NAME_KEY: name,
            GROUP_STAGE_KEY: stage.value,
            GROUP_PARAMETER_NAMES_KEY: names,
        }
    )


def _stage_group_definitions(
    configuration: StageConfiguration,
) -> tuple[dict[str, object], ...]:
    selected = _current_stage_entries(configuration)
    stage = configuration.stage
    recipe = _stage_recipe(stage)
    groups: list[dict[str, object]] = []

    if stage is TrainingStage.F:
        source_lr = _finite_float(
            recipe.get("source_lr"), name="Stage-F source LR", positive=True
        )
        trilora_lr = _finite_float(
            recipe.get("trilora_lr"), name="Stage-F TriLoRA LR", positive=True
        )
        bias_lr = _finite_float(
            recipe.get("bias_lr"), name="Stage-F intercept LR", positive=True
        )
        matrix_decay = _finite_float(
            recipe.get("matrix_weight_decay"),
            name="Stage-F matrix weight decay",
        )
        no_decay = _finite_float(
            recipe.get("norm_gate_bias_weight_decay"),
            name="Stage-F norm/gate/bias weight decay",
        )
        if matrix_decay < 0.0 or no_decay != 0.0:
            raise OptimizerContractError("Stage-F weight-decay recipe drifted")
        buckets: dict[str, list[tuple[str, nn.Parameter]]] = {
            "source_matrix_time_mixer": [],
            "source_norm_bias": [],
            "trilora_matrix": [],
            "trilora_source_gate": [],
            "camera_intercept": [],
            "object_intercept": [],
        }
        source_decay_suffixes = (
            ".time_embedding",
            ".down.weight",
            ".up.weight",
            ".temporal_mixer.weight",
        )
        source_no_decay_suffixes = (
            ".input_norm.weight",
            ".input_norm.bias",
            ".down.bias",
            ".up.bias",
            ".temporal_mixer.bias",
            ".output_norm.weight",
            ".output_norm.bias",
        )
        for entry in selected:
            name, _ = entry
            if name.startswith("source_core."):
                if name.endswith(source_decay_suffixes):
                    buckets["source_matrix_time_mixer"].append(entry)
                elif name.endswith(source_no_decay_suffixes):
                    buckets["source_norm_bias"].append(entry)
                else:
                    raise OptimizerContractError(
                        f"unknown Stage-F source parameter: {name}"
                    )
            elif name.startswith("wrappers."):
                if any(
                    marker in name
                    for marker in (
                        ".camera_A.",
                        ".camera_B.",
                        ".object_A.",
                        ".object_B.",
                    )
                ) and name.endswith(".weight"):
                    buckets["trilora_matrix"].append(entry)
                elif any(
                    marker in name
                    for marker in (
                        ".camera_source_gate.",
                        ".object_source_gate.",
                    )
                ) and name.endswith(".weight"):
                    buckets["trilora_source_gate"].append(entry)
                else:
                    raise OptimizerContractError(
                        f"unknown Stage-F TriLoRA parameter: {name}"
                    )
            elif name == "decision.b_c":
                buckets["camera_intercept"].append(entry)
            elif name == "decision.b_o":
                buckets["object_intercept"].append(entry)
            else:
                raise OptimizerContractError(
                    f"undeclared Stage-F optimizer parameter: {name}"
                )
        values = (
            (
                "source_matrix_time_mixer_decay",
                buckets["source_matrix_time_mixer"],
                source_lr,
                matrix_decay,
            ),
            (
                "source_norm_bias_no_decay",
                buckets["source_norm_bias"],
                source_lr,
                no_decay,
            ),
            (
                "trilora_matrix_decay",
                buckets["trilora_matrix"],
                trilora_lr,
                matrix_decay,
            ),
            (
                "trilora_source_gate_no_decay",
                buckets["trilora_source_gate"],
                trilora_lr,
                no_decay,
            ),
            (
                "camera_intercept_no_decay",
                buckets["camera_intercept"],
                bias_lr,
                no_decay,
            ),
            (
                "object_intercept_no_decay",
                buckets["object_intercept"],
                bias_lr,
                no_decay,
            ),
        )
        for name, parameters, lr, decay in values:
            _append_group(
                groups,
                stage=stage,
                name=name,
                values=parameters,
                lr=lr,
                weight_decay=decay,
            )
    elif stage is TrainingStage.G:
        lr = _finite_float(
            recipe.get("shared_trilora_lr"),
            name="Stage-G Shared TriLoRA LR",
            positive=True,
        )
        decay = _finite_float(recipe.get("weight_decay"), name="Stage-G weight decay")
        if decay < 0.0 or any(
            not name.startswith("wrappers.")
            or not any(marker in name for marker in (".shared_A.", ".shared_B."))
            or not name.endswith(".weight")
            for name, _ in selected
        ):
            raise OptimizerContractError("Stage-G Shared TriLoRA ABI drifted")
        _append_group(
            groups,
            stage=stage,
            name="shared_trilora_decay",
            values=selected,
            lr=lr,
            weight_decay=decay,
        )
    elif stage is TrainingStage.R:
        lr = _finite_float(
            recipe.get("explicit_residual_lr"),
            name="Stage-R explicit residual LR",
            positive=True,
        )
        decay = _finite_float(recipe.get("weight_decay"), name="Stage-R weight decay")
        expected_suffixes = {
            f"explicit_residual.{owner}.{leaf}"
            for owner in ("camera", "object")
            for leaf in (
                "down.weight",
                "up.weight",
                "gate.weight",
                "gate.bias",
            )
        }
        if decay < 0.0 or {name for name, _ in selected} != expected_suffixes:
            raise OptimizerContractError("Stage-R residual parameter ABI drifted")
        _append_group(
            groups,
            stage=stage,
            name="explicit_residual_decay",
            values=selected,
            lr=lr,
            weight_decay=decay,
        )
    else:  # guarded by _current_stage_entries
        raise OptimizerContractError("FROZEN has no optimizer")

    flattened_names = tuple(
        parameter_name
        for group in groups
        for parameter_name in group[GROUP_PARAMETER_NAMES_KEY]  # type: ignore[union-attr]
    )
    flattened_parameters = tuple(
        parameter
        for group in groups
        for parameter in group["params"]  # type: ignore[union-attr]
    )
    selected_names = tuple(name for name, _ in selected)
    selected_ids = {id(parameter) for _, parameter in selected}
    if (
        len(flattened_names) != len(set(flattened_names))
        or len(flattened_parameters)
        != len({id(parameter) for parameter in flattened_parameters})
        or set(flattened_names) != set(selected_names)
        or {id(parameter) for parameter in flattened_parameters} != selected_ids
    ):
        raise OptimizerContractError(
            "optimizer groups are not one exact cover of the stage allowlist"
        )
    return tuple(groups)


def build_stage_adamw(configuration: StageConfiguration) -> torch.optim.AdamW:
    """Build a pristine AdamW covering each current-stage parameter once."""

    groups = _stage_group_definitions(configuration)
    betas, eps = _common_recipe()
    optimizer = torch.optim.AdamW(
        list(groups),
        lr=float(groups[0]["lr"]),
        betas=betas,
        eps=eps,
        weight_decay=0.0,
        amsgrad=False,
    )
    audit_stage_adamw(optimizer, configuration, state_mode="pristine")
    return optimizer


def _validate_initialized_state(
    optimizer: torch.optim.AdamW,
    expected_parameters: Sequence[nn.Parameter],
) -> None:
    expected_ids = {id(parameter) for parameter in expected_parameters}
    state_by_id = {id(parameter): value for parameter, value in optimizer.state.items()}
    if set(state_by_id) != expected_ids:
        raise OptimizerContractError(
            "initialized AdamW state does not exactly cover the stage parameters"
        )
    step_values: list[int] = []
    for parameter in expected_parameters:
        state = state_by_id[id(parameter)]
        if not isinstance(state, Mapping) or set(state) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise OptimizerContractError("AdamW parameter-state keys drifted")
        step = state["step"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        step_value = (
            float(step.detach().cpu())
            if isinstance(step, torch.Tensor) and step.numel() == 1
            else float("nan")
        )
        if (
            not isinstance(step, torch.Tensor)
            or step.numel() != 1
            or not bool(torch.isfinite(step).all())
            or step_value < 1.0
            or not step_value.is_integer()
        ):
            raise OptimizerContractError("AdamW step state is invalid")
        step_values.append(int(step_value))
        for name, value in (("exp_avg", exp_avg), ("exp_avg_sq", exp_avg_sq)):
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != parameter.shape
                or value.device != parameter.device
                or value.dtype != parameter.dtype
                or not bool(torch.isfinite(value).all())
            ):
                raise OptimizerContractError(f"AdamW {name} state is invalid")
        if bool((exp_avg_sq < 0).any()):
            raise OptimizerContractError("AdamW exp_avg_sq state is negative")
    if len(set(step_values)) != 1:
        raise OptimizerContractError("AdamW parameter step counters differ")


def audit_stage_adamw(
    optimizer: torch.optim.Optimizer,
    configuration: StageConfiguration,
    *,
    state_mode: str,
) -> Mapping[str, object]:
    """Audit group coverage, recipe values, and pristine/initialized state."""

    if type(optimizer) is not torch.optim.AdamW:
        raise OptimizerContractError("optimizer must be exactly torch.optim.AdamW")
    if state_mode not in {"pristine", "initialized"}:
        raise OptimizerContractError(
            "optimizer state_mode must be pristine or initialized"
        )
    expected_groups = _stage_group_definitions(configuration)
    betas, eps = _common_recipe()
    if len(optimizer.param_groups) != len(expected_groups):
        raise OptimizerContractError("AdamW parameter-group count drifted")
    observed_ids: list[int] = []
    observed_names: list[str] = []
    group_audit: list[Mapping[str, object]] = []
    for observed, expected in zip(optimizer.param_groups, expected_groups, strict=True):
        expected_parameters = tuple(expected["params"])  # type: ignore[arg-type]
        expected_names = tuple(
            expected[GROUP_PARAMETER_NAMES_KEY]  # type: ignore[arg-type]
        )
        parameters = tuple(observed.get("params", ()))
        names = tuple(observed.get(GROUP_PARAMETER_NAMES_KEY, ()))
        if (
            observed.get(GROUP_NAME_KEY) != expected[GROUP_NAME_KEY]
            or observed.get(GROUP_STAGE_KEY) != configuration.stage.value
            or names != expected_names
            or len(parameters) != len(expected_parameters)
            or any(
                actual is not wanted
                for actual, wanted in zip(parameters, expected_parameters, strict=True)
            )
        ):
            raise OptimizerContractError("AdamW group identity or ordering drifted")
        if (
            float(observed.get("lr", float("nan"))) != float(expected["lr"])
            or float(observed.get("weight_decay", float("nan")))
            != float(expected["weight_decay"])
            or tuple(observed.get("betas", ())) != betas
            or float(observed.get("eps", float("nan"))) != eps
            or bool(observed.get("amsgrad", True)) is not False
        ):
            raise OptimizerContractError("AdamW group hyperparameters drifted")
        if any(
            not isinstance(parameter, nn.Parameter) or not parameter.requires_grad
            for parameter in parameters
        ):
            raise OptimizerContractError(
                "AdamW contains a frozen, base, or non-Parameter value"
            )
        observed_ids.extend(id(parameter) for parameter in parameters)
        observed_names.extend(names)
        group_audit.append(
            {
                "name": observed[GROUP_NAME_KEY],
                "parameter_count": len(parameters),
                "lr": float(observed["lr"]),
                "weight_decay": float(observed["weight_decay"]),
            }
        )
    selected = _current_stage_entries(configuration)
    expected_ids = {id(parameter) for _, parameter in selected}
    expected_names = {name for name, _ in selected}
    if (
        len(observed_ids) != len(set(observed_ids))
        or len(observed_names) != len(set(observed_names))
        or set(observed_ids) != expected_ids
        or set(observed_names) != expected_names
    ):
        raise OptimizerContractError(
            "AdamW parameters do not exactly equal the current-stage allowlist"
        )
    if state_mode == "pristine":
        if optimizer.state:
            raise OptimizerContractError("pristine AdamW unexpectedly has state")
    else:
        _validate_initialized_state(
            optimizer, tuple(parameter for _, parameter in selected)
        )
    return {
        "stage": configuration.stage.value,
        "optimizer": "AdamW",
        "state_mode": state_mode,
        "parameter_count": len(observed_ids),
        "groups": tuple(group_audit),
        "betas": betas,
        "eps": eps,
        "exact_stage_allowlist_cover": True,
        "frozen_or_base_parameters": 0,
    }


__all__ = [
    "GROUP_NAME_KEY",
    "GROUP_PARAMETER_NAMES_KEY",
    "GROUP_STAGE_KEY",
    "OptimizerContractError",
    "audit_stage_adamw",
    "build_stage_adamw",
]
