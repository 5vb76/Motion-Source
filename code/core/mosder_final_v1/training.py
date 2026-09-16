"""Trainable parameter groups and optimizer settings for the F/G/R stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import torch
from torch import nn

from .bridge import DualSourceExplicitResidual
from .contract import FORMAL_TRAINING_AUTHORIZED, HOLD_STATUS
from .core import MoSDeRMotionSourceCore
from .routing import FactorSpanDecision, SourceConditionedTriLoRALinear


class TrainingContractError(RuntimeError):
    """A stage owner, optimizer, or release invariant failed."""


class TrainingStage(str, Enum):
    F = "F"  # source-separated factor learning
    G = "G"  # shared native-language alignment
    R = "R"  # explicit raw-plus-refined language residual
    FROZEN = "FROZEN"


def normalize_stage(stage: TrainingStage | str) -> TrainingStage:
    if isinstance(stage, TrainingStage):
        return stage
    if not isinstance(stage, str):
        raise TypeError("stage must be F, G, R, or FROZEN")
    normalized = stage.strip().upper().replace("STAGE-", "")
    if normalized == "P":
        raise TrainingContractError("MoSDeR has no trajectory Stage-P")
    try:
        return TrainingStage(normalized)
    except ValueError as error:
        raise TrainingContractError("stage must be F, G, R, or FROZEN") from error


@dataclass(frozen=True, slots=True)
class NamedOwnerPartitions:
    camera: tuple[tuple[str, nn.Parameter], ...]
    object: tuple[tuple[str, nn.Parameter], ...]
    shared: tuple[tuple[str, nn.Parameter], ...]
    residual: tuple[tuple[str, nn.Parameter], ...]

    @property
    def all(self) -> tuple[tuple[str, nn.Parameter], ...]:
        return self.camera + self.object + self.shared + self.residual

    def names(self) -> Mapping[str, tuple[str, ...]]:
        return {
            "camera": tuple(name for name, _ in self.camera),
            "object": tuple(name for name, _ in self.object),
            "shared": tuple(name for name, _ in self.shared),
            "residual": tuple(name for name, _ in self.residual),
        }

    def validate(self) -> None:
        values = self.all
        names = tuple(name for name, _ in values)
        identities = tuple(id(parameter) for _, parameter in values)
        if (
            any(not owner for owner in self.names().values())
            or len(names) != len(set(names))
            or len(identities) != len(set(identities))
        ):
            raise TrainingContractError(
                "MoSDeR parameter owners are empty, overlapping, or duplicated"
            )


@dataclass(frozen=True, slots=True)
class StageConfiguration:
    stage: TrainingStage
    owners: NamedOwnerPartitions
    trainable_names: tuple[str, ...]


def named_owner_partitions(
    *,
    source_core: MoSDeRMotionSourceCore,
    wrappers: Mapping[str, SourceConditionedTriLoRALinear],
    decision: FactorSpanDecision,
    residual: DualSourceExplicitResidual,
) -> NamedOwnerPartitions:
    if not isinstance(source_core, MoSDeRMotionSourceCore):
        raise TypeError("source_core must be MoSDeRMotionSourceCore")
    if (
        not isinstance(wrappers, Mapping)
        or not wrappers
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(wrapper, SourceConditionedTriLoRALinear)
            for name, wrapper in wrappers.items()
        )
        or len({id(wrapper) for wrapper in wrappers.values()}) != len(wrappers)
    ):
        raise TrainingContractError("TriLoRA wrapper registry is invalid")
    if not isinstance(decision, FactorSpanDecision):
        raise TypeError("decision must be FactorSpanDecision")
    if not isinstance(residual, DualSourceExplicitResidual):
        raise TypeError("residual must be DualSourceExplicitResidual")

    source = source_core.owner_named_parameters()
    camera = list(
        (f"source_core.{name}", parameter) for name, parameter in source["camera"]
    )
    obj = list(
        (f"source_core.{name}", parameter) for name, parameter in source["object"]
    )
    shared: list[tuple[str, nn.Parameter]] = []
    for path, wrapper in wrappers.items():
        local = wrapper.owner_named_parameters()
        camera.extend(
            (f"wrappers.{path}.{name}", parameter)
            for name, parameter in local["camera"]
        )
        obj.extend(
            (f"wrappers.{path}.{name}", parameter)
            for name, parameter in local["object"]
        )
        shared.extend(
            (f"wrappers.{path}.{name}", parameter)
            for name, parameter in local["shared"]
        )
    camera.append(("decision.b_c", decision.b_c))
    obj.append(("decision.b_o", decision.b_o))
    residual_values = tuple(residual.named_parameters(prefix="explicit_residual"))
    output = NamedOwnerPartitions(
        camera=tuple(camera),
        object=tuple(obj),
        shared=tuple(shared),
        residual=residual_values,
    )
    output.validate()
    return output


def configure_training_stage(
    stage: TrainingStage | str,
    *,
    installed_model: nn.Module,
    source_core: MoSDeRMotionSourceCore,
    wrappers: Mapping[str, SourceConditionedTriLoRALinear],
    decision: FactorSpanDecision,
    residual: DualSourceExplicitResidual,
) -> StageConfiguration:
    """Apply the exact F/G/R trainability partition.

    F: Camera/Object source adapters, same-owner TriLoRA, and b_c/b_o.
    G: Shared TriLoRA only.
    R: The eight explicit-residual tensors only.
    """

    if not isinstance(installed_model, nn.Module):
        raise TypeError("installed_model must be the complete live model")
    normalized = normalize_stage(stage)
    owners = named_owner_partitions(
        source_core=source_core,
        wrappers=wrappers,
        decision=decision,
        residual=residual,
    )
    live_module_ids = {id(module) for module in installed_model.modules()}
    required_module_ids = {
        id(source_core),
        id(decision),
        id(residual),
        *(id(wrapper) for wrapper in wrappers.values()),
    }
    if not required_module_ids.issubset(live_module_ids):
        raise TrainingContractError(
            "stage configuration received modules outside the installed graph"
        )
    live_parameter_ids = {id(parameter) for parameter in installed_model.parameters()}
    owner_parameter_ids = {id(parameter) for _, parameter in owners.all}
    if not owner_parameter_ids.issubset(live_parameter_ids):
        raise TrainingContractError(
            "one or more owner parameters are not registered in the live model"
        )

    # Freeze the complete VLM first.  Re-enable only the selected method owner
    # below; callers cannot accidentally leave an upstream backbone parameter
    # live by configuring the plugin in isolation.
    for parameter in installed_model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    source_core.configure_stage(normalized.value)
    wrapper_owners: tuple[str, ...]
    if normalized is TrainingStage.F:
        wrapper_owners = ("camera", "object")
    elif normalized is TrainingStage.G:
        wrapper_owners = ("shared",)
    else:
        wrapper_owners = ()
    for wrapper in wrappers.values():
        wrapper.set_trainable_owners(wrapper_owners)
    for parameter in decision.parameters():
        parameter.requires_grad_(normalized is TrainingStage.F)
        parameter.grad = None
    for parameter in residual.parameters():
        parameter.requires_grad_(normalized is TrainingStage.R)
        parameter.grad = None

    allowed_owner_names: tuple[str, ...]
    if normalized is TrainingStage.F:
        allowed_owner_names = owners.names()["camera"] + owners.names()["object"]
    elif normalized is TrainingStage.G:
        allowed_owner_names = owners.names()["shared"]
    elif normalized is TrainingStage.R:
        allowed_owner_names = owners.names()["residual"]
    else:
        allowed_owner_names = ()
    observed = tuple(name for name, parameter in owners.all if parameter.requires_grad)
    if set(observed) != set(allowed_owner_names):
        raise TrainingContractError(
            "observed trainable tensors differ from the stage owner allowlist"
        )
    if any(
        parameter.requires_grad
        for wrapper in wrappers.values()
        for parameter in wrapper.base.parameters()
    ):
        raise TrainingContractError("a family-native W0 parameter became trainable")
    observed_live_ids = {
        id(parameter)
        for parameter in installed_model.parameters()
        if parameter.requires_grad
    }
    expected_live_ids = {
        id(parameter) for name, parameter in owners.all if name in allowed_owner_names
    }
    if observed_live_ids != expected_live_ids:
        raise TrainingContractError(
            "live-model trainables differ from the selected method owner"
        )
    return StageConfiguration(normalized, owners, observed)


# Conservative pre-freeze defaults derived from the completed development
# workers.  These are not permission to run formal data while Gate 2 is held.
PREFREEZE_OPTIMIZER_RECIPE: Mapping[str, object] = {
    "optimizer": "AdamW",
    "betas": [0.9, 0.999],
    "eps": 1.0e-8,
    "gradient_clip_max_norm": 1.0,
    "warmup_fraction": 0.05,
    "scheduler": "cosine_to_0.1x",
    "F": {
        "source_lr": 1.0e-4,
        "trilora_lr": 2.0e-4,
        "bias_lr": 1.0e-2,
        "matrix_weight_decay": 0.01,
        "norm_gate_bias_weight_decay": 0.0,
    },
    "G": {"shared_trilora_lr": 1.0e-4, "weight_decay": 0.01},
    "R": {"explicit_residual_lr": 5.0e-4, "weight_decay": 0.01},
}


def formal_training_preflight_status() -> Mapping[str, object]:
    """Return the current release boundary without touching any formal data."""

    return {
        "architecture_frozen": True,
        "architecture_selection_hash_seal_complete": True,
        "standalone_reference_graph_ready": True,
        "stage_owner_contract_ready": True,
        "three_family_semantic_abi_frozen": True,
        "three_family_real_runner_smoke_complete": True,
        "three_family_real_smoke_is_accuracy": False,
        "formal_runner_complete": False,
        "formal_checkpoint_resume_smoke_complete": False,
        "formal_protocol_and_seed_seal_complete": False,
        "formal_data_role_authority_complete": False,
        "formal_training_authorized": FORMAL_TRAINING_AUTHORIZED,
        "hold_status": HOLD_STATUS,
        "confirmation_a_opened": False,
        "final_b_opened": False,
        "held_roles_opened": False,
        "may_launch_formal_8k_2k": False,
        "next_safe_action": (
            "complete_formal_runner_checkpoint_resume_seed_and_data_role_"
            "seals_without_opening_held_roles"
        ),
    }


def assert_formal_training_authorized() -> None:
    if not FORMAL_TRAINING_AUTHORIZED:
        raise TrainingContractError(
            f"formal training is blocked by {HOLD_STATUS}; held roles stay closed"
        )


__all__ = [
    "NamedOwnerPartitions",
    "PREFREEZE_OPTIMIZER_RECIPE",
    "StageConfiguration",
    "TrainingContractError",
    "TrainingStage",
    "assert_formal_training_authorized",
    "configure_training_stage",
    "formal_training_preflight_status",
    "named_owner_partitions",
    "normalize_stage",
]
