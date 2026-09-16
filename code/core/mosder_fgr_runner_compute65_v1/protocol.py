"""Settings for the bounded-budget F1/G1/R1 experiment.

F and G use fixed endpoints. R compares the G parent at step zero with the
trained R endpoint after one epoch."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from typing import Any, Final, Mapping


SCHEMA_VERSION: Final[str] = "mosder_fgr_run_protocol_compute65_v1"
ENVELOPE_SCHEMA_VERSION: Final[str] = "mosder_fgr_protocol_envelope_compute65_v1"
METHOD_NAME: Final[str] = "MoSDeR"
METHOD_VERSION: Final[str] = "MoSDeR-v1"
HOLD_STATUS: Final[str] = "HOLD_GATE2_V3"
STAGES: Final[tuple[str, ...]] = ("F", "G", "R")
FAMILIES: Final[tuple[str, ...]] = (
    "qwen3_vl_8b",
    "molmo2_o_7b",
    "nvila_lite_8b",
)
VALIDATION_SCHEDULE: Final[str] = "r_step0_parent_g_then_r_epoch1_only"
FIXED_ENDPOINT_SELECTION: Final[str] = "fixed_epoch1_endpoint_no_formal_validation"
R_SELECTION: Final[str] = (
    "best_of_r_step0_parent_g_and_r_epoch1_by_higher_strict_exact_then_"
    "min_state_recall_then_factor_consistency_then_lower_span_nll_then_"
    "earlier_step"
)


class ProtocolContractError(RuntimeError):
    """A Compute65 protocol identity or execution choice is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ProtocolContractError(f"{name} must be a positive integer")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolContractError(f"{name} must be a lowercase SHA256")
    return value


@dataclass(frozen=True, slots=True)
class RunProtocol:
    """Every choice that may change the Compute65 trajectory or selection."""

    family: str
    seed: int
    train_rows: int
    validation_rows: int
    train_projection_sha256: str
    validation_projection_sha256: str
    consumed_field_seal_sha256: str
    rgb20_bridge_seal_sha256: str
    architecture_seal_sha256: str
    runtime_manifest_sha256: str
    prompt_sha256: str
    tokenizer_manifest_sha256: str
    epochs_f: int = 1
    epochs_g: int = 1
    epochs_r: int = 1
    accumulation_f: int = 8
    accumulation_g: int = 8
    accumulation_r: int = 8
    checkpoint_every_optimizer_steps: int = 100
    validation_schedule: str = VALIDATION_SCHEDULE
    stage_f_selection: str = FIXED_ENDPOINT_SELECTION
    stage_g_selection: str = FIXED_ENDPOINT_SELECTION
    stage_r_selection: str = R_SELECTION
    optimizer: str = "AdamW"
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1.0e-8
    warmup_fraction: float = 0.05
    cosine_floor: float = 0.1
    gradient_clip_max_norm: float = 1.0
    gradient_checkpointing: bool = False
    model_mode: str = "eval"
    formal_training_authorized: bool = False
    hold_status: str = HOLD_STATUS
    confirmation_a_opened: bool = False
    final_b_opened: bool = False

    def validated(self) -> "RunProtocol":
        if self.family not in FAMILIES:
            raise ProtocolContractError(f"unsupported family: {self.family!r}")
        if self.seed != 20260902:
            raise ProtocolContractError("Compute65 requires seed 20260902")
        for name in (
            "train_rows",
            "validation_rows",
            "epochs_f",
            "epochs_g",
            "epochs_r",
            "accumulation_f",
            "accumulation_g",
            "accumulation_r",
            "checkpoint_every_optimizer_steps",
        ):
            _positive_int(getattr(self, name), name)
        if (self.epochs_f, self.epochs_g, self.epochs_r) != (1, 1, 1):
            raise ProtocolContractError("Compute65 requires exactly F1/G1/R1")
        if (self.accumulation_f, self.accumulation_g, self.accumulation_r) != (
            8,
            8,
            8,
        ):
            raise ProtocolContractError(
                "Compute65 requires global accumulation 8 in every stage"
            )
        if self.checkpoint_every_optimizer_steps != 100:
            raise ProtocolContractError(
                "Compute65 requires checkpoint interval 100 optimizer steps"
            )
        for name in (
            "train_projection_sha256",
            "validation_projection_sha256",
            "consumed_field_seal_sha256",
            "rgb20_bridge_seal_sha256",
            "architecture_seal_sha256",
            "runtime_manifest_sha256",
            "prompt_sha256",
            "tokenizer_manifest_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.validation_schedule != VALIDATION_SCHEDULE:
            raise ProtocolContractError("Compute65 Validation schedule drifted")
        if (
            self.stage_f_selection != FIXED_ENDPOINT_SELECTION
            or self.stage_g_selection != FIXED_ENDPOINT_SELECTION
            or self.stage_r_selection != R_SELECTION
        ):
            raise ProtocolContractError("Compute65 checkpoint selection drifted")
        if (
            self.optimizer != "AdamW"
            or tuple(self.betas) != (0.9, 0.999)
            or self.epsilon != 1.0e-8
            or self.warmup_fraction != 0.05
            or self.cosine_floor != 0.1
            or self.gradient_clip_max_norm != 1.0
        ):
            raise ProtocolContractError("optimizer/scheduler recipe drifted")
        numeric = (
            *self.betas,
            self.epsilon,
            self.warmup_fraction,
            self.cosine_floor,
            self.gradient_clip_max_norm,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ProtocolContractError("protocol contains non-finite numerics")
        if (
            self.gradient_checkpointing is not False
            or self.model_mode != "eval"
            or self.formal_training_authorized is not False
            or self.hold_status != HOLD_STATUS
            or self.confirmation_a_opened is not False
            or self.final_b_opened is not False
        ):
            raise ProtocolContractError("negative authority or execution mode drifted")
        return self

    def as_dict(self) -> Mapping[str, Any]:
        self.validated()
        value = asdict(self)
        value["schema_version"] = SCHEMA_VERSION
        value["method_name"] = METHOD_NAME
        value["method_version"] = METHOD_VERSION
        value["stages"] = list(STAGES)
        value["betas"] = list(self.betas)
        return value

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    def epochs(self, stage: str) -> int:
        if stage not in STAGES:
            raise ProtocolContractError(f"unknown stage: {stage!r}")
        return int(getattr(self, f"epochs_{stage.lower()}"))

    def accumulation(self, stage: str) -> int:
        if stage not in STAGES:
            raise ProtocolContractError(f"unknown stage: {stage!r}")
        return int(getattr(self, f"accumulation_{stage.lower()}"))


def protocol_envelope(protocol: RunProtocol) -> Mapping[str, Any]:
    payload = protocol.validated().as_dict()
    return {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "protocol": payload,
        "protocol_sha256": canonical_sha256(payload),
        "authority": {
            "formal_training_authorized": False,
            "hold_status": HOLD_STATUS,
            "confirmation_a_opened": False,
            "final_b_opened": False,
            "held_roles_opened": False,
        },
    }


def run_protocol_from_mapping(value: Mapping[str, Any]) -> RunProtocol:
    if not isinstance(value, Mapping):
        raise ProtocolContractError("protocol root must be a mapping")
    payload = value.get("protocol") if "protocol" in value else value
    if not isinstance(payload, Mapping):
        raise ProtocolContractError("protocol payload is absent")
    expected_extras = {
        "schema_version": SCHEMA_VERSION,
        "method_name": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "stages": list(STAGES),
    }
    for name, wanted in expected_extras.items():
        if payload.get(name) != wanted:
            raise ProtocolContractError(f"protocol {name} differs")
    field_names = {field.name for field in fields(RunProtocol)}
    if set(payload) != field_names | set(expected_extras):
        raise ProtocolContractError("protocol field inventory differs")
    arguments = {name: payload[name] for name in field_names}
    arguments["betas"] = tuple(arguments["betas"])
    protocol = RunProtocol(**arguments).validated()
    if "protocol" in value and value.get("protocol_sha256") != protocol.identity_sha256:
        raise ProtocolContractError("protocol envelope SHA256 differs")
    return protocol


__all__ = [
    "ENVELOPE_SCHEMA_VERSION",
    "FAMILIES",
    "FIXED_ENDPOINT_SELECTION",
    "HOLD_STATUS",
    "METHOD_NAME",
    "METHOD_VERSION",
    "ProtocolContractError",
    "R_SELECTION",
    "RunProtocol",
    "SCHEMA_VERSION",
    "STAGES",
    "VALIDATION_SCHEDULE",
    "canonical_json_bytes",
    "canonical_sha256",
    "protocol_envelope",
    "run_protocol_from_mapping",
]
