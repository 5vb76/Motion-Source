"""Experiment settings and identity hashes for the F/G/R runner.

Protocol construction preserves the recorded release status. The runner
validates its release receipt separately before accessing training data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from typing import Any, Final, Mapping


SCHEMA_VERSION: Final[str] = "mosder_fgr_run_protocol_candidate_v2"
METHOD_NAME: Final[str] = "MoSDeR"
METHOD_VERSION: Final[str] = "MoSDeR-v1"
HOLD_STATUS: Final[str] = "HOLD_GATE2_V3"
STAGES: Final[tuple[str, ...]] = ("F", "G", "R")
FAMILIES: Final[tuple[str, ...]] = (
    "qwen3_vl_8b",
    "molmo2_o_7b",
    "nvila_lite_8b",
)


class ProtocolContractError(RuntimeError):
    """A run identity or frozen execution choice is invalid."""


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
    """Every choice that may change the training trajectory or selection."""

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
    epochs_f: int = 4
    epochs_g: int = 4
    epochs_r: int = 2
    accumulation_f: int = 8
    accumulation_g: int = 8
    accumulation_r: int = 8
    checkpoint_every_optimizer_steps: int = 100
    validation_schedule: str = "end_of_each_epoch"
    stage_f_selection: str = (
        "higher_adt_macro_recall_then_lower_factor_loss_then_earlier_step"
    )
    stage_g_selection: str = (
        "higher_strict_exact_then_min_state_recall_then_factor_consistency_"
        "then_lower_span_nll_then_earlier_step"
    )
    stage_r_selection: str = (
        "higher_strict_exact_then_min_state_recall_then_factor_consistency_"
        "then_lower_span_nll_then_earlier_step"
    )
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
        if type(self.seed) is not int or self.seed < 0:
            raise ProtocolContractError("seed must be a nonnegative integer")
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
        if self.validation_schedule != "end_of_each_epoch":
            raise ProtocolContractError("validation schedule drifted")
        expected_selection = {
            "stage_f_selection": (
                "higher_adt_macro_recall_then_lower_factor_loss_then_earlier_step"
            ),
            "stage_g_selection": (
                "higher_strict_exact_then_min_state_recall_then_factor_consistency_"
                "then_lower_span_nll_then_earlier_step"
            ),
            "stage_r_selection": (
                "higher_strict_exact_then_min_state_recall_then_factor_consistency_"
                "then_lower_span_nll_then_earlier_step"
            ),
        }
        for name, expected in expected_selection.items():
            if getattr(self, name) != expected:
                raise ProtocolContractError(f"{name} drifted")
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
    payload = protocol.as_dict()
    return {
        "schema_version": "mosder_fgr_protocol_envelope_candidate_v2",
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
    "FAMILIES",
    "HOLD_STATUS",
    "METHOD_NAME",
    "METHOD_VERSION",
    "ProtocolContractError",
    "RunProtocol",
    "SCHEMA_VERSION",
    "STAGES",
    "canonical_json_bytes",
    "canonical_sha256",
    "protocol_envelope",
    "run_protocol_from_mapping",
]
