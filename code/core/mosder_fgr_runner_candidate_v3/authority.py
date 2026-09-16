"""Validate the release receipt required by the standalone runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Mapping

from .protocol import RunProtocol


SCHEMA_VERSION: Final[str] = "mosder_independent_release_authority_v1"


class AuthorityContractError(RuntimeError):
    """Independent data/release authority is absent or inconsistent."""


def load_release_authority(path: Path, protocol: RunProtocol) -> Mapping[str, Any]:
    """Load a separately issued receipt; the runner cannot self-authorize it."""

    if not path.is_file() or path.is_symlink():
        raise AuthorityContractError("independent release-authority file is absent")
    try:
        value = json.loads(path.read_bytes())
    except Exception as error:
        raise AuthorityContractError("release-authority JSON is invalid") from error
    expected_true = {
        "formal_data_role_authority_complete",
        "formal_training_authorized",
        "formal_validation_authorized",
        "gate2_successor_pass",
        "formal_protocol_authorized",
    }
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol_sha256") != protocol.identity_sha256
        or any(value.get(name) is not True for name in expected_true)
        or value.get("held_roles_opened") is not False
        or value.get("confirmation_a_opened") is not False
        or value.get("final_b_opened") is not False
        or value.get("supersedes_hold_status") != "HOLD_GATE2_V3"
        or value.get("successor_gate_status") != "PASS"
    ):
        raise AuthorityContractError("independent release authority is incomplete")
    for name in (
        "issuer",
        "data_authority_receipt_sha256",
        "gate2_successor_receipt_sha256",
    ):
        field = value.get(name)
        if not isinstance(field, str) or not field:
            raise AuthorityContractError(f"authority field is absent: {name}")
    return value


__all__ = ["AuthorityContractError", "SCHEMA_VERSION", "load_release_authority"]
