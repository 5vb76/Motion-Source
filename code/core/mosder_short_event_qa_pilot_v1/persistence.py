"""Atomic per-question records and exact run-identity resume."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Final, Mapping

from .contract import RESULT_SCHEMA_VERSION, canonical_json_bytes, canonical_sha256


RUN_SCHEMA_VERSION: Final[str] = "mosder_short_event_qa_run_v1"


class ShortEventPersistenceError(RuntimeError):
    """A run identity or persisted question record is inconsistent."""


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _load_canonical(path: Path) -> Mapping[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except Exception as error:
        raise ShortEventPersistenceError(f"invalid JSON record: {path}") from error
    if not isinstance(value, Mapping) or raw != canonical_json_bytes(value) + b"\n":
        raise ShortEventPersistenceError(f"noncanonical JSON record: {path}")
    return value


class ResultStore:
    """One immutable file per question; successful rows resume by default."""

    def __init__(self, root: str | Path, run_payload: Mapping[str, Any]) -> None:
        self.root = Path(root).resolve()
        if self.root == Path("/"):
            raise ShortEventPersistenceError("output root cannot be filesystem root")
        payload = dict(run_payload)
        payload["schema_version"] = RUN_SCHEMA_VERSION
        payload["nonauthorizing"] = True
        payload["training_performed"] = False
        payload["labels_loaded_by_inference"] = False
        self.run_sha256 = canonical_sha256(payload)
        payload["run_sha256"] = self.run_sha256
        self.run_payload = payload
        self.records_dir = self.root / "records"
        self.root.mkdir(parents=True, exist_ok=True)
        self.records_dir.mkdir(parents=True, exist_ok=True)
        run_path = self.root / "RUN.json"
        if run_path.exists():
            if _load_canonical(run_path) != payload:
                raise ShortEventPersistenceError(
                    "output directory belongs to a different run identity"
                )
        else:
            _atomic_write(run_path, payload)

    def record_path(self, question_id: str) -> Path:
        if not isinstance(question_id, str) or not question_id:
            raise ShortEventPersistenceError("question_id is invalid")
        digest = hashlib.sha256(question_id.encode("utf-8")).hexdigest()
        return self.records_dir / f"{digest}.json"

    def existing(self, question_id: str) -> Mapping[str, Any] | None:
        path = self.record_path(question_id)
        if not path.exists():
            return None
        value = _load_canonical(path)
        if (
            value.get("schema_version") != RESULT_SCHEMA_VERSION
            or value.get("run_sha256") != self.run_sha256
            or value.get("question_id") != question_id
            or value.get("status") not in {"success", "error"}
        ):
            raise ShortEventPersistenceError("persisted question record differs")
        return value

    def should_skip(self, question_id: str, *, retry_errors: bool) -> bool:
        existing = self.existing(question_id)
        if existing is None:
            return False
        return existing["status"] == "success" or not retry_errors

    def write(self, question_id: str, record: Mapping[str, Any]) -> Path:
        value = dict(record)
        if (
            value.get("schema_version") != RESULT_SCHEMA_VERSION
            or value.get("question_id") != question_id
            or value.get("status") not in {"success", "error"}
        ):
            raise ShortEventPersistenceError("question result contract differs")
        value["run_sha256"] = self.run_sha256
        path = self.record_path(question_id)
        _atomic_write(path, value)
        return path

    def records(self) -> tuple[Mapping[str, Any], ...]:
        values = tuple(
            _load_canonical(path) for path in sorted(self.records_dir.glob("*.json"))
        )
        if any(value.get("run_sha256") != self.run_sha256 for value in values):
            raise ShortEventPersistenceError("record directory mixes run identities")
        return values


__all__ = ["RUN_SCHEMA_VERSION", "ResultStore", "ShortEventPersistenceError"]
