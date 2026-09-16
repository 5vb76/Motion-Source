#!/usr/bin/env python3
"""Record a protocol together with its runtime versions and file hashes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import torch

from family_backends_v1 import discover_local_family
from .checkpoint import configure_reproducible_numeric_environment
from .data import EXPECTED_BUNDLE_HASHES, sha256_file
from .manifest import code_manifest, runtime_environment_manifest
from .protocol import (
    RunProtocol,
    canonical_json_bytes,
    canonical_sha256,
    protocol_envelope,
)
from .runner import (
    ARCHITECTURE_SEAL,
    BRIDGE_SEAL_RECEIPT,
    PROJECTION_RECEIPT,
    RUNNER_CODE,
    _verify_protocol_static_bindings,
)


class ProtocolSealError(RuntimeError):
    """A protocol cannot be sealed onto the exact intended runtime."""


def build_protocol(
    *,
    family: str,
    seed: int,
    epochs_f: int,
    epochs_g: int,
    epochs_r: int,
    accumulation: int,
    checkpoint_every: int,
) -> RunProtocol:
    # Must precede the first CUDA availability/device query.
    configure_reproducible_numeric_environment()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ProtocolSealError(
            "protocol sealing requires exactly one visible target GPU"
        )
    discovery = discover_local_family(family)
    environment = runtime_environment_manifest()
    runtime_sha = canonical_sha256(
        {
            "runner_code": code_manifest(RUNNER_CODE)["manifest_sha256"],
            "environment": environment["environment_sha256"],
            "static_family": discovery["artifact_control_sha256"],
        }
    )
    try:
        receipt = json.loads(PROJECTION_RECEIPT.read_bytes())
        prompt_sha = str(receipt["consumed_field_contract"]["query_sha256"])
    except Exception as error:
        raise ProtocolSealError("consumed-field receipt is invalid") from error
    protocol = RunProtocol(
        family=family,
        seed=seed,
        train_rows=7959,
        validation_rows=1991,
        train_projection_sha256=EXPECTED_BUNDLE_HASHES["train"],
        validation_projection_sha256=EXPECTED_BUNDLE_HASHES["validation"],
        consumed_field_seal_sha256=sha256_file(PROJECTION_RECEIPT),
        rgb20_bridge_seal_sha256=sha256_file(BRIDGE_SEAL_RECEIPT),
        architecture_seal_sha256=sha256_file(ARCHITECTURE_SEAL),
        runtime_manifest_sha256=runtime_sha,
        prompt_sha256=prompt_sha,
        tokenizer_manifest_sha256=str(discovery["artifact_control_sha256"]),
        epochs_f=epochs_f,
        epochs_g=epochs_g,
        epochs_r=epochs_r,
        accumulation_f=accumulation,
        accumulation_g=accumulation,
        accumulation_r=accumulation,
        checkpoint_every_optimizer_steps=checkpoint_every,
    ).validated()
    _verify_protocol_static_bindings(protocol, discovery)
    return protocol


def create_once(path: Path, payload: bytes) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = payload + b"\n"
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != encoded:
            raise ProtocolSealError("existing protocol envelope differs")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs-f", type=int, default=4)
    parser.add_argument("--epochs-g", type=int, default=4)
    parser.add_argument("--epochs-r", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    protocol = build_protocol(
        family=arguments.family,
        seed=arguments.seed,
        epochs_f=arguments.epochs_f,
        epochs_g=arguments.epochs_g,
        epochs_r=arguments.epochs_r,
        accumulation=arguments.accumulation,
        checkpoint_every=arguments.checkpoint_every,
    )
    envelope = protocol_envelope(protocol)
    create_once(arguments.output, canonical_json_bytes(envelope))
    print(
        json.dumps(
            {
                "status": "PASS_PROTOCOL_SEALED_NONAUTHORIZING",
                "family": protocol.family,
                "protocol_sha256": protocol.identity_sha256,
                "output": str(arguments.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
