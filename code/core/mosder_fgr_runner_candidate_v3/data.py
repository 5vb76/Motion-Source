"""Load 20-frame RGB examples from projected benchmark records."""

from __future__ import annotations

import hashlib
import importlib
import json
from multiprocessing import shared_memory
from multiprocessing.connection import Listener
import os
import secrets
import subprocess
import tempfile
import threading
from pathlib import Path
import sys
from typing import Any, Final, Mapping

import numpy as np
from family_backends_v1 import RGB20Request
from mosder_consumed_field_projection_v2 import projection
from mosder_final_v1.language import canonical_answer

from .engine import TrainingExample
from .protocol import HOLD_STATUS
from .sampler import SampleRef, STATE_ORDER
from .strict_v3_rgb20_bridge import (
    bridge_identity,
    dependency_binding,
    expected_metadata_closure,
    ORIGINAL_ROLE_BINDINGS,
)


SCHEMA_VERSION: Final[str] = "mosder_projected_lazy_dataset_v2"
RGB20_IPC_PROTOCOL: Final[str] = "mosder_rgb20_ipc_v1"
# LOCAL_PATH: Projected train/validation manifests configured in projection.py.
DEFAULT_BUNDLE_ROOT: Final[Path] = projection.DEFAULT_OUTPUT_ROOT
# LOCAL_PATH: Python environment used by the separate RGB20 decoding worker.
FORMAL_RGB20_INTERPRETER: Final[Path] = Path("/root/miniconda3/bin/python")
FORMAL_RGB20_WORKER: Final[Path] = Path(__file__).with_name("rgb20_loader_worker.py")
FORMAL_RGB20_WORKER_STARTUP_TIMEOUT_SECONDS: Final[float] = 60.0
FORMAL_RGB20_WORKER_REQUEST_TIMEOUT_SECONDS: Final[float] = 300.0
# LOCAL_PATH: External pixel policy used to validate decoded frames.
FORMAL_RGB20_POLICY: Final[Path] = Path(
    "/root/autodl-tmp/tst_plugin_formal_preconfirmation_v2/"
    "GATE6_POLICY_FREEZES_V6/"
    "gate6_pixel_v4.m5_da7c4431d873e802f02099d6a168ac47ae55d4c5c91d8a50c4f672a96c4ddd68.frozen.json"
)
FORMAL_RGB20_POLICY_SHA256: Final[str] = (
    "73cdda361c20fca300e071719694e32f1108a369f165f45079d833699be0370c"
)
# LOCAL_PATH: PyAV/Pillow module files in the decoding worker's Python environment.
FORMAL_RGB20_AV_MODULE: Final[Path] = Path(
    "/root/miniconda3/lib/python3.12/site-packages/av/__init__.py"
)
FORMAL_RGB20_PIL_MODULE: Final[Path] = Path(
    "/root/miniconda3/lib/python3.12/site-packages/PIL/Image.py"
)
EXPECTED_BUNDLE_HASHES: Final[Mapping[str, str]] = {
    "train": "0cd6e87c2d1833d426400566b2b5cbb6d68c507f8c8a1e9cbdee809f3d8f538b",
    "validation": "288cd2623250033e3b8fdff636ed8cda249cd899f9dd30ba9f1edc3d7320f422",
}


class DatasetContractError(RuntimeError):
    """Projected membership, supervision, or RGB20 resolution drifted."""


def _validated_runtime_timestamps(
    model_input: Mapping[str, Any],
    interpolation: Mapping[str, Any],
    source_dataset: str,
) -> tuple[int, ...]:
    """Resolve the two sealed timestamp forms without turning them into labels."""

    timestamp_contract = model_input.get("timestamps")
    visual_contract = model_input.get("rgb20")
    if not isinstance(timestamp_contract, Mapping) or not isinstance(
        visual_contract, Mapping
    ):
        raise DatasetContractError("projected timestamp contract is absent")
    if not (
        timestamp_contract.get("consumed_for_deterministic_order_validation") is True
        and timestamp_contract.get("learned_model_feature") is False
    ):
        raise DatasetContractError("projected timestamp role drifted")
    source = timestamp_contract.get("source")
    decoded_values = interpolation.get("frame_timestamps")
    if not isinstance(source, Mapping) or not isinstance(decoded_values, list):
        raise DatasetContractError("runtime timestamp source is invalid")
    if not (
        len(decoded_values) == 20
        and all(type(value) is int for value in decoded_values)
        and all(right > left for left, right in zip(decoded_values, decoded_values[1:]))
    ):
        raise DatasetContractError("decoded timestamps are not strictly ordered RGB20")
    timestamps = tuple(decoded_values)
    if source_dataset in projection.LOCAL_SOURCES:
        if set(source) != {"kind", "values"} or source.get("kind") != "timestamp_ns":
            raise DatasetContractError("local projected timestamp ABI drifted")
        projected_values = source.get("values")
        if not (
            isinstance(projected_values, list)
            and tuple(projected_values) == timestamps
            and all(type(value) is int for value in projected_values)
        ):
            raise DatasetContractError(
                "local decoded timestamps differ from projection"
            )
    elif source_dataset == projection.TACO_SOURCE:
        if set(source) != {
            "kind",
            "source_frame_ordinals",
            "decoded_timestamps_required_at_runtime",
        } or not (
            source.get("kind")
            == "decoded_timestamp_ns_from_bound_container_and_ordinals"
            and source.get("decoded_timestamps_required_at_runtime") is True
        ):
            raise DatasetContractError("TACO projected timestamp ABI drifted")
        ordinals = source.get("source_frame_ordinals")
        if not (
            isinstance(ordinals, list)
            and len(ordinals) == 20
            and all(type(value) is int for value in ordinals)
            and all(right > left for left, right in zip(ordinals, ordinals[1:]))
            and visual_contract.get("source_frame_ordinals") == ordinals
        ):
            raise DatasetContractError("TACO bound frame ordinals drifted")
    else:
        raise DatasetContractError("unsupported projected timestamp source")
    return timestamps


class _SharedLoadedRGB20:
    def __init__(
        self, payload: Mapping[str, Any], owned_shared: shared_memory.SharedMemory
    ) -> None:
        name = payload.get("shared_memory_name")
        shape = payload.get("shape")
        if (
            not isinstance(name, str)
            or not isinstance(shape, list)
            or len(shape) != 4
            or shape[0] != 20
            or shape[-1] != 3
            or payload.get("dtype") != "uint8"
        ):
            raise DatasetContractError("isolated loader payload is invalid")
        if owned_shared.name != name:
            raise DatasetContractError("isolated RGB20 shared-memory name differs")
        self._shared = owned_shared
        array = np.ndarray(
            tuple(map(int, shape)), dtype=np.uint8, buffer=self._shared.buf
        )
        expected_nbytes = int(array.nbytes)
        if (
            payload.get("nbytes") != expected_nbytes
            or self._shared.size != expected_nbytes
            or payload.get("rgb20_sha256") != hashlib.sha256(array).hexdigest()
        ):
            raise DatasetContractError("isolated RGB20 shared-memory binding differs")
        self.frames = tuple(array[index] for index in range(20))
        self.interpolation = dict(payload["interpolation"])
        self.case_id = str(payload["case_id"])
        self.source_dataset = str(payload["source_dataset"])
        self.split_role = str(payload["split_role"])
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self.frames = ()
        self.interpolation = {}
        self._shared.close()
        self._shared.unlink()
        self._closed = True


class IsolatedFormalRGB20Loader:
    """Persistent hash-bound reader process outside the family model environment.

    The frozen pixel policy binds PyAV/Pillow in ``/root/miniconda3``.  Family
    environments intentionally carry different copies of those packages, so a
    normal multiprocessing spawn would still violate the policy.  Launching the
    small worker with the bound interpreter also prevents model remote code from
    pre-registering a conflicting top-level ``scripts`` package.
    """

    def __init__(self, role: str) -> None:
        if role not in {"train", "validation"}:
            raise DatasetContractError("isolated RGB20 role is invalid")
        self.role = role
        if not FORMAL_RGB20_INTERPRETER.is_file():
            raise DatasetContractError("bound formal RGB20 interpreter is absent")
        if not FORMAL_RGB20_WORKER.is_file():
            raise DatasetContractError("formal RGB20 worker is absent")
        authkey = secrets.token_bytes(32)
        self._ipc_directory = tempfile.TemporaryDirectory(prefix="mosder_rgb20_loader_")
        socket_path = Path(self._ipc_directory.name) / "reader.sock"
        self._listener = Listener(str(socket_path), family="AF_UNIX", authkey=authkey)
        environment = os.environ.copy()
        environment.update(
            {
                "MOSDER_RGB20_IPC_SOCKET": str(socket_path),
                "MOSDER_RGB20_IPC_AUTHKEY_HEX": authkey.hex(),
                "MOSDER_RGB20_ALLOWED_ROLE": role,
                "PYTHONNOUSERSITE": "1",
            }
        )
        # A model launcher may inject its own Python packages here.  The worker
        # inserts the repository itself and otherwise uses only its bound env.
        for inherited_name in (
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONSTARTUP",
            "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV",
            "VIRTUAL_ENV",
        ):
            environment.pop(inherited_name, None)
        self._stderr = tempfile.TemporaryFile(mode="w+b")
        self._process = subprocess.Popen(
            [
                str(FORMAL_RGB20_INTERPRETER),
                "-I",
                "-B",
                "-u",
                str(FORMAL_RGB20_WORKER),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr,
            env=environment,
            close_fds=True,
        )
        self._lock = threading.Lock()
        self._closed = False
        try:
            # ``Listener`` exposes no public accept timeout.  Its AF_UNIX
            # implementation owns a normal socket, for which this operation is
            # stable across supported CPython versions.
            self._listener._listener._socket.settimeout(  # type: ignore[attr-defined]
                FORMAL_RGB20_WORKER_STARTUP_TIMEOUT_SECONDS
            )
            self._connection = self._listener.accept()
        except BaseException as error:
            details = self._worker_stderr()
            self.close()
            raise DatasetContractError(
                {
                    "reason": "isolated formal loader failed to connect",
                    "worker_stderr": details,
                }
            ) from error
        finally:
            self._listener.close()
        if not self._connection.poll(FORMAL_RGB20_WORKER_STARTUP_TIMEOUT_SECONDS):
            details = self._worker_stderr()
            self.close()
            raise DatasetContractError(
                {
                    "reason": "isolated formal loader startup timed out",
                    "worker_stderr": details,
                }
            )
        ready = self._connection.recv()
        expected_interpreter = str(FORMAL_RGB20_INTERPRETER.resolve())
        expected_worker = str(FORMAL_RGB20_WORKER.resolve())
        # LOCAL_PATH: Must match the external loader imported by the RGB20 worker.
        expected_loader = str(
            Path(
                "/root/story2_camera_object_motion/scripts/"
                "formal_rgb20_input_loader_v1.py"
            ).resolve()
        )
        expected_ready = {
            "protocol": RGB20_IPC_PROTOCOL,
            "status": "READY",
            "interpreter_realpath": expected_interpreter,
            "worker_realpath": expected_worker,
            "formal_loader_realpath": expected_loader,
            "pixel_policy_realpath": str(FORMAL_RGB20_POLICY.resolve()),
            "pixel_policy_sha256": FORMAL_RGB20_POLICY_SHA256,
            "av_module_realpath": str(FORMAL_RGB20_AV_MODULE.resolve()),
            "pil_module_realpath": str(FORMAL_RGB20_PIL_MODULE.resolve()),
            "allowed_role": role,
            "strict_operational_manifest": {
                "path": str(ORIGINAL_ROLE_BINDINGS[role]["path"]),
                "sha256": ORIGINAL_ROLE_BINDINGS[role]["sha256"],
                "rows": ORIGINAL_ROLE_BINDINGS[role]["rows"],
            },
            "resolver_dependencies": dict(dependency_binding()),
            "bridge_identity": dict(bridge_identity()),
            "metadata_closure": dict(expected_metadata_closure(role)),
        }
        if not (isinstance(ready, Mapping) and dict(ready) == expected_ready):
            details = self._worker_stderr()
            self.close()
            raise DatasetContractError(
                {
                    "reason": "isolated formal loader failed startup",
                    "worker": ready,
                    "worker_stderr": details,
                }
            )

    def _worker_stderr(self) -> str:
        try:
            self._stderr.flush()
            self._stderr.seek(0)
            return self._stderr.read(16 * 1024).decode("utf-8", errors="replace")
        except Exception:
            return ""

    def load_case(
        self,
        role: str,
        source: str,
        case_id: str,
        source_binding: Mapping[str, Any],
    ) -> _SharedLoadedRGB20:
        if role != self.role:
            raise DatasetContractError("isolated RGB20 request crosses its role")
        if self._closed or self._process.poll() is not None:
            raise DatasetContractError("isolated formal loader is not alive")
        with self._lock:
            request_id = secrets.token_hex(16)
            self._connection.send(
                {
                    "protocol": RGB20_IPC_PROTOCOL,
                    "request_id": request_id,
                    "role": role,
                    "source_dataset": source,
                    "case_id": case_id,
                    "source_binding": dict(source_binding),
                }
            )
            if not self._connection.poll(FORMAL_RGB20_WORKER_REQUEST_TIMEOUT_SECONDS):
                raise DatasetContractError("isolated formal loader request timed out")
            value = self._connection.recv()
            if not (
                isinstance(value, Mapping)
                and value.get("protocol") == RGB20_IPC_PROTOCOL
                and value.get("status") == "DECODED"
                and value.get("request_id") == request_id
                and value.get("case_id") == case_id
                and value.get("source_dataset") == source
                and value.get("split_role") == role
            ):
                raise DatasetContractError(
                    {"reason": "isolated formal loader failed", "worker": value}
                )
            shape = value.get("shape")
            if not (
                isinstance(shape, list)
                and len(shape) == 4
                and shape[0] == 20
                and all(type(dimension) is int and dimension > 0 for dimension in shape)
                and shape[-1] == 3
                and value.get("dtype") == "uint8"
                and value.get("nbytes") == int(np.prod(shape, dtype=np.int64))
            ):
                raise DatasetContractError("isolated decoded RGB20 metadata is invalid")
            owned_shared = shared_memory.SharedMemory(
                create=True, size=int(value["nbytes"])
            )
            try:
                self._connection.send(
                    {
                        "protocol": RGB20_IPC_PROTOCOL,
                        "status": "BUFFER",
                        "request_id": request_id,
                        "shared_memory_name": owned_shared.name,
                    }
                )
                if not self._connection.poll(
                    FORMAL_RGB20_WORKER_REQUEST_TIMEOUT_SECONDS
                ):
                    raise DatasetContractError(
                        "isolated formal loader buffer copy timed out"
                    )
                copied = self._connection.recv()
                expected_copied = {
                    "protocol": RGB20_IPC_PROTOCOL,
                    "status": "OK",
                    "request_id": request_id,
                    "shared_memory_name": owned_shared.name,
                }
                if not isinstance(copied, Mapping) or dict(copied) != expected_copied:
                    raise DatasetContractError(
                        {"reason": "isolated RGB20 copy failed", "worker": copied}
                    )
                payload = {**dict(value), "shared_memory_name": owned_shared.name}
                loaded = _SharedLoadedRGB20(payload, owned_shared)
            except Exception:
                owned_shared.close()
                try:
                    owned_shared.unlink()
                except FileNotFoundError:
                    pass
                raise
        if (
            loaded.case_id != case_id
            or loaded.source_dataset != source
            or loaded.split_role != role
        ):
            loaded.close()
            raise DatasetContractError(
                "isolated formal loader response identity differs"
            )
        return loaded

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        connection = getattr(self, "_connection", None)
        if connection is not None and self._process.poll() is None:
            try:
                connection.send(None)
            except Exception:
                pass
        try:
            self._process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5.0)
        if connection is not None:
            connection.close()
        listener = getattr(self, "_listener", None)
        if listener is not None:
            try:
                listener.close()
            except Exception:
                pass
        self._stderr.close()
        ipc_directory = getattr(self, "_ipc_directory", None)
        if ipc_directory is not None:
            ipc_directory.cleanup()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ProjectedMoSDeRDataset:
    """Keep only byte offsets/identities; decode pixels one selected row at a time."""

    def __init__(
        self,
        role: str,
        *,
        bundle_root: Path = DEFAULT_BUNDLE_ROOT,
        allow_validation: bool = False,
        replay_projection_sources: bool = False,
        isolated_loader: bool = False,
    ) -> None:
        if role not in {"train", "validation"}:
            raise DatasetContractError("dataset role must be train or validation")
        if role == "validation" and allow_validation is not True:
            raise DatasetContractError(
                "Validation requires an explicit runner-level authorization flag"
            )
        self.role = role
        self.isolated_loader = isolated_loader
        self.bundle_root = bundle_root.resolve()
        try:
            verification = projection.verify_bundle(
                self.bundle_root, replay_sources=replay_projection_sources
            )
        except Exception as error:
            raise DatasetContractError(
                "consumed-field bundle verification failed"
            ) from error
        receipt = verification.get("receipt")
        if not isinstance(receipt, Mapping):
            # The verifier's public return is deliberately compact in some
            # versions; independently bind the canonical receipt below.
            receipt_path = self.bundle_root / projection.RECEIPT_NAME
            try:
                receipt = json.loads(receipt_path.read_bytes())
            except Exception as error:
                raise DatasetContractError("projection receipt is invalid") from error
        authority = receipt.get("authority")
        if not isinstance(authority, Mapping) or not (
            authority.get("formal_training_authorized") is False
            and authority.get("formal_validation_authorized") is False
            and authority.get("underlying_gate2_status") == HOLD_STATUS
        ):
            raise DatasetContractError("projection negative authority drifted")
        output = receipt.get("outputs", {}).get(role)
        if not isinstance(output, Mapping):
            raise DatasetContractError("projection receipt lacks role output")
        self.path = self.bundle_root / projection.OUTPUT_MANIFEST_NAMES[role]
        expected_sha = EXPECTED_BUNDLE_HASHES[role]
        if (
            output.get("filename") != self.path.name
            or output.get("sha256") != expected_sha
            or sha256_file(self.path) != expected_sha
        ):
            raise DatasetContractError("projected manifest SHA256 differs")
        self.manifest_sha256 = expected_sha
        self._offsets: list[int] = []
        self._refs: list[SampleRef] = []
        self._scan_membership()
        if len(self._refs) != int(output.get("rows", -1)):
            raise DatasetContractError("projected manifest row count differs")
        self._formal_loader: Any | None = None

    @property
    def refs(self) -> tuple[SampleRef, ...]:
        return tuple(self._refs)

    def _scan_membership(self) -> None:
        seen: set[str] = set()
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                try:
                    row = json.loads(line)
                except Exception as error:
                    raise DatasetContractError(
                        "projected JSONL row is invalid"
                    ) from error
                if not isinstance(row, Mapping):
                    raise DatasetContractError("projected JSONL row is not an object")
                identity = row.get("audit_identity")
                supervision = row.get("supervision")
                authority = row.get("authority")
                if not all(
                    isinstance(value, Mapping)
                    for value in (identity, supervision, authority)
                ):
                    raise DatasetContractError("projected row contract is incomplete")
                case_id = identity.get("case_id")
                state = supervision.get("state")
                source = identity.get("source_dataset")
                if (
                    not isinstance(case_id, str)
                    or not case_id
                    or case_id in seen
                    or state not in STATE_ORDER
                    or not isinstance(source, str)
                    or not source
                    or identity.get("split_role") != self.role
                    or identity.get("state") != state
                    or supervision.get("canonical_answer")
                    != canonical_answer(str(state))
                    or supervision.get("dense_numeric_trajectory_target_consumed")
                    is not False
                    or authority.get("formal_training_authorized") is not False
                    or authority.get("underlying_gate2_status") != HOLD_STATUS
                ):
                    raise DatasetContractError(
                        "projected row identity/supervision drifted"
                    )
                seen.add(case_id)
                self._offsets.append(offset)
                self._refs.append(SampleRef(case_id, str(state), source).validated())

    def __len__(self) -> int:
        return len(self._refs)

    def _row(self, index: int) -> Mapping[str, Any]:
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        with self.path.open("rb") as handle:
            handle.seek(self._offsets[index])
            line = handle.readline()
        try:
            row = json.loads(line)
        except Exception as error:
            raise DatasetContractError(
                "selected projected row cannot be parsed"
            ) from error
        if row.get("audit_identity", {}).get("case_id") != self._refs[index].case_id:
            raise DatasetContractError("selected row offset/identity drifted")
        return row

    def _loader(self) -> Any:
        if self._formal_loader is None:
            if self.isolated_loader:
                self._formal_loader = IsolatedFormalRGB20Loader(self.role)
                return self._formal_loader
            # LOCAL_PATH: External source checkout used for in-process RGB loading.
            repository = Path("/root/story2_camera_object_motion")
            scripts = Path("/root/story2_camera_object_motion/scripts")
            if str(repository) not in sys.path:
                sys.path.insert(0, str(repository))
            # Some family remote-code trees create their own top-level
            # ``scripts`` namespace before the formal loader is imported.
            # Extend that namespace explicitly instead of deleting or
            # replacing it, preserving both dependency graphs.
            namespace = sys.modules.get("scripts")
            if namespace is not None:
                namespace_path = getattr(namespace, "__path__", None)
                if namespace_path is None:
                    raise DatasetContractError(
                        "a non-package module shadows the repository scripts namespace"
                    )
                if str(scripts) not in namespace_path:
                    namespace_path.insert(0, str(scripts))
            try:
                module = importlib.import_module("scripts.formal_rgb20_input_loader_v1")
                FormalRGB20InputLoader = module.FormalRGB20InputLoader
            except Exception as error:
                raise DatasetContractError(
                    "formal RGB20 loader import failed"
                ) from error
            self._formal_loader = FormalRGB20InputLoader()
        return self._formal_loader

    def close(self) -> None:
        if self._formal_loader is not None and hasattr(self._formal_loader, "close"):
            self._formal_loader.close()
        self._formal_loader = None

    def __getitem__(self, index: int) -> TrainingExample:
        row = self._row(index)
        ref = self._refs[index]
        loader_request = row.get("loader_request")
        model_input = row.get("model_input")
        source_binding = row.get("source_binding")
        if not all(
            isinstance(value, Mapping)
            for value in (loader_request, model_input, source_binding)
        ):
            raise DatasetContractError("selected projected loader request is absent")
        if loader_request != {
            "case_id": ref.case_id,
            "resolver": "formal_rgb20_input_loader_v1",
            "source_dataset": ref.source_dataset,
            "split_role": self.role,
        }:
            raise DatasetContractError("projected loader request drifted")
        loader = self._loader()
        loaded = (
            loader.load_case(
                self.role,
                ref.source_dataset,
                ref.case_id,
                source_binding,
            )
            if self.isolated_loader
            else loader.load_case(self.role, ref.source_dataset, ref.case_id)
        )
        try:
            interpolation = loaded.interpolation
            timestamps = _validated_runtime_timestamps(
                model_input, interpolation, ref.source_dataset
            )
            boxes = interpolation["interpolated_boxes_xyxy_half_open_float"]
            if (
                len(loaded.frames) != 20
                or len(timestamps) != 20
                or len(boxes) != 20
                or model_input.get("target_query") != projection.QUERY_TEXT
            ):
                raise DatasetContractError("resolved RGB20 differs from projection")
            request = RGB20Request(
                frames=tuple(loaded.frames),
                timestamps_ns=timestamps,
                prompt=projection.QUERY_TEXT,
                oracle_boxes_xyxy=[list(map(float, box)) for box in boxes],
                request_id=ref.case_id,
            )
            request.validated()
            return TrainingExample(
                case_id=ref.case_id,
                state=ref.state,
                source_dataset=ref.source_dataset,
                request=request,
                release=loaded.close,
            )
        except Exception:
            loaded.close()
            raise


__all__ = [
    "DEFAULT_BUNDLE_ROOT",
    "DatasetContractError",
    "EXPECTED_BUNDLE_HASHES",
    "FORMAL_RGB20_INTERPRETER",
    "FORMAL_RGB20_POLICY",
    "FORMAL_RGB20_POLICY_SHA256",
    "FORMAL_RGB20_WORKER",
    "IsolatedFormalRGB20Loader",
    "ProjectedMoSDeRDataset",
    "RGB20_IPC_PROTOCOL",
    "SCHEMA_VERSION",
    "sha256_file",
]
