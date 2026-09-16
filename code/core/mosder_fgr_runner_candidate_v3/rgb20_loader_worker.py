#!/usr/bin/env python3
"""Decode RGB frames in the dataset-specific Python environment.

This script runs separately from the model process so pixel readers can use
their own dependencies without importing the model backend."""

from __future__ import annotations

import os
from multiprocessing import resource_tracker, shared_memory
from multiprocessing.connection import Client
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping


# LOCAL_PATH: External RGB20 source checkout; align with the bridge configuration.
REPOSITORY = Path("/root/story2_camera_object_motion")
RGB20_IPC_PROTOCOL = "mosder_rgb20_ipc_v1"


def _connect() -> Any:
    socket_path = os.environ.pop("MOSDER_RGB20_IPC_SOCKET")
    authkey = bytes.fromhex(os.environ.pop("MOSDER_RGB20_IPC_AUTHKEY_HEX"))
    return Client(socket_path, family="AF_UNIX", authkey=authkey)


def _error(error: BaseException) -> Mapping[str, Any]:
    return {
        "protocol": RGB20_IPC_PROTOCOL,
        "status": "ERROR",
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }


def _serve(connection: Any, allowed_role: str) -> None:
    import hashlib
    import numpy as np

    runner_root = Path(__file__).resolve().parent
    if str(runner_root) not in sys.path:
        sys.path.insert(0, str(runner_root))
    from strict_v3_rgb20_bridge import (
        RoleScopedStrictV3RGB20Loader,
        bridge_identity,
        dependency_binding,
    )

    loader = RoleScopedStrictV3RGB20Loader(allowed_role)
    formal_loader_module = loader.formal_module
    import av
    from PIL import Image

    connection.send(
        {
            "protocol": RGB20_IPC_PROTOCOL,
            "status": "READY",
            "interpreter_realpath": str(Path(sys.executable).resolve()),
            "worker_realpath": str(Path(__file__).resolve()),
            "formal_loader_realpath": str(
                Path(formal_loader_module.__file__).resolve()
            ),
            "pixel_policy_realpath": str(
                formal_loader_module.FROZEN_PIXEL_POLICY.resolve()
            ),
            "pixel_policy_sha256": (formal_loader_module.FROZEN_PIXEL_POLICY_SHA256),
            "av_module_realpath": str(Path(av.__file__).resolve()),
            "pil_module_realpath": str(Path(Image.__file__).resolve()),
            "allowed_role": allowed_role,
            "strict_operational_manifest": dict(loader.manifest_binding),
            "resolver_dependencies": dict(dependency_binding()),
            "bridge_identity": dict(bridge_identity()),
            "metadata_closure": dict(loader.metadata_closure),
        }
    )
    while True:
        message = connection.recv()
        if message is None:
            return
        if not isinstance(message, Mapping) or set(message) != {
            "protocol",
            "request_id",
            "role",
            "source_dataset",
            "case_id",
            "source_binding",
        }:
            raise RuntimeError("RGB20 worker request contract differs")
        if message.get("protocol") != RGB20_IPC_PROTOCOL:
            raise RuntimeError("RGB20 worker protocol differs")
        loaded = loader.load_case(
            str(message["role"]),
            str(message["source_dataset"]),
            str(message["case_id"]),
            message["source_binding"],
        )
        try:
            frames = tuple(np.asarray(frame) for frame in loaded.frames)
            if len(frames) != 20 or any(
                frame.dtype != np.uint8
                or frame.ndim != 3
                or frame.shape[-1] != 3
                or frame.shape != frames[0].shape
                for frame in frames
            ):
                raise RuntimeError("isolated RGB20 array is invalid")
            shape = (20, *frames[0].shape)
            nbytes = int(sum(frame.nbytes for frame in frames))
            rgb_digest = hashlib.sha256()
            for frame in frames:
                rgb_digest.update(frame)
            request_id = str(message["request_id"])
            decoded = {
                "protocol": RGB20_IPC_PROTOCOL,
                "status": "DECODED",
                "request_id": request_id,
                "shape": list(shape),
                "dtype": "uint8",
                "nbytes": nbytes,
                "rgb20_sha256": rgb_digest.hexdigest(),
                "case_id": loaded.case_id,
                "source_dataset": loaded.source_dataset,
                "split_role": loaded.split_role,
                "interpolation": {
                    key: value
                    for key, value in loaded.interpolation.items()
                    if key != "pair_masks"
                },
            }
            connection.send(decoded)
            buffer_request = connection.recv()
            if not (
                isinstance(buffer_request, Mapping)
                and set(buffer_request)
                == {
                    "protocol",
                    "status",
                    "request_id",
                    "shared_memory_name",
                }
                and buffer_request.get("protocol") == RGB20_IPC_PROTOCOL
                and buffer_request.get("status") == "BUFFER"
                and buffer_request.get("request_id") == request_id
                and isinstance(buffer_request.get("shared_memory_name"), str)
            ):
                raise RuntimeError("RGB20 worker buffer request differs")
            shared = shared_memory.SharedMemory(
                name=str(buffer_request["shared_memory_name"]), create=False
            )
            try:
                if shared.size != nbytes:
                    raise RuntimeError("RGB20 worker shared-memory size differs")
                target = np.ndarray(shape, dtype=np.uint8, buffer=shared.buf)
                for index, frame in enumerate(frames):
                    target[index] = frame
                shared.close()
                resource_tracker.unregister(shared._name, "shared_memory")
                connection.send(
                    {
                        "protocol": RGB20_IPC_PROTOCOL,
                        "status": "OK",
                        "request_id": request_id,
                        "shared_memory_name": str(buffer_request["shared_memory_name"]),
                    }
                )
            except BaseException:
                try:
                    shared.close()
                finally:
                    try:
                        resource_tracker.unregister(shared._name, "shared_memory")
                    except Exception:
                        pass
                raise
        finally:
            loaded.close()


def main() -> int:
    allowed_role = os.environ.pop("MOSDER_RGB20_ALLOWED_ROLE")
    connection = _connect()
    try:
        try:
            _serve(connection, allowed_role)
        except BaseException as error:
            try:
                connection.send(_error(error))
            except Exception:
                pass
            return 1
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
