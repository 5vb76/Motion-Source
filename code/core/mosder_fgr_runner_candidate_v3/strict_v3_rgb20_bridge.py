"""Role-scoped Strict-V3 V2 RGB20 resolver for MoSDeR.

The resolver validates all records for a dataset role, then loads its
projected membership. Dense trajectory supervision is used only by the
upstream row validator and is not returned to the MoSDeR model process.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Final, Mapping

import numpy as np


# LOCAL_PATH: External source checkout containing the RGB20 loader and materializer.
REPOSITORY = Path("/root/story2_camera_object_motion")
MATERIALIZER_ROOT = (
    REPOSITORY / "experiments/tst_plugin_o18_strict_v3_feature_materializer_v1"
)
MATERIALIZER_CONTRACT = MATERIALIZER_ROOT / "materializer_contract_v1.py"
MATERIALIZER_CONTRACT_SHA256 = (
    "4470ed04bac87f5a665596402de450f524e4efd1df64d47b636af2438810860e"
)
MATERIALIZER_BACKENDS = MATERIALIZER_ROOT / "family_backends_v1.py"
MATERIALIZER_BACKENDS_SHA256 = (
    "1738d7309ef2e37c58f0befdbeb4dc98960e7166cb3b5298fecede30107a2822"
)
# LOCAL_PATH: Prepared TACO RGB assets for each data role on the target machine.
TACO_ROLE_ROOTS = {
    role: Path(
        "/root/autodl-tmp/tst_plugin_taco_allocentric_object_only_v1/"
        f"selected_marker_removed_rgb_v4/{role}"
    )
    for role in ("train", "validation")
}
# LOCAL_PATH: Input train/validation manifests consumed by the RGB resolver.
ORIGINAL_ROLE_BINDINGS: Final[Mapping[str, Mapping[str, Any]]] = {
    "train": {
        "path": Path(
            "/root/autodl-tmp/tst_plugin_formal_preconfirmation_v2/"
            "TST_PLUGIN_O18_STRICT_V3_OPERATIONAL_DATA_CONTRACT_V2_NONAUTHORIZING/"
            "train_o18_strict_v3_operational_v2.jsonl"
        ),
        "sha256": "2595242cec6b2e652180d5c9b0107b2a2900fb78b919cc1f5808a0b098c075d6",
        "rows": 8000,
    },
    "validation": {
        "path": Path(
            "/root/autodl-tmp/tst_plugin_formal_preconfirmation_v2/"
            "TST_PLUGIN_O18_STRICT_V3_OPERATIONAL_DATA_CONTRACT_V2_NONAUTHORIZING/"
            "validation_o18_strict_v3_operational_v2.jsonl"
        ),
        "sha256": "f428b2d38c38804b7eec6bc59ed8e2f0587bd8a2a8c5368ec4fab7c96e733ee6",
        "rows": 2000,
    },
}

BRIDGE_SCHEMA_VERSION: Final[str] = "mosder_strict_v3_rgb20_bridge_candidate_v2"
BRIDGE_RESOLVER_NAME: Final[str] = (
    "strict_v3_additive_role_scoped_sanitized_membership_cache_v2"
)
HOLD_STATUS: Final[str] = "HOLD_GATE2_V3"
PROJECTION_ROW_SCHEMA: Final[str] = (
    "mosder_v1_sanitized_consumed_field_projection_case_v2"
)
# LOCAL_PATH: Projected manifests; keep aligned with projection.DEFAULT_OUTPUT_ROOT.
PROJECTION_ROOT: Final[Path] = Path(
    "/root/autodl-tmp/tst_native_adapter_o18_sandbox_v1/"
    "MOSDER_V1_SANITIZED_CONSUMED_FIELD_PROJECTION_V2_NONAUTHORIZING"
)
PROJECTION_ROLE_BINDINGS: Final[Mapping[str, Mapping[str, Any]]] = {
    "train": {
        "path": PROJECTION_ROOT / "train_mosder_v1_sanitized_consumed_fields_v2.jsonl",
        "sha256": "0cd6e87c2d1833d426400566b2b5cbb6d68c507f8c8a1e9cbdee809f3d8f538b",
        "rows": 7959,
        "unique_visual_windows": 4778,
        "by_source": {
            "ADT-LiteOffice": 6321,
            "HOT3D": 346,
            "TACO-V1-allocentric": 1292,
        },
    },
    "validation": {
        "path": PROJECTION_ROOT
        / "validation_mosder_v1_sanitized_consumed_fields_v2.jsonl",
        "sha256": "288cd2623250033e3b8fdff636ed8cda249cd899f9dd30ba9f1edc3d7320f422",
        "rows": 1991,
        "unique_visual_windows": 1389,
        "by_source": {
            "ADT-LiteOffice": 1550,
            "HOT3D": 66,
            "TACO-V1-allocentric": 375,
        },
    },
}
# LOCAL_PATH: Sanitized role manifests required to validate projected membership.
SANITIZED_ROLE_BINDINGS: Final[Mapping[str, Mapping[str, Any]]] = {
    "train": {
        "path": Path(
            "/root/autodl-tmp/tst_na_v3_trainval_ready_candidate_v1/"
            "TRAIN_SANITIZED_V1/train_o18_strict_v3_quality_retained_v1.jsonl"
        ),
        "sha256": "69f04339f384709fe1871f0f760f32083df27cd0726f51d98ca2450c32416ba9",
        "rows": 7959,
    },
    "validation": {
        "path": Path(
            "/root/autodl-tmp/tst_na_v3_trainval_ready_candidate_v1/"
            "VALIDATION_SANITIZED_V1/validation_o18_strict_v3_quality_retained_v1.jsonl"
        ),
        "sha256": "681a96b3a4e520ed08c78318d59a033f6cfd8b319c582bd1ca0812cc2a9338df",
        "rows": 1991,
    },
}
# Sealed from the deterministic, role-complete metadata index.  These are not
# pixel hashes: startup resolves and reverse-validates all provenance/anchor
# metadata but decodes pixels lazily.
EXPECTED_ROLE_CACHE_INDEX_SHA256: Final[Mapping[str, str]] = {
    "train": "a4a8f3d3574f5dcb5b2090898d6fe8c0333b8dd327134da6d2c06bba7f8fdc75",
    "validation": "9d14cb0db7c5e04d929e5fdd73344c6d3339c428a92ddd26dac841e8475d643d",
}


class StrictV3RGB20BridgeError(RuntimeError):
    """A role, manifest, source binding, evidence, or decode invariant failed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _canonical_json_line(value: Any) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_materializer_modules() -> tuple[ModuleType, ModuleType]:
    for path, expected in (
        (MATERIALIZER_CONTRACT, MATERIALIZER_CONTRACT_SHA256),
        (MATERIALIZER_BACKENDS, MATERIALIZER_BACKENDS_SHA256),
    ):
        if not path.is_file() or _sha256_file(path) != expected:
            raise StrictV3RGB20BridgeError(
                f"bound Strict-V3 materializer module drifted: {path}"
            )
    if str(MATERIALIZER_ROOT) not in sys.path:
        sys.path.insert(0, str(MATERIALIZER_ROOT))
    contract = importlib.import_module("materializer_contract_v1")
    backends = importlib.import_module("family_backends_v1")
    if (
        Path(str(contract.__file__)).resolve() != MATERIALIZER_CONTRACT.resolve()
        or Path(str(backends.__file__)).resolve() != MATERIALIZER_BACKENDS.resolve()
    ):
        raise StrictV3RGB20BridgeError(
            "Strict-V3 materializer module resolved from an unexpected path"
        )
    return contract, backends


def dependency_binding() -> Mapping[str, Any]:
    return {
        "materializer_contract": {
            "path": str(MATERIALIZER_CONTRACT),
            "sha256": MATERIALIZER_CONTRACT_SHA256,
        },
        "materializer_backends": {
            "path": str(MATERIALIZER_BACKENDS),
            "sha256": MATERIALIZER_BACKENDS_SHA256,
        },
    }


def bridge_identity() -> Mapping[str, Any]:
    """Return resolver identity, input contracts, and release status."""

    return {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "effective_resolver": BRIDGE_RESOLVER_NAME,
        "projected_loader_request_resolver_preserved": ("formal_rgb20_input_loader_v1"),
        "resolution": {
            "startup": (
                "validate_full_original_then_filter_sanitized_role_metadata_once"
            ),
            "request_lookup": "O(1)_identity_dictionary",
            "pixel_decode": "lazy_per_requested_case",
            "pixel_cache_at_startup": False,
            "role_crossing_permitted": False,
        },
        "model_process_return_contract": [
            "source-native uint8 RGB20",
            "interpolated oracle box track metadata",
            "runtime timestamp/order metadata",
        ],
        "dense_numeric_trajectory_returned_to_model_process": False,
        "authority": {
            "formal_training_authorized": False,
            "formal_validation_authorized": False,
            "ready_to_train": False,
            "underlying_gate2_status": HOLD_STATUS,
            "hold_changed": False,
            "held_role_payload_accessed": False,
        },
    }


def expected_metadata_closure(role: str) -> Mapping[str, Any]:
    if role not in PROJECTION_ROLE_BINDINGS:
        raise StrictV3RGB20BridgeError("metadata closure role is invalid")
    specification = PROJECTION_ROLE_BINDINGS[role]
    return {
        "schema_version": "mosder_strict_v3_rgb20_metadata_closure_v2",
        "status": (
            "PASS_ROLE_SCOPED_SANITIZED_METADATA_CLOSURE_V2_"
            "NONAUTHORIZING_HOLD_GATE2_V3"
        ),
        "role": role,
        "resolved_case_rows": int(specification["rows"]),
        "unique_identities": int(specification["rows"]),
        "unique_physical_windows": int(specification["unique_visual_windows"]),
        "unique_visual_window_binding_keys": int(
            specification["unique_visual_windows"]
        ),
        "visual_key_physical_window_bijection": True,
        "source_binding_rows_verified_against_projection": int(specification["rows"]),
        "by_source": dict(specification["by_source"]),
        "projection_manifest": {
            "path": str(specification["path"]),
            "sha256": str(specification["sha256"]),
            "rows": int(specification["rows"]),
        },
        "sanitized_membership_manifest": {
            "path": str(SANITIZED_ROLE_BINDINGS[role]["path"]),
            "sha256": str(SANITIZED_ROLE_BINDINGS[role]["sha256"]),
            "rows": int(SANITIZED_ROLE_BINDINGS[role]["rows"]),
        },
        "original_operational_manifest": {
            "path": str(ORIGINAL_ROLE_BINDINGS[role]["path"]),
            "sha256": str(ORIGINAL_ROLE_BINDINGS[role]["sha256"]),
            "rows": int(ORIGINAL_ROLE_BINDINGS[role]["rows"]),
        },
        "cache_index_sha256": EXPECTED_ROLE_CACHE_INDEX_SHA256[role],
        "pixel_windows_decoded_at_startup": 0,
        "cache_lookup": "O(1)_identity_dictionary",
        "role_scoped": True,
        "held_role_payload_accessed": False,
        "formal_training_authorized": False,
        "formal_validation_authorized": False,
        "underlying_gate2_status": HOLD_STATUS,
        "hold_changed": False,
    }


@dataclass
class StrictV3LoadedRGB20:
    case_id: str
    source_dataset: str
    split_role: str
    frames: tuple[np.ndarray, ...]
    interpolation: Mapping[str, Any]
    _backing: Any = None

    def close(self) -> None:
        self.frames = ()
        self.interpolation = {}
        self._backing = None


class _RoleScopedResolver:
    """Use the tested resolver methods with an exact one-role row map."""

    def __new__(
        cls,
        base_class: type[Any],
        formal_module: ModuleType,
        formal_loader: Any,
        rows: Mapping[tuple[str, str, str], Mapping[str, Any]],
    ) -> Any:
        instance = object.__new__(base_class)
        instance.formal_module = formal_module
        instance.formal_loader = formal_loader
        instance.rows = rows
        return instance


class RoleScopedStrictV3RGB20Loader:
    """Decode projected membership after validating its dataset role."""

    def __init__(self, role: str) -> None:
        if role not in {"train", "validation"}:
            raise StrictV3RGB20BridgeError("Strict-V3 RGB20 role is invalid")
        self.role = role
        contract, backends = _load_materializer_modules()
        self.contract = contract
        self.backends = backends
        self.formal_module = backends.load_formal_rgb20_module()
        self.formal_loader = self._reader_only_formal_loader()
        self.rows, self.source_bindings = self._load_role_rows()
        self.resolver = _RoleScopedResolver(
            backends.StrictV3FormalRGB20Resolver,
            self.formal_module,
            self.formal_loader,
            self.rows,
        )
        self._ordered_identities = tuple(self.rows)
        self._resolved_by_identity, closure_base = self._resolve_role_metadata()
        self._metadata_closure = self._verify_projection_role_bindings(closure_base)

    @property
    def metadata_closure(self) -> Mapping[str, Any]:
        return dict(self._metadata_closure)

    @property
    def manifest_binding(self) -> Mapping[str, Any]:
        binding = self.contract.MANIFESTS[self.role]
        observed = {
            "path": str(binding["path"]),
            "sha256": str(binding["sha256"]),
            "rows": int(binding["rows"]),
        }
        expected = {
            "path": str(ORIGINAL_ROLE_BINDINGS[self.role]["path"]),
            "sha256": str(ORIGINAL_ROLE_BINDINGS[self.role]["sha256"]),
            "rows": int(ORIGINAL_ROLE_BINDINGS[self.role]["rows"]),
        }
        if observed != expected:
            raise StrictV3RGB20BridgeError(
                "original operational manifest binding drifted"
            )
        return observed

    def _reader_only_formal_loader(self) -> Any:
        module = self.formal_module
        loader = module.FormalRGB20InputLoader.__new__(module.FormalRGB20InputLoader)
        try:
            loader.pixel_policy, _ = module.pixel_v4.read_policy(
                module.FROZEN_PIXEL_POLICY, module.FROZEN_PIXEL_POLICY_SHA256
            )
            module.pixel_v4.verify_implementation_and_readers(loader.pixel_policy)
        except Exception as error:
            raise StrictV3RGB20BridgeError(
                "frozen role-scoped pixel reader policy failed"
            ) from error
        if (
            module.sha256_file(Path(module.anchor_v1.__file__).resolve())
            != module.ANCHOR_RESOLVER_SHA256
            or module.anchor_v1.BOX_ANCHOR_MODEL_FRAME_INDICES
            != list(module.BOX_ANCHOR_SLOTS)
            or module.anchor_v1.BOX_TO_MASK_CONTRACT != module.BOX_TO_MASK_CONTRACT
            or not module.ADT_HELPER.is_file()
        ):
            raise StrictV3RGB20BridgeError(
                "frozen role-scoped anchor/ADT helper contract differs"
            )
        role_roots = {
            current_role: root.resolve(strict=True)
            for current_role, root in TACO_ROLE_ROOTS.items()
        }
        if role_roots["train"] == role_roots["validation"]:
            raise StrictV3RGB20BridgeError("TACO role roots alias")
        loader.data_side = None
        loader.taco_roots = {self.role: role_roots[self.role]}
        return loader

    def _load_role_rows(
        self,
    ) -> tuple[
        Mapping[tuple[str, str, str], Mapping[str, Any]],
        Mapping[tuple[str, str, str], Mapping[str, Any]],
    ]:
        strict = self.contract.load_strict_loader()
        binding = self.contract.MANIFESTS[self.role]
        path = Path(binding["path"])
        try:
            payload = strict._impl._read_bound_file(path, str(binding["sha256"]))
        except Exception as error:
            raise StrictV3RGB20BridgeError(
                "cannot securely read role-scoped Strict-V3 manifest"
            ) from error
        raw_lines = payload.splitlines(keepends=True)
        if len(raw_lines) != int(binding["rows"]):
            raise StrictV3RGB20BridgeError(
                "role-scoped Strict-V3 manifest row count differs"
            )
        original_rows: list[Mapping[str, Any]] = []
        original_lines: list[bytes] = []
        original_identities: set[tuple[str, str, str]] = set()
        taco_root = self.formal_loader.taco_roots[self.role]
        for index, raw_line in enumerate(raw_lines):
            try:
                raw = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise StrictV3RGB20BridgeError(
                    "role-scoped Strict-V3 row is invalid JSON"
                ) from error
            if (
                not isinstance(raw, dict)
                or self.contract.canonical_line(raw) != raw_line
            ):
                raise StrictV3RGB20BridgeError(
                    "role-scoped Strict-V3 row is not canonical JSONL"
                )
            try:
                case = strict.load_operational_case(
                    raw, expected_role=self.role, expected_row_index=index
                )
            except Exception as error:
                raise StrictV3RGB20BridgeError(
                    "role-scoped Strict-V3 row failed its frozen loader"
                ) from error
            identity = tuple(case.identity)
            if len(identity) != 3 or identity in original_identities:
                raise StrictV3RGB20BridgeError(
                    "role-scoped Strict-V3 identity is invalid or duplicated"
                )
            if identity != (
                str(raw["identity"]["source_dataset"]),
                self.role,
                str(raw["identity"]["case_id"]),
            ):
                raise StrictV3RGB20BridgeError(
                    "role-scoped Strict-V3 identity ordering differs"
                )
            if identity[0] == self.formal_module.TACO_SOURCE:
                container_path = Path(
                    str(raw["rgb20"]["formal_RGB_container"]["path"])
                ).resolve(strict=True)
                try:
                    container_path.relative_to(taco_root)
                except ValueError as error:
                    raise StrictV3RGB20BridgeError(
                        "TACO operational media escapes its role root"
                    ) from error
            original_identities.add(identity)
            original_rows.append(raw)
            original_lines.append(raw_line)
        if len(original_rows) != int(binding["rows"]):
            raise StrictV3RGB20BridgeError(
                "role-scoped original Strict-V3 identity closure differs"
            )

        sanitized = SANITIZED_ROLE_BINDINGS[self.role]
        sanitized_path = Path(sanitized["path"])
        if (
            not sanitized_path.is_file()
            or _sha256_file(sanitized_path) != sanitized["sha256"]
        ):
            raise StrictV3RGB20BridgeError(
                "sanitized retained role manifest path/hash differs"
            )
        sanitized_lines = sanitized_path.read_bytes().splitlines(keepends=True)
        if len(sanitized_lines) != int(sanitized["rows"]):
            raise StrictV3RGB20BridgeError("sanitized retained role row count differs")
        rows: dict[tuple[str, str, str], Mapping[str, Any]] = {}
        source_bindings: dict[tuple[str, str, str], Mapping[str, Any]] = {}
        previous_original_index = -1
        for sanitized_index, raw_line in enumerate(sanitized_lines):
            try:
                raw = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise StrictV3RGB20BridgeError(
                    "sanitized retained role row is invalid JSON"
                ) from error
            selection = (
                raw.get("selection_row_binding") if isinstance(raw, dict) else None
            )
            original_index = (
                selection.get("row_index_zero_based")
                if isinstance(selection, Mapping)
                else None
            )
            if not (
                isinstance(raw, dict)
                and self.contract.canonical_line(raw) == raw_line
                and type(original_index) is int
                and previous_original_index < original_index < len(original_rows)
                and raw_line == original_lines[original_index]
            ):
                raise StrictV3RGB20BridgeError(
                    "sanitized row is not an ordered original operational row"
                )
            try:
                case = strict.load_operational_case(
                    raw,
                    expected_role=self.role,
                    expected_row_index=original_index,
                )
            except Exception as error:
                raise StrictV3RGB20BridgeError(
                    "sanitized retained row failed the frozen operational loader"
                ) from error
            identity = tuple(case.identity)
            if not (
                len(identity) == 3
                and identity in original_identities
                and identity not in rows
                and identity
                == (
                    str(raw["identity"]["source_dataset"]),
                    self.role,
                    str(raw["identity"]["case_id"]),
                )
                and original_rows[original_index]["record_payload_sha256"]
                == raw["record_payload_sha256"]
            ):
                raise StrictV3RGB20BridgeError(
                    "sanitized/original Strict-V3 identity closure differs"
                )
            rows[identity] = raw
            source_bindings[identity] = {
                "sanitized_manifest_sha256": str(sanitized["sha256"]),
                "sanitized_row_index_zero_based": sanitized_index,
                "original_operational_manifest_sha256": str(binding["sha256"]),
                "original_operational_row_index_zero_based": original_index,
                "source_record_payload_sha256": raw["record_payload_sha256"],
                "source_rgb20_canonical_sha256": self.contract.canonical_sha256(
                    raw["rgb20"]
                ),
                "source_box_record_canonical_sha256": self.contract.canonical_sha256(
                    raw["anchors"]["box"]
                ),
                "selector_row_canonical_sha256": raw["selector_row_canonical_sha256"],
                # The projection's source iterator hashes the canonical JSON
                # payload without its JSONL framing newline.
                "source_canonical_line_sha256": hashlib.sha256(
                    raw_line[:-1]
                ).hexdigest(),
            }
            previous_original_index = original_index
        if len(rows) != int(sanitized["rows"]):
            raise StrictV3RGB20BridgeError(
                "sanitized role-scoped Strict-V3 identity closure differs"
            )
        return rows, source_bindings

    def _resolve_role_metadata(
        self,
    ) -> tuple[Mapping[tuple[str, str, str], Any], Mapping[str, Any]]:
        """Resolve the complete allowed role once, without decoding pixels."""

        requests = tuple(
            (role, source, case_id)
            for source, role, case_id in self._ordered_identities
        )
        try:
            resolved_items = self.resolver.resolve_cases(requests)
        except Exception as error:
            raise StrictV3RGB20BridgeError(
                "role-complete Strict-V3 metadata resolution failed"
            ) from error
        expected = PROJECTION_ROLE_BINDINGS[self.role]
        if len(resolved_items) != len(requests) or len(requests) != int(
            expected["rows"]
        ):
            raise StrictV3RGB20BridgeError(
                "role-complete resolved metadata row count differs"
            )

        resolved_by_identity: dict[tuple[str, str, str], Any] = {}
        physical_to_visual: dict[str, str] = {}
        visual_to_physical: dict[str, str] = {}
        source_counts: Counter[str] = Counter()
        for identity, resolved in zip(
            self._ordered_identities, resolved_items, strict=True
        ):
            source, role, case_id = identity
            row = getattr(resolved, "row", None)
            records = getattr(resolved, "records", None)
            visual_key = getattr(resolved, "visual_window_binding_key", None)
            raw_identity = self.rows[identity].get("identity")
            if not (
                isinstance(row, Mapping)
                and isinstance(records, Mapping)
                and isinstance(raw_identity, Mapping)
                and row.get("source_dataset") == source
                and row.get("split_role") == role == self.role
                and row.get("case_id") == case_id
                and raw_identity.get("source_dataset") == source
                and raw_identity.get("split_role") == role
                and raw_identity.get("case_id") == case_id
                and row.get("physical_window_id")
                == raw_identity.get("physical_window_id")
                and _is_sha256(visual_key)
                and self.formal_loader._visual_window_binding_key(row, records)
                == visual_key
            ):
                raise StrictV3RGB20BridgeError(
                    "resolved metadata identity/visual binding differs"
                )
            if identity in resolved_by_identity:
                raise StrictV3RGB20BridgeError(
                    "role-complete metadata cache duplicated an identity"
                )
            physical_window_id = str(row["physical_window_id"])
            prior_visual = physical_to_visual.setdefault(
                physical_window_id, str(visual_key)
            )
            prior_physical = visual_to_physical.setdefault(
                str(visual_key), physical_window_id
            )
            if prior_visual != visual_key or prior_physical != physical_window_id:
                raise StrictV3RGB20BridgeError(
                    "physical-window/visual-key mapping is not bijective"
                )
            resolved_by_identity[identity] = resolved
            source_counts[source] += 1

        if not (
            set(resolved_by_identity) == set(self.rows)
            and len(physical_to_visual) == int(expected["unique_visual_windows"])
            and len(visual_to_physical) == int(expected["unique_visual_windows"])
            and dict(source_counts) == dict(expected["by_source"])
        ):
            raise StrictV3RGB20BridgeError(
                "role-complete metadata cache closure differs"
            )
        return resolved_by_identity, {
            "unique_physical_windows": len(physical_to_visual),
            "unique_visual_window_binding_keys": len(visual_to_physical),
            "by_source": dict(source_counts),
        }

    def _verify_projection_role_bindings(
        self, closure_base: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Bind every cached identity back to the immutable MoSDeR view."""

        specification = PROJECTION_ROLE_BINDINGS[self.role]
        projection_path = Path(specification["path"])
        if (
            not projection_path.is_file()
            or _sha256_file(projection_path) != specification["sha256"]
        ):
            raise StrictV3RGB20BridgeError(
                "role-scoped MoSDeR projection path/hash binding failed"
            )
        cache_digest = hashlib.sha256()
        projected_identities: set[tuple[str, str, str]] = set()
        projected_physical_windows: set[str] = set()
        with projection_path.open("rb") as handle:
            for index, raw_line in enumerate(handle):
                try:
                    projected = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise StrictV3RGB20BridgeError(
                        "role-scoped MoSDeR projection row is invalid JSON"
                    ) from error
                if not (
                    isinstance(projected, dict)
                    and _canonical_json_line(projected) == raw_line
                    and projected.get("schema_version") == PROJECTION_ROW_SCHEMA
                ):
                    raise StrictV3RGB20BridgeError(
                        "role-scoped MoSDeR projection row is not canonical"
                    )
                payload = dict(projected)
                declared_payload_sha = payload.pop("projected_payload_sha256", None)
                identity_payload = projected.get("audit_identity")
                source_binding = projected.get("source_binding")
                authority = projected.get("authority")
                if not all(
                    isinstance(value, Mapping)
                    for value in (identity_payload, source_binding, authority)
                ):
                    raise StrictV3RGB20BridgeError(
                        "role-scoped MoSDeR projection binding is incomplete"
                    )
                assert isinstance(identity_payload, Mapping)
                assert isinstance(source_binding, Mapping)
                assert isinstance(authority, Mapping)
                identity = (
                    str(identity_payload.get("source_dataset")),
                    str(identity_payload.get("split_role")),
                    str(identity_payload.get("case_id")),
                )
                if not (
                    index < len(self._ordered_identities)
                    and identity == self._ordered_identities[index]
                    and identity not in projected_identities
                    and identity in self._resolved_by_identity
                    and dict(source_binding) == dict(self.source_bindings[identity])
                    and identity_payload.get("physical_window_id")
                    == self.rows[identity]["identity"].get("physical_window_id")
                    and projected.get("loader_request")
                    == {
                        "case_id": identity[2],
                        "resolver": "formal_rgb20_input_loader_v1",
                        "source_dataset": identity[0],
                        "split_role": self.role,
                    }
                    and _is_sha256(declared_payload_sha)
                    and hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
                    == declared_payload_sha
                    and authority.get("formal_training_authorized") is False
                    and authority.get("formal_validation_authorized") is False
                    and authority.get("underlying_gate2_status") == HOLD_STATUS
                    and authority.get("hold_changed") is False
                    and authority.get("held_role_payload_accessed_by_projection")
                    is False
                ):
                    raise StrictV3RGB20BridgeError(
                        "projected/operational/cache binding differs"
                    )
                projected_identities.add(identity)
                projected_physical_windows.add(
                    str(identity_payload["physical_window_id"])
                )
                resolved = self._resolved_by_identity[identity]
                cache_digest.update(
                    _canonical_json_line(
                        {
                            "identity": list(identity),
                            "physical_window_id": identity_payload[
                                "physical_window_id"
                            ],
                            "visual_window_binding_key": (
                                resolved.visual_window_binding_key
                            ),
                            "source_binding": dict(source_binding),
                            "projected_payload_sha256": declared_payload_sha,
                        }
                    )
                )

        observed_digest = cache_digest.hexdigest()
        expected_closure = expected_metadata_closure(self.role)
        observed_closure = {
            **dict(expected_closure),
            "resolved_case_rows": len(self._resolved_by_identity),
            "unique_identities": len(projected_identities),
            "unique_physical_windows": len(projected_physical_windows),
            "unique_visual_window_binding_keys": int(
                closure_base["unique_visual_window_binding_keys"]
            ),
            "source_binding_rows_verified_against_projection": len(
                projected_identities
            ),
            "by_source": dict(closure_base["by_source"]),
            "cache_index_sha256": observed_digest,
        }
        expected_digest = EXPECTED_ROLE_CACHE_INDEX_SHA256[self.role]
        if not (
            set(projected_identities) == set(self._ordered_identities)
            and len(projected_identities) == int(specification["rows"])
            and len(projected_physical_windows)
            == int(specification["unique_visual_windows"])
            and observed_digest == expected_digest
            and observed_closure == expected_closure
        ):
            raise StrictV3RGB20BridgeError(
                "role-scoped bridge metadata receipt differs"
            )
        return observed_closure

    def load_case(
        self,
        role: str,
        source_dataset: str,
        case_id: str,
        expected_source_binding: Mapping[str, Any],
    ) -> StrictV3LoadedRGB20:
        identity = (source_dataset, role, case_id)
        if role != self.role or identity not in self.rows:
            raise StrictV3RGB20BridgeError(
                "RGB20 request is outside the exact role-scoped membership"
            )
        observed_binding = self.source_bindings[identity]
        if dict(expected_source_binding) != dict(observed_binding):
            raise StrictV3RGB20BridgeError(
                "projected/source operational row binding differs"
            )
        try:
            # Resolution/provenance lookup is O(1); only the selected RGB20
            # visual is decoded on demand.
            resolved = self._resolved_by_identity[identity]
            frames, frame_evidence, source_evidence, backing = (
                self.resolver.decode_resolved_window(resolved)
            )
            self.resolver._verify_cached_visual_for_case(resolved, frame_evidence)
            if len(frames) != 20 or any(
                frame.dtype != np.uint8
                or frame.ndim != 3
                or frame.shape[2] != 3
                or frame.shape != frames[0].shape
                for frame in frames
            ):
                raise StrictV3RGB20BridgeError(
                    "formal Strict-V3 decoder did not return source-native RGB20"
                )
            height, width, _ = frames[0].shape
            if source_dataset in self.formal_module.LOCAL_SOURCES:
                boxes, anchor_timestamps, _ = self.resolver._local_anchor_inputs(
                    resolved.records["gate6_anchor_box_record"],
                    frame_evidence,
                    width,
                    height,
                )
                timestamps = tuple(
                    int(item["frozen_timestamp_ns"]) for item in frame_evidence
                )
            else:
                boxes, anchor_timestamps, _ = self.resolver._taco_anchor_inputs(
                    resolved.row, frame_evidence, width, height
                )
                timestamps = tuple(
                    int(item["decoded_timestamp_ns"]) for item in frame_evidence
                )
            interpolation_with_masks = (
                self.formal_module.interpolate_and_mask_native_rgb20(
                    anchor_boxes=boxes,
                    anchor_timestamps=anchor_timestamps,
                    frame_timestamps=timestamps,
                    width=width,
                    height=height,
                )
            )
            interpolation = {
                key: value
                for key, value in interpolation_with_masks.items()
                if key != "pair_masks"
            }
            del interpolation_with_masks
            if source_evidence.get("pixels_persisted") is True:
                raise StrictV3RGB20BridgeError(
                    "formal Strict-V3 decoder unexpectedly persisted pixels"
                )
            return StrictV3LoadedRGB20(
                case_id=case_id,
                source_dataset=source_dataset,
                split_role=role,
                frames=tuple(frames),
                interpolation=interpolation,
                _backing=backing,
            )
        except StrictV3RGB20BridgeError:
            raise
        except Exception as error:
            raise StrictV3RGB20BridgeError(
                "role-scoped Strict-V3 RGB20 resolution/decode failed"
            ) from error


__all__ = [
    "MATERIALIZER_BACKENDS",
    "MATERIALIZER_BACKENDS_SHA256",
    "MATERIALIZER_CONTRACT",
    "MATERIALIZER_CONTRACT_SHA256",
    "BRIDGE_RESOLVER_NAME",
    "BRIDGE_SCHEMA_VERSION",
    "EXPECTED_ROLE_CACHE_INDEX_SHA256",
    "PROJECTION_ROLE_BINDINGS",
    "RoleScopedStrictV3RGB20Loader",
    "StrictV3LoadedRGB20",
    "StrictV3RGB20BridgeError",
    "bridge_identity",
    "dependency_binding",
    "expected_metadata_closure",
]
