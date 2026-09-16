#!/usr/bin/env python3
"""Native RGB video interfaces for Qwen3-VL, Molmo2, and NVILA.

Each backend maps 20 RGB frames, a prompt, and a target box track to the
family's visual encoder, projector, and language decoder. Model loading checks
the pinned family environment. MoSDeR attaches its modules in
``mosder_final_v1.family_backend``; ordinary LoRA uses the helpers here."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterable,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from tst_na_o18_core_v1 import (
    PhysicalOutput,
    SourceConditionedDualLoRALinear,
    TsTNativeAdapterO18Core,
    configure_stage_l,
    configure_stage_p,
    dual_lora_source_context,
)


SCHEMA_VERSION = "tst_native_adapter_family_backends_v1"
FRAME_COUNT = 20
TEMPORAL_UNITS = 10
PHYSICAL_RESIDUAL_SCALE = 0.05


class NativeBackendError(RuntimeError):
    """A family, runtime, tensor, adapter, or online-forward contract failed."""


class NativeBackendBlocked(NativeBackendError):
    """A real family path is known but cannot be executed in this runtime."""


@dataclass(frozen=True)
class FamilyBinding:
    key: str
    display_name: str
    model_root: Path
    interpreter: Path
    torch_version: str
    transformers_version: str
    peft_version: str
    architecture: str
    hidden_size: int
    decoder_path: str
    decoder_block_count: int
    decoder_block_class_fragment: str
    post_projector_path: str
    native_visual_temporal: int
    native_visual_grid_hw: tuple[int, int] | None
    spatial_merge: int
    video_pixel_budget_primary: int | None
    video_pixel_budget_memory_fallback: int | None
    video_pixel_budget_shortest_edge: int | None
    native_source_lora_attention_leaves: tuple[str, ...]
    ordinary_lora_attention_leaves: tuple[str, ...]
    ordinary_lora_mlp_leaves: tuple[str, ...]
    native_generate_path: str
    native_hidden_path: str
    reference_code_root: Path | None = None

    def block_path(self, index: int) -> str:
        if not 0 <= index < self.decoder_block_count:
            raise NativeBackendError(
                f"{self.display_name} decoder layer {index} is outside "
                f"[0,{self.decoder_block_count})"
            )
        return f"{self.decoder_path}.{index}"


# LOCAL_PATH: Set each model_root and interpreter for the target machine;
# model loading also checks the Python environment and package versions below.
FAMILY_BINDINGS: Mapping[str, FamilyBinding] = {
    "qwen3_vl_8b": FamilyBinding(
        key="qwen3_vl_8b",
        display_name="Qwen3-VL-8B",
        model_root=Path("/root/autodl-tmp/models/hf/Qwen__Qwen3-VL-8B-Instruct"),
        interpreter=Path("/root/autodl-tmp/envs/fca_qwen_py312/bin/python3.12"),
        torch_version="2.8.0+cu128",
        transformers_version="5.13.0",
        peft_version="0.18.1",
        architecture="Qwen3VLForConditionalGeneration",
        hidden_size=4096,
        decoder_path="model.language_model.layers",
        decoder_block_count=36,
        decoder_block_class_fragment="Qwen3VLTextDecoderLayer",
        post_projector_path="model.visual",
        native_visual_temporal=10,
        native_visual_grid_hw=None,
        spatial_merge=2,
        video_pixel_budget_primary=25_165_824,
        video_pixel_budget_memory_fallback=6_291_456,
        video_pixel_budget_shortest_edge=4_096,
        native_source_lora_attention_leaves=("self_attn.q_proj", "self_attn.v_proj"),
        ordinary_lora_attention_leaves=(
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ),
        ordinary_lora_mlp_leaves=("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
        native_generate_path="Qwen3VLForConditionalGeneration.generate",
        native_hidden_path="Qwen3VLCausalLMOutputWithPast.hidden_states",
    ),
    "molmo2_o_7b": FamilyBinding(
        key="molmo2_o_7b",
        display_name="Molmo2-O-7B",
        model_root=Path("/root/autodl-tmp/models/hf/allenai__Molmo2-O-7B"),
        interpreter=Path("/root/autodl-tmp/envs/molmo2/bin/python"),
        torch_version="2.8.0+cu128",
        transformers_version="4.57.1",
        peft_version="0.15.2",
        architecture="Molmo2ForConditionalGeneration",
        hidden_size=4096,
        decoder_path="model.transformer.blocks",
        decoder_block_count=32,
        decoder_block_class_fragment="Molmo2",
        post_projector_path="model.vision_backbone",
        native_visual_temporal=20,
        native_visual_grid_hw=(9, 9),
        spatial_merge=1,
        video_pixel_budget_primary=None,
        video_pixel_budget_memory_fallback=None,
        video_pixel_budget_shortest_edge=None,
        native_source_lora_attention_leaves=("self_attn.att_proj",),
        ordinary_lora_attention_leaves=("self_attn.att_proj", "self_attn.attn_out"),
        ordinary_lora_mlp_leaves=("mlp.ff_proj", "mlp.ff_out"),
        native_generate_path="Molmo2ForConditionalGeneration.generate",
        native_hidden_path="Molmo2CausalLMOutputWithPast.hidden_states",
    ),
    "nvila_lite_8b": FamilyBinding(
        key="nvila_lite_8b",
        display_name="NVILA-Lite-8B",
        model_root=Path("/root/autodl-tmp/models/story2/NVILA-Lite-8B"),
        interpreter=Path("/root/miniconda3/envs/vlm-llava/bin/python"),
        torch_version="2.8.0+cu128",
        transformers_version="4.46.3",
        peft_version="0.15.2",
        architecture="LlavaLlamaModel",
        hidden_size=3584,
        decoder_path="llm.model.layers",
        decoder_block_count=28,
        decoder_block_class_fragment="Qwen2DecoderLayer",
        post_projector_path="mm_projector",
        native_visual_temporal=20,
        native_visual_grid_hw=(11, 11),
        spatial_merge=1,
        video_pixel_budget_primary=None,
        video_pixel_budget_memory_fallback=None,
        video_pixel_budget_shortest_edge=None,
        native_source_lora_attention_leaves=("self_attn.q_proj", "self_attn.v_proj"),
        ordinary_lora_attention_leaves=(
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ),
        ordinary_lora_mlp_leaves=("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
        native_generate_path="LlavaLlamaModel.generate -> llm.generate(inputs_embeds=...)",
        native_hidden_path="CausalLMOutputWithPast.hidden_states",
        # LOCAL_PATH: NVILA requires this external 4d-rgpt source checkout.
        reference_code_root=Path(
            "/root/story2_camera_object_motion/references/4d-rgpt"
        ),
    ),
}


def binding_for(family: str) -> FamilyBinding:
    try:
        return FAMILY_BINDINGS[family]
    except KeyError as error:
        raise NativeBackendError(f"unsupported family: {family!r}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _weight_manifest(root: Path, index_relative: str) -> Mapping[str, Any]:
    index_path = root / index_relative
    if not index_path.is_file():
        raise NativeBackendError(f"weight index absent: {index_path}")
    value = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise NativeBackendError(f"invalid weight map: {index_path}")
    shards = sorted(set(map(str, weight_map.values())))
    rows = []
    for shard in shards:
        path = root / Path(index_relative).parent / shard
        if not path.is_file():
            raise NativeBackendError(f"listed model shard absent: {path}")
        rows.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size})
    return {
        "index_path": str(index_path),
        "index_sha256": _sha256_file(index_path),
        "parameter_key_count": len(weight_map),
        "shards": rows,
        "shard_size_manifest_sha256": _canonical_sha256(rows),
    }


def _assert_index_parameter_keys(
    root: Path,
    index_relative: str,
    required_keys: Sequence[str],
) -> Mapping[str, Any]:
    """Prove selected native Linear weights exist without loading a shard."""
    index_path = root / index_relative
    value = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = value.get("weight_map", {})
    missing = [key for key in required_keys if key not in weight_map]
    if missing:
        raise NativeBackendError(
            {
                "reason": "bound native projection keys absent from weight index",
                "index": str(index_path),
                "missing": missing,
            }
        )
    control = {key: str(weight_map[key]) for key in sorted(required_keys)}
    return {
        "required_parameter_to_shard": control,
        "required_parameter_control_sha256": _canonical_sha256(control),
    }


def discover_local_family(family: str) -> Mapping[str, Any]:
    """Read only local configs/indexes; never instantiate an 8B model."""
    binding = binding_for(family)
    root = binding.model_root
    config_path = root / "config.json"
    if not config_path.is_file():
        raise NativeBackendError(f"model config absent: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or binding.architecture not in architectures:
        raise NativeBackendError(
            f"{binding.display_name} architecture drift: {architectures!r}"
        )

    if family == "qwen3_vl_8b":
        text = config.get("text_config", {})
        vision = config.get("vision_config", {})
        video_config_path = root / "video_preprocessor_config.json"
        if not video_config_path.is_file():
            raise NativeBackendError("Qwen video preprocessor config is absent")
        video_config = json.loads(video_config_path.read_text(encoding="utf-8"))
        video_size = video_config.get("size", {})
        observed = {
            "decoder_blocks": text.get("num_hidden_layers"),
            "hidden_size": text.get("hidden_size"),
            "vision_blocks": vision.get("depth"),
            "vision_hidden_size": vision.get("hidden_size"),
            "vision_out_hidden_size": vision.get("out_hidden_size"),
            "spatial_merge": vision.get("spatial_merge_size"),
            "video_longest_edge": video_size.get("longest_edge"),
            "video_shortest_edge": video_size.get("shortest_edge"),
        }
        expected = {
            "decoder_blocks": 36,
            "hidden_size": 4096,
            "vision_blocks": 27,
            "vision_hidden_size": 1152,
            "vision_out_hidden_size": 4096,
            "spatial_merge": 2,
            "video_longest_edge": binding.video_pixel_budget_primary,
            "video_shortest_edge": binding.video_pixel_budget_shortest_edge,
        }
        weights = [_weight_manifest(root, "model.safetensors.index.json")]
        key_control = _assert_index_parameter_keys(
            root,
            "model.safetensors.index.json",
            [
                f"model.language_model.layers.{index}.{leaf}.weight"
                for index in (0, binding.decoder_block_count - 1)
                for leaf in binding.native_source_lora_attention_leaves
            ],
        )
    elif family == "molmo2_o_7b":
        text = config.get("text_config", {})
        vision = config.get("vit_config", {})
        adapter = config.get("adapter_config", {})
        observed = {
            "decoder_blocks": text.get("num_hidden_layers"),
            "hidden_size": text.get("hidden_size"),
            "vision_blocks": vision.get("num_hidden_layers"),
            "vision_hidden_size": vision.get("hidden_size"),
            "projected_hidden_size": adapter.get("text_hidden_size"),
            "projector_vit_layers": adapter.get("vit_layers"),
        }
        expected = {
            "decoder_blocks": 32,
            "hidden_size": 4096,
            "vision_blocks": 27,
            "vision_hidden_size": 1152,
            "projected_hidden_size": 4096,
            "projector_vit_layers": [-3, -9],
        }
        weights = [_weight_manifest(root, "model.safetensors.index.json")]
        key_control = _assert_index_parameter_keys(
            root,
            "model.safetensors.index.json",
            [
                f"model.transformer.blocks.{index}.{leaf}.weight"
                for index in (0, binding.decoder_block_count - 1)
                for leaf in binding.native_source_lora_attention_leaves
            ],
        )
    else:
        llm = config.get("llm_cfg", {})
        vision = config.get("vision_tower_cfg", {})
        projector = config.get("mm_projector_cfg", {})
        observed = {
            "decoder_blocks": llm.get("num_hidden_layers"),
            "hidden_size": llm.get("hidden_size"),
            "vision_blocks": vision.get("num_hidden_layers"),
            "vision_hidden_size": vision.get("hidden_size"),
            "projector_type": projector.get("mm_projector_type"),
            "top_level_hidden_size": config.get("hidden_size"),
        }
        expected = {
            "decoder_blocks": 28,
            "hidden_size": 3584,
            "vision_blocks": 27,
            "vision_hidden_size": 1152,
            "projector_type": "mlp_downsample_3x3_fix",
            "top_level_hidden_size": 3584,
        }
        weights = [_weight_manifest(root, "llm/model.safetensors.index.json")]
        key_control = _assert_index_parameter_keys(
            root,
            "llm/model.safetensors.index.json",
            [
                f"model.layers.{index}.{leaf}.weight"
                for index in (0, binding.decoder_block_count - 1)
                for leaf in binding.native_source_lora_attention_leaves
            ],
        )
        for relative in (
            "vision_tower/model.safetensors",
            "mm_projector/model.safetensors",
        ):
            path = root / relative
            if not path.is_file():
                raise NativeBackendError(f"NVILA component absent: {path}")
            weights.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "control_identity": "single_safetensors_file_size_plus_config_hash",
                }
            )
    if observed != expected:
        raise NativeBackendError(
            {
                "family": family,
                "expected_config": expected,
                "observed": observed,
            }
        )

    module_contract = {
        "decoder_container": binding.decoder_path,
        "decoder_block_count": binding.decoder_block_count,
        "post_projector": binding.post_projector_path,
        "native_hidden": binding.native_hidden_path,
        "native_generate": binding.native_generate_path,
        "native_source_conditioned_lora_leaves": list(
            binding.native_source_lora_attention_leaves
        ),
        "native_source_conditioned_lora_formula": (
            "W0(x)+Bc(Ac(x)*tanh(Gc(mean(camera_source))))+"
            "Bo(Ao(x)*tanh(Go(mean(object_source))))"
        ),
        "unified_physical_core": (
            "Camera/Object family-width [10,D] native sources must feed both "
            "Camera18/Object18 and masked native-token reinjection"
        ),
        "physical_residual_scale_fixed_nontrainable": PHYSICAL_RESIDUAL_SCALE,
        "video_pixel_budget": {
            "primary": binding.video_pixel_budget_primary,
            "memory_fallback": binding.video_pixel_budget_memory_fallback,
            "shortest_edge": binding.video_pixel_budget_shortest_edge,
            "fallback_trigger": (
                "memory/OOM predeclared only; never selected from accuracy"
                if binding.video_pixel_budget_memory_fallback is not None
                else None
            ),
        },
        "ordinary_lora_attention_baseline_leaves": list(
            binding.ordinary_lora_attention_leaves
        ),
        "ordinary_lora_is_not_tst_na": True,
        "legacy_block_residual_adapter_paths_not_decoder_lora": [
            binding.block_path(i) for i in range(binding.decoder_block_count)
        ],
    }
    artifact = {
        "model_root": str(root),
        "config_sha256": _sha256_file(config_path),
        "weight_controls": weights,
        "selected_native_projection_key_control": key_control,
    }
    if family == "qwen3_vl_8b":
        artifact["video_preprocessor_config_sha256"] = _sha256_file(
            root / "video_preprocessor_config.json"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "display_name": binding.display_name,
        "status": "PASS_STATIC_LOCAL_ARTIFACT_AND_MODULE_CONTRACT_NOT_REAL_FORWARD",
        "ready_for_real_online_forward": False,
        "reason": "8B weights were not loaded by static discovery",
        "artifact": artifact,
        "artifact_control_sha256": _canonical_sha256(artifact),
        "config_observed": observed,
        "module_contract": module_contract,
        "runtime_required": {
            "interpreter": str(binding.interpreter),
            "torch": binding.torch_version,
            "transformers": binding.transformers_version,
            "peft": binding.peft_version,
        },
    }


def discover_all_local_families() -> Mapping[str, Mapping[str, Any]]:
    return {family: discover_local_family(family) for family in FAMILY_BINDINGS}


def audit_runtime(binding: FamilyBinding) -> Mapping[str, Any]:
    observed = {
        "interpreter": str(Path(sys.executable).resolve()),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
    }
    expected = {
        "interpreter": str(binding.interpreter.resolve()),
        "torch": binding.torch_version,
        "transformers": binding.transformers_version,
        "peft": binding.peft_version,
    }
    compatible = observed == expected
    return {"compatible": compatible, "expected": expected, "observed": observed}


def require_runtime(binding: FamilyBinding) -> Mapping[str, Any]:
    audit = audit_runtime(binding)
    if not audit["compatible"]:
        raise NativeBackendBlocked(
            {
                "reason": "family runtime mismatch",
                "family": binding.display_name,
                **audit,
            }
        )
    return audit


def _as_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        raise NativeBackendError(
            "RGB20 frames must be source RGB arrays/PIL images; tensor input is "
            "rejected so a cached feature cannot masquerade as online RGB"
        )
    if isinstance(frame, np.ndarray):
        array = frame
    elif hasattr(frame, "convert") and hasattr(frame, "size"):
        array = np.asarray(frame.convert("RGB"))
    else:
        raise NativeBackendError(f"unsupported raw RGB frame type: {type(frame)!r}")
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise NativeBackendError("every RGB20 frame must be uint8 HWC RGB")
    return np.ascontiguousarray(array)


@dataclass(frozen=True)
class RGB20Request:
    frames: Sequence[Any]
    timestamps_ns: Sequence[int]
    prompt: str
    oracle_boxes_xyxy: Sequence[Sequence[float]]
    request_id: str = "unbound_sandbox_request"

    def validated(self) -> "ValidatedRGB20":
        if len(self.frames) != FRAME_COUNT:
            raise NativeBackendError(
                f"RGB20 requires exactly 20 frames, got {len(self.frames)}"
            )
        arrays = tuple(_as_rgb_array(frame) for frame in self.frames)
        shape = arrays[0].shape
        if any(frame.shape != shape for frame in arrays[1:]):
            raise NativeBackendError("RGB20 frame dimensions differ")
        timestamps = tuple(self.timestamps_ns)
        if len(timestamps) != FRAME_COUNT or any(
            type(x) is not int for x in timestamps
        ):
            raise NativeBackendError(
                "timestamps_ns must contain exactly 20 Python integers"
            )
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise NativeBackendError("timestamps_ns must be strictly increasing")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise NativeBackendError(
                "native VLM forward requires a nonempty language prompt"
            )
        boxes = np.asarray(self.oracle_boxes_xyxy, dtype=np.float64)
        if boxes.shape != (FRAME_COUNT, 4) or not np.isfinite(boxes).all():
            raise NativeBackendError("oracle box track must be finite [20,4] xyxy")
        if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
            raise NativeBackendError("oracle boxes must be nonempty half-open xyxy")
        height, width = shape[:2]
        if (
            np.any(boxes[:, 0] < 0)
            or np.any(boxes[:, 1] < 0)
            or np.any(boxes[:, 2] > width)
            or np.any(boxes[:, 3] > height)
        ):
            raise NativeBackendError("oracle boxes fall outside RGB frame bounds")
        return ValidatedRGB20(
            frames=arrays,
            timestamps_ns=timestamps,
            prompt=self.prompt.strip(),
            boxes_xyxy=np.ascontiguousarray(boxes),
            request_id=str(self.request_id),
            height=height,
            width=width,
        )


@dataclass(frozen=True)
class ValidatedRGB20:
    frames: tuple[np.ndarray, ...]
    timestamps_ns: tuple[int, ...]
    prompt: str
    boxes_xyxy: np.ndarray
    request_id: str
    height: int
    width: int

    @property
    def offsets_seconds(self) -> np.ndarray:
        origin = self.timestamps_ns[0]
        return np.asarray(
            [(value - origin) / 1_000_000_000.0 for value in self.timestamps_ns],
            dtype=np.float64,
        )

    @property
    def sampled_fps(self) -> float:
        deltas = np.diff(np.asarray(self.timestamps_ns, dtype=np.int64))
        return float(1_000_000_000.0 / np.median(deltas))


def project_frame_box_masks(
    boxes_xyxy: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    grid_width: int,
    grid_height: int,
) -> torch.Tensor:
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    if boxes.shape != (FRAME_COUNT, 4) or not np.isfinite(boxes).all():
        raise NativeBackendError("box projection requires finite [20,4]")
    if any(
        type(value) is not int or value <= 0
        for value in (image_width, image_height, grid_width, grid_height)
    ):
        raise NativeBackendError("image/grid sizes must be positive integers")
    masks = torch.zeros((FRAME_COUNT, grid_height, grid_width), dtype=torch.bool)
    for frame_index, box in enumerate(boxes):
        x0 = max(0, min(grid_width, math.floor(box[0] * grid_width / image_width)))
        y0 = max(0, min(grid_height, math.floor(box[1] * grid_height / image_height)))
        x1 = max(0, min(grid_width, math.ceil(box[2] * grid_width / image_width)))
        y1 = max(0, min(grid_height, math.ceil(box[3] * grid_height / image_height)))
        if x1 <= x0 or y1 <= y0:
            raise NativeBackendError("a projected target box is empty")
        masks[frame_index, y0:y1, x0:x1] = True
    if any(not bool(mask.any()) or bool(mask.all()) for mask in masks):
        raise NativeBackendError(
            "every native-frame target mask must be nonempty and nonfull"
        )
    return masks


def adjacent_unit_masks(frame_masks: torch.Tensor) -> torch.Tensor:
    if (
        frame_masks.dtype is not torch.bool
        or frame_masks.ndim != 3
        or frame_masks.shape[0] != FRAME_COUNT
    ):
        raise NativeBackendError("adjacent-unit projection expects bool [20,H,W]")
    masks = frame_masks.reshape(TEMPORAL_UNITS, 2, *frame_masks.shape[1:]).any(dim=1)
    if any(not bool(mask.any()) or bool(mask.all()) for mask in masks):
        raise NativeBackendError(
            "every 10-unit target mask must be nonempty and nonfull"
        )
    return masks


@dataclass
class NativeVisualBundle:
    family: str
    raw_native_tokens: torch.Tensor
    modified_native_tokens: torch.Tensor
    native_target_mask: torch.Tensor
    native_context_mask: torch.Tensor
    unit10_tokens: torch.Tensor
    unit10_target_mask: torch.Tensor
    unit10_context_mask: torch.Tensor
    target_pooled: torch.Tensor
    context_pooled: torch.Tensor
    residual_reinjected: bool
    physical_output: Any | None = None

    def validate(self, hidden_size: int) -> None:
        expected_prefix = tuple(self.native_target_mask.shape)
        if self.raw_native_tokens.shape[:-1] != expected_prefix:
            raise NativeBackendError("native token/mask shapes differ")
        if self.modified_native_tokens.shape != self.raw_native_tokens.shape:
            raise NativeBackendError("modified native tokens changed shape")
        if self.raw_native_tokens.shape[-1] != hidden_size:
            raise NativeBackendError("native post-projector width drifted")
        if self.unit10_tokens.shape[0] != TEMPORAL_UNITS:
            raise NativeBackendError("visual bundle is not 10 temporal units")
        if self.unit10_target_mask.shape != self.unit10_tokens.shape[:-1]:
            raise NativeBackendError("10-unit target mask/token shapes differ")
        if self.target_pooled.shape != (TEMPORAL_UNITS, hidden_size):
            raise NativeBackendError("target pooled ABI differs from [10,D]")
        if self.context_pooled.shape != (TEMPORAL_UNITS, hidden_size):
            raise NativeBackendError("context pooled ABI differs from [10,D]")
        for tensor in (
            self.raw_native_tokens,
            self.modified_native_tokens,
            self.target_pooled,
            self.context_pooled,
        ):
            if not bool(torch.isfinite(tensor).all()):
                raise NativeBackendError("visual bundle contains non-finite values")


def _pool_target_context(
    unit_tokens: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if unit_tokens.ndim != 4 or unit_tokens.shape[0] != TEMPORAL_UNITS:
        raise NativeBackendError("unit visual tokens must be [10,H,W,D]")
    if (
        target_mask.dtype is not torch.bool
        or target_mask.shape != unit_tokens.shape[:-1]
    ):
        raise NativeBackendError("unit target mask differs from visual grid")
    target_mask = target_mask.to(unit_tokens.device)
    target = torch.stack(
        [unit_tokens[i][target_mask[i]].mean(dim=0) for i in range(TEMPORAL_UNITS)]
    )
    context = torch.stack(
        [unit_tokens[i][~target_mask[i]].mean(dim=0) for i in range(TEMPORAL_UNITS)]
    )
    return target, context


class LowRankResidualAdapter(nn.Module):
    """A classic bottleneck residual adapter, not an unrestricted PEFT LoRA."""

    def __init__(self, hidden_size: int, rank: int, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0 or rank >= hidden_size:
            raise NativeBackendError("adapter rank must be in (0, hidden_size)")
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        # Keep the initial residual negligible while allowing ``down.weight``
        # to receive a live gradient in the mandatory first-batch preflight.
        nn.init.normal_(self.up.weight, mean=0.0, std=1.0e-5)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up(self.dropout(F.silu(self.down(self.norm(value)))))


class PhysicalAdapterPair(nn.Module):
    """A visual-residual ablation only; it is not the TsT-NA Stage-P core.

    It has no Camera18/Object18 heads, so the family backend never instantiates
    it on the primary path.  Keeping the class makes the ablation explicit
    without silently conflating it with trajectory-supervised Stage-P.
    """

    def __init__(self, hidden_size: int, rank: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.camera = LowRankResidualAdapter(hidden_size, rank, dropout)
        self.object = LowRankResidualAdapter(hidden_size, rank, dropout)

    def forward(self, bundle: NativeVisualBundle) -> tuple[torch.Tensor, torch.Tensor]:
        return self.camera(bundle.context_pooled), self.object(bundle.target_pooled)


class SourceReadAdapterPair(nn.Module):
    """Block-output residual adapters; explicitly *not* decoder LoRA.

    This remains available as an ablation/debug primitive.  It must never be
    reported as the Stage-L source-read LoRA because it does not modify a
    native decoder projection's effective weight.
    """

    def __init__(self, hidden_size: int, rank: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.camera = LowRankResidualAdapter(hidden_size, rank, dropout)
        self.object = LowRankResidualAdapter(hidden_size, rank, dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        camera_mask: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.ndim != 3:
            raise NativeBackendError("decoder hidden state must be [B,S,D]")
        if (
            camera_mask.shape != hidden.shape[:2]
            or object_mask.shape != hidden.shape[:2]
        ):
            raise NativeBackendError(
                "source-read token masks do not align with decoder hidden state"
            )
        if camera_mask.dtype is not torch.bool or object_mask.dtype is not torch.bool:
            raise NativeBackendError("source-read masks must be bool")
        if bool((camera_mask & object_mask).any()):
            raise NativeBackendError("Camera/Object source-read supports overlap")
        camera_delta = self.camera(hidden) * camera_mask.unsqueeze(-1).to(hidden)
        object_delta = self.object(hidden) * object_mask.unsqueeze(-1).to(hidden)
        return hidden + camera_delta + object_delta


def resolve_unique_module(model: nn.Module, logical_path: str) -> tuple[str, nn.Module]:
    modules = dict(model.named_modules())
    if logical_path in modules:
        return logical_path, modules[logical_path]
    suffix = f".{logical_path}"
    matches = [
        (name, module) for name, module in modules.items() if name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise NativeBackendError(
            {
                "reason": "module path did not resolve uniquely",
                "logical_path": logical_path,
                "matches": [name for name, _ in matches],
            }
        )
    return matches[0]


def replace_resolved_module(
    model: nn.Module, resolved_path: str, replacement: nn.Module
) -> None:
    """Replace one already-resolved child module without suffix guessing."""
    parent_path, separator, leaf = resolved_path.rpartition(".")
    parent = model if not separator else dict(model.named_modules()).get(parent_path)
    if not isinstance(parent, nn.Module):
        raise NativeBackendError(
            f"module parent disappeared before replacement: {resolved_path}"
        )
    if leaf.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(leaf)] = replacement
    else:
        current = getattr(parent, leaf, None)
        if not isinstance(current, nn.Module):
            raise NativeBackendError(
                f"module leaf disappeared before replacement: {resolved_path}"
            )
        setattr(parent, leaf, replacement)


def discover_live_modules(
    model: nn.Module,
    binding: FamilyBinding,
) -> Mapping[str, Any]:
    decoder_name, decoder = resolve_unique_module(model, binding.decoder_path)
    if not isinstance(decoder, (nn.ModuleList, nn.Sequential)):
        raise NativeBackendError(
            "native decoder container is not an ordered module collection"
        )
    if len(decoder) != binding.decoder_block_count:
        raise NativeBackendError("native decoder block count drifted")
    block_classes = [type(block).__name__ for block in decoder]
    if any(binding.decoder_block_class_fragment not in name for name in block_classes):
        raise NativeBackendError(
            {"unexpected_decoder_block_classes": sorted(set(block_classes))}
        )
    projector_name, projector = resolve_unique_module(
        model, binding.post_projector_path
    )
    return {
        "decoder_resolved_path": decoder_name,
        "decoder_block_count": len(decoder),
        "decoder_block_classes": sorted(set(block_classes)),
        "post_projector_resolved_path": projector_name,
        "post_projector_class": type(projector).__name__,
    }


def ordinary_lora_target_modules(
    model: nn.Module,
    binding: FamilyBinding,
    layer_indices: Sequence[int],
    *,
    include_mlp: bool = False,
) -> tuple[str, ...]:
    """Return exact live Linear paths for the ordinary-LoRA baseline only."""
    leaves = binding.ordinary_lora_attention_leaves
    if include_mlp:
        leaves += binding.ordinary_lora_mlp_leaves
    output = []
    for index in layer_indices:
        block = binding.block_path(int(index))
        for leaf in leaves:
            logical = f"{block}.{leaf}"
            name, module = resolve_unique_module(model, logical)
            if not isinstance(module, nn.Linear):
                raise NativeBackendError(f"ordinary LoRA target is not Linear: {name}")
            output.append(name)
    if len(output) != len(set(output)):
        raise NativeBackendError("ordinary LoRA target discovery produced duplicates")
    return tuple(output)


def source_conditioned_lora_target_modules(
    model: nn.Module,
    binding: FamilyBinding,
    layer_indices: Sequence[int],
) -> tuple[str, ...]:
    """Resolve the exact q/v (or Molmo fused att_proj) Stage-L targets."""
    indices = tuple(map(int, layer_indices))
    if not indices or len(indices) != len(set(indices)):
        raise NativeBackendError(
            "source-conditioned LoRA layers must be nonempty and unique"
        )
    output = []
    for index in indices:
        block = binding.block_path(index)
        for leaf in binding.native_source_lora_attention_leaves:
            name, module = resolve_unique_module(model, f"{block}.{leaf}")
            if not isinstance(module, nn.Linear):
                raise NativeBackendError(
                    {
                        "reason": "native source-conditioned LoRA target is not an unwrapped Linear",
                        "path": name,
                        "observed_class": type(module).__name__,
                    }
                )
            output.append(name)
    if len(output) != len(set(output)):
        raise NativeBackendError(
            "source-conditioned LoRA target discovery produced duplicates"
        )
    return tuple(output)


def freeze_original_base(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise NativeBackendError("original VLM did not freeze completely")


def _is_sandbox_adapter_parameter(name: str) -> bool:
    lora_components = (
        ".camera_A.",
        ".camera_B.",
        ".camera_source_gate.",
        ".object_A.",
        ".object_B.",
        ".object_source_gate.",
    )
    return (
        "._tst_na_unified_core." in name
        or "._tst_na_source_read." in name
        or any(fragment in name for fragment in lora_components)
    )


def _bound_core_and_lora_wrappers(
    model: nn.Module,
) -> tuple[TsTNativeAdapterO18Core | None, dict[str, SourceConditionedDualLoRALinear]]:
    cores = [
        module
        for _, module in model.named_modules()
        if isinstance(module, TsTNativeAdapterO18Core)
    ]
    if len(cores) > 1:
        raise NativeBackendError("model contains more than one TsT-NA physical core")
    wrappers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, SourceConditionedDualLoRALinear)
    }
    if len({id(module) for module in wrappers.values()}) != len(wrappers):
        raise NativeBackendError("one dual-LoRA wrapper occupies multiple module paths")
    return (cores[0] if cores else None), wrappers


def _expected_stage_trainable_names(model: nn.Module, stage: str) -> set[str]:
    core, wrappers = _bound_core_and_lora_wrappers(model)
    normalized = stage.upper()
    if normalized == "P":
        if core is None:
            return set()
        core_path = next(
            name for name, module in model.named_modules() if module is core
        )
        return {
            f"{core_path}.{name}"
            for name in core.expected_trainable_parameter_names("P")
        }
    if normalized == "L":
        return {
            f"{path}.{local_name}"
            for path, wrapper in wrappers.items()
            for local_name, _ in wrapper.adapter_named_parameters()
        }
    if normalized == "BLOCK_ADAPTER":
        return {
            name
            for name, _ in model.named_parameters()
            if "._tst_na_source_read." in name
        }
    if normalized == "FROZEN":
        return set()
    raise NativeBackendError(f"invalid allowlist stage: {stage!r}")


def set_trainable_stage(model: nn.Module, stage: str) -> tuple[str, ...]:
    normalized = stage.upper()
    if normalized not in {"P", "L", "BLOCK_ADAPTER", "FROZEN"}:
        raise NativeBackendError(f"unknown training stage: {stage!r}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    core, wrappers = _bound_core_and_lora_wrappers(model)
    if normalized == "P":
        if core is None:
            raise NativeBackendError(
                "Stage-P requires the real TsTNativeAdapterO18Core"
            )
        configure_stage_p(core, wrappers)
    elif normalized == "L":
        if core is None:
            raise NativeBackendError("Stage-L requires the real frozen physical core")
        configure_stage_l(core, wrappers)
    elif normalized == "BLOCK_ADAPTER":
        if core is not None:
            core.configure_stage("FROZEN")
        for wrapper in wrappers.values():
            wrapper.set_adapter_trainable(False)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_("._tst_na_source_read." in name)
    elif core is not None:
        core.configure_stage("FROZEN")
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if normalized != "FROZEN" and not trainable:
        raise NativeBackendError(
            f"stage {normalized} has no declared trainable parameters"
        )
    audit_trainable_allowlist(model, normalized)
    return tuple(trainable)


def audit_trainable_allowlist(model: nn.Module, stage: str) -> Mapping[str, Any]:
    normalized = stage.upper()
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if normalized not in {"P", "L", "BLOCK_ADAPTER", "FROZEN"}:
        raise NativeBackendError(f"invalid allowlist stage: {stage!r}")
    expected = _expected_stage_trainable_names(model, normalized)
    if set(trainable) != expected:
        raise NativeBackendError(
            {
                "missing_declared_trainable_parameters": sorted(
                    expected - set(trainable)
                ),
                "undeclared_trainable_parameters": sorted(set(trainable) - expected),
            }
        )
    return {
        "stage": normalized,
        "trainable_names": trainable,
        "trainable_count": len(trainable),
        "trainable_numel": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


@dataclass(frozen=True)
class ParameterStamp:
    shape: tuple[int, ...]
    dtype: str
    numel: int
    tensor_version: int
    digest: str


def _sampled_flat_indices(numel: int, *, device: torch.device) -> torch.Tensor:
    """Return 64 exact, endpoint-inclusive indices without float rounding."""
    if type(numel) is not int or numel <= 64 or numel - 1 > torch.iinfo(torch.long).max:
        raise NativeBackendError(
            "sampled tensor size is outside the exact index domain"
        )
    values = [(index * (numel - 1)) // 63 for index in range(64)]
    if (
        values[0] != 0
        or values[-1] != numel - 1
        or any(right <= left for left, right in zip(values, values[1:]))
        or any(not 0 <= value < numel for value in values)
    ):
        raise NativeBackendError("exact sampled tensor indices are invalid")
    return torch.tensor(values, dtype=torch.long, device=device)


def _tensor_digest(parameter: torch.Tensor, mode: str) -> str:
    if mode not in {"sampled", "full"}:
        raise NativeBackendError("tensor hash mode must be sampled or full")
    flat = parameter.detach().reshape(-1)
    if mode == "sampled" and flat.numel() > 64:
        indices = _sampled_flat_indices(flat.numel(), device=flat.device)
        flat = flat.index_select(0, indices)
    bytes_value = flat.cpu().contiguous().view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(bytes_value).hexdigest()


def snapshot_frozen_base(
    model: nn.Module,
    *,
    mode: str = "sampled",
) -> Mapping[str, ParameterStamp]:
    output: dict[str, ParameterStamp] = {}
    for name, parameter in model.named_parameters():
        if _is_sandbox_adapter_parameter(name):
            continue
        if parameter.requires_grad:
            raise NativeBackendError(
                f"base parameter is unexpectedly trainable: {name}"
            )
        output[name] = ParameterStamp(
            shape=tuple(parameter.shape),
            dtype=str(parameter.dtype),
            numel=parameter.numel(),
            tensor_version=int(parameter._version),
            digest=_tensor_digest(parameter, mode),
        )
    if not output:
        raise NativeBackendError("no frozen base parameters were found")
    return output


def assert_frozen_base_unchanged(
    model: nn.Module,
    before: Mapping[str, ParameterStamp],
    *,
    mode: str = "sampled",
) -> Mapping[str, ParameterStamp]:
    after = snapshot_frozen_base(model, mode=mode)
    if set(after) != set(before):
        raise NativeBackendError("frozen base parameter names changed")
    changed = [name for name in before if before[name] != after[name]]
    if changed:
        raise NativeBackendError(
            {"frozen_base_parameters_changed": changed[:32], "count": len(changed)}
        )
    return after


def audit_live_gradients(
    model: nn.Module, *, require_every_trainable: bool
) -> Mapping[str, Any]:
    missing, nonfinite, zero, live = [], [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise NativeBackendError(
                    f"frozen parameter received a gradient: {name}"
                )
            continue
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
        elif not bool(torch.isfinite(gradient).all()):
            nonfinite.append(name)
        elif not bool(torch.count_nonzero(gradient)):
            zero.append(name)
        else:
            live.append(name)
    if nonfinite or (require_every_trainable and (missing or zero)):
        raise NativeBackendError(
            {
                "missing_trainable_gradients": missing,
                "zero_trainable_gradients": zero,
                "nonfinite_trainable_gradients": nonfinite,
            }
        )
    return {"live": live, "missing": missing, "zero": zero, "nonfinite": nonfinite}


@dataclass
class PreparedNativeInputs:
    request: ValidatedRGB20
    model_kwargs: Mapping[str, Any]
    input_ids: torch.Tensor
    prompt_token_count: int
    raw_visual_grid_thw: tuple[int, int, int] | None
    answer_text: str | None
    audit: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class _ExecutionState:
    backend_id: int
    binding: FamilyBinding
    prepared: PreparedNativeInputs
    bundle: NativeVisualBundle | None = None
    visual_hook_calls: int = 0
    residual_reinjections: int = 0
    decoder_prefill_calls: dict[str, int] = field(default_factory=dict)
    decoder_lora_prefill_calls: dict[str, int] = field(default_factory=dict)
    source_context_stack: ExitStack = field(default_factory=ExitStack)


_ACTIVE_EXECUTION: ContextVar[_ExecutionState | None] = ContextVar(
    "tst_na_active_native_execution",
    default=None,
)


def _replace_first_hidden(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    hidden: torch.Tensor,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    copied = dict(kwargs)
    if args and isinstance(args[0], torch.Tensor):
        return (hidden, *args[1:]), copied
    if isinstance(copied.get("hidden_states"), torch.Tensor):
        copied["hidden_states"] = hidden
        return args, copied
    raise NativeBackendError("decoder hook could not locate the hidden-state input")


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value:
        return _first_tensor(value[0])
    if hasattr(value, "last_hidden_state"):
        return _first_tensor(value.last_hidden_state)
    raise NativeBackendError(f"hook output has no tensor: {type(value)!r}")


def _uniform_floating_parameter_spec(
    module: nn.Module,
    *,
    label: str,
) -> tuple[torch.device, torch.dtype]:
    """Return the one device/dtype owned by a numerically isolated module.

    The family-native graph can run in BF16/FP16 while the physical core keeps
    FP32 master parameters.  That is an intentional precision boundary, so it
    must be bridged explicitly rather than relying on implicit promotion or an
    ambient autocast context.
    """
    parameters = tuple(module.parameters())
    if not parameters:
        raise NativeBackendError(f"{label} has no parameters")
    devices = {parameter.device for parameter in parameters}
    dtypes = {parameter.dtype for parameter in parameters}
    if len(devices) != 1 or len(dtypes) != 1:
        raise NativeBackendError(
            {
                "reason": f"{label} parameters do not share one device/dtype",
                "devices": sorted(map(str, devices)),
                "dtypes": sorted(map(str, dtypes)),
            }
        )
    device = next(iter(devices))
    dtype = next(iter(dtypes))
    if not dtype.is_floating_point:
        raise NativeBackendError(f"{label} parameter dtype is not floating point")
    return device, dtype


class NativeVisualBridgeHook:
    def __init__(self, backend: "BaseNativeBackend") -> None:
        self.backend = backend

    def __call__(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: Any,
    ) -> Any:
        del module, args, kwargs
        state = _ACTIVE_EXECUTION.get()
        if state is None or state.backend_id != id(self.backend):
            raise NativeBackendError(
                "native post-projector ran without an active online RGB20 request"
            )
        if state.visual_hook_calls:
            raise NativeBackendError(
                "native post-projector ran more than once for one RGB20 request"
            )
        state.visual_hook_calls += 1
        raw_tokens, restore = self.backend._native_visual_from_hook_output(
            output, state.prepared
        )
        bundle = self.backend._build_visual_bundle(raw_tokens, state.prepared)
        physical = self.backend.unified_physical_core
        modified = raw_tokens
        reinjected = False
        if physical is not None:
            physical_device, physical_dtype = _uniform_floating_parameter_spec(
                physical,
                label="unified physical core",
            )
            target_pooled = bundle.target_pooled.unsqueeze(0).to(
                device=physical_device,
                dtype=physical_dtype,
            )
            context_pooled = bundle.context_pooled.unsqueeze(0).to(
                device=physical_device,
                dtype=physical_dtype,
            )
            physical_output = physical.physical_forward(
                target_pooled,
                context_pooled,
            )
            self.backend._validate_physical_output(physical_output)
            modified = self.backend._reinject_physical_residuals(
                bundle,
                camera_delta=physical_output.camera_source[0],
                object_delta=physical_output.object_source[0],
            )
            bundle.physical_output = physical_output
            reinjected = True
            state.residual_reinjections += 1
            self.backend._bind_source_lora_context(state, physical_output)
        bundle.modified_native_tokens = modified
        bundle.residual_reinjected = reinjected
        bundle.validate(self.backend.binding.hidden_size)
        state.bundle = bundle
        return restore(modified)


class SourceReadBlockHook:
    def __init__(
        self,
        backend: "BaseNativeBackend",
        resolved_path: str,
        pair: SourceReadAdapterPair,
    ) -> None:
        self.backend = backend
        self.resolved_path = resolved_path
        self.pair = pair

    def __call__(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        del module
        state = _ACTIVE_EXECUTION.get()
        if state is None or state.backend_id != id(self.backend):
            raise NativeBackendError(
                "source-read adapter ran without active online RGB20 state"
            )
        hidden = (
            args[0]
            if args and isinstance(args[0], torch.Tensor)
            else kwargs.get("hidden_states")
        )
        if not isinstance(hidden, torch.Tensor):
            raise NativeBackendError("source-read block did not receive hidden states")
        # During cached generation, source visual tokens are present only in the
        # prefill.  Their adapted K/V state is already in the native cache.
        if hidden.shape[1] == 1 and state.decoder_prefill_calls.get(
            self.resolved_path, 0
        ):
            return args, dict(kwargs)
        camera_mask, object_mask = self.backend._decoder_source_masks(state, hidden)
        updated = self.pair(hidden, camera_mask, object_mask)
        state.decoder_prefill_calls[self.resolved_path] = (
            state.decoder_prefill_calls.get(self.resolved_path, 0) + 1
        )
        return _replace_first_hidden(args, kwargs, updated)


class CoreLoRACallAuditHook:
    """Fail closed if a bound core wrapper runs outside an online request."""

    def __init__(self, backend: "BaseNativeBackend", resolved_path: str) -> None:
        self.backend = backend
        self.resolved_path = resolved_path

    def __call__(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> None:
        del module, args, kwargs
        state = _ACTIVE_EXECUTION.get()
        if state is None or state.backend_id != id(self.backend):
            raise NativeBackendError(
                "native dual-LoRA projection ran outside an online RGB20 request"
            )
        state.decoder_lora_prefill_calls[self.resolved_path] = (
            state.decoder_lora_prefill_calls.get(self.resolved_path, 0) + 1
        )


class HiddenStateTap:
    """A removable native-block hook; first prefill only, no implicit detach."""

    def __init__(
        self, model: nn.Module, logical_paths: Sequence[str], *, detach: bool = False
    ) -> None:
        self.model = model
        self.logical_paths = tuple(logical_paths)
        self.detach = detach
        self.handles: list[Any] = []
        self.values: dict[str, torch.Tensor] = {}
        self.call_counts: dict[str, int] = {}

    def __enter__(self) -> "HiddenStateTap":
        if self.handles:
            raise NativeBackendError("hidden-state tap was entered twice")
        for logical in self.logical_paths:
            resolved, module = resolve_unique_module(self.model, logical)

            def hook(
                _module: nn.Module,
                _args: tuple[Any, ...],
                output: Any,
                *,
                name: str = resolved,
            ) -> None:
                tensor = _first_tensor(output)
                self.call_counts[name] = self.call_counts.get(name, 0) + 1
                if (
                    name not in self.values
                    or tensor.shape[1] > self.values[name].shape[1]
                ):
                    self.values[name] = tensor.detach() if self.detach else tensor

            self.handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def require_prefill(self) -> Mapping[str, torch.Tensor]:
        if len(self.values) != len(self.logical_paths):
            raise NativeBackendError(
                {
                    "missing_hidden_hooks": list(self.logical_paths),
                    "captured": list(self.values),
                }
            )
        return dict(self.values)


@dataclass
class NativeForwardResult:
    family: str
    output: Any
    logits: torch.Tensor
    last_hidden_state: torch.Tensor
    block_hidden_states: Mapping[str, torch.Tensor]
    visual: NativeVisualBundle
    online_rgb20: bool = True
    cached_features_used: bool = False


@dataclass(frozen=True)
class TeacherForcedScore:
    family: str
    text: str
    token_count: int
    log_probability_sum: float
    log_probability_mean: float


@dataclass
class TeacherForcedTensorScore:
    """Differentiable native-LM answer score for Stage-L optimization."""

    family: str
    text: str
    answer_token_ids: torch.Tensor
    token_log_probabilities: torch.Tensor
    negative_log_likelihood: torch.Tensor

    def validate(self) -> None:
        if self.answer_token_ids.ndim != 1 or self.answer_token_ids.numel() == 0:
            raise NativeBackendError("differentiable native score has no answer tokens")
        if self.token_log_probabilities.shape != self.answer_token_ids.shape:
            raise NativeBackendError("native answer token/log-prob shapes differ")
        if self.negative_log_likelihood.ndim != 0:
            raise NativeBackendError("native teacher-forced loss must be scalar")
        if not bool(torch.isfinite(self.token_log_probabilities).all()):
            raise NativeBackendError("native token log probabilities are non-finite")
        if not bool(torch.isfinite(self.negative_log_likelihood)):
            raise NativeBackendError("native teacher-forced loss is non-finite")


@dataclass(frozen=True)
class NativeGeneration:
    family: str
    text: str
    generated_token_ids: tuple[int, ...]


@runtime_checkable
class FamilyNativeBackendProtocol(Protocol):
    binding: FamilyBinding

    def freeze_base(self) -> None: ...
    def prepare_rgb20(
        self, request: RGB20Request, *, answer_text: str | None = None
    ) -> PreparedNativeInputs: ...
    def native_forward(
        self, request: RGB20Request, *, capture_layers: Sequence[int] = ()
    ) -> NativeForwardResult: ...
    def attach_unified_physical_core(
        self, core: TsTNativeAdapterO18Core
    ) -> tuple[str, ...]: ...
    def attach_source_conditioned_lora(
        self,
        *,
        layer_indices: Sequence[int],
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> tuple[str, ...]: ...
    def attach_source_read_adapters(
        self, *, layer_indices: Sequence[int], rank: int, dropout: float = 0.0
    ) -> tuple[str, ...]: ...
    def teacher_forced_loss(
        self, request: RGB20Request, candidate_text: str
    ) -> TeacherForcedTensorScore: ...
    def teacher_forced_score(
        self, request: RGB20Request, candidate_text: str
    ) -> TeacherForcedScore: ...
    def generate(
        self, request: RGB20Request, **generation_kwargs: Any
    ) -> NativeGeneration: ...
    def register_hidden_tap(
        self, layer_indices: Sequence[int], *, detach: bool = False
    ) -> ContextManager[HiddenStateTap]: ...


class BaseNativeBackend(ABC):
    """Shared online execution, visual reinjection, scoring, and hook logic."""

    def __init__(
        self,
        *,
        model: nn.Module,
        processor: Any,
        binding: FamilyBinding,
        device: torch.device | str,
        runtime_audit: Mapping[str, Any],
        artifact_audit: Mapping[str, Any],
        video_budget_tier: str = "stock_primary",
        video_budget_reason: str | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.binding = binding
        self.device = torch.device(device)
        self.runtime_audit = dict(runtime_audit)
        self.artifact_audit = dict(artifact_audit)
        if binding.video_pixel_budget_primary is None:
            if video_budget_tier != "stock_primary" or video_budget_reason is not None:
                raise NativeBackendError("video pixel budget tier applies only to Qwen")
            self.video_budget_tier = "family_fixed_native_grid"
            self.video_pixel_budget: int | None = None
            self.video_budget_reason = None
        else:
            budgets = {
                "stock_primary": binding.video_pixel_budget_primary,
                "memory_fallback": binding.video_pixel_budget_memory_fallback,
            }
            if video_budget_tier not in budgets or budgets[video_budget_tier] is None:
                raise NativeBackendError(
                    "Qwen video budget tier must be stock_primary or memory_fallback"
                )
            self.video_budget_tier = video_budget_tier
            self.video_pixel_budget = int(budgets[video_budget_tier])
            if video_budget_tier == "stock_primary":
                if video_budget_reason is not None:
                    raise NativeBackendError(
                        "stock Qwen budget does not accept a fallback reason"
                    )
                self.video_budget_reason = None
            else:
                allowed_reasons = {"preflight_oom", "preflight_memory_limit"}
                if video_budget_reason not in allowed_reasons:
                    raise NativeBackendError(
                        "memory_fallback requires preflight_oom or preflight_memory_limit"
                    )
                self.video_budget_reason = video_budget_reason
        self.unified_physical_core: TsTNativeAdapterO18Core | None = None
        self._visual_handle: Any | None = None
        self._source_handles: list[Any] = []
        self._source_lora_paths: list[str] = []
        self._source_lora_wrappers: dict[str, SourceConditionedDualLoRALinear] = {}
        self._source_lora_handles: list[Any] = []
        self.freeze_base()
        self.live_module_audit = discover_live_modules(self.model, self.binding)
        _, post_projector = resolve_unique_module(
            self.model, self.binding.post_projector_path
        )
        self._visual_handle = post_projector.register_forward_hook(
            NativeVisualBridgeHook(self),
            with_kwargs=True,
        )

    def close(self) -> None:
        if self._visual_handle is not None:
            self._visual_handle.remove()
            self._visual_handle = None
        for handle in self._source_handles:
            handle.remove()
        self._source_handles.clear()
        for handle in self._source_lora_handles:
            handle.remove()
        self._source_lora_handles.clear()

    def freeze_base(self) -> None:
        freeze_original_base(self.model)

    @property
    def physical_residual_scale(self) -> float:
        """Fixed, parameter-free native residual scale."""
        return PHYSICAL_RESIDUAL_SCALE

    def attach_unified_physical_core(
        self,
        core: TsTNativeAdapterO18Core,
    ) -> tuple[str, ...]:
        """Attach the one temporal/O18/native-residual Stage-P module.

        The core is required to use ``physical_dim == feature_dim == family D``.
        Thus its exact trajectory-supervised Camera/Object source is both read
        by the O18 heads and injected into native tokens; no second projection
        or unsupervised residual adapter exists.
        """
        if self.unified_physical_core is not None:
            raise NativeBackendError("a unified physical core is already attached")
        if not isinstance(core, TsTNativeAdapterO18Core):
            raise NativeBackendError(
                "unified physical core must be the shared TsTNativeAdapterO18Core"
            )
        if core.feature_dim != self.binding.hidden_size:
            raise NativeBackendError(
                "physical core feature_dim must equal family native D"
            )
        if core.physical_dim != self.binding.hidden_size:
            raise NativeBackendError(
                "physical_dim must equal family native D; latent-to-native projection is forbidden"
            )
        _, module = resolve_unique_module(self.model, self.binding.post_projector_path)
        if hasattr(module, "_tst_na_unified_core"):
            raise NativeBackendError(
                "post-projector already owns a unified physical core"
            )
        core = core.to(self.device)
        module.add_module("_tst_na_unified_core", core)
        self.unified_physical_core = core
        return set_trainable_stage(self.model, "P")

    def attach_source_read_adapters(
        self,
        *,
        layer_indices: Sequence[int],
        rank: int,
        dropout: float = 0.0,
    ) -> tuple[str, ...]:
        """Attach the block-residual ablation; this is not Stage-L LoRA."""
        if self._source_handles:
            raise NativeBackendError("source-read hooks are already attached")
        indices = tuple(map(int, layer_indices))
        if not indices or len(indices) != len(set(indices)):
            raise NativeBackendError(
                "source-read layer indices must be nonempty and unique"
            )
        for index in indices:
            logical = self.binding.block_path(index)
            resolved, block = resolve_unique_module(self.model, logical)
            if hasattr(block, "_tst_na_source_read"):
                raise NativeBackendError(
                    f"source-read adapter already exists: {resolved}"
                )
            pair = SourceReadAdapterPair(self.binding.hidden_size, rank, dropout).to(
                self.device
            )
            block.add_module("_tst_na_source_read", pair)
            hook = SourceReadBlockHook(self, resolved, pair)
            self._source_handles.append(
                block.register_forward_pre_hook(hook, with_kwargs=True)
            )
        return set_trainable_stage(self.model, "BLOCK_ADAPTER")

    def attach_source_conditioned_lora(
        self,
        *,
        layer_indices: Sequence[int],
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> tuple[str, ...]:
        """Wrap selected native q/v or fused att_proj Linear projections.

        This is the only Stage-L method in this module that may be called
        decoder LoRA.  It modifies the effective native projection while
        leaving the original weight frozen.  The only implementation is the
        shared core ``SourceConditionedDualLoRALinear``; its two rank-space
        gates read the frozen Camera/Object physical sources respectively.
        """
        if self._source_lora_paths:
            raise NativeBackendError(
                "source-conditioned decoder LoRA is already attached"
            )
        if self.unified_physical_core is None:
            raise NativeBackendError(
                "attach the shared physical core before Stage-L LoRA"
            )
        if dropout != 0.0:
            raise NativeBackendError(
                "shared SourceConditionedDualLoRALinear has no dropout; nonzero request rejected"
            )
        targets = source_conditioned_lora_target_modules(
            self.model,
            self.binding,
            layer_indices,
        )
        installed: list[str] = []
        try:
            for resolved in targets:
                _, module = resolve_unique_module(self.model, resolved)
                if not isinstance(module, nn.Linear):
                    raise NativeBackendError(
                        f"LoRA target changed before wrapping: {resolved}"
                    )
                replacement = SourceConditionedDualLoRALinear(
                    module,
                    source_dim=self.unified_physical_core.physical_dim,
                    rank=rank,
                    alpha=alpha,
                )
                replace_resolved_module(self.model, resolved, replacement)
                self._source_lora_wrappers[resolved] = replacement
                self._source_lora_handles.append(
                    replacement.register_forward_pre_hook(
                        CoreLoRACallAuditHook(self, resolved),
                        with_kwargs=True,
                    )
                )
                installed.append(resolved)
        except Exception as error:
            # Partial wrapping changes model structure and cannot be safely
            # guessed back after an arbitrary exception.  Make that state
            # explicit and unusable rather than silently continuing.
            if installed:
                self._source_lora_paths[:] = installed
                raise NativeBackendBlocked(
                    {
                        "reason": "partial source-conditioned LoRA installation; discard backend instance",
                        "installed": installed,
                    }
                ) from error
            raise
        self._source_lora_paths[:] = installed
        return set_trainable_stage(self.model, "L")

    def register_hidden_tap(
        self,
        layer_indices: Sequence[int],
        *,
        detach: bool = False,
    ) -> HiddenStateTap:
        paths = [self.binding.block_path(int(index)) for index in layer_indices]
        return HiddenStateTap(self.model, paths, detach=detach)

    @abstractmethod
    def prepare_rgb20(
        self,
        request: RGB20Request,
        *,
        answer_text: str | None = None,
    ) -> PreparedNativeInputs:
        raise NotImplementedError

    @abstractmethod
    def _native_visual_from_hook_output(
        self,
        output: Any,
        prepared: PreparedNativeInputs,
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], Any]]:
        raise NotImplementedError

    @abstractmethod
    def _decoder_source_masks(
        self,
        state: _ExecutionState,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def _decode_generated(
        self,
        generated: torch.Tensor,
        prepared: PreparedNativeInputs,
    ) -> NativeGeneration:
        raise NotImplementedError

    def _build_visual_bundle(
        self,
        native_tokens: torch.Tensor,
        prepared: PreparedNativeInputs,
    ) -> NativeVisualBundle:
        if native_tokens.ndim != 4:
            raise NativeBackendError("native post-projector tokens must be [T,H,W,D]")
        temporal, height, width, hidden = native_tokens.shape
        if (
            temporal != self.binding.native_visual_temporal
            or hidden != self.binding.hidden_size
        ):
            raise NativeBackendError(
                {
                    "expected_native_visual": [
                        self.binding.native_visual_temporal,
                        "H",
                        "W",
                        self.binding.hidden_size,
                    ],
                    "observed": list(native_tokens.shape),
                }
            )
        if (
            self.binding.native_visual_grid_hw is not None
            and (height, width) != self.binding.native_visual_grid_hw
        ):
            raise NativeBackendError("fixed family-native post-projector grid drifted")
        frame_masks = project_frame_box_masks(
            prepared.request.boxes_xyxy,
            image_width=prepared.request.width,
            image_height=prepared.request.height,
            grid_width=width,
            grid_height=height,
        ).to(native_tokens.device)
        unit_masks = adjacent_unit_masks(frame_masks)
        if temporal == TEMPORAL_UNITS:
            native_target = unit_masks
            unit_tokens = native_tokens
        elif temporal == FRAME_COUNT:
            native_target = frame_masks
            unit_tokens = native_tokens.reshape(
                TEMPORAL_UNITS, 2, height, width, hidden
            ).mean(dim=1)
        else:
            raise NativeBackendError(
                "family visual temporal axis cannot map to 10 adjacent units"
            )
        target, context = _pool_target_context(unit_tokens, unit_masks)
        return NativeVisualBundle(
            family=self.binding.key,
            raw_native_tokens=native_tokens,
            modified_native_tokens=native_tokens,
            native_target_mask=native_target,
            native_context_mask=~native_target,
            unit10_tokens=unit_tokens,
            unit10_target_mask=unit_masks,
            unit10_context_mask=~unit_masks,
            target_pooled=target,
            context_pooled=context,
            residual_reinjected=False,
        )

    def _validate_physical_output(self, output: PhysicalOutput) -> None:
        if not isinstance(output, PhysicalOutput):
            raise NativeBackendError(
                "shared core physical_forward returned the wrong type"
            )
        expected_source = (1, TEMPORAL_UNITS, self.binding.hidden_size)
        if output.camera_source.shape != expected_source:
            raise NativeBackendError("Camera physical source ABI differs from [1,10,D]")
        if output.object_source.shape != expected_source:
            raise NativeBackendError("Object physical source ABI differs from [1,10,D]")
        if output.camera18_physical.shape != (1, 18):
            raise NativeBackendError("Camera18 physical ABI differs from [1,18]")
        if output.object18_physical.shape != (1, 18):
            raise NativeBackendError("Object18 physical ABI differs from [1,18]")
        tensors = (
            output.camera_source,
            output.object_source,
            output.camera18_physical,
            output.object18_physical,
        )
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise NativeBackendError("shared physical core produced non-finite values")

    def _bind_source_lora_context(
        self,
        state: _ExecutionState,
        output: PhysicalOutput,
    ) -> None:
        if not self._source_lora_wrappers:
            return
        if set(self._source_lora_wrappers) != set(self._source_lora_paths):
            raise NativeBackendError("bound Stage-L wrapper registry drifted")
        state.source_context_stack.enter_context(
            dual_lora_source_context(
                self._source_lora_wrappers,
                output.camera_source,
                output.object_source,
            )
        )

    def _reinject_physical_residuals(
        self,
        bundle: NativeVisualBundle,
        *,
        camera_delta: torch.Tensor,
        object_delta: torch.Tensor,
    ) -> torch.Tensor:
        expected = (TEMPORAL_UNITS, self.binding.hidden_size)
        if camera_delta.shape != expected or object_delta.shape != expected:
            raise NativeBackendError("physical residuals must each have shape [10,D]")
        if not bool(torch.isfinite(camera_delta).all()) or not bool(
            torch.isfinite(object_delta).all()
        ):
            raise NativeBackendError("physical residual contains non-finite values")
        temporal = bundle.raw_native_tokens.shape[0]
        native_tokens = bundle.raw_native_tokens
        if temporal == TEMPORAL_UNITS:
            camera_native = camera_delta
            object_native = object_delta
        elif temporal == FRAME_COUNT:
            camera_native = camera_delta.repeat_interleave(2, dim=0)
            object_native = object_delta.repeat_interleave(2, dim=0)
        else:
            raise NativeBackendError(
                "cannot expand physical residual to native temporal grid"
            )
        scale = self.physical_residual_scale
        if not math.isfinite(scale) or not 0.0 < scale <= 0.1:
            raise NativeBackendError(
                "physical residual scale left the frozen [0,0.1] contract"
            )
        # Preserve the family-native ABI after the FP32 physical computation.
        # Implicit FP32 promotion here would feed FP32 activations into a
        # BF16/FP16 native decoder and fail at its next frozen LayerNorm/Linear.
        camera_native = camera_native.to(
            device=native_tokens.device,
            dtype=native_tokens.dtype,
        )
        object_native = object_native.to(
            device=native_tokens.device,
            dtype=native_tokens.dtype,
        )
        camera_grid = (
            scale
            * camera_native[:, None, None, :]
            * bundle.native_context_mask[..., None].to(native_tokens)
        )
        object_grid = (
            scale
            * object_native[:, None, None, :]
            * bundle.native_target_mask[..., None].to(native_tokens)
        )
        modified = native_tokens + camera_grid + object_grid
        if (
            modified.device != native_tokens.device
            or modified.dtype != native_tokens.dtype
        ):
            raise NativeBackendError(
                "physical residual reinjection changed native device/dtype"
            )
        return modified

    def _model_kwargs_to_device(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        def move(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.to(self.device)
            if isinstance(value, list):
                return [move(item) for item in value]
            if isinstance(value, tuple):
                return tuple(move(item) for item in value)
            if isinstance(value, Mapping):
                return {key: move(item) for key, item in value.items()}
            return value

        return {key: move(value) for key, value in kwargs.items()}

    def _generation_model_kwargs(
        self,
        prepared: PreparedNativeInputs,
    ) -> dict[str, Any]:
        return self._model_kwargs_to_device(prepared.model_kwargs)

    @contextmanager
    def _activated(self, prepared: PreparedNativeInputs) -> Iterable[_ExecutionState]:
        state = _ExecutionState(
            backend_id=id(self),
            binding=self.binding,
            prepared=prepared,
        )
        token = _ACTIVE_EXECUTION.set(state)
        try:
            yield state
        finally:
            try:
                state.source_context_stack.close()
            finally:
                _ACTIVE_EXECUTION.reset(token)

    def _require_execution_complete(self, state: _ExecutionState) -> NativeVisualBundle:
        if state.visual_hook_calls != 1 or state.bundle is None:
            raise NativeBackendError(
                "native forward did not traverse the bound post-projector exactly once"
            )
        if self.unified_physical_core is not None and state.residual_reinjections != 1:
            raise NativeBackendError(
                "attached unified physical core did not reinject exactly once"
            )
        if self._source_handles and (
            len(state.decoder_prefill_calls) != len(self._source_handles)
            or any(count != 1 for count in state.decoder_prefill_calls.values())
        ):
            raise NativeBackendError("block-residual adapter prefill count drifted")
        if self._source_lora_paths and (
            set(state.decoder_lora_prefill_calls) != set(self._source_lora_paths)
            or any(count < 1 for count in state.decoder_lora_prefill_calls.values())
        ):
            raise NativeBackendError(
                "source-conditioned decoder-LoRA call coverage drifted"
            )
        return state.bundle

    def native_forward(
        self,
        request: RGB20Request,
        *,
        capture_layers: Sequence[int] = (),
    ) -> NativeForwardResult:
        prepared = self.prepare_rgb20(request)
        kwargs = self._model_kwargs_to_device(prepared.model_kwargs)
        tap = self.register_hidden_tap(capture_layers, detach=False)
        with self._activated(prepared) as state, tap:
            output = self.model(
                **kwargs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        visual = self._require_execution_complete(state)
        logits = getattr(output, "logits", None)
        hidden_states = getattr(output, "hidden_states", None)
        if not isinstance(logits, torch.Tensor) or not hidden_states:
            raise NativeBackendError(
                "native model did not return logits and hidden states"
            )
        last_hidden = hidden_states[-1]
        if not isinstance(last_hidden, torch.Tensor):
            raise NativeBackendError("native final hidden state is not a tensor")
        return NativeForwardResult(
            family=self.binding.key,
            output=output,
            logits=logits,
            last_hidden_state=last_hidden,
            block_hidden_states=tap.require_prefill() if capture_layers else {},
            visual=visual,
        )

    def _teacher_forced_answer_logits_and_ids(
        self,
        *,
        prepared: PreparedNativeInputs,
        shifted_logits: torch.Tensor,
        shifted_ids: torch.Tensor,
        answer_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select answer-aligned logits/IDs for a non-expanding native family."""
        del prepared
        if shifted_logits.ndim != 3 or shifted_ids.ndim != 2:
            raise NativeBackendError(
                "teacher-forced shifted tensors have invalid ranks"
            )
        if shifted_logits.shape[:2] != shifted_ids.shape or shifted_ids.shape[0] != 1:
            raise NativeBackendError("teacher-forced shifted logit/ID lengths differ")
        if answer_mask.ndim != 1 or answer_mask.shape[0] != shifted_ids.shape[1]:
            raise NativeBackendError("teacher-forced answer mask length drifted")
        answer_logits = shifted_logits[0, answer_mask]
        answer_ids = shifted_ids[0, answer_mask]
        if answer_logits.shape[0] != answer_ids.numel() or answer_ids.numel() == 0:
            raise NativeBackendError("teacher-forced answer logit/ID count drifted")
        return answer_logits, answer_ids

    def teacher_forced_loss(
        self,
        request: RGB20Request,
        candidate_text: str,
    ) -> TeacherForcedTensorScore:
        if not isinstance(candidate_text, str) or not candidate_text.strip():
            raise NativeBackendError("teacher-forced candidate text must be nonempty")
        prompt_prepared = self.prepare_rgb20(request)
        prepared = self.prepare_rgb20(request, answer_text=candidate_text.strip())
        prompt_ids = prompt_prepared.input_ids[0]
        full_ids = prepared.input_ids[0]
        prefix = prepared.prompt_token_count
        if prefix != len(prompt_ids) or len(full_ids) <= prefix:
            raise NativeBackendError(
                "teacher-forced prompt/answer token boundary is invalid"
            )
        if not torch.equal(full_ids[:prefix].cpu(), prompt_ids.cpu()):
            raise NativeBackendError(
                "native prompt is not an exact prefix of teacher-forced input"
            )
        kwargs = self._model_kwargs_to_device(prepared.model_kwargs)
        with self._activated(prepared) as state:
            output = self.model(
                **kwargs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        self._require_execution_complete(state)
        logits = output.logits
        ids = prepared.input_ids.to(logits.device)
        shifted_logits = logits[:, :-1, :]
        shifted_ids = ids[:, 1:]
        token_positions = torch.arange(1, ids.shape[1], device=ids.device)
        answer_mask = token_positions >= prefix
        if not bool(answer_mask.any()):
            raise NativeBackendError("teacher-forced answer has no scoreable tokens")
        # Some native multimodal prompts (Molmo2 in particular) contain visual
        # placeholder IDs from an additional embedding table which are outside
        # the original LM-head vocabulary.  They are valid *inputs* but are not
        # valid next-token labels.  Select the answer positions before gather;
        # gathering the whole prompt first would index the LM head with those
        # visual-only IDs even though the subsequent answer mask discards them.
        answer_logits, answer_ids = self._teacher_forced_answer_logits_and_ids(
            prepared=prepared,
            shifted_logits=shifted_logits,
            shifted_ids=shifted_ids,
            answer_mask=answer_mask,
        )
        log_probs = answer_logits.log_softmax(dim=-1)
        selected = log_probs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1)
        if not bool(torch.isfinite(selected).all()):
            raise NativeBackendError("teacher-forced native score is non-finite")
        result = TeacherForcedTensorScore(
            family=self.binding.key,
            text=candidate_text.strip(),
            answer_token_ids=answer_ids,
            token_log_probabilities=selected,
            negative_log_likelihood=-selected.mean(),
        )
        result.validate()
        return result

    def score_tensor(
        self,
        request: RGB20Request,
        candidate_text: str,
    ) -> TeacherForcedTensorScore:
        """Alias whose name makes the differentiable return explicit."""
        return self.teacher_forced_loss(request, candidate_text)

    def teacher_forced_score(
        self,
        request: RGB20Request,
        candidate_text: str,
    ) -> TeacherForcedScore:
        tensor_score = self.teacher_forced_loss(request, candidate_text)
        selected = tensor_score.token_log_probabilities
        return TeacherForcedScore(
            family=self.binding.key,
            text=candidate_text.strip(),
            token_count=int(selected.numel()),
            log_probability_sum=float(selected.sum().detach().cpu()),
            log_probability_mean=float(selected.mean().detach().cpu()),
        )

    def generate(
        self, request: RGB20Request, **generation_kwargs: Any
    ) -> NativeGeneration:
        prepared = self.prepare_rgb20(request)
        kwargs = self._generation_model_kwargs(prepared)
        generation = {"max_new_tokens": 64, "do_sample": False, **generation_kwargs}
        with torch.inference_mode(), self._activated(prepared) as state:
            generated = self.model.generate(**kwargs, **generation)
        self._require_execution_complete(state)
        if (
            not isinstance(generated, torch.Tensor)
            or generated.ndim != 2
            or generated.shape[0] != 1
        ):
            raise NativeBackendError(
                "native generate did not return one token sequence"
            )
        return self._decode_generated(generated, prepared)


class QwenNativeBackend(BaseNativeBackend):
    def prepare_rgb20(
        self,
        request: RGB20Request,
        *,
        answer_text: str | None = None,
    ) -> PreparedNativeInputs:
        item = request.validated()
        from transformers.video_utils import VideoMetadata

        offsets = item.offsets_seconds
        metadata = VideoMetadata(
            total_num_frames=FRAME_COUNT,
            fps=item.sampled_fps,
            width=item.width,
            height=item.height,
            duration=float(offsets[-1]),
            video_backend="tst_na_in_memory_rgb20",
            frames_indices=list(range(FRAME_COUNT)),
        )
        messages: list[Mapping[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": list(item.frames)},
                    {"type": "text", "text": item.prompt},
                ],
            }
        ]
        add_generation_prompt = answer_text is None
        if answer_text is not None:
            messages.append({"role": "assistant", "content": answer_text})
        pixel_budget = getattr(
            self,
            "video_pixel_budget",
            self.binding.video_pixel_budget_primary,
        )
        budget_tier = getattr(self, "video_budget_tier", "stock_primary")
        budget_reason = getattr(self, "video_budget_reason", None)
        shortest_edge = self.binding.video_pixel_budget_shortest_edge
        if pixel_budget is None or shortest_edge is None:
            raise NativeBackendError("Qwen explicit video pixel budget is unbound")
        values = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
            return_dict=True,
            processor_kwargs={
                "do_sample_frames": False,
                "video_metadata": [metadata],
                "videos_kwargs": {
                    "size": {
                        "longest_edge": pixel_budget,
                        "shortest_edge": shortest_edge,
                    },
                },
            },
        )
        if (
            "input_ids" not in values
            or "video_grid_thw" not in values
            or "pixel_values_videos" not in values
        ):
            raise NativeBackendError(
                "Qwen native processor omitted language or video tensors"
            )
        grid = tuple(map(int, values["video_grid_thw"][0].tolist()))
        if grid[0] != TEMPORAL_UNITS or grid[1] % 2 or grid[2] % 2:
            raise NativeBackendError(
                f"Qwen native video grid cannot map to post-projector ABI: {grid}"
            )
        native_grid = (grid[0], grid[1] // 2, grid[2] // 2)
        native_token_count = math.prod(native_grid)
        video_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        observed_video_tokens = int((values["input_ids"] == video_token_id).sum())
        if observed_video_tokens != native_token_count:
            raise NativeBackendError(
                {
                    "reason": "Qwen processor/native visual token count drifted",
                    "grid": grid,
                    "expected_native_tokens": native_token_count,
                    "observed_video_tokens": observed_video_tokens,
                }
            )
        if answer_text is None:
            prompt_count = int(values["input_ids"].shape[1])
        else:
            prompt_count = int(self.prepare_rgb20(request).input_ids.shape[1])
        return PreparedNativeInputs(
            request=item,
            model_kwargs=dict(values),
            input_ids=values["input_ids"],
            prompt_token_count=prompt_count,
            raw_visual_grid_thw=grid,
            answer_text=answer_text,
            audit={
                "processor": type(self.processor).__name__,
                "online_video": True,
                "video_pixel_budget_tier": budget_tier,
                "video_pixel_budget_reason": budget_reason,
                "video_pixel_budget_longest_edge": pixel_budget,
                "video_pixel_budget_shortest_edge": shortest_edge,
                "raw_visual_grid_thw": grid,
                "native_postmerge_grid_thw": native_grid,
                "native_decoder_visual_token_count": native_token_count,
                "input_token_count": int(values["input_ids"].shape[1]),
            },
        )

    def _native_visual_from_hook_output(
        self,
        output: Any,
        prepared: PreparedNativeInputs,
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], Any]]:
        pooler = getattr(output, "pooler_output", None)
        grid = prepared.raw_visual_grid_thw
        if not isinstance(pooler, torch.Tensor) or grid is None:
            raise NativeBackendError("Qwen visual module did not return pooler_output")
        temporal, raw_h, raw_w = grid
        height, width = raw_h // 2, raw_w // 2
        expected = temporal * height * width
        if pooler.ndim != 2 or pooler.shape != (expected, self.binding.hidden_size):
            raise NativeBackendError(
                {"Qwen_pooler_shape": list(pooler.shape), "expected_tokens": expected}
            )
        native = pooler.reshape(temporal, height, width, self.binding.hidden_size)

        def restore(modified: torch.Tensor) -> Any:
            copied = copy(output)
            copied.pooler_output = modified.reshape_as(pooler)
            return copied

        return native, restore

    def _decoder_source_masks(
        self,
        state: _ExecutionState,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.bundle is None:
            raise NativeBackendError(
                "Qwen decoder ran before visual bundle construction"
            )
        ids = state.prepared.input_ids.to(hidden.device)
        video_id = int(self.model.config.video_token_id)
        positions = ids == video_id
        target_flat = state.bundle.native_target_mask.reshape(-1).to(hidden.device)
        context_flat = state.bundle.native_context_mask.reshape(-1).to(hidden.device)
        if hidden.shape[:2] != ids.shape or int(positions.sum()) != target_flat.numel():
            raise NativeBackendError("Qwen video-token/native-grid alignment drifted")
        camera = torch.zeros_like(positions)
        obj = torch.zeros_like(positions)
        camera[positions] = context_flat
        obj[positions] = target_flat
        return camera, obj

    def _decode_generated(
        self,
        generated: torch.Tensor,
        prepared: PreparedNativeInputs,
    ) -> NativeGeneration:
        trimmed = generated[0, prepared.input_ids.shape[1] :]
        text = self.processor.tokenizer.decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return NativeGeneration(
            self.binding.key, text, tuple(map(int, trimmed.tolist()))
        )


class MolmoNativeBackend(BaseNativeBackend):
    def prepare_rgb20(
        self,
        request: RGB20Request,
        *,
        answer_text: str | None = None,
    ) -> PreparedNativeInputs:
        item = request.validated()
        from transformers.video_utils import VideoMetadata

        video = np.stack(item.frames, axis=0)
        metadata = VideoMetadata(
            total_num_frames=FRAME_COUNT,
            fps=item.sampled_fps,
            width=item.width,
            height=item.height,
            duration=float(item.offsets_seconds[-1]),
            video_backend="tst_na_in_memory_rgb20",
            frames_indices=list(range(FRAME_COUNT)),
        )
        messages: list[Mapping[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": item.prompt},
                    {"type": "video", "video": "in_memory_rgb20"},
                ],
            }
        ]
        # Always form the exact native generation prefix first.  Molmo2's
        # shipped chat template inserts a different whitespace token when an
        # assistant message is supplied, so templating the full conversation
        # would make the prompt cease to be an exact teacher-forcing prefix.
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if answer_text is not None:
            text += answer_text.strip()
        values = self.processor(
            text=text,
            videos=video,
            video_metadata=metadata,
            do_sample_frames=False,
            return_tensors="pt",
        )
        required = {
            "input_ids",
            "pixel_values_videos",
            "video_token_pooling",
            "video_grids",
        }
        if not required.issubset(values):
            raise NativeBackendError(
                "Molmo native processor omitted language or video tensors"
            )
        grid = tuple(map(int, values["video_grids"][0].tolist()))
        if grid != (FRAME_COUNT, 9, 9):
            raise NativeBackendError(f"Molmo post-projector grid drifted: {grid}")
        if answer_text is None:
            prompt_count = int(values["input_ids"].shape[1])
        else:
            prompt_count = int(self.prepare_rgb20(request).input_ids.shape[1])
        return PreparedNativeInputs(
            request=item,
            model_kwargs=dict(values),
            input_ids=values["input_ids"],
            prompt_token_count=prompt_count,
            raw_visual_grid_thw=grid,
            answer_text=answer_text,
            audit={"processor": type(self.processor).__name__, "online_video": True},
        )

    def _native_visual_from_hook_output(
        self,
        output: Any,
        prepared: PreparedNativeInputs,
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], Any]]:
        if not isinstance(output, torch.Tensor):
            raise NativeBackendError("Molmo vision_backbone output is not a tensor")
        original_shape = output.shape
        flat = output[0] if output.ndim == 3 and output.shape[0] == 1 else output
        expected = FRAME_COUNT * 9 * 9
        if flat.ndim != 2 or flat.shape != (expected, self.binding.hidden_size):
            raise NativeBackendError({"Molmo_postprojector_shape": list(output.shape)})
        native = flat.reshape(FRAME_COUNT, 9, 9, self.binding.hidden_size)

        def restore(modified: torch.Tensor) -> torch.Tensor:
            flat_modified = modified.reshape(expected, self.binding.hidden_size)
            return (
                flat_modified.unsqueeze(0)
                if len(original_shape) == 3
                else flat_modified
            )

        return native, restore

    def _decoder_source_masks(
        self,
        state: _ExecutionState,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.bundle is None:
            raise NativeBackendError(
                "Molmo decoder ran before visual bundle construction"
            )
        ids = state.prepared.input_ids.to(hidden.device)
        patch_id = int(self.model.config.image_patch_id)
        positions = ids == patch_id
        target_flat = state.bundle.native_target_mask.reshape(-1).to(hidden.device)
        context_flat = state.bundle.native_context_mask.reshape(-1).to(hidden.device)
        if hidden.shape[:2] != ids.shape or int(positions.sum()) != target_flat.numel():
            raise NativeBackendError("Molmo patch-token/native-grid alignment drifted")
        camera = torch.zeros_like(positions)
        obj = torch.zeros_like(positions)
        camera[positions] = context_flat
        obj[positions] = target_flat
        return camera, obj

    def _decode_generated(
        self,
        generated: torch.Tensor,
        prepared: PreparedNativeInputs,
    ) -> NativeGeneration:
        trimmed = generated[0, prepared.input_ids.shape[1] :]
        text = self.processor.tokenizer.decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return NativeGeneration(
            self.binding.key, text, tuple(map(int, trimmed.tolist()))
        )


class NVILANativeBackend(BaseNativeBackend):
    def prepare_rgb20(
        self,
        request: RGB20Request,
        *,
        answer_text: str | None = None,
    ) -> PreparedNativeInputs:
        item = request.validated()
        from PIL import Image
        from llava import conversation as conversation_lib
        from llava.constants import MEDIA_TOKENS
        from llava.mm_utils import process_images
        from llava.utils.tokenizer import tokenize_conversation

        conversation_lib.default_conversation = conversation_lib.conv_templates[
            "qwen-2"
        ].copy()
        messages = [
            {
                "from": "human",
                "value": f"{MEDIA_TOKENS['video']}\n{item.prompt}",
            }
        ]
        add_generation_prompt = answer_text is None
        if answer_text is not None:
            messages.append({"from": "gpt", "value": answer_text})
        input_ids = tokenize_conversation(
            messages,
            self.model.tokenizer,
            add_generation_prompt=add_generation_prompt,
        ).unsqueeze(0)
        pil_frames = [Image.fromarray(frame, mode="RGB") for frame in item.frames]
        pixels = process_images(
            pil_frames,
            self.model.vision_tower.image_processor,
            self.model.config,
        )
        if not isinstance(pixels, torch.Tensor) or pixels.shape != (
            FRAME_COUNT,
            3,
            448,
            448,
        ):
            raise NativeBackendError(
                {"NVILA_processed_RGB20_shape": list(getattr(pixels, "shape", ()))}
            )
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if answer_text is None:
            prompt_count = int(input_ids.shape[1])
        else:
            prompt_count = int(self.prepare_rgb20(request).input_ids.shape[1])
        kwargs = {
            "input_ids": input_ids,
            "media": {"video": [pixels]},
            "media_config": {"video": {}},
            "attention_mask": attention_mask,
            "packing": False,
        }
        return PreparedNativeInputs(
            request=item,
            model_kwargs=kwargs,
            input_ids=input_ids,
            prompt_token_count=prompt_count,
            raw_visual_grid_thw=(FRAME_COUNT, 11, 11),
            answer_text=answer_text,
            audit={"processor": "llava.mm_utils.process_images", "online_video": True},
        )

    def _model_kwargs_to_device(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        moved = super()._model_kwargs_to_device(kwargs)
        try:
            pixels = moved["media"]["video"][0]
        except (KeyError, IndexError, TypeError) as error:
            raise NativeBackendError(
                "NVILA online video tensor structure drifted"
            ) from error
        if not isinstance(pixels, torch.Tensor) or not pixels.is_floating_point():
            raise NativeBackendError("NVILA online video pixels are not floating point")
        try:
            vision_dtype = next(self.model.vision_tower.parameters()).dtype
        except (AttributeError, StopIteration) as error:
            raise NativeBackendError(
                "NVILA vision tower dtype is unavailable"
            ) from error
        moved["media"]["video"][0] = pixels.to(dtype=vision_dtype)
        return moved

    def _generation_model_kwargs(
        self,
        prepared: PreparedNativeInputs,
    ) -> dict[str, Any]:
        kwargs = self._model_kwargs_to_device(prepared.model_kwargs)
        # LlavaMetaForCausalLM.generate consumes media/media_config but forwards
        # all remaining keys directly to llm.generate; ``packing`` is a native
        # forward-only kwarg and would be rejected by GenerationMixin.
        kwargs.pop("packing", None)
        return kwargs

    def _teacher_forced_answer_logits_and_ids(
        self,
        *,
        prepared: PreparedNativeInputs,
        shifted_logits: torch.Tensor,
        shifted_ids: torch.Tensor,
        answer_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align original answer IDs to NVILA's expanded video-token logits."""
        ids = prepared.input_ids
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise NativeBackendError("NVILA teacher forcing requires batch size one")
        full_length = int(ids.shape[1])
        prefix = int(prepared.prompt_token_count)
        if prefix <= 0 or prefix >= full_length:
            raise NativeBackendError(
                "NVILA teacher-forced prompt/answer boundary is invalid"
            )
        if shifted_ids.shape != (1, full_length - 1):
            raise NativeBackendError("NVILA original shifted ID length drifted")
        if not torch.equal(shifted_ids, ids[:, 1:].to(shifted_ids.device)):
            raise NativeBackendError(
                "NVILA shifted IDs do not match the original token sequence"
            )
        expected_answer_mask = (
            torch.arange(1, full_length, device=answer_mask.device) >= prefix
        )
        if answer_mask.ndim != 1 or not torch.equal(answer_mask, expected_answer_mask):
            raise NativeBackendError("NVILA original answer mask contract drifted")

        media_id = int(self.model.tokenizer.media_token_ids["video"])
        locations = torch.nonzero(ids[0] == media_id, as_tuple=False).flatten()
        if locations.numel() != 1:
            raise NativeBackendError(
                "NVILA teacher forcing requires exactly one video placeholder"
            )
        placeholder = int(locations[0])
        if placeholder >= prefix:
            raise NativeBackendError(
                "NVILA video placeholder must occur only in the prompt"
            )
        if bool((ids[0, prefix:] == media_id).any()):
            raise NativeBackendError("NVILA answer suffix contains a media token")

        newline_count = len(self.model.tokenizer("\n").input_ids)
        if newline_count <= 0:
            raise NativeBackendError(
                "NVILA video encoder newline token contract is empty"
            )
        if self.binding.native_visual_grid_hw != (11, 11):
            raise NativeBackendError("NVILA native visual grid contract drifted")
        media_length = FRAME_COUNT * (11 * 11 + newline_count)
        expanded_prompt_length = prefix - 1 + media_length
        expected_expanded_length = full_length - 1 + media_length
        if shifted_logits.ndim != 3 or shifted_logits.shape[0] != 1:
            raise NativeBackendError(
                "NVILA expanded teacher-forced logits have invalid rank"
            )
        if shifted_logits.shape[1] + 1 != expected_expanded_length:
            raise NativeBackendError(
                {
                    "reason": "NVILA expanded teacher-forced logit length drifted",
                    "expected": expected_expanded_length - 1,
                    "observed": int(shifted_logits.shape[1]),
                    "original_length": full_length,
                    "media_length": media_length,
                    "newline_tokens_per_frame": newline_count,
                }
            )

        answer_ids = ids[0, prefix:].to(shifted_logits.device)
        answer_count = int(answer_ids.numel())
        start = expanded_prompt_length - 1
        stop = start + answer_count
        if answer_count <= 0 or start < 0 or stop > shifted_logits.shape[1]:
            raise NativeBackendError(
                "NVILA expanded answer-logit slice is out of bounds"
            )
        answer_logits = shifted_logits[0, start:stop]
        if answer_logits.shape[0] != answer_count:
            raise NativeBackendError("NVILA expanded answer logit/ID count drifted")
        return answer_logits, answer_ids

    def _native_visual_from_hook_output(
        self,
        output: Any,
        prepared: PreparedNativeInputs,
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], Any]]:
        del prepared
        if not isinstance(output, torch.Tensor) or output.shape != (
            FRAME_COUNT,
            11 * 11,
            self.binding.hidden_size,
        ):
            raise NativeBackendError(
                {"NVILA_postprojector_shape": list(getattr(output, "shape", ()))}
            )
        native = output.reshape(FRAME_COUNT, 11, 11, self.binding.hidden_size)

        def restore(modified: torch.Tensor) -> torch.Tensor:
            return modified.reshape_as(output)

        return native, restore

    def _decoder_source_masks(
        self,
        state: _ExecutionState,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.bundle is None:
            raise NativeBackendError(
                "NVILA decoder ran before visual bundle construction"
            )
        ids = state.prepared.input_ids.to(hidden.device)
        media_id = int(self.model.tokenizer.media_token_ids["video"])
        locations = torch.nonzero(ids[0] == media_id, as_tuple=False).flatten()
        if locations.numel() != 1 or ids.shape[0] != 1 or hidden.shape[0] != 1:
            raise NativeBackendError(
                "NVILA requires one video placeholder and batch size one"
            )
        placeholder = int(locations[0])
        newline_count = len(self.model.tokenizer("\n").input_ids)
        if newline_count <= 0:
            raise NativeBackendError(
                "NVILA video encoder newline token contract is empty"
            )
        target_frames = state.bundle.native_target_mask.reshape(FRAME_COUNT, -1).to(
            hidden.device
        )
        context_frames = state.bundle.native_context_mask.reshape(FRAME_COUNT, -1).to(
            hidden.device
        )
        false_suffix = torch.zeros(
            (FRAME_COUNT, newline_count), dtype=torch.bool, device=hidden.device
        )
        target_media = torch.cat((target_frames, false_suffix), dim=1).reshape(-1)
        context_media = torch.cat((context_frames, false_suffix), dim=1).reshape(-1)
        expected_length = ids.shape[1] - 1 + target_media.numel()
        if hidden.shape[1] != expected_length:
            raise NativeBackendError(
                {
                    "reason": "NVILA expanded video-token length drifted",
                    "expected": expected_length,
                    "observed": hidden.shape[1],
                    "newline_tokens_per_frame": newline_count,
                }
            )
        camera = torch.zeros(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        obj = torch.zeros_like(camera)
        camera[0, placeholder : placeholder + context_media.numel()] = context_media
        obj[0, placeholder : placeholder + target_media.numel()] = target_media
        return camera, obj

    def _decode_generated(
        self,
        generated: torch.Tensor,
        prepared: PreparedNativeInputs,
    ) -> NativeGeneration:
        del prepared
        tokens = generated[0]
        text = self.model.tokenizer.decode(
            tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return NativeGeneration(
            self.binding.key, text, tuple(map(int, tokens.tolist()))
        )


def _dtype_from_name(name: str) -> torch.dtype:
    values = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return values[name]
    except KeyError as error:
        raise NativeBackendError(f"unsupported model dtype: {name!r}") from error


def load_local_backend(
    family: str,
    *,
    device: str | torch.device = "cuda:0",
    dtype: str | None = None,
    qwen_video_budget_tier: str = "stock_primary",
    qwen_video_budget_reason: str | None = None,
) -> BaseNativeBackend:
    """Load one complete local VLM.  No downloads and no offline-feature mode."""
    binding = binding_for(family)
    dtype = (
        ("float16" if family == "nvila_lite_8b" else "bfloat16")
        if dtype is None
        else dtype
    )
    if family == "nvila_lite_8b" and dtype != "float16":
        raise NativeBackendError("bound NVILA loader is float16-only")
    if family != "qwen3_vl_8b" and (
        qwen_video_budget_tier != "stock_primary"
        or qwen_video_budget_reason is not None
    ):
        raise NativeBackendError(
            "Qwen video budget tier cannot be set for another family"
        )
    if family == "qwen3_vl_8b":
        if (
            qwen_video_budget_tier == "stock_primary"
            and qwen_video_budget_reason is not None
        ):
            raise NativeBackendError(
                "stock Qwen budget does not accept a fallback reason"
            )
        if (
            qwen_video_budget_tier == "memory_fallback"
            and qwen_video_budget_reason
            not in {
                "preflight_oom",
                "preflight_memory_limit",
            }
        ):
            raise NativeBackendError(
                "Qwen memory fallback requires a predeclared memory/OOM reason"
            )
        if qwen_video_budget_tier not in {"stock_primary", "memory_fallback"}:
            raise NativeBackendError("unknown Qwen video budget tier")
    runtime = require_runtime(binding)
    artifact = discover_local_family(family)
    if not binding.model_root.is_dir():
        raise NativeBackendBlocked(f"local model root absent: {binding.model_root}")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch_dtype = _dtype_from_name(dtype)
    device_value = torch.device(device)

    if family == "qwen3_vl_8b":
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            binding.model_root, local_files_only=True
        )
        model = AutoModelForImageTextToText.from_pretrained(
            binding.model_root,
            dtype=torch_dtype,
            attn_implementation="sdpa",
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(device_value)
        backend_type: type[BaseNativeBackend] = QwenNativeBackend
    elif family == "molmo2_o_7b":
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            binding.model_root,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            binding.model_root,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(device_value)
        backend_type = MolmoNativeBackend
    else:
        if (
            binding.reference_code_root is None
            or not binding.reference_code_root.is_dir()
        ):
            raise NativeBackendBlocked("bound NVILA reference code root is absent")
        resolved_reference_root = binding.reference_code_root.resolve()
        if str(resolved_reference_root) not in sys.path:
            sys.path.insert(0, str(resolved_reference_root))
        os.environ.setdefault("VILA_ATTN_IMPLEMENTATION", "sdpa")
        import llava

        if (
            Path(llava.__file__).resolve()
            != (resolved_reference_root / "llava/__init__.py").resolve()
        ):
            raise NativeBackendBlocked(
                "imported llava package is not the bound local reference"
            )
        model = llava.load(
            str(binding.model_root),
            device=str(device_value),
            device_map={"": str(device_value)},
            attn_implementation="sdpa",
        )
        processor = model.tokenizer
        model.config.geometry_injection = False
        model.config.disable_distillation = True
        model.config.num_video_frames = FRAME_COUNT
        model.config.image_aspect_ratio = "resize"
        forbidden = (
            getattr(model, "geometry_projector", None),
            getattr(model, "distillation", None),
            getattr(model, "encoder_l4p", None),
        )
        if any(value is not None for value in forbidden):
            raise NativeBackendBlocked(
                "NVILA loaded a forbidden external geometry/distillation path"
            )
        backend_type = NVILANativeBackend
    model.train(False)
    return backend_type(
        model=model,
        processor=processor,
        binding=binding,
        device=device_value,
        runtime_audit=runtime,
        artifact_audit=artifact,
        video_budget_tier=qwen_video_budget_tier,
        video_budget_reason=qwen_video_budget_reason,
    )


__all__ = [
    "BaseNativeBackend",
    "FAMILY_BINDINGS",
    "FamilyBinding",
    "FamilyNativeBackendProtocol",
    "HiddenStateTap",
    "LowRankResidualAdapter",
    "MolmoNativeBackend",
    "NVILANativeBackend",
    "NativeBackendBlocked",
    "NativeBackendError",
    "NativeForwardResult",
    "NativeGeneration",
    "NativeVisualBundle",
    "PHYSICAL_RESIDUAL_SCALE",
    "PhysicalOutput",
    "PhysicalAdapterPair",
    "PreparedNativeInputs",
    "QwenNativeBackend",
    "RGB20Request",
    "SourceReadAdapterPair",
    "SourceConditionedDualLoRALinear",
    "TsTNativeAdapterO18Core",
    "TeacherForcedScore",
    "TeacherForcedTensorScore",
    "adjacent_unit_masks",
    "assert_frozen_base_unchanged",
    "audit_live_gradients",
    "audit_runtime",
    "audit_trainable_allowlist",
    "binding_for",
    "discover_all_local_families",
    "discover_live_modules",
    "discover_local_family",
    "freeze_original_base",
    "load_local_backend",
    "ordinary_lora_target_modules",
    "project_frame_box_masks",
    "require_runtime",
    "resolve_unique_module",
    "set_trainable_stage",
    "snapshot_frozen_base",
    "source_conditioned_lora_target_modules",
]


if __name__ == "__main__":
    print(json.dumps(discover_all_local_families(), indent=2, sort_keys=True))
