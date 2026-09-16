"""Connect soft query routing to the native MoSDeR backend hooks.

Questions select spatial weights without input boxes. The backend keeps the
native visual tensor layout and reinjects camera/object residuals at each grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import numpy as np
import torch

# LOCAL_PATH: External backend source is prepended and can override repository modules.
EXPERIMENT_DIR = Path(
    "/root/story2_camera_object_motion/experiments/tst_native_adapter_o18_sandbox_v1"
)
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
from family_backends_v1 import NativeVisualBundle, _as_rgb_array, resolve_unique_module
from mosder_final_v1.contract import BASE_SOURCE_RESIDUAL_SCALE
from mosder_final_v1.family_backend import MolmoMoSDeRBackend
from mosder_final_v1.routing import FactorRoute
from soft_query_router import SoftQueryRouter


def question_stem_from_prompt(prompt: str) -> str:
    """Conservative support for the local newline-delimited A/B/C/D format.

    Fail closed for inline answer options. This function does not identify a
    unique object in a multi-referent question; such mappings require a data
    eligibility audit. No answer text is an argument.
    """
    stem = re.split(
        r"(?im)^\s*(?:options?|choices?)\s*:|^\s*(?:[A-H][.)]|\([A-H]\))\s+",
        prompt,
        maxsplit=1,
    )[0].strip()
    stem = re.sub(r"(?is)\s*Only reply with the best option\.?\s*$", "", stem).strip()
    if re.search(r"\s(?:[A-D][.)]|\([A-D]\))\s+\S", stem):
        raise ValueError("inline options require explicit audited question_stem")
    if not stem:
        raise ValueError("empty question stem")
    return stem


@dataclass(frozen=True)
class VideoQuery20Request:
    """Twenty RGB frames and a question stem used for spatial grounding."""

    frames: Sequence[Any]
    timestamps_ns: Sequence[int]
    prompt: str
    request_id: str = "unpromoted_soft_query_successor"
    # Must be a prefix of the prompt. This prevents adding GT target names.
    question_stem: str | None = None

    def validated(self):
        if len(self.frames) != 20:
            raise ValueError("exactly twenty original RGB frames are required")
        frames = tuple(_as_rgb_array(x) for x in self.frames)
        if any(x.shape != frames[0].shape for x in frames):
            raise ValueError("frame dimensions differ")
        stamps = tuple(self.timestamps_ns)
        if (
            len(stamps) != 20
            or any(type(x) is not int for x in stamps)
            or any(b <= a for a, b in zip(stamps, stamps[1:]))
        ):
            raise ValueError("timestamps must be twenty increasing Python integers")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("nonempty native prompt required")
        prompt = self.prompt.strip()
        stem = (
            self.question_stem.strip()
            if self.question_stem is not None
            else question_stem_from_prompt(prompt)
        )
        if not stem or not prompt.startswith(stem):
            raise ValueError("grounding text must be the original question prefix")
        if question_stem_from_prompt(stem) != stem:
            raise ValueError("grounding text still contains options/instructions")
        return ValidatedVideoQuery20(
            frames,
            stamps,
            prompt,
            str(self.request_id),
            frames[0].shape[0],
            frames[0].shape[1],
            stem,
        )


@dataclass(frozen=True)
class ValidatedVideoQuery20:
    """Validated frame sequence, timing, prompt, and grounding text."""

    frames: tuple[np.ndarray, ...]
    timestamps_ns: tuple[int, ...]
    prompt: str
    request_id: str
    height: int
    width: int
    question_stem: str

    @property
    def offsets_seconds(self):
        return np.asarray(
            [(x - self.timestamps_ns[0]) / 1e9 for x in self.timestamps_ns],
            dtype=np.float64,
        )

    @property
    def sampled_fps(self):
        return float(
            1e9 / np.median(np.diff(np.asarray(self.timestamps_ns, dtype=np.int64)))
        )


def frozen_query_embedding(model, tokenizer, stem: str) -> torch.Tensor:
    """Mean native input embeddings, no visual placeholders or answer tokens.

    This cheap initialization is not an assertion that embedding similarity
    already grounds objects. Query-token order is discarded by mean pooling.
    """
    embedding = model.get_input_embeddings()
    if any(parameter.requires_grad for parameter in embedding.parameters()):
        raise ValueError("query embedding must use frozen W0")
    values = tokenizer(stem, add_special_tokens=False, return_tensors="pt")
    token_ids = values["input_ids"]
    if token_ids.ndim != 2 or token_ids.shape[0] != 1 or token_ids.numel() == 0:
        raise ValueError("invalid stem tokens")
    device = next(embedding.parameters()).device
    with torch.no_grad():
        result = embedding(token_ids.to(device))[0].float().mean(0)
    return result.detach()


class SoftQueryBackendMixin:
    """Override only spatial pooling and spatial residual assignment."""

    def attach_query_router(self, router: SoftQueryRouter, *, mode: str = "query"):
        if hasattr(self, "query_router"):
            raise ValueError("query router already attached")
        if router.hidden_size != self.binding.hidden_size or mode not in (
            "query",
            "uniform",
        ):
            raise ValueError("router width/mode mismatch")
        _, owner = resolve_unique_module(self.model, self.binding.post_projector_path)
        # The registered name is part of existing state_dict keys.
        owner.add_module(
            "_successor_query_router", router.to(self.device, dtype=torch.float32)
        )
        self.query_router, self.query_router_mode = router, mode
        self.last_soft_routing = None

    def _grounding_embedding(self, request: ValidatedVideoQuery20):
        return frozen_query_embedding(
            self.model, self.processor.tokenizer, request.question_stem
        )

    def _build_visual_bundle(self, native_tokens, prepared):
        if not isinstance(prepared.request, ValidatedVideoQuery20):
            raise ValueError("successor accepts only box-free VideoQuery20Request")
        if (
            native_tokens.shape[0] != self.binding.native_visual_temporal
            or native_tokens.shape[-1] != self.binding.hidden_size
        ):
            raise ValueError("native temporal/hidden ABI drifted")
        if (
            self.binding.native_visual_grid_hw is not None
            and tuple(native_tokens.shape[1:3]) != self.binding.native_visual_grid_hw
        ):
            raise ValueError("native grid ABI drifted")
        soft = self.query_router(
            native_tokens,
            self._grounding_embedding(prepared.request),
            mode=self.query_router_mode,
        )
        self.last_soft_routing = soft
        return NativeVisualBundle(
            family=self.binding.key,
            raw_native_tokens=native_tokens,
            modified_native_tokens=native_tokens,
            native_target_mask=soft.target_weights,
            native_context_mask=soft.context_weights,
            unit10_tokens=soft.unit_tokens,
            unit10_target_mask=soft.unit_target_weights,
            unit10_context_mask=1 - soft.unit_target_weights,
            target_pooled=soft.target_pooled,
            context_pooled=soft.context_pooled,
            residual_reinjected=False,
        )

    def _reinject_physical_residuals(self, bundle, *, camera_delta, object_delta):
        native_tokens, route = bundle.raw_native_tokens, self.active_factor_route
        if (
            camera_delta.shape != (10, self.binding.hidden_size)
            or object_delta.shape != camera_delta.shape
        ):
            raise ValueError("source ABI must be [10,D]")

        def broadcast_residual(source, weight):
            source = (
                source
                if native_tokens.shape[0] == 10
                else source.repeat_interleave(2, 0)
            )
            return (
                source.to(native_tokens)[:, None, None, :]
                * weight.to(native_tokens)[..., None]
            )

        result = native_tokens
        if route in (FactorRoute.CAMERA_FACTOR, FactorRoute.FULL_LANGUAGE):
            result = result + BASE_SOURCE_RESIDUAL_SCALE * broadcast_residual(
                camera_delta, bundle.native_context_mask
            )
        if route in (FactorRoute.OBJECT_FACTOR, FactorRoute.FULL_LANGUAGE):
            result = result + BASE_SOURCE_RESIDUAL_SCALE * broadcast_residual(
                object_delta, bundle.native_target_mask
            )
        if route is FactorRoute.FULL_LANGUAGE:
            # R residuals detach their source input. QA also uses the direct
            # 0.05 source path above and source-conditioned TriLoRA.
            camera_residual = self.explicit_residual.camera(camera_delta[None])[0]
            object_residual = self.explicit_residual.object(object_delta[None])[0]
            result = (
                result
                + broadcast_residual(camera_residual, bundle.native_context_mask)
                + broadcast_residual(object_residual, bundle.native_target_mask)
            )
            self._mosder_explicit_call_counts["camera"] += 1
            self._mosder_explicit_call_counts["object"] += 1
        if (
            result.shape != native_tokens.shape
            or result.dtype != native_tokens.dtype
            or not bool(torch.isfinite(result).all())
        ):
            raise ValueError("soft reinjection changed native ABI")
        return result

    def _decoder_source_masks(self, state, hidden):
        # Boolean block adapters would discard the soft routing weights.
        raise RuntimeError(
            "soft successor supports TriLoRA, not hard-mask block adapters"
        )


class SoftQueryMolmoBackend(SoftQueryBackendMixin, MolmoMoSDeRBackend):
    """Molmo backend using query-weighted target and context pooling."""


def assemble_soft_query_backend(
    fresh_base, router: SoftQueryRouter, *, mode: str = "query"
):
    """Reassemble a fresh raw Molmo backend; load frozen parent weights later."""
    if fresh_base.binding.key != "molmo2_o_7b":
        raise ValueError("only Molmo's candidate class is currently provided")
    if fresh_base.unified_physical_core is not None or fresh_base._source_lora_paths:
        raise ValueError("fresh raw backend required; no mutation of existing parent")
    values = {
        key: getattr(fresh_base, key)
        for key in (
            "model",
            "processor",
            "binding",
            "device",
            "runtime_audit",
            "artifact_audit",
        )
    }
    fresh_base.close()
    backend = SoftQueryMolmoBackend(**values)
    try:
        backend.attach_mosder_plugin()
        backend.attach_mosder_decoder()
        backend.attach_query_router(router, mode=mode)
    except Exception:
        backend.close()
        raise
    return backend
