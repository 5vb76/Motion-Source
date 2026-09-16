"""Local runtime-only interventions. No original source file is modified."""

import types
from dataclasses import replace

import torch


def constant_source_gate(self, source, projection):
    """Keep the gate projection while removing dependence on video content."""
    # Same input vector for all videos. Retain matrix dimensions and initialization.
    # This matches nominal count, not functional capacity of video conditioning.
    with torch.autocast(device_type=source.device.type, enabled=False):
        feature_count = source.shape[-1]
        fixed = torch.arange(
            1, feature_count + 1, device=source.device, dtype=torch.float32
        )
        fixed = fixed - fixed.mean()
        fixed = fixed / fixed.square().mean().sqrt()
        summary = fixed.unsqueeze(0).expand(source.shape[0], -1)
        gate = torch.tanh(
            torch.nn.functional.linear(summary, projection.weight.float(), None)
        )
    return gate.to(projection.weight.dtype)


def global_bundle(original, native_tokens, prepared):
    """Give the target and context branches the same global visual average."""
    bundle = original(native_tokens, prepared)
    global_pool = bundle.unit10_tokens.mean(dim=(1, 2))
    return replace(bundle, target_pooled=global_pool, context_pooled=global_pool)


def apply_intervention(backend, arm):
    """Install the selected inference intervention without changing parameters."""
    if arm == "full":
        return
    if arm == "global_evidence":
        original = backend._build_visual_bundle

        def build(self, native_tokens, prepared):
            return global_bundle(original, native_tokens, prepared)

        backend._build_visual_bundle = types.MethodType(build, backend)
    elif arm == "constant_gate":
        assert backend._mosder_wrappers
        for wrapper in backend._mosder_wrappers.values():
            wrapper._source_gate = types.MethodType(constant_source_gate, wrapper)
    else:
        raise ValueError(arm)
