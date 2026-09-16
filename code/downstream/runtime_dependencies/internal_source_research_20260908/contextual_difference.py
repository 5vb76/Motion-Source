"""Subtract context differences from the object stream without adding weights.

Relative mode preserves the object appearance features and camera stream.
It changes feature differences only; it does not estimate geometric camera
motion. Importing the module leaves existing backends unchanged.
"""

from pathlib import Path
import sys
from types import MethodType

import torch
from torch.nn import functional as F

# LOCAL_PATH: External backend source is prepended and can override repository modules.
EXPERIMENT_DIR = Path(
    "/root/story2_camera_object_motion/experiments/tst_native_adapter_o18_sandbox_v1"
)
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
from mosder_final_v1.core import (
    MoSDeRMotionSourceCore,
    MotionSourceContractError,
    MotionSourceOutput,
    _require_source_input,
)


def normalized_object_inputs(adapter, target, context):
    """Return the existing appearance input and common-mode-rejected delta.

    Kept explicit for checks of the hypothesis, not exposed to the decoder as
    physical labels. There is no trainable new weight or learned coefficient.
    """
    normalized = adapter.input_norm(target)
    context_normalized = adapter.input_norm(context)
    delta = torch.zeros_like(normalized)
    context_delta = torch.zeros_like(context_normalized)
    delta[:, 1:] = normalized[:, 1:] - normalized[:, :-1]
    context_delta[:, 1:] = context_normalized[:, 1:] - context_normalized[:, :-1]
    return normalized, delta - context_delta


def relative_physical_forward(core, target_features, context_features):
    for value, name in (
        (target_features, "target features"),
        (context_features, "context features"),
    ):
        _require_source_input(value, name=name, hidden_size=core.hidden_size)
    if target_features.shape[0] != context_features.shape[0]:
        raise MotionSourceContractError("target/context feature batch sizes differ")
    camera_source = core.camera_source_adapter(context_features)
    adapter = core.object_source_adapter
    normalized, delta = normalized_object_inputs(
        adapter, target_features, context_features
    )
    bottleneck = F.gelu(adapter.down(torch.cat((normalized, delta), dim=-1)))
    latent = adapter.up(bottleneck) + adapter.time_embedding.unsqueeze(0)
    mixed = adapter.temporal_mixer(latent.transpose(1, 2)).transpose(1, 2)
    object_source = adapter.output_norm(latent + F.gelu(mixed))
    if object_source.shape != target_features.shape or not bool(
        torch.isfinite(object_source).all()
    ):
        raise MotionSourceContractError("relative Object output ABI differs")
    return MotionSourceOutput(camera_source=camera_source, object_source=object_source)


class ContextualDifferenceCore(MoSDeRMotionSourceCore):
    """Standalone option with exact original state_dict keys and shapes.

    Explicit mode is required so constructing the class cannot silently select
    a new experimental behavior. Mode metadata must be stored by the runner;
    parameters alone cannot distinguish original and relative execution.
    """

    def __init__(self, hidden_size, *, mode, **kwargs):
        if mode not in ("original", "relative"):
            raise ValueError("mode must be original or relative")
        super().__init__(hidden_size, **kwargs)
        self.contextual_difference_mode = mode

    def physical_forward(self, target_features, context_features):
        if self.contextual_difference_mode == "original":
            return super().physical_forward(target_features, context_features)
        if self.contextual_difference_mode != "relative":
            raise ValueError("invalid contextual difference mode")
        return relative_physical_forward(self, target_features, context_features)


def _bound_physical_forward(core, target_features, context_features):
    if core.contextual_difference_mode == "relative":
        return relative_physical_forward(core, target_features, context_features)
    if core.contextual_difference_mode == "original":
        return core._contextual_difference_original_physical_forward(
            target_features, context_features
        )
    raise ValueError("invalid contextual difference mode")


def install_contextual_difference(backend, enabled=True):
    """Opt in on a fresh successor backend; preserve all parameter objects.

    Repeated calls toggle the mode. Does not edit original source files, reload
    a checkpoint, change trainability, or add parameters/modules. All shared
    aliases to the existing source core therefore observe the same behavior.
    The runner must record enabled/mode separately in its protocol/checkpoint.
    """
    if type(enabled) is not bool:
        raise TypeError("enabled must be bool")
    core = backend.source_core
    if not isinstance(core, MoSDeRMotionSourceCore):
        raise TypeError("backend.source_core must be a native MoSDeR source core")
    if isinstance(core, ContextualDifferenceCore):
        core.contextual_difference_mode = "relative" if enabled else "original"
        return core
    if not hasattr(core, "_contextual_difference_original_physical_forward"):
        core._contextual_difference_original_physical_forward = core.physical_forward
        core.physical_forward = MethodType(_bound_physical_forward, core)
    core.contextual_difference_mode = "relative" if enabled else "original"
    return core
