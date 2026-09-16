"""Zero source-owned outputs at inference; no parameter edits or gate substitution."""

import torch


def apply_intervention(backend, arm):
    """Install the selected inference intervention without changing parameters."""
    if arm == "full":
        return
    owner = {"no_camera": "camera", "no_object": "object"}[arm]
    backend._branch_knockout_stats = {}
    backend._branch_knockout_handles = []

    def make_zero_output_hook(name):
        def hook(module, inputs, output):
            assert isinstance(output, torch.Tensor)
            stats = backend._branch_knockout_stats.setdefault(
                name, {"calls": 0, "max_abs_after": 0.0}
            )
            stats["calls"] += 1
            return torch.zeros_like(output)

        return hook

    modules = [
        ("source_adapter", getattr(backend.source_core, owner + "_source_adapter")),
        ("explicit_residual", getattr(backend.explicit_residual, owner)),
    ]
    modules += [
        ("source_lora:" + name, getattr(wrapper, owner + "_B"))
        for name, wrapper in backend._mosder_wrappers.items()
    ]
    for name, module in modules:
        backend._branch_knockout_handles.append(
            module.register_forward_hook(make_zero_output_hook(name))
        )
