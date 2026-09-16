"""Same-parent incremental ordinary-LoRA control, isolated from MoSDeR-v1.

Parent and grounder parameters stay frozen. New residuals are applied on
all three explicit FactorRoutes, so attribution and QA see the same control.
This tests WHERE an incremental update helps; it does not establish MoSDeR's
source structure is superior to an ordinary model trained from raw weights.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import weakref

import torch
from torch import nn
from torch.nn import functional as F

# LOCAL_PATH: External backend source is prepended and can override repository modules.
EXPERIMENT_DIR = Path(
    "/root/story2_camera_object_motion/experiments/tst_native_adapter_o18_sandbox_v1"
)
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
from family_backends_v1 import resolve_unique_module
from mosder_final_v1.contract import FACTORSPAN_LAYERS
from mosder_final_v1.routing import FactorRoute, SourceConditionedTriLoRALinear


class OrdinaryUpdateResidual(nn.Module):
    """FP32 ordinary BA residual with scale=1, no source/query gate.

    A is random, B is exactly zero. Thus initial predictions equal the frozen
    parent. The first loss has zero A-gradient by construction and nonzero
    B-gradient; A receives gradients after B becomes nonzero.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, *, seed: int):
        super().__init__()
        if min(in_features, out_features, rank) <= 0 or rank >= min(
            in_features, out_features
        ):
            raise ValueError("invalid ordinary residual dimensions")
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        with torch.random.fork_rng(devices=[]):
            self.A = nn.Linear(in_features, rank, bias=False, dtype=torch.float32)
            self.B = nn.Linear(rank, out_features, bias=False, dtype=torch.float32)
        self.reset_parameters(seed=seed)

    def reset_parameters(self, *, seed: int):
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        initial = torch.empty(self.rank, self.in_features, dtype=torch.float32)
        initial.uniform_(
            -1 / math.sqrt(self.in_features),
            1 / math.sqrt(self.in_features),
            generator=generator,
        )
        with torch.no_grad():
            self.A.weight.copy_(initial.to(self.A.weight.device))
            self.B.weight.zero_()
        self.zero_grad(set_to_none=True)

    def forward(self, hidden: torch.Tensor):
        if hidden.shape[-1] != self.in_features or not hidden.is_floating_point():
            raise ValueError("ordinary-update hidden ABI mismatch")
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return F.linear(F.linear(hidden.float(), self.A.weight), self.B.weight)


@dataclass(frozen=True)
class UpdateTarget:
    path: str
    rank: int
    in_features: int
    out_features: int
    parent_kind: str


class SameParentOrdinaryUpdate:
    """Owns hook lifecycle and independent checkpoint state, not parent weights."""

    def __init__(self, backend, targets: tuple[UpdateTarget, ...], *, seed: int):
        if not targets or len({x.path for x in targets}) != len(targets):
            raise ValueError("nonempty unique update targets required")
        self.backend = weakref.ref(backend)
        self.targets, self.seed = targets, int(seed)
        self.adapters, self.handles = {}, []
        self.calls = Counter()
        self.parent_parameters = tuple(backend.model.named_parameters())
        # Preflight all targets before mutation.
        live = {}
        for target in targets:
            resolved, module = resolve_unique_module(backend.model, target.path)
            if resolved != target.path or hasattr(module, "_successor_ordinary_update"):
                raise ValueError("target path drifted or already owns update")
            base = (
                module.base
                if isinstance(module, SourceConditionedTriLoRALinear)
                else module
            )
            if not isinstance(base, nn.Linear) or (
                base.in_features,
                base.out_features,
            ) != (target.in_features, target.out_features):
                raise ValueError("projection shape differs from declared control")
            live[target.path] = module
        for _, parameter in self.parent_parameters:
            parameter.requires_grad_(False)
            parameter.grad = None
        try:
            for index, target in enumerate(targets):
                module = live[target.path]
                base = (
                    module.base
                    if isinstance(module, SourceConditionedTriLoRALinear)
                    else module
                )
                adapter = OrdinaryUpdateResidual(
                    target.in_features,
                    target.out_features,
                    target.rank,
                    seed=self.seed + index,
                )
                adapter.to(device=base.weight.device, dtype=torch.float32)
                module.add_module("_successor_ordinary_update", adapter)
                self.adapters[target.path] = adapter
                self.handles.append(
                    module.register_forward_hook(
                        self._hook_for(target.path), with_kwargs=True
                    )
                )
            self.configure_trainable()
        except Exception:
            self.close()
            raise

    def _hook_for(self, path):
        def hook(module, args, kwargs, output):
            del module
            backend = self.backend()
            if backend is None:
                raise RuntimeError("control backend was destroyed")
            route = backend.active_factor_route
            if route not in tuple(FactorRoute):
                raise RuntimeError("ordinary update requires an explicit valid route")
            hidden = args[0] if args else kwargs.get("hidden", kwargs.get("input"))
            if not isinstance(hidden, torch.Tensor) or not isinstance(
                output, torch.Tensor
            ):
                raise ValueError("projection hook requires tensor input/output")
            delta = self.adapters[path](hidden).to(output)
            if delta.shape != output.shape or not bool(torch.isfinite(delta).all()):
                raise ValueError("ordinary update output ABI differs")
            self.calls[(path, route.value)] += 1
            return output + delta

        return hook

    def configure_trainable(self):
        backend = self.backend()
        if backend is None:
            raise RuntimeError("backend unavailable")
        for _, parameter in self.parent_parameters:
            parameter.requires_grad_(False)
            parameter.grad = None
        for adapter in self.adapters.values():
            adapter.requires_grad_(True)
        expected = {id(parameter) for _, parameter in self.named_parameters()}
        actual = {
            id(parameter)
            for parameter in backend.model.parameters()
            if parameter.requires_grad
        }
        if actual != expected:
            raise RuntimeError("ordinary update trainable allowlist differs")
        return tuple(self.named_parameters())

    def named_parameters(self):
        for path, adapter in self.adapters.items():
            for name, parameter in adapter.named_parameters():
                yield f"{path}.{name}", parameter

    def parameters(self):
        return (parameter for _, parameter in self.named_parameters())

    def state_dict(self, *, cpu: bool = True):
        return {
            name: (parameter.detach().cpu() if cpu else parameter.detach()).clone()
            for name, parameter in self.named_parameters()
        }

    def load_state_dict(self, state):
        expected = dict(self.named_parameters())
        if set(state) != set(expected):
            raise ValueError("ordinary update checkpoint keys differ")
        for name, parameter in expected.items():
            value = state[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != parameter.shape
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"invalid control checkpoint tensor: {name}")
        with torch.no_grad():
            for name, parameter in expected.items():
                parameter.copy_(state[name].to(parameter))

    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self.seed = int(seed)
        for index, adapter in enumerate(self.adapters.values()):
            adapter.reset_parameters(seed=self.seed + index)
        self.calls.clear()

    def assert_parent_frozen(self):
        bad = [
            name
            for name, parameter in self.parent_parameters
            if parameter.requires_grad or parameter.grad is not None
        ]
        if bad:
            raise RuntimeError(f"frozen parent gained gradient/trainability: {bad[:5]}")
        return True

    def inventory(self):
        return {
            "seed": self.seed,
            "ordinary_trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters()
            ),
            "ordinary_trainable_tensors": sum(1 for _ in self.parameters()),
            "frozen_parent_parameters": sum(
                parameter.numel() for _, parameter in self.parent_parameters
            ),
            "targets": [vars(target) for target in self.targets],
            "scale": 1.0,
            "initial_B_exact_zero": all(
                not bool(torch.count_nonzero(a.B.weight))
                for a in self.adapters.values()
            ),
            "applied_routes": [r.value for r in FactorRoute],
            "call_counts": {
                f"{path}::{route}": count for (path, route), count in self.calls.items()
            },
            "claim_scope": "same-parent incremental update location only; ordinary arm adds total model capacity",
        }

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        backend = self.backend()
        if backend is not None:
            for path in tuple(self.adapters):
                _, module = resolve_unique_module(backend.model, path)
                if (
                    getattr(module, "_successor_ordinary_update", None)
                    is self.adapters[path]
                ):
                    delattr(module, "_successor_ordinary_update")
        self.adapters.clear()


def install_same_parent_ordinary_update(
    backend, *, seed: int = 0
) -> SameParentOrdinaryUpdate:
    """Molmo: four existing TriLoRA att_proj + four raw attn_out projections.

    att_proj: 4096 -> 12288, rank 30 in layers 27/28/29/30.
    attn_out: 4096 -> 4096, ranks 33/32/32/31 respectively.
    Total: 3,014,656 extra trainables; MoSDeR update arm has 3,014,944.
    """
    if backend.binding.key != "molmo2_o_7b":
        raise ValueError("published matched budget applies only to local Molmo")
    backend._require_complete()
    targets = []
    expected_wrappers = set()
    for layer, out_rank in zip(
        FACTORSPAN_LAYERS["molmo2_o_7b"], (33, 32, 32, 31), strict=True
    ):
        for leaf, rank, out_size, kind in (
            ("self_attn.att_proj", 30, 12288, "frozen_parent_TriLoRA"),
            ("self_attn.attn_out", out_rank, 4096, "frozen_native_Linear"),
        ):
            path, module = resolve_unique_module(
                backend.model, backend.binding.block_path(layer) + "." + leaf
            )
            if kind == "frozen_parent_TriLoRA":
                if (
                    not isinstance(module, SourceConditionedTriLoRALinear)
                    or backend._mosder_wrappers.get(path) is not module
                ):
                    raise ValueError(
                        "att_proj must be the declared frozen parent TriLoRA"
                    )
                expected_wrappers.add(path)
            elif not isinstance(module, nn.Linear):
                raise ValueError("attn_out must remain an original native Linear")
            targets.append(UpdateTarget(path, rank, 4096, out_size, kind))
    if expected_wrappers != set(backend._mosder_wrappers):
        raise ValueError("parent TriLoRA inventory differs from four att_proj targets")
    total = sum(x.rank * (x.in_features + x.out_features) for x in targets)
    if total != 3014656:
        raise ValueError("ordinary update budget drifted")
    return SameParentOrdinaryUpdate(backend, tuple(targets), seed=seed)
