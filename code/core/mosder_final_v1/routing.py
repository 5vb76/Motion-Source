"""Camera, object, and shared LoRA branches with four-state factor decoding."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterator, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .contract import (
    STATE_ORDER,
    TEMPORAL_UNITS,
    TRILORA_ALPHA,
    TRILORA_RANK,
)


SOURCE_GATE_DTYPE = torch.float32


class RoutingContractError(RuntimeError):
    """A route, source, parameter-owner, or decision invariant failed."""


class FactorRoute(str, Enum):
    CAMERA_FACTOR = "CAMERA_FACTOR"
    OBJECT_FACTOR = "OBJECT_FACTOR"
    FULL_LANGUAGE = "FULL_LANGUAGE"


def normalize_route(route: FactorRoute | str) -> FactorRoute:
    if isinstance(route, FactorRoute):
        return route
    if not isinstance(route, str):
        raise TypeError("route must be a FactorRoute or string")
    normalized = route.strip().upper().replace("-", "_")
    try:
        return FactorRoute(normalized)
    except ValueError as error:
        raise RoutingContractError(f"unknown MoSDeR route: {route!r}") from error


def _require_source(
    source: torch.Tensor | None,
    *,
    name: str,
    source_dim: int,
) -> torch.Tensor:
    if not isinstance(source, torch.Tensor) or not source.is_floating_point():
        raise RoutingContractError(f"{name} must be a floating tensor")
    if source.ndim != 3 or tuple(source.shape[1:]) != (
        TEMPORAL_UNITS,
        source_dim,
    ):
        raise RoutingContractError(
            f"{name} must have shape [B,{TEMPORAL_UNITS},{source_dim}]"
        )
    if not bool(torch.isfinite(source).all()):
        raise RoutingContractError(f"{name} contains non-finite values")
    return source


@dataclass(frozen=True, slots=True)
class TriLoRAOutput:
    route: FactorRoute
    base_output: torch.Tensor
    camera_residual: torch.Tensor | None
    object_residual: torch.Tensor | None
    shared_residual: torch.Tensor | None
    output: torch.Tensor


class SourceConditionedTriLoRALinear(nn.Module):
    """Frozen native Linear plus Camera/Object/Shared low-rank updates.

    Each factor route uses its corresponding source and LoRA branch. Source
    tensors keep their gradients in Stage F, allowing the factor loss to train
    the temporal adapters. Stages G/R freeze those adapters.
    """

    OWNER_PREFIXES = {
        "camera": ("camera_A.", "camera_B.", "camera_source_gate."),
        "object": ("object_A.", "object_B.", "object_source_gate."),
        "shared": ("shared_A.", "shared_B."),
    }

    def __init__(
        self,
        base: nn.Linear,
        *,
        source_dim: int,
        rank: int = TRILORA_RANK,
        alpha: float = TRILORA_ALPHA,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be an nn.Linear")
        if type(source_dim) is not int or source_dim <= 0:
            raise ValueError("source_dim must be positive")
        if type(rank) is not int or not 0 < rank < min(
            base.in_features, base.out_features
        ):
            raise ValueError("rank is outside the native Linear dimensions")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be positive and finite")
        self.base = base
        self.source_dim = source_dim
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)

        self.camera_A = nn.Linear(base.in_features, rank, bias=False)
        self.camera_B = nn.Linear(rank, base.out_features, bias=False)
        self.camera_source_gate = nn.Linear(source_dim, rank, bias=False)
        self.object_A = nn.Linear(base.in_features, rank, bias=False)
        self.object_B = nn.Linear(rank, base.out_features, bias=False)
        self.object_source_gate = nn.Linear(source_dim, rank, bias=False)
        self.shared_A = nn.Linear(base.in_features, rank, bias=False)
        self.shared_B = nn.Linear(rank, base.out_features, bias=False)

        self._active_route: FactorRoute | None = None
        self._camera_source: torch.Tensor | None = None
        self._object_source: torch.Tensor | None = None
        self.reset_adapter_parameters()
        self._move_adapters_to_base()
        self.set_trainable_owners(())

    def reset_adapter_parameters(self) -> None:
        for module in (self.camera_A, self.object_A, self.shared_A):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5.0))
        for module in (self.camera_B, self.object_B, self.shared_B):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
        nn.init.xavier_uniform_(self.camera_source_gate.weight)
        nn.init.xavier_uniform_(self.object_source_gate.weight)

    def _move_adapters_to_base(self) -> None:
        kwargs = {
            "device": self.base.weight.device,
            "dtype": self.base.weight.dtype,
        }
        for module in (
            self.camera_A,
            self.camera_B,
            self.camera_source_gate,
            self.object_A,
            self.object_B,
            self.object_source_gate,
            self.shared_A,
            self.shared_B,
        ):
            module.to(**kwargs)

    def set_adapter_master_dtype(self, dtype: torch.dtype) -> None:
        """Set only trainable adapter masters; frozen native ``W0`` is untouched."""

        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("adapter master dtype must be floating point")
        for module in (
            self.camera_A,
            self.camera_B,
            self.camera_source_gate,
            self.object_A,
            self.object_B,
            self.object_source_gate,
            self.shared_A,
            self.shared_B,
        ):
            module.to(device=self.base.weight.device, dtype=dtype)
        if any(parameter.requires_grad for parameter in self.base.parameters()):
            raise RoutingContractError("wrapped native W0 became trainable")

    def requires_grad_(
        self, requires_grad: bool = True
    ) -> "SourceConditionedTriLoRALinear":
        if not isinstance(requires_grad, bool):
            raise TypeError("requires_grad must be bool")
        super().requires_grad_(requires_grad)
        self.base.requires_grad_(False)
        return self

    def owner_named_parameters(
        self,
    ) -> Mapping[str, tuple[tuple[str, nn.Parameter], ...]]:
        parameters = tuple(self.named_parameters())
        output: dict[str, tuple[tuple[str, nn.Parameter], ...]] = {}
        for owner, prefixes in self.OWNER_PREFIXES.items():
            output[owner] = tuple(
                (name, parameter)
                for name, parameter in parameters
                if name.startswith(prefixes)
            )
        return output

    def set_trainable_owners(self, owners: tuple[str, ...]) -> None:
        normalized = tuple(str(owner).strip().lower() for owner in owners)
        if len(normalized) != len(set(normalized)) or any(
            owner not in self.OWNER_PREFIXES for owner in normalized
        ):
            raise RoutingContractError("TriLoRA owner allowlist is invalid")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        partitions = self.owner_named_parameters()
        for owner in normalized:
            for _, parameter in partitions[owner]:
                parameter.requires_grad_(True)
        self.base.requires_grad_(False)
        if any(parameter.requires_grad for parameter in self.base.parameters()):
            raise RoutingContractError("wrapped native W0 became trainable")

    @contextmanager
    def route_context(
        self,
        route: FactorRoute | str,
        *,
        camera_source: torch.Tensor | None = None,
        object_source: torch.Tensor | None = None,
    ) -> Iterator[None]:
        normalized = normalize_route(route)
        camera: torch.Tensor | None = None
        obj: torch.Tensor | None = None
        if normalized is FactorRoute.CAMERA_FACTOR:
            if object_source is not None:
                raise RoutingContractError("CAMERA_FACTOR forbids an Object source")
            camera = _require_source(
                camera_source,
                name="Camera source",
                source_dim=self.source_dim,
            )
        elif normalized is FactorRoute.OBJECT_FACTOR:
            if camera_source is not None:
                raise RoutingContractError("OBJECT_FACTOR forbids a Camera source")
            obj = _require_source(
                object_source,
                name="Object source",
                source_dim=self.source_dim,
            )
        else:
            camera = _require_source(
                camera_source,
                name="FULL_LANGUAGE Camera source",
                source_dim=self.source_dim,
            )
            obj = _require_source(
                object_source,
                name="FULL_LANGUAGE Object source",
                source_dim=self.source_dim,
            )
            if camera.shape[0] != obj.shape[0]:
                raise RoutingContractError("Camera/Object source batches differ")

        gate_device = self.camera_source_gate.weight.device
        previous = (
            self._active_route,
            self._camera_source,
            self._object_source,
        )
        self._active_route = normalized
        # Deliberately preserve autograd in Stage-F.  ``Tensor.to`` remains
        # differentiable when a device/dtype bridge is required.
        self._camera_source = (
            None
            if camera is None
            else camera.to(device=gate_device, dtype=SOURCE_GATE_DTYPE)
        )
        self._object_source = (
            None if obj is None else obj.to(device=gate_device, dtype=SOURCE_GATE_DTYPE)
        )
        try:
            yield
        finally:
            (
                self._active_route,
                self._camera_source,
                self._object_source,
            ) = previous

    def _source_gate(self, source: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        with torch.autocast(device_type=source.device.type, enabled=False):
            summary = source.to(dtype=SOURCE_GATE_DTYPE).mean(dim=1)
            gate = torch.tanh(
                F.linear(
                    summary,
                    projection.weight.to(dtype=SOURCE_GATE_DTYPE),
                    None,
                )
            )
        return gate.to(dtype=projection.weight.dtype)

    def _owned_residual(
        self,
        hidden: torch.Tensor,
        *,
        source: torch.Tensor | None,
        gate_projection: nn.Linear,
        down: nn.Linear,
        up: nn.Linear,
        owner: str,
    ) -> torch.Tensor:
        if source is None or source.shape[0] != hidden.shape[0]:
            raise RoutingContractError(f"{owner} source/hidden batch differs")
        broadcast = (hidden.shape[0],) + (1,) * (hidden.ndim - 2) + (self.rank,)
        gate = self._source_gate(source, gate_projection).reshape(broadcast)
        return up(down(hidden) * gate) * self.scaling

    def forward_with_components(self, hidden: torch.Tensor) -> TriLoRAOutput:
        if (
            not isinstance(hidden, torch.Tensor)
            or not hidden.is_floating_point()
            or hidden.ndim < 2
            or hidden.shape[-1] != self.base.in_features
            or not bool(torch.isfinite(hidden).all())
        ):
            raise RoutingContractError("native Linear input ABI differs")
        if any(parameter.requires_grad for parameter in self.base.parameters()):
            raise RoutingContractError("wrapped native W0 must stay frozen")
        route = self._active_route
        if route is None:
            raise RoutingContractError("TriLoRA route_context is not active")
        base = self.base(hidden)
        camera: torch.Tensor | None = None
        obj: torch.Tensor | None = None
        shared: torch.Tensor | None = None
        if route is FactorRoute.CAMERA_FACTOR:
            camera = self._owned_residual(
                hidden,
                source=self._camera_source,
                gate_projection=self.camera_source_gate,
                down=self.camera_A,
                up=self.camera_B,
                owner="Camera",
            )
            output = base + camera
        elif route is FactorRoute.OBJECT_FACTOR:
            obj = self._owned_residual(
                hidden,
                source=self._object_source,
                gate_projection=self.object_source_gate,
                down=self.object_A,
                up=self.object_B,
                owner="Object",
            )
            output = base + obj
        else:
            camera = self._owned_residual(
                hidden,
                source=self._camera_source,
                gate_projection=self.camera_source_gate,
                down=self.camera_A,
                up=self.camera_B,
                owner="Camera",
            )
            obj = self._owned_residual(
                hidden,
                source=self._object_source,
                gate_projection=self.object_source_gate,
                down=self.object_A,
                up=self.object_B,
                owner="Object",
            )
            shared = self.shared_B(self.shared_A(hidden)) * self.scaling
            output = base + camera + obj + shared
        return TriLoRAOutput(route, base, camera, obj, shared, output)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.forward_with_components(hidden).output


@dataclass(frozen=True, slots=True)
class FactorSpanRouteScores:
    camera_neither: torch.Tensor
    camera_camera_only: torch.Tensor
    object_neither: torch.Tensor
    object_camera_only: torch.Tensor
    object_object_only: torch.Tensor
    object_both: torch.Tensor


@dataclass(frozen=True, slots=True)
class FactorMargins:
    camera: torch.Tensor
    object: torch.Tensor


@dataclass(frozen=True, slots=True)
class FactorLogits:
    camera: torch.Tensor
    object: torch.Tensor


def factor_span_margins(scores: FactorSpanRouteScores) -> FactorMargins:
    if not isinstance(scores, FactorSpanRouteScores):
        raise TypeError("scores must be FactorSpanRouteScores")
    values = (
        scores.camera_neither,
        scores.camera_camera_only,
        scores.object_neither,
        scores.object_camera_only,
        scores.object_object_only,
        scores.object_both,
    )
    if not values:
        raise RoutingContractError("factor scores are empty")
    shape, device = values[0].shape, values[0].device
    if any(
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.shape != shape
        or value.device != device
        or not bool(torch.isfinite(value).all())
        for value in values
    ):
        raise RoutingContractError("factor score ABI differs")
    c_n, c_c, o_n, o_c, o_o, o_b = (value.to(dtype=torch.float32) for value in values)
    return FactorMargins(
        camera=c_c - c_n,
        object=0.5 * ((o_o - o_n) + (o_b - o_c)),
    )


class FactorSpanDecision(nn.Module):
    """Two learned Train-only intercepts; inference threshold is fixed at zero."""

    def __init__(self) -> None:
        super().__init__()
        self.b_c = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.b_o = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, margins: FactorMargins) -> FactorLogits:
        if not isinstance(margins, FactorMargins):
            raise TypeError("margins must be FactorMargins")
        return FactorLogits(
            camera=margins.camera.to(torch.float32) + self.b_c,
            object=margins.object.to(torch.float32) + self.b_o,
        )

    def from_scores(self, scores: FactorSpanRouteScores) -> FactorLogits:
        return self(factor_span_margins(scores))


def four_state_from_logits(logits: FactorLogits) -> tuple[str, ...]:
    if not isinstance(logits, FactorLogits):
        raise TypeError("logits must be FactorLogits")
    if (
        logits.camera.shape != logits.object.shape
        or not bool(torch.isfinite(logits.camera).all())
        or not bool(torch.isfinite(logits.object).all())
    ):
        raise RoutingContractError("factor logit ABI differs")
    camera = (logits.camera > 0).reshape(-1).tolist()
    obj = (logits.object > 0).reshape(-1).tolist()
    index = {
        (False, False): STATE_ORDER[0],
        (True, False): STATE_ORDER[1],
        (False, True): STATE_ORDER[2],
        (True, True): STATE_ORDER[3],
    }
    return tuple(index[(bool(c), bool(o))] for c, o in zip(camera, obj, strict=True))


__all__ = [
    "FactorLogits",
    "FactorMargins",
    "FactorRoute",
    "FactorSpanDecision",
    "FactorSpanRouteScores",
    "RoutingContractError",
    "SourceConditionedTriLoRALinear",
    "TriLoRAOutput",
    "factor_span_margins",
    "four_state_from_logits",
    "normalize_route",
]
