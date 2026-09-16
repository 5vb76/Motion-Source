"""Camera and object temporal adapters over ten pooled video units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .contract import SOURCE_RANK, TEMPORAL_UNITS


class MotionSourceContractError(RuntimeError):
    """A source tensor, owner, or trainability contract was violated."""


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_source_input(
    value: torch.Tensor,
    *,
    name: str,
    hidden_size: int,
) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise MotionSourceContractError(f"{name} must be a floating tensor")
    if value.ndim != 3 or tuple(value.shape[1:]) != (
        TEMPORAL_UNITS,
        hidden_size,
    ):
        raise MotionSourceContractError(
            f"{name} must have shape [B,{TEMPORAL_UNITS},{hidden_size}], "
            f"got {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all()):
        raise MotionSourceContractError(f"{name} contains non-finite values")


class TemporalMotionSourceAdapter(nn.Module):
    """Map pooled features and adjacent differences to a ten-unit source sequence.

    Inputs and outputs have shape ``[batch, 10, hidden_size]``. Camera and
    object branches each own an instance of this adapter.
    """

    def __init__(self, hidden_size: int, rank: int = SOURCE_RANK) -> None:
        super().__init__()
        self.hidden_size = _positive_int(hidden_size, "hidden_size")
        self.rank = _positive_int(rank, "rank")
        if self.rank >= self.hidden_size:
            raise ValueError("source rank must be smaller than hidden_size")
        self.input_norm = nn.LayerNorm(self.hidden_size)
        self.down = nn.Linear(2 * self.hidden_size, self.rank)
        self.up = nn.Linear(self.rank, self.hidden_size)
        self.time_embedding = nn.Parameter(
            torch.empty(TEMPORAL_UNITS, self.hidden_size)
        )
        self.temporal_mixer = nn.Linear(TEMPORAL_UNITS, TEMPORAL_UNITS)
        self.output_norm = nn.LayerNorm(self.hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.input_norm.reset_parameters()
        self.down.reset_parameters()
        self.up.reset_parameters()
        self.temporal_mixer.reset_parameters()
        self.output_norm.reset_parameters()
        nn.init.normal_(self.time_embedding, mean=0.0, std=0.02)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        _require_source_input(
            value,
            name="motion-source input",
            hidden_size=self.hidden_size,
        )
        normalized = self.input_norm(value)
        delta = torch.zeros_like(normalized)
        delta[:, 1:] = normalized[:, 1:] - normalized[:, :-1]
        bottleneck = F.gelu(self.down(torch.cat((normalized, delta), dim=-1)))
        latent = self.up(bottleneck) + self.time_embedding.unsqueeze(0)
        mixed = self.temporal_mixer(latent.transpose(1, 2)).transpose(1, 2)
        output = self.output_norm(latent + F.gelu(mixed))
        if output.shape != value.shape or not bool(torch.isfinite(output).all()):
            raise MotionSourceContractError("motion-source output ABI differs")
        return output


@dataclass(frozen=True, slots=True)
class MotionSourceOutput:
    """Camera and object features, each shaped ``[batch, 10, hidden_size]``."""

    camera_source: torch.Tensor
    object_source: torch.Tensor


class MoSDeRMotionSourceCore(nn.Module):
    """Independent Camera/context and Object/target temporal adapters."""

    VALID_STAGES = frozenset({"F", "G", "R", "FROZEN"})

    def __init__(self, hidden_size: int, *, rank: int = SOURCE_RANK) -> None:
        super().__init__()
        self.hidden_size = _positive_int(hidden_size, "hidden_size")
        self.rank = _positive_int(rank, "rank")
        self.camera_source_adapter = TemporalMotionSourceAdapter(
            self.hidden_size, self.rank
        )
        self.object_source_adapter = TemporalMotionSourceAdapter(
            self.hidden_size, self.rank
        )
        self._active_stage = "FROZEN"
        self.configure_stage("FROZEN")

    @property
    def active_stage(self) -> str:
        return self._active_stage

    def owner_named_parameters(
        self,
    ) -> dict[str, tuple[tuple[str, nn.Parameter], ...]]:
        return {
            "camera": tuple(
                self.camera_source_adapter.named_parameters(
                    prefix="camera_source_adapter"
                )
            ),
            "object": tuple(
                self.object_source_adapter.named_parameters(
                    prefix="object_source_adapter"
                )
            ),
        }

    def expected_trainable_parameter_names(self, stage: str) -> tuple[str, ...]:
        normalized = _normalize_stage(stage)
        if normalized == "F":
            partitions = self.owner_named_parameters()
            return tuple(
                name for owner in ("camera", "object") for name, _ in partitions[owner]
            )
        return ()

    def configure_stage(self, stage: str) -> tuple[str, ...]:
        normalized = _normalize_stage(stage)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        if normalized == "F":
            for parameter in self.parameters():
                parameter.requires_grad_(True)
        self._active_stage = normalized
        observed = self.trainable_parameter_names()
        expected = self.expected_trainable_parameter_names(normalized)
        if set(observed) != set(expected):
            raise MotionSourceContractError(
                "source-core trainability differs from its owner allowlist"
            )
        return observed

    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        )

    def physical_forward(
        self,
        target_features: torch.Tensor,
        context_features: torch.Tensor,
    ) -> MotionSourceOutput:
        _require_source_input(
            target_features,
            name="target features",
            hidden_size=self.hidden_size,
        )
        _require_source_input(
            context_features,
            name="context features",
            hidden_size=self.hidden_size,
        )
        if target_features.shape[0] != context_features.shape[0]:
            raise MotionSourceContractError("target/context feature batch sizes differ")
        camera_source = self.camera_source_adapter(context_features)
        object_source = self.object_source_adapter(target_features)
        return MotionSourceOutput(
            camera_source=camera_source,
            object_source=object_source,
        )

    def forward(
        self,
        target_features: torch.Tensor,
        context_features: torch.Tensor,
    ) -> MotionSourceOutput:
        return self.physical_forward(target_features, context_features)


def _normalize_stage(stage: str) -> str:
    if not isinstance(stage, str):
        raise TypeError("stage must be F, G, R, or FROZEN")
    normalized = stage.strip().upper().replace("STAGE-", "")
    if normalized == "P":
        raise ValueError("MoSDeR has no Stage-P or numeric trajectory heads")
    if normalized not in MoSDeRMotionSourceCore.VALID_STAGES:
        raise ValueError("stage must be F, G, R, or FROZEN")
    return normalized


def flatten_named_parameters(
    values: Iterable[tuple[str, nn.Parameter]],
) -> tuple[nn.Parameter, ...]:
    return tuple(parameter for _, parameter in values)


__all__ = [
    "MoSDeRMotionSourceCore",
    "MotionSourceContractError",
    "MotionSourceOutput",
    "TemporalMotionSourceAdapter",
    "flatten_named_parameters",
]
