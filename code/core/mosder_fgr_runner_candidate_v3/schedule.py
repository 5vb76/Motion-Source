"""Explicit warmup/cosine scheduler with an exact portable state."""

from __future__ import annotations

import math
from typing import Any, Final, Mapping

import torch


SCHEMA_VERSION: Final[str] = "mosder_warmup_5pct_cosine_to_0p1_v1"


class ScheduleContractError(RuntimeError):
    """Scheduler construction or resume state differs from the protocol."""


class WarmupCosineToFloor:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        warmup_fraction: float = 0.05,
        floor: float = 0.1,
    ) -> None:
        if type(total_steps) is not int or total_steps <= 0:
            raise ScheduleContractError("total_steps must be positive int")
        if not 0.0 < warmup_fraction < 1.0 or not 0.0 < floor <= 1.0:
            raise ScheduleContractError("scheduler fractions are invalid")
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_fraction = float(warmup_fraction)
        self.warmup_steps = max(1, int(math.ceil(total_steps * warmup_fraction)))
        self.floor = float(floor)
        self.base_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)
        if not self.base_lrs or any(lr <= 0.0 for lr in self.base_lrs):
            raise ScheduleContractError("optimizer base LR is invalid")
        self.completed_steps = 0
        self._apply_for_upcoming_step()

    def multiplier(self, upcoming_step: int) -> float:
        if type(upcoming_step) is not int or upcoming_step < 0:
            raise ScheduleContractError("upcoming step is invalid")
        if upcoming_step < self.warmup_steps:
            return float(upcoming_step + 1) / float(self.warmup_steps)
        if self.total_steps <= self.warmup_steps + 1:
            return self.floor
        progress = min(
            1.0,
            float(upcoming_step - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps - 1),
        )
        return self.floor + (1.0 - self.floor) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    def _apply_for_upcoming_step(self) -> None:
        multiplier = self.multiplier(self.completed_steps)
        for group, base_lr in zip(
            self.optimizer.param_groups, self.base_lrs, strict=True
        ):
            group["lr"] = base_lr * multiplier

    def step(self) -> None:
        if self.completed_steps >= self.total_steps:
            raise ScheduleContractError("scheduler stepped beyond total_steps")
        self.completed_steps += 1
        self._apply_for_upcoming_step()

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "total_steps": self.total_steps,
            "warmup_fraction": self.warmup_fraction,
            "warmup_steps": self.warmup_steps,
            "floor": self.floor,
            "base_lrs": list(self.base_lrs),
            "completed_steps": self.completed_steps,
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        expected = {
            "schema_version": SCHEMA_VERSION,
            "total_steps": self.total_steps,
            "warmup_fraction": self.warmup_fraction,
            "warmup_steps": self.warmup_steps,
            "floor": self.floor,
            "base_lrs": list(self.base_lrs),
        }
        for name, wanted in expected.items():
            if value.get(name) != wanted:
                raise ScheduleContractError(f"scheduler checkpoint {name} differs")
        completed = value.get("completed_steps")
        if type(completed) is not int or not 0 <= completed <= self.total_steps:
            raise ScheduleContractError("scheduler completed_steps is invalid")
        self.completed_steps = completed
        self._apply_for_upcoming_step()


__all__ = ["SCHEMA_VERSION", "ScheduleContractError", "WarmupCosineToFloor"]
