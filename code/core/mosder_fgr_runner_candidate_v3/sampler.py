"""Deterministic full-coverage state-stratified sampler with exact resume."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any, Final, Mapping, Sequence

from .protocol import canonical_sha256


SCHEMA_VERSION: Final[str] = "mosder_full_coverage_state_sampler_v1"
STATE_ORDER: Final[tuple[str, ...]] = (
    "neither",
    "camera_only",
    "object_only",
    "both",
)


class SamplerContractError(RuntimeError):
    """Sampler membership, permutation, or cursor violated its ABI."""


@dataclass(frozen=True, slots=True)
class SampleRef:
    case_id: str
    state: str
    source_dataset: str

    def validated(self) -> "SampleRef":
        if (
            not isinstance(self.case_id, str)
            or not self.case_id
            or self.state not in STATE_ORDER
            or not isinstance(self.source_dataset, str)
            or not self.source_dataset
        ):
            raise SamplerContractError("sample reference is invalid")
        return self


class FullCoverageStateSampler:
    """Shuffle within state, then proportionally interleave without replacement.

    Every epoch is an exact permutation of membership even when state counts are
    unequal.  Choosing the state with the smallest emitted fraction spreads each
    state across the whole epoch instead of leaving a long majority-state tail.
    """

    def __init__(
        self,
        rows: Sequence[SampleRef],
        *,
        seed: int,
        epochs: int,
    ) -> None:
        if type(seed) is not int or seed < 0:
            raise SamplerContractError("sampler seed must be nonnegative int")
        if type(epochs) is not int or epochs <= 0:
            raise SamplerContractError("sampler epochs must be positive int")
        self.rows = tuple(row.validated() for row in rows)
        if not self.rows or len({row.case_id for row in self.rows}) != len(self.rows):
            raise SamplerContractError("membership is empty or has duplicate case IDs")
        self.groups = {
            state: tuple(
                index for index, row in enumerate(self.rows) if row.state == state
            )
            for state in STATE_ORDER
        }
        if any(not values for values in self.groups.values()):
            raise SamplerContractError("all four states must be represented")
        self.seed = seed
        self.epochs = epochs
        self.epoch = 0
        self.cursor = 0
        self._order = self._make_order(0)

    def _derived_seed(self, epoch: int, label: str) -> int:
        payload = f"{SCHEMA_VERSION}|{self.seed}|{epoch}|{label}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    def _make_order(self, epoch: int) -> tuple[int, ...]:
        queues: dict[str, list[int]] = {}
        for state, values in self.groups.items():
            queue = list(values)
            random.Random(self._derived_seed(epoch, state)).shuffle(queue)
            queues[state] = queue
        tie_order = list(STATE_ORDER)
        random.Random(self._derived_seed(epoch, "tie_order")).shuffle(tie_order)
        priority = {state: rank for rank, state in enumerate(tie_order)}
        emitted = {state: 0 for state in STATE_ORDER}
        offsets = {state: 0 for state in STATE_ORDER}
        output: list[int] = []
        while len(output) < len(self.rows):
            available = [
                state for state in STATE_ORDER if offsets[state] < len(queues[state])
            ]
            # Compare integer cross-products to avoid float/platform drift.
            chosen = available[0]
            for state in available[1:]:
                left = emitted[state] * len(self.groups[chosen])
                right = emitted[chosen] * len(self.groups[state])
                if left < right or (
                    left == right and priority[state] < priority[chosen]
                ):
                    chosen = state
            output.append(queues[chosen][offsets[chosen]])
            offsets[chosen] += 1
            emitted[chosen] += 1
        if len(output) != len(self.rows) or set(output) != set(range(len(self.rows))):
            raise SamplerContractError("epoch order is not an exact permutation")
        return tuple(output)

    @property
    def membership_sha256(self) -> str:
        return canonical_sha256(
            [
                {
                    "case_id": row.case_id,
                    "state": row.state,
                    "source_dataset": row.source_dataset,
                }
                for row in self.rows
            ]
        )

    @property
    def order_sha256(self) -> str:
        return canonical_sha256([self.rows[index].case_id for index in self._order])

    @property
    def exhausted(self) -> bool:
        return self.epoch == self.epochs

    @property
    def at_epoch_end(self) -> bool:
        return self.epoch < self.epochs and self.cursor == len(self.rows)

    def next_index(self) -> int | None:
        if self.exhausted:
            return None
        if self.cursor == len(self._order):
            self.epoch += 1
            self.cursor = 0
            if self.exhausted:
                return None
            self._order = self._make_order(self.epoch)
        value = self._order[self.cursor]
        self.cursor += 1
        return value

    def finish_epoch(self) -> bool:
        """Advance only from an observed epoch boundary; return run completion."""

        if not self.at_epoch_end:
            raise SamplerContractError("finish_epoch requires an epoch-end cursor")
        self.epoch += 1
        self.cursor = 0
        if not self.exhausted:
            self._order = self._make_order(self.epoch)
        return self.exhausted

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "seed": self.seed,
            "epochs": self.epochs,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "states": list(STATE_ORDER),
            "membership_sha256": self.membership_sha256,
            "order_sha256": None if self.exhausted else self.order_sha256,
            "order": None if self.exhausted else list(self._order),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        expected = {
            "schema_version": SCHEMA_VERSION,
            "seed": self.seed,
            "epochs": self.epochs,
            "states": list(STATE_ORDER),
            "membership_sha256": self.membership_sha256,
        }
        for name, wanted in expected.items():
            if value.get(name) != wanted:
                raise SamplerContractError(f"sampler checkpoint {name} differs")
        epoch = value.get("epoch")
        cursor = value.get("cursor")
        if type(epoch) is not int or type(cursor) is not int:
            raise SamplerContractError("sampler checkpoint cursor is invalid")
        if not 0 <= epoch <= self.epochs:
            raise SamplerContractError("sampler epoch is outside its run")
        if epoch == self.epochs:
            if (
                cursor != 0
                or value.get("order_sha256") is not None
                or value.get("order") is not None
            ):
                raise SamplerContractError("completed sampler state is invalid")
            self.epoch, self.cursor = epoch, cursor
            return
        if not 0 <= cursor <= len(self.rows):
            raise SamplerContractError("sampler cursor is outside its epoch")
        order = self._make_order(epoch)
        observed_sha = canonical_sha256([self.rows[index].case_id for index in order])
        recorded_order = value.get("order")
        if (
            not isinstance(recorded_order, list)
            or recorded_order != list(order)
            or value.get("order_sha256") != observed_sha
        ):
            raise SamplerContractError("sampler epoch permutation hash differs")
        self.epoch, self.cursor, self._order = epoch, cursor, order


__all__ = [
    "FullCoverageStateSampler",
    "SCHEMA_VERSION",
    "STATE_ORDER",
    "SampleRef",
    "SamplerContractError",
]
