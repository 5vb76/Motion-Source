"""Sequential best-Validation F -> G -> R orchestration for MoSDeR-v1."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Final, Mapping

from .authority import load_release_authority
from .checkpoint import (
    build_checkpoint,
    commit_pointer,
    create_once_checkpoint,
    load_committed_checkpoint,
    load_method_parameter_state,
    restore_checkpoint,
    set_global_seed,
    sha256_file,
)
from .data import ProjectedMoSDeRDataset
from .engine import (
    MoSDeRExecutionAdapter,
    OptimizerBoundary,
    StageRuntime,
    run_current_epoch,
    stage_total_optimizer_steps,
)
from .protocol import RunProtocol, STAGES, canonical_json_bytes
from .sampler import FullCoverageStateSampler
from .selection import BestValidationTracker, ValidationCandidate
from .validation import evaluate_with_rng_isolation


SCHEMA_VERSION: Final[str] = "mosder_formal_fgr_orchestrator_candidate_v1"


class OrchestratorContractError(RuntimeError):
    """Formal stage transaction or parent selection failed closed."""


def _stage_seed(protocol: RunProtocol, stage: str) -> int:
    payload = f"{protocol.identity_sha256}|{protocol.seed}|{stage}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _create_once_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise OrchestratorContractError(f"immutable JSON already exists: {path.name}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise OrchestratorContractError(
                "immutable JSON publication raced"
            ) from error
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


class FamilyRun:
    def __init__(
        self,
        backend: Any,
        protocol: RunProtocol,
        *,
        output_dir: Path,
        release_authority_path: Path,
        run_start_manifest_sha256: str,
    ) -> None:
        protocol.validated()
        self.authority = load_release_authority(release_authority_path, protocol)
        self.protocol = protocol
        if (
            not isinstance(run_start_manifest_sha256, str)
            or len(run_start_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in run_start_manifest_sha256
            )
        ):
            raise OrchestratorContractError("run-start manifest SHA256 is invalid")
        self.run_start_manifest_sha256 = run_start_manifest_sha256
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock_handle = (self.output_dir / "RUN.lock").open("a+b")
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise OrchestratorContractError("another process owns this run") from error
        self.train = ProjectedMoSDeRDataset("train", isolated_loader=True)
        self.validation = ProjectedMoSDeRDataset(
            "validation", allow_validation=True, isolated_loader=True
        )
        if (
            len(self.train) != protocol.train_rows
            or len(self.validation) != protocol.validation_rows
            or self.train.manifest_sha256 != protocol.train_projection_sha256
            or self.validation.manifest_sha256 != protocol.validation_projection_sha256
        ):
            raise OrchestratorContractError("protocol/data projection identity differs")
        self.adapter = MoSDeRExecutionAdapter(backend)
        self.method_parameters = self.adapter.method_parameters
        self.global_optimizer_step = 0
        self.parent_lineage: Mapping[str, Any] = {}

    def close(self) -> None:
        if getattr(self, "train", None) is not None:
            self.train.close()
        if getattr(self, "validation", None) is not None:
            self.validation.close()
        if getattr(self, "_lock_handle", None) is not None:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()
            self._lock_handle = None

    def _checkpoint(
        self,
        *,
        stage_dir: Path,
        name: str,
        runtime: StageRuntime,
        sampler: FullCoverageStateSampler,
        tracker: BestValidationTracker,
        stage_complete: bool,
    ) -> tuple[Path, str]:
        value = build_checkpoint(
            protocol=self.protocol,
            stage=runtime.stage,
            stage_complete=stage_complete,
            stage_optimizer_step=runtime.stage_optimizer_step,
            global_optimizer_step=runtime.global_optimizer_step,
            micro_examples_seen=runtime.micro_examples_seen,
            method_parameters=self.method_parameters,
            optimizer=runtime.optimizer,
            scheduler=runtime.scheduler,
            sampler=sampler,
            best_validation=tracker.state_dict(),
            frozen_base_digest=self.adapter.frozen_base_digest,
            parent_lineage=self.parent_lineage,
        )
        path = stage_dir / name
        return path, create_once_checkpoint(path, value)

    def _validate_candidate(
        self,
        *,
        stage_dir: Path,
        runtime: StageRuntime,
        tracker: BestValidationTracker,
        checkpoint_path: Path,
        checkpoint_sha256: str,
        epoch_one_based: int,
        kind: str = "trained",
        attempt: int,
    ) -> bool:
        metrics = evaluate_with_rng_isolation(
            self.adapter, self.validation, stage=runtime.stage
        )
        receipt = {
            "schema_version": "mosder_validation_receipt_candidate_v1",
            "protocol_sha256": self.protocol.identity_sha256,
            "stage": runtime.stage,
            "epoch_one_based": epoch_one_based,
            "optimizer_step": runtime.stage_optimizer_step,
            "checkpoint_name": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_sha256,
            "metrics": metrics,
            "authority_receipt": dict(self.authority),
            "confirmation_a_opened": False,
            "final_b_opened": False,
            "held_roles_opened": False,
        }
        receipt_path = stage_dir / (
            f"attempt_{attempt:04d}__validation_epoch_{epoch_one_based:03d}_"
            f"step_{runtime.stage_optimizer_step:08d}.json"
        )
        receipt_sha = _create_once_json(receipt_path, receipt)
        candidate = ValidationCandidate(
            stage=runtime.stage,
            epoch_one_based=epoch_one_based,
            optimizer_step=runtime.stage_optimizer_step,
            checkpoint_name=checkpoint_path.name,
            checkpoint_sha256=checkpoint_sha256,
            receipt_sha256=receipt_sha,
            metrics=dict(metrics["selection_metrics"]),
            kind=kind,
        )
        improved = tracker.consider(candidate)
        if improved:
            commit_pointer(
                stage_dir,
                pointer_name="BEST.json",
                checkpoint_name=checkpoint_path.name,
                checkpoint_sha256=checkpoint_sha256,
                protocol=self.protocol,
                stage=runtime.stage,
                stage_optimizer_step=runtime.stage_optimizer_step,
                global_optimizer_step=runtime.global_optimizer_step,
            )
        # Validation configures FROZEN.  Restore the same stage owner without
        # replacing its optimizer or scheduler.
        configuration = self.adapter.backend.configure_mosder_stage(runtime.stage)
        runtime.configuration = configuration
        runtime.active_parameters = tuple(
            (name, parameter)
            for name, parameter in configuration.owner_configuration.owners.all
            if name in set(configuration.owner_configuration.trainable_names)
        )
        return improved

    def _run_stage(self, stage: str) -> Mapping[str, Any]:
        stage_dir = self.output_dir / f"stage_{stage.lower()}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        existing_attempts = []
        for path in stage_dir.iterdir():
            prefix = path.name.split("__", 1)[0]
            if prefix.startswith("attempt_") and prefix[8:].isdigit():
                existing_attempts.append(int(prefix[8:]))
        attempt = max(existing_attempts, default=0) + 1
        tracker = BestValidationTracker(stage)
        sampler = FullCoverageStateSampler(
            self.train.refs,
            seed=_stage_seed(self.protocol, stage),
            epochs=self.protocol.epochs(stage),
        )
        total_steps = stage_total_optimizer_steps(
            len(self.train),
            epochs=self.protocol.epochs(stage),
            accumulation=self.protocol.accumulation(stage),
        )
        runtime = self.adapter.configure_stage(
            stage,
            total_optimizer_steps=total_steps,
            protocol=self.protocol,
            global_optimizer_step=self.global_optimizer_step,
        )

        latest_pointer = stage_dir / "LATEST.json"
        if latest_pointer.is_file():
            latest, _ = load_committed_checkpoint(
                stage_dir, pointer_name="LATEST.json", protocol=self.protocol
            )
            progress = restore_checkpoint(
                latest,
                protocol=self.protocol,
                expected_stage=stage,
                method_parameters=self.method_parameters,
                optimizer=runtime.optimizer,
                scheduler=runtime.scheduler,
                sampler=sampler,
                frozen_base_digest=self.adapter.frozen_base_digest,
            )
            if progress["parent_lineage"] != dict(self.parent_lineage):
                raise OrchestratorContractError("resume parent lineage differs")
            tracker.load_state_dict(progress["best_validation"])
            runtime.stage_optimizer_step = int(progress["stage_optimizer_step"])
            runtime.global_optimizer_step = int(progress["global_optimizer_step"])
            runtime.micro_examples_seen = int(progress["micro_examples_seen"])
            self.global_optimizer_step = runtime.global_optimizer_step

        if stage == "R" and runtime.stage_optimizer_step == 0 and not tracker.history:
            checkpoint, digest = self._checkpoint(
                stage_dir=stage_dir,
                name=f"attempt_{attempt:04d}__candidate_r_step0_parent_g.pt",
                runtime=runtime,
                sampler=sampler,
                tracker=tracker,
                stage_complete=False,
            )
            self._validate_candidate(
                stage_dir=stage_dir,
                runtime=runtime,
                tracker=tracker,
                checkpoint_path=checkpoint,
                checkpoint_sha256=digest,
                epoch_one_based=0,
                kind="r_step0_parent_g",
                attempt=attempt,
            )

        def optimizer_boundary(
            current: StageRuntime,
            current_sampler: FullCoverageStateSampler,
            boundary: OptimizerBoundary,
        ) -> None:
            # Never publish an epoch-end cursor as a resumable mid-epoch
            # transaction.  The epoch callback must complete and finish_epoch
            # must advance the sampler before resume_after_epoch becomes LATEST.
            if current_sampler.at_epoch_end:
                return
            interval = self.protocol.checkpoint_every_optimizer_steps
            if boundary.stage_optimizer_step % interval:
                return
            path, digest = self._checkpoint(
                stage_dir=stage_dir,
                name=(
                    f"attempt_{attempt:04d}__resume_step_"
                    f"{boundary.stage_optimizer_step:08d}.pt"
                ),
                runtime=current,
                sampler=current_sampler,
                tracker=tracker,
                stage_complete=False,
            )
            commit_pointer(
                stage_dir,
                pointer_name="LATEST.json",
                checkpoint_name=path.name,
                checkpoint_sha256=digest,
                protocol=self.protocol,
                stage=stage,
                stage_optimizer_step=current.stage_optimizer_step,
                global_optimizer_step=current.global_optimizer_step,
            )

        while not sampler.exhausted:
            epoch = sampler.epoch + 1

            def epoch_boundary(
                current: StageRuntime,
                current_sampler: FullCoverageStateSampler,
                boundaries: Any,
            ) -> None:
                del boundaries
                path, digest = self._checkpoint(
                    stage_dir=stage_dir,
                    name=(
                        f"attempt_{attempt:04d}__candidate_epoch_{epoch:03d}_"
                        f"step_{current.stage_optimizer_step:08d}.pt"
                    ),
                    runtime=current,
                    sampler=current_sampler,
                    tracker=tracker,
                    # The sampler advances only after the epoch callback;
                    # completion is committed by resume_after_epoch below.
                    stage_complete=False,
                )
                self._validate_candidate(
                    stage_dir=stage_dir,
                    runtime=current,
                    tracker=tracker,
                    checkpoint_path=path,
                    checkpoint_sha256=digest,
                    epoch_one_based=epoch,
                    attempt=attempt,
                )

            run_current_epoch(
                self.adapter,
                runtime,
                sampler,
                self.train,
                accumulation=self.protocol.accumulation(stage),
                gradient_clip_max_norm=self.protocol.gradient_clip_max_norm,
                on_optimizer_boundary=optimizer_boundary,
                on_epoch_boundary=epoch_boundary,
            )
            resume_path, resume_sha = self._checkpoint(
                stage_dir=stage_dir,
                name=(
                    f"attempt_{attempt:04d}__resume_after_epoch_{epoch:03d}_"
                    f"step_{runtime.stage_optimizer_step:08d}.pt"
                ),
                runtime=runtime,
                sampler=sampler,
                tracker=tracker,
                stage_complete=sampler.exhausted,
            )
            commit_pointer(
                stage_dir,
                pointer_name="LATEST.json",
                checkpoint_name=resume_path.name,
                checkpoint_sha256=resume_sha,
                protocol=self.protocol,
                stage=stage,
                stage_optimizer_step=runtime.stage_optimizer_step,
                global_optimizer_step=runtime.global_optimizer_step,
            )
        if tracker.best is None:
            raise OrchestratorContractError(f"Stage-{stage} selected no checkpoint")
        best, best_pointer = load_committed_checkpoint(
            stage_dir, pointer_name="BEST.json", protocol=self.protocol
        )
        if (
            best_pointer.get("checkpoint_name") != tracker.best.checkpoint_name
            or best_pointer.get("checkpoint_sha256") != tracker.best.checkpoint_sha256
        ):
            raise OrchestratorContractError(
                f"Stage-{stage} BEST pointer/tracker selection differs"
            )
        load_method_parameter_state(
            self.method_parameters, best["method_parameter_state"]
        )
        self.global_optimizer_step = runtime.global_optimizer_step
        self.parent_lineage = {
            "parent_stage": stage,
            "parent_checkpoint_name": tracker.best.checkpoint_name,
            "parent_checkpoint_sha256": tracker.best.checkpoint_sha256,
            "parent_receipt_sha256": tracker.best.receipt_sha256,
        }
        return {
            "stage": stage,
            "optimizer_steps": runtime.stage_optimizer_step,
            "global_optimizer_step": runtime.global_optimizer_step,
            "best": tracker.best.as_dict(),
            "best_pointer": dict(best_pointer),
            "tracker": tracker.state_dict(),
        }

    def run(self) -> Mapping[str, Any]:
        set_global_seed(self.protocol.seed)
        stages: dict[str, Any] = {}
        parent_sha: str | None = None
        try:
            for stage in STAGES:
                result = dict(self._run_stage(stage))
                result["selected_parent_checkpoint_sha256"] = parent_sha
                parent_sha = result["best"]["checkpoint_sha256"]
                stages[stage] = result
            completion = {
                "schema_version": SCHEMA_VERSION,
                "status": "PASS_FORMAL_FGR_TRAINING_AND_VALIDATION_CHECKPOINT_SELECTION",
                "protocol_sha256": self.protocol.identity_sha256,
                "family": self.protocol.family,
                "run_start_manifest_sha256": self.run_start_manifest_sha256,
                "stages": stages,
                "selected_final_checkpoint_sha256": parent_sha,
                "confirmation_a_opened": False,
                "final_b_opened": False,
                "held_roles_opened": False,
            }
            completion_sha = _create_once_json(
                self.output_dir / "TRAINING_COMPLETION.json", completion
            )
            return {**completion, "completion_sha256": completion_sha}
        finally:
            self.adapter.backend.configure_mosder_stage("FROZEN")


__all__ = ["FamilyRun", "OrchestratorContractError", "SCHEMA_VERSION"]
