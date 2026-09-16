"""Protocol variant with a bounded F/G/R training budget.

This package supplies the compute-budget settings used by the QA runtime.
The execution engine lives in ``mosder_fgr_runner_candidate_v3``."""

from .protocol import RunProtocol

__all__ = ["RunProtocol"]
