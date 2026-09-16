"""Attach the shared query router to NVILA native and MoSDeR backends."""

from pathlib import Path
import sys

# LOCAL_PATH: External grounding/backend sources are prepended and can override repository modules.
GROUNDING_MODEL_DIR = Path(
    "/root/autodl-tmp/mosder_reference_research_20260906/internal_grounding_physics_qa_v1/model"
)
EXPERIMENT_DIR = Path(
    "/root/story2_camera_object_motion/experiments/tst_native_adapter_o18_sandbox_v1"
)
for import_path in [GROUNDING_MODEL_DIR, EXPERIMENT_DIR]:
    sys.path.insert(0, str(import_path))
from soft_query_backend import (
    SoftQueryBackendMixin,
    VideoQuery20Request as VideoQuery20Request,
    frozen_query_embedding,
)
from soft_query_router import SoftQueryRouter as SoftQueryRouter
from mosder_final_v1.family_backend import NVILAMoSDeRBackend
from family_backends_v1 import NVILANativeBackend


class NVILAQueryMixin(SoftQueryBackendMixin):
    """Read question embeddings from NVILA's native language model."""

    def _grounding_embedding(self, request):
        return frozen_query_embedding(
            self.model.llm, self.model.tokenizer, request.question_stem
        )


class SoftQueryNVILA(NVILAQueryMixin, NVILAMoSDeRBackend):
    """NVILA with the MoSDeR source modules and soft spatial routing."""

    pass


class RawQueryNVILA(NVILAQueryMixin, NVILANativeBackend):
    """Frozen native NVILA with a router for feature extraction."""

    pass


def assemble_soft_query_backend(fresh_base, router, *, mode="query", raw=False):
    """Transfer a fresh backend into the requested NVILA routing variant."""
    assert (
        fresh_base.binding.key == "nvila_lite_8b"
        and fresh_base.unified_physical_core is None
        and not fresh_base._source_lora_paths
    )
    values = {
        attribute: getattr(fresh_base, attribute)
        for attribute in (
            "model",
            "processor",
            "binding",
            "device",
            "runtime_audit",
            "artifact_audit",
        )
    }
    fresh_base.close()
    backend = (RawQueryNVILA if raw else SoftQueryNVILA)(**values)
    if not raw:
        backend.attach_mosder_plugin()
        backend.attach_mosder_decoder()
    backend.attach_query_router(router, mode=mode)
    return backend
