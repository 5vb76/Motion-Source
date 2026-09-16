"""Frozen, box-free native inference adapter. Importing never loads a VLM."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import importlib.util
import math
import sys
from typing import Mapping
import numpy as np

RUNTIME_DIR = Path(__file__).resolve().parent
DIAGNOSTIC_DIR = RUNTIME_DIR.parent
GROUNDING_DIR = DIAGNOSTIC_DIR.parent / "internal_grounding_physics_qa_v1"
# LOCAL_PATH: Requires external backend source plus local router/parent checkpoints below.
EXPERIMENT_DIR = Path(
    "/root/story2_camera_object_motion/experiments/tst_native_adapter_o18_sandbox_v1"
)
# LOCAL_PATH: This relative router checkpoint is not included in the repository.
GROUND_PATH = GROUNDING_DIR / "runs/grounder/FINAL.pt"
GROUND_SHA = "3f8ce265f0f7cc94b2c8d963ae3aca77443c68f64f1ca915492a32c145063068"
PARENT_PATH = Path(
    "/root/autodl-tmp/tst_native_adapter_o18_sandbox_v1/MOSDER_FINAL_TRAINING_V1/recoveries/seed_20260902/molmo2_o_7b_validation_abi_v1/stage_r/attempt_0001__candidate_epoch_001_step_00000647.pt"
)
PARENT_SHA = "a4770a322b079c8f6d1a133b19741302f54b289e00b7f7a9448679ee9c10815b"
RAW_CONDITIONS = (
    "raw",
    "generic",
    "reference_camera",
    "reference_object",
    "reference_both",
    "predicted_both",
)
GENERATION = {"max_new_tokens": 32, "do_sample": False}
GENERIC = "请分别考虑相机运动和题目所问目标相对于固定场景的运动，再根据视频回答原问题。不要把仅由相机运动引起的画面变化直接当成目标运动。"
ENDING = "请仍只按原问题要求作答。"
CAMERA_TEMPLATE = "相机：{state}。"
OBJECT_TEMPLATE = "题目所问目标相对于固定场景：{state}。"
STATE_TEXT = {"moving": "有运动", "static": "静止", "unknown": "未知"}
FACTOR_SUFFIX = (
    "For the target referred to by the question above, describe only its motion sources. "
    "Judge camera motion relative to the fixed scene and target motion relative to the fixed scene. "
    "Camera-induced apparent image motion alone does not count as target motion. "
    "Use exactly the four-line Camera / Object / State / Description format."
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_legacy_common():
    """Load the grounding helpers under a separate module identity."""
    name = "_frozen_igpq_common_for_omni"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, GROUNDING_DIR / "common.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


# LOCAL_PATH: External backend and grounding model sources can override repository modules.
def ensure_import_paths():
    for import_path in (EXPERIMENT_DIR, GROUNDING_DIR / "model"):
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))


@dataclass(frozen=True)
class OriginalInput:
    """Input fields allowed to reach the box-free inference backend."""

    ordinal: int
    qa_id: str
    video_id: str
    question_type: str
    question_stem: str
    prompt: str
    rgb20_npz_path: str
    rgb20_npz_sha256: str
    timestamps_ns: tuple[int, ...]
    rgb_sha256: str

    @classmethod
    def from_row(cls, row: Mapping):
        # Exact whitelist: no captions, ref notes, labels, entities or answers.
        if row.get("prepared") is False:
            raise ValueError("fixed item was not prepared")
        out = cls(
            ordinal=row["ordinal"],
            qa_id=row["qa_id"],
            video_id=row["video_id"],
            question_type=row["question_type"],
            question_stem=row["question_stem"],
            prompt=row["prompt"],
            rgb20_npz_path=row["rgb20_npz_path"],
            rgb20_npz_sha256=row["rgb20_npz_sha256"],
            timestamps_ns=tuple(row["timestamps_ns"]),
            rgb_sha256=row["rgb_sha256"],
        )
        if type(out.ordinal) is not int or out.ordinal < 0:
            raise ValueError("invalid ordinal")
        if out.question_type not in ("ynqa", "mcqa"):
            raise ValueError("unsupported question type")
        if not out.question_stem or out.question_stem != out.question_stem.strip():
            raise ValueError("original stem must be exact stripped text")
        if not out.prompt.startswith(out.question_stem):
            raise ValueError("original stem is not prompt prefix")
        if (
            len(out.timestamps_ns) != 20
            or any(type(t) is not int for t in out.timestamps_ns)
            or any(b <= a for a, b in zip(out.timestamps_ns, out.timestamps_ns[1:]))
        ):
            raise ValueError("invalid fixed20 timestamps")
        return out


def load_pixels(item: OriginalInput):
    """Verify and load a clip containing only RGB frames and timestamps."""
    # LOCAL_PATH: rgb20_npz_path in each manifest row must resolve to a local RGB archive.
    if sha(item.rgb20_npz_path) != item.rgb20_npz_sha256:
        raise ValueError("NPZ SHA mismatch")
    with np.load(item.rgb20_npz_path, allow_pickle=False) as archive:
        if set(archive.files) != {"rgb", "timestamps_ns"}:
            raise ValueError(
                "input NPZ must contain only rgb+timestamps_ns; boxes forbidden"
            )
        rgb = archive["rgb"].copy()
        timestamps = archive["timestamps_ns"].copy()
    if (
        rgb.dtype != np.uint8
        or rgb.ndim != 4
        or rgb.shape[0] != 20
        or rgb.shape[-1] != 3
    ):
        raise ValueError("RGB20 ABI mismatch")
    if tuple(timestamps.tolist()) != item.timestamps_ns:
        raise ValueError("pixel timestamps mismatch")
    if hashlib.sha256(rgb.tobytes(order="C")).hexdigest() != item.rgb_sha256:
        raise ValueError("RGB byte hash mismatch")
    return rgb


def provided_states(row: Mapping):
    """Consume only approved enum values and validity booleans, never notes."""
    result = {}
    for factor in ("camera", "object"):
        valid = row[factor + "_reference_valid"]
        if type(valid) is not bool:
            raise ValueError("reference validity must be Boolean")
        if not valid:
            result[factor] = "unknown"
            continue
        state = row[factor]
        if state not in STATE_TEXT:
            raise ValueError("invalid approved state enum")
        result[factor] = state
    return result


def predicted_states(row: Mapping):
    if row.get("success") is not True:
        raise ValueError("source prediction unavailable; no raw/unknown fallback")
    result = {factor: row[factor] for factor in ("camera", "object")}
    if any(v not in ("moving", "static") for v in result.values()):
        raise ValueError("predicted source must use frozen binary decision")
    return result


def condition_prompt(
    item: OriginalInput, condition: str, states: Mapping | None = None
):
    if condition == "raw":
        return item.prompt
    if condition not in RAW_CONDITIONS:
        raise ValueError("invalid raw condition")
    used = {"camera": "unknown", "object": "unknown"}
    factors = {
        "generic": (),
        "reference_camera": ("camera",),
        "reference_object": ("object",),
        "reference_both": ("camera", "object"),
        "predicted_both": ("camera", "object"),
    }[condition]
    for factor in factors:
        if states is None or states[factor] not in STATE_TEXT:
            raise ValueError("missing/invalid condition state")
        used[factor] = states[factor]
    return (
        item.prompt
        + "\n\n"
        + GENERIC
        + "\n"
        + CAMERA_TEMPLATE.format(state=STATE_TEXT[used["camera"]])
        + "\n"
        + OBJECT_TEMPLATE.format(state=STATE_TEXT[used["object"]])
        + "\n"
        + ENDING
    )


def native_request(item: OriginalInput, rgb, prompt: str):
    ensure_import_paths()
    from soft_query_backend import VideoQuery20Request

    # Stem remains the original question even after factor/information suffixes.
    request = VideoQuery20Request(
        tuple(rgb), list(item.timestamps_ns), prompt, item.qa_id, item.question_stem
    )
    clean = request.validated()
    if clean.question_stem != item.question_stem:
        raise ValueError("router question stem drift")
    return request


def source_request(item: OriginalInput, rgb):
    # Deliberately no reference-state argument or original answer option input.
    return native_request(item, rgb, item.question_stem + "\n\n" + FACTOR_SUFFIX)


def factor_prediction(backend, request):
    """Exact original 2+4 full canonical-answer mean-log-score decision."""
    ensure_import_paths()
    import torch
    from mosder_final_v1.language import canonical_answer
    from mosder_final_v1.routing import (
        FactorRoute,
        FactorSpanRouteScores,
        factor_span_margins,
        four_state_from_logits,
    )

    calls = (
        ("camera_neither", "neither", FactorRoute.CAMERA_FACTOR),
        ("camera_camera_only", "camera_only", FactorRoute.CAMERA_FACTOR),
        ("object_neither", "neither", FactorRoute.OBJECT_FACTOR),
        ("object_camera_only", "camera_only", FactorRoute.OBJECT_FACTOR),
        ("object_object_only", "object_only", FactorRoute.OBJECT_FACTOR),
        ("object_both", "both", FactorRoute.OBJECT_FACTOR),
    )
    before = backend.route_success_counts()
    values = {}
    details = []
    with torch.inference_mode():
        for field, state, route in calls:
            answer = canonical_answer(state)
            score = backend.teacher_forced_loss(request, answer, route=route)
            ids, token_scores = score.answer_token_ids, score.token_log_probabilities
            if (
                score.text != answer
                or ids.ndim != 1
                or token_scores.shape != ids.shape
                or ids.numel() == 0
            ):
                raise ValueError("canonical native answer-span ABI mismatch")
            if not bool(torch.isfinite(token_scores).all()) or bool(
                (token_scores > 0).any()
            ):
                raise ValueError("nonfinite/invalid token scores")
            scalar = token_scores.float().mean()
            values[field] = scalar
            details.append(
                {
                    "field": field,
                    "route": route.value,
                    "canonical_state": state,
                    "canonical_text_sha256": text_sha(answer),
                    "answer_token_ids": ids.detach().cpu().tolist(),
                    "token_log_probabilities": token_scores.float()
                    .detach()
                    .cpu()
                    .tolist(),
                    "mean_answer_log_score": float(scalar.cpu()),
                }
            )
        scores = FactorSpanRouteScores(**values)
        margins = factor_span_margins(scores)
        logits = backend.factor_decision.from_scores(scores)
        state = four_state_from_logits(logits)
    if len(state) != 1:
        raise ValueError("factor decoder returned nonscalar state")
    after = backend.route_success_counts()
    delta = {k: after[k] - before[k] for k in before}
    if delta != {"CAMERA_FACTOR": 2, "OBJECT_FACTOR": 4, "FULL_LANGUAGE": 0}:
        raise ValueError("factor route count must be exact2/4/0")
    camera_logit, object_logit = float(logits.camera.cpu()), float(logits.object.cpu())
    if not math.isfinite(camera_logit) or not math.isfinite(object_logit):
        raise ValueError("nonfinite source logits")
    return {
        "camera": "moving" if camera_logit > 0 else "static",
        "object": "moving" if object_logit > 0 else "static",
        "state": state[0],
        "q_camera": camera_logit,
        "q_object": object_logit,
        "camera_margin": float(margins.camera.cpu()),
        "object_margin": float(margins.object.cpu()),
        "b_camera": float(backend.factor_decision.b_c.detach().cpu()),
        "b_object": float(backend.factor_decision.b_o.detach().cpu()),
        "decision_rule": "q>0 is moving; zero is static; original learned intercepts",
        "score_records": details,
        "route_call_delta": delta,
    }


def freeze_all(backend):
    backend.model.eval()
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    if any(module.training for module in backend.model.modules()):
        raise ValueError("all modules must be eval")


def parameter_stamp(model):
    if any(
        parameter.requires_grad or parameter.grad is not None
        for parameter in model.parameters()
    ):
        raise ValueError("inference parameter gained trainability/gradient")
    if any(module.training for module in model.modules()):
        raise ValueError("inference module left eval mode")
    return tuple(
        (name, id(parameter), parameter._version)
        for name, parameter in model.named_parameters()
    )


def all_parameter_hashes(model):
    """Full byte hashes before/after phase, CPU streaming one parameter at time."""
    import torch

    result = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad or parameter.grad is not None:
            raise ValueError("non-frozen inference parameter")
        value = parameter.detach().cpu().contiguous()
        result[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": value.numel(),
            "sha256": hashlib.sha256(
                value.reshape(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
    return result


def open_frozen_backend(loaded, *, internal: bool):
    """Load the pinned router and optional parent, then freeze all parameters."""
    import torch

    ensure_import_paths()
    legacy = load_legacy_common()
    from soft_query_router import SoftQueryRouter

    if sha(GROUND_PATH) != GROUND_SHA:
        raise ValueError("preselected grounder hash mismatch")
    router = SoftQueryRouter(4096, rank=64)
    router.load_state_dict(
        torch.load(GROUND_PATH, map_location="cpu", weights_only=True)
    )
    if internal:
        if sha(PARENT_PATH) != PARENT_SHA:
            raise ValueError("preselected original R647 hash mismatch")
        backend, identity = legacy.parent_on_raw(loaded.backend, router)
        if (
            Path(identity["checkpoint_path"]) != PARENT_PATH
            or identity["checkpoint_sha256"] != PARENT_SHA
        ):
            raise ValueError("parent selection drift")
    else:
        backend = legacy.raw_boxfree_backend(loaded.backend, router)
        identity = {
            "raw_native_model": True,
            "MoSDeR_method_loaded": False,
            "router_mode": "uniform_no_plugin_no_pixel_modification",
        }
    loaded.backend = backend
    freeze_all(backend)
    return backend, identity


def prompt_contract():
    return {
        "raw_conditions": list(RAW_CONDITIONS),
        "generation": GENERATION,
        "generic_instruction": GENERIC,
        "ending": ENDING,
        "camera_template": CAMERA_TEMPLATE,
        "object_template": OBJECT_TEMPLATE,
        "state_enum_translation": STATE_TEXT,
        "auxiliary_prompt_rule": "original prompt + two newlines + generic + newline + camera line + newline + object line + newline + ending",
        "generic_values": {"camera": "unknown", "object": "unknown"},
        "factor_suffix": FACTOR_SUFFIX,
        "factor_prompt_rule": "exact original question_stem + two newlines + factor_suffix; no facts or QA options",
        "router_stem_rule": "explicit unchanged original question_stem for every request",
        "invalid_reference_field": "unknown",
        "source_prediction_error": "predicted_both unavailable; no fallback",
        "target_entities": "no caption/ref-derived target name is used",
        "fixed_parent_path": str(PARENT_PATH),
        "fixed_parent_sha256": PARENT_SHA,
        "fixed_grounder_path": str(GROUND_PATH),
        "fixed_grounder_sha256": GROUND_SHA,
        "no_training": True,
        "new_diagnostic_not_frozen_method_promotion": True,
    }


HERE = RUNTIME_DIR
NEW = DIAGNOSTIC_DIR
OLD = GROUNDING_DIR
WORK = EXPERIMENT_DIR
