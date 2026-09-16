"""CPU checks for model interfaces and training behavior used by the entry points."""

import pytest
import torch
from torch import nn

from mosder_final_v1.bridge import (
    BoundedTemporalSourceResidual,
    DualSourceExplicitResidual,
    inject_route_sources,
)
from mosder_final_v1.core import MoSDeRMotionSourceCore
from mosder_final_v1.language import canonical_answer, parse_four_line_language
from mosder_final_v1.routing import (
    FactorRoute,
    FactorSpanDecision,
    SourceConditionedTriLoRALinear,
)
from mosder_final_v1.training import configure_training_stage
from mosder_fgr_runner_candidate_v3.sampler import (
    FullCoverageStateSampler,
    SampleRef,
    SamplerContractError,
)


@pytest.mark.parametrize("state", ["neither", "camera_only", "object_only", "both"])
def test_canonical_answers_preserve_state(state):
    answer = canonical_answer(state)
    assert len(answer.splitlines()) == 4
    assert parse_four_line_language(answer).state == state


def test_source_branches_depend_on_their_own_inputs():
    torch.manual_seed(7)
    model = MoSDeRMotionSourceCore(hidden_size=64, rank=8)
    target = torch.randn(2, 10, 64)
    context = torch.randn_like(target)
    original = model(target, context)
    changed = model(target + torch.randn_like(target), context)
    assert original.camera_source.shape == (2, 10, 64)
    torch.testing.assert_close(original.camera_source, changed.camera_source)
    assert not torch.equal(original.object_source, changed.object_source)


@pytest.mark.parametrize("stage", ["F", "G", "R", "FROZEN"])
def test_stage_changes_train_only_the_expected_modules(stage):
    model = nn.ModuleDict(
        {
            "source": MoSDeRMotionSourceCore(hidden_size=64, rank=8),
            "projection": SourceConditionedTriLoRALinear(
                nn.Linear(64, 64), source_dim=64
            ),
            "decision": FactorSpanDecision(),
            "residual": DualSourceExplicitResidual(64),
            "backbone": nn.Linear(64, 64),
        }
    )
    configure_training_stage(
        stage,
        installed_model=model,
        source_core=model["source"],
        wrappers={"projection": model["projection"]},
        decision=model["decision"],
        residual=model["residual"],
    )
    active = {name for name, value in model.named_parameters() if value.requires_grad}
    if stage == "F":
        assert any(name.startswith("source.") for name in active)
        assert {"decision.b_c", "decision.b_o"} <= active
        assert not any(
            "shared_" in name or name.startswith("residual.") for name in active
        )
    elif stage == "G":
        assert active == {"projection.shared_A.weight", "projection.shared_B.weight"}
    elif stage == "R":
        assert len(active) == 8
        assert all(name.startswith("residual.") for name in active)
    else:
        assert not active
    assert not any(name.startswith("backbone.") or ".base." in name for name in active)


@pytest.mark.parametrize("route", list(FactorRoute))
@pytest.mark.parametrize("frame_count", [10, 20])
def test_visual_injection_respects_route_and_region(route, frame_count):
    raw = torch.zeros(1, frame_count, 1, 2, 64)
    target_mask = torch.zeros(raw.shape[:-1], dtype=torch.bool)
    target_mask[..., 0] = True
    camera = torch.full((1, 10, 64), 2.0)
    obj = torch.full_like(camera, 4.0)
    result = inject_route_sources(
        raw_native_tokens=raw,
        target_mask=target_mask,
        context_mask=~target_mask,
        camera_source=camera,
        object_source=obj,
        route=route,
        explicit_residual=DualSourceExplicitResidual(64),
    )
    expected = torch.zeros_like(raw)
    if route in (FactorRoute.CAMERA_FACTOR, FactorRoute.FULL_LANGUAGE):
        expected[~target_mask] = 0.1
    if route in (FactorRoute.OBJECT_FACTOR, FactorRoute.FULL_LANGUAGE):
        expected[target_mask] = 0.2
    torch.testing.assert_close(result.output_native_tokens, expected)


def test_explicit_residual_trains_without_backpropagating_into_source():
    torch.manual_seed(11)
    residual = BoundedTemporalSourceResidual(64)
    source = torch.randn(1, 10, 64, requires_grad=True)
    assert torch.count_nonzero(residual(source)) == 0
    with torch.no_grad():
        residual.up.weight.normal_(std=0.05)
    output = residual(source)
    assert output.abs().max() <= 0.1
    output.square().sum().backward()
    assert source.grad is None
    assert torch.count_nonzero(residual.up.weight.grad) > 0


def test_sampler_resume_preserves_remaining_order_and_rejects_changed_membership():
    states = ("neither", "camera_only", "object_only", "both")
    rows = [SampleRef(str(index), states[index % 4], "fixture") for index in range(12)]
    sampler = FullCoverageStateSampler(rows, seed=19, epochs=2)
    prefix = [sampler.next_index() for _ in range(7)]
    state = sampler.state_dict()
    restored = FullCoverageStateSampler(rows, seed=19, epochs=2)
    restored.load_state_dict(state)
    remaining = list(iter(sampler.next_index, None))
    assert remaining == list(iter(restored.next_index, None))
    all_indices = prefix + remaining
    assert sorted(all_indices[:12]) == sorted(all_indices[12:]) == list(range(12))
    changed = [SampleRef("changed", rows[0].state, "fixture"), *rows[1:]]
    with pytest.raises(SamplerContractError):
        FullCoverageStateSampler(changed, seed=19, epochs=2).load_state_dict(state)
