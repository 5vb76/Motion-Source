"""Query-conditioned spatial pooling over native visual tokens.

The router predicts a target weight for each token and pools target/context
features into ten temporal units. This module contains no model-loading code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SoftRoutingOutput:
    membership_logits: torch.Tensor  # [T,H,W]
    target_weights: torch.Tensor  # independent probabilities, not spatial softmax
    context_weights: torch.Tensor
    target_pooled: torch.Tensor  # [10,D]
    context_pooled: torch.Tensor
    unit_tokens: torch.Tensor
    unit_target_weights: torch.Tensor
    # Small weights permit absence, but without absence labels this is NOT a
    # calibrated probability that the queried target is absent or unknown.
    absence_calibrated: bool = False


def pool_soft_sources(native_tokens: torch.Tensor, weights: torch.Tensor):
    """Pool each frame before adjacent pairing; no full-visibility requirement.

    Clamp by one token rather than epsilon: all-zero target weights yield a
    zero target pool instead of normalizing arbitrarily tiny noise to unit mass.
    Context means non-target visual context, NOT guaranteed static background.
    """
    if native_tokens.ndim != 4 or native_tokens.shape[0] not in (10, 20):
        raise ValueError("native tokens must be [10|20,H,W,D]")
    if weights.shape != native_tokens.shape[:-1] or not weights.is_floating_point():
        raise ValueError("weights must be floating [T,H,W]")
    if not bool(torch.isfinite(weights).all()) or bool(
        ((weights < 0) | (weights > 1)).any()
    ):
        raise ValueError("weights must be finite in [0,1]")
    value, weights = native_tokens.float(), weights.float()
    context = 1 - weights
    target_pool = (value * weights[..., None]).sum((1, 2)) / weights.sum(
        (1, 2)
    ).clamp_min(1)[:, None]
    context_pool = (value * context[..., None]).sum((1, 2)) / context.sum(
        (1, 2)
    ).clamp_min(1)[:, None]
    if value.shape[0] == 20:
        target_pool = target_pool.reshape(10, 2, -1).mean(1)
        context_pool = context_pool.reshape(10, 2, -1).mean(1)
        unit_tokens = value.reshape(10, 2, *value.shape[1:]).mean(1)
        unit_weights = weights.reshape(10, 2, *weights.shape[1:]).mean(1)
    else:
        unit_tokens, unit_weights = value, weights
    return target_pool, context_pool, unit_tokens, unit_weights


class SoftQueryRouter(nn.Module):
    """Low-rank query/token matching in the frozen native embedding space.

    A token-wise visual bias is query-independent; the bilinear term contains
    query dependence. Query embeddings must be computed from the question
    stem only, never the teacher-forced answer or answer options.
    """

    def __init__(self, hidden_size: int, rank: int = 64):
        super().__init__()
        if not 0 < rank < hidden_size:
            raise ValueError("rank must be in (0,hidden_size)")
        self.hidden_size, self.rank = hidden_size, rank
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.visual_key = nn.Linear(hidden_size, rank, bias=False)
        self.query_key = nn.Linear(hidden_size, rank, bias=False)
        self.visual_bias = nn.Linear(rank, 1)

    def forward(
        self,
        native_tokens: torch.Tensor,
        query_embedding: torch.Tensor,
        *,
        mode: str = "query",
    ) -> SoftRoutingOutput:
        if native_tokens.ndim != 4 or native_tokens.shape[-1] != self.hidden_size:
            raise ValueError("native tokens must be [T,H,W,D]")
        if query_embedding.shape != (self.hidden_size,):
            raise ValueError("query embedding must have shape [D]")
        if not bool(torch.isfinite(native_tokens).all()) or not bool(
            torch.isfinite(query_embedding).all()
        ):
            raise ValueError("nonfinite visual/query features")
        if mode == "uniform":
            logits = torch.zeros(
                native_tokens.shape[:-1],
                device=native_tokens.device,
                dtype=torch.float32,
            )
        elif mode == "query":
            with torch.autocast(device_type=native_tokens.device.type, enabled=False):
                visual = self.visual_key(self.norm(native_tokens.float()))
                query = self.query_key(self.norm(query_embedding.float()))
                logits = (visual * query).sum(-1) / math.sqrt(self.rank)
                logits = logits + self.visual_bias(visual).squeeze(-1)
        else:
            raise ValueError("mode must be query or uniform")
        weights = logits.sigmoid()
        target, context, units, unit_weights = pool_soft_sources(native_tokens, weights)
        return SoftRoutingOutput(
            logits, weights, 1 - weights, target, context, units, unit_weights
        )


def box_grid_occupancy(
    normalized_box, height: int, width: int, *, device=None
) -> torch.Tensor:
    """Fraction of each native token cell occupied by one normalized xyxy box.

    This is rectangle-cell intersection area, not center-in-box thresholding.
    Thus even a positive-area box much smaller than one token keeps a positive
    target. Summed occupancy / (height*width) equals normalized box area.
    """
    if type(height) is not int or type(width) is not int or min(height, width) <= 0:
        raise ValueError("positive integer grid dimensions required")
    box = torch.as_tensor(normalized_box, device=device, dtype=torch.float32)
    if box.shape != (4,) or not bool(torch.isfinite(box).all()):
        raise ValueError("normalized xyxy box must be finite [4]")
    if bool(((box < 0) | (box > 1)).any()) or bool((box[2:] <= box[:2]).any()):
        raise ValueError("box must have positive area within [0,1]")
    x0 = torch.arange(width, device=box.device, dtype=box.dtype) / width
    y0 = torch.arange(height, device=box.device, dtype=box.dtype) / height
    x_overlap = (
        torch.minimum(x0 + 1 / width, box[2]) - torch.maximum(x0, box[0])
    ).clamp_min(0)
    y_overlap = (
        torch.minimum(y0 + 1 / height, box[3]) - torch.maximum(y0, box[1])
    ).clamp_min(0)
    return (y_overlap[:, None] * x_overlap[None, :] * (height * width)).clamp(0, 1)


def sparse_box_anchor_loss(
    logits: torch.Tensor, target_masks: torch.Tensor, known_frames: torch.Tensor
) -> torch.Tensor:
    """Balanced per-frame BCE on observed official-box anchors only.

    Boxes supervise coarse box occupancy, NOT true object segmentation. Every
    unannotated frame has zero loss, including missing/occluded/out-of-range
    frames. known_frames is a TRAIN-only evidence mask, never an input to the
    router. A full/empty observed mask is supported without dividing by zero.
    """
    if logits.ndim != 3 or target_masks.shape != logits.shape:
        raise ValueError("logits and targets must share [T,H,W]")
    if known_frames.shape != (logits.shape[0],) or known_frames.dtype != torch.bool:
        raise ValueError("known_frames must be bool [T]")
    if not bool(known_frames.any()):
        raise ValueError("no supervised anchors; caller must skip this example")
    predicted_logits, truth = (
        logits[known_frames],
        target_masks[known_frames].to(logits),
    )
    if not bool(torch.isfinite(truth).all()) or bool(((truth < 0) | (truth > 1)).any()):
        raise ValueError("observed targets must be finite in [0,1]")
    # Separate positive/negative logistic terms. Multiplying a soft-target BCE
    # by truth again would square/cross-weight target fractions incorrectly.
    positive_mass, negative_mass = truth.sum((1, 2)), (1 - truth).sum((1, 2))
    positive_loss = (F.softplus(-predicted_logits) * truth).sum(
        (1, 2)
    ) / positive_mass.clamp_min(torch.finfo(predicted_logits.dtype).tiny)
    negative_loss = (F.softplus(predicted_logits) * (1 - truth)).sum(
        (1, 2)
    ) / negative_mass.clamp_min(torch.finfo(predicted_logits.dtype).tiny)
    groups = (positive_mass > 0).to(logits) + (negative_mass > 0).to(logits)
    return ((positive_loss + negative_loss) / groups.clamp_min(1)).mean()
