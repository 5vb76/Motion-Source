"""Short-event multiple-choice QA inputs, inference, and result storage."""

from .contract import (
    ALLOWED_DATA_ROLES,
    ARMS,
    SCORING_STRATEGIES,
    IndependentLabel,
    PredictedState,
    QuestionSpec,
)
from .evaluator import score_question
from .manifest import load_question_assets, materialize_target_visible
from .runner import run_assets

__all__ = [
    "ALLOWED_DATA_ROLES",
    "ARMS",
    "SCORING_STRATEGIES",
    "IndependentLabel",
    "PredictedState",
    "QuestionSpec",
    "load_question_assets",
    "materialize_target_visible",
    "run_assets",
    "score_question",
]
