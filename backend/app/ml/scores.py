"""Map real model logits to a REAL or FAKE label.

The percentage is the softmax score of the chosen class. It is a model
confidence score. It is not a calibrated probability that the label is correct.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn.functional import softmax

from app.ml.metadata import ModelMetadata
from app.schemas.detection import PredictionLabel, VerdictLabel

CONFIDENCE_NOTE = (
    "This percentage is the model's confidence score for the chosen label. "
    "It is not a calibrated probability that the label is correct."
)


@dataclass
class ClassDecision:
    prediction: PredictionLabel
    verdict: VerdictLabel
    manipulated_score: float
    authentic_score: float
    confidence_percent: float
    explanation: str


def logits_to_manipulated_score(logits: torch.Tensor, metadata: ModelMetadata) -> float:
    """Convert raw 2-class logits to the manipulated-class softmax score in [0, 1]."""
    decision = decision_from_logits(logits, metadata, subject="image")
    return decision.manipulated_score


def decision_from_logits(logits: torch.Tensor, metadata: ModelMetadata, subject: str) -> ClassDecision:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.shape[-1] != 2:
        raise ValueError("Expected binary logits with 2 classes.")
    probabilities = softmax(logits, dim=-1)[0]
    authentic = float(probabilities[metadata.authentic_index].item())
    manipulated = float(probabilities[metadata.manipulated_index].item())
    return decision_from_scores(authentic, manipulated, subject)


def decision_from_scores(authentic_score: float, manipulated_score: float, subject: str) -> ClassDecision:
    """Choose the class with the higher model score. A tie goes to FAKE."""
    if not 0.0 <= authentic_score <= 1.0 or not 0.0 <= manipulated_score <= 1.0:
        raise ValueError("Class scores must be inside [0, 1].")
    if manipulated_score >= authentic_score:
        verdict: VerdictLabel = "FAKE"
        prediction: PredictionLabel = "likely_manipulated"
        chosen = manipulated_score
        quality = "manipulated"
    else:
        verdict = "REAL"
        prediction = "likely_authentic"
        chosen = authentic_score
        quality = "real"
    return ClassDecision(
        prediction=prediction,
        verdict=verdict,
        manipulated_score=manipulated_score,
        authentic_score=authentic_score,
        confidence_percent=round(chosen * 100, 1),
        explanation=_explanation(subject, quality),
    )


def decision_from_manipulated_score(manipulated_score: float, subject: str) -> ClassDecision:
    """Use a single manipulated-class score when the paired authentic score is its complement."""
    if not 0.0 <= manipulated_score <= 1.0:
        raise ValueError("Manipulated score must be inside [0, 1].")
    return decision_from_scores(1.0 - manipulated_score, manipulated_score, subject)


def _explanation(subject: str, quality: str) -> str:
    if subject == "video":
        return (
            f"This video appears more likely to be {quality} based on the model's analysis of the sampled frames."
        )
    if subject == "camera frame":
        return f"This camera frame appears more likely to be {quality} based on the model's analysis."
    return f"This image appears more likely to be {quality} based on the model's analysis."
