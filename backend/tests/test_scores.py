import torch

from app.ml.metadata import ModelMetadata
from app.ml.scores import decision_from_logits, decision_from_manipulated_score, logits_to_manipulated_score


def test_softmax_maps_manipulated_index():
    metadata = ModelMetadata()
    logits = torch.tensor([[-4.0, 4.0]])
    score = logits_to_manipulated_score(logits, metadata)
    assert score > 0.9


def test_higher_manipulated_logit_is_fake():
    metadata = ModelMetadata()
    decision = decision_from_logits(torch.tensor([[-4.0, 4.0]]), metadata, "image")
    assert decision.verdict == "FAKE"
    assert decision.prediction == "likely_manipulated"
    assert decision.confidence_percent > 90
    assert "manipulated" in decision.explanation
    assert decision.manipulated_score > 0.9


def test_higher_authentic_logit_is_real():
    metadata = ModelMetadata()
    decision = decision_from_logits(torch.tensor([[4.0, -4.0]]), metadata, "image")
    assert decision.verdict == "REAL"
    assert decision.prediction == "likely_authentic"
    assert decision.confidence_percent > 90
    assert decision.explanation == "This image appears more likely to be real based on the model's analysis."
    assert decision.manipulated_score < 0.1


def test_tie_uses_fake_and_reports_that_class_score():
    decision = decision_from_manipulated_score(0.5, "image")
    assert decision.verdict == "FAKE"
    assert decision.confidence_percent == 50.0


def test_selected_class_percent_is_not_always_the_fake_score():
    decision = decision_from_manipulated_score(0.2, "video")
    assert decision.verdict == "REAL"
    assert decision.confidence_percent == 80.0
    assert "video" in decision.explanation
