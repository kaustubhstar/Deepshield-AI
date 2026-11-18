import { describe, expect, it } from "vitest";
import { formatPercent, historyConfidence, presentResult } from "./result";
import type { DetectionResponse } from "./types";

function response(partial: Partial<DetectionResponse>): DetectionResponse {
  return {
    status: "completed",
    prediction: "likely_authentic",
    verdict: "REAL",
    confidence_percent: 87,
    explanation: "This image appears more likely to be real based on the model's analysis.",
    score: 0.13,
    score_type: "model_score",
    model_name: "deepshield_artifact_cnn",
    model_version: "test",
    processing_time_ms: 12,
    warnings: [],
    ...partial,
  };
}

describe("presentResult", () => {
  it("shows the backend verdict and percent", () => {
    const shown = presentResult(response({}));
    expect(shown).toEqual({
      verdict: "REAL",
      confidencePercent: 87,
      explanation: "This image appears more likely to be real based on the model's analysis.",
    });
  });

  it("does not invent a label when the model did not return one", () => {
    expect(presentResult(response({ status: "unavailable", verdict: null, confidence_percent: null, explanation: null }))).toBeNull();
    expect(presentResult(response({ verdict: null, explanation: "Not enough readable frames to choose REAL or FAKE." }))).toBeNull();
  });
});

describe("historyConfidence", () => {
  it("uses the stored selected-class percent", () => {
    expect(historyConfidence({ prediction: "likely_authentic", score: 0.13, confidence_percent: 87 })).toBe(87);
  });

  it("recovers the selected class from an older manipulated score", () => {
    expect(historyConfidence({ prediction: "likely_authentic", score: 0.2, confidence_percent: null })).toBe(80);
    expect(historyConfidence({ prediction: "likely_manipulated", score: 0.92, confidence_percent: null })).toBe(92);
    expect(historyConfidence({ prediction: "inconclusive", score: 0.4, confidence_percent: null })).toBeNull();
  });
});

describe("formatPercent", () => {
  it("keeps one decimal only when needed", () => {
    expect(formatPercent(87)).toBe("87");
    expect(formatPercent(87.5)).toBe("87.5");
  });
});
