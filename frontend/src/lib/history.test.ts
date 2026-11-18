import { describe, expect, it } from "vitest";
import { summarize } from "./history";
import type { HistoryItem } from "./types";
import { predictionLabel } from "./types";

function item(partial: Partial<HistoryItem>): HistoryItem {
  return {
    id: "1",
    type: "image",
    timestamp: "2026-01-01T00:00:00.000Z",
    prediction: "likely_authentic",
    score: 0.1,
    processing_time_ms: 10,
    model_version: "test",
    ...partial,
  };
}

describe("history summary", () => {
  it("does not invent counts", () => {
    const summary = summarize([]);
    expect(summary.total).toBe(0);
    expect(summary.avgMs).toBeNull();
  });

  it("counts stored predictions only", () => {
    const summary = summarize([
      item({ id: "a", prediction: "likely_manipulated", processing_time_ms: 20 }),
      item({ id: "b", prediction: "analysis_unavailable", processing_time_ms: 0 }),
    ]);
    expect(summary.total).toBe(2);
    expect(summary.manipulated).toBe(1);
    expect(summary.unavailable).toBe(1);
    expect(summary.avgMs).toBe(20);
  });
});

describe("labels", () => {
  it("maps API enums to display text", () => {
    expect(predictionLabel("likely_manipulated")).toBe("FAKE");
    expect(predictionLabel("likely_authentic")).toBe("REAL");
    expect(predictionLabel("analysis_unavailable")).toBe("Analysis unavailable");
  });
});
