import type { DetectionResponse, HistoryItem } from "./types";

export interface PresentedResult {
  verdict: "REAL" | "FAKE";
  confidencePercent: number;
  explanation: string;
}

/** Show only a label the backend actually returned. Never invent REAL or FAKE here. */
export function presentResult(result: DetectionResponse): PresentedResult | null {
  if (result.status !== "completed" || (result.verdict !== "REAL" && result.verdict !== "FAKE")) {
    return null;
  }
  if (result.confidence_percent === null || result.confidence_percent === undefined) return null;
  if (!result.explanation) return null;
  return {
    verdict: result.verdict,
    confidencePercent: result.confidence_percent,
    explanation: result.explanation,
  };
}

export function historyConfidence(item: Pick<HistoryItem, "prediction" | "score" | "confidence_percent">): number | null {
  if (item.confidence_percent !== null && item.confidence_percent !== undefined) {
    return item.confidence_percent;
  }
  if (item.score === null) return null;
  if (item.prediction === "likely_manipulated") return Math.round(item.score * 1000) / 10;
  if (item.prediction === "likely_authentic") return Math.round((1 - item.score) * 1000) / 10;
  return null;
}

export function formatPercent(value: number): string {
  const rounded = Math.round(value * 10) / 10;
  return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
}
