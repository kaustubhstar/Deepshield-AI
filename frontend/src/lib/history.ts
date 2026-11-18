import type { DetectionResponse, HistoryItem } from "./types";

const KEY = "deepshield.history.v1";
const LIMIT = 40;

export function loadHistory(): HistoryItem[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as HistoryItem[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveHistoryItem(item: Omit<HistoryItem, "id" | "timestamp"> & { timestamp?: string }) {
  const next: HistoryItem = {
    ...item,
    id: crypto.randomUUID(),
    timestamp: item.timestamp ?? new Date().toISOString(),
  };
  const items = [next, ...loadHistory()].slice(0, LIMIT);
  window.localStorage.setItem(KEY, JSON.stringify(items));
  return items;
}

export function clearHistory() {
  window.localStorage.removeItem(KEY);
}

export function summarize(items: HistoryItem[]) {
  const completed = items.filter((item) => item.prediction !== "analysis_unavailable");
  const counts = {
    total: items.length,
    manipulated: items.filter((i) => i.prediction === "likely_manipulated").length,
    authentic: items.filter((i) => i.prediction === "likely_authentic").length,
    inconclusive: items.filter((i) => i.prediction === "inconclusive").length,
    unavailable: items.filter((i) => i.prediction === "analysis_unavailable").length,
  };
  const avgMs =
    completed.length === 0
      ? null
      : Math.round(completed.reduce((sum, item) => sum + item.processing_time_ms, 0) / completed.length);
  return { ...counts, avgMs };
}

export function recordFromResponse(type: HistoryItem["type"], response: DetectionResponse) {
  return saveHistoryItem({
    type,
    prediction: response.prediction,
    score: response.score,
    confidence_percent: response.confidence_percent,
    processing_time_ms: response.processing_time_ms,
    model_version: response.model_version,
  });
}
