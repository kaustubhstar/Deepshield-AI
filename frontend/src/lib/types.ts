export type Prediction =
  | "likely_manipulated"
  | "likely_authentic"
  | "inconclusive"
  | "analysis_unavailable";

export type AnalysisStatus = "completed" | "unavailable" | "failed";

export interface FrameResult {
  timestamp_sec: number;
  prediction: Prediction;
  score: number | null;
  face_detected: boolean;
  warning: string | null;
}

export interface DetectionResponse {
  status: AnalysisStatus;
  prediction: Prediction;
  verdict: "REAL" | "FAKE" | null;
  confidence_percent: number | null;
  explanation: string | null;
  score: number | null;
  score_type: "model_score" | "calibrated_probability";
  model_name: string;
  model_version: string;
  processing_time_ms: number;
  warnings: string[];
  evidence?: Record<string, unknown> | null;
  frames_sampled?: number;
  frames_analyzed?: number;
  frame_results?: FrameResult[];
  aggregation_method?: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  environment: string;
}

export interface ReadyResponse {
  ready: boolean;
  model_loaded: boolean;
  checkpoint_present: boolean;
  reason: string | null;
}

export interface ModelInfoResponse {
  loaded: boolean;
  model_name: string | null;
  model_version: string | null;
  architecture: string | null;
  device: string | null;
  checkpoint_path: string | null;
  score_type: string;
  authentic_threshold: number | null;
  manipulated_threshold: number | null;
  classes: string[] | null;
  metrics: Record<string, unknown> | null;
  notes: string[];
}

export interface HistoryItem {
  id: string;
  type: "image" | "video" | "live";
  timestamp: string;
  prediction: Prediction;
  score: number | null;
  confidence_percent?: number | null;
  processing_time_ms: number;
  model_version: string;
}

export function predictionLabel(value: Prediction): string {
  switch (value) {
    case "likely_manipulated":
      return "FAKE";
    case "likely_authentic":
      return "REAL";
    case "inconclusive":
      return "Not enough evidence";
    default:
      return "Analysis unavailable";
  }
}
