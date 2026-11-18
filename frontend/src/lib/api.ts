import type {
  DetectionResponse,
  HealthResponse,
  ModelInfoResponse,
  ReadyResponse,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

async function parseError(response: Response): Promise<never> {
  let detail = `Request failed (${response.status})`;
  try {
    const body = await response.json();
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    /* keep default */
  }
  throw new ApiError(detail, response.status);
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`, { signal });
  if (!response.ok) await parseError(response);
  return response.json();
}

export async function getReady(signal?: AbortSignal): Promise<ReadyResponse> {
  const response = await fetch(`${API_BASE}/ready`, { signal });
  if (!response.ok) await parseError(response);
  return response.json();
}

export async function getModelInfo(signal?: AbortSignal): Promise<ModelInfoResponse> {
  const response = await fetch(`${API_BASE}/model-info`, { signal });
  if (!response.ok) await parseError(response);
  return response.json();
}

async function postMedia(path: string, file: Blob, filename: string): Promise<DetectionResponse> {
  const body = new FormData();
  body.append("file", file, filename);
  const response = await fetch(`${API_BASE}${path}`, { method: "POST", body });
  if (!response.ok) await parseError(response);
  return response.json();
}

export function detectImage(file: File) {
  return postMedia("/api/detect/image", file, file.name);
}

export function detectVideo(file: File) {
  return postMedia("/api/detect/video", file, file.name);
}

export function detectFrame(blob: Blob) {
  return postMedia("/api/detect/frame", blob, "frame.jpg");
}

export async function startVideoJob(file: File): Promise<{ job_id: string; status: string }> {
  const body = new FormData();
  body.append("file", file, file.name);
  const response = await fetch(`${API_BASE}/api/detect/video/jobs`, { method: "POST", body });
  if (!response.ok) await parseError(response);
  return response.json();
}

export async function getVideoJob(jobId: string): Promise<DetectionResponse & { job_status: string; progress: number; message: string | null; job_id: string }> {
  const response = await fetch(`${API_BASE}/api/detect/video/jobs/${jobId}`);
  if (!response.ok) await parseError(response);
  return response.json();
}
