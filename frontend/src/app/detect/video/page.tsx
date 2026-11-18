"use client";

import { ResultPanel } from "@/components/ResultPanel";
import { getVideoJob, startVideoJob } from "@/lib/api";
import { recordFromResponse } from "@/lib/history";
import type { DetectionResponse } from "@/lib/types";
import { useMemo, useState } from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function VideoDetectPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [jobMessage, setJobMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResponse | null>(null);

  const chartData = useMemo(
    () =>
      (result?.frame_results ?? []).map((frame) => ({
        t: frame.timestamp_sec,
        score: frame.score ?? null,
      })),
    [result],
  );

  async function analyze() {
    if (!file) return;
    setBusy(true);
    setError(null);
    setResult(null);
    setProgress(0);
    setJobMessage("Uploading…");
    try {
      const created = await startVideoJob(file);
      setJobMessage(`Job ${created.job_id.slice(0, 8)}… queued`);
      for (;;) {
        const job = await getVideoJob(created.job_id);
        setProgress(job.progress);
        setJobMessage(job.message);
        if (job.job_status === "completed" || job.job_status === "failed") {
          setResult(job);
          if (job.job_status === "completed") {
            recordFromResponse("video", job);
          } else {
            setError(job.message || "Video analysis failed.");
          }
          break;
        }
        await new Promise((resolve) => window.setTimeout(resolve, 400));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold">Video detection</h1>
        <p className="mt-2 text-slate-400">
          MP4, WebM, or MOV. The backend checks sampled frames with the same model and shows REAL or FAKE from
          those scores. If too few frames can be read, it says so instead of inventing a label.
        </p>
      </div>
      <input
        type="file"
        accept="video/mp4,video/webm,video/quicktime,.mp4,.webm,.mov"
        onChange={(event) => {
          const next = event.target.files?.[0] ?? null;
          setFile(next);
          setPreview(next ? URL.createObjectURL(next) : null);
          setResult(null);
          setProgress(0);
        }}
      />
      {preview ? <video src={preview} controls className="w-full max-w-3xl rounded-2xl" /> : null}
      <button
        type="button"
        disabled={!file || busy}
        onClick={analyze}
        className="rounded-full bg-sky-400 px-6 py-3 font-medium text-slate-950 disabled:opacity-40"
      >
        {busy ? "Analyzing frames…" : "Analyze video"}
      </button>
      {busy ? (
        <div className="panel p-4">
          <p className="text-sm text-slate-300">{jobMessage}</p>
          <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-800">
            <div className="h-full bg-sky-400" style={{ width: `${Math.round(progress * 100)}%` }} />
          </div>
        </div>
      ) : null}
      {error ? <p className="text-rose-300">{error}</p> : null}
      {result ? (
        <>
          <ResultPanel result={result} />
          <section className="panel p-5">
            <h2 className="font-semibold">Frame timeline</h2>
            <p className="mt-1 text-sm text-slate-400">
              Sampled {result.frames_sampled ?? 0} frames, analyzed {result.frames_analyzed ?? 0}. Each point is that
              frame&apos;s FAKE model score, from 0 to 1.
            </p>
            {chartData.length > 0 ? (
              <div className="mt-4 h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <XAxis dataKey="t" stroke="#93a4bd" />
                    <YAxis domain={[0, 1]} stroke="#93a4bd" />
                    <Tooltip />
                    <Line type="monotone" dataKey="score" stroke="#38bdf8" dot connectNulls={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <p className="mt-3 text-slate-400">No frame scores to plot.</p>
            )}
          </section>
        </>
      ) : null}
    </div>
  );
}
