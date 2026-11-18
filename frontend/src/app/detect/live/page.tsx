"use client";

import { ResultPanel } from "@/components/ResultPanel";
import { detectFrame } from "@/lib/api";
import { captureJpeg, listVideoDevices, permissionMessage, startStream, stopStream } from "@/lib/camera";
import { recordFromResponse } from "@/lib/history";
import type { DetectionResponse } from "@/lib/types";
import { useCallback, useEffect, useRef, useState } from "react";

export default function LiveDetectPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const inFlight = useRef(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState<string>("");
  const [facing, setFacing] = useState<"user" | "environment">("user");
  const [running, setRunning] = useState(false);
  const [intervalMs, setIntervalMs] = useState(2000);
  const [periodic, setPeriodic] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [latency, setLatency] = useState<number | null>(null);

  useEffect(() => {
    return () => stopStream(streamRef.current);
  }, []);

  async function start() {
    setError(null);
    try {
      const stream = await startStream(deviceId || undefined, deviceId ? undefined : facing);
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setRunning(true);
      const listed = await listVideoDevices();
      setDevices(listed);
    } catch (err) {
      setError(permissionMessage(err));
    }
  }

  function stop() {
    stopStream(streamRef.current);
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setRunning(false);
    setPeriodic(false);
  }

  const analyzeFrame = useCallback(async () => {
    if (!videoRef.current || inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    setError(null);
    const started = performance.now();
    try {
      const blob = await captureJpeg(videoRef.current);
      const response = await detectFrame(blob);
      setResult(response);
      setLatency(Math.round(performance.now() - started));
      recordFromResponse("live", response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Frame analysis failed.");
    } finally {
      inFlight.current = false;
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    if (!periodic || !running) return;
    const id = window.setInterval(() => {
      void analyzeFrame();
    }, intervalMs);
    return () => window.clearInterval(id);
  }, [periodic, running, intervalMs, analyzeFrame]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold">Live camera</h1>
        <p className="mt-2 text-slate-400">
          Permission is requested only after Start camera. Only captured still frames are sent. The feed is not recorded.
          Ordinary webcams cannot reliably detect every deepfake, replay, or mask attack.
        </p>
      </div>
      <video ref={videoRef} autoPlay playsInline muted className="w-full max-w-3xl rounded-2xl bg-black" />
      <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap">
        <button type="button" onClick={running ? stop : start} className="rounded-full bg-sky-400 px-6 py-3 font-medium text-slate-950">
          {running ? "Stop camera" : "Start camera"}
        </button>
        <button type="button" disabled={!running || busy} onClick={analyzeFrame} className="rounded-full border border-sky-400/30 px-6 py-3 disabled:opacity-40">
          {busy ? "Analyzing…" : "Capture frame"}
        </button>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={periodic} disabled={!running} onChange={(e) => setPeriodic(e.target.checked)} />
          Periodic analysis
        </label>
      </div>
      {devices.length > 1 ? (
        <label className="block text-sm">
          Camera
          <select
            className="mt-1 w-full rounded-xl bg-slate-900 p-3"
            value={deviceId}
            onChange={(event) => {
              setDeviceId(event.target.value);
              if (running) {
                stop();
                window.setTimeout(() => {
                  void start();
                }, 0);
              }
            }}
          >
            {devices.map((device) => (
              <option key={device.deviceId} value={device.deviceId}>
                {device.label || "Camera"}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      <div className="flex flex-col gap-3 sm:flex-row">
        <button
          type="button"
          className={`rounded-full border px-5 py-3 ${facing === "user" ? "border-sky-300 bg-sky-400/10" : "border-sky-400/30"}`}
          onClick={() => {
            setDeviceId("");
            setFacing("user");
          }}
        >
          Front camera
        </button>
        <button
          type="button"
          className={`rounded-full border px-5 py-3 ${facing === "environment" ? "border-sky-300 bg-sky-400/10" : "border-sky-400/30"}`}
          onClick={() => {
            setDeviceId("");
            setFacing("environment");
          }}
        >
          Rear camera
        </button>
      </div>
      <p className="text-sm text-slate-400">Stop and start the camera after switching front/rear or device.</p>
      <label className="block text-sm">
        Interval (ms)
        <input
          type="number"
          min={1000}
          step={500}
          value={intervalMs}
          onChange={(event) => setIntervalMs(Number(event.target.value))}
          className="mt-1 w-full rounded-xl bg-slate-900 p-3"
        />
      </label>
      {latency !== null ? <p className="text-sm text-slate-400">Round-trip latency: {latency} ms</p> : null}
      {error ? <p className="text-rose-300">{error}</p> : null}
      {result ? <ResultPanel result={result} /> : null}
    </div>
  );
}
