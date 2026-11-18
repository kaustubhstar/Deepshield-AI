"use client";

import { getHealth, getModelInfo, getReady } from "@/lib/api";
import type { ModelInfoResponse, ReadyResponse } from "@/lib/types";
import { useEffect, useState } from "react";

export function BackendStatus() {
  const [ready, setReady] = useState<ReadyResponse | null>(null);
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);
  const [healthOk, setHealthOk] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([getHealth(controller.signal), getReady(controller.signal), getModelInfo(controller.signal)])
      .then(([health, readyBody, model]) => {
        setHealthOk(health.status === "ok");
        setReady(readyBody);
        setInfo(model);
      })
      .catch(() => {
        setHealthOk(false);
        setError("Backend is unreachable. Start the FastAPI server or check NEXT_PUBLIC_API_URL.");
      });
    return () => controller.abort();
  }, []);

  return (
    <section className="panel p-5">
      <h2 className="text-lg font-semibold">Backend status</h2>
      <ul className="mt-3 space-y-2 text-sm">
        <li>API health: {healthOk === null ? "checking…" : healthOk ? "ok" : "down"}</li>
        <li>Model loaded: {ready ? String(ready.model_loaded) : "unknown"}</li>
        <li>Checkpoint present: {ready ? String(ready.checkpoint_present) : "unknown"}</li>
        {ready?.reason ? <li className="text-amber-200">{ready.reason}</li> : null}
        {info?.model_version ? <li>Version: {info.model_version}</li> : null}
        {error ? <li className="text-rose-300">{error}</li> : null}
      </ul>
    </section>
  );
}
