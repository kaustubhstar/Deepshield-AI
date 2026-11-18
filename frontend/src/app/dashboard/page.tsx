"use client";

import { BackendStatus } from "@/components/BackendStatus";
import { loadHistory, summarize } from "@/lib/history";
import { historyConfidence } from "@/lib/result";
import { predictionLabel } from "@/lib/types";
import Link from "next/link";
import { useEffect, useState } from "react";

export default function DashboardPage() {
  const [summary, setSummary] = useState(summarize([]));
  const [recent, setRecent] = useState(loadHistory());

  useEffect(() => {
    const items = loadHistory();
    setRecent(items.slice(0, 8));
    setSummary(summarize(items));
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold">Dashboard</h1>
        <p className="mt-2 text-slate-400">
          Counts below come from this browser&apos;s analysis history. They are not live worldwide statistics.
        </p>
      </div>
      <BackendStatus />
      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[
          ["Stored analyses", summary.total],
          ["FAKE", summary.manipulated],
          ["REAL", summary.authentic],
          ["Avg latency (ms)", summary.avgMs ?? "n/a"],
        ].map(([label, value]) => (
          <div key={String(label)} className="panel p-5">
            <p className="text-sm text-slate-400">{label}</p>
            <p className="mt-2 text-2xl font-semibold">{value}</p>
          </div>
        ))}
      </section>
      <section className="grid gap-4 md:grid-cols-3">
        <Link className="panel p-5" href="/detect/image">
          Analyze an image
        </Link>
        <Link className="panel p-5" href="/detect/video">
          Analyze a video
        </Link>
        <Link className="panel p-5" href="/detect/live">
          Open live camera
        </Link>
      </section>
      <section className="panel p-5">
        <h2 className="text-lg font-semibold">Recent results in this browser</h2>
        {recent.length === 0 ? (
          <p className="mt-3 text-slate-400">No local history yet.</p>
        ) : (
          <ul className="mt-3 divide-y divide-white/10">
            {recent.map((item) => (
              <li key={item.id} className="flex flex-wrap items-center justify-between gap-2 py-3 text-sm">
                <span className="capitalize">{item.type}</span>
                <span>{predictionLabel(item.prediction)}</span>
                <span className="font-mono text-sky-200">
                  {historyConfidence(item) === null ? "n/a" : `${historyConfidence(item)}%`}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
