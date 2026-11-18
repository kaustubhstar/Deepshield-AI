"use client";

import { clearHistory, loadHistory } from "@/lib/history";
import { historyConfidence } from "@/lib/result";
import { predictionLabel } from "@/lib/types";
import { useEffect, useState } from "react";

export default function HistoryPage() {
  const [items, setItems] = useState(loadHistory());

  useEffect(() => {
    setItems(loadHistory());
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-semibold">Analysis history</h1>
          <p className="mt-2 max-w-2xl text-slate-400">
            Stored only in this browser. It is not synced across phones and laptops. Original media is not saved.
          </p>
        </div>
        <button
          type="button"
          className="rounded-full border border-rose-400/40 px-5 py-3 text-rose-200"
          onClick={() => {
            clearHistory();
            setItems([]);
          }}
        >
          Clear history
        </button>
      </div>
      {items.length === 0 ? (
        <p className="panel p-6 text-slate-400">No analyses stored yet.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead className="text-slate-400">
              <tr>
                <th className="p-3">When</th>
                <th className="p-3">Type</th>
                <th className="p-3">Prediction</th>
                <th className="p-3">Model confidence</th>
                <th className="p-3">ms</th>
                <th className="p-3">Model</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={item.id} className="border-t border-white/10">
                  <td className="p-3">{new Date(item.timestamp).toLocaleString()}</td>
                  <td className="p-3 capitalize">{item.type}</td>
                  <td className="p-3">{predictionLabel(item.prediction)}</td>
                  <td className="p-3 font-mono">
                    {historyConfidence(item) === null ? "n/a" : `${historyConfidence(item)}%`}
                  </td>
                  <td className="p-3">{item.processing_time_ms}</td>
                  <td className="p-3">{item.model_version}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
