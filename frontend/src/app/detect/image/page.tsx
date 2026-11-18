"use client";

import { ResultPanel } from "@/components/ResultPanel";
import { detectImage } from "@/lib/api";
import { recordFromResponse } from "@/lib/history";
import type { DetectionResponse } from "@/lib/types";
import { useRef, useState } from "react";

const ACCEPT = "image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp";

export default function ImageDetectPage() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResponse | null>(null);

  function onFile(next: File | null) {
    setResult(null);
    setError(null);
    setFile(next);
    setPreview(next ? URL.createObjectURL(next) : null);
  }

  async function analyze() {
    if (!file) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const response = await detectImage(file);
      setResult(response);
      recordFromResponse("image", response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed.");
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    onFile(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold">Image detection</h1>
        <p className="mt-2 text-slate-400">JPEG, PNG, and WEBP. The file is sent to your FastAPI backend only when you click Analyze.</p>
      </div>
      <section
        className="panel flex min-h-48 flex-col items-center justify-center gap-3 p-6 text-center"
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          onFile(event.dataTransfer.files[0] ?? null);
        }}
      >
        <p>Drag and drop an image, or browse.</p>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="max-w-full text-sm"
          onChange={(event) => onFile(event.target.files?.[0] ?? null)}
        />
      </section>
      <div className="flex flex-col gap-3 sm:flex-row">
        <button
          type="button"
          disabled={!file || busy}
          onClick={analyze}
          className="rounded-full bg-sky-400 px-6 py-3 font-medium text-slate-950 disabled:opacity-40"
        >
          {busy ? "Analyzing…" : "Analyze image"}
        </button>
        <button type="button" onClick={reset} className="rounded-full border border-sky-400/30 px-6 py-3">
          Reset
        </button>
      </div>
      {error ? <p className="text-rose-300">{error}</p> : null}
      {busy ? <p className="text-slate-300">Checking the image with the loaded model…</p> : null}
      {result ? (
        <ResultPanel result={result} fileName={file?.name} previewUrl={preview} onAnalyzeAnother={reset} />
      ) : preview ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={preview} alt="Selected preview" className="max-h-80 w-full rounded-2xl object-contain" />
      ) : null}
    </div>
  );
}
