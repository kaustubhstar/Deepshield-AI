import type { DetectionResponse } from "@/lib/types";
import { formatPercent, presentResult } from "@/lib/result";

const CONFIDENCE_NOTE =
  "This percentage is the model's confidence score for the chosen label. It is not a calibrated chance that the label is correct.";

export function ResultPanel({
  result,
  fileName,
  previewUrl,
  onAnalyzeAnother,
}: {
  result: DetectionResponse;
  fileName?: string | null;
  previewUrl?: string | null;
  onAnalyzeAnother?: () => void;
}) {
  const presented = presentResult(result);
  const tone = presented?.verdict === "FAKE" ? "text-rose-300" : "text-emerald-300";
  const bar = presented?.verdict === "FAKE" ? "bg-rose-400" : "bg-emerald-400";

  return (
    <section className="panel p-5 sm:p-6" aria-live="polite">
      <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Detection result</p>
      {presented ? (
        <>
          <h2 className={`mt-3 text-4xl font-semibold tracking-wide sm:text-5xl ${tone}`}>{presented.verdict}</h2>
          <p className={`mt-2 text-2xl font-semibold sm:text-3xl ${tone}`}>
            {formatPercent(presented.confidencePercent)}% model confidence
          </p>
          <div className="mt-4 h-3 overflow-hidden rounded-full bg-slate-800" aria-hidden>
            <div
              className={`h-full ${bar}`}
              style={{ width: `${Math.min(100, Math.max(0, presented.confidencePercent))}%` }}
            />
          </div>
          <p className="mt-4 text-base text-slate-100">{presented.explanation}</p>
          <p className="mt-2 text-sm text-slate-400">{CONFIDENCE_NOTE}</p>
        </>
      ) : (
        <>
          <h2 className="mt-3 text-2xl font-semibold text-amber-200">No result</h2>
          <p className="mt-3 text-slate-200">
            {result.explanation || result.warnings[0] || "The model did not return a REAL or FAKE result."}
          </p>
        </>
      )}
      <dl className="mt-5 grid gap-3 text-sm sm:grid-cols-2">
        {fileName ? (
          <div>
            <dt className="text-slate-400">File</dt>
            <dd className="break-all">{fileName}</dd>
          </div>
        ) : null}
        <div>
          <dt className="text-slate-400">Status</dt>
          <dd className="capitalize">{result.status}</dd>
        </div>
      </dl>
      {previewUrl ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={previewUrl} alt="Uploaded preview" className="mt-4 max-h-80 w-full rounded-2xl object-contain" />
      ) : null}
      {presented && result.warnings.length > 0 ? (
        <ul className="mt-4 list-disc space-y-1 pl-5 text-sm text-amber-100">
          {result.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      ) : null}
      {onAnalyzeAnother ? (
        <button
          type="button"
          onClick={onAnalyzeAnother}
          className="mt-5 w-full rounded-full border border-sky-400/40 px-5 py-3 sm:w-auto"
        >
          Analyse another image
        </button>
      ) : null}
    </section>
  );
}
