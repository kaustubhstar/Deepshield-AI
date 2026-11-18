import { Camera, ImageIcon, ShieldAlert, Video } from "lucide-react";
import Link from "next/link";

const ACTIONS = [
  { href: "/detect/image", title: "Image analysis", desc: "Upload JPEG, PNG, or WEBP.", icon: ImageIcon },
  { href: "/detect/video", title: "Video analysis", desc: "Sample frames from MP4 or WebM.", icon: Video },
  { href: "/detect/live", title: "Live camera", desc: "Analyze selected webcam frames.", icon: Camera },
];

export default function HomePage() {
  return (
    <div className="space-y-12">
      <section className="panel glow p-8 sm:p-12">
        <p className="text-xs uppercase tracking-[0.25em] text-sky-300">Real-time deepfake screening</p>
        <h1 className="mt-4 max-w-3xl text-4xl font-semibold leading-tight sm:text-5xl">
          DeepShield AI inspects faces with a CNN you train, not a hardcoded score.
        </h1>
        <p className="mt-5 max-w-2xl text-slate-300">
          This PBL project sends images, video frames, and optional camera captures to a FastAPI
          inference service. Until you install a trained checkpoint, the API reports analysis
          unavailable instead of inventing confidence numbers.
        </p>
        <div className="mt-8 flex flex-col gap-3 sm:flex-row">
          <Link href="/dashboard" className="rounded-full bg-sky-400 px-6 py-3 text-center font-medium text-slate-950">
            Get started
          </Link>
          <Link href="/about" className="rounded-full border border-sky-400/30 px-6 py-3 text-center">
            Methodology
          </Link>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        {ACTIONS.map((item) => (
          <Link key={item.href} href={item.href} className="panel p-6 hover:border-sky-300/40">
            <item.icon className="h-8 w-8 text-sky-300" aria-hidden />
            <h2 className="mt-4 text-xl font-semibold">{item.title}</h2>
            <p className="mt-2 text-sm text-slate-400">{item.desc}</p>
          </Link>
        ))}
      </section>

      <section className="panel p-6">
        <h2 className="text-xl font-semibold">How the detector works</h2>
        <ol className="mt-4 list-decimal space-y-2 pl-5 text-slate-300">
          <li>A face crop is taken when a face is found.</li>
          <li>The loaded CNN scores the picture as real or manipulated.</li>
          <li>The higher score is shown as REAL or FAKE, from 0% to 100%.</li>
          <li>That percentage is the model&apos;s confidence score, not a calibrated chance of being correct.</li>
        </ol>
      </section>

      <aside className="flex gap-3 rounded-2xl border border-amber-400/30 bg-amber-400/10 p-5 text-amber-100">
        <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0" aria-hidden />
        <p>
          AI detectors miss some fakes and flag some real media. Webcam analysis cannot reliably catch
          every face swap, replay, or mask. Do not treat a single score as proof.
        </p>
      </aside>
    </div>
  );
}
