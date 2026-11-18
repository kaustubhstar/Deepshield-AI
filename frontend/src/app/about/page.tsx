export default function AboutPage() {
  return (
    <article className="prose prose-invert max-w-3xl space-y-6 text-slate-300">
      <h1 className="text-3xl font-semibold text-white">Methodology and limitations</h1>
      <p>
        The installed detector is a compact face CNN (artifact_cnn). Training can also build EfficientNet-B0,
        but this copy&apos;s weights are the smaller network. The checkpoint file is still named
        deepshield_efficientnet_b0.pt because an older trainer always used that name. The server loads the
        architecture recorded in the metadata and checks that the tensors match.
      </p>
      <h2 className="text-xl font-semibold text-white">Scores vs accuracy</h2>
      <p>
        A result shows REAL or FAKE and the model&apos;s confidence for that label, from 0% to 100%. Validation
        accuracy on the training machine was about 67%, and the image files are not on this Mac, so no new held-out
        test score was measured here. The percentage on one photo is not the chance that the label is correct.
      </p>
      <h2 className="text-xl font-semibold text-white">Video aggregation</h2>
      <p>
        Videos use evenly spaced frames and the same model. The overall label uses the middle FAKE score. If too
        few frames can be read, the page does not invent REAL or FAKE. Nearby frames are similar, so a video score
        is still only a model confidence score.
      </p>
      <h2 className="text-xl font-semibold text-white">Privacy</h2>
      <p>
        Uploads are processed in memory or short-lived temp files and are not kept as a media archive. History stays in
        localStorage on the current device.
      </p>
      <h2 className="text-xl font-semibold text-white">What this cannot do</h2>
      <ul className="list-disc pl-5">
        <li>Guarantee 97% accuracy until you train and evaluate on a held-out, leakage-safe split.</li>
        <li>Detect every face swap, voice clone, mask, or screen replay from a laptop webcam.</li>
        <li>Replace forensic examination or legal expert analysis.</li>
      </ul>
    </article>
  );
}
