# Dataset guide (you must obtain data yourself)

DeepShield AI does **not** download datasets. You must confirm the license and download files on a machine you control.

## Recommended academic options

Record the name, URL, license, and date of access in `models/model_metadata.json` when you train.

1. **FaceForensics++**
   - Source: https://github.com/ondyari/FaceForensics
   - Typical labels after face crop: authentic original frames vs Face2Face / FaceSwap / NeuralTextures / Deepfakes.
   - License: research request / academic terms. Follow the authors' form; do not redistribute.
   - Split by **source video id**, not by random frames.

2. **Celeb-DF (v2)**
   - Source: https://github.com/yuezunli/celeb-deepfakeforensics
   - Real celebrity videos vs corresponding deepfake videos.
   - License: research use; request access from the authors.

3. **Deepfake Detection Challenge (DFDC) preview**
   - Source: https://ai.meta.com/datasets/dfdc/
   - Large and noisy. Use only if you can store it and respect the license.

## Manifest format

Create a CSV named `manifest.csv`:

```text
path,label,source_id
C:/data/real/videoA/frame_0001.jpg,authentic,videoA
C:/data/fake/videoA_swap/frame_0001.jpg,manipulated,videoA_swap
```

Rules:

- `label` is only `authentic` or `manipulated`.
- `source_id` is the originating video (or identity). All frames from one video share one id.
- Training code groups by `source_id` then splits 70% / 15% / 15% so neighboring frames cannot leak into the test set.

## Suggested preprocessing

1. Extract frames at a low rate (for example 1 fps) from each video.
2. Detect the largest face and crop with ~25% padding (the training dataset can be pre-cropped; the API also crops at inference).
3. Save JPEG/PNG crops at least 128 px on the short side.
4. Keep a mapping from crop path → source video → manipulation method → compression level if you want extra evaluation slices.

## Target accuracy

97% test accuracy is a **goal**, not a shipped result. After `training/evaluate.py` finishes, report the numbers in `models/test_metrics.json` unchanged.

## Small fixture data

Automated tests use tiny synthetic PNGs. They are not a deepfake dataset and must not be cited as experimental results.
