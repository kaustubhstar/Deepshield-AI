# DeepShield AI — Real-Time AI-Powered Deepfake Detection

## Abstract

DeepShield AI is a college Project-Based Learning web application that screens images, short videos, and selected live-camera frames for possible facial manipulation. A Next.js interface talks to a FastAPI service that runs an EfficientNet-B0 binary classifier (authentic vs manipulated). This repository ships the full software pipeline. It does **not** ship a trained deepfake checkpoint, so production inference returns `analysis_unavailable` until you train and install weights.

## Problem statement

Synthetic face videos and images are easy to generate and hard for people to judge. Students need a transparent detector that shows real model outputs, not a demo that hardcodes 97% confidence.

## Objectives

1. Detect possibly manipulated still images through a trained CNN.
2. Sample video frames and aggregate scores with a documented method.
3. Analyze selected webcam frames on laptop and compatible phones.
4. Keep scores honest: dataset accuracy is not per-file confidence.
5. Document training, evaluation, limits, and deployment.

## Existing system limitations

Many public demos either wrap an ImageNet object classifier, show random percentages, or hide that no task-specific model is loaded. Forensic tools also fail on compression, unseen generators, and webcam replays.

## Proposed system

```text
Browser UI → HTTPS JSON/multipart API → one loaded PyTorch model → face crop → logits → softmax → REAL or FAKE
```

The installed weights are a compact `artifact_cnn`. EfficientNet-B0 is still a training option, but it is not the checkpoint currently loaded. History stays in the browser. Media is not archived.

## Architecture

```text
Phone / laptop browser (Next.js)
        |  HTTPS multipart uploads
        v
FastAPI (Uvicorn)
  /health  /ready  /model-info
  /api/detect/image|video|frame
        |
        v
EfficientNet-B0 (optional training architecture) or the installed artifact CNN (2 logits) + OpenCV Haar face crop
```

## Modules

| Module | Role |
| --- | --- |
| `frontend/src/app` | Pages: home, dashboard, image, video, live, history, about |
| `backend/app/api` | HTTP endpoints |
| `backend/app/ml` | Architecture, preprocess, faces, scores, video, metrics |
| `backend/training` | Train and evaluate |
| `backend/models` | Installed artifact_cnn checkpoint (`deepshield_efficientnet_b0.pt` plus `model_metadata.json`) |

## Algorithms

1. **Transfer learning.** EfficientNet-B0 is a CNN that already knows generic visual features. Training replaces the last layer with 2 outputs and updates weights on a deepfake dataset.
2. **Face crop.** Haar cascade finds the largest face. If none is found, the full frame is classified and a warning is added.
3. **Score.** Softmax of the two logits. The higher class is shown as REAL or FAKE. The percentage is that class's model confidence score. It is not a calibrated probability.
4. **Video.** Uniform frame indices, per-frame scores, then the median. Too few readable frames produces no REAL/FAKE label.

## Dataset information

See `docs/DATASET.md`. You must obtain a licensed set (FaceForensics++, Celeb-DF, or similar). Splits group by `source_id` so frames from one video cannot appear in both train and test.

## CNN and transfer learning (viva notes)

A CNN slides small filters over an image to find edges, textures, then object-like patterns. EfficientNet-B0 is a compact CNN. Transfer learning starts from ImageNet weights (optional `--pretrained-backbone`) and then learns “real face vs manipulated face”. Using ImageNet weights **without** this second training is not deepfake detection. This project refuses to serve an `untrained` checkpoint.

## Model evaluation results

The installed checkpoint was selected on the validation split. Those numbers are stored in `backend/models/model_metadata.json`:

- Accuracy: 0.673125
- Manipulated recall: 0.4475
- ROC-AUC: 0.76055

The 16,000-row manifest is balanced and its official train, validation, and test rows do not share a `source_id`. The image files themselves are not on this Mac, so `training/evaluate.py` was not run and there is no held-out `test_metrics.json`. Do not report the validation numbers as a test result. 97% is a target, not a result.

## API endpoints

| Method | Path | Meaning |
| --- | --- | --- |
| GET | `/health` | Process is up |
| GET | `/ready` | Compatible trained model is loaded |
| GET | `/model-info` | Architecture, version, thresholds, notes |
| POST | `/api/detect/image` | Image inference |
| POST | `/api/detect/video` | Synchronous short-video analysis |
| POST | `/api/detect/video/jobs` | Background video job |
| GET | `/api/detect/video/jobs/{id}` | Job poll |
| POST | `/api/detect/frame` | Live camera still |

## Setup (simple language)

### 1. Tools on this Mac

- Python 3.12 from Homebrew: `/opt/homebrew/bin/python3.12`
- Project environment: `backend/.venv312`
- Node.js 20 and npm

### 2. Backend

```text
cd backend
/opt/homebrew/bin/python3.12 -m venv .venv312
.venv312/bin/python -m pip install -r requirements.txt
cp .env.example .env
.venv312/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000/health. `/ready` is true when `models/deepshield_efficientnet_b0.pt` and `models/model_metadata.json` load. The current weights are `artifact_cnn`.

Use http://localhost:3000 for the site. CORS also allows http://127.0.0.1:3000.

### 3. Frontend

```text
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

Open http://localhost:3000. An image result shows REAL or FAKE and a 0–100% model confidence score from the API. If the model is missing, the page shows an error and does not invent a score.

### 4. Tests

```text
cd backend
.venv312/bin/python -m pytest
.venv312/bin/ruff check app tests training
```

```text
cd frontend
npm test
npm run typecheck
npm run build
```

### 5. Optional fixture

This trains on synthetic color placeholders, not deepfakes. Do not report that accuracy as deepfake performance. It would replace the installed checkpoint, so skip it unless you intend to overwrite `backend/models`.

### 6. Training later

When the image files are on this machine:

```text
cd backend
.venv312/bin/python training/train.py --manifest path/to/manifest.csv --architecture artifact_cnn --epochs 8 --input-size 128 --output-dir models
.venv312/bin/python training/evaluate.py --manifest path/to/manifest.csv
```

Set `CHECKPOINT_FILENAME` to the saved `deepshield_<architecture>.pt` name, then restart Uvicorn. `/ready` is true only when the version is not `untrained` and the weight shapes match the metadata.

## Testing

```text
cd backend
.venv312/bin/python -m pytest
.venv312/bin/ruff check app tests training
```

```text
cd frontend
npm test
```

Fixture weights in pytest are **not** production detectors.

## Deployment

See `docs/DEPLOYMENT.md`. For phones, `NEXT_PUBLIC_API_URL` must be a public HTTPS API. `localhost` on a phone is the phone itself.

## Limitations

- The installed detector is artifact_cnn, validation accuracy about 67%. It is not a finished deepfake detector.
- The image archive is not on this Mac, so held-out test metrics were not recomputed.
- Haar faces miss profile views and tiny faces.
- Video median is not a probability.
- Webcam screening is weak against replay and masks.
- 97% accuracy is unmet.
- A model confidence score is not the chance that REAL or FAKE is correct.

## Future scope

Calibration (temperature scaling), identity-aware splits, generator-wise evaluation, stronger face detectors, optional accounts with PostgreSQL, and a smaller ONNX runtime for cheap hosts.

## References and licenses

Cite only sources you actually used. Starter references (read the licenses before download):

- Tan & Le, EfficientNet, ICML 2019.
- Rössler et al., FaceForensics++, ICCV 2019. https://github.com/ondyari/FaceForensics
- Li et al., Celeb-DF. https://github.com/yuezunli/celeb-deepfakeforensics
- FastAPI, Next.js, PyTorch, OpenCV, torchvision documentation.

Do not invent papers or accuracy numbers for the viva.

## Phase notes

Phase 1 inspected an empty workspace. Later phases added the frontend, FastAPI service, ML package, video/live paths, history, tests, and docs. Inference is implemented but **gated** on a real checkpoint.
