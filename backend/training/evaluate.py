"""Evaluate a saved checkpoint on the held-out test sources.

Uses the same source-id split and seed as training. Do not change the seed
to hunt for a higher test score.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ml.architecture import estimate_model_mb, resolve_device
from app.ml.checkpoint import load_checkpoint_bundle
from app.ml.dataset import FaceCropDataset, read_manifest, split_by_source, splits_from_manifest
from app.ml.metrics import compute_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DeepShield checkpoint on test sources")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "deepshield_efficientnet_b0.pt")
    parser.add_argument("--metadata", type=Path, default=ROOT / "models" / "model_metadata.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "test_metrics.json")
    parser.add_argument("--cache-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = resolve_device(args.device)
    model, metadata, _ = load_checkpoint_bundle(args.checkpoint, args.metadata, device)
    samples = read_manifest(args.manifest)
    splits = splits_from_manifest(samples)
    if splits is None:
        splits = split_by_source(samples, seed=args.seed)
    if not splits["test"]:
        raise SystemExit("Test split is empty. Add more distinct source_id values.")
    dataset = FaceCropDataset(splits["test"], metadata, augment=False, cache=args.cache_images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    y_true = []
    y_pred = []
    y_score = []
    latencies = []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            start = time.perf_counter()
            logits = model(images)
            latencies.append((time.perf_counter() - start) / images.size(0))
            probs = torch.softmax(logits, dim=1)[:, metadata.manipulated_index]
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_score.extend(probs.cpu().tolist())

    import numpy as np

    metrics = compute_classification_metrics(np.array(y_true), np.array(y_pred), np.array(y_score))
    report = {
        "model_name": metadata.model_name,
        "model_version": metadata.model_version,
        "n_test_images": len(dataset),
        "n_test_sources": splits["source_counts"]["test"],
        "metrics": metrics,
        "mean_inference_latency_ms": round(float(np.mean(latencies) * 1000), 2),
        "model_size_mb": estimate_model_mb(model),
        "hardware": str(device),
        "target_accuracy_note": (
            "97% is a project target, not a claimed result. Use metrics.accuracy from this file."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
