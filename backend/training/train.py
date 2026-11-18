"""Train EfficientNet-B0 for binary deepfake classification.

Example:
  python training/train.py --manifest data/manifest.csv --epochs 10 --output-dir models
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ml.architecture import build_classifier, estimate_model_mb, resolve_device
from app.ml.dataset import FaceCropDataset, read_manifest, split_by_source, splits_from_manifest
from app.ml.metadata import ModelMetadata, save_metadata, utc_now
from app.ml.metrics import compute_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepShield EfficientNet-B0 classifier")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with path,label,source_id")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--architecture", default="efficientnet_b0", choices=("efficientnet_b0", "artifact_cnn"))
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--cache-images",
        action="store_true",
        help="Resize the split into memory before training so epochs are not disk-bound",
    )
    parser.add_argument("--dataset-name", default="user_provided")
    parser.add_argument("--dataset-source", default="see docs/DATASET.md")
    parser.add_argument("--dataset-license", default="unknown_until_recorded")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--log-every", type=int, default=50, help="Print training progress every N batches")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool, log_every: int = 0):
    model.train(train)
    total_loss = 0.0
    n = 0
    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        n += labels.size(0)
        if train and log_every and step % log_every == 0:
            print(json.dumps({"batches": step, "images": n, "running_loss": round(total_loss / max(n, 1), 4)}), flush=True)
    return total_loss / max(n, 1)


def collect_predictions(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_score = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_score.extend(probs.cpu().tolist())
    import numpy as np

    return np.array(y_true), np.array(y_pred), np.array(y_score)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = resolve_device(args.device)
    samples = read_manifest(args.manifest)
    splits = splits_from_manifest(samples)
    if splits is None:
        splits = split_by_source(samples, seed=args.seed)
    print(json.dumps({"source_counts": splits["source_counts"], "n_train": len(splits["train"]), "n_val": len(splits["val"])}, indent=2))
    if not splits["train"] or not splits["val"]:
        raise SystemExit("Need samples in both train and val after source-id splitting.")

    metadata = ModelMetadata(
        model_name=f"deepshield_{args.architecture}",
        architecture=args.architecture,
        dataset_name=args.dataset_name,
        dataset_source=args.dataset_source,
        dataset_license=args.dataset_license,
        input_size=args.input_size,
        split_strategy=(
            "official_split_column"
            if samples and samples[0].split
            else "group_by_source_id_then_70_15_15"
        ),
        random_seed=args.seed,
        trained_at=utc_now(),
        hardware=str(device),
        notes=[
            "ImageNet backbone weights are used only as initialization when --pretrained-backbone is set.",
            "Validation metrics select the checkpoint. Test metrics must be computed separately with evaluate.py.",
        ]
        + (["Backbone weights were frozen; only the classifier head was trained."] if args.freeze_backbone else []),
    )
    print(json.dumps({"cache_images": args.cache_images, "input_size": args.input_size}), flush=True)
    train_ds = FaceCropDataset(splits["train"], metadata, augment=True, cache=args.cache_images)
    val_ds = FaceCropDataset(splits["val"], metadata, augment=False, cache=args.cache_images)
    print(json.dumps({"cached_train": len(train_ds), "cached_val": len(val_ds)}), flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    counts = Counter(s.label for s in splits["train"])
    weight = torch.tensor(
        [1.0 / max(counts.get(0, 1), 1), 1.0 / max(counts.get(1, 1), 1)],
        dtype=torch.float32,
        device=device,
    )
    weight = weight / weight.sum() * 2

    model = build_classifier(args.architecture, pretrained_backbone=args.pretrained_backbone).to(device)
    if args.freeze_backbone:
        if not hasattr(model, "features"):
            raise SystemExit(f"{args.architecture} does not support --freeze-backbone.")
        for param in model.features.parameters():
            param.requires_grad = False
    criterion = nn.CrossEntropyLoss(weight=weight)
    trainable = (p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    patience_left = args.patience
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True, log_every=args.log_every
        )
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        y_true, y_pred, y_score = collect_predictions(model, val_loader, device)
        val_metrics = compute_classification_metrics(y_true, y_pred, y_score)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_metrics["accuracy"]}
        history.append(row)
        print(json.dumps(row))
        if val_loss < best_val:
            best_val = val_loss
            patience_left = args.patience
            metadata.metrics = {"validation": val_metrics, "history": history}
            metadata.model_version = f"val_selected_{utc_now()[:19].replace(':', '')}"
            bundle = {"state_dict": model.state_dict(), "metadata": metadata.to_json()}
            checkpoint_path = args.output_dir / f"deepshield_{args.architecture}.pt"
            torch.save(bundle, checkpoint_path)
            save_metadata(args.output_dir / "model_metadata.json", metadata)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping on validation loss.")
                break

    elapsed = time.time() - start
    summary = {
        "elapsed_sec": round(elapsed, 2),
        "model_size_mb": estimate_model_mb(model),
        "device": str(device),
        "best_val_loss": best_val,
        "checkpoint": str(args.output_dir / f"deepshield_{args.architecture}.pt"),
        "architecture": args.architecture,
    }
    print(json.dumps(summary, indent=2))
    print("Next step: python training/evaluate.py --manifest ... --split-file is not required; evaluate.py re-splits with the same seed.")


if __name__ == "__main__":
    main()
