"""Build a DeepShield manifest from the 140k Real vs Fake Faces folders.

Expected layout (Kaggle 140k-real-and-fake-faces):

    <root>/train/real
    <root>/train/fake
    <root>/valid/real
    <root>/valid/fake
    <root>/test/real
    <root>/test/fake

Labels written for this project are authentic (real) and manipulated (fake).
Official train/valid/test folders are kept. Pass per-class caps to train faster
on CPU; 0 means use every image in that split.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLIT_DIRS = (("train", "train"), ("valid", "val"), ("test", "test"))
CLASS_DIRS = (("real", "authentic"), ("fake", "manipulated"))


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise SystemExit(f"Missing dataset folder: {folder}")
    files = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    files.sort(key=lambda path: path.name)
    return files


def take(files: list[Path], limit: int, rng: random.Random) -> list[Path]:
    if limit <= 0 or limit >= len(files):
        return files
    return sorted(rng.sample(files, limit), key=lambda path: path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a real/fake face manifest")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=0)
    parser.add_argument("--val-per-class", type=int, default=0)
    parser.add_argument("--test-per-class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limits = {"train": args.train_per_class, "val": args.val_per_class, "test": args.test_per_class}
    rng = random.Random(args.seed)
    rows: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for folder_name, split_name in SPLIT_DIRS:
        for class_dir, label in CLASS_DIRS:
            files = list_images(args.root / folder_name / class_dir)
            chosen = take(files, limits[split_name], rng)
            counts[f"{split_name}/{label}"] = len(chosen)
            for path in chosen:
                rows.append(
                    {
                        "path": str(path.resolve()),
                        "label": label,
                        "source_id": f"{split_name}_{label}_{path.stem}",
                        "split": split_name,
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "source_id", "split"])
        writer.writeheader()
        writer.writerows(rows)
    print({"output": str(args.output), "rows": len(rows), **counts})


if __name__ == "__main__":
    main()
