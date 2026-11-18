"""Dataset loading with source-video grouping to reduce leakage."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from app.ml.metadata import ModelMetadata

CLASS_TO_INDEX = {"authentic": 0, "manipulated": 1}


SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}


@dataclass
class Sample:
    path: Path
    label: int
    source_id: str
    split: str | None = None


def read_manifest(csv_path: Path) -> list[Sample]:
    """CSV columns: path,label,source_id

    label must be authentic or manipulated.
    source_id should be the originating video or identity, never a random frame id.
    """
    samples: list[Sample] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "label", "source_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest must contain columns {sorted(required)}")
        has_split = "split" in (reader.fieldnames or [])
        for row in reader:
            label_name = row["label"].strip().lower()
            if label_name not in CLASS_TO_INDEX:
                raise ValueError(f"Unknown label '{label_name}' in {csv_path}")
            split_name = None
            if has_split and row.get("split", "").strip():
                raw_split = row["split"].strip().lower()
                if raw_split not in SPLIT_ALIASES:
                    raise ValueError(f"Unknown split '{raw_split}' in {csv_path}")
                split_name = SPLIT_ALIASES[raw_split]
            samples.append(
                Sample(
                    path=Path(row["path"]),
                    label=CLASS_TO_INDEX[label_name],
                    source_id=row["source_id"].strip(),
                    split=split_name,
                )
            )
    return samples


def splits_from_manifest(samples: list[Sample]) -> dict[str, list[Sample]] | None:
    """Use an official split column when every row has one.

    Returns None when the manifest has no split labels, so callers can fall
    back to grouping by source_id.
    """
    labeled = [sample for sample in samples if sample.split]
    if not labeled:
        return None
    if len(labeled) != len(samples):
        raise ValueError("Manifest split column must be set on every row, or on none.")
    buckets: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    for sample in samples:
        assert sample.split is not None
        key = SPLIT_ALIASES.get(sample.split.strip().lower())
        if key is None:
            raise ValueError(f"Unknown split '{sample.split}'")
        buckets[key].append(sample)
    source_counts = {
        name: len({sample.source_id for sample in rows}) for name, rows in buckets.items()
    }
    return {
        "train": buckets["train"],
        "val": buckets["val"],
        "test": buckets["test"],
        "source_counts": source_counts,
    }


def split_by_source(
    samples: list[Sample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[Sample]]:
    """Assign every source_id to exactly one split so frames from one video cannot leak."""
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.source_id].append(sample)
    source_ids = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(source_ids)
    n = len(source_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_ids = set(source_ids[:n_train])
    val_ids = set(source_ids[n_train : n_train + n_val])
    test_ids = set(source_ids[n_train + n_val :])
    return {
        "train": [s for sid in train_ids for s in grouped[sid]],
        "val": [s for sid in val_ids for s in grouped[sid]],
        "test": [s for sid in test_ids for s in grouped[sid]],
        "source_counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
    }


class FaceCropDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        metadata: ModelMetadata,
        augment: bool,
        cache: bool = False,
    ) -> None:
        self.samples = samples
        self.cached: list[Image.Image] | None = None
        if cache:
            # Resize once so later epochs are not blocked on disk.
            size = (metadata.input_size, metadata.input_size)
            cached: list[Image.Image] = []
            total = len(samples)
            for index, sample in enumerate(samples, start=1):
                with Image.open(sample.path) as image:
                    cached.append(image.convert("RGB").resize(size, Image.BILINEAR))
                if index == total or index % 2000 == 0:
                    print(f"cached {index}/{total}", flush=True)
            self.cached = cached
        if augment and self.cached is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((metadata.input_size + 32, metadata.input_size + 32)),
                    transforms.RandomCrop(metadata.input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=metadata.image_mean, std=metadata.image_std),
                ]
            )
        elif augment:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=metadata.image_mean, std=metadata.image_std),
                ]
            )
        elif self.cached is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((metadata.input_size, metadata.input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=metadata.image_mean, std=metadata.image_std),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=metadata.image_mean, std=metadata.image_std),
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        if self.cached is not None:
            tensor = self.transform(self.cached[index])
            return tensor, sample.label
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, sample.label
