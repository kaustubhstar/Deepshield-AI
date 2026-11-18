"""Create a tiny synthetic dataset so the API can be launched without a licensed deepfake corpus.

This is NOT a deepfake dataset. Images are color-tinted placeholders with a learnable
class cue (cooler authentic vs warmer manipulated). Do not cite fixture accuracy as
deepfake detection performance.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def draw_placeholder(path: Path, manipulated: bool, seed: int) -> None:
    image = Image.new("RGB", (160, 160), color=(18, 24, 38))
    draw = ImageDraw.Draw(image)
    if manipulated:
        face = (170 + (seed % 20), 70 + (seed % 15), 70)
        mouth = (140, 40, 40)
    else:
        face = (70, 90 + (seed % 20), 160 + (seed % 15))
        mouth = (40, 70, 140)
    draw.ellipse((30, 20, 130, 140), fill=face)
    draw.ellipse((55, 50, 75, 70), fill=(20, 20, 20))
    draw.ellipse((85, 50, 105, 70), fill=(20, 20, 20))
    draw.ellipse((60, 90, 100, 110), fill=mouth)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write synthetic train/val/test placeholders")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "fixture")
    parser.add_argument("--sources-per-class", type=int, default=12)
    parser.add_argument("--frames-per-source", type=int, default=2)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for label, manipulated in (("authentic", False), ("manipulated", True)):
        for source in range(args.sources_per_class):
            source_id = f"{label}_src{source:02d}"
            for frame in range(args.frames_per_source):
                rel = Path(label) / source_id / f"frame_{frame:02d}.png"
                path = args.output_dir / rel
                draw_placeholder(path, manipulated=manipulated, seed=source * 10 + frame)
                rows.append({"path": str(path.resolve()), "label": label, "source_id": source_id})

    manifest = args.output_dir / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "source_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} images and {manifest}")
    print("This fixture is not a licensed deepfake dataset.")


if __name__ == "__main__":
    main()
