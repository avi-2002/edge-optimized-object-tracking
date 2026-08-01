"""Convert a small BDD100K detection subset into the project's YOLO format.

Example (copies 500 train and 150 validation images):
    python -m edge_tracker.prepare_bdd100k \
        --source /Users/avinashraval/Downloads/bdd100k
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2


CLASS_TO_ID = {"person": 0, "car": 1, "bus": 2, "truck": 3}


@dataclass(frozen=True)
class BddExample:
    image_path: Path
    objects: list[dict]


def relevant_objects(annotation_path: Path) -> list[dict]:
    """Read BDD100K's per-image JSON and retain usable target-category boxes."""
    annotation = json.loads(annotation_path.read_text())
    frames = annotation.get("frames", [])
    if not frames:
        return []
    objects = frames[0].get("objects", [])
    return [
        obj
        for obj in objects
        if obj.get("category") in CLASS_TO_ID and "box2d" in obj
    ]


def available_examples(source_root: Path, split: str) -> list[BddExample]:
    """Find BDD images that have at least one selected class with a box."""
    labels_dir = source_root / "labels" / split
    images_dir = source_root / "images" / "100k" / split
    if not labels_dir.is_dir() or not images_dir.is_dir():
        raise FileNotFoundError(f"BDD100K {split!r} folders were not found under {source_root}")

    examples = []
    for annotation_path in sorted(labels_dir.glob("*.json")):
        image_path = images_dir / f"{annotation_path.stem}.jpg"
        objects = relevant_objects(annotation_path)
        if image_path.exists() and objects:
            examples.append(BddExample(image_path=image_path, objects=objects))
    return examples


def yolo_lines(objects: list[dict], image_width: int, image_height: int) -> list[str]:
    """Convert pixel corner boxes into normalized YOLO center/width/height lines."""
    lines: list[str] = []
    for obj in objects:
        box = obj["box2d"]
        x1 = max(0.0, min(float(box["x1"]), image_width))
        y1 = max(0.0, min(float(box["y1"]), image_height))
        x2 = max(0.0, min(float(box["x2"]), image_width))
        y2 = max(0.0, min(float(box["y2"]), image_height))
        if x2 <= x1 or y2 <= y1:
            continue
        x_center = ((x1 + x2) / 2) / image_width
        y_center = ((y1 + y2) / 2) / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        lines.append(f"{CLASS_TO_ID[obj['category']]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return lines


def export_split(examples: list[BddExample], destination_root: Path, split: str) -> int:
    """Copy images and save corresponding YOLO text labels."""
    images_destination = destination_root / "images" / split
    labels_destination = destination_root / "labels" / split
    images_destination.mkdir(parents=True, exist_ok=True)
    labels_destination.mkdir(parents=True, exist_ok=True)

    exported = 0
    for example in examples:
        image = cv2.imread(str(example.image_path))
        if image is None:
            print(f"Skipping unreadable image: {example.image_path}")
            continue
        height, width = image.shape[:2]
        lines = yolo_lines(example.objects, width, height)
        if not lines:
            continue
        shutil.copy2(example.image_path, images_destination / example.image_path.name)
        (labels_destination / f"{example.image_path.stem}.txt").write_text("\n".join(lines) + "\n")
        exported += 1
    return exported


def choose_examples(examples: list[BddExample], count: int, rng: random.Random, split: str) -> list[BddExample]:
    if len(examples) < count:
        raise ValueError(f"Requested {count} {split} examples, but only found {len(examples)}")
    return rng.sample(examples, count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Extracted BDD100K folder")
    parser.add_argument("--destination", type=Path, default=Path("data/traffic"))
    parser.add_argument("--train-count", type=int, default=500)
    parser.add_argument("--val-count", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Inspect source data without copying files")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train = choose_examples(available_examples(args.source, "train"), args.train_count, rng, "train")
    val = choose_examples(available_examples(args.source, "val"), args.val_count, rng, "val")
    print(f"Selected {len(train)} train and {len(val)} validation images with target objects.")
    if args.dry_run:
        print("Dry run complete; no files were copied.")
        return

    train_exported = export_split(train, args.destination, "train")
    val_exported = export_split(val, args.destination, "val")
    print(f"Exported {train_exported} train and {val_exported} validation image/label pairs to {args.destination}")


if __name__ == "__main__":
    main()
