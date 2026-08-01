"""Validate a YOLO-format dataset before spending time on training.

Run from the project root:
    python -m edge_tracker.validate_dataset --data data/traffic.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def validate_label(label_path: Path, class_count: int) -> list[str]:
    """Return human-readable errors in one YOLO label file."""
    errors: list[str] = []
    for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"{label_path}:{line_number}: expected 5 values, found {len(fields)}")
            continue
        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = map(float, fields[1:])
        except ValueError:
            errors.append(f"{label_path}:{line_number}: values must be numeric")
            continue
        if not 0 <= class_id < class_count:
            errors.append(f"{label_path}:{line_number}: class ID {class_id} is outside 0..{class_count - 1}")
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
            errors.append(f"{label_path}:{line_number}: normalized box values are invalid")
    return errors


def validate_split(dataset_root: Path, split: str, class_count: int) -> tuple[int, list[str]]:
    """Check that every image in one split has a valid matching label file."""
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    if not images_dir.exists():
        return 0, [f"Missing image directory: {images_dir}"]
    if not labels_dir.exists():
        return 0, [f"Missing label directory: {labels_dir}"]

    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    errors: list[str] = []
    if not images:
        errors.append(f"No images found in required split: {images_dir}")
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            errors.append(f"Missing label for image: {image_path}")
        else:
            errors.extend(validate_label(label_path, class_count))
    return len(images), errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="YOLO dataset YAML file")
    args = parser.parse_args()
    config = yaml.safe_load(args.data.read_text())
    dataset_root = Path(config["path"])
    class_count = len(config["names"])

    all_errors: list[str] = []
    for split in ("train", "val"):
        image_count, errors = validate_split(dataset_root, split, class_count)
        print(f"{split}: {image_count} image(s)")
        all_errors.extend(errors)

    if all_errors:
        print("\nDataset validation failed:")
        print("\n".join(f"- {error}" for error in all_errors))
        raise SystemExit(1)
    print("\nDataset validation passed.")


if __name__ == "__main__":
    main()
