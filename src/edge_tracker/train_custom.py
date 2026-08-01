"""Fine-tune the small YOLO model on the custom traffic dataset.

Run only after `validate_dataset` passes:
    python -m edge_tracker.train_custom --epochs 50 --batch 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from edge_tracker.detect_video import choose_device

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def from_project_root(path: Path) -> Path:
    """Resolve normal relative CLI paths independently of the shell's CWD."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/traffic.yaml"))
    parser.add_argument("--model", type=Path, default=Path("models/yolo11n.pt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, or mps")
    parser.add_argument("--name", default="traffic_yolo11n")
    args = parser.parse_args()

    data_path = from_project_root(args.data)
    model_path = from_project_root(args.model)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {data_path}")
    model = YOLO(str(model_path))
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=choose_device(args.device),
        # An absolute path prevents Ultralytics from nesting output under runs/detect.
        project=str(PROJECT_ROOT / "runs" / "train"),
        name=args.name,
        pretrained=True,
        plots=True,
    )
    print(f"Training complete. Results are in: {results.save_dir}")


if __name__ == "__main__":
    main()
