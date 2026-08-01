"""Evaluate a trained custom model on the validation split.

Example:
    python -m edge_tracker.evaluate_custom --model runs/train/traffic_yolo11n/weights/best.pt
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
    parser.add_argument("--model", type=Path, required=True, help="Trained best.pt file")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto", help="auto, cpu, or mps")
    args = parser.parse_args()

    data_path = from_project_root(args.data)
    model_path = from_project_root(args.model)
    if not model_path.is_file():
        saved_models = sorted(PROJECT_ROOT.glob("runs/**/weights/best.pt"))
        available = "\n".join(f"- {path.relative_to(PROJECT_ROOT)}" for path in saved_models)
        hint = f"\nAvailable best.pt files:\n{available}" if available else ""
        raise FileNotFoundError(f"Trained model not found: {model_path}.{hint}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {data_path}")

    metrics = YOLO(str(model_path)).val(
        data=str(data_path), imgsz=args.imgsz, device=choose_device(args.device)
    )
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")


if __name__ == "__main__":
    main()
