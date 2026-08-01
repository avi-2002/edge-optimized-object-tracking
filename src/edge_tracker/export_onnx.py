"""Export a trained PyTorch YOLO model to a fixed-shape ONNX file.

Example:
    python -m edge_tracker.export_onnx \
        --model runs/detect/runs/train/traffic_yolo11n/weights/best.pt \
        --output models/exports/traffic_yolo11n.onnx
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def from_project_root(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Trained .pt model")
    parser.add_argument("--output", type=Path, required=True, help="Destination .onnx file")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model_path = from_project_root(args.model)
    output_path = from_project_root(args.output)
    if not model_path.is_file():
        raise FileNotFoundError(f"PyTorch model not found: {model_path}")

    # Fixed 640x640 input makes the PyTorch/ONNX comparison fair and predictable.
    exported_path = Path(
        YOLO(str(model_path)).export(
            format="onnx",
            imgsz=args.imgsz,
            dynamic=False,
            simplify=False,
            device="cpu",
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported_path, output_path)
    print(f"Exported ONNX model: {output_path} ({output_path.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
