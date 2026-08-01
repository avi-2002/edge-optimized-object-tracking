"""Evaluate a trained custom model on the validation split.

Example:
    python -m edge_tracker.evaluate_custom --model runs/train/traffic_yolo11n/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from edge_tracker.detect_video import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/traffic.yaml"))
    parser.add_argument("--model", type=Path, required=True, help="Trained best.pt file")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto", help="auto, cpu, or mps")
    args = parser.parse_args()

    metrics = YOLO(str(args.model)).val(data=str(args.data), imgsz=args.imgsz, device=choose_device(args.device))
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")


if __name__ == "__main__":
    main()
