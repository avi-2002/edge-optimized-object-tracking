"""Detect objects in every frame of a video with a pretrained YOLO model.

Run from the project root:
    python -m edge_tracker.detect_video --input data/raw/traffic.mp4 \
        --output outputs/traffic_detected.mp4 --classes person car bus truck
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from edge_tracker.video_basics import VideoInfo, get_video_info


def choose_device(requested_device: str) -> str:
    """Use the requested accelerator, or choose Apple Metal when it is available."""
    if requested_device != "auto":
        return requested_device
    return "mps" if torch.backends.mps.is_available() else "cpu"


def resolve_class_ids(class_names: list[str] | None, model_names: dict[int, str]) -> list[int] | None:
    """Translate user-friendly names such as 'person' into YOLO class IDs."""
    if not class_names:
        return None  # None tells YOLO to keep every class it knows.

    name_to_id = {name.lower(): class_id for class_id, name in model_names.items()}
    unknown = [name for name in class_names if name.lower() not in name_to_id]
    if unknown:
        available = ", ".join(sorted(name_to_id))
        raise ValueError(f"Unknown class(es): {', '.join(unknown)}. Available: {available}")
    return [name_to_id[name.lower()] for name in class_names]


def color_for_class(class_id: int) -> tuple[int, int, int]:
    """Make a repeatable OpenCV BGR color for each object class."""
    return ((37 * class_id) % 180 + 50, (17 * class_id) % 180 + 50, (29 * class_id) % 180 + 50)


def draw_detections(frame: np.ndarray, result) -> tuple[np.ndarray, int]:
    """Draw YOLO's raw box predictions ourselves, rather than hiding them in a helper."""
    annotated = frame.copy()
    detection_count = 0
    if result.boxes is None:
        return annotated, detection_count

    for box in result.boxes:
        # xyxy means the corners: (left, top, right, bottom), in pixel coordinates.
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        class_name = result.names[class_id]
        color = color_for_class(class_id)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        detection_count += 1
    return annotated, detection_count


def detect_video(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    confidence: float,
    class_names: list[str] | None,
    device: str,
    max_frames: int | None,
) -> VideoInfo:
    """Run detection independently on each frame and save an annotated MP4."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    model = YOLO(str(model_path))  # Downloads the small pretrained model the first time.
    class_ids = resolve_class_ids(class_names, model.names)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"OpenCV could not open video: {input_path}")
    info = get_video_info(capture)
    if not all((info.width > 0, info.height > 0, info.fps > 0)):
        capture.release()
        raise ValueError(f"Invalid video metadata: {info}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not create output video: {output_path}")

    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success or (max_frames is not None and frame_index >= max_frames):
                break

            start = time.perf_counter()
            result = model.predict(
                frame, conf=confidence, classes=class_ids, device=device, verbose=False
            )[0]
            inference_seconds = time.perf_counter() - start
            annotated, detection_count = draw_detections(frame, result)
            fps = 1 / inference_seconds if inference_seconds else 0.0
            cv2.putText(
                annotated,
                f"detections={detection_count} | inference={fps:.1f} FPS | device={device}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(annotated)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    print(f"Processed {frame_index} frame(s) on {device} -> {output_path}")
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/yolo11n.pt"))
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument("--classes", nargs="+", default=None, help="Optional classes, e.g. person car")
    parser.add_argument("--device", default="auto", help="auto, cpu, or mps")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    info = detect_video(
        args.input,
        args.output,
        args.model,
        args.confidence,
        args.classes,
        device,
        args.max_frames,
    )
    print(f"Input: {info.width}x{info.height}, {info.fps:.2f} FPS, {info.frame_count} frames")


if __name__ == "__main__":
    main()
