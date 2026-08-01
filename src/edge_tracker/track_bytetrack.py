"""Track objects with YOLO + ByteTrack.

Run from the project root:
    python -m edge_tracker.track_bytetrack --input data/raw/traffic.mp4 \
        --output outputs/traffic_bytetrack.mp4 --classes person car bus truck
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from edge_tracker.detect_video import choose_device, color_for_class, resolve_class_ids
from edge_tracker.video_basics import VideoInfo, get_video_info


def draw_bytetrack_results(frame: np.ndarray, result) -> tuple[np.ndarray, set[int]]:
    """Draw the track IDs added by Ultralytics' ByteTrack implementation."""
    annotated = frame.copy()
    active_ids: set[int] = set()
    if result.boxes is None:
        return annotated, active_ids

    for box in result.boxes:
        # `box.id` is None when a detection was not assigned a stable track.
        if box.id is None:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        track_id = int(box.id.item())
        color = color_for_class(class_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{result.names[class_id]} #{track_id} {confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        active_ids.add(track_id)
    return annotated, active_ids


def track_video(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    confidence: float,
    class_names: list[str] | None,
    device: str,
    max_frames: int | None,
) -> VideoInfo:
    """Run YOLO detections and let ByteTrack associate them across frames."""
    model = YOLO(str(model_path))
    class_ids = resolve_class_ids(class_names, model.names)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"OpenCV could not open video: {input_path}")
    info = get_video_info(capture)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not create output video: {output_path}")

    frame_index = 0
    all_seen_ids: set[int] = set()
    try:
        while True:
            success, frame = capture.read()
            if not success or (max_frames is not None and frame_index >= max_frames):
                break
            start = time.perf_counter()
            # persist=True preserves the Kalman-filter/association state between frames.
            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=confidence,
                classes=class_ids,
                device=device,
                verbose=False,
            )[0]
            elapsed = time.perf_counter() - start
            annotated, active_ids = draw_bytetrack_results(frame, result)
            all_seen_ids.update(active_ids)
            fps = 1 / elapsed if elapsed else 0.0
            cv2.putText(
                annotated,
                f"ByteTrack | active={len(active_ids)} | IDs seen={len(all_seen_ids)} | {fps:.1f} FPS",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(annotated)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    print(f"Processed {frame_index} frame(s); ByteTrack issued {len(all_seen_ids)} ID(s) -> {output_path}")
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/yolo11n.pt"))
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, or mps")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    track_video(
        args.input,
        args.output,
        args.model,
        args.confidence,
        args.classes,
        choose_device(args.device),
        args.max_frames,
    )


if __name__ == "__main__":
    main()
