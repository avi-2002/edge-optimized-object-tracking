"""Track objects with ByteTrack and add trajectories plus line-crossing analytics.

Run from the project root:
    python -m edge_tracker.analyze_video --input data/raw/traffic.mp4 \
        --output outputs/traffic_analytics.mp4 --classes person car bus truck
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from edge_tracker.analytics import TrackObservation, VideoAnalytics
from edge_tracker.detect_video import choose_device, resolve_class_ids
from edge_tracker.track_bytetrack import draw_bytetrack_results
from edge_tracker.video_basics import VideoInfo, get_video_info


def observations_from_result(result) -> list[TrackObservation]:
    """Keep only detections to which ByteTrack assigned a stable ID."""
    if result.boxes is None:
        return []
    observations = []
    for box in result.boxes:
        if box.id is not None:
            observations.append(
                TrackObservation(
                    track_id=int(box.id.item()),
                    class_id=int(box.cls.item()),
                    bbox=box.xyxy[0].cpu().numpy(),
                )
            )
    return observations


def analyze_video(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    confidence: float,
    class_names: list[str] | None,
    device: str,
    line_fraction: float,
    max_frames: int | None,
) -> VideoInfo:
    """Process a video, saving boxes, track IDs, trajectories, and count metrics."""
    if not 0.0 < line_fraction < 1.0:
        raise ValueError("line_fraction must be between 0 and 1 (exclusive)")
    model = YOLO(str(model_path))
    class_ids = resolve_class_ids(class_names, model.names)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"OpenCV could not open video: {input_path}")
    info = get_video_info(capture)
    analytics = VideoAnalytics(line_y=int(info.height * line_fraction))
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
            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=confidence,
                classes=class_ids,
                device=device,
                verbose=False,
            )[0]
            analytics.record_inference_time(time.perf_counter() - start)
            analytics.update(observations_from_result(result))
            boxes_and_ids, _ = draw_bytetrack_results(frame, result)
            writer.write(analytics.draw_overlay(boxes_and_ids))
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    print(
        f"Processed {frame_index} frame(s); unique IDs={analytics.unique_tracks}, "
        f"up={analytics.up_crossings}, down={analytics.down_crossings} -> {output_path}"
    )
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/yolo11n.pt"))
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, or mps")
    parser.add_argument(
        "--line-fraction",
        type=float,
        default=0.50,
        help="Horizontal line location as a fraction of image height, e.g. 0.50",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_video(
        args.input,
        args.output,
        args.model,
        args.confidence,
        args.classes,
        choose_device(args.device),
        args.line_fraction,
        args.max_frames,
    )


if __name__ == "__main__":
    main()
