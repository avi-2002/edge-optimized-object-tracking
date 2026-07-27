"""Inspect and annotate a video, one frame at a time.

Run from the project root:
    python -m edge_tracker.video_basics --input data/raw/my_video.mp4 \
        --output outputs/annotated_video.mp4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """Properties reported by the video file's decoder."""

    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        """Approximate duration; some codecs may not report frame count exactly."""
        return self.frame_count / self.fps if self.fps else 0.0


def get_video_info(capture: cv2.VideoCapture) -> VideoInfo:
    """Read metadata from an already-open video capture object."""
    return VideoInfo(
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(capture.get(cv2.CAP_PROP_FPS)),
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )


def draw_frame_annotation(frame: np.ndarray, frame_index: int, fps: float) -> np.ndarray:
    """Return a copy of *frame* with simple Phase-1 information overlaid.

    OpenCV uses BGR color order, so (0, 255, 0) is green, not RGB red.
    """
    annotated = frame.copy()
    timestamp = frame_index / fps if fps else 0.0
    label = f"frame={frame_index}  time={timestamp:.2f}s"
    cv2.putText(
        annotated,
        label,
        (20, 35),  # x, y position measured from the top-left corner
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,  # font scale
        (0, 255, 0),  # BGR green
        2,  # line thickness in pixels
        cv2.LINE_AA,
    )
    return annotated


def process_video(input_path: Path, output_path: Path, max_frames: int | None = None) -> VideoInfo:
    """Read an input video frame-by-frame and save an annotated copy."""
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"OpenCV could not open video: {input_path}")

    info = get_video_info(capture)
    if info.width <= 0 or info.height <= 0 or info.fps <= 0:
        capture.release()
        raise ValueError(f"Invalid video metadata: {info}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # mp4v is a broadly supported MP4 codec for local learning exercises.
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), codec, info.fps, (info.width, info.height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not create output video: {output_path}")

    processed_frames = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break  # End of video, or an unreadable frame.

            writer.write(draw_frame_annotation(frame, processed_frames, info.fps))
            processed_frames += 1
            if max_frames is not None and processed_frames >= max_frames:
                break
    finally:
        # Always release file handles, including if processing raises an error.
        capture.release()
        writer.release()

    print(f"Processed {processed_frames} frame(s) -> {output_path}")
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to an input video file")
    parser.add_argument("--output", type=Path, required=True, help="Path for the annotated MP4")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit for a quick test; processes every frame by default",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = process_video(args.input, args.output, args.max_frames)
    print(
        f"Input metadata: {info.width}x{info.height}, {info.fps:.2f} FPS, "
        f"{info.frame_count} frames, about {info.duration_seconds:.2f} seconds"
    )


if __name__ == "__main__":
    main()
