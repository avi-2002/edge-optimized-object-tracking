"""Create a tiny synthetic video so Phase 1 can be tested without downloads.

Run from the project root:
    python -m edge_tracker.make_sample_video --output data/raw/phase1_sample.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def make_sample_video(output_path: Path, fps: int = 24, seconds: int = 4) -> None:
    """Save moving shapes in a video; the shapes stand in for future objects."""
    width, height = 640, 360
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not create sample video: {output_path}")

    try:
        for frame_index in range(fps * seconds):
            frame = np.full((height, width, 3), (35, 25, 20), dtype=np.uint8)
            progress = frame_index / (fps * seconds - 1)
            # The x-coordinate changes over time; this is the motion tracking will later model.
            x = int(40 + progress * (width - 160))
            cv2.rectangle(frame, (x, 130), (x + 90, 230), (255, 160, 30), -1)
            cv2.circle(frame, (width - x, 105), 25, (30, 220, 255), -1)
            cv2.putText(frame, "Synthetic Phase 1 Video", (20, 325), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (240, 240, 240), 2, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()

    print(f"Created {output_path} ({fps} FPS, {seconds} seconds)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_sample_video(args.output)


if __name__ == "__main__":
    main()
