"""Assign persistent IDs to detected objects with a small, readable IoU tracker.

Run from the project root:
    python -m edge_tracker.track_video --input data/raw/traffic.mp4 \
        --output outputs/traffic_tracked.mp4 --classes person car bus truck
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from edge_tracker.detect_video import choose_device, color_for_class, resolve_class_ids
from edge_tracker.video_basics import VideoInfo, get_video_info


@dataclass(frozen=True)
class Detection:
    """One YOLO prediction in a single frame."""

    bbox: np.ndarray  # [left, top, right, bottom] in pixels
    class_id: int
    confidence: float


@dataclass
class Track:
    """State that survives from one frame to the next."""

    track_id: int
    bbox: np.ndarray
    class_id: int
    confidence: float
    missed_frames: int = 0


def iou(first: np.ndarray, second: np.ndarray) -> float:
    """Intersection over Union: overlap area divided by total covered area."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


class IoUTracker:
    """A minimal multi-object tracker based on greedy IoU matching.

    It is intentionally simple for learning: same-class boxes with enough overlap
    keep the existing ID. ByteTrack, which we will add later, also uses motion
    prediction and a stronger matching strategy for crowded/occluded scenes.
    """

    def __init__(self, iou_threshold: float = 0.30, max_missed_frames: int = 15) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.tracks: list[Track] = []
        self.next_track_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        """Match current detections to previous tracks and return active tracks."""
        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self.tracks):
            for detection_index, detection in enumerate(detections):
                if track.class_id == detection.class_id:
                    candidates.append((iou(track.bbox, detection.bbox), track_index, detection_index))

        # Greedy matching: choose the strongest unmatched overlap first.
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for overlap, track_index, detection_index in sorted(candidates, reverse=True):
            if overlap < self.iou_threshold:
                break
            if track_index in matched_tracks or detection_index in matched_detections:
                continue
            track = self.tracks[track_index]
            detection = detections[detection_index]
            track.bbox = detection.bbox
            track.confidence = detection.confidence
            track.missed_frames = 0
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        for track_index, track in enumerate(self.tracks):
            if track_index not in matched_tracks:
                track.missed_frames += 1

        for detection_index, detection in enumerate(detections):
            if detection_index not in matched_detections:
                self.tracks.append(
                    Track(self.next_track_id, detection.bbox, detection.class_id, detection.confidence)
                )
                self.next_track_id += 1

        self.tracks = [track for track in self.tracks if track.missed_frames <= self.max_missed_frames]
        return [track for track in self.tracks if track.missed_frames == 0]


def detections_from_result(result) -> list[Detection]:
    """Convert Ultralytics result objects into the small Detection type above."""
    if result.boxes is None:
        return []
    return [
        Detection(
            bbox=box.xyxy[0].cpu().numpy(),
            class_id=int(box.cls.item()),
            confidence=float(box.conf.item()),
        )
        for box in result.boxes
    ]


def draw_tracks(frame: np.ndarray, tracks: list[Track], names: dict[int, str]) -> np.ndarray:
    """Draw current boxes using the ID kept by IoUTracker."""
    annotated = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = track.bbox.astype(int)
        color = color_for_class(track.class_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{names[track.class_id]} #{track.track_id} {track.confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def track_video(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    confidence: float,
    class_names: list[str] | None,
    device: str,
    max_frames: int | None,
    iou_threshold: float,
) -> VideoInfo:
    """Detect every frame, match its boxes to prior boxes, and write an MP4."""
    model = YOLO(str(model_path))
    class_ids = resolve_class_ids(class_names, model.names)
    tracker = IoUTracker(iou_threshold=iou_threshold)
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
            result = model.predict(frame, conf=confidence, classes=class_ids, device=device, verbose=False)[0]
            tracks = tracker.update(detections_from_result(result))
            elapsed = time.perf_counter() - start
            all_seen_ids.update(track.track_id for track in tracks)
            annotated = draw_tracks(frame, tracks, result.names)
            fps = 1 / elapsed if elapsed else 0.0
            cv2.putText(
                annotated,
                f"active={len(tracks)} | IDs seen={len(all_seen_ids)} | {fps:.1f} FPS",
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

    print(f"Processed {frame_index} frame(s); issued {len(all_seen_ids)} track ID(s) -> {output_path}")
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
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    track_video(
        args.input,
        args.output,
        args.model,
        args.confidence,
        args.classes,
        device,
        args.max_frames,
        args.iou_threshold,
    )


if __name__ == "__main__":
    main()
