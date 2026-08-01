"""Stateful, model-independent video analytics for tracked bounding boxes."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TrackObservation:
    """One object that has a tracker ID in the current video frame."""

    track_id: int
    class_id: int
    bbox: np.ndarray  # [left, top, right, bottom]

    @property
    def centroid(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox.astype(int)
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class VideoAnalytics:
    """Maintain trajectory and line-crossing state while a video is processed."""

    def __init__(self, line_y: int, history_size: int = 30) -> None:
        self.line_y = line_y
        self.histories: dict[int, deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self.previous_centroids: dict[int, tuple[int, int]] = {}
        self.track_classes: dict[int, int] = {}
        self.counted_crossings: set[tuple[int, str]] = set()
        self.up_crossings = 0
        self.down_crossings = 0
        self.recent_inference_seconds: deque[float] = deque(maxlen=30)

    def update(self, observations: list[TrackObservation]) -> None:
        """Record current tracks and count a crossing when a centroid changes sides."""
        for observation in observations:
            track_id = observation.track_id
            centroid = observation.centroid
            previous = self.previous_centroids.get(track_id)

            if previous is not None:
                if previous[1] < self.line_y <= centroid[1]:
                    self._count_crossing(track_id, "down")
                elif previous[1] > self.line_y >= centroid[1]:
                    self._count_crossing(track_id, "up")

            self.histories[track_id].append(centroid)
            self.previous_centroids[track_id] = centroid
            self.track_classes.setdefault(track_id, observation.class_id)

    def _count_crossing(self, track_id: int, direction: str) -> None:
        """Count each track at most once in each direction."""
        event = (track_id, direction)
        if event in self.counted_crossings:
            return
        self.counted_crossings.add(event)
        if direction == "down":
            self.down_crossings += 1
        else:
            self.up_crossings += 1

    def record_inference_time(self, seconds: float) -> None:
        self.recent_inference_seconds.append(seconds)

    @property
    def inference_fps(self) -> float:
        if not self.recent_inference_seconds:
            return 0.0
        return 1 / (sum(self.recent_inference_seconds) / len(self.recent_inference_seconds))

    @property
    def unique_tracks(self) -> int:
        return len(self.track_classes)

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw the virtual line, trajectories, and summary metrics on a frame."""
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        cv2.line(annotated, (0, self.line_y), (width - 1, self.line_y), (0, 255, 255), 2)
        cv2.putText(
            annotated,
            "counting line",
            (20, max(25, self.line_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        for history in self.histories.values():
            if len(history) >= 2:
                cv2.polylines(annotated, [np.array(history, dtype=np.int32)], False, (255, 255, 0), 2)

        summary = (
            f"unique IDs={self.unique_tracks} | up={self.up_crossings} | "
            f"down={self.down_crossings} | inference={self.inference_fps:.1f} FPS"
        )
        cv2.rectangle(annotated, (10, 10), (min(width - 10, 760), 45), (20, 20, 20), -1)
        cv2.putText(
            annotated,
            summary,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated
