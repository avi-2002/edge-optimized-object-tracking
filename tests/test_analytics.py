import numpy as np

from edge_tracker.analytics import TrackObservation, VideoAnalytics


def observation(track_id: int, center_y: int) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        class_id=0,
        bbox=np.array([20, center_y - 10, 40, center_y + 10]),
    )


def test_downward_crossing_is_counted_once() -> None:
    analytics = VideoAnalytics(line_y=100)
    analytics.update([observation(1, 80)])
    analytics.update([observation(1, 120)])
    analytics.update([observation(1, 140)])

    assert analytics.down_crossings == 1
    assert analytics.up_crossings == 0
    assert analytics.unique_tracks == 1


def test_upward_crossing_is_counted() -> None:
    analytics = VideoAnalytics(line_y=100)
    analytics.update([observation(7, 120)])
    analytics.update([observation(7, 80)])

    assert analytics.down_crossings == 0
    assert analytics.up_crossings == 1
