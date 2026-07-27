import numpy as np

from edge_tracker.video_basics import draw_frame_annotation


def test_annotation_does_not_change_input_frame() -> None:
    """Drawing should produce a new image instead of mutating the original frame."""
    frame = np.zeros((80, 320, 3), dtype=np.uint8)

    annotated = draw_frame_annotation(frame, frame_index=25, fps=25.0)

    assert np.array_equal(frame, np.zeros((80, 320, 3), dtype=np.uint8))
    assert not np.array_equal(annotated, frame)
