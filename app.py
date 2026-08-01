"""Streamlit interface for Edge-Optimized Object Detection & Tracking."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edge_tracker.analyze_video import analyze_video  # noqa: E402
from edge_tracker.detect_video import choose_device  # noqa: E402


CUSTOM_MODEL = PROJECT_ROOT / "assets" / "traffic_yolo11n.pt"
FALLBACK_MODEL = PROJECT_ROOT / "models" / "yolo11n.pt"
TRAFFIC_CLASSES = ["person", "car", "bus", "truck"]
PERFORMANCE_PROFILES = {
    "Fast (recommended for 4K / long video)": {"imgsz": 416, "output_scale": 0.5, "frame_stride": 2},
    "Balanced": {"imgsz": 512, "output_scale": 0.75, "frame_stride": 1},
    "Quality (slowest)": {"imgsz": 640, "output_scale": 1.0, "frame_stride": 1},
}


def model_choices() -> dict[str, Path]:
    """Offer the custom model when packaged, otherwise keep a pretrained fallback."""
    choices = {"Pretrained YOLO (COCO classes)": FALLBACK_MODEL}
    if CUSTOM_MODEL.is_file():
        choices = {"Custom traffic model (recommended)": CUSTOM_MODEL, **choices}
    return choices


def process_upload(
    uploaded_file,
    model_path: Path,
    classes: list[str],
    confidence: float,
    line_fraction: float,
    performance: dict[str, float | int],
):
    """Write the uploaded video temporarily and return its processed MP4 bytes."""
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    with tempfile.TemporaryDirectory(prefix="edge_tracker_") as temporary_directory:
        temporary_directory = Path(temporary_directory)
        input_path = temporary_directory / f"input{suffix}"
        output_path = temporary_directory / "annotated.mp4"
        input_path.write_bytes(uploaded_file.getvalue())
        analyze_video(
            input_path=input_path,
            output_path=output_path,
            model_path=model_path,
            confidence=confidence,
            class_names=classes or None,
            device=choose_device("auto"),
            line_fraction=line_fraction,
            max_frames=None,
            imgsz=int(performance["imgsz"]),
            output_scale=float(performance["output_scale"]),
            frame_stride=int(performance["frame_stride"]),
        )
        return output_path.read_bytes()


st.set_page_config(page_title="Edge Object Tracker", page_icon="🎯", layout="wide")
st.title("🎯 Edge-Optimized Object Detection & Tracking")
st.caption("Upload a traffic video to detect objects, track stable IDs, draw trajectories, and count line crossings.")

with st.sidebar:
    st.header("Processing settings")
    selected_name = st.selectbox("Model", list(model_choices()))
    selected_model = model_choices()[selected_name]
    confidence = st.slider("Detection confidence", 0.10, 0.90, 0.40, 0.05)
    line_fraction = st.slider("Counting-line height", 0.10, 0.90, 0.50, 0.05)
    classes = st.multiselect("Classes to detect", TRAFFIC_CLASSES, default=TRAFFIC_CLASSES)
    selected_profile = st.selectbox("Performance profile", list(PERFORMANCE_PROFILES))
    performance = PERFORMANCE_PROFILES[selected_profile]
    st.caption("The counting line is measured from the top of the video frame.")
    st.caption("Fast mode uses a 416px model input, 50% output resolution, and every second frame.")

uploaded_file = st.file_uploader(
    "Upload an MP4, MOV, AVI, or MKV video", type=["mp4", "mov", "avi", "mkv"], max_upload_size=200
)

if uploaded_file is not None:
    st.video(uploaded_file.getvalue())
    if st.button("Process video", type="primary"):
        with st.spinner("Detecting, tracking, and computing video analytics. This can take a few minutes on CPU."):
            try:
                output_bytes = process_upload(
                    uploaded_file,
                    selected_model,
                    classes,
                    confidence,
                    line_fraction,
                    performance,
                )
                st.session_state["processed_video"] = output_bytes
                st.session_state["processed_name"] = f"tracked_{Path(uploaded_file.name).stem}.mp4"
            except Exception as error:
                st.exception(error)

if "processed_video" in st.session_state:
    st.success("Processing complete.")
    st.video(st.session_state["processed_video"], format="video/mp4")
    st.download_button(
        "Download annotated video",
        data=st.session_state["processed_video"],
        file_name=st.session_state["processed_name"],
        mime="video/mp4",
    )

st.divider()
st.markdown(
    "**How it works:** YOLO detects `person`, `car`, `bus`, and `truck` in each frame; "
    "ByteTrack links detections across frames; trajectories and a virtual line produce movement analytics."
)
