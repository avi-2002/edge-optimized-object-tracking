# Edge-Optimized Object Detection & Tracking

A learning-first computer-vision project that detects objects in video, assigns stable IDs over time, measures simple analytics, and compares edge-friendly inference options.

## Project goals

1. Understand video processing, object detection, and multi-object tracking from first principles.
2. Build a reproducible local pipeline before creating a web application.
3. Benchmark the accuracy/speed/model-size trade-offs of an edge-oriented model.
4. Deploy a small Streamlit demo and document the work clearly.

## Roadmap

- [x] Phase 0 - repository and Python environment
- [x] Phase 1 - OpenCV video fundamentals
- [x] Phase 2 - pretrained object detection
- [x] Phase 3 - ByteTrack multi-object tracking
- [x] Phase 4 - counting, trajectories, and performance metrics
- [x] Phase 5 - custom training and evaluation
- [x] Phase 6 - ONNX export and edge optimization
- [ ] Phase 7 - Streamlit application and deployment

## Repository layout

```text
src/edge_tracker/  Application modules
tests/             Automated checks
data/raw/          Original videos (not committed)
data/processed/    Derived data (not committed)
models/            Downloaded/trained weights (not committed)
outputs/           Annotated videos and images (not committed)
benchmarks/        Reproducible performance results
notebooks/         Optional exploratory notebooks
```

## Environment

This project targets Python 3.11. Create and activate the virtual environment:

```bash
source .venv/bin/activate
python --version
```

Deactivate it when you are done:

```bash
deactivate
```

## Learning notes

Keep a `learning-notes.md` in this repository. At the end of every phase, answer:

1. What goes into the program and what comes out?
2. What key concept did I learn?
3. What surprised or confused me?
4. What evidence shows the phase works?
