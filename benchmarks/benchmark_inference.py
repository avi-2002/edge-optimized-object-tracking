"""Benchmark equivalent PyTorch and ONNX YOLO inference on one representative image."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def from_project_root(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def benchmark(model_path: Path, image, imgsz: int, warmup_runs: int, measured_runs: int) -> dict[str, float]:
    """Measure full predict-call latency, including preprocessing and postprocessing."""
    model = YOLO(str(model_path))
    for _ in range(warmup_runs):
        model.predict(image, imgsz=imgsz, device="cpu", verbose=False)

    latencies_ms = []
    for _ in range(measured_runs):
        start = time.perf_counter()
        model.predict(image, imgsz=imgsz, device="cpu", verbose=False)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    return {
        "model_size_mib": round(model_path.stat().st_size / 1024**2, 3),
        "mean_latency_ms": round(statistics.mean(latencies_ms), 3),
        "median_latency_ms": round(statistics.median(latencies_ms), 3),
        "p95_latency_ms": round(sorted(latencies_ms)[int(0.95 * (len(latencies_ms) - 1))], 3),
        "mean_fps": round(1000 / statistics.mean(latencies_ms), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True, help="Representative validation image")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--measured-runs", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/phase6_results.json"))
    args = parser.parse_args()

    pytorch_path = from_project_root(args.pytorch)
    onnx_path = from_project_root(args.onnx)
    image_path = from_project_root(args.image)
    output_path = from_project_root(args.output)
    for path in (pytorch_path, onnx_path, image_path):
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    results = {
        "setup": {
            "image": str(image_path.relative_to(PROJECT_ROOT)),
            "imgsz": args.imgsz,
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.measured_runs,
            "device": "cpu",
        },
        "pytorch": benchmark(pytorch_path, image, args.imgsz, args.warmup_runs, args.measured_runs),
        "onnx": benchmark(onnx_path, image, args.imgsz, args.warmup_runs, args.measured_runs),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Saved benchmark: {output_path}")


if __name__ == "__main__":
    main()
