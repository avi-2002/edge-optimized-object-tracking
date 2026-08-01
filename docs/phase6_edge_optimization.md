# Phase 6: Edge optimization

## Goal

Compare the trained PyTorch model (`.pt`) against an exported ONNX model (`.onnx`) on the same image, at the same 640×640 input resolution, on CPU.

## Why ONNX?

ONNX is a portable model graph format. ONNX Runtime can execute it without loading the PyTorch framework, which can make CPU deployment faster and more portable. It does not magically improve detection accuracy: exported FP32 ONNX should make the same predictions as the PyTorch model within small numerical differences.

## Steps

```bash
source .venv/bin/activate
python -m pip install 'onnx>=1.16,<2' 'onnxruntime>=1.18,<2'

export PYTHONPATH="$PWD/src"

python -m edge_tracker.export_onnx \
  --model runs/detect/runs/train/traffic_yolo11n/weights/best.pt \
  --output models/exports/traffic_yolo11n.onnx

python benchmarks/benchmark_inference.py \
  --pytorch runs/detect/runs/train/traffic_yolo11n/weights/best.pt \
  --onnx models/exports/traffic_yolo11n.onnx \
  --image data/traffic/images/val/REPLACE_WITH_A_REAL_FILENAME.jpg
```

## What to compare

- **Model size (MiB):** storage cost; lower is useful on constrained devices.
- **Mean / median latency (ms):** time to process one image; lower is better.
- **P95 latency:** a slow-but-realistic near-worst case, useful for interactive video.
- **FPS:** reciprocal of latency; higher is better.
- **Accuracy:** keep the same validation metrics for FP32 ONNX. Quantized INT8 models must be revalidated because accuracy can change.

## Next experiment

After this fair FP32 comparison, export INT8 using representative traffic data. Quantization lowers numerical precision to reduce model size and potentially speed up compatible edge hardware, but it can lower accuracy. Report the accuracy/latency/size trade-off rather than claiming it is automatically better.
