# Phase 5: Custom detector training

## Goal

Fine-tune a small pretrained YOLO model to recognize `person`, `car`, `bus`, and `truck` in footage similar to the footage you will deploy on. Pretraining gives the model general visual features; fine-tuning adapts it to your camera angle, lighting, and scene.

## Dataset layout

```text
data/traffic/
  images/train/       70% of images
  images/val/         20% of images
  images/test/        10% of images (optional until final evaluation)
  labels/train/       matching .txt labels
  labels/val/         matching .txt labels
  labels/test/        matching .txt labels
```

Each image `images/train/frame_001.jpg` needs `labels/train/frame_001.txt`. One object per line uses YOLO format:

```text
class_id x_center y_center width height
```

All four box values are normalized from 0 to 1. Example: `1 0.50 0.60 0.20 0.15` is a `car` whose box is centered at 50% width and 60% height.

## BDD100K conversion

The downloaded BDD100K package stores annotations as one JSON file per image. `prepare_bdd100k.py` keeps only `person`, `car`, `bus`, and `truck`, maps them to this project's class IDs, converts the corner boxes to normalized YOLO labels, and copies a random subset.

```bash
python -m edge_tracker.prepare_bdd100k \
  --source /Users/avinashraval/Downloads/bdd100k \
  --train-count 500 \
  --val-count 150
```

Use `--dry-run` first to confirm the source without copying. The 500/150 subset is appropriate for a first CPU training experiment; increase it only after completing one full training/evaluation cycle.

## Recommended learning workflow

1. Collect 150-300 varied frames from videos resembling your target scene. Include night/day, near/far objects, partial occlusion, and negative images with no target objects.
2. Annotate every target object with a tool such as CVAT, Label Studio, or Roboflow Annotate. Export in YOLO detection format.
3. Keep the train/validation split at the **video level**, not random adjacent frames. Otherwise nearly identical frames leak into validation and produce misleading metrics.
4. Run `python -m edge_tracker.validate_dataset --data data/traffic.yaml` until it passes.
5. Train first with 50 epochs and batch size 4 on this M1 system. Reduce `--batch` if memory fails.
6. Read the saved `results.png`, then evaluate `best.pt` using `evaluate_custom.py`. It is saved under `runs/train/<run name>/weights/best.pt` for new runs.
7. Compare custom and pretrained models on an untouched video. Report mAP, FPS, model size, common false positives, and common missed objects.

## Concepts to study

- Transfer learning and fine-tuning
- YOLO normalized bounding-box format
- Train/validation/test splits and data leakage
- Data augmentation
- Precision, recall, mAP50, and mAP50-95
- Overfitting and early stopping
