# Learning notes

## Phase 0 - Environment and repository

### What I set up

- A dedicated Python virtual environment (`.venv`) so this project's packages do not affect other projects.
- A Git repository so changes can be saved as understandable milestones.
- A repository layout that keeps source code, tests, raw data, trained models, and generated output separate.

### Key ideas in my own words

< Ok, so now i am starting this project of object detection , it is divided into 8 phases and the 0th step is all about intialization .>

### Evidence

<!-- Add the output of `python --version` and `git status --short` here. -->
< Python 3.14.6 >
< So i got understanding of virtual environment. it is place where we can install specific libraries related to our project which does not hinder the other files.>
< and git ignore tells the git to ignopre specific installations or files which are as such not necessary keeping the repo short and sweet.>
### Questions I still have

<!-- Add questions here. Questions are a useful part of learning. -->

## Phase 1 - Video fundamentals

### Concepts to explain after the exercise

- A video is a sequence of image frames plus timing information (FPS).
- An image is a NumPy array: height x width x color channels.
- OpenCV uses BGR color order by default, whereas many other tools use RGB.
- `VideoCapture.read()` returns one frame at a time; `VideoWriter.write()` saves one frame at a time.

### Evidence to add

<Processed 651 frame(s) -> outputs/my-video-annotated.mp4
Input metadata: 2160x3840, 59.94 FPS, 651 frames, about 10.86 seconds>

### Questions I still have

<!-- Add questions here. -->

## Phase 2 - Object detection

### Concepts to explain after the exercise

- Detection answers “what objects are in this frame, and where are they?”
- A bounding box uses `(x1, y1, x2, y2)`: left, top, right, and bottom pixel positions.
- A class ID is a numeric label; the model maps it to a name such as `person` or `car`.
- Confidence is the model's estimated certainty, not a guarantee of correctness.
- A detector processes each frame independently. It does not yet know that a person in frame 10 is the same person in frame 11.

### Evidence to add

<!-- Record your video, model, confidence threshold, device, and example true/false detections. -->

### Questions I still have

<!-- Add questions here. -->

## Phase 3 - Multi-object tracking

### What I built

- An IoU-based tracker in `track_video.py` that assigns an ID to each detected object and attempts to keep that ID in later frames.
- A ByteTrack pipeline in `track_bytetrack.py`, using YOLO's built-in ByteTrack integration and the `lap` matching library.
- Annotated labels such as `person #1 0.88`, where `#1` is the track ID and `0.88` is the detector confidence.

### Core ideas

- **Detection** looks at one frame and answers: “What is here, and where?”
- **Tracking** compares the current frame with earlier frames and answers: “Is this the same object as before?”
- An ID is not a class label. Two people can both have class `person`, but should receive different IDs such as `person #1` and `person #2`.
- **IoU (Intersection over Union)** measures how much two boxes overlap: `overlap area / combined area`. A high IoU suggests that two boxes may describe the same object in consecutive frames.
- The learning tracker matches same-class boxes with IoU at least `0.30`; otherwise, it creates a new ID.
- ByteTrack is more robust than this simple tracker because it uses motion prediction and also associates lower-confidence detections. It handles short missed detections and occlusion better.
- A track ID is useful, but it is not automatically a perfect count: rapid motion, overlapping people, or a missed detection can cause an ID switch.

### Evidence to add

<!-- Run the command below on a people/vehicle video and paste the final output here. -->

```bash
python -m edge_tracker.track_bytetrack \
  --input data/raw/my-video.mp4 \
  --output outputs/my-video-bytetrack.mp4 \
  --classes person car bus truck \
  --confidence 0.40 \
  --device cpu
```

<!-- Observe one object for several frames. Did it retain its ID? Note any ID switch or missed detection. -->

### Questions I still have

<!-- Why does IoU matching fail when objects cross paths? How does a Kalman filter help ByteTrack predict the next box location? -->

## Phase 4 - Trajectories and video analytics

### What I built

- A trajectory: the recent centroid positions of each tracked ID, drawn as a line behind the object.
- A horizontal virtual counting line, placed at a configurable fraction of the video height.
- Upward and downward crossing counters, plus a rolling inference-FPS metric.

### Core ideas

- A bounding-box centroid is its center point: `((x1 + x2) / 2, (y1 + y2) / 2)`.
- A trajectory is a short history of those center points; it shows an object's motion path.
- A crossing event occurs when the centroid moves from one side of the counting line to the other.
- Each `(track ID, direction)` pair is counted only once. This prevents a stationary object near the line from being counted repeatedly.
- **Inference FPS** measures how quickly the model-and-tracker pipeline processes frames. It is different from the video file's FPS, which is the original playback rate.
- A unique track ID is an approximation, not a guaranteed real-world count: tracker ID switches can overcount and missed detections can undercount.

### Evidence to add

```bash
python -m edge_tracker.analyze_video \
  --input data/raw/my-video.mp4 \
  --output outputs/my-video-analytics.mp4 \
  --classes person car bus truck \
  --line-fraction 0.50 \
  --device cpu
```

<!-- Record the final unique-ID, up, and down totals. Did the line sit at a meaningful place in your video? -->

### Questions I still have

<!-- How would I count only cars? How can I stop an ID switch from creating an extra count? -->

## Phase 5 - Custom model training

### What I completed

- Prepared a BDD100K subset with 500 training and 150 validation images for `person`, `car`, `bus`, and `truck`.
- Converted BDD100K JSON corner boxes to YOLO normalized label files and passed dataset validation.
- Fine-tuned pretrained `yolo11n.pt` for 20 epochs with batch size 2 on CPU.
- Loaded the resulting `best.pt` successfully and evaluated the custom model.

### Core ideas

- **Transfer learning** starts from a pretrained model and adapts it to a smaller, task-specific dataset.
- A YOLO label line is `class_id x_center y_center width height`; box values are normalized from 0 to 1.
- Keep train, validation, and test images separate. Split by source video, not by neighboring frames, to prevent data leakage.
- Precision answers “when the model predicts an object, how often is it right?” Recall answers “of the real objects, how many did it find?” mAP summarizes detection quality across confidence thresholds.
- A model that performs very well on training images but poorly on validation images is overfitting.

### Evidence

```text
Dataset validation: passed
Training images: 500
Validation images: 150
Model: yolo11n.pt (pretrained)
Training: 20 epochs, batch size 2, CPU
Final validation precision: 45.38%
Final validation recall: 29.79%
Final validation mAP50: 31.10%
Final validation mAP50-95: 17.03%
Saved model: runs/detect/runs/train/traffic_yolo11n/weights/best.pt
```

### Interpretation

- `mAP50` uses a lenient 0.50 IoU match threshold; `mAP50-95` averages stricter thresholds from 0.50 to 0.95, so it is normally lower.
- This first model is a baseline, not a final accuracy claim. The low recall means it misses many real objects; more varied images, better class balance, more epochs, and error analysis are the next improvements.

### Questions I still have

<!-- Why should videos, rather than individual frames, define the data split? How does mAP50 differ from mAP50-95? -->
