# Methodology

## Overview

This document explains the technical approach behind each component of the
AI Surveillance & Anomaly Detection System.  Every module is deliberately
kept as a clear, explainable rule-based system — not a black-box neural
network — so that the logic can be audited, tuned, and improved.

---

## 1. Object Detection

**Model:** YOLOv8 (nano by default, configurable)  
**Library:** Ultralytics

YOLOv8 is a single-stage anchor-free object detector.  On each frame:

1. The frame is resized to the model's input size (640×640 by default).
2. The CNN backbone extracts multi-scale feature maps.
3. Prediction heads output bounding box coordinates, class probabilities,
   and objectness scores.
4. Non-Maximum Suppression (NMS) removes overlapping boxes below the
   configured IoU threshold.

We restrict detections to four COCO classes:

| Class ID | Label     | Rationale                           |
|----------|-----------|-------------------------------------|
| 0        | person    | Primary subject of analysis         |
| 24       | backpack  | Common abandoned-object candidate   |
| 26       | handbag   | Common abandoned-object candidate   |
| 28       | suitcase  | High-priority abandoned-object risk |

---

## 2. Object Tracking

**Algorithm:** ByteTrack (via Ultralytics built-in integration)

ByteTrack uses a two-stage association strategy:

1. **High-confidence detections** are matched to existing tracks using
   the Hungarian algorithm with IoU as the cost metric.
2. **Low-confidence detections** (often partial occlusions) are matched
   to unmatched tracks in a second pass, recovering tracks that would
   otherwise be lost.

Each confirmed track receives a persistent integer ID.  The system
maintains a centroid history deque (last 60 positions) per track, which
is used by downstream anomaly modules.

---

## 3. Fall Detection Logic

Fall detection relies on the observation that a standing person's
bounding box is **taller than it is wide**, while a fallen person's
box is **wider than it is tall**.

**Criterion 1 — Aspect ratio check:**
```
aspect_ratio = bbox_width / bbox_height
fall_suspected = aspect_ratio >= threshold  # default 1.2
```

**Criterion 2 — Position check:**
The bottom of the bounding box must be in the lower 70 % of the frame,
ensuring we don't flag people waving horizontally or leaning.

**Criterion 3 — Sudden drop (supplementary):**
If the centroid Y position drops more than `sudden_drop_pixels` in a
single frame, it is treated as an instantaneous fall indicator.

An alert fires only when criteria are met for `consecutive_frames`
frames in a row, reducing false positives from brief posture changes.

---

## 4. Abandoned Object Detection

**Step 1 — Stationarity check:**  
For each bag/backpack/suitcase track, the detector compares the current
centroid to the last recorded reference centroid.  If displacement is
below `stationary_pixel_threshold` pixels, the object has not moved.

**Step 2 — Timer:**  
A frame-based timer starts at first detection.  If the object remains
stationary for `stationary_seconds`, we proceed to step 3.

**Step 3 — Person proximity:**  
A Euclidean distance check determines whether any person centroid is
within `proximity_threshold` pixels of the object.  If **no person is
within range**, the object is flagged as abandoned.

**Reset:** If the object moves or a person approaches, the timer resets.

---

## 5. Crowd Analysis

At each frame, the system counts tracked persons and appends the count to
a rolling window.  The smoothed (rolling average) count is displayed on
the overlay.

A **crowd alert** fires when the raw count exceeds `crowd_threshold` for
`alert_frames` consecutive frames.  This debounce prevents alert spam
from momentary counting errors.

---

## 6. Anomaly Detection Rules

### 6a. Suspicious Inactivity

1. Compute per-frame centroid displacement for each person track.
2. If displacement < `stationary_pixel_threshold` px for
   `inactivity_seconds` seconds consecutively, raise an alert.
3. Reset when the person moves.

### 6b. Erratic Movement

1. Collect per-frame displacement values in a short rolling window
   (15 frames).
2. Count frames where displacement > `erratic_movement_threshold` px.
3. If `erratic_frames_required` frames in the window exceed the
   threshold, raise an erratic-movement alert.

### 6c. Restricted Zone Breach

1. User defines one or more polygons in `config.yaml`.
2. Each frame, a ray-casting point-in-polygon test checks whether a
   person's centroid is inside a zone.
3. If inside, a zone-breach alert is raised with the zone name.

---

## 7. Drawing & Visualisation

- Bounding boxes are colour-coded by class (green for persons, orange for
  objects) and turn red when under an active alert.
- Track IDs and confidence scores are shown on each box.
- A motion trail (last 20 centroids) visualises recent movement paths.
- A semi-transparent HUD panel shows FPS, people count, and alert count.
- Alert labels are drawn above the offending bounding box.
