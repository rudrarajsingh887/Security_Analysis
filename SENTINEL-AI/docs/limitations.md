# Limitations and Future Work

## Current Limitations

### 1. Rule-Based Anomaly Detection

All anomaly logic in this system consists of hand-crafted geometric
heuristics.  This means:

- **False positives are common.**  A person sitting on a bench will
  trigger the inactivity alert.  A child running will likely trigger
  the erratic-movement alert.
- **Rules do not generalise.**  Thresholds that work for one camera
  angle or scene will need manual re-tuning for another.
- **No semantic understanding.**  The system cannot distinguish
  "person helping an injured colleague" from "person ignoring a fallen
  colleague."

### 2. Fall Detection Limitations

- Relies entirely on bounding-box aspect ratio.  A person crouching to
  pick something up may trigger a false fall alert.
- Cannot distinguish voluntary lying down (e.g. on a yoga mat) from an
  emergency fall.
- The ideal solution is skeleton-based pose estimation (e.g. MediaPipe
  Pose, OpenPose) combined with a trained action classifier.

### 3. Tracking Reliability

- ByteTrack can lose and re-acquire tracks during heavy occlusion,
  resetting track IDs and any associated timers.
- The system has not been evaluated against standard MOT benchmarks
  (MOT17, MOT20).
- Re-ID (re-identification across camera cuts) is not supported.

### 4. Abandoned Object Detection

- Relies on consistent object track IDs; ID switches reset the
  stationarity timer.
- Does not handle objects that are briefly occluded by passing people.
- The proximity heuristic can fail in crowded scenes.

### 5. Dataset and Evaluation

- The system has not been trained or evaluated on a labelled dataset.
- No precision / recall / F1 metrics are reported.
- Optimal thresholds are scene-dependent and have not been
  empirically validated.

### 6. Hardware Constraints

- The nano YOLOv8 model was chosen for CPU compatibility, but real-time
  performance (>25 FPS at 1080p) requires a GPU.
- Processing is single-threaded; a production system would pipeline
  frame capture, inference, and display.

---

## Privacy and Ethical Considerations

- **This system must not be deployed on real CCTV footage without
  explicit legal authorisation and informed consent.**
- Automated surveillance systems carry significant risks of misuse,
  bias, and chilling effects on public behaviour.
- Any production deployment must comply with GDPR, the UK ICO
  surveillance camera code of practice, and relevant local laws.
- The system should not be used to profile, discriminate, or make
  automated decisions about individuals.

---

## Future Work

| Improvement | Description | Effort |
|-------------|-------------|--------|
| Pose-based fall detection | Replace aspect-ratio heuristic with MediaPipe Pose | Medium |
| Action recognition | Fine-tune a temporal model (SlowFast, VideoMAE) for violence / running / falling | High |
| Re-identification | Cross-camera person Re-ID with OSNet or CLIP features | High |
| Anomaly ML model | Replace hand-crafted rules with an unsupervised anomaly detector (Isolation Forest, Autoencoder on motion features) | Medium |
| Evaluation suite | Benchmark against public datasets (CUHK-Avenue, ShanghaiTech) | Medium |
| RTSP support | Pull live feeds from IP cameras | Low |
| GPU acceleration | TensorRT export for YOLOv8, async pipeline | Medium |
| Alert logging | Write timestamped alerts to a database or CSV | Low |
| Email/SMS notifications | Trigger alerts via Twilio / SendGrid | Low |
| Multi-camera support | Process multiple streams simultaneously | High |

---

## Honest Assessment for Recruiters

This project demonstrates competence in:

- Applying pre-trained deep learning models (YOLOv8) to real video.
- Implementing a complete computer vision pipeline from detection to
  visualisation.
- Designing a modular, maintainable Python codebase.
- Building a usable Streamlit interface for non-technical stakeholders.
- Thinking critically about the limitations of rule-based systems and
  articulating the path to production.

It does **not** claim production-grade surveillance accuracy, and the
author is transparent about where learning models would be required to
improve reliability.
