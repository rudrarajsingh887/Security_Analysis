"""
tracker.py
----------
Multi-object tracker built on top of Ultralytics' built-in ByteTrack
integration.  Assigns persistent IDs to detected bounding boxes across
frames and maintains centroid history for motion analysis.

ByteTrack is preferred over DeepSORT for this project because:
  - It ships with Ultralytics — no extra install required.
  - It is fast enough to run on CPU.
  - It handles occlusion well with its two-stage matching.

If you want DeepSORT, replace the `update` method with a call to the
`deep_sort_realtime` package (commented out below).
"""

from collections import defaultdict, deque
from typing import List, Dict, Any, Tuple

import numpy as np
from loguru import logger

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Run: pip install ultralytics")


class ObjectTracker:
    """
    Wraps YOLO + ByteTrack to produce tracked detections with persistent IDs.

    Parameters
    ----------
    model_name : str
        Same YOLO weights used for detection (tracking reuses the model).
    confidence_threshold : float
    iou_threshold : float
    target_classes : list[int]
    max_history : int
        How many past centroid positions to keep per track.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        target_classes: List[int] = None,
        max_history: int = 60,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes or [0, 24, 26, 28]
        self.max_history = max_history

        logger.info(f"Loading tracker model: {model_name}")
        self.model = YOLO(model_name)

        # centroid_history[track_id] = deque of (cx, cy) tuples
        self.centroid_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=max_history)
        )

        # frame_count_seen[track_id] = number of frames this track was visible
        self.frame_count_seen: Dict[int, int] = defaultdict(int)

        logger.success("Tracker ready (ByteTrack via Ultralytics)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection + ByteTrack on one frame.

        Returns
        -------
        list of dict:
            track_id  : int   — persistent ID across frames
            bbox      : [x1, y1, x2, y2]
            class_id  : int
            label     : str
            conf      : float
            centroid  : (cx, cy)
            history   : list of (cx, cy) — past positions
        """
        if frame is None or frame.size == 0:
            return []

        results = self.model.track(
            frame,
            persist=True,            # keep ByteTrack state between calls
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.target_classes,
            tracker="bytetrack.yaml", # use ByteTrack (ships with ultralytics)
            verbose=False,
        )

        tracks = []
        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue  # no tracked boxes in this frame

            for box in boxes:
                track_id = int(box.id[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Compute centroid
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Update history
                self.centroid_history[track_id].append((cx, cy))
                self.frame_count_seen[track_id] += 1

                label = self._class_label(cls_id)

                tracks.append(
                    {
                        "track_id": track_id,
                        "bbox": [x1, y1, x2, y2],
                        "class_id": cls_id,
                        "label": label,
                        "conf": round(conf, 3),
                        "centroid": (cx, cy),
                        "history": list(self.centroid_history[track_id]),
                    }
                )

        return tracks

    def get_centroid_history(self, track_id: int) -> List[Tuple[int, int]]:
        """Return the centroid trail for a given track ID."""
        return list(self.centroid_history.get(track_id, []))

    def get_frames_seen(self, track_id: int) -> int:
        """Return how many frames a track has been active."""
        return self.frame_count_seen.get(track_id, 0)

    def reset(self) -> None:
        """Clear all track state (call between video files)."""
        self.centroid_history.clear()
        self.frame_count_seen.clear()
        logger.info("Tracker state reset.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _class_label(cls_id: int) -> str:
        labels = {0: "person", 24: "backpack", 26: "handbag", 28: "suitcase"}
        return labels.get(cls_id, f"class_{cls_id}")
