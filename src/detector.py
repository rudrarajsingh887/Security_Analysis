"""
detector.py
-----------
Wraps Ultralytics YOLOv8 to perform object detection on individual frames.

Responsibilities:
  - Load the YOLO model once at startup.
  - Run inference on a frame and return structured detection results.
  - Filter detections by class ID and confidence threshold.
"""

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from loguru import logger

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "Ultralytics is not installed. Run: pip install ultralytics"
    )


# COCO class names we care about (subset for readability)
COCO_CLASSES = {
    0: "person",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
}


class ObjectDetector:
    """
    YOLOv8-based object detector.

    Parameters
    ----------
    model_name : str
        YOLO model weights file or Ultralytics model identifier
        e.g. 'yolov8n.pt', 'yolov8s.pt'.  The file is downloaded
        automatically on first run if it is not found locally.
    confidence_threshold : float
        Detections below this score are discarded (0–1).
    iou_threshold : float
        NMS IoU threshold (0–1).
    target_classes : list[int]
        COCO class IDs to keep.  Others are silently dropped.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        target_classes: List[int] = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes or [0, 24, 26, 28]

        logger.info(f"Loading YOLO model: {model_name}")
        model_path = Path(model_name)
        if not model_path.exists():
            logger.info(
                f"Model '{model_name}' not found locally — "
                "Ultralytics will download it automatically."
            )
        self.model = YOLO(model_name)
        logger.success(f"Model loaded: {model_name}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on a single BGR frame.

        Returns
        -------
        list of dict, each containing:
            bbox    : [x1, y1, x2, y2]   (ints, pixel coords)
            class_id: int
            label   : str  (e.g. 'person', 'backpack')
            conf    : float
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received — skipping detection.")
            return []

        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.target_classes,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "class_id": cls_id,
                        "label": COCO_CLASSES.get(cls_id, f"class_{cls_id}"),
                        "conf": round(conf, 3),
                    }
                )

        return detections

    def detect_persons(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Convenience method: return only person detections."""
        return [d for d in self.detect(frame) if d["class_id"] == 0]

    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Convenience method: return only bag/suitcase detections."""
        return [d for d in self.detect(frame) if d["class_id"] in (24, 26, 28)]
