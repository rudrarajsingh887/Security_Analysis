"""
helpers.py
----------
Shared test utilities for constructing synthetic "track" dictionaries that
match the output format of `ObjectTracker.update()`.

This lets the rule-based anomaly modules (fall detection, abandoned object,
crowd analysis, anomaly detector) be unit-tested with plain Python data —
no YOLO model, video file, or network access required.
"""

from typing import List, Optional, Tuple

# Mirrors the COCO subset used throughout src/
CLASS_LABELS = {0: "person", 24: "backpack", 26: "handbag", 28: "suitcase"}


def make_track(
    track_id: int,
    bbox: List[int],
    class_id: int = 0,
    conf: float = 0.9,
    history: Optional[List[Tuple[int, int]]] = None,
) -> dict:
    """
    Build a synthetic tracked-detection dict matching ObjectTracker output.

    Parameters
    ----------
    track_id : persistent track ID
    bbox     : [x1, y1, x2, y2] pixel coordinates
    class_id : COCO class ID (0=person, 24=backpack, 26=handbag, 28=suitcase)
    conf     : detection confidence
    history  : optional centroid history; defaults to a single-point history
               at the current centroid

    Returns
    -------
    dict with keys: track_id, bbox, class_id, label, conf, centroid, history
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    return {
        "track_id": track_id,
        "bbox": list(bbox),
        "class_id": class_id,
        "label": CLASS_LABELS.get(class_id, f"class_{class_id}"),
        "conf": conf,
        "centroid": (cx, cy),
        "history": history or [(cx, cy)],
    }
