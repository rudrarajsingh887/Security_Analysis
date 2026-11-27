"""
utils.py
--------
Drawing utilities, colour helpers, and small shared functions used
across the project.
"""

import time
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
from loguru import logger


# ---------------------------------------------------------------
# Colour map  (BGR)
# ---------------------------------------------------------------
COLOURS = {
    "person":       (0, 200, 0),       # green
    "backpack":     (0, 165, 255),     # orange
    "handbag":      (0, 165, 255),
    "suitcase":     (0, 165, 255),
    "alert":        (0, 0, 230),       # red
    "fall":         (0, 0, 255),       # bright red
    "abandoned":    (0, 0, 200),
    "crowd":        (0, 50, 255),
    "inactivity":   (180, 0, 255),
    "erratic":      (255, 0, 180),
    "zone_breach":  (0, 0, 180),
    "white":        (255, 255, 255),
    "black":        (0, 0, 0),
    "dark_panel":   (20, 20, 20),
}

ALERT_LABEL_MAP = {
    "fall":                  "⚠ FALL DETECTED",
    "abandoned_object":      "⚠ ABANDONED OBJECT",
    "crowd_density":         "⚠ HIGH CROWD DENSITY",
    "suspicious_inactivity": "⚠ SUSPICIOUS INACTIVITY",
    "erratic_movement":      "⚠ ERRATIC MOVEMENT",
    "zone_breach":           "⚠ ZONE BREACH",
}


# ---------------------------------------------------------------
# Bounding box + track drawing
# ---------------------------------------------------------------

def draw_tracks(
    frame: np.ndarray,
    tracks: List[Dict[str, Any]],
    alert_ids: Optional[set] = None,
) -> np.ndarray:
    """
    Draw bounding boxes, track IDs, confidence scores, and motion trails.

    Parameters
    ----------
    frame     : BGR frame (modified in place)
    tracks    : list of tracker output dicts
    alert_ids : set of track_ids currently under an alert (drawn in red)
    """
    alert_ids = alert_ids or set()

    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        tid = t["track_id"]
        label = t["label"]
        conf = t["conf"]
        history = t.get("history", [])

        # Choose colour
        if tid in alert_ids:
            colour = COLOURS["alert"]
        else:
            colour = COLOURS.get(label, COLOURS["person"])

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # Label text: "person #7  0.88"
        text = f"{label} #{tid}  {conf:.2f}"
        _draw_label(frame, text, (x1, y1 - 4), colour)

        # Motion trail (last 20 centroids)
        _draw_trail(frame, history[-20:], colour)

    return frame


def draw_alerts(
    frame: np.ndarray,
    alerts: List[Dict[str, Any]],
) -> np.ndarray:
    """Overlay red boxes and alert labels for all active alerts."""
    for alert in alerts:
        x1, y1, x2, y2 = alert["bbox"]
        alert_type = alert.get("alert_type", "alert")
        label_text = ALERT_LABEL_MAP.get(alert_type, f"⚠ {alert_type.upper()}")
        colour = COLOURS.get(alert_type, COLOURS["alert"])

        # Thick red rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
        _draw_label(frame, label_text, (x1, y1 - 20), colour, font_scale=0.50)

    return frame


def draw_info_panel(
    frame: np.ndarray,
    fps: float,
    person_count: int,
    alerts: List[Dict[str, Any]],
    crowd_info: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Draw a semi-transparent status panel in the top-left corner.
    """
    panel_h, panel_w = 130, 260
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), COLOURS["dark_panel"], -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    lines = [
        f"FPS: {fps:.1f}",
        f"People: {person_count}",
        f"Alerts: {len(alerts)}",
    ]
    if crowd_info:
        lines.append(f"Crowd avg: {crowd_info.get('smoothed_count', 0):.0f}")

    for i, line in enumerate(lines):
        y = 22 + i * 26
        cv2.putText(
            frame, line, (8, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58,
            COLOURS["white"], 1, cv2.LINE_AA,
        )

    return frame


def draw_restricted_zones(
    frame: np.ndarray,
    zones: List[Dict[str, Any]],
) -> np.ndarray:
    """Draw restricted-zone polygons on the frame."""
    for zone in zones:
        pts = np.array(zone["points"], dtype=np.int32)
        cv2.polylines(frame, [pts], True, (0, 0, 200), 2)
        # Label at the top-left point of the polygon
        lx, ly = pts[0]
        _draw_label(frame, zone["name"], (lx, ly - 6), (0, 0, 200))
    return frame


# ---------------------------------------------------------------
# FPS tracker
# ---------------------------------------------------------------

class FPSCounter:
    """Rolling-window FPS estimator."""

    def __init__(self, window: int = 30) -> None:
        self._timestamps = []
        self._window = window

    def tick(self) -> float:
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) > self._window:
            self._timestamps.pop(0)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / max(elapsed, 1e-6)


# ---------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------

def open_video(source) -> cv2.VideoCapture:
    """
    Open a video source (file path or webcam index).
    Raises FileNotFoundError for missing files.
    """
    if isinstance(source, str):
        import os
        if source.isdigit():
            source = int(source)
        elif not os.path.isfile(source):
            raise FileNotFoundError(f"Video file not found: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Could not open video source: {source}")
    return cap


def get_video_properties(cap: cv2.VideoCapture) -> Dict:
    """Return basic properties of an open VideoCapture."""
    return {
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":    cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------

def _draw_label(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    colour: Tuple[int, int, int],
    font_scale: float = 0.48,
) -> None:
    """Draw a filled-background label for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    # Clamp to frame edges
    x = max(x, 0)
    y = max(y, th + baseline)
    cv2.rectangle(
        frame,
        (x, y - th - baseline),
        (x + tw, y + baseline),
        colour, -1
    )
    cv2.putText(
        frame, text, (x, y),
        font, font_scale,
        COLOURS["white"], thickness, cv2.LINE_AA,
    )


def _draw_trail(
    frame: np.ndarray,
    history: List[Tuple[int, int]],
    colour: Tuple[int, int, int],
) -> None:
    """Draw a fading motion trail from centroid history."""
    n = len(history)
    for i in range(1, n):
        alpha = i / n            # fade older points
        c = tuple(int(v * alpha) for v in colour)
        cv2.line(frame, history[i - 1], history[i], c, 2)
