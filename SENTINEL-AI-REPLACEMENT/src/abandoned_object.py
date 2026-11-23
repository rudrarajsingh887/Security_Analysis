"""
abandoned_object.py
-------------------
Detects bags / backpacks / suitcases that have been left unattended.

Algorithm
---------
1. For each tracked bag/backpack/suitcase:
     a. Record the frame timestamp when it was first seen.
     b. Track its centroid displacement across frames.
     c. If displacement stays below a pixel threshold for
        `stationary_seconds` or more, the object is deemed stationary.
2. Check whether any person is within `proximity_threshold` pixels.
3. If stationary AND no nearby person → raise an "abandoned object" alert.

Reset: once the object disappears from tracking (picked up), its record
       is deleted.

Limitations
-----------
  - Relies on consistent tracking IDs.  If the tracker loses and
    re-acquires the object, the timer resets.
  - Does not account for objects partially behind other objects.
  - A robust solution would use background subtraction + blob analysis.
"""

import time
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from loguru import logger

# Object COCO class IDs we treat as potentially abandonable
OBJECT_CLASS_IDS = {24, 26, 28}  # backpack, handbag, suitcase


class AbandonedObjectDetector:
    """
    Monitors luggage / bag track IDs for stationary + unattended patterns.

    Parameters
    ----------
    stationary_seconds : float
        Seconds an object must stay still before being flagged.
    proximity_threshold : int
        Pixel distance within which a person is considered 'attending'
        the object.
    stationary_pixel_threshold : int
        Max centroid drift (pixels) per frame that still counts as
        'stationary'.
    min_visible_frames : int
        Object must be visible for at least this many frames before
        timing starts (avoids false alerts on new detections).
    fps : float
        Video FPS — used to convert frame counts to seconds.
    """

    def __init__(
        self,
        stationary_seconds: float = 10.0,
        proximity_threshold: int = 120,
        stationary_pixel_threshold: int = 10,
        min_visible_frames: int = 15,
        fps: float = 25.0,
    ) -> None:
        self.stationary_seconds = stationary_seconds
        self.proximity_threshold = proximity_threshold
        self.stationary_pixel_threshold = stationary_pixel_threshold
        self.min_visible_frames = min_visible_frames
        self.fps = max(fps, 1.0)

        # State per object track
        # { track_id: { 'first_seen': int, 'last_moved': int,
        #               'ref_centroid': (cx, cy), 'frames_seen': int,
        #               'alerted': bool } }
        self._objects: Dict[int, Dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        tracks: List[Dict[str, Any]],
        frame_idx: int,
    ) -> List[Dict[str, Any]]:
        """
        Update the state machine and return active abandoned-object alerts.

        Parameters
        ----------
        tracks   : combined list of all tracks (persons + objects)
        frame_idx: current frame index (used as a frame-based clock)

        Returns
        -------
        list of alert dicts:
            track_id, bbox, centroid, alert_type='abandoned_object',
            seconds_unattended
        """
        object_tracks = [t for t in tracks if t["class_id"] in OBJECT_CLASS_IDS]
        person_tracks = [t for t in tracks if t["class_id"] == 0]
        active_obj_ids = {t["track_id"] for t in object_tracks}

        # Remove records for objects that disappeared
        for tid in list(self._objects.keys()):
            if tid not in active_obj_ids:
                del self._objects[tid]

        alerts = []

        for obj_track in object_tracks:
            tid = obj_track["track_id"]
            cx, cy = obj_track["centroid"]

            if tid not in self._objects:
                # First time we see this object — initialise record
                self._objects[tid] = {
                    "first_seen": frame_idx,
                    "last_moved": frame_idx,
                    "ref_centroid": (cx, cy),
                    "frames_seen": 1,
                    "alerted": False,
                }
                continue

            record = self._objects[tid]
            record["frames_seen"] += 1

            # Check movement since last reference position
            rx, ry = record["ref_centroid"]
            displacement = np.hypot(cx - rx, cy - ry)

            if displacement > self.stationary_pixel_threshold:
                # Object moved — reset the timer
                record["last_moved"] = frame_idx
                record["ref_centroid"] = (cx, cy)
                record["alerted"] = False
            else:
                # Object has not moved since ref_centroid was recorded
                pass

            # Only evaluate after minimum visible frames
            if record["frames_seen"] < self.min_visible_frames:
                continue

            frames_stationary = frame_idx - record["last_moved"]
            seconds_stationary = frames_stationary / self.fps

            if seconds_stationary < self.stationary_seconds:
                continue

            # Check if any person is nearby
            person_nearby = self._is_person_nearby(
                (cx, cy), person_tracks
            )

            if not person_nearby:
                if not record["alerted"]:
                    record["alerted"] = True
                    logger.warning(
                        f"[ABANDONED OBJECT] Track {tid} | label={obj_track['label']} "
                        f"| stationary for {seconds_stationary:.1f}s"
                    )
                alerts.append(
                    {
                        "track_id": tid,
                        "bbox": obj_track["bbox"],
                        "centroid": obj_track["centroid"],
                        "label": obj_track["label"],
                        "alert_type": "abandoned_object",
                        "seconds_unattended": round(seconds_stationary, 1),
                    }
                )
            else:
                # Someone is near — cancel alert
                record["alerted"] = False

        return alerts

    def reset(self) -> None:
        self._objects.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_person_nearby(
        self,
        obj_centroid: Tuple[int, int],
        person_tracks: List[Dict[str, Any]],
    ) -> bool:
        """Return True if any person centroid is within proximity_threshold."""
        ox, oy = obj_centroid
        for person in person_tracks:
            px, py = person["centroid"]
            dist = np.hypot(px - ox, py - oy)
            if dist <= self.proximity_threshold:
                return True
        return False
