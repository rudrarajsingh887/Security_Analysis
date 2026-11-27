"""
anomaly_detector.py
-------------------
Rule-based anomaly detector for suspicious individual behaviour:

  1. Prolonged inactivity  — person standing still for too long.
  2. Erratic movement      — centroid jumping unusually far between frames.
  3. Restricted zone entry — person centroid inside a user-defined polygon ROI.

Design note
-----------
This is an MVP rule-based system, NOT a trained behavioural model.
It will produce false positives in crowded scenes, with camera shake,
or when people naturally pause (e.g. reading a notice board).
Use the thresholds in config.yaml to calibrate for your scene.
"""

from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from loguru import logger


class AnomalyDetector:
    """
    Raises alerts for suspicious behavioural patterns.

    Parameters
    ----------
    inactivity_seconds : float
        Seconds a person must be stationary before 'suspicious inactivity'.
    stationary_pixel_threshold : int
        Max pixels/frame movement that still counts as 'stationary'.
    erratic_movement_threshold : int
        Centroid displacement per frame above which movement is 'erratic'.
    erratic_frames_required : int
        How many erratic frames in a short window trigger the alert.
    fps : float
        Video FPS, used to convert frame counts → seconds.
    restricted_zones : list[dict]
        Optional list of {'name': str, 'points': [[x,y], ...]} dicts.
    """

    def __init__(
        self,
        inactivity_seconds: float = 20.0,
        stationary_pixel_threshold: int = 8,
        erratic_movement_threshold: int = 80,
        erratic_frames_required: int = 5,
        fps: float = 25.0,
        restricted_zones: Optional[List[Dict]] = None,
    ) -> None:
        self.inactivity_seconds = inactivity_seconds
        self.stationary_pixel_threshold = stationary_pixel_threshold
        self.erratic_movement_threshold = erratic_movement_threshold
        self.erratic_frames_required = erratic_frames_required
        self.fps = max(fps, 1.0)
        self.restricted_zones = restricted_zones or []

        # Convert zone polygons to numpy arrays once
        self._zone_arrays = [
            {
                "name": z["name"],
                "pts": np.array(z["points"], dtype=np.int32),
            }
            for z in self.restricted_zones
        ]

        # Per-track state
        self._stationary_since: Dict[int, int] = {}   # frame idx
        self._last_centroid: Dict[int, Tuple[int, int]] = {}
        self._erratic_window: Dict[int, List[float]] = defaultdict(list)
        self._alerted_inactivity: Dict[int, bool] = defaultdict(bool)
        self._zone_inside: Dict[Tuple[int, str], bool] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        tracks: List[Dict[str, Any]],
        frame_idx: int,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate all person tracks for anomalous behaviour.

        Parameters
        ----------
        tracks    : all current tracked objects
        frame_idx : current frame index

        Returns
        -------
        List of alert dicts with keys:
            track_id, centroid, alert_type, detail
        """
        person_tracks = [t for t in tracks if t["class_id"] == 0]
        active_ids = {t["track_id"] for t in person_tracks}

        # Clean up state for disappeared tracks
        for tid in list(self._stationary_since.keys()):
            if tid not in active_ids:
                self._stationary_since.pop(tid, None)
                self._last_centroid.pop(tid, None)
                self._erratic_window.pop(tid, None)
                self._alerted_inactivity[tid] = False
         # Clean up zone history for disappeared tracks
        for key in list(self._zone_inside):
            if key[0] not in active_ids:
                self._zone_inside.pop(key, None)       

        alerts = []

        for track in person_tracks:
            tid = track["track_id"]
            cx, cy = track["centroid"]
        
            # ---- Movement delta ----
            if tid in self._last_centroid:
                lx, ly = self._last_centroid[tid]
                delta = np.hypot(cx - lx, cy - ly)
            else:
                delta = 0.0
            self._last_centroid[tid] = (cx, cy)

            # ---- Inactivity check ----
            if delta <= self.stationary_pixel_threshold:
                if tid not in self._stationary_since:
                    self._stationary_since[tid] = frame_idx
                frames_still = frame_idx - self._stationary_since[tid]
                seconds_still = frames_still / self.fps

                if (
                    seconds_still >= self.inactivity_seconds
                    and not self._alerted_inactivity[tid]
                ):
                    self._alerted_inactivity[tid] = True
                    logger.warning(
                        f"[INACTIVITY] Track {tid} stationary "
                        f"for {seconds_still:.1f}s"
                    )
                    alerts.append(self._make_alert(
                        tid, track, "suspicious_inactivity",
                        f"Still for {seconds_still:.0f}s"
                    ))
                elif self._alerted_inactivity.get(tid):
                    # Continue flagging while condition persists
                    alerts.append(self._make_alert(
                        tid, track, "suspicious_inactivity", "ongoing"
                    ))
            else:
                # Person moved — reset inactivity clock
                self._stationary_since.pop(tid, None)
                self._alerted_inactivity[tid] = False

            # ---- Erratic movement check ----
            window = self._erratic_window[tid]
            window.append(delta)
            if len(window) > 15:           # keep a short rolling window
                window.pop(0)

            erratic_frames = sum(
                1 for d in window if d > self.erratic_movement_threshold
            )
            if erratic_frames >= self.erratic_frames_required:
                alerts.append(self._make_alert(
                    tid, track, "erratic_movement",
                    f"{erratic_frames} erratic frames in last {len(window)}"
                ))

                        # ---- Restricted zone check ----
            for zone in self._zone_arrays:
                zone_key = (tid, zone["name"])
                inside = self._point_in_polygon((cx, cy), zone["pts"])

                previous_inside = self._zone_inside.get(zone_key)

                # First observation: establish baseline, not an entry.
                if previous_inside is None:
                    self._zone_inside[zone_key] = inside

                    if inside:
                        alerts.append(self._make_alert(
                            tid, track, "zone_presence",
                            f"Already inside: {zone['name']}"
                        ))

                # Genuine outside-to-inside transition.
                elif inside and not previous_inside:
                    self._zone_inside[zone_key] = True

                    logger.warning(
                        f"[ZONE BREACH] Track {tid} entered '{zone['name']}'"
                    )

                    alerts.append(self._make_alert(
                        tid, track, "zone_breach",
                        f"Entered zone: {zone['name']}"
                    ))

                # Continuing presence: not a new intrusion incident.
                elif inside:
                    alerts.append(self._make_alert(
                        tid, track, "zone_presence",
                        f"In zone: {zone['name']}"
                    ))

                # Person exited; allow a future entry event.
                else:
                    self._zone_inside[zone_key] = False

        return alerts

    def reset(self) -> None:
        self._stationary_since.clear()
        self._last_centroid.clear()
        self._erratic_window.clear()
        self._alerted_inactivity.clear()
        self._zone_inside.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_alert(
        track_id: int,
        track: Dict[str, Any],
        alert_type: str,
        detail: str,
    ) -> Dict[str, Any]:
        return {
            "track_id": track_id,
            "bbox": track["bbox"],
            "centroid": track["centroid"],
            "alert_type": alert_type,
            "detail": detail,
        }

    @staticmethod
    def _point_in_polygon(
        point: Tuple[int, int],
        polygon: np.ndarray,
    ) -> bool:
        """
        Ray-casting polygon test.
        Returns True if `point` is inside `polygon`.
        """
        x, y = point
        n = len(polygon)
        inside = False
        px, py = polygon[0]
        for i in range(1, n + 1):
            qx, qy = polygon[i % n]
            if y > min(py, qy):
                if y <= max(py, qy):
                    if x <= max(px, qx):
                        if py != qy:
                            xinters = (y - py) * (qx - px) / (qy - py) + px
                        if px == qx or x <= xinters:
                            inside = not inside
            px, py = qx, qy
        return inside
