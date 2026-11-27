"""
crowd_analysis.py
-----------------
Counts people per frame and raises crowd-density alerts when the count
exceeds a configurable threshold for a sustained number of frames.

Also provides a rolling-average smoothed count to avoid alert spam
from single-frame counting errors.
"""

from collections import deque
from typing import List, Dict, Any

from loguru import logger


class CrowdAnalyser:
    """
    Simple people counter with threshold-based alerting.

    Parameters
    ----------
    crowd_threshold : int
        Number of people that constitutes a crowd.
    alert_frames : int
        Number of consecutive over-threshold frames before an alert fires.
    smoothing_window : int
        Rolling window size for smoothing the raw count.
    """

    def __init__(
        self,
        crowd_threshold: int = 8,
        alert_frames: int = 5,
        smoothing_window: int = 10,
    ) -> None:
        self.crowd_threshold = crowd_threshold
        self.alert_frames = alert_frames

        # Rolling window of raw person counts
        self._count_history: deque = deque(maxlen=smoothing_window)

        # How many consecutive frames have been above threshold
        self._consecutive_high: int = 0

        # Prevent re-alerting once density is already known to be high
        self._alert_active: bool = False

        # Running statistics
        self.total_person_detections: int = 0
        self.peak_count: int = 0
        self.alert_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        tracks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Update crowd statistics for this frame.

        Parameters
        ----------
        tracks : all current tracked objects (persons + bags etc.)

        Returns
        -------
        dict:
            raw_count      : int   — persons in this frame
            smoothed_count : float — rolling average
            is_crowd       : bool  — is count above threshold?
            alert          : bool  — new alert this frame?
            alert_type     : str   — 'crowd_density' if alerting
        """
        # Count only person class
        person_count = sum(1 for t in tracks if t["class_id"] == 0)

        self._count_history.append(person_count)
        smoothed = sum(self._count_history) / len(self._count_history)

        self.total_person_detections += person_count
        self.peak_count = max(self.peak_count, person_count)

        is_crowd = person_count >= self.crowd_threshold
        new_alert = False

        if is_crowd:
            self._consecutive_high += 1
        else:
            self._consecutive_high = 0
            self._alert_active = False  # crowd dispersed — re-arm alert

        if (
            self._consecutive_high >= self.alert_frames
            and not self._alert_active
        ):
            self._alert_active = True
            new_alert = True
            self.alert_count += 1
            logger.warning(
                f"[CROWD ALERT] {person_count} people detected "
                f"(threshold={self.crowd_threshold})"
            )

        return {
            "raw_count": person_count,
            "smoothed_count": round(smoothed, 1),
            "is_crowd": is_crowd,
            "alert": new_alert,
            "alert_type": "crowd_density" if new_alert else None,
        }

    def summary(self) -> Dict[str, Any]:
        """Return overall statistics across all processed frames."""
        return {
            "peak_count": self.peak_count,
            "crowd_alerts_raised": self.alert_count,
            "crowd_threshold": self.crowd_threshold,
        }

    def reset(self) -> None:
        self._count_history.clear()
        self._consecutive_high = 0
        self._alert_active = False
        self.total_person_detections = 0
        self.peak_count = 0
        self.alert_count = 0
