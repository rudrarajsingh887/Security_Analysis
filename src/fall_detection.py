"""
fall_detection.py
-----------------
Rule-based fall detection using bounding-box geometry.

Logic overview
--------------
A person is flagged as having fallen when ALL of the following are true
for `consecutive_frames` in a row:

  1. Aspect ratio  (width / height) >= aspect_ratio_threshold
     A standing person is tall and narrow; a fallen person is wide and short.

  2. The bounding-box BOTTOM EDGE is in the lower portion of the frame
     (the person is close to the ground, not just waving their arms).

  3. Optionally, a sudden downward drop in centroid Y position was
     observed in a recent frame (sudden_drop_pixels).

Limitations
-----------
  - Fails for people lying on elevated surfaces (beds, sofas).
  - High false-positive rate in crowded scenes where boxes overlap.
  - Does not distinguish voluntary lying from an emergency fall.
  - A proper solution would use pose estimation (e.g. MediaPipe Pose)
    or a trained action-recognition model.
"""

from collections import defaultdict
from typing import Dict, List, Any

from loguru import logger


class FallDetector:
    """
    Tracks per-person bounding-box geometry and raises a fall alert when
    the aspect-ratio criterion persists for enough consecutive frames.

    Parameters
    ----------
    aspect_ratio_threshold : float
        w/h ratio above which a bounding box looks 'horizontal'.
    consecutive_frames : int
        How many consecutive suspicious frames trigger an alert.
    sudden_drop_pixels : int
        Downward centroid shift (pixels) that is treated as an
        instantaneous fall indicator (optional supplementary check).
    frame_height : int
        Total video frame height — used to decide whether the box is
        near the ground.  Updated dynamically on each call.
    """

    def __init__(
        self,
        aspect_ratio_threshold: float = 1.2,
        consecutive_frames: int = 8,
        sudden_drop_pixels: int = 40,
    ) -> None:
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.consecutive_frames = consecutive_frames
        self.sudden_drop_pixels = sudden_drop_pixels

        # suspicious_count[track_id] = consecutive frame count
        self._suspicious_count: Dict[int, int] = defaultdict(int)
        # alerted[track_id] = True once an alert has been raised (avoids spam)
        self._alerted: Dict[int, bool] = defaultdict(bool)
        # prev_centroid_y[track_id] = last cy value
        self._prev_cy: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        tracks: List[Dict[str, Any]],
        frame_height: int,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate each person track for fall criteria.

        Parameters
        ----------
        tracks : list of tracked person dicts (output of ObjectTracker.update)
        frame_height : int  — height of the current frame in pixels

        Returns
        -------
        List of dicts for tracks where a fall alert is active:
            track_id, bbox, centroid, alert_type='fall'
        """
        person_tracks = [t for t in tracks if t["class_id"] == 0]
        active_ids = {t["track_id"] for t in person_tracks}

        # Reset counters for IDs that are no longer visible
        for tid in list(self._suspicious_count.keys()):
            if tid not in active_ids:
                self._suspicious_count[tid] = 0
                self._alerted[tid] = False

        alerts = []

        for track in person_tracks:
            tid = track["track_id"]
            x1, y1, x2, y2 = track["bbox"]
            cx, cy = track["centroid"]

            width = x2 - x1
            height = max(y2 - y1, 1)  # avoid division by zero
            aspect_ratio = width / height

            # Check for sudden downward drop
            sudden_drop = False
            if tid in self._prev_cy:
                drop = cy - self._prev_cy[tid]
                if drop > self.sudden_drop_pixels:
                    sudden_drop = True

            self._prev_cy[tid] = cy

            # Check aspect-ratio criterion
            is_horizontal = aspect_ratio >= self.aspect_ratio_threshold

            # Check that the person is actually near the ground
            # (bottom of box is in the lower 70 % of the frame)
            near_ground = y2 > (0.30 * frame_height)

            if (is_horizontal and near_ground) or sudden_drop:
                self._suspicious_count[tid] += 1
            else:
                # Reset — person recovered or was not suspicious
                self._suspicious_count[tid] = 0
                self._alerted[tid] = False

            if (
                self._suspicious_count[tid] >= self.consecutive_frames
                and not self._alerted[tid]
            ):
                self._alerted[tid] = True
                logger.warning(
                    f"[FALL ALERT] Track {tid} | AR={aspect_ratio:.2f} | "
                    f"consecutive={self._suspicious_count[tid]}"
                )
                alerts.append(
                    {
                        "track_id": tid,
                        "bbox": track["bbox"],
                        "centroid": track["centroid"],
                        "alert_type": "fall",
                        "detail": f"AR={aspect_ratio:.2f}, "
                                  f"frames={self._suspicious_count[tid]}",
                    }
                )
            elif self._alerted[tid]:
                # Keep re-reporting while the condition persists
                alerts.append(
                    {
                        "track_id": tid,
                        "bbox": track["bbox"],
                        "centroid": track["centroid"],
                        "alert_type": "fall",
                        "detail": "ongoing",
                    }
                )

        return alerts

    def reset(self) -> None:
        self._suspicious_count.clear()
        self._alerted.clear()
        self._prev_cy.clear()
