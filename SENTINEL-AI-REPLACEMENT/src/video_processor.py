"""
video_processor.py
------------------
Orchestrates the full frame-by-frame processing pipeline:

  Frame → Tracker → FallDetector → AbandonedObjectDetector
        → CrowdAnalyser → AnomalyDetector → Drawing → Output

Returns structured metrics that can be consumed by app.py (Streamlit)
or main.py (CLI).
"""

from pathlib import Path
from typing import Optional, Dict, Any, Generator

import cv2
import numpy as np
from loguru import logger

from src.tracker import ObjectTracker
from src.fall_detection import FallDetector
from src.abandoned_object import AbandonedObjectDetector
from src.crowd_analysis import CrowdAnalyser
from src.anomaly_detector import AnomalyDetector
from src.utils import (
    draw_tracks, draw_alerts, draw_info_panel,
    draw_restricted_zones, FPSCounter, open_video,
    get_video_properties,
)


class VideoProcessor:
    """
    End-to-end video analysis pipeline.

    Parameters
    ----------
    config : dict
        Parsed config.yaml content.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config

        det_cfg   = config["detection"]
        trk_cfg   = config["tracking"]
        fall_cfg  = config["fall_detection"]
        aban_cfg  = config["abandoned_object"]
        crowd_cfg = config["crowd_analysis"]
        anom_cfg  = config["suspicious_behaviour"]
        zones     = config.get("restricted_zones", {}).get("zones", [])

        # ---- Initialise sub-modules ----
        self.tracker = ObjectTracker(
            model_name=det_cfg["model_name"],
            confidence_threshold=det_cfg["confidence_threshold"],
            iou_threshold=det_cfg["iou_threshold"],
            target_classes=det_cfg["target_classes"],
        )
        self.fall_detector = FallDetector(
            aspect_ratio_threshold=fall_cfg["aspect_ratio_threshold"],
            consecutive_frames=fall_cfg["consecutive_frames"],
            sudden_drop_pixels=fall_cfg["sudden_drop_pixels"],
        )
        self.abandoned_detector = AbandonedObjectDetector(
            stationary_seconds=aban_cfg["stationary_seconds"],
            proximity_threshold=aban_cfg["proximity_threshold"],
            min_visible_frames=aban_cfg["min_visible_frames"],
        )
        self.crowd_analyser = CrowdAnalyser(
            crowd_threshold=crowd_cfg["crowd_threshold"],
            alert_frames=crowd_cfg["alert_frames"],
        )
        self.anomaly_detector = AnomalyDetector(
            inactivity_seconds=anom_cfg["inactivity_seconds"],
            stationary_pixel_threshold=anom_cfg["stationary_pixel_threshold"],
            erratic_movement_threshold=anom_cfg["erratic_movement_threshold"],
            erratic_frames_required=anom_cfg["erratic_frames_required"],
            restricted_zones=zones,
        )

        self.fps_counter = FPSCounter()

        # Aggregate session metrics
        self.metrics: Dict[str, Any] = self._init_metrics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_video(
        self,
        input_source,
        output_path: Optional[str] = None,
        show_window: bool = False,
        frame_callback=None,
    ) -> Dict[str, Any]:
        """
        Process a full video file (or webcam stream).

        Parameters
        ----------
        input_source  : str path or int webcam index
        output_path   : where to write the annotated output MP4
        show_window   : display OpenCV preview window (CLI use)
        frame_callback: optional callable(annotated_frame, metrics) for
                        streaming progress updates (Streamlit use)

        Returns
        -------
        Final session metrics dict.
        """
        self._reset_all()
        self.metrics = self._init_metrics()

        try:
            cap = open_video(input_source)
        except (FileNotFoundError, IOError) as exc:
            logger.error(str(exc))
            raise

        props = get_video_properties(cap)
        fps = props["fps"]
        width = props["width"]
        height = props["height"]
        total_frames = props["frame_count"]

        logger.info(
            f"Processing: {input_source} | "
            f"{width}×{height} @ {fps:.1f} FPS | "
            f"{total_frames} frames"
        )

        # Update fps-dependent sub-modules
        self.abandoned_detector.fps = fps
        self.anomaly_detector.fps = fps

        # ---- Output writer ----
        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"Writing output to: {output_path}")

        cfg_vid = self.cfg.get("video", {})
        target_w = cfg_vid.get("resize_width")
        target_h = cfg_vid.get("resize_height")

        zones = self.cfg.get("restricted_zones", {}).get("zones", [])
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Optional resize for speed
                if target_w and target_h:
                    frame = cv2.resize(frame, (target_w, target_h))
                    height, width = frame.shape[:2]

                # ---- Pipeline ----
                tracks = self.tracker.update(frame)

                fall_alerts  = self.fall_detector.update(tracks, height)
                aban_alerts  = self.abandoned_detector.update(tracks, frame_idx)
                crowd_info   = self.crowd_analyser.update(tracks)
                anom_alerts  = self.anomaly_detector.update(tracks, frame_idx)

                all_alerts = fall_alerts + aban_alerts + anom_alerts
                if crowd_info["alert"]:
                    all_alerts.append({
                        "bbox": [0, 0, 1, 1],
                        "alert_type": "crowd_density",
                        "centroid": (width // 2, height // 2),
                    })

                # ---- Update metrics ----
                alert_ids = {a["track_id"] for a in all_alerts if "track_id" in a}
                self._update_metrics(fall_alerts, aban_alerts, crowd_info, anom_alerts)

                # ---- Drawing ----
                annotated = frame.copy()
                if zones:
                    annotated = draw_restricted_zones(annotated, zones)
                annotated = draw_tracks(annotated, tracks, alert_ids)
                annotated = draw_alerts(annotated, all_alerts)

                display_fps = self.fps_counter.tick()
                annotated = draw_info_panel(
                    annotated, display_fps,
                    crowd_info["raw_count"],
                    all_alerts, crowd_info
                )

                if writer:
                    writer.write(annotated)

                if show_window:
                    cv2.imshow("AI Surveillance", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("User quit — stopping.")
                        break

                if frame_callback:
                    frame_callback(annotated, dict(self.metrics))

                frame_idx += 1

        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

        self.metrics["total_frames_processed"] = frame_idx
        logger.success(f"Done. {frame_idx} frames processed.")
        return dict(self.metrics)

    def process_frame_stream(
        self,
        input_source,
    ) -> Generator:
        """
        Generator that yields (annotated_frame, current_metrics) per frame.
        Useful for Streamlit st.image() live preview without writing a file.
        """
        self._reset_all()
        self.metrics = self._init_metrics()

        try:
            cap = open_video(input_source)
        except (FileNotFoundError, IOError) as exc:
            logger.error(str(exc))
            raise

        props = get_video_properties(cap)
        fps = props["fps"]
        height = props["height"]
        self.abandoned_detector.fps = fps
        self.anomaly_detector.fps = fps

        zones = self.cfg.get("restricted_zones", {}).get("zones", [])
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                cfg_vid = self.cfg.get("video", {})
                target_w = cfg_vid.get("resize_width")
                target_h = cfg_vid.get("resize_height")
                if target_w and target_h:
                    frame = cv2.resize(frame, (target_w, target_h))
                    height = frame.shape[0]

                tracks      = self.tracker.update(frame)
                fall_alerts = self.fall_detector.update(tracks, height)
                aban_alerts = self.abandoned_detector.update(tracks, frame_idx)
                crowd_info  = self.crowd_analyser.update(tracks)
                anom_alerts = self.anomaly_detector.update(tracks, frame_idx)

                all_alerts = fall_alerts + aban_alerts + anom_alerts
                alert_ids  = {a["track_id"] for a in all_alerts if "track_id" in a}

                self._update_metrics(fall_alerts, aban_alerts, crowd_info, anom_alerts)

                annotated = frame.copy()
                if zones:
                    annotated = draw_restricted_zones(annotated, zones)
                annotated = draw_tracks(annotated, tracks, alert_ids)
                annotated = draw_alerts(annotated, all_alerts)
                annotated = draw_info_panel(
                    annotated, self.fps_counter.tick(),
                    crowd_info["raw_count"], all_alerts, crowd_info
                )

                frame_idx += 1
                self.metrics["total_frames_processed"] = frame_idx

                yield annotated, dict(self.metrics)

        finally:
            cap.release()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset_all(self) -> None:
        self.tracker.reset()
        self.fall_detector.reset()
        self.abandoned_detector.reset()
        self.crowd_analyser.reset()
        self.anomaly_detector.reset()

    @staticmethod
    def _init_metrics() -> Dict[str, Any]:
        return {
            "total_frames_processed": 0,
            "total_fall_alerts": 0,
            "total_abandoned_alerts": 0,
            "total_crowd_alerts": 0,
            "total_inactivity_alerts": 0,
            "total_erratic_alerts": 0,
            "total_zone_alerts": 0,
            "peak_person_count": 0,
        }

    def _update_metrics(self, fall, aban, crowd, anom) -> None:
        m = self.metrics
        m["total_fall_alerts"] += len(fall)
        m["total_abandoned_alerts"] += len(aban)
        if crowd.get("alert"):
            m["total_crowd_alerts"] += 1
        m["peak_person_count"] = max(
            m["peak_person_count"], crowd["raw_count"]
        )
        for a in anom:
            t = a.get("alert_type", "")
            if t == "suspicious_inactivity":
                m["total_inactivity_alerts"] += 1
            elif t == "erratic_movement":
                m["total_erratic_alerts"] += 1
            elif t == "zone_breach":
                m["total_zone_alerts"] += 1
