"""
Unit tests for src/anomaly_detector.py — AnomalyDetector.

Covers suspicious inactivity, erratic movement, and restricted-zone
breach detection, plus the point-in-polygon helper.
"""

import numpy as np

from src.anomaly_detector import AnomalyDetector
from tests.helpers import make_track

ZONES = [{"name": "Restricted Area", "points": [[0, 0], [100, 0], [100, 100], [0, 100]]}]


def test_stationary_person_triggers_inactivity_alert():
    detector = AnomalyDetector(
        inactivity_seconds=1.0,
        stationary_pixel_threshold=5,
        erratic_movement_threshold=1000,  # disable erratic checks for this test
        erratic_frames_required=100,
        fps=10.0,
    )
    box = [500, 500, 540, 600]  # centroid = (520, 550), held fixed

    alerts = []
    for frame_idx in range(15):
        track = make_track(track_id=1, bbox=box, class_id=0)
        alerts = detector.update([track], frame_idx)

    assert any(a["alert_type"] == "suspicious_inactivity" for a in alerts)


def test_moving_person_never_triggers_inactivity():
    detector = AnomalyDetector(
        inactivity_seconds=1.0,
        stationary_pixel_threshold=5,
        erratic_movement_threshold=1000,
        erratic_frames_required=100,
        fps=10.0,
    )

    alerts = []
    for frame_idx in range(15):
        x = 500 + frame_idx * 10  # 10px/frame, above stationary threshold
        box = [x, 500, x + 40, 600]
        track = make_track(track_id=1, bbox=box, class_id=0)
        alerts = detector.update([track], frame_idx)

    assert not any(a["alert_type"] == "suspicious_inactivity" for a in alerts)


def test_erratic_movement_detected():
    detector = AnomalyDetector(
        inactivity_seconds=1000,  # disable inactivity for this test
        stationary_pixel_threshold=5,
        erratic_movement_threshold=50,
        erratic_frames_required=2,
        fps=10.0,
    )

    alerts = []
    positions = [500, 700, 500, 700, 500, 700]  # large jumps every frame
    for frame_idx, x in enumerate(positions):
        box = [x, 500, x + 40, 600]
        track = make_track(track_id=2, bbox=box, class_id=0)
        alerts = detector.update([track], frame_idx)

    assert any(a["alert_type"] == "erratic_movement" for a in alerts)


def test_small_movements_do_not_trigger_erratic_alert():
    detector = AnomalyDetector(
        inactivity_seconds=1000,
        stationary_pixel_threshold=2,
        erratic_movement_threshold=50,
        erratic_frames_required=2,
        fps=10.0,
    )

    alerts = []
    for frame_idx in range(10):
        x = 500 + frame_idx * 10  # 10px/frame -> below erratic threshold of 50
        box = [x, 500, x + 40, 600]
        track = make_track(track_id=2, bbox=box, class_id=0)
        alerts = detector.update([track], frame_idx)

    assert not any(a["alert_type"] == "erratic_movement" for a in alerts)


def test_restricted_zone_breach():
    detector = AnomalyDetector(
        inactivity_seconds=1000,
        stationary_pixel_threshold=5,
        erratic_movement_threshold=1000,
        erratic_frames_required=100,
        fps=10.0,
        restricted_zones=ZONES,
    )

    # First observation outside the restricted zone
    outside_box = [110, 40, 130, 60]
    outside_track = make_track(
        track_id=3,
        bbox=outside_box,
        class_id=0,
    )

    # Same person enters the restricted zone
    inside_box = [40, 40, 60, 60]
    inside_track = make_track(
        track_id=3,
        bbox=inside_box,
        class_id=0,
    )

    detector.update([outside_track], 0)

    alerts = detector.update([inside_track], 1)

    assert any(
        a["alert_type"] == "zone_breach"
        for a in alerts
    )


def test_outside_zone_does_not_breach():
    detector = AnomalyDetector(
        inactivity_seconds=1000,
        stationary_pixel_threshold=5,
        erratic_movement_threshold=1000,
        erratic_frames_required=100,
        fps=10.0,
        restricted_zones=ZONES,
    )

    outside_box = [480, 480, 520, 520]  # centroid = (500, 500) -> outside zone
    track = make_track(track_id=4, bbox=outside_box, class_id=0)
    alerts = detector.update([track], 0)

    assert not any(a["alert_type"] == "zone_breach" for a in alerts)


def test_point_in_polygon_static_method():
    polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    assert AnomalyDetector._point_in_polygon((50, 50), polygon) is True
    assert AnomalyDetector._point_in_polygon((150, 150), polygon) is False


def test_disappearing_track_clears_state():
    """Removing a track should clean up all internal per-track state."""
    detector = AnomalyDetector(
        inactivity_seconds=1.0,
        stationary_pixel_threshold=5,
        erratic_movement_threshold=1000,
        erratic_frames_required=100,
        fps=10.0,
        restricted_zones=ZONES,
    )
    box = [40, 40, 60, 60]

    track = make_track(track_id=5, bbox=box, class_id=0)
    detector.update([track], 0)
    assert 5 in detector._last_centroid

    detector.update([], 1)  # track disappears

    assert 5 not in detector._last_centroid
    assert 5 not in detector._stationary_since
def test_first_observation_inside_is_presence_not_breach():
    detector = AnomalyDetector(
        inactivity_seconds=1000,
        restricted_zones=ZONES,
    )

    track = make_track(
        track_id=5,
        bbox=[40, 40, 60, 60],
        class_id=0,
    )

    alerts = detector.update([track], 0)

    assert any(
        a["alert_type"] == "zone_presence"
        for a in alerts
    )

    assert not any(
        a["alert_type"] == "zone_breach"
        for a in alerts
    )


def test_continuing_inside_does_not_create_new_breach():
    detector = AnomalyDetector(
        inactivity_seconds=1000,
        restricted_zones=ZONES,
    )

    outside_track = make_track(
        track_id=6,
        bbox=[110, 40, 130, 60],
        class_id=0,
    )

    inside_track = make_track(
        track_id=6,
        bbox=[40, 40, 60, 60],
        class_id=0,
    )

    detector.update([outside_track], 0)

    entry_alerts = detector.update([inside_track], 1)
    ongoing_alerts = detector.update([inside_track], 2)

    assert any(
        a["alert_type"] == "zone_breach"
        for a in entry_alerts
    )

    assert not any(
        a["alert_type"] == "zone_breach"
        for a in ongoing_alerts
    )

    assert any(
        a["alert_type"] == "zone_presence"
        for a in ongoing_alerts
    )