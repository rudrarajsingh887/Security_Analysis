"""
Unit tests for src/fall_detection.py — FallDetector.

These tests use synthetic bounding boxes to verify the aspect-ratio +
ground-proximity heuristic without requiring YOLO, a video file, or
network access.
"""

from src.fall_detection import FallDetector
from tests.helpers import make_track

FRAME_HEIGHT = 720


def test_standing_person_never_alerts():
    """A tall, narrow box (AR < threshold) should never trigger a fall alert."""
    detector = FallDetector(
        aspect_ratio_threshold=1.2, consecutive_frames=3, sudden_drop_pixels=40
    )
    standing_box = [100, 50, 160, 350]  # width=60, height=300 -> AR=0.2

    alerts = []
    for _ in range(10):
        track = make_track(track_id=1, bbox=standing_box, class_id=0)
        alerts = detector.update([track], FRAME_HEIGHT)

    assert alerts == []


def test_fallen_person_triggers_alert_after_consecutive_frames():
    """A wide, short box near the ground should alert only after N consecutive frames."""
    detector = FallDetector(
        aspect_ratio_threshold=1.2, consecutive_frames=3, sudden_drop_pixels=40
    )
    fallen_box = [100, 600, 300, 680]  # width=200, height=80 -> AR=2.5, near ground

    alerts = []
    for frame_idx in range(5):
        track = make_track(track_id=1, bbox=fallen_box, class_id=0)
        alerts = detector.update([track], FRAME_HEIGHT)
        if frame_idx < 2:
            assert alerts == [], "Should not alert before consecutive_frames is reached"

    assert len(alerts) == 1
    assert alerts[0]["alert_type"] == "fall"
    assert alerts[0]["track_id"] == 1


def test_wide_box_high_in_frame_does_not_alert():
    """A wide box near the TOP of the frame (e.g. distant crouching) should not alert."""
    detector = FallDetector(
        aspect_ratio_threshold=1.2, consecutive_frames=3, sudden_drop_pixels=40
    )
    # width=120, height=60 -> AR=2.0 (looks "fallen"), but y2=80 is high in frame
    high_wide_box = [100, 20, 220, 80]

    alerts = []
    for _ in range(10):
        track = make_track(track_id=2, bbox=high_wide_box, class_id=0)
        alerts = detector.update([track], FRAME_HEIGHT)

    assert alerts == []


def test_recovery_clears_active_alert():
    """If the person stands back up, the alert should stop being reported."""
    detector = FallDetector(
        aspect_ratio_threshold=1.2, consecutive_frames=2, sudden_drop_pixels=40
    )
    fallen_box = [100, 600, 300, 680]
    standing_box = [150, 300, 210, 600]  # AR = 60/300 = 0.2

    # Trigger the fall alert
    alerts = []
    for _ in range(3):
        track = make_track(track_id=1, bbox=fallen_box, class_id=0)
        alerts = detector.update([track], FRAME_HEIGHT)
    assert any(a["alert_type"] == "fall" for a in alerts)

    # Person stands back up -> alert should clear
    track = make_track(track_id=1, bbox=standing_box, class_id=0)
    alerts = detector.update([track], FRAME_HEIGHT)
    assert alerts == []


def test_disappearing_track_resets_state():
    """Removing a track from the input should reset its internal counters."""
    detector = FallDetector(
        aspect_ratio_threshold=1.2, consecutive_frames=3, sudden_drop_pixels=40
    )
    fallen_box = [100, 600, 300, 680]

    # Build up 2 suspicious frames (not yet enough to alert)
    for _ in range(2):
        track = make_track(track_id=1, bbox=fallen_box, class_id=0)
        detector.update([track], FRAME_HEIGHT)

    # Track disappears for one frame
    detector.update([], FRAME_HEIGHT)

    # Track reappears - counter should have reset, so 2 more frames
    # should NOT yet trigger an alert (needs 3 consecutive)
    alerts = []
    for _ in range(2):
        track = make_track(track_id=1, bbox=fallen_box, class_id=0)
        alerts = detector.update([track], FRAME_HEIGHT)

    assert alerts == []
