"""
Unit tests for src/abandoned_object.py — AbandonedObjectDetector.

Verifies the stationarity-timer + person-proximity heuristic using
synthetic bag and person tracks across multiple simulated frames.
"""

from src.abandoned_object import AbandonedObjectDetector
from tests.helpers import make_track


def test_stationary_unattended_bag_triggers_alert():
    """A bag that doesn't move and has no nearby person should be flagged."""
    detector = AbandonedObjectDetector(
        stationary_seconds=1.0,
        proximity_threshold=50,
        stationary_pixel_threshold=5,
        min_visible_frames=2,
        fps=10.0,  # 1 second == 10 frames
    )
    bag_box = [400, 400, 440, 440]  # centroid = (420, 420), fixed

    alerts = []
    for frame_idx in range(15):
        bag = make_track(track_id=10, bbox=bag_box, class_id=28)  # suitcase
        alerts = detector.update([bag], frame_idx)

    assert len(alerts) == 1
    assert alerts[0]["alert_type"] == "abandoned_object"
    assert alerts[0]["track_id"] == 10
    assert alerts[0]["label"] == "suitcase"


def test_bag_with_nearby_person_never_alerts():
    """If a person stays close to the bag, it should never be 'abandoned'."""
    detector = AbandonedObjectDetector(
        stationary_seconds=1.0,
        proximity_threshold=100,
        stationary_pixel_threshold=5,
        min_visible_frames=2,
        fps=10.0,
    )
    bag_box = [400, 400, 440, 440]      # centroid = (420, 420)
    person_box = [410, 410, 450, 600]   # centroid = (430, 505), distance ~85.6

    alerts = []
    for frame_idx in range(15):
        bag = make_track(track_id=10, bbox=bag_box, class_id=28)
        person = make_track(track_id=1, bbox=person_box, class_id=0)
        alerts = detector.update([bag, person], frame_idx)

    assert alerts == []


def test_moving_bag_never_alerts():
    """A bag that keeps moving every frame should never be 'stationary'."""
    detector = AbandonedObjectDetector(
        stationary_seconds=1.0,
        proximity_threshold=50,
        stationary_pixel_threshold=5,
        min_visible_frames=2,
        fps=10.0,
    )

    alerts = []
    for frame_idx in range(15):
        # Move the bag 20px to the right every frame -> always "moving"
        x_offset = frame_idx * 20
        bag_box = [400 + x_offset, 400, 440 + x_offset, 440]
        bag = make_track(track_id=10, bbox=bag_box, class_id=28)
        alerts = detector.update([bag], frame_idx)

    assert alerts == []


def test_alert_clears_once_person_arrives():
    """An active alert should stop once a person approaches the object."""
    detector = AbandonedObjectDetector(
        stationary_seconds=1.0,
        proximity_threshold=60,
        stationary_pixel_threshold=5,
        min_visible_frames=2,
        fps=10.0,
    )
    bag_box = [400, 400, 440, 440]  # centroid = (420, 420)

    # Phase 1: bag alone -> should alert
    alerts = []
    for frame_idx in range(15):
        bag = make_track(track_id=10, bbox=bag_box, class_id=28)
        alerts = detector.update([bag], frame_idx)
    assert len(alerts) == 1

    # Phase 2: a person arrives right next to the bag (centroid distance = 30)
    person_box = [400, 400, 440, 500]
    bag = make_track(track_id=10, bbox=bag_box, class_id=28)
    person = make_track(track_id=1, bbox=person_box, class_id=0)
    alerts = detector.update([bag, person], 15)

    assert alerts == []


def test_object_removed_from_tracking_clears_state():
    """Once a tracked object disappears, its record should be discarded."""
    detector = AbandonedObjectDetector(
        stationary_seconds=1.0,
        proximity_threshold=50,
        stationary_pixel_threshold=5,
        min_visible_frames=2,
        fps=10.0,
    )
    bag_box = [400, 400, 440, 440]

    for frame_idx in range(15):
        bag = make_track(track_id=10, bbox=bag_box, class_id=28)
        detector.update([bag], frame_idx)

    assert 10 in detector._objects  # internal record exists while tracked

    # Object picked up / leaves the scene
    detector.update([], 15)

    assert 10 not in detector._objects
