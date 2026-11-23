"""
Unit tests for src/crowd_analysis.py — CrowdAnalyser.

Verifies person counting, the consecutive-frame alert debounce, and
the re-arming behaviour after a crowd disperses.
"""

from src.crowd_analysis import CrowdAnalyser
from tests.helpers import make_track


def _people(n: int) -> list:
    """Build n synthetic person tracks with distinct IDs."""
    return [
        make_track(track_id=i, bbox=[i * 10, 100, i * 10 + 30, 300], class_id=0)
        for i in range(n)
    ]


def test_below_threshold_never_alerts():
    analyser = CrowdAnalyser(crowd_threshold=5, alert_frames=3, smoothing_window=5)
    for _ in range(10):
        result = analyser.update(_people(3))
        assert result["is_crowd"] is False
        assert result["alert"] is False


def test_alert_fires_only_after_consecutive_high_frames():
    analyser = CrowdAnalyser(crowd_threshold=5, alert_frames=3, smoothing_window=5)

    results = [analyser.update(_people(6)) for _ in range(5)]  # all above threshold

    assert results[0]["alert"] is False
    assert results[1]["alert"] is False
    assert results[2]["alert"] is True
    assert results[2]["alert_type"] == "crowd_density"
    # Alert should not repeat every frame while still crowded
    assert results[3]["alert"] is False
    assert results[4]["alert"] is False


def test_alert_rearms_after_crowd_disperses():
    analyser = CrowdAnalyser(crowd_threshold=5, alert_frames=2, smoothing_window=5)

    analyser.update(_people(6))
    second = analyser.update(_people(6))
    assert second["alert"] is True

    dispersed = analyser.update(_people(1))
    assert dispersed["is_crowd"] is False

    analyser.update(_people(6))
    again = analyser.update(_people(6))
    assert again["alert"] is True


def test_peak_count_and_summary():
    analyser = CrowdAnalyser(crowd_threshold=100, alert_frames=3)  # never alerts

    analyser.update(_people(2))
    analyser.update(_people(7))
    analyser.update(_people(4))

    summary = analyser.summary()
    assert summary["peak_count"] == 7
    assert summary["crowd_alerts_raised"] == 0
    assert summary["crowd_threshold"] == 100


def test_smoothed_count_is_rolling_average():
    analyser = CrowdAnalyser(crowd_threshold=100, alert_frames=10, smoothing_window=2)

    first = analyser.update(_people(4))
    assert first["smoothed_count"] == 4.0

    second = analyser.update(_people(8))  # window=2 -> avg(4, 8) = 6.0
    assert second["smoothed_count"] == 6.0


def test_non_person_tracks_are_ignored():
    """Bags/suitcases in the same frame should not count toward crowd size."""
    analyser = CrowdAnalyser(crowd_threshold=5, alert_frames=3, smoothing_window=5)

    tracks = _people(2) + [
        make_track(track_id=99, bbox=[0, 0, 40, 40], class_id=28)  # suitcase
    ]
    result = analyser.update(tracks)
    assert result["raw_count"] == 2
