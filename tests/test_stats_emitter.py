from unittest.mock import patch

from frigate.application.stats.emitter import StatsEmitter


def test_startup_stats_are_live_until_periodic_history_exists():
    emitter = object.__new__(StatsEmitter)
    emitter.config = object()
    emitter.stats_tracking = object()
    emitter.hwaccel_errors = {}
    emitter.stats_history = []

    with patch(
        "frigate.application.stats.emitter.stats_snapshot",
        side_effect=[{"sequence": 1}, {"sequence": 2}],
    ) as snapshot:
        assert emitter.get_latest_stats() == {"sequence": 1}
        assert emitter.get_latest_stats() == {"sequence": 2}

    assert snapshot.call_count == 2
    assert emitter.stats_history == []

    emitter.stats_history.append({"sequence": 3})
    with patch("frigate.application.stats.emitter.stats_snapshot") as snapshot:
        assert emitter.get_latest_stats() == {"sequence": 3}
        snapshot.assert_not_called()
