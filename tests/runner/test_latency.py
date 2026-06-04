from offline_llm_eval.runner.latency import (
    LatencyMetrics,
    LatencyTracker,
    measure_latency,
)


class FakeClock:
    def __init__(self, *ticks: float) -> None:
        self._ticks = list(ticks)

    def __call__(self) -> float:
        return self._ticks.pop(0)


def test_measure_latency_同期処理のlatency_msを返す() -> None:
    result = measure_latency(lambda: "ok", clock=FakeClock(10.0, 10.123))

    assert result.value == "ok"
    assert result.metrics == LatencyMetrics(
        latency_ms=123,
        time_to_first_event_ms=None,
        time_to_first_answer_delta_ms=None,
    )


def test_latency_tracker_first_eventとfirst_answer_deltaを記録する() -> None:
    tracker = LatencyTracker(clock=FakeClock(1.0, 1.02, 1.08, 1.2))

    tracker.mark_first_event()
    tracker.mark_first_answer()
    metrics = tracker.finish()

    assert metrics == LatencyMetrics(
        latency_ms=200,
        time_to_first_event_ms=20,
        time_to_first_answer_delta_ms=60,
    )


def test_latency_tracker_first_markerは上書きしない() -> None:
    tracker = LatencyTracker(clock=FakeClock(1.0, 1.01, 1.03, 1.05))

    tracker.mark_first_event()
    tracker.mark_first_event()
    tracker.mark_first_answer()
    tracker.mark_first_answer()
    metrics = tracker.finish()

    assert metrics.time_to_first_event_ms == 10
    assert metrics.time_to_first_answer_delta_ms == 20
    assert metrics.latency_ms == 50


def test_latency_tracker_eventなしならanswer_deltaはnullになる() -> None:
    tracker = LatencyTracker(clock=FakeClock(1.0, 1.05, 1.2))

    tracker.mark_first_answer()
    metrics = tracker.finish()

    assert metrics.time_to_first_event_ms is None
    assert metrics.time_to_first_answer_delta_ms is None
    assert metrics.latency_ms == 200


def test_latency_tracker_clockが戻っても負数を返さない() -> None:
    tracker = LatencyTracker(clock=FakeClock(2.0, 1.9))

    metrics = tracker.finish()

    assert metrics.latency_ms == 0
