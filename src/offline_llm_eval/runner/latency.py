import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

MILLISECONDS_PER_SECOND: Final = 1000

type Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class LatencyMetrics:
    latency_ms: int
    time_to_first_event_ms: int | None = None
    time_to_first_answer_delta_ms: int | None = None


@dataclass(frozen=True, slots=True)
class TimedResult[T]:
    value: T
    metrics: LatencyMetrics


class LatencyTracker:
    def __init__(self, clock: Clock = time.perf_counter) -> None:
        self._clock = clock
        self._started_at = clock()
        self._first_event_at: float | None = None
        self._first_answer_at: float | None = None

    def mark_first_event(self) -> None:
        if self._first_event_at is None:
            self._first_event_at = self._clock()

    def mark_first_answer(self) -> None:
        if self._first_answer_at is None:
            self._first_answer_at = self._clock()

    def finish(self) -> LatencyMetrics:
        return self.snapshot(finished_at=self._clock())

    def snapshot(self, *, finished_at: float | None = None) -> LatencyMetrics:
        end_at = finished_at if finished_at is not None else self._clock()
        return LatencyMetrics(
            latency_ms=_elapsed_ms(self._started_at, end_at),
            time_to_first_event_ms=_time_to_first_event_ms(
                started_at=self._started_at,
                first_event_at=self._first_event_at,
            ),
            time_to_first_answer_delta_ms=_time_to_first_answer_delta_ms(
                first_event_at=self._first_event_at,
                first_answer_at=self._first_answer_at,
            ),
        )


def measure_latency[T](
    operation: Callable[[], T],
    *,
    clock: Clock = time.perf_counter,
) -> TimedResult[T]:
    tracker = LatencyTracker(clock=clock)
    value = operation()
    return TimedResult(value=value, metrics=tracker.finish())


def _time_to_first_event_ms(
    *,
    started_at: float,
    first_event_at: float | None,
) -> int | None:
    if first_event_at is None:
        return None
    return _elapsed_ms(started_at, first_event_at)


def _time_to_first_answer_delta_ms(
    *,
    first_event_at: float | None,
    first_answer_at: float | None,
) -> int | None:
    if first_event_at is None or first_answer_at is None:
        return None
    return _elapsed_ms(first_event_at, first_answer_at)


def _elapsed_ms(started_at: float, finished_at: float) -> int:
    elapsed_seconds = max(0.0, finished_at - started_at)
    return round(elapsed_seconds * MILLISECONDS_PER_SECOND)
