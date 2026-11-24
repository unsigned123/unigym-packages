"""P9 retention: the event journal is bounded and losses are reported.

The engine never retains a full episode history by default.  This module pins
both halves of that claim: the in-memory journal stops at ``max_events`` and
every dropped occurrence is still counted in the run report.
"""

from __future__ import annotations

from unigym_bench.engine import StreamingEngine
from unigym_bench.events import Event
from unigym_bench.sink import BenchmarkSink
from unigym_bench.specs import RetentionSpec
from _support import ScriptedMonitor, TickScript, make_bundle


def _event(name: str = "tick") -> Event:
    return Event(name=name, code=f"task.record.{name}", env_ids=(0,))


def _sink_with_events(
    *, ticks: int, retention: RetentionSpec
) -> tuple[BenchmarkSink, StreamingEngine]:
    monitor = ScriptedMonitor(
        "task",
        {t: TickScript(events=(_event("a"), _event("b"))) for t in range(ticks)},
    )
    engine = StreamingEngine([monitor], retention=retention)
    return BenchmarkSink(engine=engine), engine


def test_event_journal_is_bounded_and_dropped_events_are_reported() -> None:
    """A cap of 7 keeps 7 of 40 occurrences and reports the other 33."""
    sink, engine = _sink_with_events(
        ticks=20, retention=RetentionSpec(max_events=7, retain_episodes=False)
    )

    for t in range(20):
        sink.on_step(t, make_bundle(t=t))

    assert len(engine.journal) == 7
    assert engine.dropped_events == 33
    assert engine.summary.event_count == 40
    assert engine.records == ()

    report = sink.on_run_end()["BenchmarkSink"]
    assert isinstance(report, dict)
    assert report["execution"] == {
        "event_total": 40,
        "dropped_events": 33,
        "terminal_conflicts": 0,
        "retention": {
            "retain_episodes": False,
            "max_events": 7,
            "event_log": True,
        },
    }
    # Default retention keeps counters, not per-record or per-event history.
    assert report["episodes"]["retained"] == 0
    assert "by_env" not in report["episodes"]
    assert "length" not in report["episodes"]


def test_disabled_event_log_drops_every_occurrence_and_reports_it() -> None:
    """``event_log: false`` stores no journal entries but counts the losses."""
    sink, engine = _sink_with_events(
        ticks=3, retention=RetentionSpec(max_events=100, event_log=False)
    )

    for t in range(3):
        sink.on_step(t, make_bundle(t=t))

    assert engine.journal == ()
    assert engine.dropped_events == 6
    assert engine.summary.event_count == 6

    report = sink.on_run_end()["BenchmarkSink"]
    assert isinstance(report, dict)
    assert report["execution"]["event_total"] == 6
    assert report["execution"]["dropped_events"] == 6
    assert report["execution"]["retention"]["event_log"] is False
