"""Pack rewind provider: JSON-safe checkpoint, atomic restore and marks."""

from __future__ import annotations

import json

import pytest

from unigym_bench.engine import REWIND_PROVIDER_NAME, REWIND_SCHEMA, StreamingEngine
from unigym_bench.json import ensure_json_value
from unigym_bench.monitors import ThresholdMonitor
from unigym_bench.sink import BenchmarkSink
from unigym_bench.specs import RetentionSpec
from _support import make_bundle, table_probe


def _sink(
    rows: list[list[float]],
    *,
    name: str = "contact",
    threshold: float = 0.5,
    terminal_outcome: str | None = "success",
    retention: RetentionSpec | None = None,
) -> tuple[BenchmarkSink, StreamingEngine, ThresholdMonitor]:
    monitor = ThresholdMonitor(
        name,
        table_probe(rows),
        threshold=threshold,
        terminal_outcome=terminal_outcome,
    )
    engine = StreamingEngine(
        [monitor],
        retention=RetentionSpec(retain_episodes=True) if retention is None else retention,
    )
    return BenchmarkSink(engine=engine), engine, monitor


def _step(sink: BenchmarkSink, t: int) -> dict[str, object]:
    bundle = make_bundle(num_envs=1, t=t)
    sink.on_step(t, bundle)
    return bundle


def _due(sink: BenchmarkSink, t: int) -> None:
    bundle = _step(sink, t)
    sink.drain_terminations()
    sink.on_episode_end([0], bundle)
    sink.on_step_end(t, bundle)


def test_provider_identity_and_payload_is_json_safe_and_private() -> None:
    sink, engine, _ = _sink([[0.0]])
    _step(sink, 0)

    provider = sink.rewind_provider
    assert provider is engine
    assert provider.name == REWIND_PROVIDER_NAME == "unigym_bench"
    state = provider.rewind_state()

    assert state["schema"] == REWIND_SCHEMA == "unigym_bench.rewind/1"
    assert set(state) == {"schema", "engine", "monitors"}
    assert ensure_json_value(state) == state
    assert json.loads(json.dumps(state)) == state

    provider.validate_rewind_state(state)
    assert provider.rewind_state() == state

    state["engine"]["open"].append(False)
    assert provider.rewind_state()["engine"]["open"] == [True]


def test_roundtrip_after_a_due_episode_restores_seq_ticks_journal_and_monitor() -> None:
    sink, engine, monitor = _sink([[0.0], [1.0], [0.0], [1.0]])
    _step(sink, 0)
    _due(sink, 1)
    assert engine.open_envs == (0,)
    snapshot = sink.rewind_provider.rewind_state()

    _step(sink, 2)
    _due(sink, 3)
    assert engine.summary.episodes == 2
    assert [record.episode_seq for record in engine.records] == [0, 1]

    sink.rewind_provider.apply_rewind_state(snapshot)

    assert engine.open_envs == (0,)
    assert len(engine.journal) == 1
    assert engine.summary.episodes == 1
    assert engine.summary.total_ticks == 2
    assert [record.episode_seq for record in engine.records] == [0]
    assert monitor.report_view()["trigger_events"] == 1

    _step(sink, 2)
    _due(sink, 3)

    assert [record.episode_seq for record in engine.records] == [0, 1]
    assert engine.records[1].ticks == 2
    assert len(engine.journal) == 2
    assert monitor.report_view()["trigger_events"] == 2


def test_pending_terminal_is_snapshotted_restored_and_drained_once() -> None:
    sink, engine, _ = _sink([[1.0]])
    _step(sink, 0)
    snapshot = sink.rewind_provider.rewind_state()
    assert set(snapshot["engine"]["pending"]) == {"0"}

    drained = sink.drain_terminations()
    assert set(drained) == {0}
    assert sink.drain_terminations() == {}

    sink.rewind_provider.apply_rewind_state(snapshot)

    restored = sink.drain_terminations()
    assert set(restored) == {0}
    assert restored[0].code == "custom.contact.threshold"
    assert sink.drain_terminations() == {}


def test_first_terminal_latch_event_counter_and_journal_length_are_restored() -> None:
    sink, engine, _ = _sink([[1.0]])
    _step(sink, 0)
    snapshot = sink.rewind_provider.rewind_state()
    assert snapshot["engine"]["latched"]["0"]["code"] == "custom.contact.threshold"

    sink.drain_terminations()
    sink.rewind_provider.apply_rewind_state(snapshot)

    assert engine.summary.event_count == 1
    assert len(engine.journal) == 1
    assert engine.journal[0].code == "task.contact.threshold_reached"
    assert engine.conflicts == {}
    assert set(sink.drain_terminations()) == {0}

    bundle = make_bundle(num_envs=1, t=0)
    sink.on_episode_end([0], bundle)
    record = engine.records[0]
    assert record.first_terminal is not None
    assert record.first_terminal.code == "custom.contact.threshold"
    assert record.verdict == "terminal"


def test_dropped_event_counter_and_capped_journal_are_restored() -> None:
    sink, engine, _ = _sink(
        [[1.0], [1.0], [1.0]],
        retention=RetentionSpec(retain_episodes=True, max_events=1),
    )
    _due(sink, 0)
    _step(sink, 1)
    snapshot = sink.rewind_provider.rewind_state()
    assert len(snapshot["engine"]["journal"]) == 1
    assert snapshot["engine"]["dropped_events"] == 1

    sink.drain_terminations()
    sink.on_episode_end([0], make_bundle(num_envs=1, t=1))
    sink.on_step_end(1, make_bundle(num_envs=1, t=1))
    _step(sink, 2)
    assert engine.dropped_events == 2

    sink.rewind_provider.apply_rewind_state(snapshot)

    assert engine.dropped_events == 1
    assert len(engine.journal) == 1
    assert engine.summary.event_count == 2


def test_monitor_snapshot_config_mismatch_is_rejected_atomically() -> None:
    source = StreamingEngine(
        [
            ThresholdMonitor("contact", table_probe([[0.0]]), threshold=0.5),
            ThresholdMonitor("stable", table_probe([[0.0]]), threshold=0.5),
        ],
        retention=RetentionSpec(retain_episodes=True),
    )
    source_sink = BenchmarkSink(engine=source)
    source_sink.on_step(0, make_bundle(num_envs=1, t=0))
    snapshot = source_sink.rewind_provider.rewind_state()

    stable = ThresholdMonitor("stable", table_probe([[1.0]]), threshold=0.5)
    target = StreamingEngine(
        [
            ThresholdMonitor("contact", table_probe([[1.0]]), threshold=0.9),
            stable,
        ],
        retention=RetentionSpec(retain_episodes=True),
    )
    target_sink = BenchmarkSink(engine=target)
    target_sink.on_step(0, make_bundle(num_envs=1, t=0))
    assert stable.report_view()["trigger_events"] == 1
    assert len(target.journal) == 2

    with pytest.raises(ValueError, match="config does not match"):
        target_sink.rewind_provider.validate_rewind_state(snapshot)
    with pytest.raises(ValueError, match="config does not match"):
        target_sink.rewind_provider.apply_rewind_state(snapshot)

    assert stable.report_view()["trigger_events"] == 1
    assert len(target.journal) == 2
    assert target.records == ()
    assert target.open_envs == (0,)
    assert target.summary.event_count == 2


def test_rewind_after_due_episode_marks_replayed_open_episode_rewound() -> None:
    sink, engine, _ = _sink([[1.0], [0.0]])
    _due(sink, 0)
    snapshot = sink.rewind_provider.rewind_state()

    _step(sink, 1)
    sink.rewind_provider.apply_rewind_state(snapshot)
    sink.on_rewind({"tick": 2, "restored_tick": 1, "snap_id": 7})
    _step(sink, 1)
    summary = sink.on_run_end()

    assert [record.verdict for record in engine.records] == ["terminal", "rewound"]
    assert [record.rewound for record in engine.records] == [False, True]
    assert engine.records[1].episode_seq == 1
    assert engine.records[1].ticks == 1
    assert summary["BenchmarkSink"]["episodes"] == 2


def test_payload_never_carries_sampler_or_signal_bus_state() -> None:
    sink, _, _ = _sink([[1.0]])
    _step(sink, 0)

    state = sink.rewind_provider.rewind_state()

    forbidden = {"sampler", "signals", "signal_bus", "bus", "episode_id"}
    seen: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                seen.add(key)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(state)
    assert seen.isdisjoint(forbidden)
