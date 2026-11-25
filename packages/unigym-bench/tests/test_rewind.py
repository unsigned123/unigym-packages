"""Pack rewind provider: JSON-safe checkpoint, atomic restore and marks."""

from __future__ import annotations

import json

import pytest

from unigym_bench.engine import REWIND_PROVIDER_NAME, REWIND_SCHEMA, StreamingEngine
from unigym_bench.flow import DefaultBenchmarkFlow, EventRule, TerminationRule
from unigym_bench.json import ensure_json_value
from unigym_bench.monitors import PredicateMonitor, threshold_config
from unigym_bench.sink import BenchmarkSink
from unigym_bench.specs import RetentionSpec
from _support import make_bundle, make_values


def _sink(
    *,
    name: str = "contact",
    threshold: float = 0.5,
    terminal_outcome: str | None = "success",
    retention: RetentionSpec | None = None,
) -> tuple[BenchmarkSink, StreamingEngine, PredicateMonitor]:
    monitor = threshold_config(
        name,
        "value:level",
        threshold=threshold,
    )
    rules = (
        ()
        if terminal_outcome is None
        else (
            TerminationRule(
                name,
                "threshold_reached",
                terminal_outcome,
                f"custom.{name}.threshold",
            ),
        )
    )
    flow = DefaultBenchmarkFlow(
        rules,
        events=(EventRule(name, "threshold_reached", "threshold_reached"),),
    )
    engine = StreamingEngine(
        [monitor],
        retention=RetentionSpec(retain_episodes=True) if retention is None else retention,
    )
    return BenchmarkSink(engine=engine, flow=flow), engine, monitor


def _bundle(t: int, row: list[float]) -> dict[str, object]:
    return make_bundle(num_envs=1, t=t, values=make_values(row))


def _step(sink: BenchmarkSink, t: int, row: list[float]) -> dict[str, object]:
    bundle = _bundle(t, row)
    sink.on_step(t, bundle)
    return bundle


def _due(sink: BenchmarkSink, t: int, row: list[float]) -> None:
    bundle = _step(sink, t, row)
    sink.drain_terminations()
    sink.on_episode_end([0], bundle)
    sink.on_reset([0])


def test_provider_identity_and_payload_is_json_safe_and_private() -> None:
    sink, engine, _ = _sink()
    _step(sink, 0, [0.0])

    provider = sink.rewind_provider
    assert provider is engine
    assert provider.name == REWIND_PROVIDER_NAME == "unigym_bench"
    state = provider.rewind_state()

    assert state["schema"] == REWIND_SCHEMA == "unigym_bench.rewind/5"
    assert state["monitors"]["contact"]["schema"] == "unigym_bench.monitor/4"
    assert set(state) == {"schema", "engine", "monitors"}
    assert ensure_json_value(state) == state
    assert json.loads(json.dumps(state)) == state

    provider.validate_rewind_state(state)
    assert provider.rewind_state() == state

    state["engine"]["open"].append(False)
    assert provider.rewind_state()["engine"]["open"] == [True]


def test_roundtrip_after_a_due_episode_restores_seq_ticks_journal_and_monitor() -> None:
    sink, engine, monitor = _sink()
    rows = [[0.0], [1.0], [0.0], [1.0]]
    _step(sink, 0, rows[0])
    _due(sink, 1, rows[1])
    assert engine.open_envs == (0,)
    snapshot = sink.rewind_provider.rewind_state()

    _step(sink, 2, rows[2])
    _due(sink, 3, rows[3])
    assert engine.summary.episodes == 2
    assert [record.episode_seq for record in engine.records] == [0, 1]

    sink.rewind_provider.apply_rewind_state(snapshot)

    assert engine.open_envs == (0,)
    assert len(engine.journal) == 1
    assert engine.summary.episodes == 1
    assert engine.summary.total_ticks == 2
    assert [record.episode_seq for record in engine.records] == [0]
    assert monitor.report_view()["fired"] == 1

    _step(sink, 2, rows[2])
    _due(sink, 3, rows[3])

    assert [record.episode_seq for record in engine.records] == [0, 1]
    assert engine.records[1].ticks == 2
    assert len(engine.journal) == 2
    assert monitor.report_view()["fired"] == 2


def test_pending_terminal_is_snapshotted_restored_and_drained_once() -> None:
    sink, engine, _ = _sink()
    _step(sink, 0, [1.0])
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
    sink, engine, _ = _sink()
    _step(sink, 0, [1.0])
    snapshot = sink.rewind_provider.rewind_state()
    assert snapshot["engine"]["latched"]["0"]["code"] == "custom.contact.threshold"

    sink.drain_terminations()
    sink.rewind_provider.apply_rewind_state(snapshot)

    assert engine.summary.event_count == 1
    assert len(engine.journal) == 1
    assert engine.journal[0].code == "task.contact.threshold_reached"
    assert engine.conflicts == {}
    assert set(sink.drain_terminations()) == {0}

    bundle = _bundle(0, [1.0])
    sink.on_episode_end([0], bundle)
    record = engine.records[0]
    assert record.first_terminal is not None
    assert record.first_terminal.code == "custom.contact.threshold"
    assert record.verdict == "terminal"


def test_dropped_event_counter_and_capped_journal_are_restored() -> None:
    sink, engine, _ = _sink(
        retention=RetentionSpec(retain_episodes=True, max_events=1),
    )
    _due(sink, 0, [1.0])
    _step(sink, 1, [1.0])
    snapshot = sink.rewind_provider.rewind_state()
    assert len(snapshot["engine"]["journal"]) == 1
    assert snapshot["engine"]["dropped_events"] == 1

    sink.drain_terminations()
    sink.on_episode_end([0], _bundle(1, [1.0]))
    sink.on_reset([0])
    _step(sink, 2, [1.0])
    assert engine.dropped_events == 2

    sink.rewind_provider.apply_rewind_state(snapshot)

    assert engine.dropped_events == 1
    assert len(engine.journal) == 1
    assert engine.summary.event_count == 2


def test_monitor_snapshot_config_mismatch_is_rejected_atomically() -> None:
    source = StreamingEngine(
        [
            threshold_config("contact", "value:level", threshold=0.5),
            threshold_config("stable", "value:level", threshold=0.5),
        ],
        retention=RetentionSpec(retain_episodes=True),
    )
    source_sink = BenchmarkSink(engine=source)
    source_sink.on_step(0, _bundle(0, [0.0]))
    snapshot = source_sink.rewind_provider.rewind_state()

    stable = threshold_config("stable", "value:level", threshold=0.5)
    target = StreamingEngine(
        [
            threshold_config("contact", "value:level", threshold=0.9),
            stable,
        ],
        retention=RetentionSpec(retain_episodes=True),
    )
    target_sink = BenchmarkSink(engine=target)
    target_sink.on_step(0, _bundle(0, [1.0]))
    assert stable.report_view()["fired"] == 1
    assert target.journal == ()

    with pytest.raises(ValueError, match="config does not match"):
        target_sink.rewind_provider.validate_rewind_state(snapshot)
    with pytest.raises(ValueError, match="config does not match"):
        target_sink.rewind_provider.apply_rewind_state(snapshot)

    assert stable.report_view()["fired"] == 1
    assert target.journal == ()
    assert target.records == ()
    assert target.open_envs == (0,)
    assert target.summary.event_count == 0


def test_rewind_after_due_episode_marks_replayed_open_episode_rewound() -> None:
    sink, engine, _ = _sink()
    _due(sink, 0, [1.0])
    snapshot = sink.rewind_provider.rewind_state()

    _step(sink, 1, [0.0])
    sink.rewind_provider.apply_rewind_state(snapshot)
    sink.on_rewind({"tick": 2, "restored_tick": 1, "snap_id": 7})
    _step(sink, 1, [0.0])
    summary = sink.on_run_end()

    assert [record.verdict for record in engine.records] == ["terminal", "rewound"]
    assert [record.rewound for record in engine.records] == [False, True]
    assert engine.records[1].episode_seq == 1
    assert engine.records[1].ticks == 1
    assert summary["BenchmarkSink"]["episodes"]["count"] == 2


def test_payload_never_carries_sampler_or_core_value_state() -> None:
    sink, _, _ = _sink()
    _step(sink, 0, [1.0])

    state = sink.rewind_provider.rewind_state()

    forbidden = {"sampler", "signals", "signal_bus", "bus", "values", "episode_id"}
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


def test_old_rewind_and_monitor_schemas_are_rejected() -> None:
    sink, engine, _ = _sink()
    _step(sink, 0, [0.0])
    state = sink.rewind_provider.rewind_state()

    old_rewind = json.loads(json.dumps(state))
    old_rewind["schema"] = "unigym_bench.rewind/4"
    with pytest.raises(ValueError, match=r"expected 'unigym_bench.rewind/5'"):
        engine.validate_rewind_state(old_rewind)
    with pytest.raises(ValueError, match=r"expected 'unigym_bench.rewind/5'"):
        engine.apply_rewind_state(old_rewind)

    old_monitor = json.loads(json.dumps(state))
    old_monitor["monitors"]["contact"]["schema"] = "unigym_bench.monitor/3"
    with pytest.raises(ValueError, match=r"expected schema 'unigym_bench.monitor/4'"):
        engine.validate_rewind_state(old_monitor)
    with pytest.raises(ValueError, match=r"expected schema 'unigym_bench.monitor/4'"):
        engine.apply_rewind_state(old_monitor)
