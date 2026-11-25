"""Streaming-engine contract: latches, drains, journal and lifecycle."""

from __future__ import annotations

import gc
import weakref

import pytest
import torch
from unigym.core.episode_termination import (
    EpisodeOutcome,
    EpisodeTermination,
    TerminationSource,
)
from unigym.values import StepValues, ValueStore

from unigym_bench.engine import EngineStateError, StreamingEngine
from unigym_bench.flow import DefaultBenchmarkFlow, EventRule, FlowDecision, TerminationRule
from unigym_bench.monitors import PredicateMonitor, threshold_config
from unigym_bench.specs import RetentionSpec
from _support import (
    ScriptedMonitor,
    TickScript,
    make_bundle,
    make_termination,
    make_values,
)


def _apply_flow_tick(
    engine: StreamingEngine, flow: DefaultBenchmarkFlow, bundle: dict[str, object]
) -> None:
    tick = engine.begin_tick(bundle)
    engine.apply_decision(tick, flow.on_tick(tick))


def test_multi_env_interleaving_latches_each_env_independently() -> None:
    monitor = threshold_config("contact", "value:level", threshold=0.5)
    flow = DefaultBenchmarkFlow(
        (TerminationRule("contact", "threshold_reached", "success", "custom.contact.threshold"),)
    )
    engine = StreamingEngine([monitor], retention=RetentionSpec(retain_episodes=True))

    _apply_flow_tick(engine, flow, make_bundle(t=0, num_envs=2, values=make_values([1.0, 0.0])))
    first = engine.drain_terminations()
    assert set(first) == {0}
    assert first[0].code == "custom.contact.threshold"
    engine.on_episode_end([0], make_bundle(t=0, num_envs=2))
    engine.on_reset([0])

    _apply_flow_tick(engine, flow, make_bundle(t=1, num_envs=2, values=make_values([0.0, 1.0])))
    second = engine.drain_terminations()
    assert set(second) == {1}
    assert second[1].code == "custom.contact.threshold"
    engine.on_episode_end([1], make_bundle(t=1, num_envs=2))

    assert engine.drain_terminations() == {}
    assert engine.conflicts == {}
    assert [
        (record.env_id, record.episode_seq, record.ticks, record.verdict)
        for record in engine.records
    ] == [
        (0, 0, 1, "terminal"),
        (1, 0, 2, "terminal"),
    ]
    assert engine.summary.episodes == 2


def test_late_terminal_for_open_latched_env_is_a_conflict() -> None:
    monitor = ScriptedMonitor(
        "late",
        {
            0: TickScript(values={"fired": torch.tensor([True])}),
            1: TickScript(values={"fired": torch.tensor([True])}),
        },
    )
    flow = DefaultBenchmarkFlow((TerminationRule("late", "fired", "success", "custom.late.first"),))
    engine = StreamingEngine([monitor])

    _apply_flow_tick(engine, flow, make_bundle(num_envs=1, t=0))
    pending = engine.drain_terminations()
    assert set(pending) == {0}
    assert pending[0].code == "custom.late.first"
    _apply_flow_tick(engine, flow, make_bundle(num_envs=1, t=1))
    assert engine.drain_terminations() == {}
    assert dict(engine.conflicts) == {0: 1}
    assert engine.conflict_count == 1


def test_terminal_for_closed_env_is_a_contract_error() -> None:
    monitor = ScriptedMonitor(
        "closed",
        {
            0: TickScript(values={"fired": torch.tensor([True])}),
            1: TickScript(values={"fired": torch.tensor([True])}),
        },
    )
    flow = DefaultBenchmarkFlow(
        (TerminationRule("closed", "fired", "success", "custom.closed.first"),)
    )
    engine = StreamingEngine([monitor], retention=RetentionSpec(retain_episodes=True))

    _apply_flow_tick(engine, flow, make_bundle(num_envs=1, t=0))
    engine.drain_terminations()
    engine.on_episode_end([0], make_bundle(num_envs=1, t=0))

    with pytest.raises(EngineStateError, match="closed env 0"):
        _apply_flow_tick(engine, flow, make_bundle(num_envs=1, t=1))
    with pytest.raises(EngineStateError, match="already closed"):
        engine.on_episode_end([0], make_bundle(num_envs=1, t=1))


def test_reset_of_open_env_restarts_in_place_without_a_record() -> None:
    monitor = ScriptedMonitor("idle", {})
    engine = StreamingEngine([monitor], retention=RetentionSpec(retain_episodes=True))
    engine.on_step(make_bundle(num_envs=1, t=0))

    engine.on_reset([0])

    assert engine.open_envs == (0,)
    assert engine.records == ()
    assert monitor.resets == [(0,)]

    engine.on_step(make_bundle(num_envs=1, t=1))
    engine.on_episode_end([0], make_bundle(num_envs=1, t=1))

    assert [(record.episode_seq, record.ticks) for record in engine.records] == [(0, 1)]


def test_reset_of_open_env_with_no_new_tick_reports_no_episode() -> None:
    engine = StreamingEngine(
        [ScriptedMonitor("idle", {})],
        retention=RetentionSpec(retain_episodes=True),
    )
    engine.on_step(make_bundle(num_envs=1, t=0))

    engine.on_reset([0])
    summary = engine.on_run_end()

    assert summary.episodes == 0
    assert engine.records == ()


def test_event_journal_cap_and_dropped_count() -> None:
    script = {tick: TickScript(values={"fired": torch.tensor([True])}) for tick in range(3)}
    flow = DefaultBenchmarkFlow(events=(EventRule("cap", "fired", "tick", code="task.cap.tick"),))
    engine = StreamingEngine(
        [ScriptedMonitor("cap", script)],
        retention=RetentionSpec(max_events=2, retain_episodes=True),
    )
    for tick in range(3):
        _apply_flow_tick(engine, flow, make_bundle(num_envs=1, t=tick))

    assert len(engine.journal) == 2
    assert engine.dropped_events == 1

    engine.on_episode_end([0], make_bundle(num_envs=1, t=2))
    assert engine.records[0].event_counts == {"task.cap.tick": 3}
    assert engine.summary.event_count == 3
    assert engine.summary.dropped_events == 1


def test_event_log_disabled_counts_every_occurrence_as_dropped() -> None:
    monitor = ScriptedMonitor(
        "disabled",
        {0: TickScript(values={"fired": torch.tensor([True])})},
    )
    flow = DefaultBenchmarkFlow(
        events=(EventRule("disabled", "fired", "tick", code="task.cap.tick"),)
    )
    engine = StreamingEngine([monitor], retention=RetentionSpec(event_log=False))

    _apply_flow_tick(engine, flow, make_bundle(num_envs=1, t=0))
    assert engine.journal == ()
    assert engine.dropped_events == 1


def test_signal_publication_is_namespaced() -> None:
    first = ScriptedMonitor(
        "first",
        {
            0: TickScript(
                values={
                    "level": torch.tensor([1.0, 2.0]),
                    "active": torch.tensor([True, False]),
                }
            )
        },
    )
    second = ScriptedMonitor(
        "second",
        {0: TickScript(values={"value": torch.tensor([3.0, 4.0])})},
    )
    store = ValueStore(num_envs=2, device="cpu")
    engine = StreamingEngine([first, second], publisher=store.publisher("sink"))

    engine.on_step(make_bundle(num_envs=2, t=0))

    snapshot = store.snapshot()
    assert set(snapshot) == {"first/level", "first/active", "second/value"}
    assert snapshot["first/level"].tolist() == [1.0, 2.0]
    assert snapshot["first/active"].tolist() == [True, False]
    assert snapshot["second/value"].tolist() == [3.0, 4.0]


def test_values_are_not_published_for_a_closed_env_tick() -> None:
    monitor = ScriptedMonitor(
        "levels",
        {
            0: TickScript(values={"level": torch.tensor([1.0, 2.0])}),
            1: TickScript(values={"level": torch.tensor([3.0, 4.0])}),
            2: TickScript(values={"level": torch.tensor([5.0, 6.0])}),
        },
    )
    store = ValueStore(num_envs=2, device="cpu")
    engine = StreamingEngine(
        [monitor],
        publisher=store.publisher("sink"),
        retention=RetentionSpec(retain_episodes=True),
    )

    engine.on_step(make_bundle(num_envs=2, t=0))
    assert store.snapshot()["levels/level"].tolist() == [1.0, 2.0]

    engine.on_episode_end([1], make_bundle(num_envs=2, t=0))
    engine.on_step(make_bundle(num_envs=2, t=1))
    assert store.snapshot()["levels/level"].tolist() == [1.0, 2.0]

    engine.on_reset([1])
    engine.on_step(make_bundle(num_envs=2, t=2))
    assert store.snapshot()["levels/level"].tolist() == [5.0, 6.0]


def test_episode_end_reset_and_reopen_start_a_new_sequence() -> None:
    monitor = ScriptedMonitor(
        "reopen",
        {
            0: TickScript(values={"fired": torch.tensor([True])}),
            1: TickScript(values={"fired": torch.tensor([True])}),
        },
    )
    flow = DefaultBenchmarkFlow(
        (TerminationRule("reopen", "fired", "success", "custom.reopen.first"),)
    )
    engine = StreamingEngine([monitor], retention=RetentionSpec(retain_episodes=True))

    _apply_flow_tick(engine, flow, make_bundle(num_envs=1, t=0))
    assert engine.drain_terminations()[0].code == "custom.reopen.first"
    engine.on_episode_end([0], make_bundle(num_envs=1, t=0))
    assert engine.open_envs == ()
    assert monitor.episode_ends == [(0,)]
    assert monitor.resets == [(0,)]

    engine.on_reset([0])
    assert engine.open_envs == (0,)
    _apply_flow_tick(engine, flow, make_bundle(num_envs=1, t=1))
    assert engine.drain_terminations()[0].code == "custom.reopen.first"
    engine.on_episode_end([0], make_bundle(num_envs=1, t=1))

    assert [(record.episode_seq, record.ticks, record.verdict) for record in engine.records] == [
        (0, 1, "terminal"),
        (1, 1, "terminal"),
    ]
    assert engine.conflicts == {}


def test_zero_tick_run_finalizes_no_episodes() -> None:
    engine = StreamingEngine([], retention=RetentionSpec(retain_episodes=True))

    summary = engine.on_run_end()

    assert summary.episodes == 0
    assert summary.total_ticks == 0
    assert summary.verdict_counts == {}
    assert summary.event_count == 0
    assert summary.dropped_events == 0
    assert summary.conflicts == 0
    assert engine.records == ()
    assert engine.journal == ()
    assert engine.num_envs is None
    assert engine.on_run_end().episodes == 0


def test_open_episodes_are_finalized_as_incomplete_at_run_end() -> None:
    monitor = ScriptedMonitor(
        "open",
        {
            0: TickScript(values={"fired": torch.tensor([True, False])}),
            1: TickScript(values={"fired": torch.tensor([False, False])}),
        },
    )
    flow = DefaultBenchmarkFlow((TerminationRule("open", "fired", "success", "custom.open.first"),))
    engine = StreamingEngine([monitor], retention=RetentionSpec(retain_episodes=True))

    _apply_flow_tick(engine, flow, make_bundle(num_envs=2, t=0))
    assert set(engine.drain_terminations()) == {0}
    _apply_flow_tick(engine, flow, make_bundle(num_envs=2, t=1))

    summary = engine.on_run_end()
    records = engine.records
    assert [(record.env_id, record.verdict, record.ticks) for record in records] == [
        (0, "incomplete", 2),
        (1, "incomplete", 2),
    ]
    assert records[0].first_terminal is not None
    assert records[0].first_terminal.code == "custom.open.first"
    assert summary.episodes == 2
    assert summary.verdict_counts == {"incomplete": 2}
    assert summary.total_ticks == 4
    assert engine.open_envs == ()


def test_due_env_without_pack_terminal_uses_core_termination_verdict() -> None:
    core = EpisodeTermination(
        primary=make_termination(
            "runtime.episode_horizon",
            outcome=EpisodeOutcome.DEADLINE,
            source=TerminationSource.RUNTIME,
        )
    )
    engine = StreamingEngine(
        [ScriptedMonitor("idle", {})],
        retention=RetentionSpec(retain_episodes=True),
    )

    engine.on_step(make_bundle(num_envs=1, t=0))
    engine.on_episode_end([0], make_bundle(num_envs=1, t=0, terminations={0: core}))

    assert engine.records[0].verdict == "deadline"


def test_terminal_source_and_code_are_validated() -> None:
    engine = StreamingEngine([ScriptedMonitor("idle", {})])
    tick = engine.begin_tick(make_bundle(num_envs=1, t=0))
    bad_source = make_termination("custom.bad.reason", source=TerminationSource.ENVIRONMENT)
    with pytest.raises(ValueError, match="source must be custom"):
        engine.apply_decision(tick, FlowDecision(terminations={0: bad_source}))

    tick = engine.begin_tick(make_bundle(num_envs=1, t=1))
    bad_code = make_termination("env.force_limit")
    with pytest.raises(ValueError, match="must start with 'custom.'"):
        engine.apply_decision(tick, FlowDecision(terminations={0: bad_code}))


def test_value_shape_is_validated() -> None:
    monitor = ScriptedMonitor(
        "bad_value",
        {0: TickScript(values={"level": torch.tensor([1.0, 2.0])})},
    )

    with pytest.raises(ValueError, match="leading dimension num_envs=1"):
        StreamingEngine([monitor]).on_step(make_bundle(num_envs=1, t=0))


def test_duplicate_monitor_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate monitor name"):
        StreamingEngine([ScriptedMonitor("same", {}), ScriptedMonitor("same", {})])


def test_engine_does_not_retain_tick_tensors_or_unrequested_records() -> None:
    engine = StreamingEngine(
        [ScriptedMonitor("hold", {})],
        retention=RetentionSpec(retain_episodes=False),
    )
    step_index = torch.zeros(1, dtype=torch.int64)
    bundle = make_bundle(num_envs=1, t=0, step_index=step_index)

    engine.on_step(bundle)
    tensor_ref = weakref.ref(step_index)

    del bundle, step_index
    gc.collect()

    assert tensor_ref() is None
    assert engine.journal == ()

    engine.on_episode_end([0], make_bundle(num_envs=1, t=0))
    assert engine.records == ()
    assert engine.summary.episodes == 1


def _final_predicate() -> "PredicateMonitor":
    return PredicateMonitor(
        "end_pred",
        {"source": "value:flag", "op": "gt", "value": 0.0},
        "episode_end",
    )


def test_episode_end_uses_final_step_values_and_publishes_final_levels() -> None:
    monitor = _final_predicate()
    store = ValueStore(num_envs=1, device="cpu")
    engine = StreamingEngine(
        [monitor],
        publisher=store.publisher("sink"),
        retention=RetentionSpec(retain_episodes=True),
    )
    bundle = make_bundle(
        num_envs=1,
        t=0,
        values=StepValues({"flag": torch.tensor([0.0])}),
        final_values=StepValues({"flag": torch.tensor([1.0])}),
    )

    engine.on_step(bundle)
    assert monitor.values()["fired"].tolist() == [False]
    assert store.snapshot()["end_pred/fired"].tolist() == [False]

    engine.on_episode_end([0], bundle)

    assert engine.records[0].event_counts == {}
    assert engine.summary.event_count == 0
    assert engine.summary.termination_code_counts == {}
    assert engine.summary.termination_outcome_counts == {}
    assert engine.drain_terminations() == {}
    assert engine.conflicts == {}
    assert store.snapshot()["end_pred/fired"].tolist() == [True]


def test_final_episode_value_never_enters_the_runner_latch_or_drain() -> None:
    tick = ScriptedMonitor(
        "tick",
        {0: TickScript(values={"fired": torch.tensor([True])})},
    )
    final = _final_predicate()
    flow = DefaultBenchmarkFlow((TerminationRule("tick", "fired", "success", "custom.tick.hit"),))
    engine = StreamingEngine(
        [tick, final],
        retention=RetentionSpec(retain_episodes=True),
    )
    bundle = make_bundle(
        num_envs=1,
        t=0,
        values=StepValues({"flag": torch.tensor([0.0])}),
        final_values=StepValues({"flag": torch.tensor([1.0])}),
    )

    _apply_flow_tick(engine, flow, bundle)
    assert set(engine.drain_terminations()) == {0}
    engine.on_episode_end([0], bundle)

    assert engine.conflicts == {}
    assert engine.drain_terminations() == {}
    assert engine.summary.termination_code_counts == {"custom.tick.hit": 1}
    assert engine.records[0].event_counts == {}


def test_episode_end_requires_the_final_step_context() -> None:
    engine = StreamingEngine([ScriptedMonitor("idle", {})])
    bundle = make_bundle(num_envs=1, t=0)
    engine.on_step(bundle)
    del bundle["final_step"]

    with pytest.raises(EngineStateError, match="final_step"):
        engine.on_episode_end([0], bundle)


def test_on_step_reads_tick_values_not_the_final_step_values() -> None:
    monitor = PredicateMonitor(
        "tick_pred",
        {"source": "value:flag", "op": "gt", "value": 0.0},
        "tick",
    )
    engine = StreamingEngine([monitor])
    bundle = make_bundle(
        num_envs=1,
        t=0,
        values=StepValues({"flag": torch.tensor([0.0])}),
        final_values=StepValues({"flag": torch.tensor([1.0])}),
    )

    engine.on_step(bundle)

    assert engine.journal == ()
    assert engine.summary.event_count == 0
