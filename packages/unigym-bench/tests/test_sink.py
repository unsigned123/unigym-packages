"""Sink shell, construction seam and immutable episode-record contract."""

from __future__ import annotations

import json

import pytest
import torch
from unigym.core.episode_termination import (
    EpisodeOutcome,
    EpisodeTermination,
    Termination,
    TerminationSource,
)
from unigym.errors import ConfigError
from unigym.signals import SignalBus

from unigym_bench.engine import StreamingEngine
from unigym_bench.events import Event, EpisodeRecord
from unigym_bench.json import ensure_json_value
from unigym_bench.sink import BenchmarkSink, make_sink
from unigym_bench.specs import RetentionSpec
from _support import (
    ScriptedMonitor,
    TickScript,
    make_bundle,
    make_sink_context,
    make_termination,
)

_FACTORY = "_support:make_threshold_monitors"


def _engine_sink(
    monitor: ScriptedMonitor,
) -> tuple[BenchmarkSink, StreamingEngine]:
    """Build one sink over a real engine, retaining episode records."""
    engine = StreamingEngine([monitor], retention=RetentionSpec(retain_episodes=True))
    return BenchmarkSink(engine=engine), engine


def test_make_sink_builds_from_a_real_context_and_monitor_factory() -> None:
    sink = make_sink(make_sink_context(options={"monitor_factory": _FACTORY}))

    assert isinstance(sink, BenchmarkSink)
    assert sink.loop_mode == "pool"
    assert sink.stop_when() is False
    assert sink.rewind_provider is None


def test_make_sink_accepts_sink_steered_with_a_publisher() -> None:
    bus = SignalBus(num_envs=1, device="cpu")
    sink = make_sink(
        make_sink_context(
            options={"loop_mode": "sink-steered", "monitor_factory": _FACTORY},
            signals=bus,
        )
    )

    assert sink.loop_mode == "sink-steered"


def test_make_sink_rejects_sink_steered_without_signals() -> None:
    with pytest.raises(ConfigError, match="requires ctx.signals"):
        make_sink(make_sink_context(options={"loop_mode": "sink-steered"}))


def test_make_sink_rejects_unknown_keys_and_loop_modes() -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        make_sink(make_sink_context(options={"typo": 1}))
    with pytest.raises(ConfigError, match="loop_mode"):
        make_sink(make_sink_context(options={"loop_mode": "closed"}))


def test_make_sink_rejects_auto_reset_false() -> None:
    with pytest.raises(ConfigError, match="runner.auto_reset: true"):
        make_sink(make_sink_context(auto_reset=False))


@pytest.mark.parametrize(
    "factory",
    [
        123,
        "not_a_module:make",
        "no_such_pack_module:make",
        "_support:MissingFactory",
    ],
)
def test_make_sink_rejects_invalid_monitor_factory_paths(factory: object) -> None:
    with pytest.raises(ConfigError, match="monitor_factory"):
        make_sink(make_sink_context(options={"monitor_factory": factory}))


def test_make_sink_engine_rejects_a_non_sequence_factory_result() -> None:
    with pytest.raises(TypeError, match="expected a sequence"):
        make_sink(make_sink_context(options={"monitor_factory": "_support:make_bundle"}))


def test_make_sink_rejects_a_non_callable_factory_target() -> None:
    with pytest.raises(ConfigError, match="is not callable"):
        make_sink(make_sink_context(options={"monitor_factory": "_support:_SINK_CONTEXT_CFG"}))


def test_make_sink_without_a_factory_builds_an_empty_honest_run() -> None:
    sink = make_sink(make_sink_context())

    assert sink.on_run_end() == {
        "BenchmarkSink": {
            "episodes": 0,
            "total_ticks": 0,
            "verdict_counts": {},
            "event_count": 0,
            "dropped_events": 0,
            "conflicts": 0,
        }
    }


def test_direct_sink_steered_construction_requires_a_publisher() -> None:
    with pytest.raises(ConfigError, match="requires a signal publisher"):
        BenchmarkSink([], loop_mode="sink-steered")

    bus = SignalBus(num_envs=1, device="cpu")
    assert BenchmarkSink([], loop_mode="sink-steered", publisher=bus).loop_mode == "sink-steered"


def test_injected_engine_sink_steered_requires_an_engine_publisher() -> None:
    with pytest.raises(ConfigError, match="requires a signal publisher"):
        BenchmarkSink(engine=StreamingEngine([]), loop_mode="sink-steered")

    bus = SignalBus(num_envs=1, device="cpu")
    sink = BenchmarkSink(engine=StreamingEngine([], publisher=bus), loop_mode="sink-steered")
    assert sink.loop_mode == "sink-steered"


def test_direct_construction_rejects_a_non_resetting_runner_assertion() -> None:
    with pytest.raises(ConfigError, match="reset_after_episode"):
        BenchmarkSink([], reset_after_episode=False)

    with pytest.raises(TypeError, match="reset_after_episode"):
        BenchmarkSink([], reset_after_episode="yes")


def test_direct_construction_rejects_engine_and_monitor_conflicts() -> None:
    engine = StreamingEngine([])

    with pytest.raises(TypeError, match="not both"):
        BenchmarkSink([], engine=engine)
    with pytest.raises(TypeError, match="owns retention and publisher"):
        BenchmarkSink(engine=engine, retention=RetentionSpec())


def test_sink_delegates_the_terminal_drain_and_reopens_after_step_end() -> None:
    monitor = ScriptedMonitor(
        "contact",
        {0: TickScript(terminals={0: make_termination("custom.contact.hit")})},
    )
    sink, engine = _engine_sink(monitor)
    bundle = make_bundle(num_envs=1, t=0)

    sink.on_step(0, bundle)
    pending = sink.drain_terminations()
    assert set(pending) == {0}
    assert pending[0].code == "custom.contact.hit"

    sink.on_episode_end([0], bundle)
    assert engine.open_envs == ()
    sink.on_step_end(0, bundle)
    assert engine.open_envs == (0,)
    assert engine.records[0].episode_seq == 0


def test_on_run_end_immediately_after_reset_skips_the_zero_tick_reopened_env() -> None:
    monitor = ScriptedMonitor(
        "contact",
        {0: TickScript(terminals={0: make_termination("custom.contact.hit")})},
    )
    sink, engine = _engine_sink(monitor)
    bundle = make_bundle(num_envs=1, t=0)

    sink.on_step(0, bundle)
    sink.drain_terminations()
    sink.on_episode_end([0], bundle)
    sink.on_step_end(0, bundle)
    assert engine.open_envs == (0,)

    summary = sink.on_run_end()

    assert summary["BenchmarkSink"]["episodes"] == 1
    assert summary["BenchmarkSink"]["verdict_counts"] == {"terminal": 1}
    assert len(engine.records) == 1
    assert engine.records[0].verdict == "terminal"
    assert engine.open_envs == ()


def test_reopened_episode_captures_new_command_and_reset_facts() -> None:
    monitor = ScriptedMonitor(
        "reopen",
        {
            0: TickScript(terminals={0: make_termination("custom.reopen.first")}),
            1: TickScript(terminals={0: make_termination("custom.reopen.second")}),
        },
    )
    sink, engine = _engine_sink(monitor)
    first = make_bundle(
        num_envs=1,
        t=0,
        cmd=torch.tensor([[0.1, 0.2]]),
        reset_state={0: {"reset_index": 0, "reset_valid": True, "reset_reason": "first"}},
    )
    sink.on_step(0, first)
    sink.drain_terminations()
    sink.on_episode_end([0], first)
    sink.on_step_end(0, first)

    second = make_bundle(
        num_envs=1,
        t=1,
        cmd=torch.tensor([[0.7, 0.8]]),
        reset_state={0: {"reset_index": 0, "reset_valid": False, "reset_reason": "second"}},
    )
    sink.on_step(1, second)
    sink.drain_terminations()
    sink.on_episode_end([0], second)

    assert [
        (record.episode_seq, record.reset_facts["reset_reason"]) for record in engine.records
    ] == [(0, "first"), (1, "second")]
    assert engine.records[0].command_at_start == pytest.approx((0.1, 0.2))
    assert engine.records[1].command_at_start == pytest.approx((0.7, 0.8))


def test_episode_record_captures_start_command_reset_facts_and_canonical_terminal() -> None:
    monitor = ScriptedMonitor(
        "hit",
        {0: TickScript(terminals={0: make_termination("custom.hit.first")})},
    )
    sink, engine = _engine_sink(monitor)
    reset_state = {0: {"reset_index": 0, "reset_valid": True, "reset_reason": "initial"}}
    bundle = make_bundle(
        num_envs=1,
        t=0,
        cmd=torch.tensor([[0.25, -0.5]]),
        reset_state=reset_state,
    )

    sink.on_step(0, bundle)
    sink.drain_terminations()
    stamped = Termination(
        outcome=EpisodeOutcome.SUCCESS,
        code="custom.hit.first",
        source=TerminationSource.CUSTOM,
        step_index=1,
        time_s=0.02,
    )
    core = Termination(
        outcome=EpisodeOutcome.FAILURE,
        code="env.fallen",
        source=TerminationSource.ENVIRONMENT,
        step_index=1,
        time_s=0.02,
    )
    bundle["terminations"] = {0: EpisodeTermination.from_terminations([core, stamped])}
    sink.on_episode_end([0], bundle)

    record = engine.records[0]
    assert record.command_at_start == (0.25, -0.5)
    assert record.reset_facts == reset_state[0]
    assert record.first_terminal is not None
    assert record.first_terminal.step_index == 1
    assert record.first_terminal.time_s == 0.02
    assert record.core_termination is not None
    assert [term.code for term in record.core_termination.all] == ["env.fallen"]
    assert record.verdict == "terminal"
    assert record.rewound is False


def test_record_keeps_the_monitor_request_when_no_canonical_stamp_is_available() -> None:
    monitor = ScriptedMonitor(
        "raw",
        {0: TickScript(terminals={0: make_termination("custom.raw.first")})},
    )
    sink, engine = _engine_sink(monitor)
    bundle = make_bundle(num_envs=1, t=0)

    sink.on_step(0, bundle)
    sink.drain_terminations()
    sink.on_episode_end([0], bundle)

    record = engine.records[0]
    assert record.first_terminal is not None
    assert record.first_terminal.code == "custom.raw.first"
    assert record.first_terminal.step_index is None
    assert record.core_termination is None


def test_record_collects_episode_event_summaries() -> None:
    monitor = ScriptedMonitor(
        "events",
        {0: TickScript(events=(Event(name="hit", code="task.events.hit", env_ids=(0,)),))},
    )
    sink, engine = _engine_sink(monitor)
    bundle = make_bundle(num_envs=1, t=0)

    sink.on_step(0, bundle)
    sink.on_episode_end([0], bundle)

    assert engine.records[0].event_counts == {"task.events.hit": 1}


def test_open_episode_is_incomplete_in_the_unique_run_summary() -> None:
    sink, _ = _engine_sink(ScriptedMonitor("open", {}))
    sink.on_step(0, make_bundle(num_envs=1, t=0))

    summary = sink.on_run_end()

    assert set(summary) == {"BenchmarkSink"}
    document = summary["BenchmarkSink"]
    assert isinstance(document, dict)
    assert document["episodes"] == 1
    assert document["verdict_counts"] == {"incomplete": 1}
    assert json.loads(json.dumps(document)) == document


def test_zero_episode_run_returns_an_honest_summary() -> None:
    sink = BenchmarkSink([])

    summary = sink.on_run_end()

    assert summary == {
        "BenchmarkSink": {
            "episodes": 0,
            "total_ticks": 0,
            "verdict_counts": {},
            "event_count": 0,
            "dropped_events": 0,
            "conflicts": 0,
        }
    }


def test_episode_record_is_frozen_and_json_safe() -> None:
    record = EpisodeRecord(
        env_id=0,
        episode_seq=1,
        ticks=3,
        verdict="incomplete",
        command_at_start=[0.1, 0.2],
        reset_facts={"reset_index": 0, "reset_valid": True},
        event_counts={"task.record.hit": 2},
    )

    assert record.command_at_start == (0.1, 0.2)
    with pytest.raises(TypeError):
        record.reset_facts["reset_index"] = 9
    with pytest.raises(TypeError):
        record.event_counts["task.record.hit"] = 9
    document = record.to_json()
    assert ensure_json_value(document) == document
    assert json.loads(json.dumps(document)) == document


def test_episode_record_deep_freezes_nested_reset_facts() -> None:
    source: dict[str, object] = {
        "nested": {"values": [1, 2], "flags": [True, None]},
        "name": "sample",
    }
    record = EpisodeRecord(
        env_id=0,
        episode_seq=0,
        ticks=1,
        verdict="incomplete",
        reset_facts=source,
    )

    source["nested"]["values"].append(3)

    nested = record.reset_facts["nested"]
    assert nested["values"] == (1, 2)
    assert nested["flags"] == (True, None)
    with pytest.raises(TypeError):
        nested["extra"] = 1
    with pytest.raises(AttributeError):
        nested["values"].append(3)


def test_episode_record_to_json_returns_a_private_deep_copy() -> None:
    record = EpisodeRecord(
        env_id=0,
        episode_seq=0,
        ticks=1,
        verdict="incomplete",
        reset_facts={"nested": {"values": [1, 2]}},
    )

    document = record.to_json()
    reset_facts = document["reset_facts"]
    assert isinstance(reset_facts, dict)
    nested = reset_facts["nested"]
    assert isinstance(nested, dict)
    nested["values"].append(99)
    nested["extra"] = True

    assert record.to_json()["reset_facts"] == {"nested": {"values": [1, 2]}}


def test_episode_record_rejects_non_json_reset_facts() -> None:
    with pytest.raises(TypeError, match="reset_facts"):
        EpisodeRecord(
            env_id=0,
            episode_seq=0,
            ticks=1,
            verdict="incomplete",
            reset_facts={"bad": torch.tensor(1.0)},
        )
