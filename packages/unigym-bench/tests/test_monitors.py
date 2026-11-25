"""Monitor protocol, configured predicates, phase and coverage extension."""

from __future__ import annotations

import gc
import math
import os
import subprocess
import sys
import weakref
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

import unigym_bench
from _support import make_context, make_value_context, make_values
from unigym.core.episode_termination import (
    EpisodeOutcome,
    Termination,
    TerminationExtension,
    TerminationSource,
)
from unigym.evaluation import EnvStepContext
from unigym_bench.conditions import ConditionError
from unigym_bench.events import EpisodeRecord, EpisodeSummary, Event
from unigym_bench.json import ensure_json_value
from unigym_bench.monitors import (
    BaseMonitor,
    Monitor,
    PhaseMonitor,
    PredicateMonitor,
    RewindableMonitor,
    event_counter_config,
    hold_config,
    safety_config,
    threshold_config,
)
from unigym_bench.report import (
    RunMetadata,
    RunSummary,
    ScenarioCoverageExtension,
    validate_report_extensions,
)
from unigym_bench.sources import SourceRef


def _assert_roundtrip(monitor: object, clone: object, frames: Sequence[EnvStepContext]) -> None:
    for frame in frames:
        monitor.on_tick(frame)
    snapshot = monitor.snapshot()
    assert ensure_json_value(snapshot) == snapshot
    clone.restore(snapshot)
    assert clone.snapshot() == snapshot
    assert list(monitor.drain_events()) == list(clone.drain_events())
    assert dict(monitor.drain_terminals()) == dict(clone.drain_terminals())
    left = monitor.signals()
    right = clone.signals()
    assert set(left) == set(right)
    for key in left:
        assert torch.equal(left[key], right[key])


def _row_context(
    row: Sequence[float],
    *,
    t: int = 0,
    dtype: torch.dtype = torch.float32,
    name: str = "level",
) -> EnvStepContext:
    return make_value_context(row, t=t, dtype=dtype, name=name)


def _summary() -> RunSummary:
    return RunSummary(
        metadata=RunMetadata(),
        episodes=(),
        episode_summary=EpisodeSummary(),
        monitor_views={},
    )


def _record(env_id: int, command: Sequence[float] | None) -> EpisodeRecord:
    return EpisodeRecord(
        env_id=env_id,
        episode_seq=0,
        ticks=1,
        verdict="incomplete",
        command_at_start=None if command is None else tuple(command),
    )


def test_base_monitor_defaults() -> None:
    monitor = BaseMonitor("base_monitor")

    assert isinstance(monitor, Monitor)
    assert not isinstance(monitor, RewindableMonitor)
    assert monitor.name == "base_monitor"
    assert dict(monitor.signals()) == {}
    assert tuple(monitor.drain_events()) == ()
    assert dict(monitor.drain_terminals()) == {}

    frame = make_context()
    monitor.on_tick(frame)
    monitor.on_episode_end((0,))
    monitor.reset([0])

    view = monitor.report_view()
    assert view == {"name": "base_monitor", "kind": "base"}
    assert ensure_json_value(view) == view


@pytest.mark.parametrize("name", ["", "Upper", "1lead", "with-dash", "with space", 3, None])
def test_monitor_name_must_be_snake_case(name: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        BaseMonitor(name)


def test_base_monitor_rejects_bad_env_ids() -> None:
    monitor = BaseMonitor("base_monitor")

    with pytest.raises(TypeError):
        monitor.reset("not ids")
    with pytest.raises(ValueError):
        monitor.reset([0, 0])


def test_configuration_helpers_return_predicate_monitors() -> None:
    monitors = [
        threshold_config("threshold", "value:level", threshold=0.5),
        hold_config("hold", "value:level", threshold=0.5, hold_steps=2),
        event_counter_config("counter", "value:flag"),
        safety_config("safety", "value:level", threshold=0.5),
    ]

    for monitor in monitors:
        assert isinstance(monitor, PredicateMonitor)
        assert isinstance(monitor, Monitor)
        assert isinstance(monitor, RewindableMonitor)
        report = monitor.report_view()
        assert ensure_json_value(report) == report
        assert report["kind"] == "predicate"


def test_threshold_configuration_parity_and_report_mapping() -> None:
    rows = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    monitor = threshold_config(
        "contact",
        SourceRef.parse("value:level"),
        threshold=0.5,
        terminal_outcome="failure",
    )

    events_per_tick = []
    terminals_per_tick = []
    for t, row in enumerate(rows):
        monitor.on_tick(_row_context(row, t=t))
        events = monitor.drain_events()
        events_per_tick.append([event.env_ids for event in events])
        terminals_per_tick.append(len(monitor.drain_terminals()))

    assert events_per_tick == [[], [(0,)], [], [(1,)]]
    assert terminals_per_tick == [0, 1, 0, 1]

    report = monitor.report_view()
    assert report["condition"] == {"source": "value:level", "op": "gte", "value": 0.5}
    assert report["trigger"] == {"kind": "edge"}
    assert report["event"] == "threshold_reached"
    assert report["terminal_reason"] == "threshold"
    assert report["terminal_outcome"] == "failure"
    assert report["fired"] == 2
    assert report["fired_counts"] == [1, 1]
    assert report["episodes"] == 0

    monitor.reset([0])
    monitor.on_tick(_row_context([1.0, 0.0], t=4))
    assert [event.env_ids for event in monitor.drain_events()] == [(0,)]
    assert monitor.report_view()["fired"] == 3
    assert monitor.report_view()["fired_counts"] == [2, 1]


def test_threshold_event_and_terminal_keep_old_code_and_outcome() -> None:
    monitor = threshold_config(
        "contact",
        "value:level",
        threshold=0.5,
        operator=">",
        terminal_outcome="failure",
    )

    monitor.on_tick(_row_context([1.0], dtype=torch.float32))
    events = monitor.drain_events()
    assert len(events) == 1
    assert events[0].name == "threshold_reached"
    assert events[0].code == "task.contact.threshold_reached"
    assert events[0].severity == "info"
    assert events[0].env_ids == (0,)
    assert events[0].data == {"trigger": "edge"}

    terminals = monitor.drain_terminals()
    assert set(terminals) == {0}
    terminal = terminals[0]
    assert terminal.outcome is EpisodeOutcome.FAILURE
    assert terminal.source is TerminationSource.CUSTOM
    assert terminal.code == "custom.contact.threshold"
    assert terminal.metrics == {}

    monitor.on_tick(_row_context([1.0], t=1))
    assert tuple(monitor.drain_terminals()) == ()


@pytest.mark.parametrize(
    ("operator", "condition_op"),
    [(">", "gt"), (">=", "gte"), ("<", "lt"), ("<=", "lte")],
)
def test_threshold_operator_maps_to_the_condition_dsl(operator: str, condition_op: str) -> None:
    monitor = threshold_config("contact", "value:level", threshold=1.0, operator=operator)

    assert monitor.report_view()["condition"]["op"] == condition_op


def test_threshold_configuration_validates_inputs() -> None:
    with pytest.raises(TypeError):
        threshold_config("bad", 7, threshold=0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        threshold_config("bad", "value:level", threshold=math.nan)
    with pytest.raises(TypeError):
        threshold_config("bad", "value:level", threshold=True)
    with pytest.raises(ValueError):
        threshold_config("bad", "value:level", threshold=0.0, operator="=")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        threshold_config("bad", "value:level", threshold=0.0, terminal_outcome="engine")
    with pytest.raises(ValueError):
        threshold_config("bad", "level", threshold=0.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_predicate_configurations_reject_non_finite_sources(bad: float) -> None:
    threshold = threshold_config("threshold", "value:level", threshold=0.5)
    hold = hold_config("hold", "value:level", threshold=0.5, hold_steps=2)
    safety = safety_config("safety", "value:level", threshold=0.5)

    for monitor in (threshold, hold, safety):
        with pytest.raises(ConditionError, match=r"value:level.*non-finite"):
            monitor.on_tick(_row_context([bad]))


def test_monitor_rejects_changed_num_envs() -> None:
    monitor = threshold_config("stable", "value:level", threshold=0.0)
    monitor.on_tick(_row_context([0.0, 0.0], t=0))

    with pytest.raises(ValueError, match="num_envs changed"):
        monitor.on_tick(_row_context([0.0], t=1))


def test_hold_configuration_parity_and_rearm() -> None:
    rows = [[1.0], [1.0], [0.0], [1.0], [1.0], [1.0]]
    monitor = hold_config(
        "hold",
        "value:level",
        threshold=0.5,
        hold_steps=2,
        terminal_outcome="success",
    )

    event_counts = []
    terminal_counts = []
    for t, row in enumerate(rows):
        monitor.on_tick(_row_context(row, t=t))
        event_counts.append(len(monitor.drain_events()))
        terminal_counts.append(len(monitor.drain_terminals()))

    assert event_counts == [0, 1, 0, 0, 1, 0]
    assert terminal_counts == [0, 1, 0, 0, 1, 0]
    report = monitor.report_view()
    assert report["trigger"] == {"kind": "hold", "steps": 2}
    assert report["event"] == "hold_reached"
    assert report["terminal_reason"] == "hold"
    assert report["fired"] == 2
    assert report["fired_counts"] == [2]

    monitor.reset([0])
    monitor.on_tick(_row_context([1.0], t=6))
    assert len(monitor.drain_events()) == 0
    monitor.on_tick(_row_context([1.0], t=7))
    assert len(monitor.drain_events()) == 1


def test_hold_validates_hold_steps() -> None:
    for value in (0, -1, True):
        with pytest.raises((TypeError, ValueError)):
            hold_config("bad", "value:level", threshold=0.0, hold_steps=value)


def test_safety_configuration_parity() -> None:
    values = [0.9, 1.0, 0.1, 0.9]
    monitor = safety_config(
        "safety",
        "value:level",
        threshold=0.8,
        operator=">",
        severity="fatal",
        terminal_outcome="failure",
    )

    events_per_tick = []
    terminals_per_tick = []
    for t, value in enumerate(values):
        monitor.on_tick(_row_context([value], t=t))
        events = monitor.drain_events()
        events_per_tick.append(len(events))
        if events:
            assert all(event.severity == "fatal" for event in events)
            assert all(event.code == "task.safety.violation" for event in events)
        terminals_per_tick.append(len(monitor.drain_terminals()))

    assert events_per_tick == [1, 0, 0, 1]
    assert terminals_per_tick == [1, 0, 0, 1]
    report = monitor.report_view()
    assert report["severity"] == "fatal"
    assert report["event"] == "violation"
    assert report["terminal_reason"] == "violation"
    assert report["fired"] == 2
    assert report["fired_counts"] == [2]

    monitor.reset([0])
    monitor.on_tick(_row_context([0.9], t=4))
    assert len(monitor.drain_events()) == 1


def test_event_counter_configuration_parity_and_report_only() -> None:
    state = {"row": [False, False]}
    monitor = event_counter_config("occurrences", "value:flag", severity="warning")

    def tick(row: list[bool], t: int) -> None:
        state["row"] = row
        monitor.on_tick(_row_context(row, t=t, dtype=torch.bool, name="flag"))

    tick([False, False], 0)
    tick([True, True], 1)
    tick([True, False], 2)
    tick([False, True], 3)
    tick([True, True], 4)

    assert monitor.signals()["counted"].tolist() == [True, False]
    assert tuple(monitor.drain_events()) == ()
    assert dict(monitor.drain_terminals()) == {}
    report = monitor.report_view()
    assert report["severity"] == "warning"
    assert report["event"] is None
    assert report["terminal_outcome"] is None
    assert report["fired"] == 4
    assert report["fired_counts"] == [2, 2]

    monitor.reset([0])
    tick([True, True], 5)
    report = monitor.report_view()
    assert report["fired"] == 5
    assert report["fired_counts"] == [3, 2]
    assert monitor.signals()["counted"].tolist() == [True, False]


def test_event_counter_rejects_bad_severity() -> None:
    with pytest.raises(ValueError):
        event_counter_config("bad", "value:flag", severity="critical")  # type: ignore[arg-type]


def test_phase_tracks_transitions_and_initializes_silently() -> None:
    values = [-1.0, 0.5, 2.0, 0.5, -2.0]
    monitor = PhaseMonitor("phase", SourceRef.parse("value:level"), boundaries=[0.0, 1.0])

    events_per_tick = []
    phases = []
    for t, value in enumerate(values):
        monitor.on_tick(_row_context([value], t=t))
        events_per_tick.append(len(monitor.drain_events()))
        phases.append(int(monitor.signals()["phase"][0]))

    assert events_per_tick == [0, 1, 1, 1, 1]
    assert phases == [0, 1, 2, 1, 0]
    assert monitor.signals()["phase"].dtype == torch.int64
    assert monitor.signals()["level"].dtype == torch.float32

    report = monitor.report_view()
    assert report["source"] == "value:level"
    assert report["transitions"] == 4
    assert report["phase_ticks"] == {"0": 2, "1": 2, "2": 1}

    monitor.reset([0])
    monitor.on_tick(_row_context([-1.0], t=0))
    assert tuple(monitor.drain_events()) == ()


def test_phase_names_episode_counts_and_snapshot() -> None:
    monitor = PhaseMonitor(
        "phase",
        "value:level",
        boundaries=[0.0, 1.0],
        phase_names=["low", "mid", "high"],
    )
    monitor.on_tick(_row_context([0.5]))
    monitor.on_episode_end((0,))

    report = monitor.report_view()
    assert report["phase_names"] == ["low", "mid", "high"]
    assert report["episode_phases"] == {"low": 0, "mid": 1, "high": 0}

    clone = PhaseMonitor(
        "phase",
        "value:level",
        boundaries=[0.0, 1.0],
        phase_names=["low", "mid", "high"],
    )
    clone.restore(monitor.snapshot())
    assert clone.snapshot() == monitor.snapshot()
    assert clone.report_view() == report


@pytest.mark.parametrize(
    "kwargs",
    [
        {"boundaries": []},
        {"boundaries": [1.0, 0.0]},
        {"boundaries": [0.0, math.inf]},
        {"boundaries": [0.0, 1.0], "phase_names": ["only", "two"]},
        {"boundaries": [0.0, 1.0], "phase_names": ["same", "same", "same"]},
    ],
)
def test_phase_validates_boundaries_and_names(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        PhaseMonitor("bad", "value:level", **kwargs)


def test_phase_source_validation() -> None:
    monitor = PhaseMonitor("phase", "value:missing", boundaries=[0.0])
    with pytest.raises(Exception, match="value source"):
        monitor.on_tick(_row_context([0.0]))

    monitor = PhaseMonitor("phase", "value:level", boundaries=[0.0])
    with pytest.raises(TypeError, match="floating"):
        monitor.on_tick(_row_context([0.0], dtype=torch.bool))
    with pytest.raises(ValueError, match="shape"):
        monitor.on_tick(make_context(num_envs=1, values=make_values([0.0, 1.0])))
    with pytest.raises(ValueError, match="finite"):
        monitor.on_tick(_row_context([math.nan]))


def test_scenario_coverage_extension_matches_float32_exactly() -> None:
    extension = ScenarioCoverageExtension("coverage", commands=[[0.1], [1.0]])
    assert validate_report_extensions([extension]) == (extension,)

    near_float32 = [0.1 + 1e-10]
    extension.on_env_end((0,), {0: _record(0, near_float32)})
    extension.on_env_end((1,), {1: _record(1, [1.0])})

    report = extension.on_run_end(_summary())
    assert report == {
        "name": "coverage",
        "kind": "scenario_coverage",
        "pool_size": 2,
        "covered": 2,
        "coverage_ratio": 1.0,
        "episode_counts": [1, 1],
        "unmatched_episodes": 0,
        "episodes": 2,
    }
    assert ensure_json_value(report) == report


def test_scenario_coverage_extension_unmatched_and_missing_command() -> None:
    extension = ScenarioCoverageExtension("coverage", commands=[[0.0, 0.0], [1.0, 1.0]])

    extension.on_env_end((0,), {0: _record(0, None)})
    extension.on_env_end((1,), {1: _record(1, [2.0, 2.0])})
    extension.on_env_end((0, 1), {0: _record(0, [0.0, 0.0]), 1: _record(1, [1.0, 1.0])})

    report = extension.on_run_end(_summary())
    assert report["pool_size"] == 2
    assert report["covered"] == 2
    assert report["coverage_ratio"] == 1.0
    assert report["episode_counts"] == [1, 1]
    assert report["unmatched_episodes"] == 2
    assert report["episodes"] == 4


def test_scenario_coverage_extension_rejects_bad_pool_and_dimension() -> None:
    for commands in (
        [],
        [[0.0], [0.0]],
        [[0.0], [1.0, 2.0]],
        [[math.nan]],
        [[math.inf]],
        [[1e39]],
    ):
        with pytest.raises((TypeError, ValueError)):
            ScenarioCoverageExtension("bad", commands=commands)

    extension = ScenarioCoverageExtension("coverage", commands=[[0.0, 0.0]])
    extension.on_env_end((0,), {0: _record(0, [0.0])})
    report = extension.on_run_end(_summary())
    assert report["unmatched_episodes"] == 1
    assert report["episodes"] == 1


def test_scenario_coverage_extension_rejects_non_finite_start_command() -> None:
    extension = ScenarioCoverageExtension("coverage", commands=[[0.0]])

    with pytest.raises(ValueError, match="not representable as float32"):
        extension.on_env_end((0,), {0: _record(0, [1e39])})


def test_scenario_coverage_extension_validates_due_records() -> None:
    extension = ScenarioCoverageExtension("coverage", commands=[[0.0]])

    with pytest.raises(TypeError):
        extension.on_env_end((0,), {0: object()})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="missing"):
        extension.on_env_end((0, 1), {0: _record(0, [0.0])})
    with pytest.raises(ValueError, match="extra"):
        extension.on_env_end((0,), {0: _record(0, [0.0]), 1: _record(1, [0.0])})
    with pytest.raises(ValueError, match="unique"):
        extension.on_env_end((0, 0), {0: _record(0, [0.0])})


def test_scenario_coverage_extension_snapshot_restore_and_config_mismatch() -> None:
    extension = ScenarioCoverageExtension("coverage", commands=[[0.0], [1.0]])
    extension.on_env_end((0, 1), {0: _record(0, [0.0]), 1: _record(1, [2.0])})

    snapshot = extension.snapshot()
    assert ensure_json_value(snapshot) == snapshot
    assert snapshot == {
        "schema": "unigym_bench.scenario_coverage/1",
        "name": "coverage",
        "commands": [[0.0], [1.0]],
        "covered": [True, False],
        "episode_counts": [1, 0],
        "unmatched": 1,
        "episodes": 2,
    }

    clone = ScenarioCoverageExtension("coverage", commands=[[0.0], [1.0]])
    clone.validate_restore(snapshot)
    clone.restore(snapshot)
    assert clone.snapshot() == snapshot
    assert clone.on_run_end(_summary()) == extension.on_run_end(_summary())

    snapshot["covered"][0] = False
    assert extension.snapshot()["covered"] == [True, False]
    clone_again = ScenarioCoverageExtension("coverage", commands=[[0.0], [1.0]])
    clone_again.restore(extension.snapshot())
    snapshot["covered"][0] = False
    assert clone_again.snapshot()["covered"] == [True, False]

    other = ScenarioCoverageExtension("coverage", commands=[[0.5], [1.0]])
    with pytest.raises(ValueError, match="command pool"):
        other.restore(extension.snapshot())

    wrong_schema = ensure_json_value(extension.snapshot())
    wrong_schema["schema"] = "unigym_bench.scenario_coverage/0"
    with pytest.raises(ValueError, match="expected schema"):
        extension.restore(wrong_schema)

    wrong_name = ensure_json_value(extension.snapshot())
    wrong_name["name"] = "other"
    with pytest.raises(ValueError, match="names extension"):
        extension.restore(wrong_name)


def test_step_context_is_not_retained() -> None:
    step_index = torch.zeros(1, dtype=torch.int64)
    context = make_context(
        num_envs=1,
        step_index=step_index,
        values=make_values([1.0]),
    )
    monitor = threshold_config("no_retain", "value:level", threshold=0.5)

    monitor.on_tick(context)
    tensor_ref = weakref.ref(step_index)
    del context, step_index
    gc.collect()

    assert tensor_ref() is None


def test_predicate_snapshot_restore_roundtrip() -> None:
    frames = [_row_context([1.0])]
    original = threshold_config(
        "threshold_rt", "value:level", threshold=0.5, terminal_outcome="failure"
    )
    clone = threshold_config(
        "threshold_rt", "value:level", threshold=0.5, terminal_outcome="failure"
    )

    _assert_roundtrip(original, clone, frames)


def test_hold_snapshot_restore_roundtrip() -> None:
    frames = [_row_context([1.0])]
    original = hold_config(
        "hold_rt", "value:level", threshold=0.5, hold_steps=1, terminal_outcome="success"
    )
    clone = hold_config(
        "hold_rt", "value:level", threshold=0.5, hold_steps=1, terminal_outcome="success"
    )

    _assert_roundtrip(original, clone, frames)


def test_event_counter_snapshot_restore_roundtrip() -> None:
    frames = [_row_context([True], dtype=torch.bool, name="flag")]
    original = event_counter_config("counter_rt", "value:flag")
    clone = event_counter_config("counter_rt", "value:flag")

    _assert_roundtrip(original, clone, frames)


def test_phase_snapshot_restore_roundtrip() -> None:
    frames = [_row_context([0.0], t=0), _row_context([2.0], t=1)]
    original = PhaseMonitor("phase_rt", "value:level", boundaries=[0.0, 1.0])
    clone = PhaseMonitor("phase_rt", "value:level", boundaries=[0.0, 1.0])

    _assert_roundtrip(original, clone, frames)


def test_safety_snapshot_restore_roundtrip() -> None:
    frames = [_row_context([1.0])]
    original = safety_config("safety_rt", "value:level", threshold=0.5, terminal_outcome="failure")
    clone = safety_config("safety_rt", "value:level", threshold=0.5, terminal_outcome="failure")

    _assert_roundtrip(original, clone, frames)


def test_snapshot_and_restore_use_private_copies() -> None:
    monitor = threshold_config("copies", "value:level", threshold=0.5, terminal_outcome="failure")
    monitor.on_tick(_row_context([1.0]))

    snapshot = monitor.snapshot()
    baseline = monitor.snapshot()
    state = snapshot["state"]
    assert isinstance(state, dict)
    fired = state["fired"]
    assert isinstance(fired, list)
    fired[0] = False
    assert monitor.snapshot() == baseline

    clone = threshold_config("copies", "value:level", threshold=0.5, terminal_outcome="failure")
    clone.restore(baseline)
    restored = clone.snapshot()
    baseline["state"]["fired"][0] = False
    assert clone.snapshot() == restored


def test_snapshot_before_first_tick_roundtrips() -> None:
    monitor = threshold_config("empty", "value:level", threshold=0.5)
    snapshot = monitor.snapshot()
    assert snapshot["num_envs"] is None
    assert snapshot["device"] is None

    clone = threshold_config("empty", "value:level", threshold=0.5)
    clone.restore(snapshot)
    assert clone.snapshot() == snapshot


def test_restore_rejects_mismatched_snapshots() -> None:
    monitor = threshold_config("strict", "value:level", threshold=0.0)
    monitor.on_tick(_row_context([0.0]))
    snapshot = monitor.snapshot()

    wrong_name = ensure_json_value(snapshot)
    wrong_name["name"] = "other"
    with pytest.raises(ValueError, match="names monitor"):
        monitor.restore(wrong_name)

    wrong_kind = ensure_json_value(snapshot)
    wrong_kind["kind"] = "phase"
    with pytest.raises(ValueError, match="expected kind"):
        monitor.restore(wrong_kind)

    wrong_schema = ensure_json_value(snapshot)
    wrong_schema["schema"] = "other/3"
    with pytest.raises(ValueError, match="expected schema"):
        monitor.restore(wrong_schema)

    tensor_state = ensure_json_value(snapshot)
    tensor_state["state"] = {"fired": torch.zeros(1)}
    with pytest.raises(TypeError):
        monitor.restore(tensor_state)


def test_restore_rejects_num_envs_mismatch() -> None:
    source = threshold_config("strict", "value:level", threshold=0.0)
    source.on_tick(_row_context([0.0, 0.0]))

    target = threshold_config("strict", "value:level", threshold=0.0)
    target.on_tick(_row_context([0.0]))

    with pytest.raises(ValueError, match="num_envs"):
        target.restore(source.snapshot())


def test_restore_rejects_different_predicate_config() -> None:
    source = threshold_config("cfg", "value:level", threshold=0.5)
    source.on_tick(_row_context([0.0]))
    clone = threshold_config("cfg", "value:level", threshold=0.75)

    with pytest.raises(ValueError, match="config"):
        clone.restore(source.snapshot())

    phase_source = PhaseMonitor("phase_cfg", "value:level", boundaries=[0.0])
    phase_source.on_tick(_row_context([0.0]))
    phase_clone = PhaseMonitor("phase_cfg", "value:other", boundaries=[0.0])
    with pytest.raises(ValueError, match="config"):
        phase_clone.restore(phase_source.snapshot())


def test_failed_restore_leaves_monitor_unchanged_and_usable() -> None:
    monitor = threshold_config("atomic", "value:level", threshold=0.5)
    monitor.on_tick(_row_context([0.0]))
    before = monitor.snapshot()

    bad = ensure_json_value(before)
    state = bad["state"]
    assert isinstance(state, dict)
    state["previous"] = ["not-a-bool"]
    with pytest.raises(TypeError):
        monitor.restore(bad)
    assert monitor.snapshot() == before

    fresh = threshold_config("atomic", "value:level", threshold=0.5)
    with pytest.raises(TypeError):
        fresh.restore(bad)
    fresh_snapshot = fresh.snapshot()
    assert fresh_snapshot["num_envs"] is None
    assert fresh_snapshot["device"] is None

    monitor.on_tick(_row_context([1.0], t=1))
    assert len(monitor.drain_events()) == 1
    fresh.on_tick(_row_context([1.0], t=1))
    assert len(fresh.drain_events()) == 1


def test_on_episode_end_validates_env_ids() -> None:
    monitor = threshold_config("episode", "value:level", threshold=0.0)
    monitor.on_tick(_row_context([0.0]))

    with pytest.raises(ValueError):
        monitor.on_episode_end((1,))
    assert monitor.report_view()["episodes"] == 0

    monitor.on_episode_end((0,))
    assert monitor.report_view()["episodes"] == 1


def test_restore_rejects_out_of_range_pending_drains() -> None:
    source = threshold_config("bounds", "value:level", threshold=0.0, terminal_outcome="failure")
    source.on_tick(_row_context([1.0]))
    snapshot = source.snapshot()

    bad_terminal = ensure_json_value(snapshot)
    terminals = bad_terminal["pending_terminals"]
    assert isinstance(terminals, dict)
    terminals["9"] = terminals["0"]
    clone = threshold_config("bounds", "value:level", threshold=0.0, terminal_outcome="failure")
    with pytest.raises(ValueError, match="outside"):
        clone.restore(bad_terminal)

    bad_event = ensure_json_value(snapshot)
    bad_event["pending_events"] = [
        {
            "name": "off_grid",
            "code": "task.bounds.off_grid",
            "env_ids": [9],
            "severity": "info",
            "data": {},
        }
    ]
    with pytest.raises(ValueError, match="outside"):
        clone.restore(bad_event)


def test_terminal_snapshot_roundtrips_negative_priority_and_extensions() -> None:
    monitor = threshold_config("term_rt", "value:level", threshold=1.0)
    monitor.on_tick(_row_context([0.0]))
    termination = Termination(
        outcome=EpisodeOutcome.FAILURE,
        code="custom.term_rt.external",
        source=TerminationSource.CUSTOM,
        step_index=3,
        time_s=0.25,
        metrics={"value": 1.0},
        extensions={
            "extra": TerminationExtension(namespace="extra", schema_version=1, data={"note": "ok"})
        },
        priority=-2,
    )
    snapshot = ensure_json_value(monitor.snapshot())
    snapshot["pending_terminals"] = {"0": termination.to_json()}

    clone = threshold_config("term_rt", "value:level", threshold=1.0)
    clone.restore(snapshot)
    restored = clone.drain_terminals()[0]

    assert restored == termination
    assert restored.priority == -2
    assert restored.extensions["extra"].data == {"note": "ok"}


def test_event_validation_and_json_roundtrip() -> None:
    event = Event(
        name="threshold_reached",
        code="task.contact.threshold_reached",
        env_ids=(0, 1),
        severity="warning",
        data={"value": 1.0, "items": [1, 2]},
    )
    payload = event.to_json()
    assert ensure_json_value(payload) == payload

    rebuilt = Event.from_json(payload)
    assert rebuilt == event
    payload["data"]["items"].append(3)
    assert rebuilt.data["items"] == [1, 2]

    with pytest.raises(ValueError):
        Event(name="bad", code="not-namespaced", env_ids=(0,))
    with pytest.raises(ValueError):
        Event(name="bad", code="task.bad", env_ids=(0,), severity="critical")
    with pytest.raises(ValueError):
        Event(name="bad", code="task.bad", env_ids=())
    with pytest.raises(ValueError):
        Event(name="bad", code="task.bad", env_ids=(0, 0))
    with pytest.raises(TypeError):
        Event(name="bad", code="task.bad", env_ids=(True,))


_FENCE_SCRIPT = """
import sys

import unigym_bench.monitors as monitors

prefixes = ("unigym.benchmarks", "unigym.runner.evaluation", "unigym.runner.episodes")
blocked = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
)
if blocked:
    raise SystemExit("forbidden core modules imported: " + ", ".join(blocked))
missing = [name for name in ("Probe", "ThresholdMonitor", "HoldMonitor",
                             "EventCounterMonitor", "SafetyMonitor",
                             "ScenarioCoverageMonitor", "TickFrame")
           if hasattr(monitors, name)]
if missing:
    raise SystemExit("retired names still exported: " + ", ".join(missing))
expected = ("event_counter_config", "hold_config", "safety_config", "threshold_config")
if not all(name in monitors.__all__ for name in expected):
    raise SystemExit("configuration helpers missing from __all__")
print("ok")
"""


def test_monitors_fresh_import_has_no_retired_names_or_core_benchmark_scaffolding() -> None:
    src_root = str(Path(unigym_bench.__file__).resolve().parents[1])
    existing = os.environ.get("PYTHONPATH")
    env = dict(os.environ)
    env["PYTHONPATH"] = src_root if not existing else os.pathsep.join([src_root, existing])

    result = subprocess.run(
        [sys.executable, "-c", _FENCE_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == "ok"
