"""Monitor protocol, built-in behavior and rewind state."""

from __future__ import annotations

import gc
import math
import os
import subprocess
import sys
import weakref
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
import torch

import unigym_bench

from _support import (
    bool_probe,
    constant_probe,
    make_frame,
    make_view,
    series_probe,
    table_probe,
)
from unigym.core.episode_termination import (
    EpisodeOutcome,
    Termination,
    TerminationExtension,
    TerminationSource,
)
from unigym_bench.events import Event
from unigym_bench.frames import TickFrame
from unigym_bench.json import ensure_json_value
from unigym_bench.monitors import (
    BaseMonitor,
    EventCounterMonitor,
    HoldMonitor,
    Monitor,
    PhaseMonitor,
    RewindableMonitor,
    SafetyMonitor,
    ScenarioCoverageMonitor,
    ThresholdMonitor,
)


def _assert_roundtrip(monitor: object, clone: object, frames: Sequence[TickFrame]) -> None:
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


def test_base_monitor_defaults() -> None:
    monitor = BaseMonitor("base_monitor")

    assert isinstance(monitor, Monitor)
    assert not isinstance(monitor, RewindableMonitor)
    assert monitor.name == "base_monitor"
    assert dict(monitor.signals()) == {}
    assert tuple(monitor.drain_events()) == ()
    assert dict(monitor.drain_terminals()) == {}

    frame = make_frame()
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


def test_builtins_satisfy_protocols_and_json_reports() -> None:
    monitors = [
        ThresholdMonitor("threshold", constant_probe([0.0]), threshold=0.5),
        HoldMonitor("hold", constant_probe([0.0]), threshold=0.5, hold_steps=2),
        EventCounterMonitor("counter", bool_probe([False])),
        PhaseMonitor("phase", constant_probe([0.0]), boundaries=[0.0]),
        SafetyMonitor("safety", constant_probe([0.0]), threshold=0.5),
        ScenarioCoverageMonitor("coverage", commands=[[0.0], [1.0]]),
    ]

    for monitor in monitors:
        assert isinstance(monitor, Monitor)
        assert isinstance(monitor, RewindableMonitor)
        report = monitor.report_view()
        assert ensure_json_value(report) == report


def test_threshold_emits_once_per_rising_edge() -> None:
    levels = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    monitor = ThresholdMonitor("contact", table_probe(levels), threshold=0.5)

    monitor.on_tick(make_frame(t=0))
    assert tuple(monitor.drain_events()) == ()

    monitor.on_tick(make_frame(t=1))
    events = monitor.drain_events()
    assert len(events) == 1
    event = events[0]
    assert event.name == "threshold_reached"
    assert event.code == "task.contact.threshold_reached"
    assert event.env_ids == (0,)
    assert event.severity == "info"
    assert event.data["value"] == 1.0

    monitor.on_tick(make_frame(t=2))
    assert tuple(monitor.drain_events()) == ()

    monitor.on_tick(make_frame(t=3))
    events = monitor.drain_events()
    assert [event.env_ids for event in events] == [(1,)]

    signals = monitor.signals()
    assert set(signals) == {"level", "active"}
    assert signals["level"].shape == (2,)
    assert signals["level"].device == torch.device("cpu")
    assert signals["active"].tolist() == [False, True]

    second = monitor.signals()
    assert second is not signals
    assert second["level"] is not signals["level"]


def test_threshold_terminal_is_custom_and_one_shot() -> None:
    monitor = ThresholdMonitor(
        "contact",
        constant_probe([1.0]),
        threshold=0.5,
        operator=">",
        terminal_outcome="failure",
    )

    monitor.on_tick(make_frame(num_envs=1))
    terminals = monitor.drain_terminals()
    assert set(terminals) == {0}
    terminal = terminals[0]
    assert terminal.outcome is EpisodeOutcome.FAILURE
    assert terminal.source is TerminationSource.CUSTOM
    assert terminal.code == "custom.contact.threshold"
    assert terminal.code.startswith("custom.")
    assert terminal.step_index is None
    assert terminal.time_s is None
    assert terminal.metrics["value"] == 1.0

    monitor.on_tick(make_frame(num_envs=1, t=1))
    assert dict(monitor.drain_terminals()) == {}


def test_threshold_reset_clears_edge_state() -> None:
    monitor = ThresholdMonitor(
        "contact",
        constant_probe([1.0]),
        threshold=0.0,
        terminal_outcome="success",
    )

    monitor.on_tick(make_frame(num_envs=1))
    assert len(monitor.drain_events()) == 1
    assert len(monitor.drain_terminals()) == 1

    monitor.reset([0])
    monitor.on_tick(make_frame(num_envs=1, t=1))
    assert len(monitor.drain_events()) == 1
    assert len(monitor.drain_terminals()) == 1


@pytest.mark.parametrize(
    ("probe", "error"),
    [
        (lambda frame: [1.0, 2.0], TypeError),
        (lambda frame: torch.tensor(1.0), ValueError),
        (lambda frame: torch.zeros(2, 1), ValueError),
        (lambda frame: torch.zeros(3), ValueError),
        (lambda frame: torch.zeros(2, dtype=torch.bool), TypeError),
        (lambda frame: torch.zeros(2, device="meta"), ValueError),
        (lambda frame: torch.tensor([math.nan, 0.0]), ValueError),
        (lambda frame: torch.tensor([math.inf, 0.0]), ValueError),
    ],
)
def test_float_probe_validates_type_shape_and_device(
    probe: Callable[[TickFrame], object], error: type[BaseException]
) -> None:
    monitor = ThresholdMonitor("probe_check", probe, threshold=0.0)

    with pytest.raises(error):
        monitor.on_tick(make_frame(num_envs=2))


def test_probe_and_operator_validation() -> None:
    with pytest.raises(TypeError):
        ThresholdMonitor("bad", 7, threshold=0.0)
    with pytest.raises(ValueError):
        ThresholdMonitor("bad", constant_probe([0.0]), threshold=math.nan)
    with pytest.raises(TypeError):
        ThresholdMonitor("bad", constant_probe([0.0]), threshold=True)
    with pytest.raises(ValueError):
        ThresholdMonitor("bad", constant_probe([0.0]), threshold=0.0, operator="=")
    with pytest.raises(ValueError):
        ThresholdMonitor(
            "bad",
            constant_probe([0.0]),
            threshold=0.0,
            terminal_outcome="engine",
        )


def test_monitor_rejects_changed_num_envs() -> None:
    monitor = ThresholdMonitor("stable", constant_probe([0.0, 0.0]), threshold=0.0)
    monitor.on_tick(make_frame(num_envs=2))

    with pytest.raises(ValueError, match="num_envs changed"):
        monitor.on_tick(make_frame(num_envs=1))


def test_hold_requires_consecutive_ticks() -> None:
    rows = [[1.0], [1.0], [0.0], [1.0], [1.0], [1.0]]
    monitor = HoldMonitor("hold", series_probe(rows), threshold=0.5, hold_steps=2)
    event_counts = []
    streaks = []
    for t in range(len(rows)):
        monitor.on_tick(make_frame(num_envs=1, t=t))
        event_counts.append(len(monitor.drain_events()))
        streaks.append(int(monitor.signals()["streak"][0]))

    assert event_counts == [0, 1, 0, 0, 1, 0]
    assert streaks == [1, 2, 0, 1, 2, 3]
    assert monitor.signals()["holding"].tolist() == [True]

    monitor.reset([0])
    assert monitor.signals()["streak"].tolist() == [0]
    assert monitor.signals()["holding"].tolist() == [False]


def test_hold_rearms_after_reset() -> None:
    monitor = HoldMonitor(
        "hold",
        constant_probe([1.0]),
        threshold=0.5,
        hold_steps=2,
        terminal_outcome="success",
    )

    monitor.on_tick(make_frame(num_envs=1))
    assert len(monitor.drain_events()) == 0
    monitor.on_tick(make_frame(num_envs=1, t=1))
    assert len(monitor.drain_events()) == 1
    assert len(monitor.drain_terminals()) == 1

    monitor.reset([0])
    monitor.on_tick(make_frame(num_envs=1, t=2))
    assert len(monitor.drain_events()) == 0
    monitor.on_tick(make_frame(num_envs=1, t=3))
    assert len(monitor.drain_events()) == 1
    assert len(monitor.drain_terminals()) == 1


def test_hold_validates_hold_steps() -> None:
    for value in (0, -1, True):
        with pytest.raises((TypeError, ValueError)):
            HoldMonitor("bad", constant_probe([0.0]), threshold=0.0, hold_steps=value)


def test_event_counter_counts_rising_edges_and_is_report_only() -> None:
    state = {"row": [False, False]}

    def probe(frame: TickFrame) -> torch.Tensor:
        return torch.tensor(state["row"], dtype=torch.bool, device=frame.device)

    monitor = EventCounterMonitor("occurrences", probe, severity="warning")

    def tick(row: list[bool], t: int) -> None:
        state["row"] = row
        monitor.on_tick(make_frame(t=t))

    tick([False, False], 0)
    tick([True, True], 1)
    tick([True, False], 2)
    tick([False, True], 3)
    tick([True, True], 4)

    assert monitor.signals()["count"].tolist() == [2, 2]
    assert tuple(monitor.drain_events()) == ()
    assert dict(monitor.drain_terminals()) == {}

    monitor.reset([0])
    tick([True, True], 5)
    assert monitor.signals()["count"].tolist() == [3, 2]
    report = monitor.report_view()
    assert report["occurrences"] == 5
    assert report["severity"] == "warning"


def test_event_counter_rejects_bad_severity() -> None:
    with pytest.raises(ValueError):
        EventCounterMonitor("bad", bool_probe([False]), severity="critical")


def test_phase_tracks_transitions_and_initializes_silently() -> None:
    values = [-1.0, 0.5, 2.0, 0.5, -2.0]
    monitor = PhaseMonitor(
        "phase",
        table_probe([[value] for value in values]),
        boundaries=[0.0, 1.0],
    )

    events_per_tick = []
    phases = []
    for t in range(len(values)):
        monitor.on_tick(make_frame(num_envs=1, t=t))
        events_per_tick.append(len(monitor.drain_events()))
        phases.append(int(monitor.signals()["phase"][0]))

    assert events_per_tick == [0, 1, 1, 1, 1]
    assert phases == [0, 1, 2, 1, 0]
    assert monitor.signals()["phase"].dtype == torch.int64
    assert monitor.signals()["level"].dtype == torch.float32

    report = monitor.report_view()
    assert report["transitions"] == 4
    assert report["phase_ticks"] == {"0": 2, "1": 2, "2": 1}

    monitor.reset([0])
    monitor.on_tick(make_frame(num_envs=1, t=0))
    assert tuple(monitor.drain_events()) == ()


def test_phase_names_and_snapshot() -> None:
    monitor = PhaseMonitor(
        "phase",
        constant_probe([0.5]),
        boundaries=[0.0, 1.0],
        phase_names=["low", "mid", "high"],
    )
    monitor.on_tick(make_frame(num_envs=1))
    monitor.on_episode_end((0,))

    report = monitor.report_view()
    assert report["phase_names"] == ["low", "mid", "high"]
    assert report["episode_phases"] == {"low": 0, "mid": 1, "high": 0}

    clone = PhaseMonitor(
        "phase",
        constant_probe([0.5]),
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
        PhaseMonitor("bad", constant_probe([0.0]), **kwargs)


def test_safety_emits_severity_and_counts_violations() -> None:
    values = [0.9, 1.0, 0.1, 0.9]
    monitor = SafetyMonitor(
        "safety",
        table_probe([[value] for value in values]),
        threshold=0.8,
        operator=">",
        severity="fatal",
        terminal_outcome="failure",
    )

    events_per_tick = []
    terminals_per_tick = []
    for t in range(len(values)):
        monitor.on_tick(make_frame(num_envs=1, t=t))
        events = monitor.drain_events()
        events_per_tick.append(len(events))
        if events:
            assert all(event.severity == "fatal" for event in events)
            assert all(event.code == "task.safety.violation" for event in events)
        terminals_per_tick.append(len(monitor.drain_terminals()))

    assert events_per_tick == [1, 0, 0, 1]
    assert terminals_per_tick == [1, 0, 0, 1]
    assert monitor.signals()["violating"].tolist() == [True]

    report = monitor.report_view()
    assert report["violation_events"] == 2
    assert report["violation_ticks"] == 3
    assert report["severity"] == "fatal"

    monitor.reset([0])
    assert monitor.signals()["violating"].tolist() == [False]


def test_scenario_coverage_tracks_start_commands() -> None:
    monitor = ScenarioCoverageMonitor("coverage", commands=[[0.0, 0.0], [1.0, 1.0]])

    commands = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    monitor.on_tick(make_frame(num_envs=2, cmd=commands))
    assert dict(monitor.signals()) == {}
    monitor.on_episode_end((0, 1))

    report = monitor.report_view()
    assert report["pool_size"] == 2
    assert report["covered"] == 2
    assert report["coverage_ratio"] == 1.0
    assert report["episode_counts"] == [1, 1]
    assert report["unmatched_episodes"] == 0
    assert report["episodes"] == 2

    monitor.reset([0, 1])
    next_commands = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    monitor.on_tick(make_frame(num_envs=2, t=1, cmd=next_commands))
    monitor.on_episode_end((0, 1))

    report = monitor.report_view()
    assert report["covered"] == 2
    assert report["episode_counts"] == [2, 1]
    assert report["unmatched_episodes"] == 1
    assert report["episodes"] == 4
    assert tuple(monitor.drain_events()) == ()
    assert dict(monitor.drain_terminals()) == {}


def test_scenario_coverage_without_tick_is_unmatched() -> None:
    monitor = ScenarioCoverageMonitor("coverage", commands=[[0.0]])
    monitor.on_episode_end((0,))

    report = monitor.report_view()
    assert report["unmatched_episodes"] == 1
    assert report["episodes"] == 1


@pytest.mark.parametrize(
    "commands",
    [
        [],
        [[0.0], [0.0]],
        [[0.0], [1.0, 2.0]],
        [[math.nan]],
        [[1e39]],
    ],
)
def test_scenario_coverage_validates_pool(commands: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ScenarioCoverageMonitor("bad", commands=commands)


@pytest.mark.parametrize(
    ("cmd", "error"),
    [
        (None, ValueError),
        (torch.zeros(2), ValueError),
        (torch.zeros(2, 3), ValueError),
        (torch.zeros(2, 2, dtype=torch.int64), TypeError),
        (torch.zeros(2, 2, device="meta"), ValueError),
        (torch.tensor([[math.nan, 0.0], [0.0, 0.0]]), ValueError),
    ],
)
def test_scenario_coverage_validates_observed_command(
    cmd: torch.Tensor | None, error: type[BaseException]
) -> None:
    monitor = ScenarioCoverageMonitor("coverage", commands=[[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(error):
        monitor.on_tick(make_frame(num_envs=2, cmd=cmd))


def test_frame_is_not_retained() -> None:
    view = make_view(1)
    frame = make_frame(num_envs=1, view=view)
    monitor = ThresholdMonitor("no_retain", constant_probe([1.0]), threshold=0.5)

    monitor.on_tick(frame)
    frame_ref = weakref.ref(frame)
    del frame
    del view
    gc.collect()

    assert frame_ref() is None


def test_threshold_snapshot_restore_roundtrip() -> None:
    frames = [make_frame(num_envs=1)]
    original = ThresholdMonitor(
        "threshold_rt", constant_probe([1.0]), threshold=0.5, terminal_outcome="failure"
    )
    clone = ThresholdMonitor(
        "threshold_rt", constant_probe([1.0]), threshold=0.5, terminal_outcome="failure"
    )

    _assert_roundtrip(original, clone, frames)


def test_hold_snapshot_restore_roundtrip() -> None:
    frames = [make_frame(num_envs=1)]
    original = HoldMonitor(
        "hold_rt", constant_probe([1.0]), threshold=0.5, hold_steps=1, terminal_outcome="success"
    )
    clone = HoldMonitor(
        "hold_rt", constant_probe([1.0]), threshold=0.5, hold_steps=1, terminal_outcome="success"
    )

    _assert_roundtrip(original, clone, frames)


def test_event_counter_snapshot_restore_roundtrip() -> None:
    frames = [make_frame(num_envs=1)]
    original = EventCounterMonitor("counter_rt", bool_probe([True]))
    clone = EventCounterMonitor("counter_rt", bool_probe([True]))

    _assert_roundtrip(original, clone, frames)


def test_phase_snapshot_restore_roundtrip() -> None:
    levels = [0.0, 2.0]
    frames = [make_frame(num_envs=1, t=0), make_frame(num_envs=1, t=1)]
    probe = table_probe([[level] for level in levels])
    original = PhaseMonitor("phase_rt", probe, boundaries=[0.0, 1.0])
    clone = PhaseMonitor("phase_rt", probe, boundaries=[0.0, 1.0])

    _assert_roundtrip(original, clone, frames)


def test_safety_snapshot_restore_roundtrip() -> None:
    frames = [make_frame(num_envs=1)]
    original = SafetyMonitor(
        "safety_rt", constant_probe([1.0]), threshold=0.5, terminal_outcome="failure"
    )
    clone = SafetyMonitor(
        "safety_rt", constant_probe([1.0]), threshold=0.5, terminal_outcome="failure"
    )

    _assert_roundtrip(original, clone, frames)


def test_scenario_coverage_snapshot_restore_roundtrip() -> None:
    commands = torch.tensor([[0.0], [1.0]])
    probe_commands = ScenarioCoverageMonitor("coverage_rt", commands=[[0.0], [1.0]])
    clone = ScenarioCoverageMonitor("coverage_rt", commands=[[0.0], [1.0]])

    probe_commands.on_tick(make_frame(num_envs=2, cmd=commands))
    snapshot = probe_commands.snapshot()
    clone.restore(snapshot)
    assert clone.snapshot() == snapshot

    probe_commands.on_episode_end((0, 1))
    clone.on_episode_end((0, 1))
    assert clone.report_view() == probe_commands.report_view()


def test_snapshot_and_restore_use_private_copies() -> None:
    monitor = ThresholdMonitor(
        "copies", constant_probe([1.0]), threshold=0.5, terminal_outcome="failure"
    )
    monitor.on_tick(make_frame(num_envs=1))

    snapshot = monitor.snapshot()
    baseline = monitor.snapshot()
    state = snapshot["state"]
    assert isinstance(state, dict)
    active = state["active"]
    assert isinstance(active, list)
    active[0] = False
    assert monitor.snapshot() == baseline

    clone = ThresholdMonitor(
        "copies", constant_probe([1.0]), threshold=0.5, terminal_outcome="failure"
    )
    clone.restore(baseline)
    restored = clone.snapshot()
    baseline["state"]["active"][0] = False
    assert clone.snapshot() == restored


def test_snapshot_before_first_tick_roundtrips() -> None:
    monitor = ThresholdMonitor("empty", constant_probe([0.0]), threshold=0.5)
    snapshot = monitor.snapshot()
    assert snapshot["num_envs"] is None
    assert snapshot["device"] is None

    clone = ThresholdMonitor("empty", constant_probe([0.0]), threshold=0.5)
    clone.restore(snapshot)
    assert clone.snapshot() == snapshot


def test_restore_rejects_mismatched_snapshots() -> None:
    monitor = ThresholdMonitor("strict", constant_probe([0.0]), threshold=0.0)
    monitor.on_tick(make_frame(num_envs=1))
    snapshot = monitor.snapshot()

    wrong_name = ensure_json_value(snapshot)
    wrong_name["name"] = "other"
    with pytest.raises(ValueError, match="names monitor"):
        monitor.restore(wrong_name)

    wrong_kind = ensure_json_value(snapshot)
    wrong_kind["kind"] = "hold"
    with pytest.raises(ValueError, match="expected kind"):
        monitor.restore(wrong_kind)

    wrong_schema = ensure_json_value(snapshot)
    wrong_schema["schema"] = "other/1"
    with pytest.raises(ValueError, match="expected schema"):
        monitor.restore(wrong_schema)

    tensor_state = ensure_json_value(snapshot)
    tensor_state["state"] = {"active": torch.zeros(1)}
    with pytest.raises(TypeError):
        monitor.restore(tensor_state)


def test_restore_rejects_num_envs_mismatch() -> None:
    source = ThresholdMonitor("strict", constant_probe([0.0, 0.0]), threshold=0.0)
    source.on_tick(make_frame(num_envs=2))

    target = ThresholdMonitor("strict", constant_probe([0.0]), threshold=0.0)
    target.on_tick(make_frame(num_envs=1))

    with pytest.raises(ValueError, match="num_envs"):
        target.restore(source.snapshot())


def test_restore_rejects_different_threshold_config() -> None:
    source = ThresholdMonitor("cfg", constant_probe([0.0]), threshold=0.5)
    source.on_tick(make_frame(num_envs=1))
    clone = ThresholdMonitor("cfg", constant_probe([0.0]), threshold=0.75)

    with pytest.raises(ValueError, match="config"):
        clone.restore(source.snapshot())


def test_restore_rejects_different_terminal_config() -> None:
    source = ThresholdMonitor(
        "cfg", constant_probe([0.0]), threshold=0.5, terminal_outcome="failure"
    )
    source.on_tick(make_frame(num_envs=1))
    clone = ThresholdMonitor("cfg", constant_probe([0.0]), threshold=0.5)

    with pytest.raises(ValueError, match="config"):
        clone.restore(source.snapshot())


def test_restore_rejects_different_coverage_pool() -> None:
    commands = torch.tensor([[0.0], [1.0]])
    source = ScenarioCoverageMonitor("cfg", commands=[[0.0], [1.0]])
    source.on_tick(make_frame(num_envs=2, cmd=commands))
    clone = ScenarioCoverageMonitor("cfg", commands=[[0.0], [2.0]])

    with pytest.raises(ValueError, match="config"):
        clone.restore(source.snapshot())


def test_failed_restore_leaves_monitor_unchanged_and_usable() -> None:
    row = {"value": [0.0]}

    def probe(frame: TickFrame) -> torch.Tensor:
        return torch.tensor(row["value"], dtype=torch.float32, device=frame.device)

    monitor = ThresholdMonitor("atomic", probe, threshold=0.5)
    monitor.on_tick(make_frame(num_envs=1))
    before = monitor.snapshot()

    bad = ensure_json_value(before)
    state = bad["state"]
    assert isinstance(state, dict)
    state["active"] = ["not-a-bool"]
    with pytest.raises(TypeError):
        monitor.restore(bad)
    assert monitor.snapshot() == before

    fresh = ThresholdMonitor("atomic", probe, threshold=0.5)
    with pytest.raises(TypeError):
        fresh.restore(bad)
    fresh_snapshot = fresh.snapshot()
    assert fresh_snapshot["num_envs"] is None
    assert fresh_snapshot["device"] is None

    row["value"] = [1.0]
    monitor.on_tick(make_frame(num_envs=1, t=1))
    assert len(monitor.drain_events()) == 1
    fresh.on_tick(make_frame(num_envs=1, t=1))
    assert len(fresh.drain_events()) == 1


def test_on_episode_end_validates_env_ids() -> None:
    monitor = ThresholdMonitor("episode", constant_probe([0.0]), threshold=0.0)
    monitor.on_tick(make_frame(num_envs=1))

    with pytest.raises(ValueError):
        monitor.on_episode_end((1,))
    assert monitor.report_view()["episodes"] == 0

    monitor.on_episode_end((0,))
    assert monitor.report_view()["episodes"] == 1


def test_restore_rejects_out_of_range_pending_drains() -> None:
    source = ThresholdMonitor(
        "bounds", constant_probe([1.0]), threshold=0.0, terminal_outcome="failure"
    )
    source.on_tick(make_frame(num_envs=1))
    snapshot = source.snapshot()

    bad_terminal = ensure_json_value(snapshot)
    terminals = bad_terminal["pending_terminals"]
    assert isinstance(terminals, dict)
    terminals["9"] = terminals["0"]
    clone = ThresholdMonitor(
        "bounds", constant_probe([1.0]), threshold=0.0, terminal_outcome="failure"
    )
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
    monitor = ThresholdMonitor("term_rt", constant_probe([0.0]), threshold=1.0)
    monitor.on_tick(make_frame(num_envs=1))
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

    clone = ThresholdMonitor("term_rt", constant_probe([0.0]), threshold=1.0)
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


def test_tick_frame_validates_tensor_contracts() -> None:
    with pytest.raises(TypeError):
        make_frame(num_envs=2, step_index=torch.zeros(2))
    with pytest.raises(ValueError):
        make_frame(num_envs=2, step_index=torch.zeros(3, dtype=torch.int64))
    with pytest.raises(TypeError):
        make_frame(num_envs=2, time_s=torch.zeros(2, dtype=torch.int64))
    with pytest.raises(ValueError):
        make_frame(num_envs=2, cmd=torch.zeros(3, 1))
    with pytest.raises(ValueError):
        make_frame(num_envs=2, reset_state={2: {"a": 1}})
    with pytest.raises(TypeError):
        make_frame(num_envs=2, reset_state={True: {"a": 1}})
    with pytest.raises(TypeError):
        make_frame(num_envs=2, reset_state={0: {"tensor": torch.zeros(1)}})


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
print(monitors.ThresholdMonitor.__name__)
"""


def test_monitors_import_no_core_benchmark_scaffolding() -> None:
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
    assert result.stdout.strip() == "ThresholdMonitor"
