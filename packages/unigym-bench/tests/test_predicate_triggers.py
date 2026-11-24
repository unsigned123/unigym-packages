"""Predicate trigger cadences, output grammar and construction validation."""

from __future__ import annotations

import pytest
import torch

from _support import make_context
from unigym.core.episode_termination import TerminationSource
from unigym.evaluation import EnvView
from unigym_bench.conditions import Comparison
from unigym_bench.json import ensure_json_value
from unigym_bench.monitors import PredicateMonitor
from unigym_bench.predicates import EmitSpec, TriggerError, TriggerSpec
from unigym_bench.sources import SourceRef


def _condition(key: str = "x", *, op: str = "gt", value: float = 0.0) -> Comparison:
    return Comparison(SourceRef.parse(f"record:{key}"), op, value)  # type: ignore[arg-type]


def _context(row: list[float], *, key: str = "x", num_envs: int | None = None):
    if num_envs is None:
        num_envs = len(row)
    view = EnvView(
        record={key: torch.tensor(row, dtype=torch.float32)},
        obs=None,
        env_id=None,
        num_envs=num_envs,
        device="cpu",
    )
    return make_context(num_envs=num_envs, view=view)


def _tick(monitor: PredicateMonitor, row: list[float]) -> list[bool]:
    monitor.on_tick(_context(row))
    signals = monitor.signals()
    assert list(signals) == ["fired"]
    return signals["fired"].tolist()


def test_tick_fires_on_every_true_tick_per_env() -> None:
    monitor = PredicateMonitor("tick_pred", _condition(), TriggerSpec("tick"))

    assert _tick(monitor, [1.0, 0.0]) == [True, False]
    assert _tick(monitor, [0.0, 1.0]) == [False, True]
    assert _tick(monitor, [1.0, 1.0]) == [True, True]

    report = monitor.report_view()
    assert report["fired"] == 4
    assert report["fired_counts"] == [2, 2]
    assert report["episode_end_fired"] == 0
    assert ensure_json_value(report) == report


def test_edge_fires_on_first_true_and_later_rising_edges() -> None:
    monitor = PredicateMonitor("edge_pred", _condition(), TriggerSpec("edge"))

    rows = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
    fired = [_tick(monitor, row) for row in rows]

    assert fired == [
        [True, False],
        [False, True],
        [False, False],
        [True, False],
    ]
    assert monitor.report_view()["fired_counts"] == [2, 1]


def test_hold_requires_consecutive_ticks_and_rearms_after_false() -> None:
    monitor = PredicateMonitor(
        "hold_pred",
        _condition(),
        TriggerSpec("hold", steps=2),
    )

    rows = [[1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    fired = [_tick(monitor, row) for row in rows]

    assert fired == [
        [False, False],
        [True, False],
        [False, True],
        [False, False],
        [False, False],
        [True, False],
    ]
    assert monitor.report_view()["fired_counts"] == [2, 1]


def test_hold_reset_rearms_the_streak() -> None:
    monitor = PredicateMonitor("hold_pred", _condition(), TriggerSpec("hold", steps=2))

    assert _tick(monitor, [1.0]) == [False]
    assert _tick(monitor, [1.0]) == [True]
    assert _tick(monitor, [1.0]) == [False]

    monitor.reset([0])
    assert _tick(monitor, [1.0]) == [False]
    assert _tick(monitor, [1.0]) == [True]
    assert monitor.report_view()["fired_counts"] == [2]


def test_once_fires_only_the_first_true_per_episode_and_reset_rearms() -> None:
    monitor = PredicateMonitor("once_pred", _condition(), TriggerSpec("once"))

    rows = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
    fired = [_tick(monitor, row) for row in rows]

    assert fired == [
        [True, False],
        [False, True],
        [False, False],
        [False, False],
    ]

    monitor.reset([0])
    assert _tick(monitor, [1.0, 0.0]) == [True, False]
    assert monitor.report_view()["fired_counts"] == [2, 1]


def test_episode_end_never_fires_on_tick_and_only_scores_due_rows() -> None:
    monitor = PredicateMonitor(
        "end_pred",
        _condition(),
        TriggerSpec("episode_end"),
        EmitSpec(
            event="episode_success",
            severity="warning",
            terminal_reason="score",
            terminal_outcome="success",
        ),
    )

    # A tick context without the condition source proves the condition is not
    # evaluated at normal ticks for an episode_end trigger.
    tick_view = EnvView(record={}, obs=None, env_id=None, num_envs=2, device="cpu")
    monitor.on_tick(make_context(num_envs=2, view=tick_view))
    assert monitor.signals()["fired"].tolist() == [False, False]
    assert tuple(monitor.drain_events()) == ()
    assert dict(monitor.drain_terminals()) == {}

    # Due env 0 and non-due env 1 are both true: only the due row may score.
    monitor.on_episode_end([0], _context([1.0, 1.0]))

    events = monitor.drain_events()
    assert len(events) == 1
    assert events[0].name == "episode_success"
    assert events[0].code == "task.end_pred.episode_success"
    assert events[0].severity == "warning"
    assert events[0].env_ids == (0,)
    assert events[0].data == {"trigger": "episode_end"}

    # The episode-end terminal is report/scoring-only and drained separately.
    episode_terminals = monitor.drain_episode_terminals()
    assert sorted(episode_terminals) == [0]
    assert episode_terminals[0].code == "custom.end_pred.score"
    assert episode_terminals[0].source is TerminationSource.CUSTOM
    assert dict(monitor.drain_episode_terminals()) == {}
    assert dict(monitor.drain_terminals()) == {}

    # The local level refreshes for the due row; the non-due row stays untouched.
    assert monitor.signals()["fired"].tolist() == [True, False]

    report = monitor.report_view()
    assert report["episode_end_fired"] == 1
    assert report["episode_end_counts"] == [1, 0]
    assert report["fired"] == 0
    assert report["episodes"] == 1

    # A later boundary refreshes the due row without scoring a non-due true row.
    monitor.on_episode_end([0], _context([0.0, 1.0]))
    assert dict(monitor.drain_episode_terminals()) == {}
    assert monitor.signals()["fired"].tolist() == [False, False]
    assert monitor.report_view()["episode_end_counts"] == [1, 0]
    assert monitor.report_view()["episodes"] == 2


def test_episode_end_without_final_context_does_not_evaluate() -> None:
    monitor = PredicateMonitor("end_pred", _condition(), TriggerSpec("episode_end"))

    monitor.on_episode_end([0, 1])

    assert monitor.report_view()["episodes"] == 2
    assert monitor.report_view()["episode_end_fired"] == 0
    assert tuple(monitor.drain_events()) == ()


def test_non_episode_end_triggers_do_not_evaluate_at_the_boundary() -> None:
    monitor = PredicateMonitor("tick_pred", _condition(), TriggerSpec("tick"))
    missing_view = EnvView(record={}, obs=None, env_id=None, num_envs=2, device="cpu")

    monitor.on_episode_end([0, 1], make_context(num_envs=2, view=missing_view))

    assert monitor.report_view()["episodes"] == 2
    assert monitor.report_view()["fired"] == 0


def test_terminal_requests_repeat_on_every_fire_without_a_monitor_latch() -> None:
    monitor = PredicateMonitor(
        "terminal_pred",
        _condition(),
        TriggerSpec("tick"),
        EmitSpec(
            event="hit",
            severity="violation",
            terminal_reason="hit",
            terminal_outcome="failure",
        ),
    )

    terminal_drains: list[list[int]] = []
    event_drains: list[int] = []
    for _ in range(3):
        monitor.on_tick(_context([1.0, 1.0]))
        drained = monitor.drain_terminals()
        terminal_drains.append(sorted(drained))
        assert all(value.code == "custom.terminal_pred.hit" for value in drained.values())
        assert all(value.source is TerminationSource.CUSTOM for value in drained.values())
        event_drains.append(len(monitor.drain_events()))

    # The engine, not the monitor, owns first-terminal conflict accounting.
    assert terminal_drains == [[0, 1], [0, 1], [0, 1]]
    assert event_drains == [1, 1, 1]


def test_events_use_pack_code_severity_and_grouped_env_ids() -> None:
    monitor = PredicateMonitor(
        "event_pred",
        _condition(),
        TriggerSpec("tick"),
        EmitSpec(event="fired", severity="fatal"),
    )

    monitor.on_tick(_context([1.0, 0.0, 1.0]))
    events = monitor.drain_events()

    assert len(events) == 1
    assert events[0].code == "task.event_pred.fired"
    assert events[0].severity == "fatal"
    assert events[0].env_ids == (0, 2)
    assert monitor.signals()["fired"].tolist() == [True, False, True]


def test_signals_return_a_fresh_tensor_each_call() -> None:
    monitor = PredicateMonitor("fresh_pred", _condition(), TriggerSpec("tick"))
    monitor.on_tick(_context([1.0, 0.0]))

    first = monitor.signals()["fired"]
    first[0] = False
    second = monitor.signals()["fired"]

    assert second.tolist() == [True, False]
    assert first is not second
    assert second.dtype == torch.bool


@pytest.mark.parametrize(
    ("kind", "match"),
    [
        ("level", "level"),
        ("always", "reserved"),
        ("never", "reserved"),
        ("fraction_true", "reserved"),
        ("hold_until_end", "reserved"),
        ("bogus", "not supported"),
    ],
)
def test_trigger_rejects_level_temporal_and_unknown_kinds(kind: str, match: str) -> None:
    with pytest.raises(TriggerError, match=match):
        TriggerSpec(kind=kind)  # type: ignore[arg-type]


def test_trigger_hold_steps_and_closed_mapping_validation() -> None:
    with pytest.raises(TriggerError, match="steps"):
        TriggerSpec("hold")
    with pytest.raises(ValueError, match=">= 1"):
        TriggerSpec("hold", steps=0)
    with pytest.raises(TypeError):
        TriggerSpec("hold", steps=True)  # type: ignore[arg-type]
    with pytest.raises(TriggerError, match="steps"):
        TriggerSpec("edge", steps=2)

    assert TriggerSpec.from_mapping({"kind": "hold", "steps": 3}) == TriggerSpec("hold", 3)
    assert TriggerSpec("hold", 3).to_mapping() == {"kind": "hold", "steps": 3}

    with pytest.raises(TriggerError, match="unknown"):
        TriggerSpec.from_mapping({"kind": "edge", "extra": 1})
    with pytest.raises(TriggerError, match="missing"):
        TriggerSpec.from_mapping({"steps": 2})
    with pytest.raises(TriggerError, match="object"):
        TriggerSpec.from_mapping(["edge"])


def test_emit_spec_validates_names_and_terminal_pairs() -> None:
    with pytest.raises(ValueError, match="value"):
        EmitSpec(value="Bad")
    with pytest.raises(ValueError, match="event"):
        EmitSpec(event="not-a-name")
    with pytest.raises(ValueError, match="together"):
        EmitSpec(terminal_reason="hit")
    with pytest.raises(ValueError, match="together"):
        EmitSpec(terminal_outcome="success")
    with pytest.raises(ValueError, match="unknown episode outcome"):
        EmitSpec(terminal_reason="hit", terminal_outcome="bogus")
    with pytest.raises(ValueError, match="unknown"):
        EmitSpec.from_mapping({"value": "fired", "extra": 1})

    spec = EmitSpec(
        value="due",
        event="passed",
        severity="warning",
        terminal_reason="score",
        terminal_outcome="success",
    )
    assert EmitSpec.from_mapping(spec.to_mapping()) == spec


def test_predicate_monitor_supports_string_and_mapping_configs() -> None:
    monitor = PredicateMonitor(
        "config_pred",
        {"source": "record:x", "op": "gt", "value": 0.0},
        "edge",
        {"event": "crossed", "severity": "info"},
    )

    assert _tick(monitor, [1.0]) == [True]
    assert monitor.report_view()["trigger"] == {"kind": "edge"}
