"""Predicate trigger cadences, output grammar and construction validation."""

from __future__ import annotations

import pytest
import torch

from _support import make_context
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
    signals = monitor.values()
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


def test_episode_end_never_fires_on_tick_and_only_refreshes_due_rows() -> None:
    monitor = PredicateMonitor(
        "end_pred",
        _condition(),
        TriggerSpec("episode_end"),
        EmitSpec(value="episode_success"),
    )

    # A tick context without the condition source proves the condition is not
    # evaluated at normal ticks for an episode_end trigger.
    tick_view = EnvView(record={}, obs=None, env_id=None, num_envs=2, device="cpu")
    monitor.on_tick(make_context(num_envs=2, view=tick_view))
    assert monitor.values()["episode_success"].tolist() == [False, False]

    # Due env 0 and non-due env 1 are both true: only the due row may refresh.
    monitor.on_episode_end([0], _context([1.0, 1.0]))

    # The local level refreshes for the due row; the non-due row stays untouched.
    assert monitor.values()["episode_success"].tolist() == [True, False]

    report = monitor.report_view()
    assert report["episode_end_fired"] == 1
    assert report["episode_end_counts"] == [1, 0]
    assert report["fired"] == 0
    assert report["episodes"] == 1

    # A later boundary refreshes the due row without touching a non-due true row.
    monitor.on_episode_end([0], _context([0.0, 1.0]))
    assert monitor.values()["episode_success"].tolist() == [False, False]
    assert monitor.report_view()["episode_end_counts"] == [1, 0]
    assert monitor.report_view()["episodes"] == 2


def test_episode_end_without_final_context_does_not_evaluate() -> None:
    monitor = PredicateMonitor("end_pred", _condition(), TriggerSpec("episode_end"))

    monitor.on_episode_end([0, 1])

    assert monitor.report_view()["episodes"] == 2
    assert monitor.report_view()["episode_end_fired"] == 0


def test_non_episode_end_triggers_do_not_evaluate_at_the_boundary() -> None:
    monitor = PredicateMonitor("tick_pred", _condition(), TriggerSpec("tick"))
    missing_view = EnvView(record={}, obs=None, env_id=None, num_envs=2, device="cpu")

    monitor.on_episode_end([0, 1], make_context(num_envs=2, view=missing_view))

    assert monitor.report_view()["episodes"] == 2
    assert monitor.report_view()["fired"] == 0


def test_repeated_fires_publish_values_without_monitor_outputs() -> None:
    monitor = PredicateMonitor("value_pred", _condition(), TriggerSpec("tick"))

    fired_per_tick: list[list[bool]] = []
    for _ in range(3):
        monitor.on_tick(_context([1.0, 1.0]))
        fired_per_tick.append(monitor.values()["fired"].tolist())

    # The flow, not the monitor, owns terminal decision and conflict accounting.
    assert fired_per_tick == [[True, True], [True, True], [True, True]]
    assert not any(
        callable(getattr(monitor, name, None))
        for name in ("drain_events", "drain_terminals", "drain_episode_terminals")
    )


def test_signals_return_a_fresh_tensor_each_call() -> None:
    monitor = PredicateMonitor("fresh_pred", _condition(), TriggerSpec("tick"))
    monitor.on_tick(_context([1.0, 0.0]))

    first = monitor.values()["fired"]
    first[0] = False
    second = monitor.values()["fired"]

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


def test_emit_spec_accepts_only_a_value_name() -> None:
    with pytest.raises(ValueError, match="value"):
        EmitSpec(value="Bad")
    with pytest.raises(TypeError):
        EmitSpec(event="not-a-name")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="unknown"):
        EmitSpec.from_mapping({"value": "fired", "extra": 1})
    with pytest.raises(ValueError, match="monitor emit has moved"):
        EmitSpec.from_mapping({"event": "hit"})
    with pytest.raises(ValueError, match="monitor emit has moved"):
        EmitSpec.from_mapping({"terminal_reason": "hit", "terminal_outcome": "failure"})

    spec = EmitSpec(value="due")
    assert EmitSpec.from_mapping(spec.to_mapping()) == spec


def test_predicate_monitor_supports_string_and_mapping_configs() -> None:
    monitor = PredicateMonitor(
        "config_pred",
        {"source": "record:x", "op": "gt", "value": 0.0},
        "edge",
    )

    assert _tick(monitor, [1.0]) == [True]
    assert monitor.report_view()["trigger"] == {"kind": "edge"}
