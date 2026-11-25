"""Predicate timeout snapshots, atomic refusal and reset/re-arm behavior."""

from __future__ import annotations

import json

import pytest
import torch

from _support import make_context
from unigym.evaluation import EnvView
from unigym_bench.conditions import Comparison
from unigym_bench.json import ensure_json_value
from unigym_bench.monitors import PredicateMonitor
from unigym_bench.predicates import EmitSpec, TriggerSpec
from unigym_bench.sources import SourceRef


def _condition() -> Comparison:
    return Comparison(SourceRef.parse("record:x"), "gt", 0.0)


def _context(row: list[float]):
    num_envs = len(row)
    view = EnvView(
        record={"x": torch.tensor(row, dtype=torch.float32)},
        obs=None,
        env_id=None,
        num_envs=num_envs,
        device="cpu",
    )
    return make_context(num_envs=num_envs, view=view)


def _monitor(trigger: TriggerSpec | None = None) -> PredicateMonitor:
    return PredicateMonitor(
        "pred",
        _condition(),
        TriggerSpec("edge") if trigger is None else trigger,
        EmitSpec(value="fired"),
    )


def _snapshot(monitor: PredicateMonitor) -> dict[str, object]:
    state = monitor.snapshot()
    assert ensure_json_value(state) == state
    return state  # type: ignore[return-value]


def test_snapshot_roundtrip_restores_state_and_values() -> None:
    source = _monitor()
    source.on_tick(_context([1.0, 0.0]))
    snapshot = _snapshot(source)
    assert json.loads(json.dumps(snapshot)) == snapshot
    assert set(snapshot) == {"schema", "name", "kind", "num_envs", "device", "config", "state"}

    clone = _monitor()
    clone.restore(snapshot)

    assert clone.snapshot() == snapshot
    assert clone.report_view() == source.report_view()
    assert clone.values()["fired"].tolist() == source.values()["fired"].tolist()


def test_snapshot_carries_episode_end_counts() -> None:
    source = _monitor(TriggerSpec("episode_end"))
    source.on_tick(_context([0.0, 0.0]))
    source.on_episode_end([0, 1], _context([1.0, 0.0]))
    snapshot = _snapshot(source)
    assert snapshot["state"]["episode_end_counts"] == [1, 0]
    assert "episode_terminals" not in snapshot["state"]

    clone = _monitor(TriggerSpec("episode_end"))
    clone.restore(snapshot)

    assert clone.snapshot() == snapshot
    assert clone.report_view()["episode_end_fired"] == 1
    assert clone.report_view()["episodes"] == 2


def test_refused_restore_is_zero_change_for_config_and_state() -> None:
    source = _monitor()
    source.on_tick(_context([1.0, 0.0]))
    snapshot = _snapshot(source)

    target = _monitor(TriggerSpec("hold", steps=2))
    target.on_tick(_context([1.0, 0.0]))
    before = target.snapshot()
    assert target.values()["fired"].tolist() == [False, False]

    with pytest.raises(ValueError, match="config does not match"):
        target.restore(snapshot)
    assert target.snapshot() == before
    assert target.values()["fired"].tolist() == [False, False]

    twin = _monitor()
    twin.on_tick(_context([0.0, 1.0]))
    twin_before = twin.snapshot()
    bad_state = ensure_json_value(snapshot)
    bad_state["state"]["previous"] = "not a bool list"
    with pytest.raises(TypeError):
        twin.restore(bad_state)
    assert twin.snapshot() == twin_before


def test_validate_restore_does_not_mutate() -> None:
    source = _monitor()
    source.on_tick(_context([1.0, 0.0]))
    snapshot = _snapshot(source)

    target = _monitor()
    target.on_tick(_context([0.0, 1.0]))
    before = target.snapshot()

    target.validate_restore(snapshot)

    assert target.snapshot() == before


def test_reset_rearms_the_edge_after_restore() -> None:
    source = _monitor()
    source.on_tick(_context([1.0]))
    snapshot = _snapshot(source)

    clone = _monitor()
    clone.restore(snapshot)
    assert clone.values()["fired"].tolist() == [True]

    clone.reset([0])
    clone.on_tick(_context([0.0]))
    assert clone.values()["fired"].tolist() == [False]
    clone.on_tick(_context([1.0]))
    assert clone.values()["fired"].tolist() == [True]


def test_reset_clears_level_but_keeps_run_counters() -> None:
    monitor = _monitor(TriggerSpec("once"))
    monitor.on_tick(_context([1.0, 1.0]))
    assert monitor.values()["fired"].tolist() == [True, True]
    assert monitor.report_view()["fired_counts"] == [1, 1]

    monitor.reset([0, 1])

    assert monitor.values()["fired"].tolist() == [False, False]
    assert monitor.report_view()["fired_counts"] == [1, 1]
    monitor.on_tick(_context([1.0, 0.0]))
    assert monitor.values()["fired"].tolist() == [True, False]
    assert monitor.report_view()["fired_counts"] == [2, 1]
