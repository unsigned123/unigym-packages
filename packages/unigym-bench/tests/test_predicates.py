"""Predicate conditions resolve record, obs and value sources without history."""

from __future__ import annotations

import pytest
import torch

from _support import make_context
from unigym.env.obs import ObservationSchema, ObservationSpec, ObsSnapshot
from unigym.evaluation import EnvStepContext, EnvView
from unigym.values import StepValues
from unigym_bench.conditions import (
    All,
    Comparison,
    ConditionError,
    TemporalOperatorError,
    parse_condition,
)
from unigym_bench.json import ensure_json_value
from unigym_bench.monitors import PredicateMonitor
from unigym_bench.predicates import TriggerSpec
from unigym_bench.sources import (
    SourceRef,
    SourceResolveError,
    SourceUnavailableError,
)


def _observation(values: dict[str, torch.Tensor], num_envs: int) -> ObsSnapshot:
    entries = {
        key: ObservationSpec(key, tuple(value.shape[1:]), "f32") for key, value in values.items()
    }
    return ObsSnapshot(
        values,
        schema=ObservationSchema(entries),
        num_envs=num_envs,
        device="cpu",
    )


def _context(
    *,
    record: dict[str, object] | None = None,
    obs: ObsSnapshot | None = None,
    values: StepValues | None = None,
    num_envs: int = 2,
) -> EnvStepContext:
    view = EnvView(
        record={} if record is None else record,
        obs=obs,
        env_id=None,
        num_envs=num_envs,
        device="cpu",
    )
    return make_context(num_envs=num_envs, view=view, values=values)


def _ref(source: str) -> SourceRef:
    return SourceRef.parse(source)


def _fired(monitor: PredicateMonitor, context: EnvStepContext) -> list[bool]:
    monitor.on_tick(context)
    return monitor.values()["fired"].tolist()


def test_record_source_condition_fires() -> None:
    monitor = PredicateMonitor(
        "record_pred",
        Comparison(_ref("record:height"), "gt", 0.5),
        TriggerSpec("tick"),
    )
    context = _context(record={"height": torch.tensor([0.6, 0.4])})

    assert _fired(monitor, context) == [True, False]


def test_obs_source_condition_fires() -> None:
    monitor = PredicateMonitor(
        "obs_pred",
        Comparison(_ref("obs:joint_pos"), "gte", 1.0),
        TriggerSpec("tick"),
    )
    obs = _observation({"joint_pos": torch.tensor([1.0, 2.0])}, num_envs=2)
    context = _context(obs=obs)

    assert _fired(monitor, context) == [True, True]


def test_value_source_condition_fires() -> None:
    monitor = PredicateMonitor(
        "value_pred",
        Comparison(_ref("value:flag"), "gt", 0.0),
        TriggerSpec("tick"),
    )
    context = _context(values=StepValues({"flag": torch.tensor([1.0, 0.0])}))

    assert _fired(monitor, context) == [True, False]


def test_json_value_source_is_not_comparable() -> None:
    monitor = PredicateMonitor(
        "json_pred",
        Comparison(_ref("value:meta"), "eq", 1.0),
        TriggerSpec("tick"),
    )
    context = _context(values=StepValues({"meta": {"nested": [1, 2]}}))

    with pytest.raises(ConditionError, match="JSON"):
        monitor.on_tick(context)


def test_missing_source_is_a_teaching_error() -> None:
    monitor = PredicateMonitor(
        "missing_pred",
        Comparison(_ref("record:absent"), "gt", 0.0),
        TriggerSpec("tick"),
    )
    context = _context(record={"present": torch.tensor([1.0, 1.0])})

    with pytest.raises(SourceResolveError, match="available record keys"):
        monitor.on_tick(context)


def test_query_source_stays_unavailable() -> None:
    monitor = PredicateMonitor(
        "query_pred",
        Comparison(_ref("query:pose"), "gt", 0.0),
        TriggerSpec("tick"),
    )

    with pytest.raises(SourceUnavailableError, match="QuerySourceRegistry"):
        monitor.on_tick(_context())


def test_all_is_logical_and_at_one_boundary_not_history() -> None:
    condition = All(
        [
            Comparison(_ref("record:a"), "gt", 0.0),
            Comparison(_ref("record:b"), "gt", 0.0),
        ]
    )
    monitor = PredicateMonitor("all_pred", condition, TriggerSpec("tick"))

    first = _context(record={"a": torch.tensor([1.0, 1.0]), "b": torch.tensor([1.0, 0.0])})
    second = _context(record={"a": torch.tensor([0.0, 1.0]), "b": torch.tensor([1.0, 1.0])})

    assert _fired(monitor, first) == [True, False]
    assert _fired(monitor, second) == [False, True]


def test_condition_to_mapping_roundtrips_through_the_parser() -> None:
    condition = All(
        [
            Comparison(_ref("record:position"), "gt", 0.25, dim=2),
            Comparison(_ref("value:flag"), "eq", 1.0),
        ]
    )
    mapping = condition.to_mapping()

    assert ensure_json_value(mapping) == mapping
    assert parse_condition(mapping) == condition
    assert _ref("record:position").to_mapping() == {"kind": "record", "key": "position"}


def test_temporal_operators_remain_reserved() -> None:
    with pytest.raises(TemporalOperatorError, match="reserved"):
        parse_condition({"always": [{"source": "record:x", "op": "gt", "value": 0.0}]})


def test_predicate_report_is_json_safe() -> None:
    monitor = PredicateMonitor(
        "report_pred",
        Comparison(_ref("record:x"), "gt", 0.0),
        TriggerSpec("hold", steps=2),
    )
    monitor.on_tick(_context(record={"x": torch.tensor([1.0, 1.0])}))

    report = monitor.report_view()

    assert report["trigger"] == {"kind": "hold", "steps": 2}
    assert report["fired"] == 0
    assert report["fired_counts"] == [0, 0]
    assert ensure_json_value(report) == report
