"""Condition DSL: comparisons, per-env AND and the closed mapping parser."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from _support import make_context
from unigym.evaluation import EnvStepContext, EnvView
from unigym.values import StepValues
from unigym_bench.conditions import (
    All,
    Comparison,
    Condition,
    ConditionError,
    ConditionParseError,
    TemporalOperatorError,
    parse_condition,
)
from unigym_bench.sources import SourceRef, SourceResolver

_RESOLVER = SourceResolver()


def _context(
    *,
    record: dict[str, object] | None = None,
    values: StepValues | None = None,
    num_envs: int = 2,
) -> EnvStepContext:
    view = EnvView(
        record={} if record is None else record,
        obs=None,
        env_id=None,
        num_envs=num_envs,
        device="cpu",
    )
    return make_context(num_envs=num_envs, view=view, values=values)


def _record_comparison(
    records: Mapping[str, torch.Tensor],
    *,
    key: str,
    op: str = "gt",
    value: float = 0.0,
    dim: int | None = None,
    num_envs: int | None = None,
) -> torch.Tensor:
    if num_envs is None:
        num_envs = int(next(iter(records.values())).shape[0])
    context = _context(record=dict(records), num_envs=num_envs)
    condition = Comparison(SourceRef.parse(f"record:{key}"), op, value, dim)  # type: ignore[arg-type]
    return condition.evaluate(context, _RESOLVER)


@pytest.mark.parametrize(
    ("op", "value", "expected"),
    [
        ("gt", 0.5, [True, False, False]),
        ("lt", 0.5, [False, True, True]),
        ("gte", 1.0, [True, False, False]),
        ("lte", -2.0, [False, True, False]),
        ("eq", 0.0, [False, False, True]),
        ("abs_gt", 1.5, [False, True, False]),
        ("abs_lt", 0.5, [False, False, True]),
        ("abs_gte", 2.0, [False, True, False]),
        ("abs_lte", 1.0, [True, False, True]),
    ],
)
def test_comparison_operators(op: str, value: float, expected: list[bool]) -> None:
    result = _record_comparison(
        {"level": torch.tensor([1.0, -2.0, 0.0])},
        key="level",
        op=op,
        value=value,
    )

    assert result.dtype == torch.bool
    assert result.device == torch.device("cpu")
    assert result.shape == (3,)
    assert result.tolist() == expected


def test_comparison_dim_indexes_the_trailing_axis() -> None:
    positions = torch.tensor([[0.0, 5.0, -1.0], [0.0, -5.0, 3.0]])

    high_z = _record_comparison({"pos": positions}, key="pos", op="gt", value=0.0, dim=2)
    high_z_negative = _record_comparison({"pos": positions}, key="pos", op="gt", value=0.0, dim=-1)
    high_x = _record_comparison({"pos": positions}, key="pos", op="gt", value=0.0, dim=0)

    assert high_z.tolist() == [False, True]
    assert torch.equal(high_z, high_z_negative)
    assert high_x.tolist() == [False, False]


def test_comparison_dim_none_requires_one_dimensional_source() -> None:
    with pytest.raises(ConditionError, match="1-D"):
        _record_comparison({"pos": torch.zeros(2, 3)}, key="pos", op="gt", value=0.0)


def test_comparison_dim_requires_two_dimensional_source() -> None:
    with pytest.raises(ConditionError, match="2-D"):
        _record_comparison({"level": torch.zeros(2)}, key="level", op="gt", value=0.0, dim=0)
    with pytest.raises(ConditionError, match="2-D"):
        _record_comparison(
            {"volume": torch.zeros(2, 3, 4)}, key="volume", op="gt", value=0.0, dim=2
        )


@pytest.mark.parametrize("dim", [3, -4])
def test_comparison_dim_out_of_range(dim: int) -> None:
    with pytest.raises(ConditionError, match="outside the trailing axis"):
        _record_comparison({"pos": torch.zeros(2, 3)}, key="pos", op="gt", value=0.0, dim=dim)


def test_comparison_shape_and_leading_axis() -> None:
    with pytest.raises(ConditionError, match="leading dimension"):
        _record_comparison({"level": torch.zeros(3)}, key="level", num_envs=2)

    context = _context(record={"level": torch.zeros(2, 2)})
    condition = Comparison(SourceRef.parse("record:level"), "gt", 0.0, 1)
    assert condition.evaluate(context, _RESOLVER).shape == (2,)


@pytest.mark.parametrize(
    ("dtype", "raw", "expected"),
    [
        (torch.bool, [True, False, True], [True, False, True]),
        (torch.int64, [1, 0, -2], [True, False, False]),
        (torch.float32, [1.0, 0.0, -2.0], [True, False, False]),
    ],
)
def test_comparison_accepts_bool_integer_and_float_sources(
    dtype: torch.dtype, raw: list[object], expected: list[bool]
) -> None:
    result = _record_comparison(
        {"level": torch.tensor(raw, dtype=dtype)}, key="level", op="gt", value=0.0
    )

    assert result.dtype == torch.bool
    assert result.tolist() == expected


def test_comparison_rejects_complex_sources() -> None:
    with pytest.raises(ConditionError, match="complex"):
        _record_comparison(
            {"level": torch.tensor([1 + 0j, 2 + 0j])}, key="level", op="gt", value=0.0
        )


def test_comparison_rejects_a_foreign_device() -> None:
    with pytest.raises(ConditionError, match="device"):
        _record_comparison({"level": torch.zeros(2, device="meta")}, key="level")


def test_comparison_rejects_json_sources() -> None:
    context = _context(values=StepValues({"meta": {"nested": 1}}))
    condition = Comparison(SourceRef.parse("value:meta"), "eq", 1.0)

    with pytest.raises(ConditionError, match="JSON"):
        condition.evaluate(context, _RESOLVER)


def test_comparison_construction_validation() -> None:
    ref = SourceRef.parse("record:level")

    assert isinstance(Comparison(ref, "gt", 1), Condition)
    assert isinstance(All([Comparison(ref, "gt", 1)]), Condition)

    with pytest.raises(TypeError, match="SourceRef"):
        Comparison("record:level", "gt", 1)  # type: ignore[arg-type]
    with pytest.raises(ConditionError, match="Comparison.op"):
        Comparison(ref, "ge", 1)  # type: ignore[arg-type]
    with pytest.raises(ConditionError, match="Comparison.op"):
        Comparison(ref, ["gt"], 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Comparison.value"):
        Comparison(ref, "gt", True)
    with pytest.raises(TypeError, match="Comparison.value"):
        Comparison(ref, "gt", "1")  # type: ignore[arg-type]
    with pytest.raises(ConditionError, match="representable"):
        Comparison(ref, "gt", 10**400)
    with pytest.raises(ConditionError, match="finite"):
        Comparison(ref, "gt", float("nan"))
    with pytest.raises(TypeError, match="Comparison.dim"):
        Comparison(ref, "gt", 1, True)
    with pytest.raises(TypeError, match="Comparison.dim"):
        Comparison(ref, "gt", 1, "1")  # type: ignore[arg-type]


def test_all_is_per_env_logical_and() -> None:
    context = _context(
        record={
            "a": torch.tensor([1.0, 3.0, -1.0]),
            "b": torch.tensor([2.0, 0.0, 4.0]),
        },
        num_envs=3,
    )
    condition = All(
        [
            Comparison(SourceRef.parse("record:a"), "gt", 0.0),
            Comparison(SourceRef.parse("record:b"), "gt", 0.0),
        ]
    )

    result = condition.evaluate(context, _RESOLVER)

    assert result.dtype == torch.bool
    assert result.device == context.device
    assert result.tolist() == [True, False, False]


def test_all_has_no_history() -> None:
    condition = All([Comparison(SourceRef.parse("record:a"), "gt", 0.0)])
    true_step = _context(record={"a": torch.tensor([1.0, 1.0])})
    false_step = _context(record={"a": torch.tensor([-1.0, -1.0])})

    assert condition.evaluate(true_step, _RESOLVER).tolist() == [True, True]
    assert condition.evaluate(false_step, _RESOLVER).tolist() == [False, False]
    assert condition.evaluate(true_step, _RESOLVER).tolist() == [True, True]


def test_all_construction_validation() -> None:
    ref = SourceRef.parse("record:a")
    comparison = Comparison(ref, "gt", 0.0)

    with pytest.raises(ConditionError, match="at least one"):
        All([])
    with pytest.raises(TypeError, match="sequence"):
        All("not conditions")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Condition"):
        All([comparison, object()])  # type: ignore[list-item]


class _BadCondition:
    """Condition-shaped object with a deliberately invalid result."""

    def __init__(self, result: object) -> None:
        self.result = result

    def evaluate(self, step: EnvStepContext, resolver: SourceResolver) -> object:
        return self.result


@pytest.mark.parametrize(
    ("result", "match"),
    [
        ([True, False], "torch.Tensor"),
        (torch.tensor([1.0, 0.0]), "torch.bool"),
        (torch.zeros(3, dtype=torch.bool), "shape"),
        (torch.zeros(2, dtype=torch.bool, device="meta"), "device"),
    ],
)
def test_all_rejects_bad_child_results(result: object, match: str) -> None:
    condition = All([_BadCondition(result)])
    context = _context()

    with pytest.raises(ConditionError, match=match):
        condition.evaluate(context, _RESOLVER)


def test_parse_leaf_and_evaluate() -> None:
    condition = parse_condition({"source": "record:pos", "op": "gte", "value": 0.5, "dim": 2})
    context = _context(record={"pos": torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])})

    assert isinstance(condition, Comparison)
    assert condition.evaluate(context, _RESOLVER).tolist() == [True, False]

    without_dim = parse_condition({"source": "record:level", "op": "gt", "value": 0.0})
    level_context = _context(record={"level": torch.tensor([1.0, -1.0])})
    assert without_dim.evaluate(level_context, _RESOLVER).tolist() == [True, False]


def test_parse_all_and_nested_all() -> None:
    document = {
        "all": [
            {"source": "record:a", "op": "gt", "value": 0.0},
            {
                "all": [
                    {"source": "record:a", "op": "lt", "value": 2.0},
                    {"source": "value:flag", "op": "eq", "value": 1.0},
                ]
            },
        ]
    }
    condition = parse_condition(document)
    context = _context(
        record={"a": torch.tensor([1.0, 3.0])},
        values=StepValues({"flag": torch.tensor([1.0, 1.0])}),
    )

    assert isinstance(condition, All)
    assert len(condition.conditions) == 2
    assert condition.evaluate(context, _RESOLVER).tolist() == [True, False]


@pytest.mark.parametrize(
    "document",
    [
        {"source": "record:a", "op": "gt", "value": 0.0, "axis": 1},
        {"source": "record:a", "op": "gt", "value": 0.0, "extra": 1},
        {"all": [{"source": "record:a", "op": "gt", "value": 0.0}], "extra": 1},
    ],
)
def test_parse_rejects_unknown_keys(document: dict[str, object]) -> None:
    with pytest.raises(ConditionParseError, match="unknown keys"):
        parse_condition(document)


def test_parse_rejects_unknown_operator() -> None:
    with pytest.raises(ConditionParseError, match="not supported"):
        parse_condition({"source": "record:a", "op": "ge", "value": 0.0})


@pytest.mark.parametrize("missing", ["source", "op", "value"])
def test_parse_requires_leaf_keys(missing: str) -> None:
    document = {"source": "record:a", "op": "gt", "value": 0.0}
    del document[missing]

    with pytest.raises(ConditionParseError, match="missing required keys"):
        parse_condition(document)


@pytest.mark.parametrize("source", ["a", "record", "record:", "record/level", "timer/a", 3])
def test_parse_rejects_bad_source(source: object) -> None:
    with pytest.raises(ConditionParseError):
        parse_condition({"source": source, "op": "gt", "value": 0.0})


@pytest.mark.parametrize("value", [True, "1", None, [1], {"v": 1}])
def test_parse_rejects_non_numeric_values(value: object) -> None:
    with pytest.raises(ConditionParseError, match="finite number"):
        parse_condition({"source": "record:a", "op": "gt", "value": value})


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_parse_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ConditionParseError, match="finite"):
        parse_condition({"source": "record:a", "op": "gt", "value": value})


def test_parse_normalizes_a_huge_value_to_a_parse_error() -> None:
    with pytest.raises(ConditionParseError, match="representable"):
        parse_condition({"source": "record:a", "op": "gt", "value": 10**400})


@pytest.mark.parametrize("dim", [True, "2", 1.0, None])
def test_parse_rejects_bad_dim(dim: object) -> None:
    with pytest.raises(ConditionParseError, match="'dim'"):
        parse_condition({"source": "record:a", "op": "gt", "value": 0.0, "dim": dim})


@pytest.mark.parametrize("document", [None, [], "condition", 3])
def test_parse_rejects_non_mapping(document: object) -> None:
    with pytest.raises(ConditionParseError, match="must be a mapping"):
        parse_condition(document)


@pytest.mark.parametrize("raw", [[], "conditions", {"source": "record:a"}])
def test_parse_all_requires_a_non_empty_list(raw: object) -> None:
    with pytest.raises(ConditionParseError):
        parse_condition({"all": raw})


@pytest.mark.parametrize(
    "operator",
    ["always", "never", "fraction_true", "hold_until_end"],
)
def test_temporal_operators_are_reserved(operator: str) -> None:
    with pytest.raises(TemporalOperatorError) as excinfo:
        parse_condition({operator: {"source": "record:a", "op": "gt", "value": 0.0}})

    message = str(excinfo.value)
    assert operator in message
    assert "reserved" in message
    assert "not implemented" in message
    assert isinstance(excinfo.value, NotImplementedError)
    assert isinstance(excinfo.value, ConditionParseError)


def test_nested_temporal_operators_are_reserved() -> None:
    document = {
        "all": [
            {"source": "record:a", "op": "gt", "value": 0.0},
            {"never": {"source": "record:a", "op": "lt", "value": 0.0}},
        ]
    }

    with pytest.raises(TemporalOperatorError, match="'never'"):
        parse_condition(document)
