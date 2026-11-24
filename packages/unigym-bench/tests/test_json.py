"""Strict JSON normalization: acceptance, normalization and rejection paths."""

from __future__ import annotations

import math
from types import MappingProxyType

import numpy as np
import pytest
import torch

from unigym_bench import JsonValue, ensure_json_value


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (True, True),
        (False, False),
        (0, 0),
        (-3, -3),
        (0.5, 0.5),
        (-1.25, -1.25),
        ("text", "text"),
        ([], []),
        ((), []),
        ([1, "a", None], [1, "a", None]),
        ((1, (2, 3)), [1, [2, 3]]),
        ({"a": 1}, {"a": 1}),
        ({"a": (1, {"b": [2.5]})}, {"a": [1, {"b": [2.5]}]}),
        ({"": "empty key"}, {"": "empty key"}),
    ],
)
def test_accepts_and_normalizes_json_values(value: object, expected: JsonValue) -> None:
    assert ensure_json_value(value) == expected


def test_normalizes_without_returning_aliases() -> None:
    source = {"items": [1, 2]}
    normalized = ensure_json_value(source)
    assert normalized == source
    assert normalized is not source
    assert isinstance(normalized, dict)
    assert normalized["items"] is not source["items"]


def test_accepts_read_only_mappings() -> None:
    assert ensure_json_value(MappingProxyType({"a": 1})) == {"a": 1}


@pytest.mark.parametrize(
    ("value", "where"),
    [
        (math.inf, "root"),
        (-math.inf, "root"),
        (math.nan, "root"),
        ({"good": 1, "bad": math.nan}, "root.bad"),
        ([1, math.inf], "root[1]"),
    ],
)
def test_rejects_non_finite_floats_with_path(value: object, where: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        ensure_json_value(value, where="root")
    assert where in str(excinfo.value)


@pytest.mark.parametrize(
    ("value", "where"),
    [
        (object(), "root"),
        (np.zeros(2), "root"),
        (np.float64(1.0), "root"),
        (torch.zeros(2), "root"),
        ({"a", "b"}, "root"),
        ({1: "x"}, "root"),
        ({"good": 1, "other": object()}, "root.other"),
        ({"a": {2: "nested"}}, "root.a"),
        ([{"x": object()}], "root[0].x"),
    ],
)
def test_rejects_non_json_types_with_path(value: object, where: str) -> None:
    with pytest.raises(TypeError) as excinfo:
        ensure_json_value(value, where="root")
    assert where in str(excinfo.value)


def test_default_root_path_is_value() -> None:
    with pytest.raises(TypeError) as excinfo:
        ensure_json_value(object())
    assert str(excinfo.value).startswith("value:")
