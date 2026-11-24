"""Strict JSON normalization owned by the pack.

``unigym`` exposes a ``JsonValue`` type alias for raw config trees before
validation.  The pack accepts plain JSON data at its report and state
boundaries, so this module provides the runtime counterpart: it normalizes
builtin scalars, sequences and string-keyed mappings into plain ``dict`` /
``list`` trees and rejects everything else with the exact path that failed.

Accepted: ``None``, ``bool``, ``int``, finite ``float``, ``str``, ``list``,
``tuple`` and ``Mapping`` with string keys.  Builtin scalar types are matched
exactly, so tensors, numpy arrays and numpy scalars are rejected rather than
silently coerced.  Values are copied, never returned by reference.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TypeAlias

__all__ = ["JsonScalar", "JsonValue", "ensure_json_value"]

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def ensure_json_value(value: object, *, where: str = "value") -> JsonValue:
    """Validate and normalize ``value`` into a plain JSON tree.

    Args:
        value: The candidate value.
        where: Human-readable path used in error messages for the root value.

    Returns:
        A newly allocated JSON tree with tuples converted to lists and mappings
        converted to dictionaries.

    Raises:
        TypeError: ``value`` is not a supported JSON type or a mapping key is
            not a string.
        ValueError: A float is not finite.
    """
    return _normalize(value, where)


def _normalize(value: object, where: str) -> JsonValue:
    if value is None:
        return None
    if type(value) is bool:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{where}: float must be finite, got {value!r}")
        return value
    if type(value) is str:
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(
                    f"{where}: mapping keys must be strings, " f"got {key!r} ({type(key).__name__})"
                )
            normalized[key] = _normalize(item, f"{where}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize(item, f"{where}[{index}]") for index, item in enumerate(value)]
    raise TypeError(
        f"{where}: expected a JSON value (None, bool, int, finite float, str, "
        f"list, tuple or string-keyed mapping), got {type(value).__name__}"
    )
