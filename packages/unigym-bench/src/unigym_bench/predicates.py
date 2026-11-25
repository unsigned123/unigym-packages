"""Predicate configuration specs shared by the monitor and task composition.

A predicate is one condition tree plus a trigger cadence and a monitor-local
value name.  This module owns the two frozen, JSON-safe configuration objects;
:class:`unigym_bench.monitors.PredicateMonitor` owns the stateful runtime.
Keeping the specs separate lets the task-document composer parse and fingerprint
a predicate without importing the monitor implementation.

A monitor publishes observations only.  Events, severities and terminals are
decisions owned by the benchmark flow; the task document expresses them in its
``flow`` section.

Trigger semantics (v1, per environment):

``tick``
    Fire on every true evaluation.  No cross-tick state.
``edge``
    Fire on a false-to-true rising edge.  The first observed row starts from
    ``previous=False``, so a true first tick fires.
``hold``
    Fire once when ``steps`` consecutive true evaluations have accumulated;
    a false evaluation clears the streak and re-arms the hold.
``once``
    Fire on the first true evaluation of an episode; later true rows are
    silent until the episode resets.
``episode_end``
    Never fires on a normal tick.  At the episode boundary it inspects the
    supplied final context and updates the value only for due rows.  A value is
    an observation; it never schedules a runner terminal.

``level`` and the reserved temporal operators (``always``, ``never``,
``fraction_true``, ``hold_until_end``) are rejected with a teaching error.
``all`` belongs to the condition grammar and is a logical AND, not a temporal
quantifier.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from .json import JsonValue

__all__ = [
    "EmitSpec",
    "TriggerError",
    "TriggerKind",
    "TriggerSpec",
]

#: Trigger cadences implemented in pack v1.
TriggerKind: TypeAlias = Literal["tick", "edge", "hold", "once", "episode_end"]

_TRIGGER_KINDS: tuple[TriggerKind, ...] = ("tick", "edge", "hold", "once", "episode_end")
_TRIGGER_KIND_SET: frozenset[str] = frozenset(_TRIGGER_KINDS)
_RESERVED_TEMPORAL: tuple[str, ...] = (
    "always",
    "never",
    "fraction_true",
    "hold_until_end",
)

_TRIGGER_KEYS: frozenset[str] = frozenset({"kind", "steps"})
_EMIT_KEYS: frozenset[str] = frozenset({"value"})
_LEGACY_EMIT_KEYS: frozenset[str] = frozenset(
    {"event", "severity", "terminal_reason", "terminal_outcome"}
)
_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")


class TriggerError(ValueError):
    """A trigger specification is unknown, reserved or malformed."""


@dataclass(frozen=True, slots=True)
class TriggerSpec:
    """Frozen, JSON-safe trigger cadence for a predicate monitor.

    Args:
        kind: One of ``tick``, ``edge``, ``hold``, ``once`` or ``episode_end``.
        steps: Required for ``hold`` and forbidden for every other kind.  It is
            the number of consecutive true evaluations required to fire.
    """

    kind: TriggerKind
    steps: int | None = None

    def __post_init__(self) -> None:
        """Validate the kind and the hold step count at construction."""
        kind = self.kind
        if not isinstance(kind, str):
            raise TriggerError(
                f"trigger kind must be a string, got {type(kind).__name__}; "
                f"expected one of {list(_TRIGGER_KINDS)}"
            )
        if kind in _RESERVED_TEMPORAL:
            raise TriggerError(
                f"temporal trigger {kind!r} is reserved and not implemented in pack v1; "
                "express episode-wide meaning with an explicit condition or use "
                f"one of {list(_TRIGGER_KINDS)}"
            )
        if kind == "level":
            raise TriggerError(
                "trigger 'level' is not implemented in pack v1; use 'tick' to fire on "
                "every true evaluation (a level is the condition mask itself, not a "
                "trigger cadence)"
            )
        if kind not in _TRIGGER_KIND_SET:
            raise TriggerError(
                f"trigger kind {kind!r} is not supported; expected one of "
                f"{list(_TRIGGER_KINDS)}"
            )
        if kind == "hold":
            if self.steps is None:
                raise TriggerError(
                    "trigger 'hold' requires a positive 'steps' count; write steps=<int >= 1>"
                )
            object.__setattr__(
                self, "steps", _require_positive_int(self.steps, "TriggerSpec.steps")
            )
            return
        if self.steps is not None:
            raise TriggerError(
                f"trigger {kind!r} does not accept 'steps'; steps is only valid for "
                "the 'hold' trigger"
            )

    @classmethod
    def from_mapping(cls, mapping: object) -> TriggerSpec:
        """Build a trigger from the closed ``{"kind": ..., "steps": ...}`` form.

        Args:
            mapping: A mapping with the required ``kind`` and optional ``steps``.

        Returns:
            The validated trigger.

        Raises:
            TriggerError: The mapping is not an object, has unknown or missing
                keys, or fails the constructor validation.
        """
        if not isinstance(mapping, Mapping):
            raise TriggerError(f"trigger mapping must be an object, got {type(mapping).__name__}")
        unknown = sorted(repr(key) for key in mapping if key not in _TRIGGER_KEYS)
        if unknown:
            raise TriggerError(
                f"trigger mapping has unknown keys {unknown}; expected keys "
                f"{sorted(_TRIGGER_KEYS)}"
            )
        if "kind" not in mapping:
            raise TriggerError(
                f"trigger mapping is missing required 'kind'; expected keys "
                f"{sorted(_TRIGGER_KEYS)}"
            )
        kind = mapping["kind"]
        if not isinstance(kind, str):
            raise TriggerError(
                f"trigger mapping 'kind' must be a string, got {type(kind).__name__}"
            )
        steps = mapping.get("steps")
        # ``kind`` and ``steps`` are validated by the constructor.
        return cls(kind=cast(TriggerKind, kind), steps=cast(int | None, steps))

    def to_mapping(self) -> dict[str, JsonValue]:
        """Return the closed, JSON-safe config form."""
        mapping: dict[str, JsonValue] = {"kind": self.kind}
        if self.steps is not None:
            mapping["steps"] = self.steps
        return mapping


@dataclass(frozen=True, slots=True)
class EmitSpec:
    """Local value name for one monitor.

    A monitor publishes observations only.  Events, severities and terminals
    are decisions owned by the benchmark flow, not by the monitor; the task
    document expresses them in its ``flow`` section.

    Args:
        value: Local level value name published by ``values()``.  Defaults to
            ``fired``.
    """

    value: str = "fired"

    def __post_init__(self) -> None:
        """Validate the local value name."""
        _require_name(self.value, "EmitSpec.value")

    @classmethod
    def from_mapping(cls, mapping: object) -> EmitSpec:
        """Build an emit spec from its closed JSON-safe mapping.

        Args:
            mapping: A mapping with the optional ``value`` key.

        Returns:
            The validated emit spec.

        Raises:
            ValueError: The mapping has unknown keys or an invalid value.
            TypeError: The mapping is not an object.
        """
        if not isinstance(mapping, Mapping):
            raise TypeError(f"emit mapping must be an object, got {type(mapping).__name__}")
        legacy = sorted(repr(key) for key in mapping if key in _LEGACY_EMIT_KEYS)
        if legacy:
            raise ValueError(
                "monitor emit has moved: use flow.termination_rules / flow.events; "
                "monitors publish values only (remove "
                f"{legacy} from emit)"
            )
        unknown = sorted(repr(key) for key in mapping if key not in _EMIT_KEYS)
        if unknown:
            raise ValueError(
                f"emit mapping has unknown keys {unknown}; expected keys " f"{sorted(_EMIT_KEYS)}"
            )
        value = mapping.get("value", "fired")
        # The constructor owns the field-level validation.
        return cls(value=cast(str, value))

    def to_mapping(self) -> dict[str, JsonValue]:
        """Return the closed, JSON-safe config form."""
        return {"value": self.value}


def _require_name(value: object, where: str) -> str:
    """Validate one local stable name."""
    if not isinstance(value, str) or not _NAME_PATTERN.fullmatch(value):
        raise ValueError(f"{where} must match [a-z][a-z0-9_]*, got {value!r}")
    return value


def _require_positive_int(value: object, where: str) -> int:
    """Validate a positive integer step count."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where}: expected an int, got {value!r}")
    if value < 1:
        raise ValueError(f"{where}: expected an int >= 1, got {value}")
    return value
