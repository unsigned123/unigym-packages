"""Predicate configuration specs shared by the monitor and task composition.

A predicate is one condition tree plus a trigger cadence and an emit
configuration.  This module owns the two frozen, JSON-safe configuration
objects; :class:`unigym_bench.monitors.PredicateMonitor` owns the stateful
runtime.  Keeping the specs separate lets the task-document composer parse and
fingerprint a predicate without importing the monitor implementation.

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
    supplied final context and fires only for due rows.  Per OD-5 its terminal
    output is report/scoring-only and is drained separately from the runner
    terminal face.

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

from unigym.core.episode_termination import EpisodeOutcome

from .events import SEVERITIES, Severity
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
_MONITOR_OUTCOMES: frozenset[EpisodeOutcome] = frozenset(
    {EpisodeOutcome.SUCCESS, EpisodeOutcome.FAILURE, EpisodeOutcome.DEADLINE}
)

_TRIGGER_KEYS: frozenset[str] = frozenset({"kind", "steps"})
_EMIT_KEYS: frozenset[str] = frozenset(
    {"value", "event", "severity", "terminal_reason", "terminal_outcome"}
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
    """Local value name and optional event/terminal outputs for one monitor.

    Args:
        value: Local level value name published by ``signals()``.  Defaults to
            ``fired``.
        event: Optional monitor-local occurrence name.  The emitted event code is
            ``task.<monitor>.<event>``.
        severity: Severity for the emitted event.
        terminal_reason: Optional stable reason used in
            ``custom.<monitor>.<reason>``.
        terminal_outcome: Optional :class:`EpisodeOutcome`.  It must be supplied
            together with ``terminal_reason``.
    """

    value: str = "fired"
    event: str | None = None
    severity: Severity = "info"
    terminal_reason: str | None = None
    terminal_outcome: EpisodeOutcome | str | None = None

    def __post_init__(self) -> None:
        """Validate names, severity and the terminal pair."""
        _require_name(self.value, "EmitSpec.value")
        if self.event is not None:
            _require_name(self.event, "EmitSpec.event")
        object.__setattr__(self, "severity", _require_severity(self.severity, "EmitSpec.severity"))
        if self.terminal_reason is not None:
            _require_name(self.terminal_reason, "EmitSpec.terminal_reason")
        outcome = _require_outcome(self.terminal_outcome, "EmitSpec.terminal_outcome")
        object.__setattr__(self, "terminal_outcome", outcome)
        if (self.terminal_reason is None) != (outcome is None):
            raise ValueError(
                "EmitSpec terminal_reason and terminal_outcome must be supplied "
                "together; write both to request a terminal or neither to skip it"
            )

    @classmethod
    def from_mapping(cls, mapping: object) -> EmitSpec:
        """Build an emit spec from its closed JSON-safe mapping.

        Args:
            mapping: A mapping with any of ``value``, ``event``, ``severity``,
                ``terminal_reason`` and ``terminal_outcome``.

        Returns:
            The validated emit spec.

        Raises:
            ValueError: The mapping has unknown keys or an invalid value.
            TypeError: The mapping is not an object.
        """
        if not isinstance(mapping, Mapping):
            raise TypeError(f"emit mapping must be an object, got {type(mapping).__name__}")
        unknown = sorted(repr(key) for key in mapping if key not in _EMIT_KEYS)
        if unknown:
            raise ValueError(
                f"emit mapping has unknown keys {unknown}; expected keys " f"{sorted(_EMIT_KEYS)}"
            )
        value = mapping.get("value", "fired")
        event = mapping.get("event")
        severity = mapping.get("severity", "info")
        reason = mapping.get("terminal_reason")
        outcome = mapping.get("terminal_outcome")
        # The constructor owns the field-level validation.
        return cls(
            value=cast(str, value),
            event=cast(str | None, event),
            severity=cast(Severity, severity),
            terminal_reason=cast(str | None, reason),
            terminal_outcome=cast(EpisodeOutcome | str | None, outcome),
        )

    def to_mapping(self) -> dict[str, JsonValue]:
        """Return the closed, JSON-safe config form."""
        outcome = self.terminal_outcome
        return {
            "value": self.value,
            "event": self.event,
            "severity": self.severity,
            "terminal_reason": self.terminal_reason,
            "terminal_outcome": None if outcome is None else outcome.value,
        }


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


def _require_severity(value: object, where: str) -> Severity:
    """Validate one event severity."""
    if not isinstance(value, str) or value not in SEVERITIES:
        raise ValueError(f"{where}: expected one of {sorted(SEVERITIES)}, got {value!r}")
    if value == "info":
        return "info"
    if value == "warning":
        return "warning"
    if value == "violation":
        return "violation"
    return "fatal"


def _require_outcome(value: object, where: str) -> EpisodeOutcome | None:
    """Validate a monitor-owned terminal outcome."""
    if value is None:
        return None
    if isinstance(value, EpisodeOutcome):
        outcome = value
    elif isinstance(value, str):
        try:
            outcome = EpisodeOutcome(value)
        except ValueError as exc:
            raise ValueError(
                f"{where}: unknown episode outcome {value!r}; "
                f"allowed: {[item.value for item in EpisodeOutcome]}"
            ) from exc
    else:
        raise TypeError(f"{where}: expected an EpisodeOutcome, str or None, got {value!r}")
    if outcome not in _MONITOR_OUTCOMES:
        raise ValueError(
            f"{where}: terminal outcome {outcome.value!r} is not owned by a monitor; "
            f"allowed: {sorted(item.value for item in _MONITOR_OUTCOMES)}"
        )
    return outcome
