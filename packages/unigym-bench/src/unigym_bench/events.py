"""Pack-owned discrete occurrence records.

An :class:`Event` is one thing that happened at a tick.  It is not a signal
(which is a per-env level) and not a terminal (which ends an env).  Events are
one-shot occurrences: a monitor drains what it emitted and the engine records
it.  The monitor owns the occurrence decision; the engine does not deduplicate
events in v1.

Codes use the core termination code grammar (``<namespace>.<stable_name>``)
so a report cannot silently invent an event schema.  ``data`` is JSON-safe and
is copied at construction and serialization; an event never retains a view,
tensor or caller-owned mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from unigym.core.episode_termination import validate_code

from .json import JsonValue, ensure_json_value

__all__ = ["SEVERITIES", "Event", "Severity"]

Severity: TypeAlias = Literal["info", "warning", "violation", "fatal"]

#: Every severity accepted for an :class:`Event`.
SEVERITIES: frozenset[str] = frozenset({"info", "warning", "violation", "fatal"})

_EVENT_KEYS = frozenset({"name", "code", "env_ids", "severity", "data"})


@dataclass(frozen=True)
class Event:
    """A discrete occurrence owned by one monitor.

    Attributes:
        name: Monitor-local occurrence name (for example
            ``threshold_reached``).
        code: Namespaced stable code validated with the core code grammar.
        env_ids: Non-empty tuple of environment ids the occurrence applies to.
        severity: One of ``info``, ``warning``, ``violation`` or ``fatal``.
        data: JSON-safe details, copied at construction and serialization.
    """

    name: str
    code: str
    env_ids: tuple[int, ...]
    severity: Severity = "info"
    data: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and copy the occurrence."""
        if not isinstance(self.name, str) or not self.name:
            raise TypeError(f"event name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.code, str):
            raise TypeError(f"event {self.name!r}: code must be a string, got {self.code!r}")
        validate_code(self.code)
        env_ids = tuple(self.env_ids)
        if not env_ids:
            raise ValueError(f"event {self.name!r}: env_ids must not be empty")
        for env_id in env_ids:
            if isinstance(env_id, bool) or not isinstance(env_id, int):
                raise TypeError(f"event {self.name!r}: env_ids must be ints, got {env_id!r}")
            if env_id < 0:
                raise ValueError(f"event {self.name!r}: env ids must be >= 0, got {env_id!r}")
        if len(set(env_ids)) != len(env_ids):
            raise ValueError(f"event {self.name!r}: env_ids must be unique, got {env_ids!r}")
        object.__setattr__(self, "env_ids", env_ids)
        if not isinstance(self.severity, str) or self.severity not in SEVERITIES:
            raise ValueError(
                f"event {self.name!r}: severity must be one of {sorted(SEVERITIES)}, "
                f"got {self.severity!r}"
            )
        if not isinstance(self.data, Mapping):
            raise TypeError(
                f"event {self.name!r}: data must be a mapping, got {type(self.data).__name__}"
            )
        try:
            normalized = ensure_json_value(dict(self.data), where=f"event {self.name!r}.data")
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"event {self.name!r}: {exc}") from exc
        if not isinstance(normalized, dict):
            raise TypeError(f"event {self.name!r}: data must normalize to an object")
        object.__setattr__(self, "data", normalized)

    def to_json(self) -> dict[str, JsonValue]:
        """Return a fresh, JSON-safe event document."""
        data = ensure_json_value(dict(self.data), where=f"event {self.name!r}.data")
        return {
            "name": self.name,
            "code": self.code,
            "env_ids": list(self.env_ids),
            "severity": self.severity,
            "data": data,
        }

    @classmethod
    def from_json(cls, payload: object) -> Event:
        """Rebuild an event from :meth:`to_json` output.

        Args:
            payload: A JSON-safe event document.

        Returns:
            A new event sharing no mutable state with ``payload``.

        Raises:
            TypeError: The payload or one of its fields has the wrong type.
            ValueError: A field is missing, unknown or invalid.
        """
        document = ensure_json_value(payload, where="event snapshot")
        if not isinstance(document, dict):
            raise TypeError(f"event snapshot: expected a mapping, got {type(document).__name__}")
        _require_keys(document, _EVENT_KEYS, "event snapshot")
        name = document["name"]
        if not isinstance(name, str):
            raise TypeError(f"event snapshot.name: expected a string, got {name!r}")
        code = document["code"]
        if not isinstance(code, str):
            raise TypeError(f"event snapshot.code: expected a string, got {code!r}")
        raw_env_ids = document["env_ids"]
        if not isinstance(raw_env_ids, list):
            raise TypeError(
                f"event snapshot.env_ids: expected a list, got {type(raw_env_ids).__name__}"
            )
        env_ids: list[int] = []
        for index, value in enumerate(raw_env_ids):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"event snapshot.env_ids[{index}]: expected an int, got {value!r}")
            env_ids.append(value)
        severity = _require_severity(document["severity"], "event snapshot.severity")
        data = document["data"]
        if not isinstance(data, dict):
            raise TypeError(f"event snapshot.data: expected an object, got {type(data).__name__}")
        return cls(
            name=name,
            code=code,
            env_ids=tuple(env_ids),
            severity=severity,
            data=data,
        )


def _require_severity(value: JsonValue, where: str) -> Severity:
    if value == "info":
        return "info"
    if value == "warning":
        return "warning"
    if value == "violation":
        return "violation"
    if value == "fatal":
        return "fatal"
    raise ValueError(f"{where}: expected one of {sorted(SEVERITIES)}, got {value!r}")


def _require_keys(document: Mapping[str, JsonValue], expected: frozenset[str], where: str) -> None:
    missing = sorted(expected - document.keys())
    unknown = sorted(document.keys() - expected)
    if missing or unknown:
        raise ValueError(
            f"{where}: expected keys {sorted(expected)}, missing {missing}, unknown {unknown}"
        )
