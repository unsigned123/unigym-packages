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
from types import MappingProxyType
from typing import Literal, TypeAlias

from unigym.core.episode_termination import Termination, validate_code

from .json import JsonValue, ensure_json_value

__all__ = [
    "SEVERITIES",
    "VERDICTS",
    "EpisodeRecord",
    "EpisodeSummary",
    "EpisodeVerdict",
    "Event",
    "Severity",
]

Severity: TypeAlias = Literal["info", "warning", "violation", "fatal"]

#: Every severity accepted for an :class:`Event`.
SEVERITIES: frozenset[str] = frozenset({"info", "warning", "violation", "fatal"})

EpisodeVerdict: TypeAlias = Literal["terminal", "deadline", "core", "incomplete", "rewound"]

#: Every verdict accepted for a finalized :class:`EpisodeRecord`.
VERDICTS: frozenset[str] = frozenset({"terminal", "deadline", "core", "incomplete", "rewound"})

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


@dataclass(frozen=True)
class EpisodeRecord:
    """A finalized episode's small, immutable facts.

    The streaming engine builds one record per due environment.  A record holds
    only scalars, the first pack terminal (if any), per-episode event counts and
    a verdict; it never retains a view, frame, tensor or backend handle.  The
    report layer owns serialization into the external report schema.
    """

    env_id: int
    episode_seq: int
    ticks: int
    verdict: EpisodeVerdict
    first_terminal: Termination | None = None
    event_counts: Mapping[str, int] = field(default_factory=dict)
    rewound: bool = False

    def __post_init__(self) -> None:
        """Validate the record and freeze its event counts."""
        if isinstance(self.env_id, bool) or not isinstance(self.env_id, int):
            raise TypeError(f"episode record.env_id: expected an int, got {self.env_id!r}")
        if self.env_id < 0:
            raise ValueError(f"episode record.env_id: expected an int >= 0, got {self.env_id}")
        _require_nonnegative_int(self.episode_seq, "episode record.episode_seq")
        _require_nonnegative_int(self.ticks, "episode record.ticks")
        if self.verdict not in VERDICTS:
            raise ValueError(
                f"episode record.verdict: expected one of {sorted(VERDICTS)}, "
                f"got {self.verdict!r}"
            )
        if self.first_terminal is not None and not isinstance(self.first_terminal, Termination):
            raise TypeError(
                "episode record.first_terminal: expected a Termination or None, "
                f"got {type(self.first_terminal).__name__}"
            )
        if self.verdict == "terminal" and self.first_terminal is None:
            raise ValueError("episode record: verdict 'terminal' requires a first_terminal")
        if not isinstance(self.event_counts, Mapping):
            raise TypeError(
                "episode record.event_counts: expected a mapping, "
                f"got {type(self.event_counts).__name__}"
            )
        counts: dict[str, int] = {}
        for code, count in self.event_counts.items():
            if not isinstance(code, str) or not code:
                raise TypeError(
                    f"episode record.event_counts key must be a non-empty str, got {code!r}"
                )
            _require_nonnegative_int(count, f"episode record.event_counts[{code!r}]")
            counts[code] = count
        object.__setattr__(self, "event_counts", MappingProxyType(counts))
        if not isinstance(self.rewound, bool):
            raise TypeError(f"episode record.rewound: expected a bool, got {self.rewound!r}")

    def to_json(self) -> dict[str, JsonValue]:
        """Return a fresh JSON-safe record document."""
        return {
            "env_id": self.env_id,
            "episode_seq": self.episode_seq,
            "ticks": self.ticks,
            "verdict": self.verdict,
            "first_terminal": (
                None if self.first_terminal is None else self.first_terminal.to_json()
            ),
            "event_counts": dict(self.event_counts),
            "rewound": self.rewound,
        }


@dataclass(frozen=True)
class EpisodeSummary:
    """Run-scoped episode totals owned by the streaming engine.

    The summary is deliberately aggregate-only: it carries no episode payloads
    and no tensors.  A report builder may combine it with retained
    :class:`EpisodeRecord` values without asking the engine to score anything.
    """

    episodes: int = 0
    total_ticks: int = 0
    verdict_counts: Mapping[str, int] = field(default_factory=dict)
    event_count: int = 0
    dropped_events: int = 0
    conflicts: int = 0

    def __post_init__(self) -> None:
        """Validate the aggregate counters and freeze the verdict map."""
        _require_nonnegative_int(self.episodes, "episode summary.episodes")
        _require_nonnegative_int(self.total_ticks, "episode summary.total_ticks")
        _require_nonnegative_int(self.event_count, "episode summary.event_count")
        _require_nonnegative_int(self.dropped_events, "episode summary.dropped_events")
        _require_nonnegative_int(self.conflicts, "episode summary.conflicts")
        if not isinstance(self.verdict_counts, Mapping):
            raise TypeError(
                "episode summary.verdict_counts: expected a mapping, "
                f"got {type(self.verdict_counts).__name__}"
            )
        counts: dict[str, int] = {}
        for verdict, count in self.verdict_counts.items():
            if verdict not in VERDICTS:
                raise ValueError(
                    f"episode summary.verdict_counts: unknown verdict {verdict!r}; "
                    f"allowed {sorted(VERDICTS)}"
                )
            _require_nonnegative_int(count, f"episode summary.verdict_counts[{verdict!r}]")
            counts[verdict] = count
        object.__setattr__(self, "verdict_counts", MappingProxyType(counts))

    def to_json(self) -> dict[str, JsonValue]:
        """Return a fresh JSON-safe summary document."""
        return {
            "episodes": self.episodes,
            "total_ticks": self.total_ticks,
            "verdict_counts": dict(self.verdict_counts),
            "event_count": self.event_count,
            "dropped_events": self.dropped_events,
            "conflicts": self.conflicts,
        }


def _require_nonnegative_int(value: object, where: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where}: expected an int, got {value!r}")
    if value < 0:
        raise ValueError(f"{where}: expected an int >= 0, got {value}")


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
