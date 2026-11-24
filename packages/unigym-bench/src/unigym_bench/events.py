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

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypeAlias

from unigym.core.episode_termination import (
    EpisodeOutcome,
    EpisodeTermination,
    Termination,
    TerminationExtension,
    TerminationSource,
    validate_code,
)

from .json import JsonValue, ensure_json_value

if TYPE_CHECKING:
    from .specs import FrozenJson

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
_RECORD_KEYS = frozenset(
    {
        "env_id",
        "episode_seq",
        "ticks",
        "verdict",
        "command_at_start",
        "first_terminal",
        "core_termination",
        "reset_facts",
        "event_counts",
        "rewound",
    }
)
_TERMINATION_KEYS = frozenset(
    {
        "outcome",
        "code",
        "source",
        "step_index",
        "time_s",
        "metrics",
        "extensions",
        "priority",
    }
)
_EPISODE_TERMINATION_KEYS = frozenset({"primary", "all"})
_TERMINATION_EXTENSION_KEYS = frozenset({"namespace", "schema_version", "data"})


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
    only scalars, the command observed on the episode's first tick, the first
    pack terminal (if any), the core/runner termination facts, the boundary-safe
    reset facts captured at episode start, per-episode event counts and a
    verdict; it never retains a view, frame, tensor or backend handle.

    ``first_terminal`` prefers the runner's canonical stamped terminal when the
    same custom request also appears in the episode-end bundle.  Otherwise it
    is the monitor's raw request; P5/P6 own any later re-stamp.  The report
    layer owns serialization into the external report schema.

    ``reset_facts`` is a deeply frozen JSON tree (read-only mappings and
    tuples); :meth:`to_json` rebuilds a fully independent mutable copy.
    """

    env_id: int
    episode_seq: int
    ticks: int
    verdict: EpisodeVerdict
    command_at_start: tuple[float, ...] | None = None
    first_terminal: Termination | None = None
    core_termination: EpisodeTermination | None = None
    reset_facts: Mapping[str, FrozenJson] = field(default_factory=dict)
    event_counts: Mapping[str, int] = field(default_factory=dict)
    rewound: bool = False

    def __post_init__(self) -> None:
        """Validate the record and freeze its mappings."""
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
        object.__setattr__(self, "command_at_start", _validate_command(self.command_at_start))
        if self.first_terminal is not None and not isinstance(self.first_terminal, Termination):
            raise TypeError(
                "episode record.first_terminal: expected a Termination or None, "
                f"got {type(self.first_terminal).__name__}"
            )
        if self.core_termination is not None and not isinstance(
            self.core_termination, EpisodeTermination
        ):
            raise TypeError(
                "episode record.core_termination: expected an EpisodeTermination or None, "
                f"got {type(self.core_termination).__name__}"
            )
        if self.verdict == "terminal" and self.first_terminal is None:
            raise ValueError("episode record: verdict 'terminal' requires a first_terminal")
        object.__setattr__(self, "reset_facts", _freeze_json_mapping(self.reset_facts))
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
            "command_at_start": (
                None if self.command_at_start is None else list(self.command_at_start)
            ),
            "first_terminal": (
                None if self.first_terminal is None else self.first_terminal.to_json()
            ),
            "core_termination": (
                None if self.core_termination is None else self.core_termination.to_json()
            ),
            "reset_facts": _thaw_json_mapping(self.reset_facts),
            "event_counts": dict(self.event_counts),
            "rewound": self.rewound,
        }

    @classmethod
    def from_json(cls, payload: object) -> EpisodeRecord:
        """Rebuild a record from :meth:`to_json` output.

        Args:
            payload: A JSON-safe record document.

        Returns:
            A new record sharing no mutable state with ``payload``.

        Raises:
            TypeError: The payload or one of its fields has the wrong type.
            ValueError: A field is missing, unknown or invalid.
        """
        document = ensure_json_value(payload, where="episode record snapshot")
        if not isinstance(document, dict):
            raise TypeError(
                "episode record snapshot: expected a mapping, " f"got {type(document).__name__}"
            )
        _require_keys(document, _RECORD_KEYS, "episode record snapshot")
        raw_env_id = document["env_id"]
        if isinstance(raw_env_id, bool) or not isinstance(raw_env_id, int):
            raise TypeError(f"episode record snapshot.env_id: expected an int, got {raw_env_id!r}")
        episode_seq = _require_int(document["episode_seq"], "episode record snapshot.episode_seq")
        ticks = _require_int(document["ticks"], "episode record snapshot.ticks")
        verdict = _require_verdict(document["verdict"], "episode record snapshot.verdict")
        first_terminal = _optional_termination(
            document["first_terminal"], "episode record snapshot.first_terminal"
        )
        core_termination = _optional_episode_termination(
            document["core_termination"], "episode record snapshot.core_termination"
        )
        reset_facts = document["reset_facts"]
        if not isinstance(reset_facts, dict):
            raise TypeError(
                "episode record snapshot.reset_facts: expected an object, "
                f"got {type(reset_facts).__name__}"
            )
        event_counts = document["event_counts"]
        if not isinstance(event_counts, dict):
            raise TypeError(
                "episode record snapshot.event_counts: expected an object, "
                f"got {type(event_counts).__name__}"
            )
        rewound = document["rewound"]
        if not isinstance(rewound, bool):
            raise TypeError(f"episode record snapshot.rewound: expected a bool, got {rewound!r}")
        return cls(
            env_id=raw_env_id,
            episode_seq=episode_seq,
            ticks=ticks,
            verdict=verdict,
            command_at_start=_validate_command(document["command_at_start"]),
            first_terminal=first_terminal,
            core_termination=core_termination,
            reset_facts=reset_facts,
            event_counts=event_counts,
            rewound=rewound,
        )


@dataclass(frozen=True)
class EpisodeSummary:
    """Run-scoped episode totals owned by the streaming engine.

    The summary is deliberately aggregate-only: it carries no episode payloads
    and no tensors.  Every counter is maintained independently of episode
    retention, so a report can still count verdicts, rewinds, terminations and
    per-env episodes when ``retain_episodes`` is false.  A report builder may
    combine it with retained :class:`EpisodeRecord` values for per-record
    detail without asking the engine to score anything.
    """

    episodes: int = 0
    total_ticks: int = 0
    verdict_counts: Mapping[str, int] = field(default_factory=dict)
    event_count: int = 0
    dropped_events: int = 0
    conflicts: int = 0
    rewound: int = 0
    termination_outcome_counts: Mapping[str, int] = field(default_factory=dict)
    termination_code_counts: Mapping[str, int] = field(default_factory=dict)
    termination_source_counts: Mapping[str, int] = field(default_factory=dict)
    episodes_by_env: Mapping[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the aggregate counters and freeze the mappings."""
        _require_nonnegative_int(self.episodes, "episode summary.episodes")
        _require_nonnegative_int(self.total_ticks, "episode summary.total_ticks")
        _require_nonnegative_int(self.event_count, "episode summary.event_count")
        _require_nonnegative_int(self.dropped_events, "episode summary.dropped_events")
        _require_nonnegative_int(self.conflicts, "episode summary.conflicts")
        _require_nonnegative_int(self.rewound, "episode summary.rewound")
        if self.rewound > self.episodes:
            raise ValueError(
                "episode summary.rewound: expected a value <= episodes, "
                f"got {self.rewound} > {self.episodes}"
            )
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
        object.__setattr__(
            self,
            "termination_outcome_counts",
            _freeze_count_mapping(
                self.termination_outcome_counts, "episode summary.termination_outcome_counts"
            ),
        )
        object.__setattr__(
            self,
            "termination_code_counts",
            _freeze_count_mapping(
                self.termination_code_counts, "episode summary.termination_code_counts"
            ),
        )
        object.__setattr__(
            self,
            "termination_source_counts",
            _freeze_count_mapping(
                self.termination_source_counts, "episode summary.termination_source_counts"
            ),
        )
        object.__setattr__(self, "episodes_by_env", _freeze_env_counts(self.episodes_by_env))

    def to_json(self) -> dict[str, JsonValue]:
        """Return a fresh JSON-safe summary document."""
        return {
            "episodes": self.episodes,
            "total_ticks": self.total_ticks,
            "verdict_counts": dict(self.verdict_counts),
            "event_count": self.event_count,
            "dropped_events": self.dropped_events,
            "conflicts": self.conflicts,
            "rewound": self.rewound,
            "termination_outcome_counts": dict(self.termination_outcome_counts),
            "termination_code_counts": dict(self.termination_code_counts),
            "termination_source_counts": dict(self.termination_source_counts),
            "episodes_by_env": {
                str(env_id): count for env_id, count in sorted(self.episodes_by_env.items())
            },
        }


def termination_from_json(payload: JsonValue, where: str) -> Termination:
    """Rebuild one core termination from its :meth:`Termination.to_json` form."""
    if not isinstance(payload, dict):
        raise TypeError(f"{where}: expected a termination object, got {type(payload).__name__}")
    _require_keys(payload, _TERMINATION_KEYS, where)
    outcome_text = payload["outcome"]
    source_text = payload["source"]
    code = payload["code"]
    if not isinstance(outcome_text, str):
        raise TypeError(f"{where}.outcome: expected a string, got {outcome_text!r}")
    if not isinstance(source_text, str):
        raise TypeError(f"{where}.source: expected a string, got {source_text!r}")
    if not isinstance(code, str):
        raise TypeError(f"{where}.code: expected a string, got {code!r}")
    try:
        outcome = EpisodeOutcome(outcome_text)
    except ValueError as exc:
        raise ValueError(f"{where}.outcome: unknown outcome {outcome_text!r}") from exc
    try:
        source = TerminationSource(source_text)
    except ValueError as exc:
        raise ValueError(f"{where}.source: unknown source {source_text!r}") from exc
    step_index = _optional_int(payload["step_index"], f"{where}.step_index")
    time_s = _optional_float(payload["time_s"], f"{where}.time_s")
    metrics = _float_mapping(payload["metrics"], f"{where}.metrics")
    extensions = _termination_extensions(payload["extensions"], f"{where}.extensions")
    priority = _require_int(payload["priority"], f"{where}.priority")
    return Termination(
        outcome=outcome,
        code=code,
        source=source,
        step_index=step_index,
        time_s=time_s,
        metrics=metrics,
        extensions=extensions,
        priority=priority,
    )


def episode_termination_from_json(payload: JsonValue, where: str) -> EpisodeTermination:
    """Rebuild one core episode termination from its JSON form."""
    if not isinstance(payload, dict):
        raise TypeError(
            f"{where}: expected an episode termination object, got {type(payload).__name__}"
        )
    _require_keys(payload, _EPISODE_TERMINATION_KEYS, where)
    raw_all = payload["all"]
    if not isinstance(raw_all, list):
        raise TypeError(f"{where}.all: expected a list, got {type(raw_all).__name__}")
    if not raw_all:
        raise ValueError(f"{where}.all: expected at least one termination")
    terms = tuple(
        termination_from_json(item, f"{where}.all[{index}]") for index, item in enumerate(raw_all)
    )
    primary = termination_from_json(payload["primary"], f"{where}.primary")
    return EpisodeTermination(primary=primary, all=terms)


def _optional_termination(payload: JsonValue, where: str) -> Termination | None:
    if payload is None:
        return None
    return termination_from_json(payload, where)


def _optional_episode_termination(payload: JsonValue, where: str) -> EpisodeTermination | None:
    if payload is None:
        return None
    return episode_termination_from_json(payload, where)


def _termination_extensions(payload: JsonValue, where: str) -> dict[str, TerminationExtension]:
    if not isinstance(payload, dict):
        raise TypeError(f"{where}: expected an object, got {type(payload).__name__}")
    extensions: dict[str, TerminationExtension] = {}
    for key, item in payload.items():
        item_where = f"{where}[{key!r}]"
        if not isinstance(item, dict):
            raise TypeError(f"{item_where}: expected an object, got {type(item).__name__}")
        _require_keys(item, _TERMINATION_EXTENSION_KEYS, item_where)
        namespace = item["namespace"]
        schema_version = item["schema_version"]
        data = item["data"]
        if not isinstance(namespace, str):
            raise TypeError(f"{item_where}.namespace: expected a string, got {namespace!r}")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise TypeError(f"{item_where}.schema_version: expected an int, got {schema_version!r}")
        if not isinstance(data, dict):
            raise TypeError(f"{item_where}.data: expected an object, got {type(data).__name__}")
        extensions[key] = TerminationExtension(
            namespace=namespace,
            schema_version=schema_version,
            data=data,
        )
    return extensions


def _float_mapping(payload: JsonValue, where: str) -> dict[str, float]:
    if not isinstance(payload, dict):
        raise TypeError(f"{where}: expected an object, got {type(payload).__name__}")
    values: dict[str, float] = {}
    for key, item in payload.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{where}[{key!r}]: expected a number, got {item!r}")
        values[key] = float(item)
    return values


def _optional_int(payload: JsonValue, where: str) -> int | None:
    if payload is None:
        return None
    return _require_int(payload, where)


def _optional_float(payload: JsonValue, where: str) -> float | None:
    if payload is None:
        return None
    if isinstance(payload, bool) or not isinstance(payload, (int, float)):
        raise TypeError(f"{where}: expected a number or None, got {payload!r}")
    return float(payload)


def _require_int(payload: JsonValue, where: str) -> int:
    if isinstance(payload, bool) or not isinstance(payload, int):
        raise TypeError(f"{where}: expected an int, got {payload!r}")
    return payload


def _require_verdict(payload: JsonValue, where: str) -> EpisodeVerdict:
    if payload == "terminal":
        return "terminal"
    if payload == "deadline":
        return "deadline"
    if payload == "core":
        return "core"
    if payload == "incomplete":
        return "incomplete"
    if payload == "rewound":
        return "rewound"
    raise ValueError(f"{where}: expected one of {sorted(VERDICTS)}, got {payload!r}")


def _validate_command(value: object) -> tuple[float, ...] | None:
    """Normalize an episode's start command into a finite float tuple."""
    if value is None:
        return None
    if not isinstance(value, (tuple, list)):
        raise TypeError(
            "episode record.command_at_start: expected a sequence of numbers or None, "
            f"got {type(value).__name__}"
        )
    command: list[float] = []
    for index, raw in enumerate(value):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(
                f"episode record.command_at_start[{index}]: expected a number, got {raw!r}"
            )
        item = float(raw)
        if not math.isfinite(item):
            raise ValueError(
                f"episode record.command_at_start[{index}]: expected a finite number, got {item!r}"
            )
        command.append(item)
    return tuple(command)


def _freeze_json_mapping(value: object) -> Mapping[str, FrozenJson]:
    """Validate and deep-freeze a JSON object into read-only containers."""
    if not isinstance(value, Mapping):
        raise TypeError(
            "episode record.reset_facts: expected a mapping, " f"got {type(value).__name__}"
        )
    try:
        normalized = ensure_json_value(dict(value), where="episode record.reset_facts")
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"episode record.reset_facts: {exc}") from exc
    if not isinstance(normalized, dict):
        raise TypeError("episode record.reset_facts: expected an object")
    return MappingProxyType({key: _freeze_json(item) for key, item in normalized.items()})


def _freeze_json(value: JsonValue) -> FrozenJson:
    """Recursively turn a JSON tree into read-only mappings and tuples."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json_mapping(value: Mapping[str, FrozenJson]) -> dict[str, JsonValue]:
    """Return a fresh, fully independent JSON object from a frozen tree."""
    return {key: _thaw_json(item) for key, item in value.items()}


def _thaw_json(value: FrozenJson) -> JsonValue:
    """Recursively rebuild mutable JSON containers from a frozen tree."""
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _require_nonnegative_int(value: object, where: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where}: expected an int, got {value!r}")
    if value < 0:
        raise ValueError(f"{where}: expected an int >= 0, got {value}")


def _freeze_count_mapping(value: object, where: str) -> Mapping[str, int]:
    """Validate and freeze a string-keyed non-negative count mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{where}: expected a mapping, got {type(value).__name__}")
    counts: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError(f"{where}: keys must be non-empty strings, got {key!r}")
        _require_nonnegative_int(count, f"{where}[{key!r}]")
        counts[key] = count
    return MappingProxyType(dict(sorted(counts.items())))


def _freeze_env_counts(value: object) -> Mapping[int, int]:
    """Validate and freeze an ``env_id -> count`` mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(
            "episode summary.episodes_by_env: expected a mapping, " f"got {type(value).__name__}"
        )
    counts: dict[int, int] = {}
    for env_id, count in value.items():
        if isinstance(env_id, bool) or not isinstance(env_id, int):
            raise TypeError(
                f"episode summary.episodes_by_env: env ids must be ints, got {env_id!r}"
            )
        if env_id < 0:
            raise ValueError(
                f"episode summary.episodes_by_env: expected env ids >= 0, got {env_id}"
            )
        _require_nonnegative_int(count, f"episode summary.episodes_by_env[{env_id}]")
        counts[env_id] = count
    return MappingProxyType(dict(sorted(counts.items())))


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
