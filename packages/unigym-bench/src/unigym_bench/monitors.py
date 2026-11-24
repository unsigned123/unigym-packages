"""Monitor protocol, convenience base and the v1 built-in monitors.

A monitor is the stateful per-env unit of a benchmark task.  It observes ticks,
publishes per-env level signals, emits one-shot events and requests per-env
episode terminals.  This module owns the monitor contract, the six legacy
built-ins and the compositional :class:`PredicateMonitor`; it owns no engine,
sink, report or scenario code.

Ownership boundaries:

- A monitor decides when an event occurs.  Events are drained one-shot and the
  engine does not deduplicate them in v1.  Built-ins emit on their own edges.
- A monitor requests terminals as core :class:`Termination` facts with
  ``source=CUSTOM`` and a ``custom.<monitor>.<reason>`` code.  The engine owns
  the first-terminal latch for an environment; a monitor keeps no competing
  latch.
- ``signals()`` returns fresh per-env level tensors with a leading dimension of
  ``num_envs``.  A monitor stores plain Python measurements, never tensors or
  views, so it cannot retain boundary data.
- ``reset(env_ids)`` clears per-env measurement state (edges, streaks, phases,
  captured commands).  Run-scoped report accumulators persist, and the last
  observed level is refreshed by the next tick rather than erased.
- A stateful built-in implements :class:`RewindableMonitor`; ``snapshot()`` is
  JSON-safe and both directions return private copies.

Built-ins that read a value take an explicit :class:`Probe` callable.  The
probe is composed at the pack's composition boundary and returns one level
tensor with shape ``(num_envs,)`` on the context device.  A probe is never
inferred from a name, so a monitor cannot silently read the wrong quantity.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, TypeVar, cast, runtime_checkable

import torch
from torch import Tensor
from unigym.core.episode_termination import (
    EpisodeOutcome,
    Termination,
    TerminationSource,
    validate_code,
)
from unigym.evaluation import EnvStepContext

from .conditions import Condition, ConditionError, parse_condition
from .events import SEVERITIES, Event, Severity, termination_from_json
from .json import JsonValue, ensure_json_value
from .predicates import EmitSpec, TriggerKind, TriggerSpec
from .sources import SourceResolver

__all__ = [
    "BaseMonitor",
    "Comparison",
    "EventCounterMonitor",
    "HoldMonitor",
    "Monitor",
    "PhaseMonitor",
    "PredicateMonitor",
    "Probe",
    "RewindableMonitor",
    "SafetyMonitor",
    "ScenarioCoverageMonitor",
    "ThresholdMonitor",
]

#: Comparison operator accepted by the value-threshold built-ins.
Comparison: TypeAlias = Literal[">", ">=", "<", "<="]

_SNAPSHOT_SCHEMA = "unigym_bench.monitor/2"
_SNAPSHOT_KEYS = frozenset(
    {
        "schema",
        "name",
        "kind",
        "num_envs",
        "device",
        "config",
        "state",
        "pending_events",
        "pending_terminals",
    }
)
_COMPARISONS: frozenset[str] = frozenset({">", ">=", "<", "<="})
_MONITOR_OUTCOMES: frozenset[EpisodeOutcome] = frozenset(
    {EpisodeOutcome.SUCCESS, EpisodeOutcome.FAILURE, EpisodeOutcome.DEADLINE}
)
_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")

_T = TypeVar("_T")


@dataclass(frozen=True)
class _DecodedMonitorSnapshot:
    """One fully validated monitor snapshot, held as private locals."""

    state: object
    num_envs: int | None
    device: torch.device | None
    pending_events: tuple[Event, ...]
    pending_terminals: dict[int, Termination]


class Probe(Protocol):
    """Read one per-env level from a tick context.

    The returned tensor must have shape ``(context.num_envs,)`` and live on
    ``context.device``.  A numeric probe returns a floating tensor; a boolean
    probe returns ``torch.bool``.  The monitor copies the values immediately
    and never retains the tensor.
    """

    def __call__(self, context: EnvStepContext) -> Tensor:
        """Return the per-env level for ``context``."""
        ...


@runtime_checkable
class Monitor(Protocol):
    """Stateful per-env unit of a benchmark task."""

    name: str

    def on_tick(self, context: EnvStepContext) -> None:
        """Observe one tick."""

    def signals(self) -> Mapping[str, Tensor]:
        """Return fresh per-env level tensors keyed by local signal name."""

    def drain_events(self) -> Sequence[Event]:
        """Return and clear the occurrences since the previous drain."""

    def drain_terminals(self) -> Mapping[int, Termination]:
        """Return and clear the terminal requests since the previous drain."""

    def on_episode_end(self, env_ids: Sequence[int], context: EnvStepContext | None = None) -> None:
        """Observe the episode boundary for the due ``env_ids``."""

    def reset(self, env_ids: Sequence[int]) -> None:
        """Clear per-env measurement state for ``env_ids``."""

    def report_view(self) -> Mapping[str, JsonValue]:
        """Return a run-scoped, JSON-safe fallback report block."""


@runtime_checkable
class RewindableMonitor(Protocol):
    """Optional capability for a monitor with rewindable private state."""

    def snapshot(self) -> JsonValue:
        """Return a JSON-safe private copy of the monitor state."""

    def restore(self, state: JsonValue) -> None:
        """Replace the monitor state from a snapshot's private copy."""


class BaseMonitor:
    """Convenience monitor with empty values and no-op lifecycle hooks.

    Subclass it for a stateless monitor or as a starting point for a custom
    stateful monitor.  :meth:`report_view` returns a small JSON-safe default;
    a subclass that contributes measurements overrides it.
    """

    kind: str = "base"

    def __init__(self, name: str) -> None:
        self._name = _require_monitor_name(name)

    @property
    def name(self) -> str:
        """The monitor name declared by the task."""
        return self._name

    def on_tick(self, context: EnvStepContext) -> None:
        """Observe one tick; the base monitor does nothing."""

    def signals(self) -> Mapping[str, Tensor]:
        """Return no signals."""
        return {}

    def drain_events(self) -> Sequence[Event]:
        """Return no events."""
        return ()

    def drain_terminals(self) -> Mapping[int, Termination]:
        """Return no terminals."""
        return {}

    def on_episode_end(self, env_ids: Sequence[int], context: EnvStepContext | None = None) -> None:
        """Observe an episode boundary; the base monitor does nothing."""
        _validate_env_ids(env_ids, None, "on_episode_end.env_ids")

    def reset(self, env_ids: Sequence[int]) -> None:
        """Clear per-env state; the base monitor has none."""
        _validate_env_ids(env_ids, None, "reset.env_ids")

    def report_view(self) -> Mapping[str, JsonValue]:
        """Return the default JSON-safe report block."""
        return {"name": self._name, "kind": self.kind}

    def _json_report(self, body: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        """Validate and copy one report block."""
        normalized = ensure_json_value(dict(body), where=f"monitor {self._name!r} report_view")
        if not isinstance(normalized, dict):
            raise TypeError(f"monitor {self._name!r} report_view must be an object")
        return normalized


class _StatefulMonitor(BaseMonitor):
    """Shared JSON state, drain and snapshot plumbing for built-in monitors."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._num_envs: int | None = None
        self._device: torch.device | None = None
        self._pending_events: list[Event] = []
        self._pending_terminals: dict[int, Termination] = {}

    def drain_events(self) -> Sequence[Event]:
        """Return and clear the pending occurrences."""
        events = tuple(self._pending_events)
        self._pending_events.clear()
        return events

    def drain_terminals(self) -> Mapping[int, Termination]:
        """Return and clear the pending terminal requests."""
        terminals = dict(self._pending_terminals)
        self._pending_terminals.clear()
        return terminals

    def snapshot(self) -> JsonValue:
        """Return a JSON-safe, private copy of the monitor state."""
        state = ensure_json_value(
            self._state_fields(),
            where=f"monitor {self._name!r} snapshot.state",
        )
        if not isinstance(state, dict):
            raise TypeError(f"monitor {self._name!r} snapshot state must be an object")
        config = ensure_json_value(
            self._config_fields(),
            where=f"monitor {self._name!r} snapshot.config",
        )
        if not isinstance(config, dict):
            raise TypeError(f"monitor {self._name!r} snapshot config must be an object")
        return {
            "schema": _SNAPSHOT_SCHEMA,
            "name": self._name,
            "kind": self.kind,
            "num_envs": self._num_envs,
            "device": None if self._device is None else str(self._device),
            "config": config,
            "state": state,
            "pending_events": [event.to_json() for event in self._pending_events],
            "pending_terminals": {
                str(env_id): termination.to_json()
                for env_id, termination in self._pending_terminals.items()
            },
        }

    def validate_restore(self, state: JsonValue) -> None:
        """Pre-flight a monitor snapshot without mutating any field.

        This is the read-only half of :meth:`restore`: it applies the same
        schema, shape and config-fingerprint judgment and is used by the pack's
        rewind provider before any state is touched.
        """
        self._decode_snapshot(state)

    def restore(self, state: JsonValue) -> None:
        """Replace the monitor state from a snapshot's private copy.

        The whole snapshot is decoded and validated into locals before any
        monitor field is mutated.  A rejected snapshot therefore leaves the
        monitor exactly as it was and still usable.
        """
        decoded = self._decode_snapshot(state)
        self._apply_state(decoded.state)
        self._num_envs = decoded.num_envs
        self._device = decoded.device
        self._pending_events = list(decoded.pending_events)
        self._pending_terminals = dict(decoded.pending_terminals)

    def _decode_snapshot(self, state: JsonValue) -> _DecodedMonitorSnapshot:
        """Decode one snapshot into private locals without mutation."""
        payload = ensure_json_value(state, where=f"monitor {self._name!r} restore")
        if not isinstance(payload, dict):
            raise TypeError(
                f"monitor {self._name!r} restore: expected an object, "
                f"got {type(payload).__name__}"
            )
        where = f"monitor {self._name!r} restore"
        _require_keys(payload, _SNAPSHOT_KEYS, where)
        if payload["schema"] != _SNAPSHOT_SCHEMA:
            raise ValueError(
                f"{where}: expected schema {_SNAPSHOT_SCHEMA!r}, " f"got {payload['schema']!r}"
            )
        if payload["name"] != self._name:
            raise ValueError(f"{where}: snapshot names monitor {payload['name']!r}")
        if payload["kind"] != self.kind:
            raise ValueError(f"{where}: expected kind {self.kind!r}, got {payload['kind']!r}")
        num_envs = _decode_optional_positive_int(payload["num_envs"], f"{where}.num_envs")
        device = _decode_optional_device(payload["device"], f"{where}.device")
        if self._num_envs is not None and num_envs is not None and num_envs != self._num_envs:
            raise ValueError(
                f"{where}: snapshot num_envs {num_envs} does not match monitor "
                f"num_envs {self._num_envs}"
            )
        if self._device is not None and device is not None and device != self._device:
            raise ValueError(
                f"{where}: snapshot device {device} does not match monitor "
                f"device {self._device}"
            )
        _require_config(payload["config"], self._config_fields(), f"{where}.config")
        raw_state = payload["state"]
        if not isinstance(raw_state, dict):
            raise TypeError(f"{where}.state: expected an object, got {type(raw_state).__name__}")
        raw_events = payload["pending_events"]
        if not isinstance(raw_events, list):
            raise TypeError(
                f"{where}.pending_events: expected a list, " f"got {type(raw_events).__name__}"
            )
        events = tuple(Event.from_json(item) for item in raw_events)
        raw_terminals = payload["pending_terminals"]
        if not isinstance(raw_terminals, dict):
            raise TypeError(
                f"{where}.pending_terminals: expected an object, "
                f"got {type(raw_terminals).__name__}"
            )
        terminals: dict[int, Termination] = {}
        for key, item in raw_terminals.items():
            env_id = _decode_env_key(key, f"{where}.pending_terminals")
            terminals[env_id] = termination_from_json(item, f"{where}.pending_terminals[{key}]")
        target_num_envs = self._num_envs if num_envs is None else num_envs
        target_device = self._device if device is None else device
        if target_num_envs is not None:
            for env_id in terminals:
                if env_id >= target_num_envs:
                    raise ValueError(
                        f"{where}.pending_terminals: env id {env_id} is outside "
                        f"[0, num_envs={target_num_envs})"
                    )
            for event in events:
                for env_id in event.env_ids:
                    if env_id >= target_num_envs:
                        raise ValueError(
                            f"{where}.pending_events: env id {env_id} is outside "
                            f"[0, num_envs={target_num_envs})"
                        )
        decoded = self._decode_state(raw_state, target_num_envs)
        return _DecodedMonitorSnapshot(
            state=decoded,
            num_envs=target_num_envs,
            device=target_device,
            pending_events=events,
            pending_terminals=terminals,
        )

    def _bind_context(self, context: EnvStepContext) -> None:
        """Bind the run shape on the first tick and reject a mismatch later."""
        if self._num_envs is None:
            self._num_envs = context.num_envs
        elif context.num_envs != self._num_envs:
            raise ValueError(
                f"monitor {self._name!r}: num_envs changed from {self._num_envs} "
                f"to {context.num_envs}"
            )
        if self._device is None:
            self._device = context.device
        elif context.device != self._device:
            raise ValueError(
                f"monitor {self._name!r}: device changed from {self._device} "
                f"to {context.device}"
            )

    def _ensure_length(self, values: list[_T], default: _T) -> list[_T]:
        """Return a per-env list, initializing it on the first tick."""
        if self._num_envs is None:
            raise RuntimeError(f"monitor {self._name!r} has no bound environment count")
        if not values:
            return [default for _ in range(self._num_envs)]
        if len(values) != self._num_envs:
            raise ValueError(
                f"monitor {self._name!r}: expected {self._num_envs} per-env values, "
                f"got {len(values)}"
            )
        return values

    def _emit_event(
        self,
        *,
        name: str,
        env_id: int,
        severity: Severity,
        data: Mapping[str, JsonValue],
    ) -> None:
        """Queue one occurrence for ``env_id``."""
        self._pending_events.append(
            Event(
                name=name,
                code=f"task.{self._name}.{name}",
                env_ids=(env_id,),
                severity=severity,
                data=data,
            )
        )

    def _emit_terminal(
        self,
        env_id: int,
        *,
        outcome: EpisodeOutcome,
        reason: str,
        metrics: Mapping[str, float],
    ) -> None:
        """Queue one terminal request; the runner stamps step and time."""
        code = f"custom.{self._name}.{reason}"
        validate_code(code)
        self._pending_terminals[env_id] = Termination(
            outcome=outcome,
            code=code,
            source=TerminationSource.CUSTOM,
            metrics=metrics,
        )

    def _state_fields(self) -> Mapping[str, JsonValue]:
        """Return the built-in's private state as a JSON-safe copy."""
        raise NotImplementedError

    def _config_fields(self) -> Mapping[str, JsonValue]:
        """Return the construction config a snapshot must match."""
        raise NotImplementedError

    def _decode_state(self, state: Mapping[str, JsonValue], num_envs: int | None) -> object:
        """Validate and decode private state without mutating the monitor."""
        raise NotImplementedError

    def _apply_state(self, decoded: object) -> None:
        """Apply a decoded state bundle after every value has been validated."""
        raise NotImplementedError

    def _signal_tensor(self, values: Sequence[float], dtype: torch.dtype) -> Tensor:
        """Build a fresh per-env tensor on the bound device."""
        if self._device is None:
            raise RuntimeError(f"monitor {self._name!r} has no bound device")
        return torch.tensor(list(values), dtype=dtype, device=self._device)


@dataclass(frozen=True)
class _ThresholdState:
    active: list[bool]
    last_values: list[float]
    trigger_events: int
    active_ticks: int
    episodes: int


class ThresholdMonitor(_StatefulMonitor):
    """Emit an event and optional terminal when a level crosses a threshold.

    The signal ``active`` reports the current condition and ``level`` the last
    observed value.  One event is emitted per environment on each rising edge,
    where ``value <operator> threshold`` becomes true.  With
    ``terminal_outcome`` set, the same edge also queues a terminal.
    """

    kind = "threshold"

    def __init__(
        self,
        name: str,
        probe: Probe,
        *,
        threshold: float,
        operator: Comparison = ">=",
        terminal_outcome: EpisodeOutcome | str | None = None,
    ) -> None:
        super().__init__(name)
        where = f"monitor {self.name!r}"
        self._probe = _require_probe(probe, f"{where}.probe")
        self._threshold = _require_finite_float(threshold, f"{where}.threshold")
        self._operator = _require_comparison(operator, f"{where}.operator")
        self._terminal_outcome = _require_outcome(terminal_outcome, f"{where}.terminal_outcome")
        self._active: list[bool] = []
        self._last_values: list[float] = []
        self._trigger_events = 0
        self._active_ticks = 0
        self._episodes = 0

    def on_tick(self, context: EnvStepContext) -> None:
        """Update the condition and emit the rising edges."""
        self._bind_context(context)
        values = _read_float_probe(self._probe, context, f"monitor {self.name!r}.probe")
        self._active = self._ensure_length(self._active, False)
        self._last_values = self._ensure_length(self._last_values, 0.0)
        for env_id, value in enumerate(values):
            active = _compare(value, self._operator, self._threshold)
            if active and not self._active[env_id]:
                self._trigger_events += 1
                self._emit_event(
                    name="threshold_reached",
                    env_id=env_id,
                    severity="info",
                    data={
                        "value": value,
                        "threshold": self._threshold,
                        "operator": self._operator,
                    },
                )
                if self._terminal_outcome is not None:
                    self._emit_terminal(
                        env_id,
                        outcome=self._terminal_outcome,
                        reason="threshold",
                        metrics={"value": value, "threshold": self._threshold},
                    )
            if active:
                self._active_ticks += 1
            self._active[env_id] = active
        self._last_values = [float(value) for value in values]

    def signals(self) -> Mapping[str, Tensor]:
        """Return the current level and condition."""
        if self._device is None or not self._last_values:
            return {}
        return {
            "level": self._signal_tensor(self._last_values, torch.float32),
            "active": torch.tensor(self._active, dtype=torch.bool, device=self._device),
        }

    def on_episode_end(self, env_ids: Sequence[int], context: EnvStepContext | None = None) -> None:
        """Count observed episodes."""
        self._episodes += len(_validate_env_ids(env_ids, self._num_envs, "on_episode_end.env_ids"))

    def reset(self, env_ids: Sequence[int]) -> None:
        """Clear the condition edge state for ``env_ids``."""
        for env_id in _validate_env_ids(env_ids, self._num_envs, "reset.env_ids"):
            if env_id < len(self._active):
                self._active[env_id] = False

    def report_view(self) -> Mapping[str, JsonValue]:
        """Return the run-scoped threshold statistics."""
        return self._json_report(
            {
                "name": self.name,
                "kind": self.kind,
                "threshold": self._threshold,
                "operator": self._operator,
                "terminal_outcome": (
                    None if self._terminal_outcome is None else self._terminal_outcome.value
                ),
                "trigger_events": self._trigger_events,
                "active_ticks": self._active_ticks,
                "episodes": self._episodes,
            }
        )

    def _state_fields(self) -> Mapping[str, JsonValue]:
        return {
            "active": list(self._active),
            "last_values": list(self._last_values),
            "trigger_events": self._trigger_events,
            "active_ticks": self._active_ticks,
            "episodes": self._episodes,
        }

    def _config_fields(self) -> Mapping[str, JsonValue]:
        return {
            "threshold": self._threshold,
            "operator": self._operator,
            "terminal_outcome": (
                None if self._terminal_outcome is None else self._terminal_outcome.value
            ),
        }

    def _decode_state(
        self, state: Mapping[str, JsonValue], num_envs: int | None
    ) -> _ThresholdState:
        where = f"monitor {self.name!r} restore.state"
        _require_state_keys(
            state,
            frozenset({"active", "last_values", "trigger_events", "active_ticks", "episodes"}),
            where,
        )
        return _ThresholdState(
            active=_decode_bool_list(state["active"], f"{where}.active", num_envs),
            last_values=_decode_float_list(state["last_values"], f"{where}.last_values", num_envs),
            trigger_events=_decode_nonnegative_int(
                state["trigger_events"], f"{where}.trigger_events"
            ),
            active_ticks=_decode_nonnegative_int(state["active_ticks"], f"{where}.active_ticks"),
            episodes=_decode_nonnegative_int(state["episodes"], f"{where}.episodes"),
        )

    def _apply_state(self, decoded: object) -> None:
        if not isinstance(decoded, _ThresholdState):
            raise TypeError(
                f"monitor {self.name!r}: invalid threshold state bundle "
                f"{type(decoded).__name__}"
            )
        self._active = decoded.active
        self._last_values = decoded.last_values
        self._trigger_events = decoded.trigger_events
        self._active_ticks = decoded.active_ticks
        self._episodes = decoded.episodes


@dataclass(frozen=True)
class _HoldState:
    streak: list[int]
    reached: list[bool]
    last_values: list[float]
    holds: int
    max_streak: int
    episodes: int


class HoldMonitor(_StatefulMonitor):
    """Emit once when a level holds a condition for consecutive ticks.

    ``streak`` reports the current consecutive count and ``holding`` whether
    the current streak satisfies the hold.  One event and optional terminal
    are emitted per environment when a streak reaches ``hold_steps``; a false
    tick resets the streak and re-arms the hold.
    """

    kind = "hold"

    def __init__(
        self,
        name: str,
        probe: Probe,
        *,
        threshold: float,
        hold_steps: int,
        operator: Comparison = ">=",
        terminal_outcome: EpisodeOutcome | str | None = None,
    ) -> None:
        super().__init__(name)
        where = f"monitor {self.name!r}"
        self._probe = _require_probe(probe, f"{where}.probe")
        self._threshold = _require_finite_float(threshold, f"{where}.threshold")
        self._operator = _require_comparison(operator, f"{where}.operator")
        self._hold_steps = _require_positive_int(hold_steps, f"{where}.hold_steps")
        self._terminal_outcome = _require_outcome(terminal_outcome, f"{where}.terminal_outcome")
        self._streak: list[int] = []
        self._reached: list[bool] = []
        self._last_values: list[float] = []
        self._holds = 0
        self._max_streak = 0
        self._episodes = 0

    def on_tick(self, context: EnvStepContext) -> None:
        """Update streaks and emit the hold edges."""
        self._bind_context(context)
        values = _read_float_probe(self._probe, context, f"monitor {self.name!r}.probe")
        self._streak = self._ensure_length(self._streak, 0)
        self._reached = self._ensure_length(self._reached, False)
        self._last_values = self._ensure_length(self._last_values, 0.0)
        for env_id, value in enumerate(values):
            if not _compare(value, self._operator, self._threshold):
                self._streak[env_id] = 0
                self._reached[env_id] = False
                continue
            self._streak[env_id] += 1
            self._max_streak = max(self._max_streak, self._streak[env_id])
            if self._reached[env_id] or self._streak[env_id] < self._hold_steps:
                continue
            self._reached[env_id] = True
            self._holds += 1
            self._emit_event(
                name="hold_reached",
                env_id=env_id,
                severity="info",
                data={
                    "value": value,
                    "threshold": self._threshold,
                    "operator": self._operator,
                    "hold_steps": self._hold_steps,
                },
            )
            if self._terminal_outcome is not None:
                self._emit_terminal(
                    env_id,
                    outcome=self._terminal_outcome,
                    reason="hold",
                    metrics={
                        "value": value,
                        "threshold": self._threshold,
                        "hold_steps": float(self._hold_steps),
                    },
                )
        self._last_values = [float(value) for value in values]

    def signals(self) -> Mapping[str, Tensor]:
        """Return the level, streak and reached flag."""
        if self._device is None or not self._last_values:
            return {}
        return {
            "level": self._signal_tensor(self._last_values, torch.float32),
            "streak": torch.tensor(self._streak, dtype=torch.int64, device=self._device),
            "holding": torch.tensor(self._reached, dtype=torch.bool, device=self._device),
        }

    def on_episode_end(self, env_ids: Sequence[int], context: EnvStepContext | None = None) -> None:
        """Count observed episodes."""
        self._episodes += len(_validate_env_ids(env_ids, self._num_envs, "on_episode_end.env_ids"))

    def reset(self, env_ids: Sequence[int]) -> None:
        """Clear streaks and reached flags for ``env_ids``."""
        for env_id in _validate_env_ids(env_ids, self._num_envs, "reset.env_ids"):
            if env_id < len(self._streak):
                self._streak[env_id] = 0
                self._reached[env_id] = False

    def report_view(self) -> Mapping[str, JsonValue]:
        """Return the run-scoped hold statistics."""
        return self._json_report(
            {
                "name": self.name,
                "kind": self.kind,
                "threshold": self._threshold,
                "operator": self._operator,
                "hold_steps": self._hold_steps,
                "terminal_outcome": (
                    None if self._terminal_outcome is None else self._terminal_outcome.value
                ),
                "holds": self._holds,
                "max_streak": self._max_streak,
                "episodes": self._episodes,
            }
        )

    def _state_fields(self) -> Mapping[str, JsonValue]:
        return {
            "streak": list(self._streak),
            "reached": list(self._reached),
            "last_values": list(self._last_values),
            "holds": self._holds,
            "max_streak": self._max_streak,
            "episodes": self._episodes,
        }

    def _config_fields(self) -> Mapping[str, JsonValue]:
        return {
            "threshold": self._threshold,
            "operator": self._operator,
            "hold_steps": self._hold_steps,
            "terminal_outcome": (
                None if self._terminal_outcome is None else self._terminal_outcome.value
            ),
        }

    def _decode_state(self, state: Mapping[str, JsonValue], num_envs: int | None) -> _HoldState:
        where = f"monitor {self.name!r} restore.state"
        _require_state_keys(
            state,
            frozenset({"streak", "reached", "last_values", "holds", "max_streak", "episodes"}),
            where,
        )
        return _HoldState(
            streak=_decode_int_list(state["streak"], f"{where}.streak", num_envs),
            reached=_decode_bool_list(state["reached"], f"{where}.reached", num_envs),
            last_values=_decode_float_list(state["last_values"], f"{where}.last_values", num_envs),
            holds=_decode_nonnegative_int(state["holds"], f"{where}.holds"),
            max_streak=_decode_nonnegative_int(state["max_streak"], f"{where}.max_streak"),
            episodes=_decode_nonnegative_int(state["episodes"], f"{where}.episodes"),
        )

    def _apply_state(self, decoded: object) -> None:
        if not isinstance(decoded, _HoldState):
            raise TypeError(
                f"monitor {self.name!r}: invalid hold state bundle " f"{type(decoded).__name__}"
            )
        self._streak = decoded.streak
        self._reached = decoded.reached
        self._last_values = decoded.last_values
        self._holds = decoded.holds
        self._max_streak = decoded.max_streak
        self._episodes = decoded.episodes


@dataclass(frozen=True)
class _EventCounterState:
    previous: list[bool]
    counts: list[int]
    occurrences: int
    episodes: int


class EventCounterMonitor(_StatefulMonitor):
    """Count boolean occurrences per environment for the report.

    The probe returns a boolean level.  Each false-to-true edge increments the
    per-env ``count`` signal and the run total.  This monitor is report-only:
    it emits no events and no terminals.  ``reset`` clears edge detection but
    keeps the run-scoped counts.  ``severity`` is report metadata only;
    occurrences are counted once per rising edge and are not severity-weighted.
    """

    kind = "event_counter"

    def __init__(
        self,
        name: str,
        probe: Probe,
        *,
        severity: Severity = "info",
    ) -> None:
        super().__init__(name)
        where = f"monitor {self.name!r}"
        self._probe = _require_probe(probe, f"{where}.probe")
        self._severity = _require_severity(severity, f"{where}.severity")
        self._previous: list[bool] = []
        self._counts: list[int] = []
        self._occurrences = 0
        self._episodes = 0

    def on_tick(self, context: EnvStepContext) -> None:
        """Count rising edges."""
        self._bind_context(context)
        current = _read_bool_probe(self._probe, context, f"monitor {self.name!r}.probe")
        self._previous = self._ensure_length(self._previous, False)
        self._counts = self._ensure_length(self._counts, 0)
        for env_id, value in enumerate(current):
            if value and not self._previous[env_id]:
                self._counts[env_id] += 1
                self._occurrences += 1
            self._previous[env_id] = value

    def signals(self) -> Mapping[str, Tensor]:
        """Return the run-scoped per-env count."""
        if self._device is None or not self._counts:
            return {}
        return {"count": torch.tensor(self._counts, dtype=torch.int64, device=self._device)}

    def on_episode_end(self, env_ids: Sequence[int], context: EnvStepContext | None = None) -> None:
        """Count observed episodes."""
        self._episodes += len(_validate_env_ids(env_ids, self._num_envs, "on_episode_end.env_ids"))

    def reset(self, env_ids: Sequence[int]) -> None:
        """Clear edge detection for ``env_ids``; counts persist."""
        for env_id in _validate_env_ids(env_ids, self._num_envs, "reset.env_ids"):
            if env_id < len(self._previous):
                self._previous[env_id] = False

    def report_view(self) -> Mapping[str, JsonValue]:
        """Return the run-scoped occurrence counts."""
        return self._json_report(
            {
                "name": self.name,
                "kind": self.kind,
                "severity": self._severity,
                "occurrences": self._occurrences,
                "counts": list(self._counts),
                "episodes": self._episodes,
            }
        )

    def _state_fields(self) -> Mapping[str, JsonValue]:
        return {
            "previous": list(self._previous),
            "counts": list(self._counts),
            "occurrences": self._occurrences,
            "episodes": self._episodes,
        }

    def _config_fields(self) -> Mapping[str, JsonValue]:
        return {"severity": self._severity}

    def _decode_state(
        self, state: Mapping[str, JsonValue], num_envs: int | None
    ) -> _EventCounterState:
        where = f"monitor {self.name!r} restore.state"
        _require_state_keys(
            state, frozenset({"previous", "counts", "occurrences", "episodes"}), where
        )
        return _EventCounterState(
            previous=_decode_bool_list(state["previous"], f"{where}.previous", num_envs),
            counts=_decode_int_list(state["counts"], f"{where}.counts", num_envs),
            occurrences=_decode_nonnegative_int(state["occurrences"], f"{where}.occurrences"),
            episodes=_decode_nonnegative_int(state["episodes"], f"{where}.episodes"),
        )

    def _apply_state(self, decoded: object) -> None:
        if not isinstance(decoded, _EventCounterState):
            raise TypeError(
                f"monitor {self.name!r}: invalid event-counter state bundle "
                f"{type(decoded).__name__}"
            )
        self._previous = decoded.previous
        self._counts = decoded.counts
        self._occurrences = decoded.occurrences
        self._episodes = decoded.episodes


_PREDICATE_STATE_KEYS: frozenset[str] = frozenset(
    {
        "previous",
        "streak",
        "reached",
        "once_fired",
        "fired",
        "fired_total",
        "fired_counts",
        "episode_end_total",
        "episode_end_counts",
        "episodes",
        "episode_terminals",
    }
)


@dataclass(frozen=True)
class _PredicateState:
    """Fully validated private predicate state, held as private locals."""

    previous: list[bool]
    streak: list[int]
    reached: list[bool]
    once_fired: list[bool]
    fired: list[bool]
    fired_total: int
    fired_counts: list[int]
    episode_end_total: int
    episode_end_counts: list[int]
    episodes: int
    episode_terminals: dict[int, Termination]


class PredicateMonitor(_StatefulMonitor):
    """One condition tree, one trigger and one emit configuration.

    The monitor resolves the condition at each evaluated boundary and stores
    only the per-env booleans, streaks, once flags and run counters.  It owns no
    view, tensor or borrowed boundary value.  ``on_tick`` publishes the trigger
    firing mask.

    Terminals follow the trigger cadence and are not latched here: the engine
    owns first-terminal conflict accounting, so a repeated firing may request a
    terminal again.  ``episode_end`` predicates are evaluated only by
    :meth:`on_episode_end` with a final context.  Their terminal output is
    report/scoring-only (OD-5) and is drained through
    :meth:`drain_episode_terminals`, never through :meth:`drain_terminals`.
    P15 consumes that separate face and records it without runner-drain
    semantics.
    """

    kind = "predicate"

    def __init__(
        self,
        name: str,
        condition: Condition | Mapping[str, object],
        trigger: TriggerSpec | TriggerKind | Mapping[str, object],
        emit: EmitSpec | Mapping[str, object] | None = None,
    ) -> None:
        """Build a predicate monitor.

        Args:
            name: Monitor name matching ``[a-z][a-z0-9_]*``.
            condition: A parsed condition or a mapping accepted by
                :func:`unigym_bench.conditions.parse_condition`.  It must expose
                a JSON-safe ``to_mapping`` for the config fingerprint.
            trigger: A :class:`TriggerSpec`, trigger kind string or closed
                trigger mapping.
            emit: Optional :class:`EmitSpec`, emit mapping or ``None`` for the
                default ``fired`` level value and no event/terminal.

        Raises:
            TypeError: An argument has the wrong type or the condition has no
                JSON-safe config form.
            ValueError: The trigger or emit configuration is invalid.
        """
        super().__init__(name)
        parsed_condition = _coerce_predicate_condition(condition)
        self._condition = parsed_condition
        condition_config = ensure_json_value(
            _condition_to_mapping(parsed_condition),
            where=f"monitor {self.name!r}.condition",
        )
        if not isinstance(condition_config, dict):
            raise TypeError(f"monitor {self.name!r}: condition config must be an object")
        self._condition_config: dict[str, JsonValue] = condition_config
        self._trigger = _coerce_trigger_spec(trigger)
        self._emit = _coerce_emit_spec(emit)
        self._resolver = SourceResolver()
        self._previous: list[bool] = []
        self._streak: list[int] = []
        self._reached: list[bool] = []
        self._once_fired: list[bool] = []
        self._fired: list[bool] = []
        self._fired_total = 0
        self._fired_counts: list[int] = []
        self._episode_end_total = 0
        self._episode_end_counts: list[int] = []
        self._episodes = 0
        self._pending_episode_terminals: dict[int, Termination] = {}

    def on_tick(self, context: EnvStepContext) -> None:
        """Evaluate the condition, advance the trigger and queue outputs.

        ``episode_end`` triggers bind the run shape but never evaluate the
        condition on a normal tick, so a final-only source can be absent here.

        Args:
            context: The step boundary borrowed for this callback.

        Raises:
            ConditionError: The condition fails or returns an invalid mask.
        """
        self._bind_context(context)
        self._ensure_predicate_state()
        if self._trigger.kind == "episode_end":
            self._fired = [False for _ in range(self._num_envs or 0)]
            return
        assert isinstance(self._num_envs, int)
        mask = self._condition.evaluate(context, self._resolver)
        current = _require_predicate_mask(mask, context, f"monitor {self.name!r}.condition")
        fired = self._apply_predicate_trigger(current)
        self._fired = fired
        if any(fired):
            self._record_predicate_firing(fired)

    def signals(self) -> Mapping[str, Tensor]:
        """Return a fresh per-env boolean tensor of the current firing mask."""
        if self._device is None or not self._fired:
            return {}
        values = torch.tensor(self._fired, dtype=torch.bool, device=self._device)
        return {self._emit.value: values}

    def on_episode_end(self, env_ids: Sequence[int], context: EnvStepContext | None = None) -> None:
        """Count the episode boundary and evaluate due ``episode_end`` rows.

        Only an ``episode_end`` trigger with a supplied final context evaluates
        the condition.  Every other trigger only counts the boundary.  A fired
        ``episode_end`` row refreshes its level bit, emits a report/scoring
        event when configured and queues a report/scoring terminal in
        :meth:`drain_episode_terminals` when configured; it never queues a
        runner terminal.

        Args:
            env_ids: The due environment ids.
            context: Optional final step context owned by the caller for this
                callback.

        Raises:
            TypeError: ``context`` is not an ``EnvStepContext``.
            ValueError: An env id is invalid.
            ConditionError: The final condition evaluation fails.
        """
        ids = _validate_env_ids(env_ids, self._num_envs, "on_episode_end.env_ids")
        if context is not None:
            if not isinstance(context, EnvStepContext):
                raise TypeError(
                    "on_episode_end.context must be an EnvStepContext or None, "
                    f"got {type(context).__name__}"
                )
            self._bind_context(context)
            ids = _validate_env_ids(env_ids, self._num_envs, "on_episode_end.env_ids")
        self._episodes += len(ids)
        if self._trigger.kind != "episode_end" or context is None:
            return
        self._ensure_predicate_state()
        if not ids:
            return
        for env_id in ids:
            self._pending_episode_terminals.pop(env_id, None)
        mask = self._condition.evaluate(context, self._resolver)
        current = _require_predicate_mask(mask, context, f"monitor {self.name!r}.condition")
        for env_id in ids:
            self._fired[env_id] = current[env_id]
        fired_ids = [env_id for env_id in ids if current[env_id]]
        if not fired_ids:
            return
        assert isinstance(self._num_envs, int)
        for env_id in fired_ids:
            self._episode_end_counts[env_id] += 1
            self._episode_end_total += 1
            self._queue_episode_terminal(env_id)
        if self._emit.event is not None:
            self._pending_events.append(
                Event(
                    name=self._emit.event,
                    code=f"task.{self.name}.{self._emit.event}",
                    env_ids=tuple(fired_ids),
                    severity=self._emit.severity,
                    data=self._predicate_event_data(),
                )
            )

    def drain_episode_terminals(self) -> Mapping[int, Termination]:
        """Return and clear report/scoring terminals from episode-end rows.

        This face is separate from :meth:`drain_terminals`: the runner must
        never drain it.  P15 records these facts in the final report without
        runner-terminal semantics.  A later boundary for the same environment
        replaces an undrained request for that environment.
        """
        terminals = dict(self._pending_episode_terminals)
        self._pending_episode_terminals.clear()
        return terminals

    def reset(self, env_ids: Sequence[int]) -> None:
        """Clear per-env trigger state; run counters and pending drains persist."""
        for env_id in _validate_env_ids(env_ids, self._num_envs, "reset.env_ids"):
            if env_id < len(self._previous):
                self._previous[env_id] = False
            if env_id < len(self._streak):
                self._streak[env_id] = 0
            if env_id < len(self._reached):
                self._reached[env_id] = False
            if env_id < len(self._once_fired):
                self._once_fired[env_id] = False
            if env_id < len(self._fired):
                self._fired[env_id] = False

    def report_view(self) -> Mapping[str, JsonValue]:
        """Return the JSON-safe config, fired counts and episode-end counts."""
        outcome = self._emit.terminal_outcome
        return self._json_report(
            {
                "name": self.name,
                "kind": self.kind,
                "condition": dict(self._condition_config),
                "trigger": self._trigger.to_mapping(),
                "emit": self._emit.to_mapping(),
                "value": self._emit.value,
                "event": self._emit.event,
                "severity": self._emit.severity,
                "terminal_reason": self._emit.terminal_reason,
                "terminal_outcome": None if outcome is None else outcome.value,
                "fired": self._fired_total,
                "fired_counts": list(self._fired_counts),
                "episode_end_fired": self._episode_end_total,
                "episode_end_counts": list(self._episode_end_counts),
                "episodes": self._episodes,
            }
        )

    def _ensure_predicate_state(self) -> None:
        """Initialize every per-env list to the bound environment count."""
        self._previous = self._ensure_length(self._previous, False)
        self._streak = self._ensure_length(self._streak, 0)
        self._reached = self._ensure_length(self._reached, False)
        self._once_fired = self._ensure_length(self._once_fired, False)
        self._fired = self._ensure_length(self._fired, False)
        self._fired_counts = self._ensure_length(self._fired_counts, 0)
        self._episode_end_counts = self._ensure_length(self._episode_end_counts, 0)

    def _apply_predicate_trigger(self, current: list[bool]) -> list[bool]:
        """Advance one trigger evaluation and return the per-env firing mask."""
        kind = self._trigger.kind
        if kind == "tick":
            return list(current)
        if kind == "edge":
            fired = [active and not previous for active, previous in zip(current, self._previous)]
            self._previous = list(current)
            return fired
        if kind == "hold":
            assert self._trigger.steps is not None
            steps = self._trigger.steps
            fired = [False for _ in current]
            for env_id, active in enumerate(current):
                if not active:
                    self._streak[env_id] = 0
                    self._reached[env_id] = False
                    continue
                self._streak[env_id] += 1
                if not self._reached[env_id] and self._streak[env_id] >= steps:
                    self._reached[env_id] = True
                    fired[env_id] = True
            return fired
        if kind == "once":
            fired = [active and not already for active, already in zip(current, self._once_fired)]
            self._once_fired = [
                already or active for active, already in zip(current, self._once_fired)
            ]
            return fired
        raise ValueError(f"monitor {self.name!r}: trigger {kind!r} cannot be evaluated on a tick")

    def _record_predicate_firing(self, fired: list[bool]) -> None:
        """Count fired rows and queue this firing's events and terminals."""
        fired_ids: list[int] = []
        for env_id, active in enumerate(fired):
            if not active:
                continue
            fired_ids.append(env_id)
            self._fired_counts[env_id] += 1
            self._fired_total += 1
            if self._emit.terminal_outcome is not None:
                assert self._emit.terminal_reason is not None
                self._emit_terminal(
                    env_id,
                    outcome=self._emit.terminal_outcome,
                    reason=self._emit.terminal_reason,
                    metrics={},
                )
        if fired_ids and self._emit.event is not None:
            self._pending_events.append(
                Event(
                    name=self._emit.event,
                    code=f"task.{self.name}.{self._emit.event}",
                    env_ids=tuple(fired_ids),
                    severity=self._emit.severity,
                    data=self._predicate_event_data(),
                )
            )

    def _queue_episode_terminal(self, env_id: int) -> None:
        """Queue one report/scoring terminal for a fired episode-end row."""
        reason = self._emit.terminal_reason
        outcome = self._emit.terminal_outcome
        if reason is None or outcome is None:
            return
        code = f"custom.{self.name}.{reason}"
        validate_code(code)
        self._pending_episode_terminals[env_id] = Termination(
            outcome=outcome,
            code=code,
            source=TerminationSource.CUSTOM,
            metrics={},
        )

    def _predicate_event_data(self) -> Mapping[str, JsonValue]:
        """Return the JSON-safe event detail payload."""
        data: dict[str, JsonValue] = {"trigger": self._trigger.kind}
        if self._trigger.steps is not None:
            data["steps"] = self._trigger.steps
        return data

    def _state_fields(self) -> Mapping[str, JsonValue]:
        return {
            "previous": list(self._previous),
            "streak": list(self._streak),
            "reached": list(self._reached),
            "once_fired": list(self._once_fired),
            "fired": list(self._fired),
            "fired_total": self._fired_total,
            "fired_counts": list(self._fired_counts),
            "episode_end_total": self._episode_end_total,
            "episode_end_counts": list(self._episode_end_counts),
            "episodes": self._episodes,
            "episode_terminals": {
                str(env_id): termination.to_json()
                for env_id, termination in sorted(self._pending_episode_terminals.items())
            },
        }

    def _config_fields(self) -> Mapping[str, JsonValue]:
        return {
            "condition": dict(self._condition_config),
            "trigger": self._trigger.to_mapping(),
            "emit": self._emit.to_mapping(),
        }

    def _decode_state(
        self, state: Mapping[str, JsonValue], num_envs: int | None
    ) -> _PredicateState:
        where = f"monitor {self.name!r} restore.state"
        unknown = sorted(repr(key) for key in state if key not in _PREDICATE_STATE_KEYS)
        missing = sorted(_PREDICATE_STATE_KEYS - set(state))
        if unknown or missing:
            raise ValueError(
                f"{where}: expected keys {sorted(_PREDICATE_STATE_KEYS)}, "
                f"missing {missing}, unknown {unknown}"
            )
        raw_terminals = state["episode_terminals"]
        if not isinstance(raw_terminals, dict):
            raise TypeError(
                f"{where}.episode_terminals: expected an object, "
                f"got {type(raw_terminals).__name__}"
            )
        episode_terminals: dict[int, Termination] = {}
        for key, item in raw_terminals.items():
            env_id = _decode_env_key(key, f"{where}.episode_terminals")
            if num_envs is not None and env_id >= num_envs:
                raise ValueError(
                    f"{where}.episode_terminals: env id {env_id} is outside "
                    f"[0, num_envs={num_envs})"
                )
            episode_terminals[env_id] = termination_from_json(
                item, f"{where}.episode_terminals[{key}]"
            )
        return _PredicateState(
            previous=_decode_bool_list(state["previous"], f"{where}.previous", num_envs),
            streak=_decode_int_list(state["streak"], f"{where}.streak", num_envs),
            reached=_decode_bool_list(state["reached"], f"{where}.reached", num_envs),
            once_fired=_decode_bool_list(state["once_fired"], f"{where}.once_fired", num_envs),
            fired=_decode_bool_list(state["fired"], f"{where}.fired", num_envs),
            fired_total=_decode_nonnegative_int(state["fired_total"], f"{where}.fired_total"),
            fired_counts=_decode_int_list(state["fired_counts"], f"{where}.fired_counts", num_envs),
            episode_end_total=_decode_nonnegative_int(
                state["episode_end_total"], f"{where}.episode_end_total"
            ),
            episode_end_counts=_decode_int_list(
                state["episode_end_counts"], f"{where}.episode_end_counts", num_envs
            ),
            episodes=_decode_nonnegative_int(state["episodes"], f"{where}.episodes"),
            episode_terminals=episode_terminals,
        )

    def _apply_state(self, decoded: object) -> None:
        if not isinstance(decoded, _PredicateState):
            raise TypeError(
                f"monitor {self.name!r}: invalid predicate state bundle "
                f"{type(decoded).__name__}"
            )
        self._previous = decoded.previous
        self._streak = decoded.streak
        self._reached = decoded.reached
        self._once_fired = decoded.once_fired
        self._fired = decoded.fired
        self._fired_total = decoded.fired_total
        self._fired_counts = decoded.fired_counts
        self._episode_end_total = decoded.episode_end_total
        self._episode_end_counts = decoded.episode_end_counts
        self._episodes = decoded.episodes
        self._pending_episode_terminals = decoded.episode_terminals


@dataclass(frozen=True)
class _PhaseState:
    phase: list[int | None]
    last_values: list[float | None]
    transitions: int
    phase_ticks: list[int]
    episode_phases: list[int]
    episodes: int


class PhaseMonitor(_StatefulMonitor):
    """Map a level to a finite phase and report transitions.

    ``boundaries`` is a strictly increasing list of finite values.  A level is
    in phase ``i`` when exactly ``i`` boundaries are less than or equal to it,
    so ``[0.0, 1.0]`` yields phases ``0``, ``1`` and ``2``.  The signal
    ``phase`` is an ``int64`` tensor and ``level`` the last value.  One
    ``phase_changed`` event is emitted per environment on each transition;
    the first observation after construction or reset initializes silently.
    """

    kind = "phase"

    def __init__(
        self,
        name: str,
        probe: Probe,
        *,
        boundaries: Sequence[float],
        phase_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__(name)
        where = f"monitor {self.name!r}"
        self._probe = _require_probe(probe, f"{where}.probe")
        self._boundaries = _require_boundaries(boundaries, f"{where}.boundaries")
        self._phase_names = _require_phase_names(
            phase_names, len(self._boundaries) + 1, f"{where}.phase_names"
        )
        self._phase: list[int | None] = []
        self._last_values: list[float | None] = []
        self._transitions = 0
        self._phase_ticks = [0] * (len(self._boundaries) + 1)
        self._episode_phases = [0] * (len(self._boundaries) + 1)
        self._episodes = 0

    def on_tick(self, context: EnvStepContext) -> None:
        """Update phases and emit transitions."""
        self._bind_context(context)
        values = _read_float_probe(self._probe, context, f"monitor {self.name!r}.probe")
        self._phase = self._ensure_length(self._phase, None)
        self._last_values = self._ensure_length(self._last_values, None)
        for env_id, value in enumerate(values):
            phase = _phase_index(value, self._boundaries)
            previous = self._phase[env_id]
            if previous is not None and previous != phase:
                self._transitions += 1
                self._emit_event(
                    name="phase_changed",
                    env_id=env_id,
                    severity="info",
                    data={"from": previous, "to": phase, "value": value},
                )
            self._phase[env_id] = phase
            self._phase_ticks[phase] += 1
        self._last_values = [float(value) for value in values]

    def signals(self) -> Mapping[str, Tensor]:
        """Return the current phase and level once every env is observed."""
        if self._device is None or not self._phase or any(p is None for p in self._phase):
            return {}
        if any(v is None for v in self._last_values):
            return {}
        return {
            "level": self._signal_tensor(
                [0.0 if v is None else v for v in self._last_values], torch.float32
            ),
            "phase": torch.tensor(self._phase, dtype=torch.int64, device=self._device),
        }

    def on_episode_end(self, env_ids: Sequence[int], context: EnvStepContext | None = None) -> None:
        """Count final phases and observed episodes."""
        ids = _validate_env_ids(env_ids, self._num_envs, "on_episode_end.env_ids")
        self._episodes += len(ids)
        for env_id in ids:
            if env_id < len(self._phase):
                phase = self._phase[env_id]
                if phase is not None:
                    self._episode_phases[phase] += 1

    def reset(self, env_ids: Sequence[int]) -> None:
        """Clear the phase observation for ``env_ids``."""
        for env_id in _validate_env_ids(env_ids, self._num_envs, "reset.env_ids"):
            if env_id < len(self._phase):
                self._phase[env_id] = None
                self._last_values[env_id] = None

    def report_view(self) -> Mapping[str, JsonValue]:
        """Return the run-scoped phase statistics."""
        return self._json_report(
            {
                "name": self.name,
                "kind": self.kind,
                "boundaries": list(self._boundaries),
                "phase_names": (None if self._phase_names is None else list(self._phase_names)),
                "transitions": self._transitions,
                "phase_ticks": {
                    self._phase_key(index): ticks for index, ticks in enumerate(self._phase_ticks)
                },
                "episode_phases": {
                    self._phase_key(index): count
                    for index, count in enumerate(self._episode_phases)
                },
                "episodes": self._episodes,
            }
        )

    def _phase_key(self, index: int) -> str:
        if self._phase_names is None:
            return str(index)
        return self._phase_names[index]

    def _state_fields(self) -> Mapping[str, JsonValue]:
        return {
            "phase": list(self._phase),
            "last_values": list(self._last_values),
            "transitions": self._transitions,
            "phase_ticks": list(self._phase_ticks),
            "episode_phases": list(self._episode_phases),
            "episodes": self._episodes,
        }

    def _config_fields(self) -> Mapping[str, JsonValue]:
        return {
            "boundaries": list(self._boundaries),
            "phase_names": None if self._phase_names is None else list(self._phase_names),
        }

    def _decode_state(self, state: Mapping[str, JsonValue], num_envs: int | None) -> _PhaseState:
        where = f"monitor {self.name!r} restore.state"
        _require_state_keys(
            state,
            frozenset(
                {"phase", "last_values", "transitions", "phase_ticks", "episode_phases", "episodes"}
            ),
            where,
        )
        phase_count = len(self._boundaries) + 1
        return _PhaseState(
            phase=_decode_optional_phase_list(
                state["phase"], f"{where}.phase", num_envs, phase_count
            ),
            last_values=_decode_optional_float_list(
                state["last_values"], f"{where}.last_values", num_envs
            ),
            transitions=_decode_nonnegative_int(state["transitions"], f"{where}.transitions"),
            phase_ticks=_decode_int_list(state["phase_ticks"], f"{where}.phase_ticks", phase_count),
            episode_phases=_decode_int_list(
                state["episode_phases"], f"{where}.episode_phases", phase_count
            ),
            episodes=_decode_nonnegative_int(state["episodes"], f"{where}.episodes"),
        )

    def _apply_state(self, decoded: object) -> None:
        if not isinstance(decoded, _PhaseState):
            raise TypeError(
                f"monitor {self.name!r}: invalid phase state bundle " f"{type(decoded).__name__}"
            )
        self._phase = decoded.phase
        self._last_values = decoded.last_values
        self._transitions = decoded.transitions
        self._phase_ticks = decoded.phase_ticks
        self._episode_phases = decoded.episode_phases
        self._episodes = decoded.episodes


@dataclass(frozen=True)
class _SafetyState:
    violating: list[bool]
    last_values: list[float]
    violation_events: int
    violation_ticks: int
    episodes: int


class SafetyMonitor(_StatefulMonitor):
    """Count a violation condition and emit severity-tagged occurrences.

    The signal ``violating`` reports the current condition and ``level`` the
    last value.  One ``violation`` event with the configured severity is
    emitted per environment on each rising edge.  With ``terminal_outcome``
    set, the same edge also queues a terminal.
    """

    kind = "safety"

    def __init__(
        self,
        name: str,
        probe: Probe,
        *,
        threshold: float,
        operator: Comparison = ">=",
        severity: Severity = "violation",
        terminal_outcome: EpisodeOutcome | str | None = None,
    ) -> None:
        super().__init__(name)
        where = f"monitor {self.name!r}"
        self._probe = _require_probe(probe, f"{where}.probe")
        self._threshold = _require_finite_float(threshold, f"{where}.threshold")
        self._operator = _require_comparison(operator, f"{where}.operator")
        self._severity = _require_severity(severity, f"{where}.severity")
        self._terminal_outcome = _require_outcome(terminal_outcome, f"{where}.terminal_outcome")
        self._violating: list[bool] = []
        self._last_values: list[float] = []
        self._violation_events = 0
        self._violation_ticks = 0
        self._episodes = 0

    def on_tick(self, context: EnvStepContext) -> None:
        """Update the violation condition and emit rising edges."""
        self._bind_context(context)
        values = _read_float_probe(self._probe, context, f"monitor {self.name!r}.probe")
        self._violating = self._ensure_length(self._violating, False)
        self._last_values = self._ensure_length(self._last_values, 0.0)
        for env_id, value in enumerate(values):
            violating = _compare(value, self._operator, self._threshold)
            if violating and not self._violating[env_id]:
                self._violation_events += 1
                self._emit_event(
                    name="violation",
                    env_id=env_id,
                    severity=self._severity,
                    data={
                        "value": value,
                        "threshold": self._threshold,
                        "operator": self._operator,
                    },
                )
                if self._terminal_outcome is not None:
                    self._emit_terminal(
                        env_id,
                        outcome=self._terminal_outcome,
                        reason="violation",
                        metrics={"value": value, "threshold": self._threshold},
                    )
            if violating:
                self._violation_ticks += 1
            self._violating[env_id] = violating
        self._last_values = [float(value) for value in values]

    def signals(self) -> Mapping[str, Tensor]:
        """Return the current level and violation condition."""
        if self._device is None or not self._last_values:
            return {}
        return {
            "level": self._signal_tensor(self._last_values, torch.float32),
            "violating": torch.tensor(self._violating, dtype=torch.bool, device=self._device),
        }

    def on_episode_end(self, env_ids: Sequence[int], context: EnvStepContext | None = None) -> None:
        """Count observed episodes."""
        self._episodes += len(_validate_env_ids(env_ids, self._num_envs, "on_episode_end.env_ids"))

    def reset(self, env_ids: Sequence[int]) -> None:
        """Clear the violation edge state for ``env_ids``."""
        for env_id in _validate_env_ids(env_ids, self._num_envs, "reset.env_ids"):
            if env_id < len(self._violating):
                self._violating[env_id] = False

    def report_view(self) -> Mapping[str, JsonValue]:
        """Return the run-scoped safety statistics."""
        return self._json_report(
            {
                "name": self.name,
                "kind": self.kind,
                "threshold": self._threshold,
                "operator": self._operator,
                "severity": self._severity,
                "terminal_outcome": (
                    None if self._terminal_outcome is None else self._terminal_outcome.value
                ),
                "violation_events": self._violation_events,
                "violation_ticks": self._violation_ticks,
                "episodes": self._episodes,
            }
        )

    def _state_fields(self) -> Mapping[str, JsonValue]:
        return {
            "violating": list(self._violating),
            "last_values": list(self._last_values),
            "violation_events": self._violation_events,
            "violation_ticks": self._violation_ticks,
            "episodes": self._episodes,
        }

    def _config_fields(self) -> Mapping[str, JsonValue]:
        return {
            "threshold": self._threshold,
            "operator": self._operator,
            "severity": self._severity,
            "terminal_outcome": (
                None if self._terminal_outcome is None else self._terminal_outcome.value
            ),
        }

    def _decode_state(self, state: Mapping[str, JsonValue], num_envs: int | None) -> _SafetyState:
        where = f"monitor {self.name!r} restore.state"
        _require_state_keys(
            state,
            frozenset(
                {"violating", "last_values", "violation_events", "violation_ticks", "episodes"}
            ),
            where,
        )
        return _SafetyState(
            violating=_decode_bool_list(state["violating"], f"{where}.violating", num_envs),
            last_values=_decode_float_list(state["last_values"], f"{where}.last_values", num_envs),
            violation_events=_decode_nonnegative_int(
                state["violation_events"], f"{where}.violation_events"
            ),
            violation_ticks=_decode_nonnegative_int(
                state["violation_ticks"], f"{where}.violation_ticks"
            ),
            episodes=_decode_nonnegative_int(state["episodes"], f"{where}.episodes"),
        )

    def _apply_state(self, decoded: object) -> None:
        if not isinstance(decoded, _SafetyState):
            raise TypeError(
                f"monitor {self.name!r}: invalid safety state bundle " f"{type(decoded).__name__}"
            )
        self._violating = decoded.violating
        self._last_values = decoded.last_values
        self._violation_events = decoded.violation_events
        self._violation_ticks = decoded.violation_ticks
        self._episodes = decoded.episodes


@dataclass(frozen=True)
class _ScenarioCoverageState:
    captured: list[tuple[float, ...] | None]
    covered: list[bool]
    episode_counts: list[int]
    unmatched: int
    episodes: int


class ScenarioCoverageMonitor(_StatefulMonitor):
    """Track which scenario commands an episode started with.

    The monitor reads the observed ``cmd`` at the first tick after a reset for
    each environment, then matches that row against the declared command pool
    at the episode boundary.  Matching is exact in ``float32``: the sampler
    owns quantization.  The report is JSON-safe and contains no per-env hidden
    axis.  This monitor emits no signals, events or terminals.
    """

    kind = "scenario_coverage"

    def __init__(self, name: str, *, commands: Sequence[Sequence[float]]) -> None:
        super().__init__(name)
        where = f"monitor {self.name!r}"
        self._commands = _require_commands(commands, f"{where}.commands")
        self._dim = len(self._commands[0])
        self._index_by_row = {row: index for index, row in enumerate(self._commands)}
        self._captured: list[tuple[float, ...] | None] = []
        self._covered = [False] * len(self._commands)
        self._episode_counts = [0] * len(self._commands)
        self._unmatched = 0
        self._episodes = 0

    def on_tick(self, context: EnvStepContext) -> None:
        """Capture each environment's command at the start of its episode."""
        self._bind_context(context)
        cmd = context.cmd
        if cmd is None:
            raise ValueError(
                f"monitor {self.name!r}: scenario coverage requires context.cmd, got None"
            )
        if cmd.ndim != 2:
            raise ValueError(
                f"monitor {self.name!r}: context.cmd must be 2-D "
                f"(num_envs, {self._dim}), got shape {tuple(cmd.shape)}"
            )
        if cmd.shape != (context.num_envs, self._dim):
            raise ValueError(
                f"monitor {self.name!r}: context.cmd must have shape "
                f"({context.num_envs}, {self._dim}), got {tuple(cmd.shape)}"
            )
        if cmd.device != context.device:
            raise ValueError(
                f"monitor {self.name!r}: context.cmd device {cmd.device} does not match "
                f"context device {context.device}"
            )
        if not cmd.is_floating_point():
            raise TypeError(
                f"monitor {self.name!r}: context.cmd must be floating point, " f"got {cmd.dtype}"
            )
        self._captured = self._ensure_length(self._captured, None)
        rows = cmd.detach().to(device="cpu", dtype=torch.float32).tolist()
        for env_id in range(context.num_envs):
            if self._captured[env_id] is not None:
                continue
            captured = tuple(float(value) for value in rows[env_id])
            for item_index, number in enumerate(captured):
                if not math.isfinite(number):
                    raise ValueError(
                        f"monitor {self.name!r}: context.cmd[{env_id}][{item_index}] "
                        f"must be finite, got {number!r}"
                    )
            self._captured[env_id] = captured

    def on_episode_end(self, env_ids: Sequence[int], context: EnvStepContext | None = None) -> None:
        """Match each ending environment's start command and clear it."""
        ids = _validate_env_ids(env_ids, self._num_envs, "on_episode_end.env_ids")
        self._episodes += len(ids)
        for env_id in ids:
            captured = self._captured[env_id] if env_id < len(self._captured) else None
            if captured is None:
                self._unmatched += 1
            else:
                index = self._index_by_row.get(captured)
                if index is None:
                    self._unmatched += 1
                else:
                    self._covered[index] = True
                    self._episode_counts[index] += 1
            if env_id < len(self._captured):
                self._captured[env_id] = None

    def reset(self, env_ids: Sequence[int]) -> None:
        """Clear captured start commands for ``env_ids``."""
        for env_id in _validate_env_ids(env_ids, self._num_envs, "reset.env_ids"):
            if env_id < len(self._captured):
                self._captured[env_id] = None

    def report_view(self) -> Mapping[str, JsonValue]:
        """Return the run-scoped command coverage."""
        covered = sum(self._covered)
        pool_size = len(self._commands)
        return self._json_report(
            {
                "name": self.name,
                "kind": self.kind,
                "pool_size": pool_size,
                "covered": covered,
                "coverage_ratio": covered / pool_size,
                "episode_counts": list(self._episode_counts),
                "unmatched_episodes": self._unmatched,
                "episodes": self._episodes,
            }
        )

    def _state_fields(self) -> Mapping[str, JsonValue]:
        return {
            "captured": [None if row is None else list(row) for row in self._captured],
            "covered": list(self._covered),
            "episode_counts": list(self._episode_counts),
            "unmatched": self._unmatched,
            "episodes": self._episodes,
        }

    def _config_fields(self) -> Mapping[str, JsonValue]:
        return {"commands": [list(row) for row in self._commands]}

    def _decode_state(
        self, state: Mapping[str, JsonValue], num_envs: int | None
    ) -> _ScenarioCoverageState:
        where = f"monitor {self.name!r} restore.state"
        _require_state_keys(
            state,
            frozenset({"captured", "covered", "episode_counts", "unmatched", "episodes"}),
            where,
        )
        return _ScenarioCoverageState(
            captured=_decode_captured_commands(
                state["captured"], f"{where}.captured", num_envs, self._dim
            ),
            covered=_decode_bool_list(state["covered"], f"{where}.covered", len(self._commands)),
            episode_counts=_decode_int_list(
                state["episode_counts"], f"{where}.episode_counts", len(self._commands)
            ),
            unmatched=_decode_nonnegative_int(state["unmatched"], f"{where}.unmatched"),
            episodes=_decode_nonnegative_int(state["episodes"], f"{where}.episodes"),
        )

    def _apply_state(self, decoded: object) -> None:
        if not isinstance(decoded, _ScenarioCoverageState):
            raise TypeError(
                f"monitor {self.name!r}: invalid scenario coverage state bundle "
                f"{type(decoded).__name__}"
            )
        self._captured = decoded.captured
        self._covered = decoded.covered
        self._episode_counts = decoded.episode_counts
        self._unmatched = decoded.unmatched
        self._episodes = decoded.episodes


def _require_monitor_name(value: object) -> str:
    if not isinstance(value, str) or not _NAME_PATTERN.fullmatch(value):
        raise ValueError(f"monitor name must match [a-z][a-z0-9_]*, got {value!r}")
    return value


def _require_probe(probe: object, where: str) -> Probe:
    if not callable(probe):
        raise TypeError(f"{where}: expected a callable probe, got {type(probe).__name__}")
    return probe


def _coerce_predicate_condition(condition: Condition | Mapping[str, object]) -> Condition:
    """Normalize a parsed condition or a closed condition mapping."""
    if isinstance(condition, Mapping):
        return parse_condition(condition)
    if not isinstance(condition, Condition):
        raise TypeError(
            "PredicateMonitor.condition must satisfy Condition (an object with "
            f"evaluate) or be a condition mapping, got {type(condition).__name__}"
        )
    return condition


def _condition_to_mapping(condition: Condition) -> dict[str, JsonValue]:
    """Return the JSON-safe config form of a built-in condition tree."""
    encode = getattr(condition, "to_mapping", None)
    if not callable(encode):
        raise TypeError(
            f"PredicateMonitor.condition of type {type(condition).__name__} has no "
            "'to_mapping' method; a predicate monitor needs a JSON-safe condition "
            "config (use Comparison/All or parse_condition)"
        )
    mapping = encode()
    if not isinstance(mapping, dict):
        raise TypeError(
            f"PredicateMonitor.condition to_mapping returned "
            f"{type(mapping).__name__}, expected an object"
        )
    return mapping


def _coerce_trigger_spec(
    trigger: TriggerSpec | TriggerKind | Mapping[str, object],
) -> TriggerSpec:
    """Normalize a trigger spec, kind string or closed mapping."""
    if isinstance(trigger, TriggerSpec):
        return trigger
    if isinstance(trigger, str):
        return TriggerSpec(kind=cast(TriggerKind, trigger))
    if isinstance(trigger, Mapping):
        return TriggerSpec.from_mapping(trigger)
    raise TypeError(
        "PredicateMonitor.trigger must be a TriggerSpec, trigger kind string or "
        f"trigger mapping, got {type(trigger).__name__}"
    )


def _coerce_emit_spec(emit: EmitSpec | Mapping[str, object] | None) -> EmitSpec:
    """Normalize an emit spec, closed mapping or ``None``."""
    if emit is None:
        return EmitSpec()
    if isinstance(emit, EmitSpec):
        return emit
    if isinstance(emit, Mapping):
        return EmitSpec.from_mapping(emit)
    raise TypeError(
        "PredicateMonitor.emit must be an EmitSpec, emit mapping or None, "
        f"got {type(emit).__name__}"
    )


def _require_predicate_mask(mask: object, context: EnvStepContext, where: str) -> list[bool]:
    """Validate a condition mask and return plain per-env booleans."""
    if not isinstance(mask, Tensor):
        raise ConditionError(f"{where} returned {type(mask).__name__}, expected a torch.Tensor")
    expected = (context.num_envs,)
    if tuple(mask.shape) != expected:
        raise ConditionError(f"{where} returned shape {tuple(mask.shape)}, expected {expected}")
    if mask.dtype != torch.bool:
        raise ConditionError(f"{where} returned dtype {mask.dtype}, expected torch.bool")
    if mask.device != context.device:
        raise ConditionError(f"{where} returned device {mask.device}, expected {context.device}")
    return [bool(value) for value in mask.tolist()]


def _require_finite_float(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{where}: expected a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{where}: expected a finite number, got {value!r}")
    return number


def _require_positive_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where}: expected an int, got {value!r}")
    if value < 1:
        raise ValueError(f"{where}: expected an int >= 1, got {value}")
    return value


def _require_comparison(value: object, where: str) -> Comparison:
    if not isinstance(value, str) or value not in _COMPARISONS:
        raise ValueError(f"{where}: expected one of {sorted(_COMPARISONS)}, got {value!r}")
    if value == ">":
        return ">"
    if value == ">=":
        return ">="
    if value == "<":
        return "<"
    return "<="


def _require_outcome(value: object, where: str) -> EpisodeOutcome | None:
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


def _require_severity(value: object, where: str) -> Severity:
    if not isinstance(value, str) or value not in SEVERITIES:
        raise ValueError(f"{where}: expected one of {sorted(SEVERITIES)}, got {value!r}")
    if value == "info":
        return "info"
    if value == "warning":
        return "warning"
    if value == "violation":
        return "violation"
    return "fatal"


def _require_boundaries(value: object, where: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{where}: expected a sequence of numbers, got {type(value).__name__}")
    boundaries = tuple(
        _require_finite_float(item, f"{where}[{index}]") for index, item in enumerate(value)
    )
    if not boundaries:
        raise ValueError(f"{where}: expected at least one boundary")
    for index in range(1, len(boundaries)):
        if boundaries[index] <= boundaries[index - 1]:
            raise ValueError(
                f"{where}: expected strictly increasing boundaries, got {list(boundaries)}"
            )
    return boundaries


def _require_phase_names(value: object, expected: int, where: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{where}: expected a sequence of strings or None, got {value!r}")
    names = tuple(value)
    if len(names) != expected:
        raise ValueError(f"{where}: expected {expected} names, got {len(names)}")
    for index, name in enumerate(names):
        if not isinstance(name, str) or not name:
            raise TypeError(f"{where}[{index}]: expected a non-empty string, got {name!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"{where}: expected unique names, got {list(names)}")
    return names


def _require_commands(value: object, where: str) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(
            f"{where}: expected a sequence of command vectors, got {type(value).__name__}"
        )
    rows = list(value)
    if not rows:
        raise ValueError(f"{where}: expected at least one command vector")
    commands: list[tuple[float, ...]] = []
    dimension: int | None = None
    for row_index, row in enumerate(rows):
        row_where = f"{where}[{row_index}]"
        if isinstance(row, (str, bytes, bytearray)) or not isinstance(row, Sequence):
            raise TypeError(f"{row_where}: expected a command vector, got {type(row).__name__}")
        numbers = tuple(
            _require_finite_float(item, f"{row_where}[{item_index}]")
            for item_index, item in enumerate(row)
        )
        if dimension is None:
            dimension = len(numbers)
            if dimension == 0:
                raise ValueError(f"{row_where}: command vector must not be empty")
        elif len(numbers) != dimension:
            raise ValueError(
                f"{row_where}: expected {dimension} values to match {where}[0], "
                f"got {len(numbers)}"
            )
        quantized = torch.tensor(numbers, dtype=torch.float32).tolist()
        for item_index, value in enumerate(quantized):
            if not math.isfinite(value):
                raise ValueError(
                    f"{row_where}[{item_index}]: value {numbers[item_index]!r} is not "
                    "finite after float32 quantization"
                )
        commands.append(tuple(float(value) for value in quantized))
    if len(set(commands)) != len(commands):
        raise ValueError(f"{where}: expected unique command vectors, got duplicates")
    return tuple(commands)


def _read_float_probe(probe: Probe, context: EnvStepContext, where: str) -> list[float]:
    value = probe(context)
    if not isinstance(value, Tensor):
        raise TypeError(f"{where}: expected a torch.Tensor, got {type(value).__name__}")
    if value.ndim != 1 or value.shape[0] != context.num_envs:
        raise ValueError(
            f"{where}: expected shape (num_envs={context.num_envs},), got {tuple(value.shape)}"
        )
    if value.device != context.device:
        raise ValueError(
            f"{where}: probe device {value.device} does not match context device {context.device}"
        )
    if not value.is_floating_point():
        raise TypeError(f"{where}: expected a floating tensor, got {value.dtype}")
    values = [float(item) for item in value.detach().to(device="cpu").tolist()]
    for index, number in enumerate(values):
        if not math.isfinite(number):
            raise ValueError(f"{where}[{index}]: expected a finite level, got {number!r}")
    return values


def _read_bool_probe(probe: Probe, context: EnvStepContext, where: str) -> list[bool]:
    value = probe(context)
    if not isinstance(value, Tensor):
        raise TypeError(f"{where}: expected a torch.Tensor, got {type(value).__name__}")
    if value.ndim != 1 or value.shape[0] != context.num_envs:
        raise ValueError(
            f"{where}: expected shape (num_envs={context.num_envs},), got {tuple(value.shape)}"
        )
    if value.device != context.device:
        raise ValueError(
            f"{where}: probe device {value.device} does not match context device {context.device}"
        )
    if value.dtype != torch.bool:
        raise TypeError(f"{where}: expected a bool tensor, got {value.dtype}")
    return [bool(item) for item in value.detach().to(device="cpu").tolist()]


def _compare(value: float, operator: Comparison, threshold: float) -> bool:
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    return value <= threshold


def _phase_index(value: float, boundaries: tuple[float, ...]) -> int:
    index = 0
    for boundary in boundaries:
        if value >= boundary:
            index += 1
        else:
            break
    return index


def _validate_env_ids(env_ids: Sequence[int], num_envs: int | None, where: str) -> tuple[int, ...]:
    if isinstance(env_ids, (str, bytes, bytearray)) or not isinstance(env_ids, Sequence):
        raise TypeError(f"{where}: expected a sequence of ints, got {type(env_ids).__name__}")
    ids: list[int] = []
    for index, env_id in enumerate(env_ids):
        if isinstance(env_id, bool) or not isinstance(env_id, int):
            raise TypeError(f"{where}[{index}]: expected an int, got {env_id!r}")
        if env_id < 0:
            raise ValueError(f"{where}[{index}]: expected an id >= 0, got {env_id}")
        if num_envs is not None and env_id >= num_envs:
            raise ValueError(f"{where}[{index}]: id {env_id} is outside [0, num_envs={num_envs})")
        ids.append(env_id)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{where}: expected unique ids, got {tuple(ids)}")
    return tuple(ids)


def _require_keys(document: Mapping[str, JsonValue], expected: frozenset[str], where: str) -> None:
    missing = sorted(expected - document.keys())
    unknown = sorted(document.keys() - expected)
    if missing or unknown:
        raise ValueError(
            f"{where}: expected keys {sorted(expected)}, missing {missing}, unknown {unknown}"
        )


def _require_state_keys(
    state: Mapping[str, JsonValue], expected: frozenset[str], where: str
) -> None:
    _require_keys(state, expected, where)


def _require_config(raw: JsonValue, current: Mapping[str, JsonValue], where: str) -> None:
    if not isinstance(raw, dict):
        raise TypeError(f"{where}: expected an object, got {type(raw).__name__}")
    expected = ensure_json_value(current, where=f"{where}.current")
    if not isinstance(expected, dict):
        raise TypeError(f"{where}: monitor config must normalize to an object")
    if raw == expected:
        return
    keys = sorted(set(raw) | set(expected))
    mismatched = [key for key in keys if raw.get(key) != expected.get(key)]
    raise ValueError(
        f"{where}: monitor config does not match the snapshot in field(s) "
        f"{mismatched}; expected {expected!r}, got {raw!r}"
    )


def _decode_optional_positive_int(value: JsonValue, where: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where}: expected a positive int or None, got {value!r}")
    if value < 1:
        raise ValueError(f"{where}: expected a positive int, got {value}")
    return value


def _decode_optional_device(value: JsonValue, where: str) -> torch.device | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"{where}: expected a device string or None, got {value!r}")
    try:
        return torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"{where}: invalid device string {value!r}: {exc}") from exc


def _decode_nonnegative_int(value: JsonValue, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where}: expected an int, got {value!r}")
    if value < 0:
        raise ValueError(f"{where}: expected an int >= 0, got {value}")
    return value


def _decode_state_list(value: JsonValue, where: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise TypeError(f"{where}: expected a list, got {type(value).__name__}")
    return value


def _expect_length(items: Sequence[JsonValue], expected: int | None, where: str) -> None:
    target = 0 if expected is None else expected
    if len(items) != target:
        raise ValueError(f"{where}: expected {target} values, got {len(items)}")


def _decode_bool_list(value: JsonValue, where: str, expected: int | None) -> list[bool]:
    items = _decode_state_list(value, where)
    _expect_length(items, expected, where)
    result: list[bool] = []
    for index, item in enumerate(items):
        if type(item) is not bool:
            raise TypeError(f"{where}[{index}]: expected a bool, got {item!r}")
        result.append(item)
    return result


def _decode_int_list(value: JsonValue, where: str, expected: int | None) -> list[int]:
    items = _decode_state_list(value, where)
    _expect_length(items, expected, where)
    result: list[int] = []
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{where}[{index}]: expected an int, got {item!r}")
        if item < 0:
            raise ValueError(f"{where}[{index}]: expected an int >= 0, got {item}")
        result.append(item)
    return result


def _decode_float_list(value: JsonValue, where: str, expected: int | None) -> list[float]:
    items = _decode_state_list(value, where)
    _expect_length(items, expected, where)
    result: list[float] = []
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{where}[{index}]: expected a number, got {item!r}")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{where}[{index}]: expected a finite number, got {item!r}")
        result.append(number)
    return result


def _decode_optional_int_list(
    value: JsonValue, where: str, expected: int | None
) -> list[int | None]:
    items = _decode_state_list(value, where)
    _expect_length(items, expected, where)
    result: list[int | None] = []
    for index, item in enumerate(items):
        if item is None:
            result.append(None)
            continue
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{where}[{index}]: expected an int or None, got {item!r}")
        if item < 0:
            raise ValueError(f"{where}[{index}]: expected an int >= 0, got {item}")
        result.append(item)
    return result


def _decode_optional_float_list(
    value: JsonValue, where: str, expected: int | None
) -> list[float | None]:
    items = _decode_state_list(value, where)
    _expect_length(items, expected, where)
    result: list[float | None] = []
    for index, item in enumerate(items):
        if item is None:
            result.append(None)
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{where}[{index}]: expected a number or None, got {item!r}")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{where}[{index}]: expected a finite number, got {item!r}")
        result.append(number)
    return result


def _decode_optional_phase_list(
    value: JsonValue, where: str, expected: int | None, phase_count: int
) -> list[int | None]:
    phases = _decode_optional_int_list(value, where, expected)
    for index, phase in enumerate(phases):
        if phase is not None and phase >= phase_count:
            raise ValueError(f"{where}[{index}]: phase {phase} is outside [0, {phase_count})")
    return phases


def _decode_captured_commands(
    value: JsonValue, where: str, expected: int | None, dimension: int
) -> list[tuple[float, ...] | None]:
    items = _decode_state_list(value, where)
    _expect_length(items, expected, where)
    result: list[tuple[float, ...] | None] = []
    for index, item in enumerate(items):
        if item is None:
            result.append(None)
            continue
        row = _decode_state_list(item, f"{where}[{index}]")
        if len(row) != dimension:
            raise ValueError(f"{where}[{index}]: expected {dimension} values, got {len(row)}")
        values: list[float] = []
        for item_index, entry in enumerate(row):
            if isinstance(entry, bool) or not isinstance(entry, (int, float)):
                raise TypeError(f"{where}[{index}][{item_index}]: expected a number, got {entry!r}")
            number = float(entry)
            if not math.isfinite(number):
                raise ValueError(
                    f"{where}[{index}][{item_index}]: expected a finite number, got {entry!r}"
                )
            values.append(number)
        result.append(tuple(values))
    return result


def _decode_env_key(key: str, where: str) -> int:
    if not key.isdigit() or str(int(key)) != key:
        raise ValueError(f"{where}: expected a non-negative integer key, got {key!r}")
    return int(key)
