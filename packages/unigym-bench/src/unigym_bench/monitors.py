"""Monitor protocol, convenience base and the configured monitor helpers.

A monitor is the stateful per-env unit of a benchmark task.  It observes ticks
and publishes per-env level values.  It produces no events and no terminals:
those decisions belong to :class:`unigym_bench.flow.DefaultBenchmarkFlow`.  This
module owns the monitor contract, the compositional :class:`PredicateMonitor`,
the specialized :class:`PhaseMonitor` and the configuration helpers that replace
the retired typed monitor classes; it owns no engine, sink, report or scenario
code.

Ownership boundaries:

- A monitor owns observation and trigger cadence only.  Its trigger collapses
  into a boolean value; a flow rule consumes that value as-is.
- A monitor never constructs an :class:`~unigym.core.episode_termination.Termination`.
  The flow owns the ``custom.<monitor>.<reason>`` decision; the engine owns the
  first-terminal latch for an environment.
- ``values()`` returns fresh per-env level tensors with a leading dimension of
  ``num_envs``.  A monitor stores plain Python measurements, never tensors or
  views, so it cannot retain boundary data.
- ``reset(env_ids)`` clears per-env measurement state (edges, streaks, phases).
  Run-scoped report accumulators persist.
- A stateful monitor implements :class:`RewindableMonitor`; ``snapshot()`` is
  JSON-safe and both directions return private copies.

A value-driven monitor names its input with an explicit
:class:`~unigym_bench.sources.SourceRef`; the monitor resolves it through a
:class:`~unigym_bench.sources.SourceResolver` at the boundary.  There is no
callable input and no fallback lookup, so a monitor cannot silently read the
wrong quantity.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, TypeVar, cast, runtime_checkable

import torch
from torch import Tensor
from unigym.evaluation import EnvStepContext

from .conditions import Condition, ConditionError, parse_condition
from .json import JsonValue, ensure_json_value
from .predicates import EmitSpec, TriggerKind, TriggerSpec
from .sources import SourceError, SourceRef, SourceResolver

__all__ = [
    "BaseMonitor",
    "Comparison",
    "Monitor",
    "PhaseMonitor",
    "PredicateMonitor",
    "RewindableMonitor",
    "event_counter_config",
    "hold_config",
    "safety_config",
    "threshold_config",
]

#: Comparison operator accepted by the configuration helpers.
Comparison: TypeAlias = Literal[">", ">=", "<", "<="]

_CONDITION_OPS: Mapping[str, str] = {
    ">": "gt",
    ">=": "gte",
    "<": "lt",
    "<=": "lte",
}

_SNAPSHOT_SCHEMA = "unigym_bench.monitor/4"
_SNAPSHOT_KEYS = frozenset(
    {
        "schema",
        "name",
        "kind",
        "num_envs",
        "device",
        "config",
        "state",
    }
)
_COMPARISONS: frozenset[str] = frozenset({">", ">=", "<", "<="})
_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")

_T = TypeVar("_T")


@dataclass(frozen=True)
class _DecodedMonitorSnapshot:
    """One fully validated monitor snapshot, held as private locals."""

    state: object
    num_envs: int | None
    device: torch.device | None


@runtime_checkable
class Monitor(Protocol):
    """Stateful per-env unit of a benchmark task."""

    name: str

    def on_tick(self, context: EnvStepContext) -> None:
        """Observe one tick."""

    def values(self) -> Mapping[str, Tensor]:
        """Return fresh per-env level tensors keyed by local value name."""

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

    def values(self) -> Mapping[str, Tensor]:
        """Return no values."""
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
    """Shared JSON state and snapshot plumbing for built-in monitors."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._num_envs: int | None = None
        self._device: torch.device | None = None

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
        target_num_envs = self._num_envs if num_envs is None else num_envs
        target_device = self._device if device is None else device
        decoded = self._decode_state(raw_state, target_num_envs)
        return _DecodedMonitorSnapshot(
            state=decoded,
            num_envs=target_num_envs,
            device=target_device,
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


def threshold_config(
    name: str,
    source: SourceRef | str,
    *,
    threshold: float,
    operator: Comparison = ">=",
) -> PredicateMonitor:
    """Configure a threshold predicate: an edge-triggered comparison.

    The old ``ThresholdMonitor`` is now one :class:`PredicateMonitor`: a
    single-source ``Comparison`` condition, an ``edge`` trigger and a value
    named ``threshold_reached``.  A runner terminal is a flow decision, not a
    monitor output.

    Args:
        name: Monitor name matching ``[a-z][a-z0-9_]*``.
        source: Per-env numeric source reference (``SourceRef`` or ``kind:key``).
        threshold: Finite comparison threshold.
        operator: One of ``>``, ``>=``, ``<`` or ``<=``.

    Returns:
        A validated predicate monitor.
    """
    where = f"threshold_config({name!r})"
    ref = _coerce_source_ref(source, f"{where}.source")
    level = _require_finite_float(threshold, f"{where}.threshold")
    comparison = _require_comparison(operator, f"{where}.operator")
    return PredicateMonitor(
        name,
        _comparison_mapping(ref, comparison, level),
        TriggerSpec(kind="edge"),
        EmitSpec(value="threshold_reached"),
    )


def hold_config(
    name: str,
    source: SourceRef | str,
    *,
    threshold: float,
    hold_steps: int,
    operator: Comparison = ">=",
) -> PredicateMonitor:
    """Configure a hold predicate: fire once after consecutive true ticks.

    The old ``HoldMonitor`` becomes a ``Comparison`` condition plus a ``hold``
    trigger and a value named ``hold_reached``.  A runner terminal is a flow
    decision, not a monitor output.

    Args:
        name: Monitor name matching ``[a-z][a-z0-9_]*``.
        source: Per-env numeric source reference.
        threshold: Finite comparison threshold.
        hold_steps: Positive number of consecutive true evaluations.
        operator: One of ``>``, ``>=``, ``<`` or ``<=``.

    Returns:
        A validated predicate monitor.
    """
    where = f"hold_config({name!r})"
    ref = _coerce_source_ref(source, f"{where}.source")
    level = _require_finite_float(threshold, f"{where}.threshold")
    comparison = _require_comparison(operator, f"{where}.operator")
    steps = _require_positive_int(hold_steps, f"{where}.hold_steps")
    return PredicateMonitor(
        name,
        _comparison_mapping(ref, comparison, level),
        TriggerSpec(kind="hold", steps=steps),
        EmitSpec(value="hold_reached"),
    )


def safety_config(
    name: str,
    source: SourceRef | str,
    *,
    threshold: float,
    operator: Comparison = ">=",
) -> PredicateMonitor:
    """Configure a safety violation predicate: an edge-triggered comparison.

    The old ``SafetyMonitor`` becomes a ``Comparison`` condition, an ``edge``
    trigger and a value named ``violation``.  A runner terminal is a flow
    decision, not a monitor output.

    Args:
        name: Monitor name matching ``[a-z][a-z0-9_]*``.
        source: Per-env numeric source reference.
        threshold: Finite comparison threshold.
        operator: One of ``>``, ``>=``, ``<`` or ``<=``.

    Returns:
        A validated predicate monitor.
    """
    where = f"safety_config({name!r})"
    ref = _coerce_source_ref(source, f"{where}.source")
    level = _require_finite_float(threshold, f"{where}.threshold")
    comparison = _require_comparison(operator, f"{where}.operator")
    return PredicateMonitor(
        name,
        _comparison_mapping(ref, comparison, level),
        TriggerSpec(kind="edge"),
        EmitSpec(value="violation"),
    )


def event_counter_config(
    name: str,
    source: SourceRef | str,
) -> PredicateMonitor:
    """Configure a report-only occurrence counter: edge-triggered true values.

    The old ``EventCounterMonitor`` becomes an ``edge`` predicate over a
    boolean-like source: a value greater than zero is true.  It publishes a
    value named ``counted``; it emits no event and no terminal.  The run report
    keeps the per-env firing counts.

    Args:
        name: Monitor name matching ``[a-z][a-z0-9_]*``.
        source: Per-env boolean or numeric source reference.

    Returns:
        A validated predicate monitor.
    """
    where = f"event_counter_config({name!r})"
    ref = _coerce_source_ref(source, f"{where}.source")
    return PredicateMonitor(
        name,
        _comparison_mapping(ref, ">", 0.0),
        TriggerSpec(kind="edge"),
        EmitSpec(value="counted"),
    )


def _comparison_mapping(
    ref: SourceRef,
    operator: Comparison,
    threshold: float,
) -> dict[str, JsonValue]:
    """Return the closed condition mapping for one source comparison."""
    return {
        "source": str(ref),
        "op": _CONDITION_OPS[operator],
        "value": float(threshold),
    }


def _coerce_source_ref(value: SourceRef | str, where: str) -> SourceRef:
    """Normalize one explicit source reference without a fallback lookup."""
    if isinstance(value, SourceRef):
        return value
    if isinstance(value, str):
        try:
            return SourceRef.parse(value)
        except SourceError as exc:
            raise ValueError(f"{where}: {exc}") from exc
    raise TypeError(f"{where}: expected a SourceRef or 'kind:key' string, got {value!r}")


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


class PredicateMonitor(_StatefulMonitor):
    """One condition tree, one trigger and one emit configuration.

    The monitor resolves the condition at each evaluated boundary and stores
    only the per-env booleans, streaks, once flags and run counters.  It owns no
    view, tensor or borrowed boundary value.  ``on_tick`` publishes the trigger
    firing mask as an observation.

    Terminals and events are flow decisions, never monitor outputs.  A repeated
    firing therefore produces a repeated true value; the flow applies its
    first-match rule and the engine owns the first-terminal latch.
    ``episode_end`` predicates are evaluated only by :meth:`on_episode_end`
    with a final context and refresh the value for due rows.
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

    def on_tick(self, context: EnvStepContext) -> None:
        """Evaluate the condition and advance the trigger.

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

    def values(self) -> Mapping[str, Tensor]:
        """Return a fresh per-env boolean tensor of the current firing mask."""
        if self._device is None or not self._fired:
            return {}
        values = torch.tensor(self._fired, dtype=torch.bool, device=self._device)
        return {self._emit.value: values}

    def on_episode_end(self, env_ids: Sequence[int], context: EnvStepContext | None = None) -> None:
        """Count the episode boundary and evaluate due ``episode_end`` rows.

        Only an ``episode_end`` trigger with a supplied final context evaluates
        the condition.  Every other trigger only counts the boundary.  A fired
        ``episode_end`` row refreshes its value; it produces no event and no
        terminal.

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

    def reset(self, env_ids: Sequence[int]) -> None:
        """Clear per-env trigger state; run counters persist."""
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
        return self._json_report(
            {
                "name": self.name,
                "kind": self.kind,
                "condition": dict(self._condition_config),
                "trigger": self._trigger.to_mapping(),
                "emit": self._emit.to_mapping(),
                "value": self._emit.value,
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
        """Count fired rows."""
        for env_id, active in enumerate(fired):
            if not active:
                continue
            self._fired_counts[env_id] += 1
            self._fired_total += 1

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


@dataclass(frozen=True)
class _PhaseState:
    phase: list[int | None]
    last_values: list[float | None]
    changed: list[bool]
    transitions: int
    phase_ticks: list[int]
    episode_phases: list[int]
    episodes: int


class PhaseMonitor(_StatefulMonitor):
    """Map one sourced level vector to a finite phase and report transitions.

    ``boundaries`` is a strictly increasing list of finite values.  A level is
    in phase ``i`` when exactly ``i`` boundaries are less than or equal to it,
    so ``[0.0, 1.0]`` yields phases ``0``, ``1`` and ``2``.  The values are
    ``phase`` (``int64``), ``level`` (the last value) and ``changed`` (a boolean
    transition mask).  The first observation after construction or reset
    initializes silently, so ``changed`` is false there.  A flow event rule may
    consume ``changed``; the monitor itself emits nothing.

    The level is resolved from the explicit ``source`` reference through a
    :class:`~unigym_bench.sources.SourceResolver` on each tick.  There is no
    callable input and the monitor retains only plain Python measurements.
    """

    kind = "phase"

    def __init__(
        self,
        name: str,
        source: SourceRef | str,
        *,
        boundaries: Sequence[float],
        phase_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__(name)
        where = f"monitor {self.name!r}"
        self._source = _coerce_source_ref(source, f"{where}.source")
        self._resolver = SourceResolver()
        self._boundaries = _require_boundaries(boundaries, f"{where}.boundaries")
        self._phase_names = _require_phase_names(
            phase_names, len(self._boundaries) + 1, f"{where}.phase_names"
        )
        self._phase: list[int | None] = []
        self._last_values: list[float | None] = []
        self._changed: list[bool] = []
        self._transitions = 0
        self._phase_ticks = [0] * (len(self._boundaries) + 1)
        self._episode_phases = [0] * (len(self._boundaries) + 1)
        self._episodes = 0

    def on_tick(self, context: EnvStepContext) -> None:
        """Update phases and the transition mask."""
        self._bind_context(context)
        values = _read_phase_source(self._resolver, self._source, context, f"monitor {self.name!r}")
        self._phase = self._ensure_length(self._phase, None)
        self._last_values = self._ensure_length(self._last_values, None)
        self._changed = self._ensure_length(self._changed, False)
        changed: list[bool] = []
        for env_id, value in enumerate(values):
            phase = _phase_index(value, self._boundaries)
            previous = self._phase[env_id]
            is_changed = previous is not None and previous != phase
            if is_changed:
                self._transitions += 1
            changed.append(is_changed)
            self._phase[env_id] = phase
            self._phase_ticks[phase] += 1
        self._last_values = [float(value) for value in values]
        self._changed = changed

    def values(self) -> Mapping[str, Tensor]:
        """Return the current phase, level and transition mask."""
        if self._device is None or not self._phase or any(p is None for p in self._phase):
            return {}
        if any(v is None for v in self._last_values):
            return {}
        return {
            "level": self._signal_tensor(
                [0.0 if v is None else v for v in self._last_values], torch.float32
            ),
            "phase": torch.tensor(self._phase, dtype=torch.int64, device=self._device),
            "changed": torch.tensor(self._changed, dtype=torch.bool, device=self._device),
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
            if env_id < len(self._changed):
                self._changed[env_id] = False

    def report_view(self) -> Mapping[str, JsonValue]:
        """Return the run-scoped phase statistics."""
        return self._json_report(
            {
                "name": self.name,
                "kind": self.kind,
                "source": str(self._source),
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
            "changed": list(self._changed),
            "transitions": self._transitions,
            "phase_ticks": list(self._phase_ticks),
            "episode_phases": list(self._episode_phases),
            "episodes": self._episodes,
        }

    def _config_fields(self) -> Mapping[str, JsonValue]:
        return {
            "source": str(self._source),
            "boundaries": list(self._boundaries),
            "phase_names": None if self._phase_names is None else list(self._phase_names),
        }

    def _decode_state(self, state: Mapping[str, JsonValue], num_envs: int | None) -> _PhaseState:
        where = f"monitor {self.name!r} restore.state"
        _require_state_keys(
            state,
            frozenset(
                {
                    "phase",
                    "last_values",
                    "changed",
                    "transitions",
                    "phase_ticks",
                    "episode_phases",
                    "episodes",
                }
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
            changed=_decode_bool_list(state["changed"], f"{where}.changed", num_envs),
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
        self._changed = decoded.changed
        self._transitions = decoded.transitions
        self._phase_ticks = decoded.phase_ticks
        self._episode_phases = decoded.episode_phases
        self._episodes = decoded.episodes


def _require_monitor_name(value: object) -> str:
    if not isinstance(value, str) or not _NAME_PATTERN.fullmatch(value):
        raise ValueError(f"monitor name must match [a-z][a-z0-9_]*, got {value!r}")
    return value


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


def _read_phase_source(
    resolver: SourceResolver,
    source: SourceRef,
    context: EnvStepContext,
    where: str,
) -> list[float]:
    """Resolve and validate one per-env floating vector for the phase monitor."""
    value = resolver.resolve(context, source)
    if not isinstance(value, Tensor):
        raise TypeError(
            f"{where}: source {source} resolved to {type(value).__name__}, "
            "expected a per-env torch.Tensor"
        )
    if value.ndim != 1 or value.shape[0] != context.num_envs:
        raise ValueError(
            f"{where}: expected shape (num_envs={context.num_envs},), got {tuple(value.shape)}"
        )
    if value.device != context.device:
        raise ValueError(
            f"{where}: source device {value.device} does not match "
            f"context device {context.device}"
        )
    if not value.is_floating_point():
        raise TypeError(f"{where}: expected a floating tensor, got {value.dtype}")
    values = [float(item) for item in value.detach().to(device="cpu").tolist()]
    for index, number in enumerate(values):
        if not math.isfinite(number):
            raise ValueError(f"{where}[{index}]: expected a finite level, got {number!r}")
    return values


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
