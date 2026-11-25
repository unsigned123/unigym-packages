"""The stable benchmark-flow seam and its default implementation.

A :class:`BenchmarkFlow` owns benchmark policy: it reads one immutable
:class:`BenchmarkTick`, decides which events and terminals apply, observes
episode boundaries, returns episode evidence and extension invocation, and
returns the pack report at run end.  The flow never mutates the engine, a
monitor or an environment; the sink and engine validate every decision before
changing run state.  Extension persistence and rewind stay engine-owned; only
lifecycle invocation moves to the flow.  A ``stateful=True`` flow implements
``snapshot()``/``restore()``; the sink binds it to the engine so its policy
state joins the single ``unigym_bench`` rewind provider.

Monitors publish values only.  :class:`DefaultBenchmarkFlow` applies explicit
:class:`TerminationRule` declarations to those values (first true rule per env
wins) and optional :class:`EventRule` declarations.  There is no hidden
``monitor active => terminal`` fallback and no flow-level trigger: the monitor
trigger is the only cadence owner.  Empty rules mean no automatic termination.

This module deliberately imports only pack-side context/event/report modules and
the core termination type; a custom flow must be able to depend on these types
without importing the engine, the sink or composition code.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, cast, runtime_checkable

import torch
from torch import Tensor
from unigym.core.episode_termination import (
    EpisodeOutcome,
    Termination,
    TerminationSource,
    validate_code,
)

from .context import BenchmarkStepContext
from .events import SEVERITIES, Event, EpisodeRecord, EpisodeSummary, Severity
from .json import JsonValue, ensure_json_value
from .query import TickQuery
from .report import ReportBuilder, ReportExtension, RunMetadata, RunSummary
from .specs import RetentionSpec

__all__ = [
    "BenchmarkFlow",
    "BenchmarkTick",
    "DefaultBenchmarkFlow",
    "EpisodeBatch",
    "EpisodeEvidence",
    "EventRule",
    "FlowDecision",
    "FlowRun",
    "FlowSequence",
    "RewindableFlow",
    "TerminationRule",
    "validate_flow",
]

_MONITOR_OUTCOMES: frozenset[EpisodeOutcome] = frozenset(
    {EpisodeOutcome.SUCCESS, EpisodeOutcome.FAILURE, EpisodeOutcome.DEADLINE}
)
_RULE_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True, slots=True)
class _MonitorTickOutput:
    """One monitor's immutable contribution to a benchmark tick.

    Attributes:
        monitor: The monitor name.
        values: Fresh per-env value tensors keyed by local value name.
    """

    monitor: str
    values: Mapping[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and copy one monitor contribution."""
        if not isinstance(self.monitor, str) or not self.monitor:
            raise TypeError(
                f"monitor tick output.monitor: expected a non-empty str, got {self.monitor!r}"
            )
        if not isinstance(self.values, Mapping):
            raise TypeError(
                "monitor tick output.values: expected a mapping, "
                f"got {type(self.values).__name__}"
            )
        values: dict[str, Tensor] = {}
        for name, value in self.values.items():
            if not isinstance(name, str) or not name:
                raise TypeError(
                    "monitor tick output.values: names must be non-empty strings, " f"got {name!r}"
                )
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"monitor tick output.values[{name!r}]: expected a tensor, "
                    f"got {type(value).__name__}"
                )
            values[name] = value
        object.__setattr__(self, "values", MappingProxyType(values))


@dataclass(frozen=True, slots=True)
class BenchmarkTick:
    """One immutable tick handed to :meth:`BenchmarkFlow.on_tick`.

    Attributes:
        step: The pack step context for this tick.  Its borrowed view is valid
            only for the callback boundary.
        monitor_outputs: Per-monitor observed values.
    """

    step: BenchmarkStepContext
    monitor_outputs: tuple[_MonitorTickOutput, ...] = ()

    def __post_init__(self) -> None:
        """Validate the tick and freeze its monitor outputs."""
        if not isinstance(self.step, BenchmarkStepContext):
            raise TypeError(
                "benchmark tick.step: expected a BenchmarkStepContext, "
                f"got {type(self.step).__name__}"
            )
        outputs = tuple(self.monitor_outputs)
        names: set[str] = set()
        for index, output in enumerate(outputs):
            if not isinstance(output, _MonitorTickOutput):
                raise TypeError(
                    f"benchmark tick.monitor_outputs[{index}]: expected a "
                    f"_MonitorTickOutput, got {type(output).__name__}"
                )
            if output.monitor in names:
                raise ValueError(
                    f"benchmark tick.monitor_outputs: duplicate monitor name " f"{output.monitor!r}"
                )
            names.add(output.monitor)
        object.__setattr__(self, "monitor_outputs", outputs)

    @property
    def query(self) -> TickQuery | None:
        """The tick snapshot shared with the step context."""
        return self.step.query

    @property
    def monitor_values(self) -> Mapping[str, Mapping[str, Tensor]]:
        """Per-monitor value mappings keyed by monitor name."""
        return MappingProxyType({output.monitor: output.values for output in self.monitor_outputs})


@dataclass(frozen=True, slots=True)
class FlowDecision:
    """The validated policy result for one tick.

    Attributes:
        events: Occurrences the flow wants journaled for this tick.
        terminations: Per-env terminal requests.  The mapping holds at most one
            request per env; the engine validates and latches the first one.
        stop: Request the run to stop after the current tick.
    """

    events: tuple[Event, ...] = ()
    terminations: Mapping[int, Termination] = field(default_factory=lambda: MappingProxyType({}))
    stop: bool = False

    def __post_init__(self) -> None:
        """Validate and freeze the decision."""
        raw_events = self.events
        if isinstance(raw_events, (str, bytes, bytearray)) or not isinstance(raw_events, Sequence):
            raise TypeError(
                "flow decision.events: expected a sequence of Event, "
                f"got {type(raw_events).__name__}"
            )
        events = tuple(raw_events)
        for index, event in enumerate(events):
            if not isinstance(event, Event):
                raise TypeError(
                    f"flow decision.events[{index}]: expected an Event, "
                    f"got {type(event).__name__}"
                )
        object.__setattr__(self, "events", events)

        if not isinstance(self.terminations, Mapping):
            raise TypeError(
                "flow decision.terminations: expected a mapping, "
                f"got {type(self.terminations).__name__}"
            )
        terminations: dict[int, Termination] = {}
        for env_id, termination in self.terminations.items():
            _validate_terminal_request(env_id, termination, "flow decision.terminations")
            terminations[env_id] = termination
        object.__setattr__(self, "terminations", MappingProxyType(terminations))

        if not isinstance(self.stop, bool):
            raise TypeError(f"flow decision.stop: expected a bool, got {self.stop!r}")


def _validate_terminal_request(env_id: object, termination: object, where: str) -> None:
    """Validate one flow-authored terminal request."""
    if isinstance(env_id, bool) or not isinstance(env_id, int):
        raise TypeError(f"{where}: keys must be int env ids, got {env_id!r}")
    if env_id < 0:
        raise ValueError(f"{where}: env id must be >= 0, got {env_id}")
    if not isinstance(termination, Termination):
        raise TypeError(
            f"{where}[{env_id}]: expected a Termination, " f"got {type(termination).__name__}"
        )


@dataclass(frozen=True, slots=True)
class EpisodeBatch:
    """The due-episode facts handed to :meth:`BenchmarkFlow.on_episode_end`.

    The batch is the flow's only view of a boundary.  It carries the due ids,
    the final step context, final monitor values, the finalized due records and
    the engine-held report extensions.  A flow may inspect records and invoke
    extension lifecycle callbacks, but it may not schedule a runner terminal
    from this callback.

    Attributes:
        due: Sorted, unique due environment ids.
        final_step: The post-``on_step`` boundary context for the due episodes.
        monitor_values: Final per-monitor value tensors.
        records: The finalized due records, keyed by env id and containing
            exactly ``due``.
        extensions: The engine-held report extensions in declaration order.
            The flow owns invocation, not persistence.
    """

    due: tuple[int, ...]
    final_step: BenchmarkStepContext
    monitor_values: Mapping[str, Mapping[str, Tensor]] = field(default_factory=dict)
    records: Mapping[int, EpisodeRecord] = field(default_factory=dict)
    extensions: tuple[ReportExtension, ...] = ()

    def __post_init__(self) -> None:
        """Validate and freeze the episode batch."""
        due = tuple(self.due)
        for env_id in due:
            if isinstance(env_id, bool) or not isinstance(env_id, int) or env_id < 0:
                raise TypeError(
                    "episode batch.due: expected non-negative int env ids, " f"got {env_id!r}"
                )
        if len(set(due)) != len(due):
            raise ValueError(f"episode batch.due: duplicate env ids {due!r}")
        object.__setattr__(self, "due", due)
        if not isinstance(self.final_step, BenchmarkStepContext):
            raise TypeError(
                "episode batch.final_step: expected a BenchmarkStepContext, "
                f"got {type(self.final_step).__name__}"
            )
        object.__setattr__(self, "monitor_values", _freeze_monitor_values(self.monitor_values))
        if not isinstance(self.records, Mapping):
            raise TypeError(
                "episode batch.records: expected a mapping, " f"got {type(self.records).__name__}"
            )
        records: dict[int, EpisodeRecord] = {}
        for env_id, record in self.records.items():
            if isinstance(env_id, bool) or not isinstance(env_id, int) or env_id < 0:
                raise TypeError(
                    "episode batch.records: keys must be non-negative int env ids, "
                    f"got {env_id!r}"
                )
            if not isinstance(record, EpisodeRecord):
                raise TypeError(
                    f"episode batch.records[{env_id}]: expected an EpisodeRecord, "
                    f"got {type(record).__name__}"
                )
            records[env_id] = record
        object.__setattr__(self, "records", MappingProxyType(records))
        object.__setattr__(self, "extensions", tuple(self.extensions))

    @property
    def query(self) -> TickQuery | None:
        """The final-boundary snapshot shared with the final step context."""
        return self.final_step.query

    @property
    def final_context(self) -> BenchmarkStepContext:
        """Alias for :attr:`final_step` used by boundary-oriented callers."""
        return self.final_step


@dataclass(frozen=True, slots=True)
class EpisodeEvidence:
    """Episode-scoped evidence returned from :meth:`BenchmarkFlow.on_episode_end`.

    Evidence is keyed by due env id and merged into that episode's evidence by
    the engine.  It is never a per-tick payload and never schedules a runner
    terminal.  Every payload must be a JSON-safe object; a non-JSON value fails
    at construction with the exact evidence path.

    Attributes:
        evidence: Due env id -> JSON object.  An empty mapping contributes
            nothing.
    """

    evidence: Mapping[int, Mapping[str, JsonValue]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        """Validate and freeze the evidence payloads."""
        if not isinstance(self.evidence, Mapping):
            raise TypeError(
                "episode evidence: expected a mapping, " f"got {type(self.evidence).__name__}"
            )
        frozen: dict[int, Mapping[str, JsonValue]] = {}
        for env_id, payload in self.evidence.items():
            if isinstance(env_id, bool) or not isinstance(env_id, int) or env_id < 0:
                raise TypeError(
                    f"episode evidence: keys must be non-negative int env ids, got {env_id!r}"
                )
            if not isinstance(payload, Mapping):
                raise TypeError(
                    f"episode evidence[{env_id}]: expected a mapping, "
                    f"got {type(payload).__name__}"
                )
            normalized = ensure_json_value(dict(payload), where=f"episode evidence[{env_id}]")
            if not isinstance(normalized, dict):
                raise TypeError(f"episode evidence[{env_id}]: expected an object")
            frozen[env_id] = MappingProxyType(normalized)
        object.__setattr__(self, "evidence", MappingProxyType(frozen))


@dataclass(frozen=True, slots=True)
class FlowRun:
    """The run-end facts handed to :meth:`BenchmarkFlow.on_run_end`.

    The flow is the report authority for the run.  It receives monitor report
    views, not live monitor objects: ``monitor_names`` preserves declaration
    order and ``monitor_views`` carries the JSON report blocks.  ``evidence``
    carries finalized episode evidence keyed by ``(env_id, episode_seq)``.
    The remaining fields are the report metadata needed to reproduce the
    current document.
    """

    metadata: RunMetadata
    summary: EpisodeSummary
    records: tuple[EpisodeRecord, ...]
    monitor_views: Mapping[str, Mapping[str, JsonValue]]
    evidence: Mapping[tuple[int, int], Mapping[str, JsonValue]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    retention: RetentionSpec = field(default_factory=RetentionSpec)
    extensions: tuple[ReportExtension, ...] = ()
    monitor_names: tuple[str, ...] = ()
    monitor_extensions: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """Validate and freeze the run bundle."""
        if not isinstance(self.metadata, RunMetadata):
            raise TypeError(
                "flow run.metadata: expected a RunMetadata, " f"got {type(self.metadata).__name__}"
            )
        if not isinstance(self.summary, EpisodeSummary):
            raise TypeError(
                "flow run.summary: expected an EpisodeSummary, "
                f"got {type(self.summary).__name__}"
            )
        object.__setattr__(self, "records", tuple(self.records))
        raw_views = self.monitor_views
        if not isinstance(raw_views, Mapping):
            raise TypeError(
                "flow run.monitor_views: expected a mapping, " f"got {type(raw_views).__name__}"
            )
        names = tuple(self.monitor_names)
        if not names:
            names = tuple(raw_views)
        checked_names: list[str] = []
        for name in names:
            if not isinstance(name, str) or not name:
                raise TypeError(
                    "flow run.monitor_names: expected non-empty strings, " f"got {name!r}"
                )
            if name not in raw_views:
                raise ValueError(f"flow run.monitor_names: monitor {name!r} has no report view")
            checked_names.append(name)
        if len(set(checked_names)) != len(checked_names):
            raise ValueError(f"flow run.monitor_names: duplicate monitor names {checked_names!r}")
        object.__setattr__(self, "monitor_names", tuple(checked_names))
        object.__setattr__(self, "monitor_views", _freeze_monitor_views(raw_views))
        if not isinstance(self.evidence, Mapping):
            raise TypeError(
                f"flow run.evidence: expected a mapping, got {type(self.evidence).__name__}"
            )
        frozen_evidence: dict[tuple[int, int], Mapping[str, JsonValue]] = {}
        for key, payload in self.evidence.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or any(
                    isinstance(part, bool) or not isinstance(part, int) or part < 0 for part in key
                )
            ):
                raise TypeError(
                    "flow run.evidence: keys must be (env_id, episode_seq) non-negative int "
                    f"tuples, got {key!r}"
                )
            if not isinstance(payload, Mapping):
                raise TypeError(
                    f"flow run.evidence[{key[0]}/{key[1]}]: expected a mapping, "
                    f"got {type(payload).__name__}"
                )
            normalized = ensure_json_value(
                dict(payload), where=f"flow run.evidence[{key[0]}/{key[1]}]"
            )
            if not isinstance(normalized, dict):
                raise TypeError(f"flow run.evidence[{key[0]}/{key[1]}]: expected an object")
            frozen_evidence[(key[0], key[1])] = MappingProxyType(normalized)
        object.__setattr__(
            self,
            "evidence",
            MappingProxyType(dict(sorted(frozen_evidence.items()))),
        )
        if not isinstance(self.retention, RetentionSpec):
            raise TypeError(
                "flow run.retention: expected a RetentionSpec, "
                f"got {type(self.retention).__name__}"
            )
        object.__setattr__(self, "extensions", tuple(self.extensions))
        if not isinstance(self.monitor_extensions, Mapping):
            raise TypeError(
                "flow run.monitor_extensions: expected a mapping, "
                f"got {type(self.monitor_extensions).__name__}"
            )
        object.__setattr__(
            self, "monitor_extensions", MappingProxyType(dict(self.monitor_extensions))
        )

    @property
    def episode_summary(self) -> EpisodeSummary:
        """Alias for :attr:`summary` matching the report summary vocabulary."""
        return self.summary


@runtime_checkable
class BenchmarkFlow(Protocol):
    """Policy seam for a benchmark run.

    ``on_tick`` and ``on_reset`` are optional: a flow that only observes
    episode boundaries or run end may omit ``on_tick``, and a flow that keeps no
    reset-driven state may omit ``on_reset``.  Because those callbacks are not
    protocol members, a flow without them still satisfies
    ``isinstance(flow, BenchmarkFlow)``.  Validation is performed by
    :func:`validate_flow`, not by ``isinstance`` alone.
    """

    name: str
    stateful: bool

    def on_episode_end(self, batch: EpisodeBatch) -> EpisodeEvidence | None:
        """Observe the due episode boundary and optionally return evidence."""

    def on_run_end(self, run: FlowRun) -> Mapping[str, JsonValue] | None:
        """Return the pack report document (or delegate to the default)."""


@runtime_checkable
class RewindableFlow(Protocol):
    """Optional checkpoint face for a :attr:`BenchmarkFlow.stateful` flow.

    ``snapshot`` returns a JSON-safe, private copy of the flow's policy state;
    ``restore`` accepts a state previously returned by ``snapshot`` and must be
    atomic enough that the engine can roll it back after an unexpected failure.
    A flow may additionally implement ``validate_restore(state)``; the engine
    calls it during checkpoint decoding so malformed payloads are refused before
    any live state is touched.
    """

    def snapshot(self) -> JsonValue:
        """Return the flow's rewindable state as JSON-safe, private data."""

    def restore(self, state: JsonValue) -> None:
        """Restore state previously returned by :meth:`snapshot`."""


def validate_flow(value: object) -> BenchmarkFlow:
    """Validate a flow instance at the sink construction boundary.

    Args:
        value: The candidate flow.

    Returns:
        The validated flow.

    Raises:
        TypeError: A required attribute or callback is missing or malformed,
            or the ``stateful`` declaration disagrees with the snapshot/restore
            capability.  ``on_tick`` and ``on_reset`` are optional; when present
            they must be callable.
    """
    name = getattr(value, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError(f"benchmark flow.name: expected a non-empty string, got {name!r}")
    stateful = getattr(value, "stateful", None)
    if not isinstance(stateful, bool):
        raise TypeError(
            f"benchmark flow {name!r}.stateful: expected an explicit bool "
            f"declaration, got {stateful!r}"
        )
    has_snapshot = callable(getattr(value, "snapshot", None))
    has_restore = callable(getattr(value, "restore", None))
    if stateful:
        if not (has_snapshot and has_restore):
            raise TypeError(
                f"benchmark flow {name!r}: stateful=True requires callable "
                "snapshot() and restore()"
            )
    elif has_snapshot or has_restore:
        raise TypeError(
            f"benchmark flow {name!r}: declares stateful=False but implements "
            "snapshot/restore; declare stateful=True or remove the capability"
        )
    on_tick = getattr(value, "on_tick", None)
    if on_tick is not None and not callable(on_tick):
        raise TypeError(f"benchmark flow {name!r}.on_tick: expected a callable or absent")
    on_reset = getattr(value, "on_reset", None)
    if on_reset is not None and not callable(on_reset):
        raise TypeError(f"benchmark flow {name!r}.on_reset: expected a callable or absent")
    for callback in ("on_episode_end", "on_run_end"):
        if not callable(getattr(value, callback, None)):
            raise TypeError(f"benchmark flow {name!r}: missing callable {callback}()")
    return cast(BenchmarkFlow, value)


class _MonitorView:
    """Report-only monitor adapter backed by one JSON report view."""

    __slots__ = ("name", "_view")

    def __init__(self, name: str, view: Mapping[str, JsonValue]) -> None:
        self.name = name
        self._view = view

    def report_view(self) -> Mapping[str, JsonValue]:
        """Return the captured report view without touching a live monitor."""
        return self._view


@dataclass(frozen=True, slots=True)
class TerminationRule:
    """One explicit value-to-terminal decision rule.

    The named ``monitor/value`` must be a per-env boolean value produced by the
    monitor trigger.  Rules are evaluated in declaration order and the first
    true rule per env wins; later true rules are not conflicts, they are simply
    not selected.

    Args:
        monitor: Monitor name that publishes the value.
        value: Local value name inside that monitor.
        outcome: Episode outcome for the produced terminal.
        code: Full ``custom.*`` termination code.
    """

    monitor: str
    value: str
    outcome: EpisodeOutcome | str
    code: str

    def __post_init__(self) -> None:
        """Validate names, outcome and code at construction."""
        _require_rule_name(self.monitor, "TerminationRule.monitor")
        _require_rule_name(self.value, "TerminationRule.value")
        object.__setattr__(self, "outcome", _require_rule_outcome(self.outcome))
        if not isinstance(self.code, str) or not self.code.startswith("custom."):
            raise ValueError(f"TerminationRule.code: expected a 'custom.*' code, got {self.code!r}")
        validate_code(self.code)


@dataclass(frozen=True, slots=True)
class EventRule:
    """One explicit value-to-event decision rule.

    For every env whose boolean value is true, the flow journals one event with
    the grouped true env ids.  The default code is ``task.<monitor>.<name>``.

    Args:
        monitor: Monitor name that publishes the value.
        value: Local value name inside that monitor.
        name: Event name.
        severity: Event severity.
        code: Optional full event code; defaults to the pack event namespace.
        data: Optional JSON-safe static event detail payload.
    """

    monitor: str
    value: str
    name: str
    severity: Severity = "info"
    code: str | None = None
    data: Mapping[str, JsonValue] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """Validate names, severity, code and data at construction."""
        _require_rule_name(self.monitor, "EventRule.monitor")
        _require_rule_name(self.value, "EventRule.value")
        _require_rule_name(self.name, "EventRule.name")
        if not isinstance(self.severity, str) or self.severity not in SEVERITIES:
            raise ValueError(
                f"EventRule.severity: expected one of {sorted(SEVERITIES)}, "
                f"got {self.severity!r}"
            )
        if self.code is None:
            code = f"task.{self.monitor}.{self.name}"
        elif isinstance(self.code, str):
            code = self.code
        else:
            raise TypeError(f"EventRule.code: expected a string or None, got {self.code!r}")
        validate_code(code)
        object.__setattr__(self, "code", code)
        if not isinstance(self.data, Mapping):
            raise TypeError(f"EventRule.data: expected a mapping, got {type(self.data).__name__}")
        normalized = ensure_json_value(dict(self.data), where="EventRule.data")
        if not isinstance(normalized, dict):
            raise TypeError("EventRule.data: expected an object")
        object.__setattr__(self, "data", MappingProxyType(normalized))


class DefaultBenchmarkFlow:
    """Stateless flow that applies explicit rules and assembles the pack report.

    Args:
        rules: Explicit termination rules in declaration order.  Empty means no
            automatic termination.
        events: Explicit event rules in declaration order.  Empty means no
            automatic events.
        report_monitor_views: Monitor names whose report views are retained.
            ``None`` retains every declared view.  A selected name that is not
            present at run end is an error; an omitted name is simply absent
            from the report's ``monitors`` block.
        report_extensions: Extension names whose lifecycle callbacks run.
            ``None`` runs every declared extension in declaration order.  A
            selected name that is not present at run end is an error; an omitted
            extension produces no ``extensions`` block.

    Raises:
        ValueError: A duplicate ``(monitor, value)`` termination rule is
            declared, a selection contains a duplicate name, or a rule is
            malformed.
    """

    name = "default"
    stateful = False

    def __init__(
        self,
        rules: Sequence[TerminationRule] = (),
        *,
        events: Sequence[EventRule] = (),
        report_monitor_views: Iterable[str] | None = None,
        report_extensions: Iterable[str] | None = None,
    ) -> None:
        """Validate and freeze the explicit rule declarations."""
        checked_rules: list[TerminationRule] = []
        seen: set[tuple[str, str]] = set()
        for index, rule in enumerate(rules):
            if not isinstance(rule, TerminationRule):
                raise TypeError(
                    f"DefaultBenchmarkFlow.rules[{index}]: expected a TerminationRule, "
                    f"got {type(rule).__name__}"
                )
            key = (rule.monitor, rule.value)
            if key in seen:
                raise ValueError(
                    "DefaultBenchmarkFlow.rules: duplicate "
                    f"(monitor={rule.monitor!r}, value={rule.value!r}) rule; the later "
                    "rule is dead policy"
                )
            seen.add(key)
            checked_rules.append(rule)
        checked_events: list[EventRule] = []
        for index, rule in enumerate(events):
            if not isinstance(rule, EventRule):
                raise TypeError(
                    f"DefaultBenchmarkFlow.events[{index}]: expected an EventRule, "
                    f"got {type(rule).__name__}"
                )
            checked_events.append(rule)
        self._rules: tuple[TerminationRule, ...] = tuple(checked_rules)
        self._events: tuple[EventRule, ...] = tuple(checked_events)
        self._report_monitor_views = _validate_report_selection(
            report_monitor_views, "DefaultBenchmarkFlow.report_monitor_views"
        )
        self._report_extensions = _validate_report_selection(
            report_extensions, "DefaultBenchmarkFlow.report_extensions"
        )

    @property
    def rules(self) -> tuple[TerminationRule, ...]:
        """The explicit termination rules in declaration order."""
        return self._rules

    @property
    def events(self) -> tuple[EventRule, ...]:
        """The explicit event rules in declaration order."""
        return self._events

    @property
    def report_monitor_views(self) -> frozenset[str] | None:
        """Selected monitor view names, or ``None`` for every declared view."""
        return None if self._report_monitor_views is None else frozenset(self._report_monitor_views)

    @property
    def report_extensions(self) -> frozenset[str] | None:
        """Selected extension names, or ``None`` for every declared extension."""
        return None if self._report_extensions is None else frozenset(self._report_extensions)

    def on_tick(self, tick: BenchmarkTick) -> FlowDecision:
        """Apply the explicit event and termination rules to this tick's values."""
        terminations: dict[int, Termination] = {}
        if self._rules:
            num_envs = tick.step.num_envs
            for rule in self._rules:
                mask = _rule_mask(tick, rule.monitor, rule.value, num_envs)
                for env_id, active in enumerate(mask):
                    if not active or env_id in terminations:
                        continue
                    terminations[env_id] = Termination(
                        outcome=rule.outcome,
                        code=rule.code,
                        source=TerminationSource.CUSTOM,
                        metrics={},
                    )
        events: list[Event] = []
        if self._events:
            num_envs = tick.step.num_envs
            for event_rule in self._events:
                mask = _rule_mask(tick, event_rule.monitor, event_rule.value, num_envs)
                true_ids = tuple(env_id for env_id, active in enumerate(mask) if active)
                if not true_ids:
                    continue
                assert event_rule.code is not None
                events.append(
                    Event(
                        name=event_rule.name,
                        code=event_rule.code,
                        env_ids=true_ids,
                        severity=event_rule.severity,
                        data=event_rule.data,
                    )
                )
        return FlowDecision(events=tuple(events), terminations=terminations)

    def on_episode_end(self, batch: EpisodeBatch) -> EpisodeEvidence | None:
        """Invoke each selected extension boundary callback exactly once.

        Extensions run in declaration order.  The engine owns their persistence
        and rewind state; the flow owns only this invocation.  The default flow
        contributes no evidence of its own.
        """
        for extension in _select_extensions(batch.extensions, self._report_extensions):
            extension.on_env_end(batch.due, batch.records)
        return None

    def on_reset(self, env_ids: Sequence[int]) -> None:
        """Observe a reset without changing default stateless behavior."""
        del env_ids

    def on_run_end(self, run: FlowRun) -> Mapping[str, JsonValue]:
        """Assemble the current pack report from the selected views/extensions."""
        return assemble_default_report(
            run,
            report_monitor_views=self._report_monitor_views,
            report_extensions=self._report_extensions,
        )


class FlowSequence:
    """Composite flow with one terminal owner and one report owner.

    The sequence runs its children in declaration order on one immutable
    :class:`BenchmarkTick`.  Each child keeps its own policy ownership:

    - Every child may journal events; events concatenate in flow order.
    - Only ``terminal_owner`` may return non-empty
      :attr:`FlowDecision.terminations` or request ``stop``.  Those decisions
      from another child are contract errors rather than silently dropped
      policy, because dropping them would hide a mis-scoped benchmark rule.
      Every other child is limited to events and episode evidence.
    - Only ``report_owner`` may return the run report.  A non-owner report is a
      contract error for the same reason.
    - Every child observes :meth:`BenchmarkFlow.on_episode_end`; returned
      evidence is namespaced by the child's flow name.  Duplicate flow names are
      impossible by construction, so two children may use the same local
      evidence key without collision: the report keeps both under their
      respective namespaces.

    ``stateful`` is true when any child is stateful.  The checkpoint is a map
    keyed by child flow name that contains only the stateful children.
    ``restore`` pre-flights every child (including an optional
    ``validate_restore``) before touching live state and rolls every child back
    when an unexpected restore failure occurs.  Rollback is best-effort: it
    assumes each child's restore also succeeds for valid state.  If a broken
    child's restore fails during rollback too, that error propagates and full
    atomicity is not guaranteed; the engine then aborts the rewind.

    Args:
        flows: Child flows in execution order.  Names must be unique, non-empty
            strings.
        terminal_owner: Name of the only child allowed to terminate.
        report_owner: Name of the only child allowed to return the run report.
        name: The sequence's own flow name.  It must be non-empty and unique
            when this sequence is itself a child of another sequence.

    Raises:
        TypeError: ``flows`` is not an iterable of valid flows, a name or owner
            is not a non-empty string, or a child misdeclares its checkpoint
            face.
        ValueError: ``flows`` is empty, two children share a name, or an owner
            does not name a declared child.
    """

    name: str
    stateful: bool

    def __init__(
        self,
        flows: Iterable[BenchmarkFlow],
        *,
        terminal_owner: str,
        report_owner: str,
        name: str = "sequence",
    ) -> None:
        """Validate the child flows, their names and both owners."""
        if not isinstance(name, str) or not name:
            raise TypeError(f"FlowSequence.name: expected a non-empty string, got {name!r}")
        if isinstance(flows, (str, bytes, bytearray)):
            raise TypeError(
                "FlowSequence.flows: expected an iterable of BenchmarkFlow, "
                f"got {type(flows).__name__}"
            )
        try:
            materialized = tuple(flows)
        except TypeError as exc:
            raise TypeError(
                "FlowSequence.flows: expected an iterable of BenchmarkFlow, "
                f"got {type(flows).__name__}"
            ) from exc
        if not materialized:
            raise ValueError("FlowSequence.flows: expected at least one child flow")
        checked: list[BenchmarkFlow] = []
        for index, flow in enumerate(materialized):
            try:
                checked.append(validate_flow(flow))
            except TypeError as exc:
                raise TypeError(f"FlowSequence.flows[{index}]: {exc}") from exc
        declared = tuple(flow.name for flow in checked)
        duplicates = sorted({child for child in declared if declared.count(child) > 1})
        if duplicates:
            raise ValueError(
                "FlowSequence.flows: duplicate child flow names "
                f"{duplicates}; every child must have a unique name"
            )
        self.name = name
        self._flows: tuple[BenchmarkFlow, ...] = tuple(checked)
        self._terminal_owner = _require_sequence_owner(
            terminal_owner, "FlowSequence.terminal_owner", declared
        )
        self._report_owner = _require_sequence_owner(
            report_owner, "FlowSequence.report_owner", declared
        )
        self._stateful_flows: tuple[BenchmarkFlow, ...] = tuple(
            flow for flow in self._flows if flow.stateful
        )
        self.stateful = bool(self._stateful_flows)
        if self.stateful:
            # validate_flow rejects a flow that declares stateful=False while
            # implementing the checkpoint face, so a stateless sequence must not
            # advertise it.  Bind the face only when there is state to save.
            self.snapshot = self._snapshot_state
            self.restore = self._restore_state
            self.validate_restore = self._validate_restore_state

    @property
    def flows(self) -> tuple[BenchmarkFlow, ...]:
        """Child flows in declaration order."""
        return self._flows

    @property
    def terminal_owner(self) -> str:
        """Name of the only child allowed to terminate."""
        return self._terminal_owner

    @property
    def report_owner(self) -> str:
        """Name of the only child allowed to return the run report."""
        return self._report_owner

    def on_tick(self, tick: BenchmarkTick) -> FlowDecision:
        """Run every child on the same immutable tick in declaration order."""
        events: list[Event] = []
        terminations: dict[int, Termination] = {}
        stop = False
        for flow in self._flows:
            callback = getattr(flow, "on_tick", None)
            if callback is None:
                continue
            decision = callback(tick)
            if not isinstance(decision, FlowDecision):
                raise TypeError(
                    f"FlowSequence child {flow.name!r}.on_tick: expected a "
                    f"FlowDecision, got {type(decision).__name__}"
                )
            if decision.terminations and flow.name != self._terminal_owner:
                raise ValueError(
                    f"FlowSequence {self.name!r}: child {flow.name!r} returned "
                    f"terminations for env ids {sorted(decision.terminations)} but "
                    f"only the terminal owner {self._terminal_owner!r} may terminate; "
                    "move the decision into the terminal owner"
                )
            if decision.stop and flow.name != self._terminal_owner:
                raise ValueError(
                    f"FlowSequence {self.name!r}: child {flow.name!r} requested a run "
                    f"stop but only the terminal owner {self._terminal_owner!r} may "
                    "stop the run; non-owner children contribute events and evidence only"
                )
            terminations.update(decision.terminations)
            events.extend(decision.events)
            stop = stop or decision.stop
        return FlowDecision(events=tuple(events), terminations=terminations, stop=stop)

    def on_episode_end(self, batch: EpisodeBatch) -> EpisodeEvidence | None:
        """Observe the boundary in every child and namespace returned evidence."""
        rows: dict[int, dict[str, JsonValue]] = {}
        for flow in self._flows:
            result = flow.on_episode_end(batch)
            if result is None:
                continue
            if not isinstance(result, EpisodeEvidence):
                raise TypeError(
                    f"FlowSequence child {flow.name!r}.on_episode_end: expected "
                    f"EpisodeEvidence or None, got {type(result).__name__}"
                )
            for env_id, payload in result.evidence.items():
                if not payload:
                    continue
                rows.setdefault(env_id, {})[flow.name] = dict(payload)
        if not rows:
            return None
        return EpisodeEvidence(rows)

    def on_reset(self, env_ids: Sequence[int]) -> None:
        """Forward the reset to every child with an ``on_reset`` callback.

        Children run in declaration order and are all notified; reset
        observation is not restricted to the terminal or report owner.  A
        child exception propagates immediately and later children are not
        called.
        """
        for flow in self._flows:
            callback = getattr(flow, "on_reset", None)
            if callable(callback):
                callback(env_ids)

    def on_run_end(self, run: FlowRun) -> Mapping[str, JsonValue] | None:
        """Collect the run report from the sole report owner."""
        report: Mapping[str, JsonValue] | None = None
        for flow in self._flows:
            result = flow.on_run_end(run)
            if result is None:
                continue
            if flow.name != self._report_owner:
                raise ValueError(
                    f"FlowSequence {self.name!r}: child {flow.name!r} returned a "
                    f"run report but only the report owner {self._report_owner!r} "
                    "may produce it; delegate the report to the report owner"
                )
            normalized = ensure_json_value(result, where=f"FlowSequence child {flow.name!r} report")
            if not isinstance(normalized, dict):
                raise TypeError(
                    f"FlowSequence child {flow.name!r}.on_run_end: expected a "
                    f"mapping report, got {type(normalized).__name__}"
                )
            report = normalized
        return report

    def _snapshot_state(self) -> dict[str, JsonValue]:
        """Return one JSON-safe snapshot per stateful child, keyed by flow name."""
        snapshots: dict[str, JsonValue] = {}
        for flow in self._stateful_flows:
            snapshots[flow.name] = ensure_json_value(
                flow.snapshot(), where=f"FlowSequence child {flow.name!r} snapshot"
            )
        return snapshots

    def _validate_restore_state(self, state: JsonValue) -> None:
        """Pre-flight a checkpoint without mutating any child."""
        self._decode_state(state)

    def _restore_state(self, state: JsonValue) -> None:
        """Restore every stateful child atomically, rolling all back on failure."""
        payloads = self._decode_state(state)
        rollback = self._snapshot_state()
        try:
            for flow in self._stateful_flows:
                flow.restore(payloads[flow.name])
        except Exception:
            for flow in self._stateful_flows:
                flow.restore(rollback[flow.name])
            raise

    def _decode_state(self, state: JsonValue) -> dict[str, JsonValue]:
        """Validate and normalize one checkpoint for every stateful child."""
        if not isinstance(state, Mapping):
            raise TypeError(
                f"FlowSequence {self.name!r} state: expected an object, "
                f"got {type(state).__name__}"
            )
        declared = tuple(flow.name for flow in self._stateful_flows)
        if set(state) != set(declared):
            raise ValueError(
                f"FlowSequence {self.name!r} state: child flow names "
                f"{sorted(map(repr, state))} do not match the live stateful "
                f"children {sorted(declared)}"
            )
        payloads: dict[str, JsonValue] = {}
        for flow in self._stateful_flows:
            payload = ensure_json_value(
                state[flow.name],
                where=f"FlowSequence {self.name!r} child {flow.name!r} state",
            )
            validate = getattr(flow, "validate_restore", None)
            if callable(validate):
                validate(payload)
            payloads[flow.name] = payload
        return payloads


def _require_sequence_owner(value: object, where: str, declared: tuple[str, ...]) -> str:
    """Validate one sequence owner name against the declared child flows."""
    if not isinstance(value, str) or not value:
        raise TypeError(f"{where}: expected a non-empty child flow name, got {value!r}")
    if value not in declared:
        raise ValueError(
            f"{where}: {value!r} is not a declared child flow; declared flows are "
            f"{list(declared)}"
        )
    return value


def _require_rule_name(value: object, where: str) -> str:
    """Validate one rule name."""
    if not isinstance(value, str) or not _RULE_NAME_PATTERN.fullmatch(value):
        raise ValueError(f"{where}: expected a name matching [a-z][a-z0-9_]*, got {value!r}")
    return value


def _require_rule_outcome(value: object) -> EpisodeOutcome:
    """Validate the outcome a flow rule may produce."""
    if isinstance(value, EpisodeOutcome):
        outcome = value
    elif isinstance(value, str):
        try:
            outcome = EpisodeOutcome(value)
        except ValueError as exc:
            raise ValueError(
                f"TerminationRule.outcome: unknown episode outcome {value!r}; "
                f"allowed: {sorted(item.value for item in _MONITOR_OUTCOMES)}"
            ) from exc
    else:
        raise TypeError(
            f"TerminationRule.outcome: expected an EpisodeOutcome or str, got {value!r}"
        )
    if outcome not in _MONITOR_OUTCOMES:
        raise ValueError(
            f"TerminationRule.outcome: {outcome.value!r} is not a flow-owned terminal "
            f"outcome; allowed: {sorted(item.value for item in _MONITOR_OUTCOMES)}"
        )
    return outcome


def _rule_mask(tick: BenchmarkTick, monitor: str, value: str, num_envs: int) -> list[bool]:
    """Return one boolean per env for a rule's monitor value."""
    monitor_values = tick.monitor_values
    if monitor not in monitor_values:
        raise ValueError(
            f"flow rule references monitor {monitor!r}, which is not present in this "
            f"tick; declared monitors are {sorted(monitor_values)}"
        )
    values = monitor_values[monitor]
    if value not in values:
        raise ValueError(
            f"flow rule references value {value!r} of monitor {monitor!r}, which is "
            f"not present in this tick; available values are {sorted(values)}"
        )
    tensor = values[value]
    if tensor.dtype is not torch.bool:
        raise TypeError(
            f"flow rule value {monitor}/{value}: expected a torch.bool tensor, "
            f"got dtype {tensor.dtype}"
        )
    if tensor.ndim != 1 or tensor.shape[0] != num_envs:
        raise ValueError(
            f"flow rule value {monitor}/{value}: expected shape (num_envs={num_envs},), "
            f"got {tuple(tensor.shape)}"
        )
    return [bool(item) for item in tensor.detach().to(device="cpu").tolist()]


def _validate_report_selection(
    value: Iterable[str] | None,
    where: str,
) -> tuple[str, ...] | None:
    """Validate an optional report selection.

    A selection may be any finite iterable (a list, tuple, set or frozenset).
    Its order is not significant: report assembly always follows declaration
    order.
    """
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{where}: expected an iterable of names or None, got {value!r}")
    try:
        items = list(value)
    except TypeError as exc:
        raise TypeError(
            f"{where}: expected an iterable of names or None, got {type(value).__name__}"
        ) from exc
    names: list[str] = []
    for name in items:
        if not isinstance(name, str) or not name:
            raise TypeError(f"{where}: expected non-empty strings, got {name!r}")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError(f"{where}: duplicate names {names!r}")
    return tuple(names)


def _select_extensions(
    extensions: Sequence[ReportExtension],
    selected: Iterable[str] | None,
) -> tuple[ReportExtension, ...]:
    """Return selected extensions in declaration order."""
    if selected is None:
        return tuple(extensions)
    wanted = set(selected)
    by_name = {extension.name: extension for extension in extensions}
    unknown = sorted(wanted - set(by_name))
    if unknown:
        raise ValueError(
            f"report_extensions: unknown extension name(s) {unknown}; "
            f"declared extensions are {sorted(by_name)}"
        )
    return tuple(extension for extension in extensions if extension.name in wanted)


def _select_monitor_names(
    names: Sequence[str],
    selected: Iterable[str] | None,
) -> tuple[str, ...]:
    """Return selected monitor names in declaration order."""
    if selected is None:
        return tuple(names)
    wanted = set(selected)
    unknown = sorted(wanted - set(names))
    if unknown:
        raise ValueError(
            f"report_monitor_views: unknown monitor name(s) {unknown}; "
            f"declared monitors are {sorted(names)}"
        )
    return tuple(name for name in names if name in wanted)


def assemble_default_report(
    run: FlowRun,
    *,
    report_monitor_views: Iterable[str] | None = None,
    report_extensions: Iterable[str] | None = None,
) -> Mapping[str, JsonValue]:
    """Build the current pack report for a run bundle.

    This is the default flow's report behavior and the sink's fallback when a
    custom flow returns ``None`` from ``on_run_end``.  The builder sees
    report-only monitor adapters; no live monitor instance is exposed.

    Args:
        run: The frozen run bundle.
        report_monitor_views: Selected monitor view names, or ``None`` for all.
        report_extensions: Selected extension names, or ``None`` for all.

    Returns:
        The validated JSON-safe pack report document.

    Raises:
        ValueError: A selection names an unknown monitor or extension, or a
            retained monitor view declares a ``report_extension`` that the same
            selection omits (which would silently drop the monitor's report
            content).
    """
    selected_monitors = _select_monitor_names(run.monitor_names, report_monitor_views)
    selected_extensions = _select_extensions(run.extensions, report_extensions)
    selected_extension_names = {extension.name for extension in selected_extensions}
    orphaned = sorted(
        f"{monitor} -> {extension}"
        for monitor in selected_monitors
        if (extension := run.monitor_extensions.get(monitor)) is not None
        and extension not in selected_extension_names
    )
    if orphaned:
        raise ValueError(
            "report selection would drop monitor report content: "
            f"{orphaned}; each retained monitor that declares report_extension "
            "requires its extension to be selected too. Select the extension, or "
            "also omit the monitor view from report_monitor_views."
        )
    summary = RunSummary(
        metadata=run.metadata,
        episodes=run.records,
        episode_summary=run.summary,
        monitor_views=run.monitor_views,
        evidence=run.evidence,
        retention=run.retention,
    )
    blocks = {extension.name: extension.on_run_end(summary) for extension in selected_extensions}
    monitors = tuple(_MonitorView(name, run.monitor_views[name]) for name in selected_monitors)
    builder = ReportBuilder(
        monitors=monitors,
        extensions=selected_extensions,
        monitor_extensions=run.monitor_extensions,
        task_monitors=run.monitor_names,
        task_extensions=tuple(extension.name for extension in run.extensions),
    )
    return builder.assemble(summary, blocks)


def _freeze_monitor_values(
    values: object,
) -> Mapping[str, Mapping[str, Tensor]]:
    """Validate and freeze one monitor-value mapping."""
    if not isinstance(values, Mapping):
        raise TypeError("monitor values: expected a mapping, " f"got {type(values).__name__}")
    frozen: dict[str, Mapping[str, Tensor]] = {}
    for name, mapping in values.items():
        if not isinstance(name, str) or not name:
            raise TypeError(f"monitor values: key must be a non-empty string, got {name!r}")
        if not isinstance(mapping, Mapping):
            raise TypeError(
                f"monitor values[{name!r}]: expected a mapping, " f"got {type(mapping).__name__}"
            )
        row: dict[str, Tensor] = {}
        for local_name, value in mapping.items():
            if not isinstance(local_name, str) or not local_name:
                raise TypeError(
                    f"monitor values[{name!r}]: names must be non-empty strings, "
                    f"got {local_name!r}"
                )
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"monitor values[{name!r}][{local_name!r}]: expected a tensor, "
                    f"got {type(value).__name__}"
                )
            row[local_name] = value
        frozen[name] = MappingProxyType(row)
    return MappingProxyType(frozen)


def _freeze_monitor_views(
    views: object,
) -> Mapping[str, Mapping[str, JsonValue]]:
    """Validate and freeze the report-orientation monitor views."""
    if not isinstance(views, Mapping):
        raise TypeError(
            "flow run.monitor_views: expected a mapping, " f"got {type(views).__name__}"
        )
    frozen: dict[str, Mapping[str, JsonValue]] = {}
    for name, view in views.items():
        if not isinstance(name, str) or not name:
            raise TypeError(f"flow run.monitor_views key must be a non-empty string, got {name!r}")
        if not isinstance(view, Mapping):
            raise TypeError(
                f"flow run.monitor_views[{name!r}]: expected a mapping, "
                f"got {type(view).__name__}"
            )
        normalized = ensure_json_value(dict(view), where=f"flow run.monitor_views[{name!r}]")
        if not isinstance(normalized, dict):
            raise TypeError(f"flow run.monitor_views[{name!r}]: expected an object")
        frozen[name] = MappingProxyType(normalized)
    return MappingProxyType(frozen)
