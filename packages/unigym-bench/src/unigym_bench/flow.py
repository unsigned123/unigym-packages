"""The stable benchmark-flow seam and its F2 default implementation.

A :class:`BenchmarkFlow` owns benchmark policy: it reads one immutable
:class:`BenchmarkTick`, decides which events and terminals apply, observes
episode boundaries and returns the pack report at run end.  The flow never
mutates the engine, a monitor or an environment; the sink and engine validate
every decision before changing run state.

F2 is a temporary bridge.  Monitors still own their event/terminal queues, so
:class:`DefaultBenchmarkFlow` reads the monitor outputs carried by the tick and
forwards them unchanged.  F3 removes that monitor output bridge and makes the
flow the only event/terminal decision owner.  This module deliberately imports
only pack-side context/event/report modules and the core termination type; a
custom flow must be able to depend on these types without importing the engine,
the sink or composition code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, cast, runtime_checkable

from torch import Tensor
from unigym.core.episode_termination import Termination

from .context import BenchmarkStepContext
from .events import Event, EpisodeRecord, EpisodeSummary
from .json import JsonValue, ensure_json_value
from .query import TickQuery
from .report import ReportBuilder, ReportExtension, RunMetadata, RunSummary
from .specs import RetentionSpec

__all__ = [
    "BenchmarkFlow",
    "BenchmarkTick",
    "DefaultBenchmarkFlow",
    "EpisodeBatch",
    "FlowDecision",
    "FlowRun",
    "validate_flow",
]


@dataclass(frozen=True, slots=True)
class _MonitorTickOutput:
    """One monitor's immutable contribution to a benchmark tick.

    F2 carries the current monitor event/terminal queues here so the default
    flow can forward them unchanged.  F3 deletes the ``events``/``terminations``
    fields and leaves only observed values.

    Attributes:
        monitor: The monitor name.
        values: Fresh per-env value tensors keyed by local value name.
        events: Occurrences drained from the monitor for this tick.
        terminations: Per-env terminal requests drained from the monitor.
    """

    monitor: str
    values: Mapping[str, Tensor] = field(default_factory=dict)
    events: tuple[Event, ...] = ()
    terminations: Mapping[int, Termination] = field(default_factory=dict)

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

        raw_events = self.events
        if isinstance(raw_events, (str, bytes, bytearray)) or not isinstance(raw_events, Sequence):
            raise TypeError(
                "monitor tick output.events: expected a sequence of Event, "
                f"got {type(raw_events).__name__}"
            )
        events = tuple(raw_events)
        for index, event in enumerate(events):
            if not isinstance(event, Event):
                raise TypeError(
                    f"monitor tick output.events[{index}]: expected an Event, "
                    f"got {type(event).__name__}"
                )
        object.__setattr__(self, "events", events)

        if not isinstance(self.terminations, Mapping):
            raise TypeError(
                "monitor tick output.terminations: expected a mapping, "
                f"got {type(self.terminations).__name__}"
            )
        terminations: dict[int, Termination] = {}
        for env_id, termination in self.terminations.items():
            if isinstance(env_id, bool) or not isinstance(env_id, int):
                raise TypeError(
                    "monitor tick output.terminations: keys must be int env ids, " f"got {env_id!r}"
                )
            if env_id < 0:
                raise ValueError(
                    f"monitor tick output.terminations: env id must be >= 0, got {env_id}"
                )
            if not isinstance(termination, Termination):
                raise TypeError(
                    f"monitor tick output.terminations[{env_id}]: expected a "
                    f"Termination, got {type(termination).__name__}"
                )
            terminations[env_id] = termination
        object.__setattr__(self, "terminations", MappingProxyType(terminations))


@dataclass(frozen=True, slots=True)
class BenchmarkTick:
    """One immutable tick handed to :meth:`BenchmarkFlow.on_tick`.

    Attributes:
        step: The pack step context for this tick.  Its borrowed view is valid
            only for the callback boundary.
        monitor_outputs: Per-monitor values and (F2 only) event/terminal queues.
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
    def monitor_events(self) -> tuple[Event, ...]:
        """Every monitor event in declaration order (F2 bridge)."""
        return tuple(event for output in self.monitor_outputs for event in output.events)

    @property
    def monitor_values(self) -> Mapping[str, Mapping[str, Tensor]]:
        """Per-monitor value mappings keyed by monitor name."""
        return MappingProxyType({output.monitor: output.values for output in self.monitor_outputs})


@dataclass(frozen=True, slots=True)
class FlowDecision:
    """The validated policy result for one tick.

    Attributes:
        events: Occurrences the flow wants journaled for this tick.
        terminations: Per-env terminal requests.  The public mapping view holds
            one request per env; F2's monitor bridge also preserves the ordered
            request list so terminal conflicts remain identical to the current
            engine behavior.
        stop: Request the run to stop after the current tick.
    """

    events: tuple[Event, ...] = ()
    terminations: Mapping[int, Termination] = field(default_factory=lambda: MappingProxyType({}))
    stop: bool = False
    _terminal_requests: tuple[tuple[int, Termination], ...] = field(
        default=(), repr=False, compare=False
    )

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

        requests = tuple(self._terminal_requests)
        if requests:
            checked_requests: list[tuple[int, Termination]] = []
            for index, request in enumerate(requests):
                if not isinstance(request, tuple) or len(request) != 2:
                    raise TypeError(
                        "flow decision terminal requests: expected (env_id, "
                        f"Termination) pairs, got {request!r} at index {index}"
                    )
                env_id, termination = request
                _validate_terminal_request(env_id, termination, "flow decision terminal requests")
                checked_requests.append((env_id, termination))
            object.__setattr__(self, "_terminal_requests", tuple(checked_requests))
        elif terminations:
            object.__setattr__(self, "_terminal_requests", tuple(terminations.items()))

        if not isinstance(self.stop, bool):
            raise TypeError(f"flow decision.stop: expected a bool, got {self.stop!r}")

    @classmethod
    def from_requests(
        cls,
        *,
        events: Sequence[Event] = (),
        requests: Sequence[tuple[int, Termination]] = (),
        stop: bool = False,
    ) -> FlowDecision:
        """Build a decision from an ordered terminal request sequence.

        A plain mapping cannot represent two monitors requesting a terminal for
        the same env; the ordered form is what preserves conflict accounting in
        the F2 monitor bridge.

        Args:
            events: Occurrences to journal.
            requests: Ordered ``(env_id, Termination)`` pairs.
            stop: Whether to request a run stop after this tick.

        Returns:
            A validated decision with the ordered request list retained.
        """
        pairs = tuple(requests)
        terminations = MappingProxyType(dict(pairs))
        return cls(
            events=tuple(events),
            terminations=terminations,
            stop=stop,
            _terminal_requests=pairs,
        )

    @property
    def terminal_requests(self) -> tuple[tuple[int, Termination], ...]:
        """The ordered terminal requests applied by the engine."""
        return self._terminal_requests


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

    F2 is intentionally minimal: it carries the due ids, the final step
    context, final monitor values and an empty record mapping.  F4 adds
    evidence and the finalized records; no flow may schedule a runner terminal
    from this callback.

    Attributes:
        due: Sorted, unique due environment ids.
        final_step: The post-``on_step`` boundary context for the due episodes.
        monitor_values: Final per-monitor value tensors.
        records: The finalized due records.  Empty until F4 owns record
            projection for the flow.
    """

    due: tuple[int, ...]
    final_step: BenchmarkStepContext
    monitor_values: Mapping[str, Mapping[str, Tensor]] = field(default_factory=dict)
    records: Mapping[int, EpisodeRecord] = field(default_factory=dict)

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

    @property
    def query(self) -> TickQuery | None:
        """The final-boundary snapshot shared with the final step context."""
        return self.final_step.query

    @property
    def final_context(self) -> BenchmarkStepContext:
        """Alias for :attr:`final_step` used by boundary-oriented callers."""
        return self.final_step


@dataclass(frozen=True, slots=True)
class FlowRun:
    """The run-end facts handed to :meth:`BenchmarkFlow.on_run_end`.

    The flow is the report authority for the run.  It receives monitor report
    views, not live monitor objects: ``monitor_names`` preserves declaration
    order and ``monitor_views`` carries the JSON report blocks.  The remaining
    fields are the report metadata needed to reproduce the current document.
    """

    metadata: RunMetadata
    summary: EpisodeSummary
    records: tuple[EpisodeRecord, ...]
    monitor_views: Mapping[str, Mapping[str, JsonValue]]
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

    ``on_tick`` is optional: a flow that only observes episode boundaries or
    run end may omit it entirely.  Validation is performed by
    :func:`validate_flow`, not by ``isinstance`` alone.
    """

    name: str
    stateful: bool

    def on_episode_end(self, batch: EpisodeBatch) -> None:
        """Observe the due episode boundary; F2 cannot add evidence."""

    def on_run_end(self, run: FlowRun) -> Mapping[str, JsonValue] | None:
        """Return the pack report document (or delegate to the default)."""


def validate_flow(value: object) -> BenchmarkFlow:
    """Validate a flow instance at the sink construction boundary.

    Args:
        value: The candidate flow.

    Returns:
        The validated flow.

    Raises:
        TypeError: A required attribute or callback is missing or malformed.
        ValueError: ``stateful`` is true; flow rewind is not part of F2.
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
        raise ValueError(
            f"benchmark flow {name!r}: stateful flows are not supported in F2; "
            "flow rewind lands with the later rewind card"
        )
    if has_snapshot or has_restore:
        raise TypeError(
            f"benchmark flow {name!r}: declares stateful=False but implements "
            "snapshot/restore; declare stateful=True or remove the capability"
        )
    on_tick = getattr(value, "on_tick", None)
    if on_tick is not None and not callable(on_tick):
        raise TypeError(f"benchmark flow {name!r}.on_tick: expected a callable or absent")
    for callback in ("on_episode_end", "on_run_end"):
        if not callable(getattr(value, callback, None)):
            raise TypeError(f"benchmark flow {name!r}: missing callable {callback}()")
    return cast(BenchmarkFlow, value)


def _monitor_bridge_decision(tick: BenchmarkTick) -> FlowDecision:
    """Forward the F2 monitor event/terminal queues as one flow decision.

    This is the single temporary bridge used by both ``engine.on_step`` (for
    the legacy engine-only path) and :class:`DefaultBenchmarkFlow`.  F3 deletes
    it with the monitor drains.

    Args:
        tick: The tick whose monitor outputs should be forwarded.

    Returns:
        A decision containing every monitor occurrence and terminal request in
        monitor declaration order.
    """
    events: list[Event] = []
    requests: list[tuple[int, Termination]] = []
    for output in tick.monitor_outputs:
        events.extend(output.events)
        for env_id in sorted(output.terminations):
            requests.append((env_id, output.terminations[env_id]))
    return FlowDecision.from_requests(events=tuple(events), requests=tuple(requests))


class _MonitorView:
    """Report-only monitor adapter backed by one JSON report view."""

    __slots__ = ("name", "_view")

    def __init__(self, name: str, view: Mapping[str, JsonValue]) -> None:
        self.name = name
        self._view = view

    def report_view(self) -> Mapping[str, JsonValue]:
        """Return the captured report view without touching a live monitor."""
        return self._view


class DefaultBenchmarkFlow:
    """Stateless flow that reproduces the current monitor and report behavior.

    The flow forwards the temporary monitor event/terminal bridge unchanged and
    assembles the pack report from the run's monitor report views and report
    extensions.  It never receives a live monitor object.  F3 removes the
    monitor bridge; F4 makes report selection flow-owned.
    """

    name = "default"
    stateful = False

    def on_tick(self, tick: BenchmarkTick) -> FlowDecision:
        """Forward the current monitor outputs for this tick."""
        return _monitor_bridge_decision(tick)

    def on_episode_end(self, batch: EpisodeBatch) -> None:
        """Observe the boundary; F2 has no episode evidence."""
        del batch

    def on_run_end(self, run: FlowRun) -> Mapping[str, JsonValue]:
        """Assemble the current pack report from the run bundle."""
        return assemble_default_report(run)


def assemble_default_report(run: FlowRun) -> Mapping[str, JsonValue]:
    """Build the current pack report for a run bundle.

    This is the default flow's report behavior and the sink's fallback when a
    custom flow returns ``None`` from ``on_run_end``.  The builder sees
    report-only monitor adapters; no live monitor instance is exposed.

    Args:
        run: The frozen run bundle.

    Returns:
        The validated JSON-safe pack report document.
    """
    summary = RunSummary(
        metadata=run.metadata,
        episodes=run.records,
        episode_summary=run.summary,
        monitor_views=run.monitor_views,
        retention=run.retention,
    )
    blocks = {extension.name: extension.on_run_end(summary) for extension in run.extensions}
    monitors = tuple(_MonitorView(name, run.monitor_views[name]) for name in run.monitor_names)
    builder = ReportBuilder(
        monitors=monitors,
        extensions=run.extensions,
        monitor_extensions=run.monitor_extensions,
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
