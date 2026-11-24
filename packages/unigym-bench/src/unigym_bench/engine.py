"""Per-run streaming engine for the external benchmark pack.

The engine is the pack's only state owner for a run.  It receives the runner
bundle once per tick, builds a :class:`~unigym_bench.frames.TickFrame`, drives
the task monitors in declaration order and owns the consequences:

- per-env open flags and the first-terminal latch;
- a run-scoped event journal with an explicit retention cap and dropped count;
- pending ``env_id -> Termination`` drains for the core runner;
- per-env episode sequence, episode-tick counts and finalized
  :class:`~unigym_bench.events.EpisodeRecord` values, including the
  JSON-safe start command and reset facts captured on the episode's first tick.

The engine never keeps an ``EnvView``, a frame, a tensor or a backend handle.
Signals are validated and published one tick at a time; they are not stored.

Lifecycle (delegated by the sink in the next card)::

    sink.on_step -> engine.on_step(bundle)
    runner drain -> engine.drain_terminations()
    runner due   -> engine.on_episode_end(due, bundle) -> due ids (sink stores)
    runner reset -> engine.on_reset(stored due)          # reopen new episodes
    runner end   -> engine.on_run_end()

The core runner does not pass reset ids to ``on_step_end``, so the sink keeps
the ids returned by ``on_episode_end`` until the post-reset callback and calls
``on_reset`` there.

A terminal request for an environment that is not open is a pack contract
error.  A terminal request for an environment that is open but already latched
is a *conflict*: it is counted and not emitted.  Signal publication is skipped
on a tick that still contains a closed environment because a level vector is
full-run and the transport has no per-env mask; a normal runner tick is fully
open.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from unigym.core.episode_termination import (
    EpisodeOutcome,
    EpisodeTermination,
    Termination,
    TerminationSource,
    validate_code,
)

from .events import (
    VERDICTS,
    EpisodeRecord,
    EpisodeSummary,
    EpisodeVerdict,
    Event,
    termination_from_json,
)
from .frames import TickFrame
from .json import JsonValue, ensure_json_value
from .monitors import Monitor
from .report import ReportExtension, validate_report_extensions
from .specs import RetentionSpec

if TYPE_CHECKING:
    from unigym.signals import SignalPublisher

#: Checkpoint payload key and schema owned by the pack's rewind provider.
#: Schema 2 adds the retention-independent run counters (rewind/termination/
#: per-env aggregates) required for an honest default-retention report.
REWIND_PROVIDER_NAME = "unigym_bench"
REWIND_SCHEMA = "unigym_bench.rewind/2"

__all__ = [
    "EngineStateError",
    "REWIND_PROVIDER_NAME",
    "REWIND_SCHEMA",
    "StreamingEngine",
]

_REWIND_KEYS = frozenset({"schema", "engine", "monitors"})
_REWIND_OPTIONAL_KEYS = frozenset({"extensions"})
_ENGINE_KEYS = frozenset(
    {
        "bound",
        "run_ended",
        "num_envs",
        "device",
        "retention",
        "open",
        "rewound",
        "episode_seq",
        "episode_ticks",
        "episode_events",
        "episode_command",
        "episode_reset",
        "latched",
        "pending",
        "conflicts",
        "conflict_total",
        "journal",
        "dropped_events",
        "event_total",
        "records",
        "episode_total",
        "tick_total",
        "verdict_counts",
        "rewound_total",
        "termination_outcome_counts",
        "termination_code_counts",
        "termination_source_counts",
        "episode_counts_by_env",
    }
)
_RETENTION_KEYS = frozenset({"retain_episodes", "max_events", "event_log"})


class EngineStateError(RuntimeError):
    """The engine was driven outside its declared per-run lifecycle."""


@dataclass(frozen=True)
class _DecodedEngineState:
    """One fully validated engine payload held as private locals."""

    bound: bool
    run_ended: bool
    num_envs: int | None
    device: torch.device | None
    open: list[bool]
    rewound: list[bool]
    episode_seq: list[int]
    episode_ticks: list[int]
    episode_events: list[dict[str, int]]
    episode_command: list[tuple[float, ...] | None]
    episode_reset: list[dict[str, JsonValue]]
    latched: dict[int, Termination]
    pending: dict[int, Termination]
    conflicts: dict[int, int]
    conflict_total: int
    journal: list[Event]
    dropped_events: int
    event_total: int
    records: list[EpisodeRecord]
    episode_total: int
    tick_total: int
    verdict_counts: dict[str, int]
    rewound_total: int
    termination_outcome_counts: dict[str, int]
    termination_code_counts: dict[str, int]
    termination_source_counts: dict[str, int]
    episode_counts_by_env: list[int]


@dataclass(frozen=True)
class _DecodedRewindState:
    """One validated checkpoint document before any live state is touched."""

    engine: _DecodedEngineState
    monitors: dict[str, JsonValue]
    extensions: dict[str, JsonValue]


class StreamingEngine:
    """One run's streaming state, monitor driver and terminal authority.

    Args:
        monitors: Monitor instances in task declaration order.  Names must be
            unique because they namespace published signals and report blocks.
        retention: Explicit event/episode retention policy.  The default is the
            pack's documented ``RetentionSpec`` default.
        publisher: Optional write-only signal publisher.  When supplied, every
            monitor level is published as ````<monitor>/<signal>````.
        extensions: Report extensions in task declaration order.  A stateful
            extension must implement snapshot/restore so the pack provider can
            restore its accumulator.
    """

    #: Structural rewind-provider name; checkpoint payload keys are namespaced.
    name: str = REWIND_PROVIDER_NAME

    def __init__(
        self,
        monitors: Sequence[Monitor],
        *,
        retention: RetentionSpec | None = None,
        publisher: SignalPublisher | None = None,
        extensions: Sequence[ReportExtension] = (),
    ) -> None:
        if isinstance(monitors, (str, bytes, bytearray)) or not isinstance(monitors, Sequence):
            raise TypeError(
                "engine monitors: expected a sequence of Monitor instances, "
                f"got {type(monitors).__name__}"
            )
        checked: list[Monitor] = []
        names: set[str] = set()
        for monitor in monitors:
            if not isinstance(monitor, Monitor):
                raise TypeError(
                    "engine monitors: expected a Monitor, " f"got {type(monitor).__name__}"
                )
            name = _require_monitor_name(monitor.name)
            if name in names:
                raise ValueError(f"engine monitors: duplicate monitor name {name!r}")
            names.add(name)
            checked.append(monitor)
        checked_extensions = validate_report_extensions(extensions)
        if retention is None:
            retention = RetentionSpec()
        elif not isinstance(retention, RetentionSpec):
            raise TypeError(
                "engine retention: expected a RetentionSpec or None, "
                f"got {type(retention).__name__}"
            )
        if publisher is not None and not callable(getattr(publisher, "publish", None)):
            raise TypeError(
                "engine publisher: expected an object with a callable publish(name, value), "
                f"got {type(publisher).__name__}"
            )

        self._monitors: tuple[Monitor, ...] = tuple(checked)
        self._extensions: tuple[ReportExtension, ...] = checked_extensions
        self._retention = retention
        self._publisher = publisher

        self._bound = False
        self._run_ended = False
        self._num_envs: int | None = None
        self._device: torch.device | None = None

        self._open: list[bool] = []
        self._rewound: list[bool] = []
        self._episode_seq: list[int] = []
        self._episode_ticks: list[int] = []
        self._episode_events: list[dict[str, int]] = []
        self._episode_command: list[tuple[float, ...] | None] = []
        self._episode_reset: list[dict[str, JsonValue]] = []

        self._latched: dict[int, Termination] = {}
        self._pending: dict[int, Termination] = {}
        self._conflicts: dict[int, int] = {}
        self._conflict_total = 0

        self._journal: list[Event] = []
        self._dropped_events = 0
        self._event_total = 0

        self._records: list[EpisodeRecord] = []
        self._episode_total = 0
        self._tick_total = 0
        self._verdict_counts: dict[str, int] = {}
        self._rewound_total = 0
        self._termination_outcome_counts: dict[str, int] = {}
        self._termination_code_counts: dict[str, int] = {}
        self._termination_source_counts: dict[str, int] = {}
        self._episode_counts_by_env: list[int] = []

    # ------------------------------------------------------------------ #
    # Public read-only state
    # ------------------------------------------------------------------ #
    @property
    def num_envs(self) -> int | None:
        """Bound environment count, or ``None`` before the first tick."""
        return self._num_envs

    @property
    def device(self) -> torch.device | None:
        """Bound tensor device, or ``None`` before the first tick."""
        return self._device

    @property
    def publisher(self) -> SignalPublisher | None:
        """The write-only signal publisher, or ``None`` when not configured."""
        return self._publisher

    @property
    def retention(self) -> RetentionSpec:
        """The explicit event/episode retention policy for this run."""
        return self._retention

    @property
    def monitors(self) -> tuple[Monitor, ...]:
        """The monitors in task declaration order."""
        return self._monitors

    @property
    def extensions(self) -> tuple[ReportExtension, ...]:
        """The report extensions in task declaration order."""
        return self._extensions

    @property
    def open_envs(self) -> tuple[int, ...]:
        """Environment ids currently inside an episode."""
        return tuple(env_id for env_id, is_open in enumerate(self._open) if is_open)

    @property
    def journal(self) -> tuple[Event, ...]:
        """Retained event occurrences in acceptance order."""
        return tuple(self._journal)

    @property
    def dropped_events(self) -> int:
        """Occurrences not written to the journal because retention dropped them."""
        return self._dropped_events

    @property
    def conflicts(self) -> Mapping[int, int]:
        """Per-env count of late terminal requests for an already-latched env."""
        return MappingProxyType(dict(self._conflicts))

    @property
    def conflict_count(self) -> int:
        """Total late terminal requests counted across the run."""
        return self._conflict_total

    @property
    def records(self) -> tuple[EpisodeRecord, ...]:
        """Finalized records retained under the declared retention policy."""
        return tuple(self._records)

    @property
    def summary(self) -> EpisodeSummary:
        """Aggregate finalized-episode totals for the current run."""
        return EpisodeSummary(
            episodes=self._episode_total,
            total_ticks=self._tick_total,
            verdict_counts=MappingProxyType(dict(self._verdict_counts)),
            event_count=self._event_total,
            dropped_events=self._dropped_events,
            conflicts=self._conflict_total,
            rewound=self._rewound_total,
            termination_outcome_counts=MappingProxyType(dict(self._termination_outcome_counts)),
            termination_code_counts=MappingProxyType(dict(self._termination_code_counts)),
            termination_source_counts=MappingProxyType(dict(self._termination_source_counts)),
            episodes_by_env=MappingProxyType(
                {env_id: count for env_id, count in enumerate(self._episode_counts_by_env)}
            ),
        )

    # ------------------------------------------------------------------ #
    # Core rewind-provider face
    # ------------------------------------------------------------------ #
    def rewind_state(self) -> dict[str, JsonValue]:
        """Return a JSON-safe, private-copied checkpoint of the run state.

        The payload owns only pack state: engine flags/counters, journal,
        latches/pending drains, report accumulators that already exist, and
        every rewindable monitor's ``snapshot()``.  The sampler cursor/RNG and
        the signal bus are separate core providers and are never duplicated
        here.
        """
        engine: dict[str, JsonValue] = {
            "bound": self._bound,
            "run_ended": self._run_ended,
            "num_envs": self._num_envs,
            "device": None if self._device is None else str(self._device),
            "retention": _retention_state(self._retention),
            "open": list(self._open),
            "rewound": list(self._rewound),
            "episode_seq": list(self._episode_seq),
            "episode_ticks": list(self._episode_ticks),
            "episode_events": [dict(counts) for counts in self._episode_events],
            "episode_command": [
                None if command is None else list(command) for command in self._episode_command
            ],
            "episode_reset": [dict(facts) for facts in self._episode_reset],
            "latched": {
                str(env_id): termination.to_json()
                for env_id, termination in sorted(self._latched.items())
            },
            "pending": {
                str(env_id): termination.to_json()
                for env_id, termination in sorted(self._pending.items())
            },
            "conflicts": {str(env_id): count for env_id, count in sorted(self._conflicts.items())},
            "conflict_total": self._conflict_total,
            "journal": [event.to_json() for event in self._journal],
            "dropped_events": self._dropped_events,
            "event_total": self._event_total,
            "records": [record.to_json() for record in self._records],
            "episode_total": self._episode_total,
            "tick_total": self._tick_total,
            "verdict_counts": dict(self._verdict_counts),
            "rewound_total": self._rewound_total,
            "termination_outcome_counts": dict(self._termination_outcome_counts),
            "termination_code_counts": dict(self._termination_code_counts),
            "termination_source_counts": dict(self._termination_source_counts),
            "episode_counts_by_env": list(self._episode_counts_by_env),
        }
        monitors: dict[str, JsonValue] = {}
        for monitor in self._monitors:
            snapshot = _monitor_snapshot(monitor)
            if snapshot is not None:
                monitors[monitor.name] = snapshot
        document: dict[str, JsonValue] = {
            "schema": REWIND_SCHEMA,
            "engine": engine,
            "monitors": monitors,
        }
        extensions: dict[str, JsonValue] = {}
        for extension in self._extensions:
            if extension.stateful:
                extensions[extension.name] = _extension_snapshot(extension)
        if extensions:
            # The ``extensions`` key is optional: a payload with no stateful
            # extension keeps the three-key shape.  It is present whenever
            # state must be restored, so a stateful extension can never silently
            # resume from a checkpoint that lacks its accumulator.
            document["extensions"] = extensions
        normalized = ensure_json_value(document, where="unigym_bench rewind state")
        if not isinstance(normalized, dict):
            raise TypeError("unigym_bench rewind state: expected an object")
        return normalized

    def validate_rewind_state(self, state: object) -> None:
        """Read-only pre-flight with the same judgment as :meth:`apply_rewind_state`."""
        self._decode_rewind_state(state)

    def apply_rewind_state(self, state: object) -> None:
        """Restore a checkpoint in place after fully decoding and validating it.

        Monitor and stateful-extension snapshots are pre-flighted with the
        read-only ``validate_restore`` capability when present.  The live state
        of every rewindable monitor and extension is captured before anything is
        touched so an unexpected restore failure rolls all of them back; engine
        fields are assigned only after every restore succeeded.
        """
        decoded = self._decode_rewind_state(state)
        monitor_rollback: dict[str, JsonValue] = {}
        for monitor in self._monitors:
            snapshot = _monitor_snapshot(monitor)
            if snapshot is not None:
                monitor_rollback[monitor.name] = snapshot
        extension_rollback: dict[str, JsonValue] = {}
        for extension in self._extensions:
            if extension.stateful:
                extension_rollback[extension.name] = _extension_snapshot(extension)
        expected_extensions = _stateful_extensions(self._extensions)
        try:
            for monitor in self._monitors:
                target = decoded.monitors.get(monitor.name)
                if target is None:
                    continue
                restore = getattr(monitor, "restore", None)
                if not callable(restore):
                    raise TypeError(
                        f"monitor {monitor.name!r}: rewind snapshot present but no "
                        "callable restore"
                    )
                restore(target)
            for name, target in decoded.extensions.items():
                restore = getattr(expected_extensions[name], "restore", None)
                if not callable(restore):
                    raise TypeError(
                        f"report extension {name!r}: rewind snapshot present but no "
                        "callable restore"
                    )
                restore(target)
        except Exception:
            for monitor in self._monitors:
                previous = monitor_rollback.get(monitor.name)
                if previous is not None:
                    monitor.restore(previous)
            for name, previous in extension_rollback.items():
                expected_extensions[name].restore(previous)
            raise
        self._apply_engine_state(decoded.engine)

    # ------------------------------------------------------------------ #
    # Runner-facing lifecycle
    # ------------------------------------------------------------------ #
    def on_step(self, bundle: Mapping[str, object]) -> None:
        """Consume one pre-reset engine step.

        Args:
            bundle: The core runner bundle.  ``view``, ``t``, ``step_index``,
                ``time_s``, ``cmd`` and ``reset_state`` are read into a borrowed
                :class:`~unigym_bench.frames.TickFrame`.

        Raises:
            EngineStateError: The run has ended or a monitor requested a
                terminal for a closed environment.
            TypeError: A bundle field, monitor output or signal violates the
                contract.
            ValueError: A shape, device, source or code is invalid.
        """
        self._require_running()
        if not isinstance(bundle, Mapping):
            raise TypeError(f"engine bundle: expected a mapping, got {type(bundle).__name__}")
        frame = self._build_frame(bundle)
        self._bind(frame)

        num_envs = self._require_bound()
        for env_id in range(num_envs):
            if self._open[env_id]:
                self._episode_ticks[env_id] += 1
        for env_id in range(num_envs):
            if self._open[env_id] and self._episode_ticks[env_id] == 1:
                self._capture_episode_start(env_id, frame)

        for monitor in self._monitors:
            monitor.on_tick(frame)

        collected: list[
            tuple[Monitor, dict[str, Tensor], tuple[Event, ...], dict[int, Termination]]
        ] = []
        for monitor in self._monitors:
            collected.append(
                (
                    monitor,
                    self._collect_signals(monitor, frame),
                    self._collect_events(monitor),
                    self._collect_terminals(monitor, num_envs),
                )
            )

        # Validate every occurrence before mutating journal/latches so a bad
        # output cannot leave a half-applied tick behind.
        for monitor, _, events, terminals in collected:
            for event in events:
                self._validate_event_envs(monitor, event)
            for env_id in terminals:
                if not self._open[env_id]:
                    raise EngineStateError(
                        f"monitor {monitor.name!r}: terminal for closed env {env_id}; "
                        f"open envs are {list(self.open_envs)}"
                    )

        if self._publisher is not None and all(self._open):
            for monitor, signals, _, _ in collected:
                for local_name, value in signals.items():
                    self._publisher.publish(f"{monitor.name}/{local_name}", value)

        for _, _, events, _ in collected:
            for event in events:
                self._record_event(event)

        for _, _, _, terminals in collected:
            for env_id in sorted(terminals):
                self._accept_terminal(env_id, terminals[env_id])

    def drain_terminations(self) -> dict[int, Termination]:
        """Return and clear the pending first-terminal facts.

        The runner calls this exactly once after ``on_step`` returns.  Only the
        first terminal for each open environment is present; conflicts are
        counted by :attr:`conflicts` instead.
        """
        pending = dict(self._pending)
        self._pending.clear()
        return pending

    def on_episode_end(self, due: object, bundle: Mapping[str, object]) -> tuple[int, ...]:
        """Finalize the due environments and clear their latches.

        Args:
            due: Sequence of env ids or a 1-D integer tensor from the runner.
            bundle: The runner's episode-end bundle.  Its ``terminations``
                mapping supplies the canonical terminal stamps and core facts;
                ``reset_facts`` are not read here because the record keeps the
                JSON-safe reset state captured on the episode's first tick.

        Returns:
            The sorted, unique due env ids.  The sink stores this tuple until
            ``on_step_end`` and then reopens exactly these envs after the
            runner's reset chain.

        Raises:
            EngineStateError: The engine is unbound/ended, a due env is closed,
                or a monitor reports a lifecycle violation.
        """
        self._require_running()
        num_envs = self._require_bound()
        if not isinstance(bundle, Mapping):
            raise TypeError(
                f"engine episode-end bundle: expected a mapping, got {type(bundle).__name__}"
            )
        due_ids = _as_env_ids(due, num_envs, "engine episode_end.due")
        if not due_ids:
            return ()
        for env_id in due_ids:
            if not self._open[env_id]:
                raise EngineStateError(
                    f"engine episode_end.due: env {env_id} is already closed; "
                    f"open envs are {list(self.open_envs)}"
                )

        records = [
            EpisodeRecord(
                env_id=env_id,
                episode_seq=self._episode_seq[env_id],
                ticks=self._episode_ticks[env_id],
                verdict=(
                    "terminal" if env_id in self._latched else self._due_verdict(bundle, env_id)
                ),
                command_at_start=self._episode_command[env_id],
                first_terminal=self._first_terminal(bundle, env_id),
                core_termination=self._core_termination(bundle, env_id),
                reset_facts=dict(self._episode_reset[env_id]),
                event_counts=dict(self._episode_events[env_id]),
                rewound=self._rewound[env_id],
            )
            for env_id in due_ids
        ]

        for monitor in self._monitors:
            monitor.on_episode_end(due_ids)
        if self._extensions:
            # Only this boundary's envs are visible, and the records are frozen;
            # an extension cannot reach another env or mutate a finalized record.
            due_records = MappingProxyType({record.env_id: record for record in records})
            for extension in self._extensions:
                extension.on_env_end(due_ids, due_records)
        for monitor in self._monitors:
            monitor.reset(due_ids)

        for record in records:
            self._finalize_record(record)
        for env_id in due_ids:
            self._latched.pop(env_id, None)
            self._pending.pop(env_id, None)
            self._episode_events[env_id].clear()
            self._episode_command[env_id] = None
            self._episode_reset[env_id].clear()
            self._open[env_id] = False
            self._rewound[env_id] = False
        return due_ids

    def on_reset(self, env_ids: object) -> None:
        """Reopen the environments the runner has just reset.

        The pack sink calls this after the runner's reset chain.  It starts a
        new episode sequence for each id and clears the per-episode tick and
        event counters.  It does not call ``monitor.reset`` again; episode-end
        finalization already did that.

        Raises:
            EngineStateError: The engine is unbound/ended or an id is open.
        """
        self._require_running()
        num_envs = self._require_bound()
        reset_ids = _as_env_ids(env_ids, num_envs, "engine reset.env_ids")
        for env_id in reset_ids:
            if self._open[env_id]:
                raise EngineStateError(
                    f"engine reset.env_ids: env {env_id} is still open; "
                    "a reset may only reopen an env finalized by on_episode_end"
                )
            if env_id in self._latched or env_id in self._pending:
                raise EngineStateError(
                    f"engine reset.env_ids: env {env_id} still has a terminal latch "
                    "or pending drain; on_episode_end must clear it first"
                )
        for env_id in reset_ids:
            self._open[env_id] = True
            self._episode_seq[env_id] += 1
            self._episode_ticks[env_id] = 0
            self._episode_events[env_id].clear()
            self._episode_command[env_id] = None
            self._episode_reset[env_id].clear()
            self._rewound[env_id] = False

    def on_run_end(self) -> EpisodeSummary:
        """Finalize still-open episodes as ``incomplete`` and return the summary.

        This is deliberately not a score: an env that never received a due
        boundary is recorded as incomplete with its observed tick count and any
        latched first terminal.  An env the sink reopened after a reset but
        that never produced a tick is not an episode yet, so it is closed
        silently instead of being recorded as a phantom zero-tick episode.  The
        method is idempotent for teardown paths.
        """
        if self._run_ended:
            return self.summary
        if self._bound:
            open_ids = self.open_envs
            recorded_ids = tuple(env_id for env_id in open_ids if self._episode_ticks[env_id] > 0)
            records = [
                EpisodeRecord(
                    env_id=env_id,
                    episode_seq=self._episode_seq[env_id],
                    ticks=self._episode_ticks[env_id],
                    verdict="rewound" if self._rewound[env_id] else "incomplete",
                    command_at_start=self._episode_command[env_id],
                    first_terminal=self._latched.get(env_id),
                    reset_facts=dict(self._episode_reset[env_id]),
                    event_counts=dict(self._episode_events[env_id]),
                    rewound=self._rewound[env_id],
                )
                for env_id in recorded_ids
            ]
            for record in records:
                self._finalize_record(record)
            for env_id in open_ids:
                self._latched.pop(env_id, None)
                self._episode_events[env_id].clear()
                self._episode_command[env_id] = None
                self._episode_reset[env_id].clear()
                self._open[env_id] = False
                self._rewound[env_id] = False
            self._pending.clear()
        self._run_ended = True
        return self.summary

    def on_rewind(self, info: Mapping[str, object]) -> None:
        """Mark every currently open episode as belonging to a forked timeline.

        The core runner calls this after the provider restore, just before the
        restored timeline's first step.  The mark rides the episode until it
        finalizes: a due record keeps its natural verdict plus ``rewound=True``;
        an episode still open at run end is reported with verdict ``rewound``.
        """
        del info
        for env_id, is_open in enumerate(self._open):
            if is_open:
                self._rewound[env_id] = True

    # ------------------------------------------------------------------ #
    # Rewind payload decode and apply
    # ------------------------------------------------------------------ #
    def _decode_rewind_state(self, state: object) -> _DecodedRewindState:
        """Decode and validate a checkpoint without mutating live state."""
        payload = ensure_json_value(state, where="unigym_bench rewind state")
        if not isinstance(payload, dict):
            raise TypeError(
                "unigym_bench rewind state: expected an object, " f"got {type(payload).__name__}"
            )
        where = "unigym_bench rewind state"
        _require_keys(payload, _REWIND_KEYS, where, optional=_REWIND_OPTIONAL_KEYS)
        if payload["schema"] != REWIND_SCHEMA:
            raise ValueError(
                f"{where}.schema: expected {REWIND_SCHEMA!r}, got {payload['schema']!r}"
            )
        raw_engine = payload["engine"]
        if not isinstance(raw_engine, dict):
            raise TypeError(f"{where}.engine: expected an object, got {type(raw_engine).__name__}")
        engine = self._decode_engine_state(raw_engine)
        raw_monitors = payload["monitors"]
        if not isinstance(raw_monitors, dict):
            raise TypeError(
                f"{where}.monitors: expected an object, got {type(raw_monitors).__name__}"
            )
        expected: dict[str, Monitor] = {}
        for monitor in self._monitors:
            if _rewind_capability(monitor) is not None:
                expected[monitor.name] = monitor
        if set(raw_monitors) != set(expected):
            raise ValueError(
                f"{where}.monitors: checkpoint monitor names "
                f"{sorted(raw_monitors)} do not match the live rewindable monitors "
                f"{sorted(expected)}"
            )
        monitors: dict[str, JsonValue] = {}
        for name, item in raw_monitors.items():
            monitor = expected[name]
            monitor_where = f"{where}.monitors[{name!r}]"
            normalized = ensure_json_value(item, where=monitor_where)
            validate = getattr(monitor, "validate_restore", None)
            if callable(validate):
                validate(normalized)
            monitors[name] = normalized
        raw_extensions = payload.get("extensions", {})
        if not isinstance(raw_extensions, dict):
            raise TypeError(
                f"{where}.extensions: expected an object, " f"got {type(raw_extensions).__name__}"
            )
        expected_extensions = _stateful_extensions(self._extensions)
        if set(raw_extensions) != set(expected_extensions):
            raise ValueError(
                f"{where}.extensions: checkpoint extension names "
                f"{sorted(raw_extensions)} do not match the live stateful extensions "
                f"{sorted(expected_extensions)}"
            )
        extensions: dict[str, JsonValue] = {}
        for name, item in raw_extensions.items():
            extension = expected_extensions[name]
            extension_where = f"{where}.extensions[{name!r}]"
            normalized = ensure_json_value(item, where=extension_where)
            validate = getattr(extension, "validate_restore", None)
            if callable(validate):
                validate(normalized)
            extensions[name] = normalized
        return _DecodedRewindState(engine=engine, monitors=monitors, extensions=extensions)

    def _decode_engine_state(self, raw: Mapping[str, JsonValue]) -> _DecodedEngineState:
        """Decode the engine-owned part of the payload into private locals."""
        where = "unigym_bench rewind state.engine"
        _require_keys(raw, _ENGINE_KEYS, where)
        bound = _require_bool(raw["bound"], f"{where}.bound")
        run_ended = _require_bool(raw["run_ended"], f"{where}.run_ended")
        num_envs = _optional_positive_int(raw["num_envs"], f"{where}.num_envs")
        device = _optional_device(raw["device"], f"{where}.device")
        if bound:
            if num_envs is None or device is None:
                raise ValueError(
                    f"{where}: a bound checkpoint requires num_envs and device, "
                    f"got num_envs={num_envs!r}, device={device!r}"
                )
            if self._bound:
                if num_envs != self._num_envs:
                    raise ValueError(
                        f"{where}.num_envs: checkpoint {num_envs} does not match "
                        f"live engine num_envs {self._num_envs}"
                    )
                if device != self._device:
                    raise ValueError(
                        f"{where}.device: checkpoint {device} does not match "
                        f"live engine device {self._device}"
                    )
        else:
            if num_envs is not None or device is not None:
                raise ValueError(
                    f"{where}: an unbound checkpoint must not carry num_envs/device, "
                    f"got {num_envs!r}/{device!r}"
                )
            if self._bound:
                raise ValueError(f"{where}: checkpoint is unbound but the live engine is bound")
        retention = _decode_retention(raw["retention"], f"{where}.retention")
        if retention != _retention_state(self._retention):
            raise ValueError(
                f"{where}.retention: checkpoint retention {retention!r} does not match "
                f"live retention {_retention_state(self._retention)!r}"
            )
        open_flags = _decode_bool_list(raw["open"], f"{where}.open", num_envs)
        rewound = _decode_bool_list(raw["rewound"], f"{where}.rewound", num_envs)
        episode_seq = _decode_int_list(raw["episode_seq"], f"{where}.episode_seq", num_envs)
        episode_ticks = _decode_int_list(raw["episode_ticks"], f"{where}.episode_ticks", num_envs)
        episode_events = _decode_event_counts_list(
            raw["episode_events"], f"{where}.episode_events", num_envs
        )
        episode_command = _decode_command_list(
            raw["episode_command"], f"{where}.episode_command", num_envs
        )
        episode_reset = _decode_reset_list(raw["episode_reset"], f"{where}.episode_reset", num_envs)
        latched = _decode_termination_map(raw["latched"], f"{where}.latched", num_envs)
        pending = _decode_termination_map(raw["pending"], f"{where}.pending", num_envs)
        for env_id in latched:
            if not open_flags[env_id]:
                raise ValueError(
                    f"{where}.latched: env {env_id} has a latch but the episode is closed"
                )
        for env_id in pending:
            if env_id not in latched:
                raise ValueError(
                    f"{where}.pending: env {env_id} has a pending drain without a latch"
                )
        conflicts = _decode_conflict_map(raw["conflicts"], f"{where}.conflicts", num_envs)
        conflict_total = _nonnegative_int(raw["conflict_total"], f"{where}.conflict_total")
        journal = _decode_journal(raw["journal"], f"{where}.journal", num_envs)
        if not self._retention.event_log:
            if journal:
                raise ValueError(
                    f"{where}.journal: checkpoint carries entries but live retention "
                    "has event_log disabled"
                )
        elif len(journal) > self._retention.max_events:
            raise ValueError(
                f"{where}.journal: checkpoint has {len(journal)} entries but live "
                f"retention max_events is {self._retention.max_events}"
            )
        records = _decode_records(raw["records"], f"{where}.records", num_envs)
        episode_total = _nonnegative_int(raw["episode_total"], f"{where}.episode_total")
        tick_total = _nonnegative_int(raw["tick_total"], f"{where}.tick_total")
        if self._retention.retain_episodes and len(records) != episode_total:
            raise ValueError(
                f"{where}.records: checkpoint retains {len(records)} records but "
                f"episode_total is {episode_total}"
            )
        if not self._retention.retain_episodes and records:
            raise ValueError(
                f"{where}.records: checkpoint carries records but live retention "
                "has retain_episodes disabled"
            )
        verdict_counts = _decode_verdict_counts(raw["verdict_counts"], f"{where}.verdict_counts")
        dropped_events = _nonnegative_int(raw["dropped_events"], f"{where}.dropped_events")
        event_total = _nonnegative_int(raw["event_total"], f"{where}.event_total")
        rewound_total = _nonnegative_int(raw["rewound_total"], f"{where}.rewound_total")
        if rewound_total > episode_total:
            raise ValueError(
                f"{where}.rewound_total: expected a value <= episode_total "
                f"{episode_total}, got {rewound_total}"
            )
        termination_outcome_counts = _decode_string_count_map(
            raw["termination_outcome_counts"], f"{where}.termination_outcome_counts"
        )
        termination_code_counts = _decode_string_count_map(
            raw["termination_code_counts"], f"{where}.termination_code_counts"
        )
        termination_source_counts = _decode_string_count_map(
            raw["termination_source_counts"], f"{where}.termination_source_counts"
        )
        episode_counts_by_env = _decode_int_list(
            raw["episode_counts_by_env"], f"{where}.episode_counts_by_env", num_envs
        )
        if sum(episode_counts_by_env) != episode_total:
            raise ValueError(
                f"{where}.episode_counts_by_env: counts sum to "
                f"{sum(episode_counts_by_env)} but episode_total is {episode_total}"
            )
        return _DecodedEngineState(
            bound=bound,
            run_ended=run_ended,
            num_envs=num_envs,
            device=device,
            open=list(open_flags),
            rewound=list(rewound),
            episode_seq=list(episode_seq),
            episode_ticks=list(episode_ticks),
            episode_events=[dict(counts) for counts in episode_events],
            episode_command=list(episode_command),
            episode_reset=[dict(facts) for facts in episode_reset],
            latched=latched,
            pending=pending,
            conflicts=conflicts,
            conflict_total=conflict_total,
            journal=journal,
            dropped_events=dropped_events,
            event_total=event_total,
            records=records,
            episode_total=episode_total,
            tick_total=tick_total,
            verdict_counts=verdict_counts,
            rewound_total=rewound_total,
            termination_outcome_counts=termination_outcome_counts,
            termination_code_counts=termination_code_counts,
            termination_source_counts=termination_source_counts,
            episode_counts_by_env=episode_counts_by_env,
        )

    def _apply_engine_state(self, decoded: _DecodedEngineState) -> None:
        """Assign the validated engine locals in place."""
        self._bound = decoded.bound
        self._run_ended = decoded.run_ended
        self._num_envs = decoded.num_envs
        self._device = decoded.device
        self._open = decoded.open
        self._rewound = decoded.rewound
        self._episode_seq = decoded.episode_seq
        self._episode_ticks = decoded.episode_ticks
        self._episode_events = decoded.episode_events
        self._episode_command = decoded.episode_command
        self._episode_reset = decoded.episode_reset
        self._latched = decoded.latched
        self._pending = decoded.pending
        self._conflicts = decoded.conflicts
        self._conflict_total = decoded.conflict_total
        self._journal = decoded.journal
        self._dropped_events = decoded.dropped_events
        self._event_total = decoded.event_total
        self._records = decoded.records
        self._episode_total = decoded.episode_total
        self._tick_total = decoded.tick_total
        self._verdict_counts = decoded.verdict_counts
        self._rewound_total = decoded.rewound_total
        self._termination_outcome_counts = decoded.termination_outcome_counts
        self._termination_code_counts = decoded.termination_code_counts
        self._termination_source_counts = decoded.termination_source_counts
        self._episode_counts_by_env = decoded.episode_counts_by_env

    # ------------------------------------------------------------------ #
    # Frame and output validation
    # ------------------------------------------------------------------ #
    def _build_frame(self, bundle: Mapping[str, object]) -> TickFrame:
        """Build the borrowed tick frame from the runner bundle."""
        view = bundle.get("view")
        if view is None:
            raise TypeError("engine bundle.view: expected an EnvView, got None")
        num_envs = getattr(view, "num_envs", None)
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise TypeError(
                "engine bundle.view.num_envs: expected a positive int, " f"got {num_envs!r}"
            )
        raw_device = getattr(view, "device", None)
        if raw_device is None:
            raise TypeError("engine bundle.view.device: expected a torch device, got None")
        try:
            device = torch.device(raw_device)
        except (TypeError, RuntimeError) as exc:
            raise TypeError(
                f"engine bundle.view.device: expected a torch device, got {raw_device!r}"
            ) from exc
        for key in ("t", "step_index"):
            if key not in bundle:
                raise TypeError(f"engine bundle.{key}: required field is missing")
        return TickFrame(
            view=view,
            t=bundle["t"],
            step_index=bundle["step_index"],
            time_s=bundle.get("time_s"),
            cmd=bundle.get("cmd"),
            reset_state=bundle.get("reset_state", {}),
            num_envs=num_envs,
            device=device,
        )

    def _bind(self, frame: TickFrame) -> None:
        """Bind the run shape on the first tick and reject later changes."""
        if not self._bound:
            self._num_envs = frame.num_envs
            self._device = frame.device
            self._open = [True] * frame.num_envs
            self._rewound = [False] * frame.num_envs
            self._episode_seq = [0] * frame.num_envs
            self._episode_ticks = [0] * frame.num_envs
            self._episode_events = [{} for _ in range(frame.num_envs)]
            self._episode_command = [None] * frame.num_envs
            self._episode_reset = [{} for _ in range(frame.num_envs)]
            self._episode_counts_by_env = [0] * frame.num_envs
            self._bound = True
            return
        if frame.num_envs != self._num_envs:
            raise ValueError(f"engine: num_envs changed from {self._num_envs} to {frame.num_envs}")
        if frame.device != self._device:
            raise ValueError(f"engine: device changed from {self._device} to {frame.device}")

    def _capture_episode_start(self, env_id: int, frame: TickFrame) -> None:
        """Record the JSON-safe command and reset facts of a new episode."""
        self._episode_reset[env_id] = dict(frame.reset_state.get(env_id, {}))
        cmd = frame.cmd
        if cmd is None:
            self._episode_command[env_id] = None
            return
        row = cmd[env_id].detach().to(device="cpu").reshape(-1).tolist()
        self._episode_command[env_id] = tuple(float(value) for value in row)

    def _termination_entry(
        self, bundle: Mapping[str, object], env_id: int
    ) -> EpisodeTermination | None:
        """Return the runner's termination fact for one env, if present."""
        terminations = bundle.get("terminations")
        if not isinstance(terminations, Mapping):
            return None
        entry = terminations.get(env_id)
        if isinstance(entry, EpisodeTermination):
            return entry
        if isinstance(entry, Termination):
            return EpisodeTermination(primary=entry)
        return None

    def _first_terminal(self, bundle: Mapping[str, object], env_id: int) -> Termination | None:
        """Return the latched terminal, preferring its canonical stamped copy."""
        latched = self._latched.get(env_id)
        if latched is None:
            return None
        entry = self._termination_entry(bundle, env_id)
        if entry is not None:
            for term in entry.all:
                if (
                    term.source is latched.source
                    and term.code == latched.code
                    and term.outcome is latched.outcome
                ):
                    return term
        return latched

    def _core_termination(
        self, bundle: Mapping[str, object], env_id: int
    ) -> EpisodeTermination | None:
        """Return the core-owned part of the runner's termination fact."""
        entry = self._termination_entry(bundle, env_id)
        if entry is None:
            return None
        core_terms = [term for term in entry.all if term.source is not TerminationSource.CUSTOM]
        if not core_terms:
            return None
        return EpisodeTermination.from_terminations(core_terms)

    def _collect_signals(self, monitor: Monitor, frame: TickFrame) -> dict[str, Tensor]:
        """Validate one monitor's fresh per-env level tensors."""
        signals = monitor.signals()
        if not isinstance(signals, Mapping):
            raise TypeError(
                f"monitor {monitor.name!r}.signals: expected a mapping, "
                f"got {type(signals).__name__}"
            )
        checked: dict[str, Tensor] = {}
        for local_name, value in signals.items():
            if not isinstance(local_name, str) or not local_name:
                raise TypeError(
                    f"monitor {monitor.name!r}.signals: names must be non-empty "
                    f"strings, got {local_name!r}"
                )
            if "/" in local_name:
                raise ValueError(
                    f"monitor {monitor.name!r}.signals: name {local_name!r} must not "
                    "contain '/'; the engine owns the '<monitor>/<signal>' namespace"
                )
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"monitor {monitor.name!r}.signals[{local_name!r}]: expected a "
                    f"tensor, got {type(value).__name__}"
                )
            if value.ndim < 1 or value.shape[0] != frame.num_envs:
                raise ValueError(
                    f"monitor {monitor.name!r}.signals[{local_name!r}]: expected leading "
                    f"dimension num_envs={frame.num_envs}, got shape {tuple(value.shape)}"
                )
            if value.device != frame.device:
                raise ValueError(
                    f"monitor {monitor.name!r}.signals[{local_name!r}]: device "
                    f"{value.device} does not match frame device {frame.device}"
                )
            checked[local_name] = value
        return checked

    def _collect_events(self, monitor: Monitor) -> tuple[Event, ...]:
        """Drain and validate one monitor's occurrences."""
        drained = monitor.drain_events()
        if isinstance(drained, (str, bytes, bytearray)) or not isinstance(drained, Sequence):
            raise TypeError(
                f"monitor {monitor.name!r}.drain_events: expected a sequence, "
                f"got {type(drained).__name__}"
            )
        events: list[Event] = []
        for event in drained:
            if not isinstance(event, Event):
                raise TypeError(
                    f"monitor {monitor.name!r}.drain_events: expected Event values, "
                    f"got {type(event).__name__}"
                )
            events.append(event)
        return tuple(events)

    def _collect_terminals(self, monitor: Monitor, num_envs: int) -> dict[int, Termination]:
        """Drain and validate one monitor's terminal requests."""
        drained = monitor.drain_terminals()
        if not isinstance(drained, Mapping):
            raise TypeError(
                f"monitor {monitor.name!r}.drain_terminals: expected a mapping, "
                f"got {type(drained).__name__}"
            )
        terminals: dict[int, Termination] = {}
        for key, value in drained.items():
            env_id = _require_env_id(key, num_envs, f"monitor {monitor.name!r}.terminals")
            if not isinstance(value, Termination):
                raise TypeError(
                    f"monitor {monitor.name!r}.terminals[{env_id}]: expected a "
                    f"Termination, got {type(value).__name__}"
                )
            if value.source is not TerminationSource.CUSTOM:
                raise ValueError(
                    f"monitor {monitor.name!r}.terminals[{env_id}]: source must be "
                    f"custom, got {value.source.value!r}"
                )
            if not value.code.startswith("custom."):
                raise ValueError(
                    f"monitor {monitor.name!r}.terminals[{env_id}]: code must start "
                    f"with 'custom.', got {value.code!r}"
                )
            validate_code(value.code)
            terminals[env_id] = value
        return terminals

    def _validate_event_envs(self, monitor: Monitor, event: Event) -> None:
        """Reject an occurrence that names a closed or unknown environment."""
        num_envs = self._require_bound()
        for env_id in event.env_ids:
            if env_id >= num_envs:
                raise ValueError(
                    f"monitor {monitor.name!r}.events[{event.name!r}]: env id {env_id} "
                    f"is outside [0, num_envs={num_envs})"
                )
            if not self._open[env_id]:
                raise EngineStateError(
                    f"monitor {monitor.name!r}.events[{event.name!r}]: env {env_id} "
                    f"is closed; open envs are {list(self.open_envs)}"
                )

    def _record_event(self, event: Event) -> None:
        """Append one occurrence to the capped journal and episode counters."""
        for env_id in event.env_ids:
            counts = self._episode_events[env_id]
            counts[event.code] = counts.get(event.code, 0) + 1
        self._event_total += 1
        if not self._retention.event_log:
            self._dropped_events += 1
        elif len(self._journal) < self._retention.max_events:
            self._journal.append(event)
        else:
            self._dropped_events += 1

    def _accept_terminal(self, env_id: int, termination: Termination) -> None:
        """Latch the first terminal per env or count a later request."""
        if env_id in self._latched:
            self._conflicts[env_id] = self._conflicts.get(env_id, 0) + 1
            self._conflict_total += 1
            return
        self._latched[env_id] = termination
        self._pending[env_id] = termination

    def _due_verdict(self, bundle: Mapping[str, object], env_id: int) -> EpisodeVerdict:
        """Label a due env without a pack terminal: deadline or core."""
        entry = self._termination_entry(bundle, env_id)
        if entry is not None and entry.primary.outcome is EpisodeOutcome.DEADLINE:
            return "deadline"
        return "core"

    def _finalize_record(self, record: EpisodeRecord) -> None:
        """Accumulate the retention-independent counters and maybe retain."""
        self._episode_total += 1
        self._tick_total += record.ticks
        self._verdict_counts[record.verdict] = self._verdict_counts.get(record.verdict, 0) + 1
        if record.rewound:
            self._rewound_total += 1
        self._episode_counts_by_env[record.env_id] += 1
        for termination in _record_terminations(record):
            outcome = termination.outcome.value
            self._termination_outcome_counts[outcome] = (
                self._termination_outcome_counts.get(outcome, 0) + 1
            )
            self._termination_code_counts[termination.code] = (
                self._termination_code_counts.get(termination.code, 0) + 1
            )
            source = termination.source.value
            self._termination_source_counts[source] = (
                self._termination_source_counts.get(source, 0) + 1
            )
        if self._retention.retain_episodes:
            self._records.append(record)

    def _require_running(self) -> None:
        if self._run_ended:
            raise EngineStateError("engine: run has already ended")

    def _require_bound(self) -> int:
        if not self._bound:
            raise EngineStateError("engine: no environment has been bound by on_step yet")
        if self._num_envs is None:
            raise EngineStateError("engine: internal run shape is not bound")
        return self._num_envs


def _require_monitor_name(name: object) -> str:
    """Validate one monitor name used as a signal namespace."""
    if not isinstance(name, str) or not name:
        raise TypeError(f"engine monitor.name: expected a non-empty str, got {name!r}")
    if "/" in name:
        raise ValueError(
            f"engine monitor.name: {name!r} must not contain '/'; names namespace signals"
        )
    return name


def _record_terminations(record: EpisodeRecord) -> tuple[Termination, ...]:
    """Return every core/pack termination fact retained by one episode record."""
    facts: list[Termination] = []
    if record.first_terminal is not None:
        facts.append(record.first_terminal)
    if record.core_termination is not None:
        facts.extend(record.core_termination.all)
    return tuple(facts)


def _require_env_id(value: object, num_envs: int, where: str) -> int:
    """Return one integer env id from ``value`` or raise a teaching error."""
    if isinstance(value, bool):
        raise TypeError(f"{where}: env ids must be ints, got {value!r}")
    try:
        env_id = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{where}: env ids must be ints, got {value!r}") from exc
    if env_id < 0 or env_id >= num_envs:
        raise ValueError(f"{where}: env id {env_id} is outside [0, num_envs={num_envs})")
    return env_id


def _as_env_ids(value: object, num_envs: int, where: str) -> tuple[int, ...]:
    """Normalize a sequence or integer tensor into sorted unique env ids."""
    if isinstance(value, Tensor):
        if value.ndim != 1:
            raise ValueError(f"{where}: expected a 1-D tensor, got shape {tuple(value.shape)}")
        if value.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{where}: expected an int32/int64 tensor, got dtype {value.dtype}")
        raw_ids: list[object] = value.detach().tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        raw_ids = list(value)
    else:
        raise TypeError(
            f"{where}: expected a sequence of env ids or an integer tensor, "
            f"got {type(value).__name__}"
        )
    ids: list[int] = []
    seen: set[int] = set()
    for raw in raw_ids:
        env_id = _require_env_id(raw, num_envs, where)
        if env_id in seen:
            raise ValueError(f"{where}: duplicate env id {env_id}")
        seen.add(env_id)
        ids.append(env_id)
    return tuple(sorted(ids))


def _retention_state(retention: RetentionSpec) -> dict[str, JsonValue]:
    """Return the construction fingerprint a checkpoint must match."""
    return {
        "retain_episodes": retention.retain_episodes,
        "max_events": retention.max_events,
        "event_log": retention.event_log,
    }


def _decode_retention(raw: JsonValue, where: str) -> dict[str, JsonValue]:
    if not isinstance(raw, dict):
        raise TypeError(f"{where}: expected an object, got {type(raw).__name__}")
    _require_keys(raw, _RETENTION_KEYS, where)
    return {
        "retain_episodes": _require_bool(raw["retain_episodes"], f"{where}.retain_episodes"),
        "max_events": _nonnegative_int(raw["max_events"], f"{where}.max_events"),
        "event_log": _require_bool(raw["event_log"], f"{where}.event_log"),
    }


def _rewind_capability(
    monitor: Monitor,
) -> tuple[Callable[[], object], Callable[[object], None]] | None:
    """Return a monitor's snapshot/restore pair, or ``None`` when stateless.

    A monitor that declares only one half of the optional capability is an
    inconsistent implementation and fails fast instead of silently
    participating in or opting out of rewind.
    """
    snapshot = getattr(monitor, "snapshot", None)
    restore = getattr(monitor, "restore", None)
    has_snapshot = callable(snapshot)
    has_restore = callable(restore)
    if has_snapshot != has_restore:
        raise TypeError(
            f"monitor {monitor.name!r}: rewind capability requires both snapshot " "and restore"
        )
    if not has_snapshot:
        return None
    return snapshot, restore


def _monitor_snapshot(monitor: Monitor) -> JsonValue | None:
    capability = _rewind_capability(monitor)
    if capability is None:
        return None
    snapshot, _ = capability
    return ensure_json_value(snapshot(), where=f"monitor {monitor.name!r} rewind snapshot")


def _stateful_extensions(
    extensions: Sequence[ReportExtension],
) -> dict[str, ReportExtension]:
    """Return the stateful extensions keyed by name.

    ``validate_report_extensions`` already guaranteed that ``stateful`` is an
    explicit bool and that a stateful extension implements snapshot/restore.
    """
    return {extension.name: extension for extension in extensions if extension.stateful}


def _extension_snapshot(extension: ReportExtension) -> JsonValue:
    """Return one stateful extension's JSON-safe, private-copied state."""
    snapshot = getattr(extension, "snapshot", None)
    if not callable(snapshot):
        raise TypeError(
            f"report extension {extension.name!r}: stateful=True requires a callable " "snapshot()"
        )
    return ensure_json_value(
        snapshot(), where=f"report extension {extension.name!r} rewind snapshot"
    )


def _require_keys(
    document: Mapping[str, JsonValue],
    expected: frozenset[str],
    where: str,
    *,
    optional: frozenset[str] = frozenset(),
) -> None:
    allowed = expected | optional
    missing = sorted(expected - document.keys())
    unknown = sorted(document.keys() - allowed)
    if missing or unknown:
        raise ValueError(
            f"{where}: expected keys {sorted(expected)} (optional {sorted(optional)}), "
            f"missing {missing}, unknown {unknown}"
        )


def _require_bool(value: JsonValue, where: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{where}: expected a bool, got {value!r}")
    return value


def _nonnegative_int(value: JsonValue, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where}: expected an int, got {value!r}")
    if value < 0:
        raise ValueError(f"{where}: expected an int >= 0, got {value}")
    return value


def _optional_positive_int(value: JsonValue, where: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where}: expected a positive int or None, got {value!r}")
    if value < 1:
        raise ValueError(f"{where}: expected a positive int, got {value}")
    return value


def _optional_device(value: JsonValue, where: str) -> torch.device | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"{where}: expected a device string or None, got {value!r}")
    try:
        return torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"{where}: invalid device string {value!r}: {exc}") from exc


def _require_list(value: JsonValue, where: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise TypeError(f"{where}: expected a list, got {type(value).__name__}")
    return value


def _require_run_length(values: Sequence[object], num_envs: int | None, where: str) -> None:
    if num_envs is None:
        if values:
            raise ValueError(f"{where}: an unbound checkpoint must not carry per-env state")
        return
    if len(values) != num_envs:
        raise ValueError(f"{where}: expected {num_envs} entries, got {len(values)}")


def _decode_bool_list(value: JsonValue, where: str, num_envs: int | None) -> list[bool]:
    items = _require_list(value, where)
    _require_run_length(items, num_envs, where)
    values: list[bool] = []
    for index, item in enumerate(items):
        values.append(_require_bool(item, f"{where}[{index}]"))
    return values


def _decode_int_list(value: JsonValue, where: str, num_envs: int | None) -> list[int]:
    items = _require_list(value, where)
    _require_run_length(items, num_envs, where)
    return [_nonnegative_int(item, f"{where}[{index}]") for index, item in enumerate(items)]


def _decode_event_counts_list(
    value: JsonValue, where: str, num_envs: int | None
) -> list[dict[str, int]]:
    items = _require_list(value, where)
    _require_run_length(items, num_envs, where)
    counts: list[dict[str, int]] = []
    for index, item in enumerate(items):
        item_where = f"{where}[{index}]"
        if not isinstance(item, dict):
            raise TypeError(f"{item_where}: expected an object, got {type(item).__name__}")
        per_env: dict[str, int] = {}
        for code, count in item.items():
            validate_code(code)
            per_env[code] = _nonnegative_int(count, f"{item_where}[{code!r}]")
        counts.append(per_env)
    return counts


def _decode_command(value: JsonValue, where: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError(f"{where}: expected a command list or None, got {value!r}")
    command: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{where}[{index}]: expected a number, got {item!r}")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{where}[{index}]: expected a finite number, got {number!r}")
        command.append(number)
    return tuple(command)


def _decode_command_list(
    value: JsonValue, where: str, num_envs: int | None
) -> list[tuple[float, ...] | None]:
    items = _require_list(value, where)
    _require_run_length(items, num_envs, where)
    return [_decode_command(item, f"{where}[{index}]") for index, item in enumerate(items)]


def _decode_reset_list(
    value: JsonValue, where: str, num_envs: int | None
) -> list[dict[str, JsonValue]]:
    items = _require_list(value, where)
    _require_run_length(items, num_envs, where)
    facts: list[dict[str, JsonValue]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(f"{where}[{index}]: expected an object, got {type(item).__name__}")
        facts.append(dict(item))
    return facts


def _decode_env_key(key: str, where: str) -> int:
    if not key.isdigit() or str(int(key)) != key:
        raise ValueError(f"{where}: expected a non-negative integer key, got {key!r}")
    return int(key)


def _decode_termination_map(
    value: JsonValue, where: str, num_envs: int | None
) -> dict[int, Termination]:
    if not isinstance(value, dict):
        raise TypeError(f"{where}: expected an object, got {type(value).__name__}")
    terminals: dict[int, Termination] = {}
    for key, item in value.items():
        env_id = _decode_env_key(key, where)
        if num_envs is None or env_id >= num_envs:
            raise ValueError(f"{where}: env id {env_id} is outside [0, num_envs={num_envs})")
        terminals[env_id] = termination_from_json(item, f"{where}[{key!r}]")
    return terminals


def _decode_conflict_map(value: JsonValue, where: str, num_envs: int | None) -> dict[int, int]:
    if not isinstance(value, dict):
        raise TypeError(f"{where}: expected an object, got {type(value).__name__}")
    conflicts: dict[int, int] = {}
    for key, item in value.items():
        env_id = _decode_env_key(key, where)
        if num_envs is None or env_id >= num_envs:
            raise ValueError(f"{where}: env id {env_id} is outside [0, num_envs={num_envs})")
        conflicts[env_id] = _nonnegative_int(item, f"{where}[{key!r}]")
    return conflicts


def _decode_journal(value: JsonValue, where: str, num_envs: int | None) -> list[Event]:
    items = _require_list(value, where)
    if num_envs is None and items:
        raise ValueError(f"{where}: an unbound checkpoint must not carry journal entries")
    journal: list[Event] = []
    for index, item in enumerate(items):
        event = Event.from_json(item)
        if num_envs is not None:
            for env_id in event.env_ids:
                if env_id >= num_envs:
                    raise ValueError(
                        f"{where}[{index}]: env id {env_id} is outside [0, num_envs={num_envs})"
                    )
        journal.append(event)
    return journal


def _decode_records(value: JsonValue, where: str, num_envs: int | None) -> list[EpisodeRecord]:
    items = _require_list(value, where)
    if num_envs is None and items:
        raise ValueError(f"{where}: an unbound checkpoint must not carry episode records")
    records: list[EpisodeRecord] = []
    for index, item in enumerate(items):
        record = EpisodeRecord.from_json(item)
        if num_envs is not None and record.env_id >= num_envs:
            raise ValueError(
                f"{where}[{index}]: env id {record.env_id} is outside " f"[0, num_envs={num_envs})"
            )
        records.append(record)
    return records


def _decode_verdict_counts(value: JsonValue, where: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise TypeError(f"{where}: expected an object, got {type(value).__name__}")
    counts: dict[str, int] = {}
    for verdict, count in value.items():
        if verdict not in VERDICTS:
            raise ValueError(f"{where}: unknown verdict {verdict!r}; allowed {sorted(VERDICTS)}")
        counts[verdict] = _nonnegative_int(count, f"{where}[{verdict!r}]")
    return counts


def _decode_string_count_map(value: JsonValue, where: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise TypeError(f"{where}: expected an object, got {type(value).__name__}")
    counts: dict[str, int] = {}
    for key, count in value.items():
        if not key:
            raise ValueError(f"{where}: keys must be non-empty strings, got {key!r}")
        counts[key] = _nonnegative_int(count, f"{where}[{key!r}]")
    return counts
