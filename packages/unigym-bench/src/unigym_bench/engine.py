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

import operator
from collections.abc import Mapping, Sequence
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

from .events import EpisodeRecord, EpisodeSummary, EpisodeVerdict, Event
from .frames import TickFrame
from .json import JsonValue
from .monitors import Monitor
from .specs import RetentionSpec

if TYPE_CHECKING:
    from unigym.signals import SignalPublisher

__all__ = ["EngineStateError", "StreamingEngine"]


class EngineStateError(RuntimeError):
    """The engine was driven outside its declared per-run lifecycle."""


class StreamingEngine:
    """One run's streaming state, monitor driver and terminal authority.

    Args:
        monitors: Monitor instances in task declaration order.  Names must be
            unique because they namespace published signals and report blocks.
        retention: Explicit event/episode retention policy.  The default is the
            pack's documented ``RetentionSpec`` default.
        publisher: Optional write-only signal publisher.  When supplied, every
            monitor level is published as ````<monitor>/<signal>````.
    """

    def __init__(
        self,
        monitors: Sequence[Monitor],
        *,
        retention: RetentionSpec | None = None,
        publisher: SignalPublisher | None = None,
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
        self._retention = retention
        self._publisher = publisher

        self._bound = False
        self._run_ended = False
        self._num_envs: int | None = None
        self._device: torch.device | None = None

        self._open: list[bool] = []
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
        )

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
            )
            for env_id in due_ids
        ]

        for monitor in self._monitors:
            monitor.on_episode_end(due_ids)
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
                    verdict="incomplete",
                    command_at_start=self._episode_command[env_id],
                    first_terminal=self._latched.get(env_id),
                    reset_facts=dict(self._episode_reset[env_id]),
                    event_counts=dict(self._episode_events[env_id]),
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
            self._pending.clear()
        self._run_ended = True
        return self.summary

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
            self._episode_seq = [0] * frame.num_envs
            self._episode_ticks = [0] * frame.num_envs
            self._episode_events = [{} for _ in range(frame.num_envs)]
            self._episode_command = [None] * frame.num_envs
            self._episode_reset = [{} for _ in range(frame.num_envs)]
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
        """Accumulate and, when retention asks, retain one finalized record."""
        self._episode_total += 1
        self._tick_total += record.ticks
        self._verdict_counts[record.verdict] = self._verdict_counts.get(record.verdict, 0) + 1
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
