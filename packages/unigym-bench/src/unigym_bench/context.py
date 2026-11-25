"""Pack-owned benchmark step context built on the core boundary context.

A :class:`BenchmarkStepContext` is the only data a monitor receives for a tick.
It subclasses the core :class:`~unigym.evaluation.EnvStepContext` and adds the
pack's own defaulted fields: the one-shot :class:`~unigym_bench.events.Event`
records produced by monitors, the per-env :class:`Termination` requests they
raised, and the tick's optional :class:`~unigym_bench.query.TickQuery` read
snapshot.  The core context owns the borrowed view, value snapshot, tick
position and reset bookkeeping; this module owns the pack extras only.

Events are copied into an immutable tuple and terminals into a read-only
mapping, so a context never aliases caller-owned mutable state.  The view is
borrowed for the callback exactly as in the core contract: read it, do not keep
it.  Value publication and final-context recording are later engine work, not
this boundary type.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from unigym.core.episode_termination import Termination
from unigym.evaluation import EnvStepContext

from .events import Event
from .query import TickQuery

__all__ = ["BenchmarkStepContext"]


@dataclass(frozen=True, slots=True)
class BenchmarkStepContext(EnvStepContext):
    """One benchmark step's read-only facts for monitors and probes.

    Attributes:
        events: One-shot occurrences produced for this tick, stored as a tuple.
        terminals: Per-env terminal requests keyed by env id, stored read-only.
        query: Optional one-tick memoized query read snapshot.  Engine ticks
            attach one; direct and offline contexts leave it ``None``.
    """

    events: tuple[Event, ...] = ()
    terminals: Mapping[int, Termination] = field(default_factory=dict)
    query: TickQuery | None = None

    def __post_init__(self) -> None:
        """Validate the core boundary, then copy and validate the pack fields."""
        super(BenchmarkStepContext, self).__post_init__()

        query = self.query
        if query is not None and not isinstance(query, TickQuery):
            raise TypeError(
                "BenchmarkStepContext.query must be a TickQuery or None, "
                f"got {type(query).__name__}"
            )

        raw_events = self.events
        if isinstance(raw_events, (str, bytes, bytearray)) or not isinstance(raw_events, Sequence):
            raise TypeError(
                "BenchmarkStepContext.events must be a sequence of Event, "
                f"got {type(raw_events).__name__}"
            )
        events = tuple(raw_events)
        for index, event in enumerate(events):
            if not isinstance(event, Event):
                raise TypeError(
                    f"BenchmarkStepContext.events[{index}] must be an Event, "
                    f"got {type(event).__name__}"
                )
        object.__setattr__(self, "events", events)

        raw_terminals = self.terminals
        if not isinstance(raw_terminals, Mapping):
            raise TypeError(
                "BenchmarkStepContext.terminals must be a mapping, "
                f"got {type(raw_terminals).__name__}"
            )
        num_envs = self.num_envs
        terminals: dict[int, Termination] = {}
        for env_id, termination in raw_terminals.items():
            if isinstance(env_id, bool) or not isinstance(env_id, int):
                raise TypeError(
                    "BenchmarkStepContext.terminals keys must be int env ids, " f"got {env_id!r}"
                )
            if env_id < 0 or env_id >= num_envs:
                raise ValueError(
                    f"BenchmarkStepContext.terminals key {env_id} is outside "
                    f"[0, num_envs={num_envs})"
                )
            if not isinstance(termination, Termination):
                raise TypeError(
                    f"BenchmarkStepContext.terminals[{env_id}] must be a Termination, "
                    f"got {type(termination).__name__}"
                )
            terminals[env_id] = termination
        object.__setattr__(self, "terminals", MappingProxyType(terminals))

    @classmethod
    def from_env_step(
        cls,
        step: EnvStepContext,
        *,
        events: Sequence[Event] = (),
        terminals: Mapping[int, Termination] | None = None,
        query: TickQuery | None = None,
    ) -> BenchmarkStepContext:
        """Wrap an already-built core context without re-reading a bundle.

        Args:
            step: The core context produced by the runner boundary.
            events: Optional one-shot occurrences to attach.
            terminals: Optional per-env terminal requests to attach.
            query: Optional tick snapshot to attach.  A wrapped context never
                inherits the source context's snapshot, so a boundary cannot
                accidentally reuse a previous tick's cache.

        Returns:
            A new benchmark context that shares only the borrowed view and
            value snapshot with ``step``.

        Raises:
            TypeError: ``step`` is not an :class:`EnvStepContext`.
        """
        if not isinstance(step, EnvStepContext):
            raise TypeError(
                "BenchmarkStepContext.from_env_step: expected an EnvStepContext, "
                f"got {type(step).__name__}"
            )
        return cls(
            view=step.view,
            values=step.values,
            t=step.t,
            step_index=step.step_index,
            time_s=step.time_s,
            cmd=step.cmd,
            reset_state=step.reset_state,
            events=events,
            terminals={} if terminals is None else terminals,
            query=query,
        )
