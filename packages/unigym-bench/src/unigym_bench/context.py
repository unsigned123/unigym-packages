"""Pack-owned benchmark step context built on the core boundary context.

A :class:`BenchmarkStepContext` is the only data a monitor receives for a tick.
It subclasses the core :class:`~unigym.evaluation.EnvStepContext` and adds the
tick's optional :class:`~unigym_bench.query.TickQuery` read snapshot.  The core
context owns the borrowed view, value snapshot, tick position and reset
bookkeeping; this module owns the pack extra only.

The context is an input carrier, never an output carrier: monitors publish
values through their protocol and the flow publishes events and terminals
through a decision.  The view is borrowed for the callback exactly as in the
core contract: read it, do not keep it.
"""

from __future__ import annotations

from dataclasses import dataclass

from unigym.evaluation import EnvStepContext

from .query import TickQuery

__all__ = ["BenchmarkStepContext"]


@dataclass(frozen=True, slots=True)
class BenchmarkStepContext(EnvStepContext):
    """One benchmark step's read-only facts for monitors and probes.

    Attributes:
        query: Optional one-tick memoized query read snapshot.  Engine ticks
            attach one; direct and offline contexts leave it ``None``.
    """

    query: TickQuery | None = None

    def __post_init__(self) -> None:
        """Validate the core boundary, then validate the pack field."""
        super(BenchmarkStepContext, self).__post_init__()

        query = self.query
        if query is not None and not isinstance(query, TickQuery):
            raise TypeError(
                "BenchmarkStepContext.query must be a TickQuery or None, "
                f"got {type(query).__name__}"
            )

    @classmethod
    def from_env_step(
        cls,
        step: EnvStepContext,
        *,
        query: TickQuery | None = None,
    ) -> BenchmarkStepContext:
        """Wrap an already-built core context without re-reading a bundle.

        Args:
            step: The core context produced by the runner boundary.
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
            query=query,
        )
