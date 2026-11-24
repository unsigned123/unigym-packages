"""The pack's core Sink shell and its narrow construction seam.

A :class:`BenchmarkSink` is the pack's runner-facing object.  It owns no
benchmark state itself: every callback delegates to the run's
:class:`~unigym_bench.engine.StreamingEngine`, which owns open flags, terminal
latches, episode records and event retention.  The sink adds only the runner
lifecycle glue and the ``on_run_end`` summary document.

Construction seam (P4 boundary)
-------------------------------

``make_sink(ctx)`` reads JSON-safe keys from ``ctx.options``:

- ``loop_mode``: ``"pool"`` (default) or ``"sink-steered"``.  A sink-steered
  run requires ``ctx.signals``; absent publisher is a construction error.
- ``monitor_factory``: optional ``"pkg.mod:callable"`` path.  The zero-argument
  callable returns the monitor sequence in task declaration order.

There is deliberately no YAML ``TaskDocument -> monitors`` composition here
yet: that is a later composition layer.  Tests and other programmatic callers
construct :class:`BenchmarkSink` directly with monitor instances or an existing
engine.  Monitor instances are not accepted through ``ctx.options`` because
they are not JSON-safe.

Lifecycle glue
--------------

The runner calls ``on_episode_end`` after the terminal drain and computes the
reset mask itself, but ``on_step_end`` carries no reset ids.  The sink therefore
stores the due ids returned by ``engine.on_episode_end`` and calls
``engine.on_reset`` from ``on_step_end``, i.e. exactly after the runner's reset
chain.  Because the sink cannot observe whether that chain ran, construction
rejects ``runner.auto_reset: false``: the core runner already rejects sink
terminals in that mode, and reopening a row the backend never reset would be
incorrect.  The direct constructor exposes the same assertion as
``reset_after_episode``.

Rewind and early-stop hooks.  ``stop_when`` never stops.  The sink exposes the
run's :class:`~unigym_bench.engine.StreamingEngine` as its core rewind provider
(``name="unigym_bench"``, schema ``unigym_bench.rewind/1``); the engine owns all
checkpointable pack state and the sink's own ``_pending_reset`` glue is empty at
every core snapshot freeze point (a snapshot is taken between completed ticks).
The sampler cursor/RNG and the ``SignalBus`` remain separate core providers.
``on_rewind`` delegates to the engine so open episodes are marked ``rewound``.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, TypeAlias, cast

from unigym.errors import ConfigError

from .engine import StreamingEngine
from .monitors import Monitor
from .specs import RetentionSpec

if TYPE_CHECKING:
    from unigym.core.episode_termination import Termination
    from unigym.run import SinkContext
    from unigym.signals import SignalPublisher

__all__ = [
    "BenchmarkSink",
    "DEFAULT_LOOP_MODE",
    "LOOP_MODES",
    "SINK_STEERED",
    "SUMMARY_KEY",
    "make_sink",
]

#: Default closed-loop mode: monitor signals are published for other consumers.
DEFAULT_LOOP_MODE = "pool"
#: Mode in which the pack sink owns the command loop through its publisher.
SINK_STEERED = "sink-steered"
#: Every closed-loop mode accepted by this construction seam.
LOOP_MODES: frozenset[str] = frozenset({DEFAULT_LOOP_MODE, SINK_STEERED})

#: Unique summary key so a fan-out sink cannot silently collide with another.
SUMMARY_KEY = "BenchmarkSink"

#: ``module:attribute`` where both sides are dotted Python identifiers.
_MONITOR_FACTORY_PATH = re.compile(
    r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
)

_OPTION_KEYS = frozenset({"loop_mode", "monitor_factory"})

_MonitorFactory: TypeAlias = Callable[[], Sequence[Monitor]]


def make_sink(ctx: SinkContext) -> BenchmarkSink:
    """Build the pack sink from a run's construction context.

    Args:
        ctx: The core construction context.  Only ``options`` and the
            publisher-only ``signals`` face are read; no live env, writer or
            logger is available here by design.

    Returns:
        A :class:`BenchmarkSink` wired to the run's signal publisher.

    Raises:
        ConfigError: An option is unknown, malformed, a factory cannot be
            resolved, or a sink-steered run has no publisher.
    """
    options = ctx.options
    if not isinstance(options, Mapping):
        raise ConfigError(
            "unigym-bench sink options: expected a mapping, " f"got {type(options).__name__}"
        )
    unknown = sorted(set(options) - _OPTION_KEYS)
    if unknown:
        raise ConfigError(
            f"unigym-bench sink options: unknown key(s) {unknown}; "
            f"allowed keys are {sorted(_OPTION_KEYS)}"
        )
    loop_mode = options.get("loop_mode", DEFAULT_LOOP_MODE)
    _validate_loop_mode(loop_mode)
    if loop_mode == SINK_STEERED and ctx.signals is None:
        raise ConfigError(
            "unigym-bench sink: loop_mode 'sink-steered' requires ctx.signals, "
            "but the build supplied no signal publisher"
        )
    auto_reset = _runner_auto_reset(ctx.spec)
    if not auto_reset:
        raise ConfigError(
            "unigym-bench sink: requires runner.auto_reset: true so the backend "
            "reset chain runs after on_episode_end; got runner.auto_reset: false. "
            "The pack sink cannot detect whether a backend reset actually ran."
        )
    raw_factory = options.get("monitor_factory")
    monitors: Sequence[Monitor] = ()
    if raw_factory is not None:
        monitors = _resolve_monitor_factory(raw_factory)()
    return BenchmarkSink(
        monitors,
        publisher=ctx.signals,
        loop_mode=loop_mode,
        reset_after_episode=auto_reset,
    )


class BenchmarkSink:
    """The pack's four-callback Sink plus the one-shot terminal drain.

    The sink is a thin lifecycle shell over a :class:`StreamingEngine`.  It
    never owns episode, latency, deadline or report state of its own.

    Args:
        monitors: Monitor instances in task declaration order.  Mutually
            exclusive with ``engine``.
        engine: An already-built engine for tests or advanced composition.
            When supplied, ``retention`` and ``publisher`` must be omitted.
        retention: Explicit event/episode retention for a sink-built engine.
        publisher: Optional write-only signal publisher for a sink-built engine.
        loop_mode: ``"pool"`` or ``"sink-steered"``.  A sink-steered sink needs
            a publisher on its engine; absent publisher is a construction error.
        reset_after_episode: Explicit assertion that the runner actually resets
            due envs.  The sink cannot observe the backend reset chain, so a
            false value is rejected instead of silently reopening envs.

    Raises:
        ConfigError: ``loop_mode`` is unknown, a sink-steered sink has no
            publisher, or ``reset_after_episode`` is false.
        TypeError: Both or neither of ``monitors``/``engine`` are supplied
            together with conflicting engine resources.
    """

    def __init__(
        self,
        monitors: Sequence[Monitor] | None = None,
        *,
        engine: StreamingEngine | None = None,
        retention: RetentionSpec | None = None,
        publisher: SignalPublisher | None = None,
        loop_mode: str = DEFAULT_LOOP_MODE,
        reset_after_episode: bool = True,
    ) -> None:
        _validate_loop_mode(loop_mode)
        if not isinstance(reset_after_episode, bool):
            raise TypeError(
                "BenchmarkSink: reset_after_episode must be a bool, "
                f"got {type(reset_after_episode).__name__}"
            )
        if not reset_after_episode:
            raise ConfigError(
                "BenchmarkSink: requires reset_after_episode (runner.auto_reset: true) "
                "so the sink only reopens envs the backend actually reset; got false"
            )
        if engine is not None:
            if monitors is not None:
                raise TypeError("BenchmarkSink: provide either monitors or engine, not both")
            if retention is not None or publisher is not None:
                raise TypeError(
                    "BenchmarkSink: an injected engine owns retention and publisher; "
                    "do not pass them separately"
                )
            self._engine = engine
        else:
            self._engine = StreamingEngine(
                () if monitors is None else monitors,
                retention=retention,
                publisher=publisher,
            )
        if loop_mode == SINK_STEERED and self._engine.publisher is None:
            raise ConfigError(
                "BenchmarkSink: loop_mode 'sink-steered' requires a signal publisher "
                "on the engine"
            )
        self._loop_mode = loop_mode
        self._pending_reset: tuple[int, ...] = ()

    @property
    def loop_mode(self) -> str:
        """The closed-loop mode this sink was built with."""
        return self._loop_mode

    @property
    def rewind_provider(self) -> StreamingEngine:
        """The run's core rewind provider, owned by the streaming engine."""
        return self._engine

    def on_step(self, t: int, bundle: dict[str, object]) -> None:
        """Delegate one pre-reset tick to the engine."""
        self._engine.on_step(bundle)

    def drain_terminations(self) -> Mapping[int, Termination]:
        """Return and clear the engine's pending first-terminal facts."""
        return self._engine.drain_terminations()

    def on_episode_end(self, env_ids: object, bundle: dict[str, object]) -> None:
        """Finalize due episodes and remember the ids for the reset reopen."""
        self._pending_reset = self._engine.on_episode_end(env_ids, bundle)

    def on_step_end(self, t: int, bundle: dict[str, object]) -> None:
        """Reopen exactly the envs the runner has just reset."""
        pending = self._pending_reset
        self._pending_reset = ()
        if pending:
            self._engine.on_reset(pending)

    def on_run_end(self) -> dict[str, object]:
        """Return the pack summary under its unique class-name key."""
        return {SUMMARY_KEY: self._engine.on_run_end().to_json()}

    def stop_when(self) -> bool:
        """P4 never requests an early stop; P5+ owns any stop policy."""
        return False

    def on_rewind(self, info: dict[str, object]) -> None:
        """Mark the currently open episodes as belonging to a forked timeline."""
        self._engine.on_rewind(info)


def _validate_loop_mode(value: object) -> None:
    """Fail fast on an unknown closed-loop mode."""
    if not isinstance(value, str) or value not in LOOP_MODES:
        raise ConfigError(
            f"unigym-bench sink loop_mode: expected one of {sorted(LOOP_MODES)}, " f"got {value!r}"
        )


def _runner_auto_reset(spec: object) -> bool:
    """Read the runner's reset policy from construction metadata.

    An absent or non-mapping ``runner`` block is the core default (auto reset).
    The value is coerced with ``bool`` exactly as ``SyncRunner`` does, so an
    explicit false-like value is rejected while an absent key stays true.
    """
    runner = getattr(spec, "runner", None)
    if not isinstance(runner, Mapping):
        return True
    return bool(runner.get("auto_reset", True))


def _resolve_monitor_factory(path: object) -> _MonitorFactory:
    """Import one ``module:callable`` monitor factory without calling it."""
    if not isinstance(path, str) or _MONITOR_FACTORY_PATH.fullmatch(path) is None:
        raise ConfigError(
            "unigym-bench sink monitor_factory: expected a 'module:callable' path "
            f"of dotted identifiers, got {path!r}"
        )
    module_name, _, attribute_path = path.partition(":")
    try:
        target: object = importlib.import_module(module_name)
    except Exception as exc:
        raise ConfigError(
            f"unigym-bench sink monitor_factory={path!r} could not be imported: {exc} "
            "| hint: the factory path is 'module:callable' and the module must be "
            "importable from the run process"
        ) from exc
    for attribute in attribute_path.split("."):
        try:
            target = getattr(target, attribute)
        except AttributeError as exc:
            raise ConfigError(
                f"unigym-bench sink monitor_factory={path!r} could not be imported: "
                f"module {module_name!r} has no attribute {attribute!r}"
            ) from exc
    if not callable(target):
        raise ConfigError(f"unigym-bench sink monitor_factory={path!r} is not callable")
    # The callable check above establishes the only structural guarantee; the
    # factory's return shape is validated by the engine when it is called.
    return cast(_MonitorFactory, target)
