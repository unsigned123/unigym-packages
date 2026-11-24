"""The pack's core Sink shell and its narrow construction seam.

A :class:`BenchmarkSink` is the pack's runner-facing object.  It owns no
benchmark state itself: every callback delegates to the run's
:class:`~unigym_bench.engine.StreamingEngine`, which owns open flags, terminal
latches, episode records and event retention.  The sink adds only the runner
lifecycle glue and the ``on_run_end`` summary document.

Construction seam (P4/P6 boundary)
----------------------------------

``make_sink(ctx)`` reads JSON-safe keys from ``ctx.options``:

- ``loop_mode``: ``"pool"`` (default) or ``"sink-steered"``.  A sink-steered
  run requires ``ctx.signals``; absent publisher is a construction error.
- ``monitor_factory``: optional ``"pkg.mod:callable"`` path.  The zero-argument
  callable returns the monitor sequence in task declaration order.
- ``extension_factory``: optional ``"pkg.mod:callable"`` path returning the
  report-extension sequence in task declaration order.
- ``monitor_extensions``: optional JSON object mapping a monitor name to the
  extension name it declares through ``report_extension``.  A declared monitor
  is omitted from the report's ``monitors`` fallback block.
- ``write_report``: optional JSON boolean (default ``false``).  When true the
  sink writes its report to :data:`unigym_bench.report.REPORT_FILENAME` under
  the construction context's ``out_dir`` at run end.  Writing is opt-in because
  it is a filesystem side effect; the fixed pack-owned filename never replaces
  the core ``benchmark_report.json`` or ``results.json``.

There is deliberately no YAML ``TaskDocument -> monitors/extensions``
composition here yet: that is a later composition layer.  Tests and other
programmatic callers construct :class:`BenchmarkSink` directly with monitor and
extension instances or an existing engine.  Instances are not accepted through
``ctx.options`` because they are not JSON-safe.

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

Report.  ``on_run_end`` finalizes the engine, drives every report extension's
``on_run_end`` once and returns the pack report under :data:`SUMMARY_KEY`.  It
never writes the core ``benchmark_report.json`` or any other core artifact.

Rewind and early-stop hooks.  ``stop_when`` never stops.  The sink exposes the
run's :class:`~unigym_bench.engine.StreamingEngine` as its core rewind provider
(``name="unigym_bench"``, schema ``unigym_bench.rewind/2``); the engine owns all
checkpointable pack state, including stateful report extensions, and the sink's
own ``_pending_reset`` glue is empty at every core snapshot freeze point (a
snapshot is taken between completed ticks).  The sampler cursor/RNG and the
``SignalBus`` remain separate core providers.  ``on_rewind`` delegates to the
engine so open episodes are marked ``rewound``.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, cast

from unigym.errors import ConfigError

from .engine import StreamingEngine
from .json import JsonValue
from .monitors import Monitor
from .report import (
    REPORT_FILENAME,
    ReportBuilder,
    ReportExtension,
    RunMetadata,
    RunSummary,
)
from .specs import RetentionSpec
from .version import __version__

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

_OPTION_KEYS = frozenset(
    {
        "loop_mode",
        "monitor_factory",
        "extension_factory",
        "monitor_extensions",
        "write_report",
    }
)

_MonitorFactory: TypeAlias = Callable[[], Sequence[Monitor]]
_ExtensionFactory: TypeAlias = Callable[[], Sequence[ReportExtension]]


def make_sink(ctx: SinkContext) -> BenchmarkSink:
    """Build the pack sink from a run's construction context.

    Args:
        ctx: The core construction context.  Only ``options`` and the
            publisher-only ``signals`` face are read; no live env, writer or
            logger is available here by design.

    Returns:
        A :class:`BenchmarkSink` wired to the run's signal publisher and report
        metadata.

    Raises:
        ConfigError: An option is unknown, malformed, a factory cannot be
            resolved, a declared reference is missing, or a sink-steered run has
            no publisher.
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
    raw_monitor_factory = options.get("monitor_factory")
    monitors: Sequence[Monitor] = ()
    if raw_monitor_factory is not None:
        monitors = _resolve_monitor_factory(raw_monitor_factory)()
    raw_extension_factory = options.get("extension_factory")
    extensions: Sequence[ReportExtension] = ()
    if raw_extension_factory is not None:
        extensions = _resolve_extension_factory(raw_extension_factory)()
    monitor_extensions: Mapping[str, str] | None = None
    raw_monitor_extensions = options.get("monitor_extensions")
    if raw_monitor_extensions is not None:
        monitor_extensions = _validate_monitor_extensions_option(
            raw_monitor_extensions,
            monitor_names=tuple(monitor.name for monitor in monitors),
            extension_names=tuple(extension.name for extension in extensions),
        )
    raw_write_report = options.get("write_report", False)
    if not isinstance(raw_write_report, bool):
        raise ConfigError(
            "unigym-bench sink write_report: expected a bool, "
            f"got {type(raw_write_report).__name__}"
        )
    metadata = RunMetadata(
        run_identity=ctx.run_identity,
        run_name=ctx.spec.run_name,
        task_name=ctx.spec.run_name,
        mode=ctx.mode,
        out_dir=ctx.out_dir,
        seed=ctx.seed,
        num_envs=ctx.num_envs,
        policy_dt=ctx.policy_dt,
        loop_mode=loop_mode,
        pack_version=__version__,
    )
    return BenchmarkSink(
        monitors,
        publisher=ctx.signals,
        loop_mode=loop_mode,
        reset_after_episode=auto_reset,
        extensions=extensions,
        metadata=metadata,
        monitor_extensions=monitor_extensions,
        write_report=raw_write_report,
    )


class BenchmarkSink:
    """The pack's four-callback Sink plus the one-shot terminal drain.

    The sink is a thin lifecycle shell over a :class:`StreamingEngine`.  It
    never owns episode, latency, deadline or monitor/extension state of its own.
    At run end it assembles and returns the pack report; it writes no core
    artifact.

    Args:
        monitors: Monitor instances in task declaration order.  Mutually
            exclusive with ``engine``.
        engine: An already-built engine for tests or advanced composition.
            When supplied, ``retention``, ``publisher`` and ``extensions`` must
            be omitted because the engine already owns them.
        retention: Explicit event/episode retention for a sink-built engine.
        publisher: Optional write-only signal publisher for a sink-built engine.
        loop_mode: ``"pool"`` or ``"sink-steered"``.  A sink-steered sink needs
            a publisher on its engine; absent publisher is a construction error.
        reset_after_episode: Explicit assertion that the runner actually resets
            due envs.  The sink cannot observe the backend reset chain, so a
            false value is rejected instead of silently reopening envs.
        extensions: Report extensions in task declaration order for a
            sink-built engine.
        metadata: Optional construction metadata for the report's ``run`` block.
            When omitted, a minimal honest metadata document is derived at run
            end (blank identity, engine-bound env count, selected loop mode).
        monitor_extensions: Optional ``monitor name -> extension name`` map from
            each monitor's declared ``report_extension``.  A declared monitor is
            omitted from the report's ``monitors`` fallback block.
        write_report: When true, write the assembled report to
            :data:`unigym_bench.report.REPORT_FILENAME` under the metadata's
            ``out_dir`` at run end.  Requires construction metadata with a
            non-empty ``out_dir``.  Never writes a core artifact.

    Raises:
        ConfigError: ``loop_mode`` is unknown, a sink-steered sink has no
            publisher, ``reset_after_episode`` is false, or ``write_report`` is
            true without a construction ``out_dir``.
        TypeError: Both or neither of ``monitors``/``engine`` are supplied, or
            an injected engine is mixed with separately supplied resources.
        ValueError: A monitor/extension name is duplicated or a declared
            ``report_extension`` reference is missing.
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
        extensions: Sequence[ReportExtension] | None = None,
        metadata: RunMetadata | None = None,
        monitor_extensions: Mapping[str, str] | None = None,
        write_report: bool = False,
    ) -> None:
        _validate_loop_mode(loop_mode)
        if not isinstance(write_report, bool):
            raise TypeError(
                "BenchmarkSink: write_report must be a bool, " f"got {type(write_report).__name__}"
            )
        if write_report and (metadata is None or not metadata.out_dir):
            raise ConfigError(
                "BenchmarkSink: write_report requires construction metadata with a "
                "non-empty out_dir; build the sink through make_sink(ctx) or pass "
                "RunMetadata(out_dir=...)"
            )
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
        if metadata is not None and not isinstance(metadata, RunMetadata):
            raise TypeError(
                "BenchmarkSink: metadata must be a RunMetadata or None, "
                f"got {type(metadata).__name__}"
            )
        if engine is not None:
            if monitors is not None:
                raise TypeError("BenchmarkSink: provide either monitors or engine, not both")
            if retention is not None or publisher is not None or extensions is not None:
                raise TypeError(
                    "BenchmarkSink: an injected engine owns retention and publisher "
                    "(and extensions); do not pass them separately"
                )
            self._engine = engine
        else:
            self._engine = StreamingEngine(
                () if monitors is None else monitors,
                retention=retention,
                publisher=publisher,
                extensions=() if extensions is None else extensions,
            )
        if loop_mode == SINK_STEERED and self._engine.publisher is None:
            raise ConfigError(
                "BenchmarkSink: loop_mode 'sink-steered' requires a signal publisher "
                "on the engine"
            )
        self._loop_mode = loop_mode
        self._metadata = metadata
        self._write_report = write_report
        self._pending_reset: tuple[int, ...] = ()
        self._builder = ReportBuilder(
            monitors=self._engine.monitors,
            extensions=self._engine.extensions,
            monitor_extensions=monitor_extensions,
        )
        self._report: dict[str, JsonValue] | None = None
        self._report_written = False

    @property
    def loop_mode(self) -> str:
        """The closed-loop mode this sink was built with."""
        return self._loop_mode

    @property
    def rewind_provider(self) -> StreamingEngine:
        """The run's core rewind provider, owned by the streaming engine."""
        return self._engine

    @property
    def report_path(self) -> Path | None:
        """The pack-owned report path when persistence is enabled, else ``None``."""
        if not self._write_report:
            return None
        if self._metadata is None:
            return None
        return Path(self._metadata.out_dir) / REPORT_FILENAME

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
        """Finalize the run, assemble the pack report and return it.

        The report is assembled once and cached, so a second teardown call
        cannot drive extension ``on_run_end`` twice.  With ``write_report`` the
        cached report is serialized to :attr:`report_path`; a failed write is
        raised and retried on the next call instead of leaving a cached
        "success".  No core artifact is written.
        """
        if self._report is None:
            aggregate = self._engine.on_run_end()
            metadata = self._metadata
            if metadata is None:
                metadata = RunMetadata(
                    num_envs=self._engine.num_envs or 0,
                    loop_mode=self._loop_mode,
                    pack_version=__version__,
                )
            summary = RunSummary(
                metadata=metadata,
                episodes=self._engine.records,
                episode_summary=aggregate,
                monitor_views={
                    monitor.name: monitor.report_view() for monitor in self._engine.monitors
                },
                retention=self._engine.retention,
            )
            blocks = {
                extension.name: extension.on_run_end(summary)
                for extension in self._engine.extensions
            }
            self._report = self._builder.assemble(summary, blocks)
        path = self.report_path
        if path is not None and not self._report_written:
            _write_report_file(path, self._report)
            self._report_written = True
        return {SUMMARY_KEY: self._report}

    def stop_when(self) -> bool:
        """P4 never requests an early stop; P5+ owns any stop policy."""
        return False

    def on_rewind(self, info: dict[str, object]) -> None:
        """Mark the currently open episodes as belonging to a forked timeline."""
        self._engine.on_rewind(info)


def _write_report_file(path: Path, report: dict[str, JsonValue]) -> None:
    """Write the pack report atomically, creating the output directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


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
    return cast(_MonitorFactory, _resolve_factory(path, "monitor_factory"))


def _resolve_extension_factory(path: object) -> _ExtensionFactory:
    """Import one ``module:callable`` report-extension factory without calling it."""
    return cast(_ExtensionFactory, _resolve_factory(path, "extension_factory"))


def _resolve_factory(path: object, option: str) -> Callable[[], object]:
    """Import one ``module:callable`` factory path without calling it."""
    if not isinstance(path, str) or _MONITOR_FACTORY_PATH.fullmatch(path) is None:
        raise ConfigError(
            f"unigym-bench sink {option}: expected a 'module:callable' path "
            f"of dotted identifiers, got {path!r}"
        )
    module_name, _, attribute_path = path.partition(":")
    try:
        target: object = importlib.import_module(module_name)
    except Exception as exc:
        raise ConfigError(
            f"unigym-bench sink {option}={path!r} could not be imported: {exc} "
            "| hint: the factory path is 'module:callable' and the module must be "
            "importable from the run process"
        ) from exc
    for attribute in attribute_path.split("."):
        try:
            target = getattr(target, attribute)
        except AttributeError as exc:
            raise ConfigError(
                f"unigym-bench sink {option}={path!r} could not be imported: "
                f"module {module_name!r} has no attribute {attribute!r}"
            ) from exc
    if not callable(target):
        raise ConfigError(f"unigym-bench sink {option}={path!r} is not callable")
    # The callable check above establishes the only structural guarantee; the
    # factory's return shape is validated when it is called.
    return cast(Callable[[], object], target)


def _validate_monitor_extensions_option(
    value: object,
    *,
    monitor_names: tuple[str, ...],
    extension_names: tuple[str, ...],
) -> Mapping[str, str] | None:
    """Validate the JSON ``monitor_extensions`` option against declared names."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ConfigError(
            "unigym-bench sink monitor_extensions: expected a JSON object mapping "
            f"monitor name to extension name, got {type(value).__name__}"
        )
    known_monitors = set(monitor_names)
    known_extensions = set(extension_names)
    validated: dict[str, str] = {}
    for monitor_name, extension_name in value.items():
        if not isinstance(monitor_name, str) or monitor_name not in known_monitors:
            raise ConfigError(
                f"unigym-bench sink monitor_extensions: unknown monitor {monitor_name!r}; "
                f"declared monitors are {sorted(known_monitors)}"
            )
        if not isinstance(extension_name, str) or extension_name not in known_extensions:
            raise ConfigError(
                f"unigym-bench sink monitor_extensions[{monitor_name!r}]: unknown "
                f"extension {extension_name!r}; declared extensions are "
                f"{sorted(known_extensions)}"
            )
        validated[monitor_name] = extension_name
    return dict(sorted(validated.items()))
