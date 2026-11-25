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
  run requires ``ctx.values``; an absent publisher is a construction error.
- ``task_document``: optional non-empty path string to a YAML task document.
  The document is loaded and composed with :func:`unigym_bench.compose.compose_task`;
  its monitors, extensions, retention and scenario-derived coverage replace the
  factory injection path.  It conflicts with ``monitor_factory``,
  ``extension_factory`` and a different ``monitor_extensions`` map.
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

A YAML ``TaskDocument`` is composed into monitors/extensions only through the
explicit ``task_document`` path option; :func:`unigym_bench.compose.compose_task`
owns that boundary.  Tests and other programmatic callers may still construct
:class:`BenchmarkSink` directly with monitor and extension instances or an
existing engine, and may use ``monitor_factory``/``extension_factory``.  Live
instances are not accepted through ``ctx.options`` because they are not
JSON-safe.

Lifecycle glue
--------------

The runner owns the episode-end boundary and calls ``on_episode_end`` when an
explicit boundary fires.  The sink finalizes the due episodes there.  When the
runner then runs a real reset chain, it calls the sink's
:class:`~unigym.sink.ResetAwareSink` ``on_reset`` capability (tick auto-reset,
``runner.reset`` and ``terminate(..., reset=True)``).  A reset of a finalized
env reopens it; a core reset of an already-open env restarts it in place and
fabricates no episode record; a pre-start reset-all leaves the engine unbound
but still notifies the flow so it can clear pre-run state.  The sink never
infers a reset from ``on_step_end`` and never owns an episode horizon, so a
pack run must end episodes with ``runner.terminate(...)`` or a pack sink
terminal rather than a core clock.

Report.  ``on_run_end`` finalizes the engine, lets the flow drive report
extensions and returns the pack report under :data:`SUMMARY_KEY`.  The sink
validates the exact schema and top-level block set, caches the report once and
never writes the core ``benchmark_report.json`` or any other core artifact.

Rewind and early-stop hooks.  ``stop_when`` returns the engine's run-scoped
stop flag.  The sink exposes the run's
:class:`~unigym_bench.engine.StreamingEngine` as its single core rewind provider
(``name="unigym_bench"``, schema ``unigym_bench.rewind/5``); the engine owns all
checkpointable pack state, including stateful report extensions and the policy
state of a stateful flow bound by this sink.  A stateless default flow adds no
checkpoint key.  A snapshot is taken between completed ticks, so there is no
provisional reset glue to save.  The sampler cursor/RNG and the core ``values``
store are separate core providers.  ``on_rewind`` delegates to the engine so
open episodes are marked ``rewound``.
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

from .compose import ComposeError, TaskPlan, compose_task
from .engine import StreamingEngine
from .flow import (
    BenchmarkFlow,
    DefaultBenchmarkFlow,
    FlowDecision,
    FlowRun,
    validate_flow,
)
from .json import JsonValue, ensure_json_value
from .monitors import Monitor
from .report import (
    REPORT_BLOCKS,
    REPORT_FILENAME,
    REPORT_SCHEMA,
    ReportExtension,
    RunMetadata,
    validate_report_configuration,
)
from .specs import RetentionSpec, SpecError, load_task_document
from .sources import SourceUnavailableError
from .version import __version__

if TYPE_CHECKING:
    from unigym.core.episode_termination import Termination
    from unigym.run import SinkContext
    from unigym.values import ValuePublisher

__all__ = [
    "BenchmarkSink",
    "DEFAULT_LOOP_MODE",
    "LOOP_MODES",
    "SINK_STEERED",
    "SUMMARY_KEY",
    "make_sink",
]

#: Default closed-loop mode: monitor values are published for other consumers.
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
        "task_document",
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
            publisher-only ``values`` face are read; no live env, writer or
            logger is available here by design.

    Returns:
        A :class:`BenchmarkSink` wired to the run's value publisher and report
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
    if loop_mode == SINK_STEERED and ctx.values is None:
        raise ConfigError(
            "unigym-bench sink: loop_mode 'sink-steered' requires ctx.values, "
            "but the build supplied no value publisher"
        )
    raw_task_document = options.get("task_document")
    raw_monitor_factory = options.get("monitor_factory")
    raw_extension_factory = options.get("extension_factory")
    plan: TaskPlan | None = None
    retention: RetentionSpec | None = None
    if raw_task_document is not None:
        if raw_monitor_factory is not None or raw_extension_factory is not None:
            raise ConfigError(
                "unigym-bench sink task_document conflicts with monitor_factory/"
                "extension_factory; a task document already declares its monitors "
                "and extensions. Use one composition path."
            )
        if not isinstance(raw_task_document, str) or not raw_task_document.strip():
            raise ConfigError(
                "unigym-bench sink task_document: expected a non-empty path string, "
                f"got {raw_task_document!r}"
            )
        try:
            document = load_task_document(raw_task_document)
            plan = compose_task(document)
        except (SpecError, ComposeError, SourceUnavailableError, OSError) as exc:
            raise ConfigError(
                f"unigym-bench sink task_document={raw_task_document!r}: {exc}"
            ) from exc
        monitors: Sequence[Monitor] = plan.monitors
        extensions: Sequence[ReportExtension] = plan.extensions
        retention = plan.retention
        plan_flow: BenchmarkFlow | None = plan.flow
    else:
        monitors = ()
        if raw_monitor_factory is not None:
            monitors = _resolve_monitor_factory(raw_monitor_factory)()
        extensions = ()
        if raw_extension_factory is not None:
            extensions = _resolve_extension_factory(raw_extension_factory)()
        plan_flow = None
    monitor_extensions: Mapping[str, str] | None = None
    raw_monitor_extensions = options.get("monitor_extensions")
    if raw_monitor_extensions is not None:
        monitor_extensions = _validate_monitor_extensions_option(
            raw_monitor_extensions,
            monitor_names=tuple(monitor.name for monitor in monitors),
            extension_names=tuple(extension.name for extension in extensions),
        )
        if plan is not None and dict(monitor_extensions) != dict(plan.monitor_extensions):
            raise ConfigError(
                "unigym-bench sink monitor_extensions conflicts with the task "
                f"document's report_extension declarations: option "
                f"{dict(sorted(monitor_extensions.items()))!r}, document "
                f"{dict(plan.monitor_extensions)!r}"
            )
    elif plan is not None:
        monitor_extensions = plan.monitor_extensions
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
        out_dir=ctx.out_dir,
        seed=ctx.seed,
        num_envs=ctx.num_envs,
        policy_dt=ctx.policy_dt,
        loop_mode=loop_mode,
        pack_version=__version__,
    )
    return BenchmarkSink(
        monitors,
        retention=retention,
        publisher=ctx.values,
        loop_mode=loop_mode,
        extensions=extensions,
        metadata=metadata,
        monitor_extensions=monitor_extensions,
        write_report=raw_write_report,
        flow=plan_flow,
    )


class BenchmarkSink:
    """The pack's runner Sink plus the one-shot terminal drain.

    The sink is a thin lifecycle shell over a :class:`StreamingEngine`.  It
    never owns episode, latency, deadline or monitor/extension state of its own.
    At run end it asks the flow for the pack report, validates and returns it;
    it writes no core
    artifact.

    Args:
        monitors: Monitor instances in task declaration order.  Mutually
            exclusive with ``engine``.
        engine: An already-built engine for tests or advanced composition.
            When supplied, ``retention``, ``publisher`` and ``extensions`` must
            be omitted because the engine already owns them.
        flow: Optional benchmark flow.  The default is a stateless
            :class:`~unigym_bench.flow.DefaultBenchmarkFlow` with no rules, so
            monitor values never terminate an episode by themselves.
        retention: Explicit event/episode retention for a sink-built engine.
        publisher: Optional write-only value publisher for a sink-built engine.
        loop_mode: ``"pool"`` or ``"sink-steered"``.  A sink-steered sink needs
            a value publisher on its engine; absent publisher is a construction
            error.
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
            publisher, or ``write_report`` is true without a construction
            ``out_dir``.
        TypeError: Both or neither of ``monitors``/``engine`` are supplied, an
            injected engine is mixed with separately supplied resources, or the
            supplied flow is malformed.
        ValueError: A monitor/extension name is duplicated or a declared
            ``report_extension`` reference is missing.
    """

    def __init__(
        self,
        monitors: Sequence[Monitor] | None = None,
        *,
        engine: StreamingEngine | None = None,
        flow: BenchmarkFlow | None = None,
        retention: RetentionSpec | None = None,
        publisher: ValuePublisher | None = None,
        loop_mode: str = DEFAULT_LOOP_MODE,
        extensions: Sequence[ReportExtension] | None = None,
        metadata: RunMetadata | None = None,
        monitor_extensions: Mapping[str, str] | None = None,
        write_report: bool = False,
    ) -> None:
        _validate_loop_mode(loop_mode)
        requested_flow: BenchmarkFlow | None = None
        if flow is not None:
            requested_flow = validate_flow(flow)
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
                "BenchmarkSink: loop_mode 'sink-steered' requires a value publisher "
                "on the engine"
            )
        self._loop_mode = loop_mode
        self._metadata = metadata
        self._write_report = write_report
        validate_report_configuration(
            monitors=self._engine.monitors,
            extensions=self._engine.extensions,
            monitor_extensions=monitor_extensions,
        )
        self._flow = DefaultBenchmarkFlow() if requested_flow is None else requested_flow
        self._fallback_flow = DefaultBenchmarkFlow()
        # Bind the sink-owned flow to the engine so a stateful flow's policy
        # state joins the single ``unigym_bench`` rewind provider.  The default
        # flow is stateless and adds no checkpoint key.
        self._engine.bind_flow(self._flow)
        self._monitor_extensions: Mapping[str, str] = (
            {} if monitor_extensions is None else dict(monitor_extensions)
        )
        self._report: dict[str, JsonValue] | None = None
        self._report_written = False

    @property
    def flow(self) -> BenchmarkFlow:
        """The validated flow composed by this sink."""
        return self._flow

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
        """Begin a tick, ask the flow, and apply the validated decision.

        ``begin_tick`` has already advanced episode tick counters and read the
        tick's monitor values before ``flow.on_tick`` runs.  If the callback
        raises, the sink discards the pending tick snapshot and re-raises; tick
        counters and observed values are not rolled back, so the run must abort
        rather than continue on partially observed state.
        """
        del t
        tick = self._engine.begin_tick(bundle)
        try:
            on_tick = getattr(self._flow, "on_tick", None)
            decision: FlowDecision | None = None
            if callable(on_tick):
                decision = on_tick(tick)
        except BaseException:
            self._engine.discard_tick(tick)
            raise
        if decision is None:
            decision = FlowDecision()
        self._engine.apply_decision(tick, decision)

    def drain_terminations(self) -> Mapping[int, Termination]:
        """Return and clear the engine's pending first-terminal facts."""
        return self._engine.drain_terminations()

    def on_episode_end(self, env_ids: object, bundle: dict[str, object]) -> None:
        """Begin the boundary, ask the flow, then commit the episode end.

        ``begin_episode_end`` has already evaluated monitors, refreshed their
        final values and built the final query snapshot before
        ``flow.on_episode_end`` runs.  If the callback raises, the sink discards
        the pending boundary and re-raises; no rollback of observed values is
        attempted, so the run must abort.
        """
        batch = self._engine.begin_episode_end(env_ids, bundle)
        if batch is None:
            return
        try:
            evidence = self._flow.on_episode_end(batch)
        except BaseException:
            self._engine.discard_episode_end(batch)
            raise
        self._engine.commit_episode_end(batch, evidence)

    def on_step_end(self, t: int, bundle: dict[str, object]) -> None:
        """Post-reset metrics hook; the reset notification owns reopening."""
        del t, bundle

    def on_reset(self, env_ids: object) -> None:
        """Apply the core reset notification without owning an episode clock.

        The core calls this after every real reset chain (pre-start reset-all,
        tick auto-reset, explicit ``reset`` and ``terminate(..., reset=True)``).
        A pre-start reset happens before the first tick binds the engine, so the
        engine side is a safe no-op: the first ``on_step`` binds the later run
        shape.  The flow is still notified so a state machine can clear pre-run
        state.  After binding, the engine reopens finalized envs and restarts
        open envs in place without fabricating an episode record, then the flow
        observes the same ids.  ``reset=False`` is never delivered by the core,
        and a flow exception propagates after the engine has applied the reset.
        """
        if self._engine.num_envs is not None:
            self._engine.on_reset(env_ids)
        on_reset = getattr(self._flow, "on_reset", None)
        if callable(on_reset):
            on_reset(env_ids)

    def on_run_end(self) -> dict[str, object]:
        """Finalize the run, let the flow own the report and return it.

        The report is assembled once and cached, so a second teardown call
        cannot drive extension ``on_run_end`` twice after a successful attempt.
        A failed assembly attempt is not cached: the next call retries the flow
        and therefore invokes extension ``on_run_end`` again.  With
        ``write_report`` the cached report is serialized to :attr:`report_path`;
        a failed write is raised and retried on the next call instead of leaving
        a cached "success".  No core artifact is written.
        """
        if self._report is None:
            aggregate = self._engine.run_summary()
            metadata = self._metadata
            if metadata is None:
                metadata = RunMetadata(
                    num_envs=self._engine.num_envs or 0,
                    loop_mode=self._loop_mode,
                    pack_version=__version__,
                )
            run = FlowRun(
                metadata=metadata,
                summary=aggregate,
                records=self._engine.records,
                monitor_views={
                    monitor.name: monitor.report_view() for monitor in self._engine.monitors
                },
                evidence=self._engine.evidence,
                retention=self._engine.retention,
                extensions=self._engine.extensions,
                monitor_names=tuple(monitor.name for monitor in self._engine.monitors),
                monitor_extensions=self._monitor_extensions,
            )
            result = self._flow.on_run_end(run)
            if result is None:
                result = self._fallback_flow.on_run_end(run)
            if not isinstance(result, Mapping):
                raise TypeError(
                    "benchmark flow.on_run_end: expected a mapping or None, "
                    f"got {type(result).__name__}"
                )
            normalized = ensure_json_value(result, where="benchmark flow report")
            if not isinstance(normalized, dict):
                raise TypeError("benchmark flow report: expected an object")
            self._report = _validate_report_document(normalized)
        path = self.report_path
        if path is not None and not self._report_written:
            _write_report_file(path, self._report)
            self._report_written = True
        return {SUMMARY_KEY: self._report}

    def stop_when(self) -> bool:
        """Return the stop flag requested by an applied flow decision."""
        return self._engine.stop_requested

    def on_rewind(self, info: dict[str, object]) -> None:
        """Mark the currently open episodes as belonging to a forked timeline."""
        self._engine.on_rewind(info)


def _validate_report_document(report: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Validate the exact report schema and top-level block set.

    The flow owns report content, not the document shape.  A flow cannot add a
    top-level block, omit a block, or return an older schema id.
    """
    keys = tuple(report)
    if keys != REPORT_BLOCKS:
        missing = sorted(set(REPORT_BLOCKS) - set(keys))
        extra = sorted(set(keys) - set(REPORT_BLOCKS))
        if missing or extra:
            raise ValueError(
                "benchmark flow report: expected exactly the top-level blocks "
                f"{list(REPORT_BLOCKS)}, missing {missing}, extra {extra}"
            )
        raise ValueError(
            "benchmark flow report: top-level blocks are out of order; "
            f"expected {list(REPORT_BLOCKS)}, got {list(keys)}"
        )
    schema = report["schema"]
    if schema != REPORT_SCHEMA:
        raise ValueError(
            f"benchmark flow report.schema: expected {REPORT_SCHEMA!r}, got {schema!r}; "
            "the previous external_benchmark_report/2 schema is rejected"
        )
    return report


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
