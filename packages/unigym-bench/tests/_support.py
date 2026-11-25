"""Construction helpers for monitor tests: real contexts and value sources.

These are test inputs, not stand-ins for pack components: every context is
built from a real :class:`unigym.evaluation.EnvView` and a
:class:`unigym.values.StepValues` snapshot, and the value helpers publish real
per-env tensors that a monitor resolves through an explicit ``value:`` source.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import torch

from unigym.core.episode_termination import (
    EpisodeOutcome,
    Termination,
    TerminationSource,
)
from unigym.evaluation import EnvStepContext, EnvView
from unigym.query import AABB, ContactManifold, Pose
from unigym.run import SinkContext
from unigym.run.spec import build_spec
from unigym.values import StepValues, ValueView
from unigym_bench.context import BenchmarkStepContext
from unigym_bench.engine import StreamingEngine
from unigym_bench.events import Event
from unigym_bench.json import JsonValue
from unigym_bench.monitors import BaseMonitor, Monitor, threshold_config
from unigym_bench.report import ScenarioCoverageExtension
from unigym_bench.sink import BenchmarkSink, make_sink


def make_view(num_envs: int = 2, *, device: str | torch.device = "cpu") -> EnvView:
    """Build a real, empty vector view with no observation snapshot."""
    return EnvView(record={}, obs=None, env_id=None, num_envs=num_envs, device=device)


def _env_values(env: object) -> tuple[int, ...] | None:
    """Return one env selection's ids for the counting query spy."""
    if env is None:
        return None
    if isinstance(env, torch.Tensor):
        return tuple(int(value) for value in env.detach().reshape(-1).tolist())
    if isinstance(env, int) and not isinstance(env, bool):
        return (int(env),)
    return tuple(int(value) for value in env)  # type: ignore[union-attr]


def _env_ids(env: object, num_envs: int) -> tuple[int, ...]:
    """Return the resolved ids for one env selection."""
    values = _env_values(env)
    return tuple(range(num_envs)) if values is None else values


class _UnusedQueryFace:
    """Explicit stand-in for a core query subface this card does not read."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"query face {name!r} is not used by this test double")


class CountingContacts:
    """Explicit spy over the narrow core contact boundary."""

    def __init__(self, num_envs: int = 2) -> None:
        self._num_envs = num_envs
        self.scan_calls: list[tuple[int, ...] | None] = []
        self.manifold_calls: list[tuple[str, tuple[int, ...] | None]] = []

    def _manifold(self, group: str, env: object) -> ContactManifold:
        ids = _env_ids(env, self._num_envs)
        return ContactManifold(
            env_ids=torch.tensor(ids, dtype=torch.int64),
            group=group,
            points=(),
        )

    def scan(self, *, env: object = None) -> ContactManifold:
        """Record the selection and return an empty real manifold."""
        self.scan_calls.append(_env_values(env))
        return self._manifold("*", env)

    def manifold(self, group: str, *, env: object = None) -> ContactManifold:
        """Record the group and selection and return an empty real manifold."""
        self.manifold_calls.append((group, _env_values(env)))
        return self._manifold(group, env)

    def jacobian(self, point: object, articulation: str) -> object:
        """Reject an unexercised read instead of returning a fake success."""
        raise AssertionError("contact jacobian is not used by this test double")

    def wrench_to_joint(self, point: object, articulation: str) -> torch.Tensor:
        """Reject an unexercised read instead of returning a fake success."""
        raise AssertionError("contact wrench_to_joint is not used by this test double")


class CountingState:
    """Explicit spy over the state subface used for a passthrough check."""

    def __init__(self, num_envs: int = 2) -> None:
        self._num_envs = num_envs
        self.body_pose_calls: list[tuple[str, str | None, tuple[int, ...] | None]] = []

    def body_pose(
        self,
        name: str,
        *,
        articulation: str | None = None,
        env: object = None,
    ) -> Pose:
        """Record the read and return one real per-env pose."""
        self.body_pose_calls.append((name, articulation, _env_values(env)))
        ids = _env_ids(env, self._num_envs)
        rows = len(ids)
        return Pose(
            env_ids=torch.tensor(ids, dtype=torch.int64),
            pos_w=torch.zeros(rows, 3),
            quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(rows, 1),
        )


class CountingGeometry:
    """Explicit spy over the geometry subface used for a passthrough check."""

    def __init__(self, num_envs: int = 2) -> None:
        self._num_envs = num_envs
        self.aabb_calls: list[tuple[object, tuple[int, ...] | None]] = []

    def aabb(self, entity: object, *, env: object = None) -> AABB:
        """Record the read and return a real zero-sized per-env box."""
        self.aabb_calls.append((entity, _env_values(env)))
        ids = _env_ids(env, self._num_envs)
        zeros = torch.zeros(len(ids), 3)
        return AABB(env_ids=torch.tensor(ids, dtype=torch.int64), min_w=zeros, max_w=zeros)


class CountingQuery:
    """Explicit spy implementing the core read subfaces TickQuery uses."""

    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.contacts = CountingContacts(num_envs)
        self.state = CountingState(num_envs)
        self.geometry = CountingGeometry(num_envs)
        self.model = _UnusedQueryFace()
        self.kinematics = _UnusedQueryFace()
        self.actuators = _UnusedQueryFace()

    @property
    def backend(self) -> str:
        """The canonical fake backend name."""
        return "counting"

    def scratch(self, env: int | None = None) -> object:
        """Reject scratch access; the evaluation snapshot never exposes it."""
        raise AssertionError("scratch is not reachable from TickQuery")

    def require(self, *ops: str) -> None:
        """Accept every capability check; the spy implements the read subset."""
        del ops

    def close(self) -> None:
        """No-op: the spy owns no resources."""


def make_query_view(num_envs: int = 2) -> tuple[EnvView, CountingQuery, list[int]]:
    """Build a real vector view whose lazy query resolves to a counting spy."""
    fake = CountingQuery(num_envs)
    factory_calls: list[int] = []

    def factory() -> CountingQuery:
        factory_calls.append(1)
        return fake

    view = EnvView(
        record={},
        obs=None,
        env_id=None,
        num_envs=num_envs,
        device="cpu",
        query_factory=factory,
    )
    return view, fake, factory_calls


def make_context(
    *,
    num_envs: int = 2,
    t: int = 0,
    cmd: torch.Tensor | None = None,
    view: EnvView | None = None,
    values: ValueView | None = None,
    reset_state: Mapping[int, JsonValue] | None = None,
    step_index: torch.Tensor | None = None,
    time_s: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> BenchmarkStepContext:
    """Build a real benchmark context from a view and value snapshot."""
    target = torch.device("cpu") if device is None else device
    if view is None:
        view = make_view(num_envs, device=target)
    if step_index is None:
        step_index = torch.zeros(num_envs, dtype=torch.int64, device=target)
    return BenchmarkStepContext(
        view=view,
        values=StepValues({}) if values is None else values,
        t=t,
        step_index=step_index,
        time_s=time_s,
        cmd=cmd,
        reset_state={} if reset_state is None else reset_state,
    )


def make_bundle(
    *,
    num_envs: int = 2,
    t: int = 0,
    cmd: torch.Tensor | None = None,
    view: EnvView | None = None,
    values: ValueView | None = None,
    reset_state: Mapping[int, JsonValue] | None = None,
    step_index: torch.Tensor | None = None,
    time_s: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
    terminations: Mapping[int, object] | None = None,
    final_view: EnvView | None = None,
    final_values: ValueView | None = None,
    final_t: int | None = None,
    final_step: EnvStepContext | None = None,
) -> dict[str, object]:
    """Build a real runner-shaped bundle carrying step and final contexts.

    The runner builds ``final_step`` after ``on_step`` and before
    ``on_episode_end``.  Tests may supply a distinct final view/value snapshot
    to prove episode-end predicates read the final boundary, not the tick one.
    """
    target = torch.device(device)
    context = make_context(
        num_envs=num_envs,
        t=t,
        cmd=cmd,
        view=view,
        values=values,
        reset_state=reset_state,
        step_index=step_index,
        time_s=time_s,
        device=target,
    )
    if final_step is None:
        final_step = make_context(
            num_envs=num_envs,
            t=t if final_t is None else final_t,
            cmd=cmd,
            view=view if final_view is None else final_view,
            values=values if final_values is None else final_values,
            reset_state=reset_state,
            step_index=step_index,
            time_s=time_s,
            device=target,
        )
    bundle: dict[str, object] = {"step": context, "final_step": final_step}
    if terminations is not None:
        bundle["terminations"] = terminations
    return bundle


def make_termination(
    code: str = "custom.test.reason",
    *,
    outcome: EpisodeOutcome = EpisodeOutcome.SUCCESS,
    source: TerminationSource = TerminationSource.CUSTOM,
    metrics: Mapping[str, float] | None = None,
) -> Termination:
    """Build a real core termination for engine tests."""
    return Termination(
        outcome=outcome,
        code=code,
        source=source,
        metrics={} if metrics is None else metrics,
    )


@dataclass(frozen=True)
class TickScript:
    """One tick's scripted monitor outputs."""

    signals: Mapping[str, torch.Tensor] = field(default_factory=dict)
    events: tuple[Event, ...] = ()
    terminals: Mapping[int, Termination] = field(default_factory=dict)


class ScriptedMonitor(BaseMonitor):
    """Test monitor that emits exactly the outputs scripted for each tick."""

    def __init__(self, name: str, script: Mapping[int, TickScript]) -> None:
        super().__init__(name)
        self._script = dict(script)
        self._signals: dict[str, torch.Tensor] = {}
        self._events: list[Event] = []
        self._terminals: dict[int, Termination] = {}
        self.episode_ends: list[tuple[int, ...]] = []
        self.final_contexts: list[EnvStepContext | None] = []
        self.resets: list[tuple[int, ...]] = []

    def on_tick(self, context: EnvStepContext) -> None:
        step = self._script.get(context.t)
        if step is None:
            self._signals = {}
            return
        self._signals = {name: value.to(context.device) for name, value in step.signals.items()}
        self._events = list(step.events)
        self._terminals = dict(step.terminals)

    def signals(self) -> Mapping[str, torch.Tensor]:
        return dict(self._signals)

    def drain_events(self) -> Sequence[Event]:
        events = tuple(self._events)
        self._events.clear()
        return events

    def drain_terminals(self) -> Mapping[int, Termination]:
        terminals = dict(self._terminals)
        self._terminals.clear()
        return terminals

    def on_episode_end(self, env_ids: Sequence[int], context: EnvStepContext | None = None) -> None:
        self.episode_ends.append(tuple(env_ids))
        self.final_contexts.append(context)

    def reset(self, env_ids: Sequence[int]) -> None:
        self.resets.append(tuple(env_ids))


def make_values(
    row: Sequence[float],
    *,
    name: str = "level",
    dtype: torch.dtype = torch.float32,
) -> StepValues:
    """Return a real step-value snapshot with one per-env row."""
    return StepValues({name: torch.tensor(list(row), dtype=dtype)})


def make_value_context(
    row: Sequence[float],
    *,
    name: str = "level",
    t: int = 0,
    num_envs: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> BenchmarkStepContext:
    """Build a real context whose value source carries one per-env row."""
    target = torch.device(device)
    tensor = torch.tensor(list(row), dtype=dtype, device=target)
    resolved_num_envs = int(tensor.shape[0]) if num_envs is None else num_envs
    return make_context(
        num_envs=resolved_num_envs,
        t=t,
        values=StepValues({name: tensor}),
        device=target,
    )


_SINK_CONTEXT_CFG: dict[str, JsonValue] = {
    "run_name": "sink-context",
    "seed": 0,
    "env": {"backend": "mujoco", "num_envs": 1},
    "world_base": {"mjcf": "model.xml"},
    "collect": {"steps": 1},
    "articulations": [{"name": "arm", "joints": [{"name": "shoulder"}]}],
    "action_spec": [
        {
            "name": "arm",
            "articulation": "arm",
            "joints": ["shoulder"],
            "target": "position",
            "role": "body",
        }
    ],
    "policies": [{"adapter": "manual", "groups": [["arm", "arm"]]}],
    "observations": {"robot": ["arm"]},
}


def make_sink_context(
    *,
    options: Mapping[str, JsonValue] | None = None,
    values: object | None = None,
    auto_reset: bool | None = None,
) -> SinkContext:
    """Build a real SinkContext over a real, backend-free RunSpec."""
    spec = build_spec(_SINK_CONTEXT_CFG, Namespace(config="run.yaml"))
    if auto_reset is not None:
        spec = replace(spec, runner={**spec.runner, "auto_reset": auto_reset})
    return SinkContext(
        spec=spec,
        out_dir="data_out/sink-context",
        num_envs=1,
        policy_dt=0.02,
        seed=0,
        action_groups=(),
        run_identity=None,
        options={} if options is None else options,
        values=values,
    )


def make_threshold_monitors() -> tuple[Monitor, ...]:
    """Importable zero-argument monitor factory used by make_sink tests."""
    return (
        threshold_config(
            "contact",
            "value:level",
            threshold=0.5,
            terminal_outcome="success",
        ),
    )


class IntegrationMonitor(BaseMonitor):
    """Minimal monitor used by the real-run integration test.

    It counts ticks and contributes one JSON-safe report view; it owns no
    tensor, view or episode state, so it can run unmodified through the core
    runner.
    """

    def __init__(self, name: str = "integration") -> None:
        super().__init__(name)
        self.ticks = 0

    def on_tick(self, context: EnvStepContext) -> None:
        """Count one observed policy tick."""
        self.ticks += 1

    def report_view(self) -> Mapping[str, JsonValue]:
        """Return the observed tick count."""
        return {"ticks": self.ticks}


def make_integration_monitors() -> tuple[Monitor, ...]:
    """Importable zero-argument monitor factory for the real-run test."""
    return (IntegrationMonitor(),)


class ContextSpyMonitor(BaseMonitor):
    """Test monitor that records each tick context without reading a query."""

    def __init__(self, name: str = "context_spy") -> None:
        super().__init__(name)
        self.contexts: list[BenchmarkStepContext] = []

    def on_tick(self, context: EnvStepContext) -> None:
        """Record this tick's context for identity assertions."""
        assert isinstance(context, BenchmarkStepContext)
        self.contexts.append(context)


class QuerySpyMonitor(ContextSpyMonitor):
    """Test monitor that reads one contact scan through the tick snapshot."""

    def on_tick(self, context: EnvStepContext) -> None:
        """Record the context and exercise the shared scan cache."""
        super().on_tick(context)
        query = self.contexts[-1].query
        if query is not None:
            query.contacts.scan()


class QueryErrorSpyMonitor(BaseMonitor):
    """Test monitor that records the error from one attempted contact read."""

    def __init__(self, name: str = "query_error_spy") -> None:
        super().__init__(name)
        self.errors: list[Exception] = []

    def on_tick(self, context: EnvStepContext) -> None:
        """Attempt a query read, recording the raised error instead of escaping."""
        assert isinstance(context, BenchmarkStepContext)
        try:
            query = context.query
            if query is None:
                raise AssertionError("engine tick did not attach a TickQuery")
            _ = query.contacts
        except Exception as exc:
            self.errors.append(exc)


class QueryFailMonitor(BaseMonitor):
    """Test monitor that records its boundary context then raises on demand."""

    def __init__(
        self,
        name: str = "query_fail",
        *,
        fail_on_tick: bool = False,
        fail_on_episode_end: bool = False,
    ) -> None:
        super().__init__(name)
        self._fail_on_tick = fail_on_tick
        self._fail_on_episode_end = fail_on_episode_end
        self.contexts: list[BenchmarkStepContext] = []
        self.final_contexts: list[BenchmarkStepContext] = []

    def on_tick(self, context: EnvStepContext) -> None:
        """Record the context, then raise when configured."""
        assert isinstance(context, BenchmarkStepContext)
        self.contexts.append(context)
        if self._fail_on_tick:
            raise RuntimeError("query fail tick")

    def on_episode_end(self, env_ids: Sequence[int], context: EnvStepContext | None = None) -> None:
        """Record the final context, then raise when configured."""
        assert isinstance(context, BenchmarkStepContext)
        self.final_contexts.append(context)
        if self._fail_on_episode_end:
            raise RuntimeError("query fail episode_end")


class BundleRecordingSink:
    """Test harness forwarding to the real pack sink while recording bundles.

    The pack sink still owns every callback and all report state; this wrapper
    only records which keys the core runner put in each bundle and which core
    ``EpisodeTermination`` facts accompanied an episode boundary.  It exists so
    the integration test can prove the retired ``bundle["env"]`` key is absent
    and compare the pack's termination counts with the core facts of the same
    run.
    """

    def __init__(self, inner: BenchmarkSink) -> None:
        self.inner = inner
        self.step_bundle_keys: list[tuple[str, ...]] = []
        self.episode_terminations: list[dict[int, object]] = []

    @property
    def rewind_provider(self) -> StreamingEngine:
        """Forward the pack engine rewind provider unchanged."""
        return self.inner.rewind_provider

    def on_step(self, t: int, bundle: dict[str, object]) -> None:
        """Record the bundle's key set, then drive the pack sink."""
        self.step_bundle_keys.append(tuple(sorted(bundle)))
        self.inner.on_step(t, bundle)

    def on_episode_end(self, env_ids: object, bundle: dict[str, object]) -> None:
        """Record this boundary's core termination facts, then delegate."""
        terminations = bundle.get("terminations")
        self.episode_terminations.append(
            dict(terminations) if isinstance(terminations, Mapping) else {}
        )
        self.inner.on_episode_end(env_ids, bundle)

    def on_step_end(self, t: int, bundle: dict[str, object]) -> None:
        """Forward the post-reset callback."""
        self.inner.on_step_end(t, bundle)

    def on_reset(self, env_ids: object) -> None:
        """Forward the reset notification to the pack sink."""
        self.inner.on_reset(env_ids)

    def on_run_end(self) -> dict[str, object]:
        """Return the pack sink's summary unchanged."""
        return self.inner.on_run_end()

    def drain_terminations(self) -> Mapping[int, object]:
        """Forward the pack terminal drain."""
        return self.inner.drain_terminations()

    def stop_when(self) -> bool:
        """Forward the pack stop policy."""
        return self.inner.stop_when()

    def on_rewind(self, info: dict[str, object]) -> None:
        """Forward the rewind notification."""
        self.inner.on_rewind(info)


def make_observed_sink(ctx: SinkContext) -> BundleRecordingSink:
    """Importable real-run sink factory wrapping ``make_sink``."""
    return BundleRecordingSink(make_sink(ctx))


class StaticReportExtension:
    """Stateless extension that returns a fixed block and keeps no state."""

    stateful = False

    def __init__(self, name: str, block: Mapping[str, JsonValue]) -> None:
        self.name = name
        self._block = block

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, object]) -> None:
        """No accumulator."""
        del env_ids, records

    def on_run_end(self, summary: object) -> Mapping[str, JsonValue]:
        """Return the fixed block."""
        del summary
        return dict(self._block)


class ViewReportExtension:
    """Stateless extension proving fallback views reach ``on_run_end``."""

    stateful = False

    def __init__(self, name: str = "contact_report") -> None:
        self.name = name

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, object]) -> None:
        """No accumulator."""
        del env_ids, records

    def on_run_end(self, summary: object) -> Mapping[str, JsonValue]:
        """Report whether the covered monitor's fallback view is visible."""
        views = getattr(summary, "monitor_views", {})
        return {"saw_view": "contact" in views}


class CaptureReportExtension:
    """Stateless extension that records boundary calls into a caller-owned list.

    The extension itself owns no accumulator; the test supplies the list, so
    the explicit ``stateful = False`` declaration stays true.
    """

    stateful = False

    def __init__(
        self, calls: list[tuple[tuple[int, ...], object]], *, name: str = "capture"
    ) -> None:
        self.name = name
        self._calls = calls

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, object]) -> None:
        """Record the due ids and the read-only mapping in the caller's list."""
        self._calls.append((tuple(env_ids), records))

    def on_run_end(self, summary: object) -> Mapping[str, JsonValue]:
        """Return the number of observed boundaries."""
        del summary
        return {"boundaries": len(self._calls)}


class CountingReportExtension:
    """Stateful extension whose accumulator must survive a rewind."""

    stateful = True

    def __init__(self, name: str = "counts") -> None:
        self.name = name
        self.episodes = 0
        self.envs: list[int] = []
        self.run_end_calls = 0

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, object]) -> None:
        """Accumulate one count per due environment."""
        del records
        ids = [int(env_id) for env_id in env_ids]
        self.episodes += len(ids)
        self.envs.extend(ids)

    def on_run_end(self, summary: object) -> Mapping[str, JsonValue]:
        """Count the call and return the accumulated state."""
        del summary
        self.run_end_calls += 1
        return {
            "episodes": self.episodes,
            "envs": list(self.envs),
            "run_end_calls": self.run_end_calls,
        }

    def snapshot(self) -> JsonValue:
        """Return a private JSON-safe copy of the accumulator."""
        return {
            "episodes": self.episodes,
            "envs": list(self.envs),
            "run_end_calls": self.run_end_calls,
        }

    def restore(self, state: JsonValue) -> None:
        """Replace the accumulator from a snapshot's private copy."""
        assert isinstance(state, dict)
        self.episodes = int(state["episodes"])
        envs = state["envs"]
        assert isinstance(envs, list)
        self.envs = [int(value) for value in envs]
        self.run_end_calls = int(state["run_end_calls"])


class NonJsonReportExtension:
    """Stateless extension that leaks a tensor into its block."""

    stateful = False
    name = "bad"

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, object]) -> None:
        """No accumulator."""
        del env_ids, records

    def on_run_end(self, summary: object) -> Mapping[str, object]:
        """Return a non-JSON value."""
        del summary
        return {"tensor": torch.tensor(1.0)}


class ListReportExtension:
    """Stateless extension that returns a non-mapping block."""

    stateful = False
    name = "listy"

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, object]) -> None:
        """No accumulator."""
        del env_ids, records

    def on_run_end(self, summary: object) -> object:
        """Return a list, which is not a report block."""
        del summary
        return [1, 2, 3]


class MissingSnapshotExtension:
    """Invalid double: declares stateful without snapshot/restore."""

    stateful = True
    name = "no_snapshot"

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, object]) -> None:
        """No accumulator."""
        del env_ids, records

    def on_run_end(self, summary: object) -> Mapping[str, JsonValue]:
        """Return an empty block."""
        del summary
        return {}


class StatelessWithSnapshotExtension:
    """Invalid double: declares stateless but implements snapshot/restore."""

    stateful = False
    name = "contradiction"

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, object]) -> None:
        """No accumulator."""
        del env_ids, records

    def on_run_end(self, summary: object) -> Mapping[str, JsonValue]:
        """Return an empty block."""
        del summary
        return {}

    def snapshot(self) -> JsonValue:
        """Return an empty snapshot."""
        return {}

    def restore(self, state: JsonValue) -> None:
        """Accept an empty snapshot."""
        del state


class UndeclaredExtension:
    """Invalid double: does not declare ``stateful``."""

    name = "undeclared"

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, object]) -> None:
        """No accumulator."""
        del env_ids, records

    def on_run_end(self, summary: object) -> Mapping[str, JsonValue]:
        """Return an empty block."""
        del summary
        return {}


class SummaryFactsReportExtension:
    """Stateless extension that projects facts from ``RunSummary`` only."""

    stateful = False
    name = "summary_facts"

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, object]) -> None:
        """No accumulator."""
        del env_ids, records

    def on_run_end(self, summary: object) -> Mapping[str, JsonValue]:
        """Return retained-episode facts visible through the summary."""
        episode_summary = getattr(summary, "episode_summary")
        retention = getattr(summary, "retention")
        return {
            "episodes": len(getattr(summary, "episodes")),
            "verdicts": dict(episode_summary.verdict_counts),
            "retain_episodes": bool(retention.retain_episodes),
        }


class LeakyReportViewMonitor(BaseMonitor):
    """Monitor whose fallback view leaks a tensor."""

    def report_view(self) -> Mapping[str, JsonValue]:
        """Return a non-JSON view."""
        return {"bad": torch.tensor(1.0)}  # type: ignore[return-value]


class RecordingReportExtension:
    """Stateful extension used by the sink construction-seam tests."""

    stateful = True

    def __init__(self, name: str = "execution") -> None:
        self.name = name
        self.env_ends: list[tuple[int, ...]] = []

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, object]) -> None:
        """Record the due env ids without retaining the records."""
        del records
        self.env_ends.append(tuple(env_ids))

    def on_run_end(self, summary: object) -> Mapping[str, JsonValue]:
        """Return a small JSON-safe block counting retained episodes and ends."""
        episodes = len(getattr(summary, "episodes"))
        return {"episodes": episodes, "env_ends": len(self.env_ends)}

    def snapshot(self) -> JsonValue:
        """Return a private JSON-safe copy of the boundary history."""
        return {"env_ends": [list(ids) for ids in self.env_ends]}

    def restore(self, state: JsonValue) -> None:
        """Replace the boundary history from a snapshot's private copy."""
        assert isinstance(state, dict)
        env_ends = state["env_ends"]
        assert isinstance(env_ends, list)
        self.env_ends = [tuple(int(value) for value in ids) for ids in env_ends]


def make_report_extensions() -> tuple[RecordingReportExtension, ...]:
    """Importable zero-argument report-extension factory used by make_sink tests."""
    return (RecordingReportExtension(),)


def make_coverage_extensions() -> tuple[ScenarioCoverageExtension, ...]:
    """Importable coverage extension factory used by the real-run test."""
    return (ScenarioCoverageExtension("coverage", commands=[[0.0]]),)


@dataclass(frozen=True)
class SamplerEnv:
    """Minimal read-only env face used by command-sampler bind tests."""

    num_envs: int = 1


class SamplerValues:
    """Static read-only value view for command-sampler closed-loop tests.

    Values are stored verbatim so a test can inject an invalid level (a list,
    an integer tensor or a wrong-shape tensor) and assert the sampler's
    teaching error.  The real transport path is the core
    :class:`unigym.values.ValueStore`.

    Args:
        values: Optional initial level values keyed by value name.
        record: When true, append every ``get`` name to :attr:`requests` and
            count every ``snapshot`` call in :attr:`snapshot_calls`.
    """

    def __init__(self, values: Mapping[str, object] | None = None, *, record: bool = False) -> None:
        self.values: dict[str, object] = {} if values is None else dict(values)
        self.requests: list[str] = []
        self.snapshot_calls = 0
        self._record = record

    def get(self, name: str) -> object:
        """Return the stored level for *name*, or ``None`` when absent."""
        if self._record:
            self.requests.append(name)
        return self.values.get(name)

    def snapshot(self) -> Mapping[str, object]:
        """Return an isolated copy of the stored levels."""
        if self._record:
            self.snapshot_calls += 1
        return dict(self.values)

    def __contains__(self, name: object) -> bool:
        """Return whether a level is present."""
        return name in self.values
