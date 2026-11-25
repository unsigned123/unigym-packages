"""Benchmark-flow seam: contracts, conductor behavior and report identity."""

from __future__ import annotations

import pytest
import torch
from unigym.core.episode_termination import TerminationSource

from unigym_bench.engine import EngineStateError, StreamingEngine
from unigym_bench.events import EpisodeSummary, Event
from unigym_bench.flow import (
    BenchmarkTick,
    DefaultBenchmarkFlow,
    EpisodeBatch,
    FlowDecision,
    FlowRun,
    _MonitorTickOutput,
    _monitor_bridge_decision,
    validate_flow,
)
from unigym_bench.report import REPORT_SCHEMA
from unigym_bench.sink import BenchmarkSink
from unigym_bench.specs import RetentionSpec
from _support import (
    CountingReportExtension,
    ScriptedMonitor,
    TickScript,
    make_bundle,
    make_context,
    make_termination,
)

_EVENT = Event(
    name="contact",
    code="custom.contact.event",
    env_ids=(0,),
    severity="warning",
)


class NoOpFlow:
    """Valid flow with no tick callback and no report contribution."""

    name = "noop"
    stateful = False

    def on_episode_end(self, batch: EpisodeBatch) -> None:
        """Observe nothing."""
        del batch

    def on_run_end(self, run: FlowRun) -> None:
        """Delegate the report to the sink's default fallback."""
        del run


class EventFlow:
    """Flow that contributes one event and requests a run stop."""

    name = "evented"
    stateful = False

    def on_tick(self, tick: BenchmarkTick) -> FlowDecision:
        """Emit the fixed event for env 0 and request a stop."""
        del tick
        return FlowDecision(events=(_EVENT,), stop=True)

    def on_episode_end(self, batch: EpisodeBatch) -> None:
        """Observe nothing."""
        del batch

    def on_run_end(self, run: FlowRun) -> None:
        """Delegate the report to the default fallback."""
        del run


class BadSourceFlow:
    """Flow whose terminal uses a non-custom source."""

    name = "bad_source"
    stateful = False

    def on_tick(self, tick: BenchmarkTick) -> FlowDecision:
        """Return a terminal the engine must reject before mutation."""
        del tick
        termination = make_termination("custom.bad.reason", source=TerminationSource.RUNTIME)
        return FlowDecision(terminations={0: termination})

    def on_episode_end(self, batch: EpisodeBatch) -> None:
        """Observe nothing."""
        del batch

    def on_run_end(self, run: FlowRun) -> None:
        """Delegate the report to the default fallback."""
        del run


class StatefulFlow(NoOpFlow):
    """Flow that declares rewindable state before F2 supports it."""

    name = "stateful"
    stateful = True


class HiddenSnapshotFlow(NoOpFlow):
    """Stateless flow that implements the rewind capability anyway."""

    name = "hidden_snapshot"
    stateful = False

    def snapshot(self) -> object:
        """Return fake state."""
        return {}

    def restore(self, state: object) -> None:
        """Ignore the fake state."""
        del state


class MissingCallbacksFlow:
    """Flow missing both required lifecycle callbacks."""

    name = "missing"
    stateful = False


def _scripted_sink(
    *flows: object, extension: CountingReportExtension | None = None
) -> tuple[BenchmarkSink, StreamingEngine, ScriptedMonitor]:
    """Build a sink over one scripted monitor with optional explicit flow."""
    monitor = ScriptedMonitor(
        "contact",
        {
            0: TickScript(
                signals={"level": torch.tensor([1.0])},
                events=(_EVENT,),
                terminals={0: make_termination("custom.contact.hit")},
            )
        },
    )
    engine = StreamingEngine(
        [monitor],
        retention=RetentionSpec(retain_episodes=True),
        extensions=() if extension is None else (extension,),
    )
    flow = flows[0] if flows else None
    sink = (
        BenchmarkSink(engine=engine, flow=flow)
        if flow is not None
        else BenchmarkSink(engine=engine)
    )
    return sink, engine, monitor


def _drive(sink: BenchmarkSink) -> dict[str, object]:
    """Drive one tick, one due boundary and run end."""
    bundle = make_bundle(num_envs=1, t=0)
    sink.on_step(0, bundle)
    sink.drain_terminations()
    sink.on_episode_end([0], bundle)
    return sink.on_run_end()


def test_default_flow_declares_a_stateless_default_contract() -> None:
    flow = DefaultBenchmarkFlow()

    assert flow.name == "default"
    assert flow.stateful is False
    assert validate_flow(flow) is flow


def test_validate_flow_accepts_a_no_op_flow_without_on_tick() -> None:
    flow = NoOpFlow()

    assert validate_flow(flow) is flow
    assert not hasattr(flow, "on_tick")


@pytest.mark.parametrize(
    ("flow", "error"),
    [
        (MissingCallbacksFlow(), TypeError),
        (HiddenSnapshotFlow(), TypeError),
        (StatefulFlow(), ValueError),
    ],
)
def test_validate_flow_rejects_malformed_or_stateful_flows(
    flow: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        validate_flow(flow)


def test_validate_flow_rejects_a_missing_name_and_non_callable_tick() -> None:
    class MissingName:
        stateful = False

        def on_episode_end(self, batch: EpisodeBatch) -> None:
            del batch

        def on_run_end(self, run: FlowRun) -> None:
            del run

    class BadTick(NoOpFlow):
        on_tick = 1

    with pytest.raises(TypeError, match="name"):
        validate_flow(MissingName())
    with pytest.raises(TypeError, match="on_tick"):
        validate_flow(BadTick())


def test_benchmark_tick_freezes_outputs_and_shares_the_query_snapshot() -> None:
    context = make_context(num_envs=2)
    output = _MonitorTickOutput(
        monitor="contact",
        values={"level": torch.tensor([1.0, 0.0])},
    )
    tick = BenchmarkTick(step=context, monitor_outputs=(output,))

    assert tick.query is context.query
    assert set(tick.monitor_values) == {"contact"}
    assert tick.monitor_values["contact"]["level"].tolist() == [1.0, 0.0]
    with pytest.raises(TypeError):
        tick.monitor_outputs[0].values["new"] = torch.tensor([1.0, 1.0])  # type: ignore[index]


def test_flow_decision_accepts_a_mapping_and_keeps_ordered_requests() -> None:
    first = make_termination("custom.first.reason")
    second = make_termination("custom.second.reason")

    mapped = FlowDecision(terminations={0: first})
    assert mapped.terminal_requests == ((0, first),)

    ordered = FlowDecision.from_requests(requests=((0, first), (0, second)))
    assert ordered.terminal_requests == ((0, first), (0, second))
    assert dict(ordered.terminations) == {0: second}
    with pytest.raises(TypeError, match="stop"):
        FlowDecision(stop=1)  # type: ignore[arg-type]


def test_sink_defaults_to_the_default_flow_and_keeps_report_schema() -> None:
    sink, _engine, _monitor = _scripted_sink()

    assert isinstance(sink.flow, DefaultBenchmarkFlow)
    report = _drive(sink)["BenchmarkSink"]
    assert report["schema"] == REPORT_SCHEMA
    assert tuple(report) == (
        "schema",
        "run",
        "tasks",
        "episodes",
        "terminations",
        "execution",
        "monitors",
        "extensions",
    )


def test_default_flow_report_is_identical_to_an_explicit_default_flow() -> None:
    implicit, _, _ = _scripted_sink(extension=CountingReportExtension())
    explicit, _, _ = _scripted_sink(DefaultBenchmarkFlow(), extension=CountingReportExtension())

    assert _drive(implicit) == _drive(explicit)


def test_default_flow_preserves_monitor_events_and_termination_conflicts() -> None:
    first = ScriptedMonitor(
        "first", {0: TickScript(terminals={0: make_termination("custom.first.reason")})}
    )
    second = ScriptedMonitor(
        "second", {0: TickScript(terminals={0: make_termination("custom.second.reason")})}
    )
    engine = StreamingEngine([first, second], retention=RetentionSpec(retain_episodes=True))
    sink = BenchmarkSink(engine=engine)
    bundle = make_bundle(num_envs=1, t=0)

    sink.on_step(0, bundle)
    pending = sink.drain_terminations()
    assert pending[0].code == "custom.first.reason"
    assert dict(engine.conflicts) == {0: 1}
    assert engine.conflict_count == 1

    sink.on_episode_end([0], bundle)
    report = sink.on_run_end()["BenchmarkSink"]
    assert report["execution"]["terminal_conflicts"] == 1


def test_default_flow_journals_tick_events() -> None:
    sink, engine, _monitor = _scripted_sink()

    _drive(sink)

    assert engine.journal == (_EVENT,)
    assert engine.summary.event_count == 1


def test_custom_flow_can_add_events_and_stop_without_a_report() -> None:
    sink, engine, _monitor = _scripted_sink(EventFlow())

    report = _drive(sink)["BenchmarkSink"]

    assert engine.journal == (_EVENT,)
    assert sink.stop_when() is True
    assert report["schema"] == REPORT_SCHEMA


def test_custom_flow_decision_is_validated_before_mutating_the_engine() -> None:
    sink, engine, _monitor = _scripted_sink(BadSourceFlow())

    with pytest.raises(ValueError, match="source must be custom"):
        sink.on_step(0, make_bundle(num_envs=1, t=0))

    assert sink.drain_terminations() == {}
    assert engine.journal == ()
    assert engine.conflict_count == 0


def test_engine_conductor_begins_and_applies_one_tick_at_a_time() -> None:
    monitor = ScriptedMonitor("contact", {0: TickScript(signals={"level": torch.tensor([1.0])})})
    engine = StreamingEngine([monitor])
    bundle = make_bundle(num_envs=1, t=0)

    tick = engine.begin_tick(bundle)
    assert tick.query is tick.step.query
    with pytest.raises(EngineStateError, match="already active"):
        engine.begin_tick(bundle)
    engine.apply_decision(tick, _monitor_bridge_decision(tick))
    next_tick = engine.begin_tick(bundle)
    assert next_tick is not tick
    engine.discard_tick(next_tick)


def test_engine_conductor_episode_boundary_commits_a_record() -> None:
    sink_monitor = ScriptedMonitor(
        "contact", {0: TickScript(terminals={0: make_termination("custom.contact.hit")})}
    )
    engine = StreamingEngine([sink_monitor], retention=RetentionSpec(retain_episodes=True))
    bundle = make_bundle(num_envs=1, t=0)

    engine.on_step(bundle)
    batch = engine.begin_episode_end([0], bundle)
    assert batch is not None
    assert batch.due == (0,)
    with pytest.raises(EngineStateError, match="already active"):
        engine.begin_episode_end([0], bundle)
    assert engine.commit_episode_end(batch, None) == (0,)
    assert engine.records[0].verdict == "terminal"


def test_episode_batch_rejects_invalid_records_and_flow_evidence_is_rejected() -> None:
    batch = EpisodeBatch(due=(0,), final_step=make_context(num_envs=1))
    with pytest.raises(TypeError):
        EpisodeBatch(due=(0,), final_step=batch.final_step, records={0: object()})  # type: ignore[dict-item]

    monitor = ScriptedMonitor("idle", {})
    engine = StreamingEngine([monitor])
    engine.on_step(make_bundle(num_envs=1, t=0))
    active = engine.begin_episode_end([0], make_bundle(num_envs=1, t=0))
    assert active is not None
    with pytest.raises(NotImplementedError, match="evidence"):
        engine.commit_episode_end(active, {0: {"score": 1.0}})


def test_default_flow_drives_each_extension_once() -> None:
    extension = CountingReportExtension()
    sink, _engine, _monitor = _scripted_sink(extension=extension)

    first = _drive(sink)
    second = sink.on_run_end()

    assert extension.run_end_calls == 1
    assert first == second


def test_benchmark_tick_rejects_duplicate_monitor_outputs() -> None:
    context = make_context(num_envs=1)
    output = _MonitorTickOutput(monitor="same")

    with pytest.raises(ValueError, match="duplicate monitor"):
        BenchmarkTick(step=context, monitor_outputs=(output, output))


def test_rewind_and_restore_clear_the_transient_stop_flag() -> None:
    engine = StreamingEngine([])
    sink = BenchmarkSink(engine=engine)
    engine.on_step(make_bundle(num_envs=1, t=0))
    snapshot = engine.rewind_state()

    tick = engine.begin_tick(make_bundle(num_envs=1, t=1))
    engine.apply_decision(tick, FlowDecision(stop=True))
    assert sink.stop_when() is True

    engine.apply_rewind_state(snapshot)
    assert sink.stop_when() is False

    tick = engine.begin_tick(make_bundle(num_envs=1, t=2))
    engine.apply_decision(tick, FlowDecision(stop=True))
    assert sink.stop_when() is True
    sink.on_rewind({})
    assert sink.stop_when() is False


def test_flow_run_accepts_minimal_report_bundle() -> None:
    from unigym_bench.report import RunMetadata

    run = FlowRun(
        metadata=RunMetadata(),
        summary=EpisodeSummary(),
        records=(),
        monitor_views={},
    )

    assert run.records == ()
    assert run.monitor_names == ()
    assert not hasattr(run, "monitors")
    assert dict(run.monitor_extensions) == {}
    assert dict(run.monitor_views) == {}
