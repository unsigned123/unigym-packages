"""Report builder, extension lifecycle, fallback and pack-owned artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from unigym.core.episode_termination import (
    EpisodeOutcome,
    EpisodeTermination,
    Termination,
    TerminationSource,
)
from unigym.core.identity import RobotIdentity, RunIdentity
from unigym.errors import ConfigError

from unigym_bench.engine import StreamingEngine
from unigym_bench.events import EpisodeRecord, EpisodeSummary
from unigym_bench.flow import (
    DefaultBenchmarkFlow,
    EpisodeEvidence,
    EventRule,
    TerminationRule,
    assemble_default_report,
)
from unigym_bench.json import ensure_json_value
from unigym_bench.monitors import BaseMonitor, PredicateMonitor
from unigym_bench.report import (
    REPORT_SCHEMA,
    ReportBuilder,
    RunMetadata,
    RunSummary,
    ScenarioCoverageExtension,
    validate_report_extensions,
)
from unigym_bench.sink import BenchmarkSink, make_sink
from unigym_bench.specs import RetentionSpec
from unigym.values import StepValues

from _support import (
    CaptureReportExtension,
    CountingReportExtension,
    LeakyReportViewMonitor,
    ListReportExtension,
    MissingSnapshotExtension,
    NonJsonReportExtension,
    ScriptedMonitor,
    StaticReportExtension,
    StatelessWithSnapshotExtension,
    SummaryFactsReportExtension,
    TickScript,
    UndeclaredExtension,
    ViewReportExtension,
    make_bundle,
    make_sink_context,
    make_termination,
)

_LAYOUT = [
    "schema",
    "run",
    "tasks",
    "episodes",
    "terminations",
    "execution",
    "monitors",
    "extensions",
]

_DEFAULT_RETENTION = {
    "retain_episodes": False,
    "max_events": 200_000,
    "event_log": True,
}


def _metadata(**overrides: object) -> RunMetadata:
    values: dict[str, object] = {
        "run_name": "report-run",
        "task_name": "report-task",
        "out_dir": "data_out/report-run",
        "seed": 3,
        "num_envs": 1,
        "policy_dt": 0.02,
        "loop_mode": "pool",
    }
    values.update(overrides)
    return RunMetadata(**values)


def _zero_execution(**retention: object) -> dict[str, object]:
    caps = dict(_DEFAULT_RETENTION)
    caps.update(retention)
    return {
        "event_total": 0,
        "dropped_events": 0,
        "terminal_conflicts": 0,
        "retention": caps,
    }


def _sink(
    monitors: list[BaseMonitor],
    *,
    extensions: list[object] | None = None,
    retention: RetentionSpec | None = None,
    flow: DefaultBenchmarkFlow | None = None,
) -> tuple[BenchmarkSink, StreamingEngine]:
    engine = StreamingEngine(
        monitors,
        retention=RetentionSpec() if retention is None else retention,
        extensions=() if extensions is None else extensions,  # type: ignore[arg-type]
    )
    return BenchmarkSink(engine=engine, metadata=_metadata(), flow=flow), engine


def test_base_fallback_and_extension_blocks_assemble_the_fixed_layout() -> None:
    monitor = BaseMonitor("contact")
    extension = StaticReportExtension("execution", {"loop_mode": "pool"})
    sink = BenchmarkSink([monitor], extensions=[extension], metadata=_metadata())

    summary = sink.on_run_end()

    assert set(summary) == {"BenchmarkSink"}
    report = summary["BenchmarkSink"]
    assert isinstance(report, dict)
    assert list(report) == _LAYOUT
    assert report["schema"] == REPORT_SCHEMA
    assert report["run"]["run_name"] == "report-run"
    assert report["run"]["task_name"] == "report-task"
    assert "mode" not in report["run"]
    assert report["run"]["seed"] == 3
    assert report["tasks"] == {
        "name": "report-task",
        "monitors": ["contact"],
        "extensions": ["execution"],
    }
    assert report["episodes"] == {
        "count": 0,
        "complete": 0,
        "incomplete": 0,
        "rewound": 0,
        "retained": 0,
        "retain_episodes": False,
        "total_ticks": 0,
        "verdicts": {},
        "by_env_counts": {},
    }
    assert "length" not in report["episodes"]
    assert "by_env" not in report["episodes"]
    assert report["terminations"] == {
        "total": 0,
        "outcome": {},
        "code": {},
        "source": {},
    }
    assert report["execution"] == _zero_execution()
    assert report["monitors"] == {"contact": {"name": "contact", "kind": "base"}}
    assert report["extensions"] == {"execution": {"loop_mode": "pool"}}
    assert ensure_json_value(report) == report
    assert json.loads(json.dumps(report)) == report


def test_monitor_with_declared_extension_is_not_duplicated_under_monitors() -> None:
    monitor = BaseMonitor("contact")
    extension = ViewReportExtension("contact_report")
    builder = ReportBuilder(
        monitors=[monitor],
        extensions=[extension],
        monitor_extensions={"contact": "contact_report"},
    )
    run_summary = RunSummary(
        metadata=_metadata(),
        episodes=(),
        episode_summary=EpisodeSummary(),
        monitor_views={"contact": {"name": "contact", "kind": "base"}},
    )

    report = builder.assemble(run_summary, {"contact_report": extension.on_run_end(run_summary)})

    assert report["monitors"] == {}
    assert report["extensions"] == {"contact_report": {"saw_view": True}}


def test_multi_env_fallback_has_no_implicit_env_axis() -> None:
    monitor = BaseMonitor("per_env")
    run_summary = RunSummary(
        metadata=_metadata(num_envs=2),
        episodes=(),
        episode_summary=EpisodeSummary(episodes_by_env={0: 0, 1: 0}),
        monitor_views={"per_env": {"values": [0.1, 0.2], "axis": "env"}},
    )

    report = ReportBuilder(monitors=[monitor]).assemble(run_summary, {})

    assert report["monitors"] == {"per_env": {"values": [0.1, 0.2], "axis": "env"}}
    assert report["episodes"]["by_env_counts"] == {"0": 0, "1": 0}


def test_zero_episode_report_is_honest() -> None:
    report = ReportBuilder(monitors=[]).assemble(
        RunSummary(
            metadata=_metadata(),
            episodes=(),
            episode_summary=EpisodeSummary(),
            monitor_views={},
        ),
        {},
    )

    assert report["episodes"] == {
        "count": 0,
        "complete": 0,
        "incomplete": 0,
        "rewound": 0,
        "retained": 0,
        "retain_episodes": False,
        "total_ticks": 0,
        "verdicts": {},
        "by_env_counts": {},
    }
    assert report["terminations"] == {
        "total": 0,
        "outcome": {},
        "code": {},
        "source": {},
    }
    assert report["execution"] == _zero_execution()


def test_incomplete_and_rewound_episodes_are_counted_and_filtered() -> None:
    terminal = make_termination("custom.contact.hit")
    incomplete = EpisodeRecord(env_id=0, episode_seq=0, ticks=2, verdict="incomplete")
    rewound = EpisodeRecord(
        env_id=0,
        episode_seq=1,
        ticks=3,
        verdict="terminal",
        first_terminal=terminal,
        rewound=True,
    )
    run_summary = RunSummary(
        metadata=_metadata(),
        episodes=(incomplete, rewound),
        episode_summary=EpisodeSummary(
            episodes=2,
            total_ticks=5,
            verdict_counts={"incomplete": 1, "terminal": 1},
            rewound=1,
            termination_outcome_counts={"success": 1},
            termination_code_counts={"custom.contact.hit": 1},
            termination_source_counts={"custom": 1},
            episodes_by_env={0: 2},
        ),
        monitor_views={},
        retention=RetentionSpec(retain_episodes=True),
    )

    report = ReportBuilder(monitors=[]).assemble(run_summary, {})

    episodes = report["episodes"]
    assert episodes["count"] == 2
    assert episodes["incomplete"] == 1
    assert episodes["rewound"] == 1
    assert episodes["complete"] == 0
    assert episodes["retained"] == 2
    assert episodes["retain_episodes"] is True
    assert episodes["by_env_counts"] == {"0": 2}
    assert episodes["length"] == {"min": 2, "max": 3, "mean": 2.5}
    assert episodes["verdicts"] == {"incomplete": 1, "terminal": 1}
    assert episodes["by_env"]["0"]["count"] == 2
    assert episodes["by_env"]["0"]["rewound"] == 1
    assert report["terminations"] == {
        "total": 1,
        "outcome": {"success": 1},
        "code": {"custom.contact.hit": 1},
        "source": {"custom": 1},
    }


def test_default_retention_terminal_episode_uses_run_counters() -> None:
    monitor = ScriptedMonitor("hit", {0: TickScript(values={"fired": torch.tensor([True])})})
    flow = DefaultBenchmarkFlow((TerminationRule("hit", "fired", "success", "custom.hit.one"),))
    sink, engine = _sink([monitor], flow=flow)
    bundle = make_bundle(num_envs=1, t=0)

    sink.on_step(0, bundle)
    sink.drain_terminations()
    sink.on_episode_end([0], bundle)
    report = sink.on_run_end()["BenchmarkSink"]

    assert engine.records == ()
    assert report["episodes"] == {
        "count": 1,
        "complete": 1,
        "incomplete": 0,
        "rewound": 0,
        "retained": 0,
        "retain_episodes": False,
        "total_ticks": 1,
        "verdicts": {"terminal": 1},
        "by_env_counts": {"0": 1},
    }
    assert "length" not in report["episodes"]
    assert "by_env" not in report["episodes"]
    assert report["terminations"] == {
        "total": 1,
        "outcome": {"success": 1},
        "code": {"custom.hit.one": 1},
        "source": {"custom": 1},
    }
    assert report["execution"] == _zero_execution()


def test_default_retention_core_episode_counts_terminations() -> None:
    monitor = ScriptedMonitor("noop", {})
    sink, engine = _sink([monitor])
    core = Termination(
        outcome=EpisodeOutcome.FAILURE,
        code="env.fallen",
        source=TerminationSource.ENVIRONMENT,
    )
    bundle = make_bundle(
        num_envs=1,
        t=0,
        terminations={0: EpisodeTermination.from_terminations([core])},
    )

    sink.on_step(0, make_bundle(num_envs=1, t=0))
    assert sink.drain_terminations() == {}
    sink.on_episode_end([0], bundle)
    report = sink.on_run_end()["BenchmarkSink"]

    assert engine.records == ()
    assert report["episodes"]["verdicts"] == {"core": 1}
    assert report["episodes"]["complete"] == 1
    assert report["episodes"]["retained"] == 0
    assert report["terminations"] == {
        "total": 1,
        "outcome": {"failure": 1},
        "code": {"env.fallen": 1},
        "source": {"environment": 1},
    }


def test_default_retention_rewound_episode_is_counted_without_records() -> None:
    monitor = ScriptedMonitor(
        "hit",
        {
            0: TickScript(values={"fired": torch.tensor([True])}),
            1: TickScript(values={"fired": torch.tensor([False])}),
        },
    )
    flow = DefaultBenchmarkFlow((TerminationRule("hit", "fired", "success", "custom.hit.one"),))
    sink, engine = _sink([monitor], flow=flow)
    first = make_bundle(num_envs=1, t=0)

    sink.on_step(0, first)
    sink.drain_terminations()
    sink.on_episode_end([0], first)
    sink.on_reset([0])
    snapshot = sink.rewind_provider.rewind_state()

    sink.on_step(1, make_bundle(num_envs=1, t=1))
    sink.rewind_provider.apply_rewind_state(snapshot)
    sink.on_rewind({})
    sink.on_step(1, make_bundle(num_envs=1, t=1))
    report = sink.on_run_end()["BenchmarkSink"]

    assert engine.records == ()
    assert report["episodes"]["count"] == 2
    assert report["episodes"]["incomplete"] == 0
    assert report["episodes"]["rewound"] == 1
    assert report["episodes"]["complete"] == 1
    assert report["episodes"]["verdicts"] == {"rewound": 1, "terminal": 1}
    assert report["episodes"]["by_env_counts"] == {"0": 2}
    assert "length" not in report["episodes"]
    assert "by_env" not in report["episodes"]
    assert report["terminations"]["total"] == 1


def test_default_retention_reports_event_cap_and_terminal_conflicts() -> None:
    monitor = ScriptedMonitor(
        "a",
        {
            0: TickScript(values={"fired": torch.tensor([True])}),
            1: TickScript(values={"fired": torch.tensor([True])}),
        },
    )
    flow = DefaultBenchmarkFlow(
        (TerminationRule("a", "fired", "success", "custom.a.hit"),),
        events=(EventRule("a", "fired", "hit", code="task.a.hit"),),
    )
    sink, engine = _sink(
        [monitor],
        retention=RetentionSpec(max_events=0),
        flow=flow,
    )
    bundle = make_bundle(num_envs=1, t=0)

    sink.on_step(0, bundle)
    assert set(sink.drain_terminations()) == {0}
    sink.on_step(1, make_bundle(num_envs=1, t=1))
    assert sink.drain_terminations() == {}
    sink.on_episode_end([0], bundle)
    report = sink.on_run_end()["BenchmarkSink"]

    assert engine.records == ()
    assert engine.journal == ()
    assert engine.dropped_events == 2
    assert engine.conflict_count == 1
    assert report["execution"] == {
        "event_total": 2,
        "dropped_events": 2,
        "terminal_conflicts": 1,
        "retention": {
            "retain_episodes": False,
            "max_events": 0,
            "event_log": True,
        },
    }
    assert report["terminations"]["total"] == 1


def test_duplicate_names_and_bad_references_fail() -> None:
    with pytest.raises(ValueError, match="duplicate extension name"):
        validate_report_extensions(
            [StaticReportExtension("dup", {}), StaticReportExtension("dup", {})]
        )
    with pytest.raises(ValueError, match="duplicate monitor name"):
        ReportBuilder(monitors=[BaseMonitor("m"), BaseMonitor("m")])
    with pytest.raises(ValueError, match="unknown extension"):
        ReportBuilder(
            monitors=[BaseMonitor("m")],
            monitor_extensions={"m": "missing"},
        )
    with pytest.raises(ValueError, match="unknown monitor"):
        ReportBuilder(
            monitors=[BaseMonitor("m")],
            extensions=[StaticReportExtension("e", {})],
            monitor_extensions={"other": "e"},
        )


def test_extension_stateful_declaration_must_match_capability() -> None:
    with pytest.raises(TypeError, match="stateful=True requires"):
        validate_report_extensions([MissingSnapshotExtension()])
    with pytest.raises(TypeError, match="declares stateful=False"):
        validate_report_extensions([StatelessWithSnapshotExtension()])
    with pytest.raises(TypeError, match="stateful"):
        validate_report_extensions([UndeclaredExtension()])


def test_stateful_extension_snapshot_restore_round_trip() -> None:
    extension = CountingReportExtension()
    extension.on_env_end((0,), {})
    state = extension.snapshot()

    extension.on_env_end((1,), {})
    assert extension.episodes == 2

    extension.restore(state)
    assert extension.episodes == 1
    assert extension.envs == [0]


def test_non_json_extension_block_fails_with_the_field_path() -> None:
    sink = BenchmarkSink([], extensions=[NonJsonReportExtension()], metadata=_metadata())

    with pytest.raises(TypeError, match="extensions.bad"):
        sink.on_run_end()


def test_non_json_monitor_view_fails() -> None:
    sink = BenchmarkSink([LeakyReportViewMonitor("leak")], metadata=_metadata())

    with pytest.raises(TypeError, match="monitor_views"):
        sink.on_run_end()


def test_non_mapping_extension_block_is_rejected() -> None:
    builder = ReportBuilder(monitors=[], extensions=[ListReportExtension()])
    run_summary = RunSummary(
        metadata=_metadata(),
        episodes=(),
        episode_summary=EpisodeSummary(),
        monitor_views={},
    )

    with pytest.raises(TypeError, match="expected a mapping or None"):
        builder.assemble(run_summary, {"listy": [1, 2, 3]})


def test_make_sink_resolves_extension_factory_and_monitor_references() -> None:
    sink = make_sink(
        make_sink_context(
            options={
                "monitor_factory": "_support:make_threshold_monitors",
                "extension_factory": "_support:make_report_extensions",
                "monitor_extensions": {"contact": "execution"},
            }
        )
    )

    report = sink.on_run_end()["BenchmarkSink"]

    assert report["monitors"] == {}
    assert report["extensions"] == {"execution": {"episodes": 0, "env_ends": 0}}


def test_make_sink_rejects_invalid_extension_factory_and_references() -> None:
    with pytest.raises(ConfigError, match="extension_factory"):
        make_sink(make_sink_context(options={"extension_factory": "no_such_pack:make"}))
    with pytest.raises(ConfigError, match="monitor_extensions"):
        make_sink(make_sink_context(options={"monitor_extensions": {"nope": "nope"}}))


def test_stateful_extension_state_round_trips_through_the_rewind_provider() -> None:
    monitor = ScriptedMonitor(
        "hit",
        {
            0: TickScript(values={"fired": torch.tensor([False])}),
            1: TickScript(values={"fired": torch.tensor([True])}),
            2: TickScript(values={"fired": torch.tensor([True])}),
        },
    )
    flow = DefaultBenchmarkFlow((TerminationRule("hit", "fired", "success", "custom.hit.one"),))
    extension = CountingReportExtension()
    sink, engine = _sink(
        [monitor],
        extensions=[extension],
        retention=RetentionSpec(retain_episodes=True),
        flow=flow,
    )

    sink.on_step(0, make_bundle(num_envs=1, t=0))
    first = make_bundle(num_envs=1, t=1)
    sink.on_step(1, first)
    sink.drain_terminations()
    sink.on_episode_end([0], first)
    sink.on_reset([0])
    assert extension.episodes == 1

    snapshot = sink.rewind_provider.rewind_state()
    assert snapshot["extensions"]["counts"] == {
        "episodes": 1,
        "envs": [0],
        "run_end_calls": 0,
    }
    sink.rewind_provider.validate_rewind_state(snapshot)

    second = make_bundle(num_envs=1, t=2)
    sink.on_step(2, second)
    sink.drain_terminations()
    sink.on_episode_end([0], second)
    assert extension.episodes == 2

    sink.rewind_provider.apply_rewind_state(snapshot)
    assert extension.episodes == 1
    assert extension.envs == [0]
    assert engine.summary.episodes == 1
    assert dict(engine.summary.episodes_by_env) == {0: 1}
    assert dict(engine.summary.termination_code_counts) == {"custom.hit.one": 1}

    replay = make_bundle(num_envs=1, t=2)
    sink.on_step(2, replay)
    sink.drain_terminations()
    sink.on_episode_end([0], replay)
    assert extension.episodes == 2
    assert extension.envs == [0, 0]
    assert engine.summary.episodes == 2


def test_scenario_coverage_extension_uses_record_start_commands_and_rewinds() -> None:
    extension = ScenarioCoverageExtension("coverage", commands=[[0.0], [1.0]])
    monitor = ScriptedMonitor(
        "hit",
        {0: TickScript(values={"fired": torch.tensor([True, False])})},
    )
    sink, engine = _sink(
        [monitor],
        extensions=[extension],
        retention=RetentionSpec(retain_episodes=True),
    )

    first = make_bundle(num_envs=2, t=0, cmd=torch.tensor([[0.0], [2.0]]))
    sink.on_step(0, first)
    sink.drain_terminations()
    sink.on_episode_end([0], first)
    sink.on_reset([0])

    state = sink.rewind_provider.rewind_state()
    assert state["extensions"]["coverage"] == {
        "schema": "unigym_bench.scenario_coverage/1",
        "name": "coverage",
        "commands": [[0.0], [1.0]],
        "covered": [True, False],
        "episode_counts": [1, 0],
        "unmatched": 0,
        "episodes": 1,
    }
    sink.rewind_provider.validate_rewind_state(state)

    second = make_bundle(num_envs=2, t=1, cmd=torch.tensor([[0.0], [2.0]]))
    sink.on_step(1, second)
    sink.on_episode_end([1], second)
    live = sink.rewind_provider.rewind_state()["extensions"]["coverage"]
    assert live["episodes"] == 2
    assert live["unmatched"] == 1

    sink.rewind_provider.apply_rewind_state(state)
    restored = sink.rewind_provider.rewind_state()["extensions"]["coverage"]
    assert restored["episodes"] == 1
    assert restored["unmatched"] == 0

    report = sink.on_run_end()["BenchmarkSink"]
    assert isinstance(report, dict)
    assert report["extensions"]["coverage"] == {
        "name": "coverage",
        "kind": "scenario_coverage",
        "pool_size": 2,
        "covered": 1,
        "coverage_ratio": 0.5,
        "episode_counts": [1, 0],
        "unmatched_episodes": 0,
        "episodes": 1,
    }
    assert report["tasks"]["extensions"] == ["coverage"]


def test_rewind_rejects_missing_extension_state_on_a_plain_engine() -> None:
    source_monitor = ScriptedMonitor(
        "hit",
        {0: TickScript(values={"fired": torch.tensor([True])})},
    )
    source_flow = DefaultBenchmarkFlow(
        (TerminationRule("hit", "fired", "success", "custom.hit.one"),)
    )
    source = StreamingEngine(
        [source_monitor],
        retention=RetentionSpec(retain_episodes=True),
        extensions=[CountingReportExtension()],
    )
    source_sink = BenchmarkSink(engine=source, metadata=_metadata(), flow=source_flow)
    bundle = make_bundle(num_envs=1, t=0)
    source_sink.on_step(0, bundle)
    source_sink.drain_terminations()
    source_sink.on_episode_end([0], bundle)
    snapshot = source_sink.rewind_provider.rewind_state()

    plain = StreamingEngine(
        [ScriptedMonitor("hit", {})],
        retention=RetentionSpec(retain_episodes=True),
    )
    plain_sink = BenchmarkSink(engine=plain, metadata=_metadata())
    plain_sink.on_step(0, make_bundle(num_envs=1, t=0))

    with pytest.raises(ValueError, match="checkpoint extension names"):
        plain_sink.rewind_provider.validate_rewind_state(snapshot)
    with pytest.raises(ValueError, match="checkpoint extension names"):
        plain_sink.rewind_provider.apply_rewind_state(snapshot)


def test_stateless_extensions_do_not_add_rewind_state() -> None:
    sink = BenchmarkSink(
        [BaseMonitor("m")],
        extensions=[StaticReportExtension("execution", {})],
        metadata=_metadata(),
    )
    sink.on_step(0, make_bundle(num_envs=1, t=0))

    state = sink.rewind_provider.rewind_state()

    assert "extensions" not in state


def test_report_builder_rejects_unknown_extension_block_names() -> None:
    builder = ReportBuilder(monitors=[], extensions=[StaticReportExtension("execution", {})])
    run_summary = RunSummary(
        metadata=_metadata(),
        episodes=(),
        episode_summary=EpisodeSummary(),
        monitor_views={},
    )

    with pytest.raises(ValueError, match="missing"):
        builder.assemble(run_summary, {})
    with pytest.raises(ValueError, match="extra"):
        builder.assemble(run_summary, {"execution": {}, "other": {}})


def test_run_identity_is_reported_when_present() -> None:
    identity = RunIdentity(
        backend="mujoco",
        device="cpu",
        num_envs=1,
        seed=5,
        policy_dt=0.02,
        sim_dt=0.002,
        articulations=(RobotIdentity(name="arm", profile="tiny"),),
    )
    sink = BenchmarkSink([BaseMonitor("m")], metadata=_metadata(run_identity=identity))

    report = sink.on_run_end()["BenchmarkSink"]

    assert report["run"]["identity"] == identity.to_json()
    assert ensure_json_value(report) == report


def test_open_episode_at_run_end_is_visible_through_the_summary() -> None:
    sink, engine = _sink(
        [BaseMonitor("m")],
        extensions=[SummaryFactsReportExtension()],
        retention=RetentionSpec(retain_episodes=True),
    )
    sink.on_step(0, make_bundle(num_envs=1, t=0))

    report = sink.on_run_end()["BenchmarkSink"]

    assert engine.records[0].verdict == "incomplete"
    assert report["episodes"]["verdicts"] == {"incomplete": 1}
    assert report["extensions"]["summary_facts"] == {
        "episodes": 1,
        "verdicts": {"incomplete": 1},
        "retain_episodes": True,
    }


def test_on_run_end_is_idempotent_and_calls_extensions_once() -> None:
    extension = CountingReportExtension()
    sink, _ = _sink(
        [BaseMonitor("m")],
        extensions=[extension],
        retention=RetentionSpec(retain_episodes=True),
    )
    sink.on_step(0, make_bundle(num_envs=1, t=0))

    first = sink.on_run_end()
    second = sink.on_run_end()

    assert first == second
    assert extension.run_end_calls == 1


def test_on_env_end_receives_only_due_envs_and_frozen_records() -> None:
    monitor = ScriptedMonitor(
        "hit",
        {0: TickScript(values={"fired": torch.tensor([False, True])})},
    )
    flow = DefaultBenchmarkFlow((TerminationRule("hit", "fired", "success", "custom.hit.one"),))
    calls: list[tuple[tuple[int, ...], object]] = []
    capture = CaptureReportExtension(calls)
    sink, _ = _sink(
        [monitor],
        extensions=[capture],
        retention=RetentionSpec(retain_episodes=True),
        flow=flow,
    )
    bundle = make_bundle(num_envs=2, t=0)

    sink.on_step(0, bundle)
    assert set(sink.drain_terminations()) == {1}
    sink.on_episode_end([1], bundle)

    assert len(calls) == 1
    env_ids, records = calls[0]
    assert env_ids == (1,)
    assert set(records) == {1}
    assert records[1].env_id == 1
    with pytest.raises(TypeError):
        records[0] = records[1]  # type: ignore[index]
    with pytest.raises(AttributeError):
        records[1].ticks = 9  # type: ignore[misc]


def test_on_run_end_writes_no_core_report_artifact(tmp_path: Path) -> None:
    sink = BenchmarkSink([BaseMonitor("m")], metadata=_metadata(out_dir=str(tmp_path)))

    report = sink.on_run_end()["BenchmarkSink"]

    assert report["run"]["out_dir"] == str(tmp_path)
    assert report["execution"] == _zero_execution()
    assert list(tmp_path.iterdir()) == []
    assert not (tmp_path / "benchmark_report.json").exists()
    assert not (tmp_path / "results.json").exists()


def test_final_episode_value_is_not_counted_as_a_termination() -> None:
    monitor = PredicateMonitor(
        "end_pred",
        {"source": "value:flag", "op": "gt", "value": 0.0},
        "episode_end",
    )
    sink, engine = _sink([monitor])
    bundle = make_bundle(
        num_envs=1,
        t=0,
        values=StepValues({"flag": torch.tensor([0.0])}),
        final_values=StepValues({"flag": torch.tensor([1.0])}),
    )

    sink.on_step(0, bundle)
    sink.on_episode_end([0], bundle)
    report = sink.on_run_end()["BenchmarkSink"]

    assert isinstance(report, dict)
    assert report["terminations"] == {
        "total": 0,
        "outcome": {},
        "code": {},
        "source": {},
    }
    assert report["execution"]["terminal_conflicts"] == 0
    assert engine.drain_terminations() == {}
    assert monitor.report_view()["episode_end_fired"] == 1


class _EvidenceFlow:
    """Minimal flow that returns fixed evidence for the due envs."""

    name = "evidence"
    stateful = False

    def on_episode_end(self, batch: object) -> EpisodeEvidence:
        """Return one evidence row per due env."""
        due = getattr(batch, "due")
        return EpisodeEvidence({env_id: {"score": float(env_id) + 0.5} for env_id in due})

    def on_run_end(self, run: object) -> dict[str, object]:
        """Use the default report so evidence is visible."""
        return dict(assemble_default_report(run))  # type: ignore[arg-type]


class _EmptyEvidenceFlow:
    """Flow that returns an empty evidence object for each due env."""

    name = "empty_evidence"
    stateful = False

    def on_episode_end(self, batch: object) -> EpisodeEvidence:
        """Return an empty payload per due env."""
        due = getattr(batch, "due")
        return EpisodeEvidence({env_id: {} for env_id in due})

    def on_run_end(self, run: object) -> dict[str, object]:
        """Use the default report so evidence filtering is observable."""
        return dict(assemble_default_report(run))  # type: ignore[arg-type]


class _OldSchemaFlow:
    """Flow that returns the retired report schema id."""

    name = "old_schema"
    stateful = False

    def on_episode_end(self, batch: object) -> None:
        """No evidence."""
        del batch

    def on_run_end(self, run: object) -> dict[str, object]:
        """Return a valid report with the previous schema id."""
        report = dict(assemble_default_report(run))  # type: ignore[arg-type]
        report["schema"] = "external_benchmark_report/2"
        return report


class _ExtraBlockFlow:
    """Flow that tries to add a top-level block."""

    name = "extra_block"
    stateful = False

    def on_episode_end(self, batch: object) -> None:
        """No evidence."""
        del batch

    def on_run_end(self, run: object) -> dict[str, object]:
        """Return a valid report plus one illegal top-level field."""
        report = dict(assemble_default_report(run))  # type: ignore[arg-type]
        report["extra"] = {}
        return report


class _OrderExtension:
    """Stateless extension that records its boundary/run-end order."""

    stateful = False

    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self._calls = calls

    def on_env_end(self, env_ids: object, records: object) -> None:
        """Record the boundary call."""
        del env_ids, records
        self._calls.append(f"{self.name}.env_end")

    def on_run_end(self, summary: object) -> Mapping[str, object]:
        """Record the run-end call and return a small block."""
        del summary
        self._calls.append(f"{self.name}.run_end")
        return {"order": len(self._calls)}


class _FailOnceExtension:
    """Stateless extension whose first run-end attempt raises."""

    stateful = False
    name = "fail_once"

    def __init__(self) -> None:
        self.run_end_calls = 0

    def on_env_end(self, env_ids: object, records: object) -> None:
        """No accumulator."""
        del env_ids, records

    def on_run_end(self, summary: object) -> Mapping[str, object]:
        """Fail on the first attempt, then report the attempt count."""
        del summary
        self.run_end_calls += 1
        if self.run_end_calls == 1:
            raise RuntimeError("first report attempt fails")
        return {"calls": self.run_end_calls}


class _OutOfOrderFlow:
    """Flow that returns the default report blocks in reverse order."""

    name = "out_of_order"
    stateful = False

    def on_episode_end(self, batch: object) -> None:
        """No evidence."""
        del batch

    def on_run_end(self, run: object) -> dict[str, object]:
        """Return the same blocks with a broken top-level order."""
        report = dict(assemble_default_report(run))  # type: ignore[arg-type]
        return {key: report[key] for key in reversed(tuple(report))}


def test_report_schema_is_3_and_has_no_evidence_key_by_default() -> None:
    sink = BenchmarkSink([BaseMonitor("m")], metadata=_metadata())
    report = sink.on_run_end()["BenchmarkSink"]

    assert report["schema"] == "external_benchmark_report/3"
    assert REPORT_SCHEMA == "external_benchmark_report/3"
    assert "evidence" not in report["episodes"]


def test_monitor_report_blocks_carry_no_terminal_fields() -> None:
    monitor = PredicateMonitor(
        "pred",
        {"source": "value:flag", "op": "gt", "value": 0.0},
        "tick",
    )
    sink = BenchmarkSink([monitor], metadata=_metadata())
    bundle = make_bundle(
        num_envs=1,
        t=0,
        values=StepValues({"flag": torch.tensor([1.0])}),
    )

    sink.on_step(0, bundle)
    block = sink.on_run_end()["BenchmarkSink"]["monitors"]["pred"]

    assert {"terminal_reason", "terminal_outcome"}.isdisjoint(block)


def test_sink_rejects_the_previous_report_schema() -> None:
    sink = BenchmarkSink([BaseMonitor("m")], flow=_OldSchemaFlow(), metadata=_metadata())

    with pytest.raises(ValueError, match="external_benchmark_report/3"):
        sink.on_run_end()


def test_sink_rejects_an_extra_top_level_report_block() -> None:
    sink = BenchmarkSink([BaseMonitor("m")], flow=_ExtraBlockFlow(), metadata=_metadata())

    with pytest.raises(ValueError, match="exactly the top-level blocks"):
        sink.on_run_end()


def test_sink_rejects_out_of_order_top_level_report_blocks() -> None:
    sink = BenchmarkSink([BaseMonitor("m")], flow=_OutOfOrderFlow(), metadata=_metadata())

    with pytest.raises(ValueError, match="out of order"):
        sink.on_run_end()


def test_default_flow_rejects_dropping_a_monitor_declared_extension() -> None:
    sink = BenchmarkSink(
        [BaseMonitor("contact")],
        extensions=[ViewReportExtension("contact_report")],
        flow=DefaultBenchmarkFlow(report_extensions=()),
        metadata=_metadata(),
        monitor_extensions={"contact": "contact_report"},
    )

    with pytest.raises(ValueError, match="contact -> contact_report"):
        sink.on_run_end()

    both_omitted = BenchmarkSink(
        [BaseMonitor("contact")],
        extensions=[ViewReportExtension("contact_report")],
        flow=DefaultBenchmarkFlow(report_monitor_views=(), report_extensions=()),
        metadata=_metadata(),
        monitor_extensions={"contact": "contact_report"},
    )
    report = both_omitted.on_run_end()["BenchmarkSink"]
    assert report["monitors"] == {}
    assert report["extensions"] == {}


def test_failed_report_attempt_retries_and_reinvokes_extension_on_run_end() -> None:
    extension = _FailOnceExtension()
    sink, _engine = _sink([BaseMonitor("m")], extensions=[extension])

    with pytest.raises(RuntimeError, match="first report attempt fails"):
        sink.on_run_end()

    report = sink.on_run_end()["BenchmarkSink"]
    assert extension.run_end_calls == 2
    assert report["extensions"]["fail_once"] == {"calls": 2}

    sink.on_run_end()
    assert extension.run_end_calls == 2


def test_missing_or_omitted_monitor_view_is_not_an_error() -> None:
    builder = ReportBuilder(monitors=[BaseMonitor("a"), BaseMonitor("b")])
    summary = RunSummary(
        metadata=_metadata(),
        episodes=(),
        episode_summary=EpisodeSummary(),
        monitor_views={"a": {"name": "a", "kind": "base"}},
    )

    report = builder.assemble(summary, {})

    assert report["monitors"] == {"a": {"name": "a", "kind": "base"}}
    assert report["tasks"]["monitors"] == ["a", "b"]


def test_default_flow_filters_monitor_views_and_extensions() -> None:
    calls: list[str] = []
    first = _OrderExtension("first", calls)
    second = _OrderExtension("second", calls)
    flow = DefaultBenchmarkFlow(
        report_monitor_views=frozenset({"kept"}),
        report_extensions=frozenset({"second"}),
    )
    sink, engine = _sink(
        [BaseMonitor("kept"), BaseMonitor("omitted")],
        extensions=[first, second],
        flow=flow,
    )
    bundle = make_bundle(num_envs=1, t=0)

    sink.on_step(0, bundle)
    sink.on_episode_end([0], bundle)
    report = sink.on_run_end()["BenchmarkSink"]

    assert engine.extensions == (first, second)
    assert calls == ["second.env_end", "second.run_end"]
    assert report["monitors"] == {"kept": {"name": "kept", "kind": "base"}}
    assert report["tasks"]["monitors"] == ["kept", "omitted"]
    assert report["extensions"] == {"second": {"order": 2}}
    assert report["tasks"]["extensions"] == ["first", "second"]


def test_default_flow_invokes_every_extension_once_in_declaration_order() -> None:
    calls: list[str] = []
    first = _OrderExtension("first", calls)
    second = _OrderExtension("second", calls)
    sink, _engine = _sink([BaseMonitor("m")], extensions=[first, second])
    bundle = make_bundle(num_envs=1, t=0)

    sink.on_step(0, bundle)
    sink.on_episode_end([0], bundle)
    sink.on_run_end()
    sink.on_run_end()

    assert calls == [
        "first.env_end",
        "second.env_end",
        "first.run_end",
        "second.run_end",
    ]


def test_default_flow_rejects_unknown_report_selections() -> None:
    monitors = [BaseMonitor("m")]
    extensions = [StaticReportExtension("ext", {})]
    unknown_view, _ = _sink(
        monitors,
        extensions=extensions,
        flow=DefaultBenchmarkFlow(report_monitor_views=("nope",)),
    )
    unknown_extension, _ = _sink(
        monitors,
        extensions=extensions,
        flow=DefaultBenchmarkFlow(report_extensions=("nope",)),
    )

    with pytest.raises(ValueError, match="unknown monitor"):
        unknown_view.on_run_end()
    with pytest.raises(ValueError, match="unknown extension"):
        unknown_extension.on_run_end()


def test_episode_evidence_is_reported_at_the_fixed_location() -> None:
    sink = BenchmarkSink(
        [BaseMonitor("m")],
        flow=_EvidenceFlow(),  # type: ignore[arg-type]
        metadata=_metadata(),
    )
    bundle = make_bundle(num_envs=1, t=0)

    sink.on_step(0, bundle)
    sink.on_episode_end([0], bundle)
    report = sink.on_run_end()["BenchmarkSink"]

    assert report["episodes"]["evidence"] == {"0/0": {"score": 0.5}}
    assert "evidence" not in report


def test_empty_evidence_entries_are_omitted() -> None:
    sink = BenchmarkSink(
        [BaseMonitor("m")],
        flow=_EmptyEvidenceFlow(),  # type: ignore[arg-type]
        metadata=_metadata(),
    )
    bundle = make_bundle(num_envs=1, t=0)

    sink.on_step(0, bundle)
    sink.on_episode_end([0], bundle)
    report = sink.on_run_end()["BenchmarkSink"]

    assert "evidence" not in report["episodes"]
    assert sink.rewind_provider.evidence == {}


def test_non_json_evidence_fails_with_its_path() -> None:
    with pytest.raises(TypeError, match=r"episode evidence\[2\]"):
        EpisodeEvidence({2: {"tensor": torch.tensor(1.0)}})


def test_engine_does_not_invoke_extension_on_env_end() -> None:
    extension = CountingReportExtension()
    monitor = ScriptedMonitor("idle", {})
    engine = StreamingEngine([monitor], extensions=[extension])
    bundle = make_bundle(num_envs=1, t=0)

    engine.on_step(bundle)
    batch = engine.begin_episode_end([0], bundle)
    assert batch is not None
    engine.commit_episode_end(batch, None)

    assert extension.episodes == 0
    assert extension.run_end_calls == 0
