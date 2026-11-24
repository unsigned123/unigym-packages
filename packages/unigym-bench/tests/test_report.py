"""Report builder, extension lifecycle, fallback and pack-owned artifacts."""

from __future__ import annotations

import json
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
from unigym_bench.events import EpisodeRecord, EpisodeSummary, Event
from unigym_bench.json import ensure_json_value
from unigym_bench.monitors import BaseMonitor, PredicateMonitor
from unigym_bench.report import (
    REPORT_SCHEMA,
    ReportBuilder,
    RunMetadata,
    RunSummary,
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
    ViewProbeReportExtension,
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
        "mode": "benchmark",
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
) -> tuple[BenchmarkSink, StreamingEngine]:
    engine = StreamingEngine(
        monitors,
        retention=RetentionSpec() if retention is None else retention,
        extensions=() if extensions is None else extensions,  # type: ignore[arg-type]
    )
    return BenchmarkSink(engine=engine, metadata=_metadata()), engine


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
    extension = ViewProbeReportExtension("contact_report")
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
    monitor = ScriptedMonitor(
        "hit", {0: TickScript(terminals={0: make_termination("custom.hit.one")})}
    )
    sink, engine = _sink([monitor])
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
        "hit", {0: TickScript(terminals={0: make_termination("custom.hit.one")})}
    )
    sink, engine = _sink([monitor])
    first = make_bundle(num_envs=1, t=0)

    sink.on_step(0, first)
    sink.drain_terminations()
    sink.on_episode_end([0], first)
    sink.on_step_end(0, first)
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
    first = ScriptedMonitor(
        "a",
        {
            0: TickScript(
                events=(Event(name="hit", code="task.a.hit", env_ids=(0,)),),
                terminals={0: make_termination("custom.a.hit")},
            )
        },
    )
    second = ScriptedMonitor("b", {0: TickScript(terminals={0: make_termination("custom.b.hit")})})
    sink, engine = _sink([first, second], retention=RetentionSpec(max_events=0))
    bundle = make_bundle(num_envs=1, t=0)

    sink.on_step(0, bundle)
    assert set(sink.drain_terminations()) == {0}
    sink.on_episode_end([0], bundle)
    report = sink.on_run_end()["BenchmarkSink"]

    assert engine.records == ()
    assert engine.journal == ()
    assert engine.dropped_events == 1
    assert engine.conflict_count == 1
    assert report["execution"] == {
        "event_total": 1,
        "dropped_events": 1,
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
            1: TickScript(terminals={0: make_termination("custom.hit.one")}),
            2: TickScript(terminals={0: make_termination("custom.hit.two")}),
        },
    )
    extension = CountingReportExtension()
    sink, engine = _sink(
        [monitor],
        extensions=[extension],
        retention=RetentionSpec(retain_episodes=True),
    )

    sink.on_step(0, make_bundle(num_envs=1, t=0))
    first = make_bundle(num_envs=1, t=1)
    sink.on_step(1, first)
    sink.drain_terminations()
    sink.on_episode_end([0], first)
    sink.on_step_end(1, first)
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


def test_rewind_rejects_missing_extension_state_on_a_plain_engine() -> None:
    source_monitor = ScriptedMonitor(
        "hit",
        {0: TickScript(terminals={0: make_termination("custom.hit.one")})},
    )
    source = StreamingEngine(
        [source_monitor],
        retention=RetentionSpec(retain_episodes=True),
        extensions=[CountingReportExtension()],
    )
    source_sink = BenchmarkSink(engine=source, metadata=_metadata())
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
        {0: TickScript(terminals={1: make_termination("custom.hit.one")})},
    )
    calls: list[tuple[tuple[int, ...], object]] = []
    capture = CaptureReportExtension(calls)
    sink, _ = _sink(
        [monitor],
        extensions=[capture],
        retention=RetentionSpec(retain_episodes=True),
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


def test_final_episode_end_terminal_is_counted_in_the_report() -> None:
    monitor = PredicateMonitor(
        "end_pred",
        {"source": "value:flag", "op": "gt", "value": 0.0},
        "episode_end",
        {"terminal_reason": "score", "terminal_outcome": "success"},
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
        "total": 1,
        "outcome": {"success": 1},
        "code": {"custom.end_pred.score": 1},
        "source": {"custom": 1},
    }
    assert report["execution"]["terminal_conflicts"] == 0
    assert engine.drain_terminations() == {}
