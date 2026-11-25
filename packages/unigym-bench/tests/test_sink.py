"""Sink shell, construction seam and immutable episode-record contract."""

from __future__ import annotations

import json
import os

import pytest
import torch
from unigym.core.episode_termination import (
    EpisodeOutcome,
    EpisodeTermination,
    Termination,
    TerminationSource,
)
from unigym.errors import ConfigError
from unigym.values import ValueStore

import unigym_bench.sink as sink_module
from unigym_bench.engine import StreamingEngine
from unigym_bench.events import EpisodeRecord
from unigym_bench.flow import DefaultBenchmarkFlow, EventRule, TerminationRule
from unigym_bench.json import ensure_json_value
from unigym_bench.report import REPORT_FILENAME, REPORT_SCHEMA, RunMetadata
from unigym_bench.sink import BenchmarkSink, make_sink
from unigym_bench.specs import RetentionSpec
from _support import (
    CountingReportExtension,
    ScriptedMonitor,
    TickScript,
    make_bundle,
    make_sink_context,
)

_FACTORY = "_support:make_threshold_monitors"


def _engine_sink(
    monitor: ScriptedMonitor,
    flow: DefaultBenchmarkFlow | None = None,
) -> tuple[BenchmarkSink, StreamingEngine]:
    """Build one sink over a real engine, retaining episode records."""
    engine = StreamingEngine([monitor], retention=RetentionSpec(retain_episodes=True))
    return BenchmarkSink(engine=engine, flow=flow), engine


def test_make_sink_builds_from_a_real_context_and_monitor_factory() -> None:
    sink = make_sink(make_sink_context(options={"monitor_factory": _FACTORY}))

    assert isinstance(sink, BenchmarkSink)
    assert sink.loop_mode == "pool"
    assert sink.stop_when() is False
    assert sink.rewind_provider is not None
    assert sink.rewind_provider.name == "unigym_bench"


def test_make_sink_accepts_sink_steered_with_a_publisher() -> None:
    store = ValueStore(num_envs=1, device="cpu")
    publisher = store.publisher("sink")
    sink = make_sink(
        make_sink_context(
            options={"loop_mode": "sink-steered", "monitor_factory": _FACTORY},
            values=publisher,
        )
    )

    assert sink.loop_mode == "sink-steered"
    assert sink.rewind_provider.publisher is publisher


def test_make_sink_rejects_sink_steered_without_values() -> None:
    with pytest.raises(ConfigError, match="requires ctx.values"):
        make_sink(make_sink_context(options={"loop_mode": "sink-steered"}))


def test_make_sink_rejects_unknown_keys_and_loop_modes() -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        make_sink(make_sink_context(options={"typo": 1}))
    with pytest.raises(ConfigError, match="loop_mode"):
        make_sink(make_sink_context(options={"loop_mode": "closed"}))


def test_make_sink_write_report_is_opt_in_and_typed() -> None:
    sink = make_sink(make_sink_context(options={"write_report": True}))

    assert sink.report_path is not None
    assert str(sink.report_path) == f"data_out/sink-context/{REPORT_FILENAME}"
    with pytest.raises(ConfigError, match="write_report"):
        make_sink(make_sink_context(options={"write_report": 1}))


def test_benchmark_sink_write_report_requires_construction_out_dir() -> None:
    with pytest.raises(ConfigError, match="write_report"):
        BenchmarkSink([], write_report=True)


def test_write_report_defaults_off_and_true_writes(tmp_path) -> None:
    out_dir = tmp_path / "nested" / "run"
    metadata = RunMetadata(out_dir=str(out_dir))

    off = BenchmarkSink([], metadata=metadata)
    assert off.report_path is None
    off_report = off.on_run_end()["BenchmarkSink"]
    assert not out_dir.exists()

    on = BenchmarkSink([], metadata=metadata, write_report=True)
    assert on.report_path == out_dir / REPORT_FILENAME
    on_report = on.on_run_end()["BenchmarkSink"]
    assert on_report == off_report
    written = json.loads((out_dir / REPORT_FILENAME).read_text(encoding="utf-8"))
    assert written == on_report


def test_write_failure_is_raised_and_retried_without_reassembling(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "run"
    extension = CountingReportExtension()
    sink = BenchmarkSink(
        [],
        extensions=(extension,),
        metadata=RunMetadata(out_dir=str(out_dir)),
        write_report=True,
    )
    real_replace = os.replace
    attempts = {"count": 0}

    def flaky_replace(source, destination) -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise OSError("simulated write failure")
        real_replace(source, destination)

    monkeypatch.setattr(sink_module.os, "replace", flaky_replace)

    with pytest.raises(OSError, match="simulated write failure"):
        sink.on_run_end()
    assert extension.run_end_calls == 1
    assert not (out_dir / REPORT_FILENAME).exists()
    assert list(out_dir.glob(f".{REPORT_FILENAME}.*.tmp")) == []

    report = sink.on_run_end()["BenchmarkSink"]
    assert extension.run_end_calls == 1
    assert attempts["count"] == 2
    written = json.loads((out_dir / REPORT_FILENAME).read_text(encoding="utf-8"))
    assert written == report


def test_make_sink_accepts_auto_reset_false_with_explicit_reset_notification() -> None:
    """The sink observes resets directly, so it no longer rejects auto_reset=false."""
    sink = make_sink(make_sink_context(auto_reset=False))

    assert isinstance(sink, BenchmarkSink)


@pytest.mark.parametrize(
    "factory",
    [
        123,
        "not_a_module:make",
        "no_such_pack_module:make",
        "_support:MissingFactory",
    ],
)
def test_make_sink_rejects_invalid_monitor_factory_paths(factory: object) -> None:
    with pytest.raises(ConfigError, match="monitor_factory"):
        make_sink(make_sink_context(options={"monitor_factory": factory}))


def test_make_sink_engine_rejects_a_non_sequence_factory_result() -> None:
    with pytest.raises(TypeError, match="expected a sequence"):
        make_sink(make_sink_context(options={"monitor_factory": "_support:make_bundle"}))


def test_make_sink_rejects_a_non_callable_factory_target() -> None:
    with pytest.raises(ConfigError, match="is not callable"):
        make_sink(make_sink_context(options={"monitor_factory": "_support:_SINK_CONTEXT_CFG"}))


def test_make_sink_without_a_factory_builds_an_empty_honest_run() -> None:
    sink = make_sink(make_sink_context())

    summary = sink.on_run_end()

    assert set(summary) == {"BenchmarkSink"}
    report = summary["BenchmarkSink"]
    assert isinstance(report, dict)
    assert report["schema"] == REPORT_SCHEMA
    assert list(report) == [
        "schema",
        "run",
        "tasks",
        "episodes",
        "terminations",
        "execution",
        "monitors",
        "extensions",
    ]
    assert report["run"]["run_name"] == "sink-context"
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
    assert report["execution"] == {
        "event_total": 0,
        "dropped_events": 0,
        "terminal_conflicts": 0,
        "retention": {
            "retain_episodes": False,
            "max_events": 200_000,
            "event_log": True,
        },
    }
    assert report["monitors"] == {}
    assert report["extensions"] == {}


def test_direct_sink_steered_construction_requires_a_publisher() -> None:
    with pytest.raises(ConfigError, match="requires a value publisher"):
        BenchmarkSink([], loop_mode="sink-steered")

    store = ValueStore(num_envs=1, device="cpu")
    sink = BenchmarkSink([], loop_mode="sink-steered", publisher=store.publisher("sink"))
    assert sink.loop_mode == "sink-steered"


def test_injected_engine_sink_steered_requires_an_engine_publisher() -> None:
    with pytest.raises(ConfigError, match="requires a value publisher"):
        BenchmarkSink(engine=StreamingEngine([]), loop_mode="sink-steered")

    store = ValueStore(num_envs=1, device="cpu")
    engine = StreamingEngine([], publisher=store.publisher("sink"))
    sink = BenchmarkSink(engine=engine, loop_mode="sink-steered")
    assert sink.loop_mode == "sink-steered"


def test_direct_construction_rejects_engine_and_monitor_conflicts() -> None:
    engine = StreamingEngine([])

    with pytest.raises(TypeError, match="not both"):
        BenchmarkSink([], engine=engine)
    with pytest.raises(TypeError, match="owns retention and publisher"):
        BenchmarkSink(engine=engine, retention=RetentionSpec())


def test_sink_delegates_the_terminal_drain_and_reopens_on_reset() -> None:
    monitor = ScriptedMonitor(
        "contact",
        {0: TickScript(values={"fired": torch.tensor([True])})},
    )
    flow = DefaultBenchmarkFlow(
        (TerminationRule("contact", "fired", "success", "custom.contact.hit"),)
    )
    sink, engine = _engine_sink(monitor, flow)
    bundle = make_bundle(num_envs=1, t=0)

    sink.on_step(0, bundle)
    pending = sink.drain_terminations()
    assert set(pending) == {0}
    assert pending[0].code == "custom.contact.hit"

    sink.on_episode_end([0], bundle)
    assert engine.open_envs == ()
    sink.on_step_end(0, bundle)
    assert engine.open_envs == ()
    sink.on_reset([0])
    assert engine.open_envs == (0,)
    assert engine.records[0].episode_seq == 0


def test_on_reset_before_the_first_tick_is_a_safe_noop() -> None:
    sink = BenchmarkSink([])

    sink.on_reset([0])

    report = sink.on_run_end()["BenchmarkSink"]
    assert report["episodes"]["count"] == 0
    assert report["terminations"]["total"] == 0


def test_on_run_end_immediately_after_reset_skips_the_zero_tick_reopened_env() -> None:
    monitor = ScriptedMonitor(
        "contact",
        {0: TickScript(values={"fired": torch.tensor([True])})},
    )
    flow = DefaultBenchmarkFlow(
        (TerminationRule("contact", "fired", "success", "custom.contact.hit"),)
    )
    sink, engine = _engine_sink(monitor, flow)
    bundle = make_bundle(num_envs=1, t=0)

    sink.on_step(0, bundle)
    sink.drain_terminations()
    sink.on_episode_end([0], bundle)
    sink.on_reset([0])
    assert engine.open_envs == (0,)

    summary = sink.on_run_end()

    assert summary["BenchmarkSink"]["episodes"]["count"] == 1
    assert summary["BenchmarkSink"]["episodes"]["verdicts"] == {"terminal": 1}
    assert summary["BenchmarkSink"]["episodes"]["retained"] == 1
    assert summary["BenchmarkSink"]["episodes"]["retain_episodes"] is True
    assert summary["BenchmarkSink"]["episodes"]["by_env_counts"] == {"0": 1}
    assert summary["BenchmarkSink"]["execution"] == {
        "event_total": 0,
        "dropped_events": 0,
        "terminal_conflicts": 0,
        "retention": {
            "retain_episodes": True,
            "max_events": 200_000,
            "event_log": True,
        },
    }
    assert len(engine.records) == 1
    assert engine.records[0].verdict == "terminal"
    assert engine.open_envs == ()


def test_reopened_episode_captures_new_command_and_reset_facts() -> None:
    monitor = ScriptedMonitor(
        "reopen",
        {
            0: TickScript(values={"fired": torch.tensor([True])}),
            1: TickScript(values={"fired": torch.tensor([True])}),
        },
    )
    flow = DefaultBenchmarkFlow(
        (TerminationRule("reopen", "fired", "success", "custom.reopen.first"),)
    )
    sink, engine = _engine_sink(monitor, flow)
    first = make_bundle(
        num_envs=1,
        t=0,
        cmd=torch.tensor([[0.1, 0.2]]),
        reset_state={0: {"reset_index": 0, "reset_valid": True, "reset_reason": "first"}},
    )
    sink.on_step(0, first)
    sink.drain_terminations()
    sink.on_episode_end([0], first)
    sink.on_reset([0])

    second = make_bundle(
        num_envs=1,
        t=1,
        cmd=torch.tensor([[0.7, 0.8]]),
        reset_state={0: {"reset_index": 0, "reset_valid": False, "reset_reason": "second"}},
    )
    sink.on_step(1, second)
    sink.drain_terminations()
    sink.on_episode_end([0], second)

    assert [
        (record.episode_seq, record.reset_facts["reset_reason"]) for record in engine.records
    ] == [(0, "first"), (1, "second")]
    assert engine.records[0].command_at_start == pytest.approx((0.1, 0.2))
    assert engine.records[1].command_at_start == pytest.approx((0.7, 0.8))


def test_episode_record_captures_start_command_reset_facts_and_canonical_terminal() -> None:
    monitor = ScriptedMonitor(
        "hit",
        {0: TickScript(values={"fired": torch.tensor([True])})},
    )
    flow = DefaultBenchmarkFlow((TerminationRule("hit", "fired", "success", "custom.hit.first"),))
    sink, engine = _engine_sink(monitor, flow)
    reset_state = {0: {"reset_index": 0, "reset_valid": True, "reset_reason": "initial"}}
    bundle = make_bundle(
        num_envs=1,
        t=0,
        cmd=torch.tensor([[0.25, -0.5]]),
        reset_state=reset_state,
    )

    sink.on_step(0, bundle)
    sink.drain_terminations()
    stamped = Termination(
        outcome=EpisodeOutcome.SUCCESS,
        code="custom.hit.first",
        source=TerminationSource.CUSTOM,
        step_index=1,
        time_s=0.02,
    )
    core = Termination(
        outcome=EpisodeOutcome.FAILURE,
        code="env.fallen",
        source=TerminationSource.ENVIRONMENT,
        step_index=1,
        time_s=0.02,
    )
    bundle["terminations"] = {0: EpisodeTermination.from_terminations([core, stamped])}
    sink.on_episode_end([0], bundle)

    record = engine.records[0]
    assert record.command_at_start == (0.25, -0.5)
    assert record.reset_facts == reset_state[0]
    assert record.first_terminal is not None
    assert record.first_terminal.step_index == 1
    assert record.first_terminal.time_s == 0.02
    assert record.core_termination is not None
    assert [term.code for term in record.core_termination.all] == ["env.fallen"]
    assert record.verdict == "terminal"
    assert record.rewound is False


def test_record_keeps_the_flow_request_when_no_canonical_stamp_is_available() -> None:
    monitor = ScriptedMonitor(
        "raw",
        {0: TickScript(values={"fired": torch.tensor([True])})},
    )
    flow = DefaultBenchmarkFlow((TerminationRule("raw", "fired", "success", "custom.raw.first"),))
    sink, engine = _engine_sink(monitor, flow)
    bundle = make_bundle(num_envs=1, t=0)

    sink.on_step(0, bundle)
    sink.drain_terminations()
    sink.on_episode_end([0], bundle)

    record = engine.records[0]
    assert record.first_terminal is not None
    assert record.first_terminal.code == "custom.raw.first"
    assert record.first_terminal.step_index is None
    assert record.core_termination is None


def test_record_collects_episode_event_summaries() -> None:
    monitor = ScriptedMonitor(
        "events",
        {0: TickScript(values={"fired": torch.tensor([True])})},
    )
    flow = DefaultBenchmarkFlow(
        events=(EventRule("events", "fired", "hit", code="task.events.hit"),)
    )
    sink, engine = _engine_sink(monitor, flow)
    bundle = make_bundle(num_envs=1, t=0)

    sink.on_step(0, bundle)
    sink.on_episode_end([0], bundle)

    assert engine.records[0].event_counts == {"task.events.hit": 1}


def test_open_episode_is_incomplete_in_the_unique_run_summary() -> None:
    sink, _ = _engine_sink(ScriptedMonitor("open", {}))
    sink.on_step(0, make_bundle(num_envs=1, t=0))

    summary = sink.on_run_end()

    assert set(summary) == {"BenchmarkSink"}
    document = summary["BenchmarkSink"]
    assert isinstance(document, dict)
    assert document["episodes"]["count"] == 1
    assert document["episodes"]["incomplete"] == 1
    assert document["episodes"]["verdicts"] == {"incomplete": 1}
    assert document["episodes"]["retained"] == 1
    assert document["episodes"]["retain_episodes"] is True
    assert document["episodes"]["length"] == {"min": 1, "max": 1, "mean": 1.0}
    assert document["episodes"]["by_env"]["0"]["incomplete"] == 1
    assert document["execution"] == {
        "event_total": 0,
        "dropped_events": 0,
        "terminal_conflicts": 0,
        "retention": {
            "retain_episodes": True,
            "max_events": 200_000,
            "event_log": True,
        },
    }
    assert json.loads(json.dumps(document)) == document


def test_zero_episode_run_returns_an_honest_summary() -> None:
    sink = BenchmarkSink([])

    summary = sink.on_run_end()

    assert set(summary) == {"BenchmarkSink"}
    report = summary["BenchmarkSink"]
    assert isinstance(report, dict)
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
    assert report["execution"] == {
        "event_total": 0,
        "dropped_events": 0,
        "terminal_conflicts": 0,
        "retention": {
            "retain_episodes": False,
            "max_events": 200_000,
            "event_log": True,
        },
    }


def test_episode_record_is_frozen_and_json_safe() -> None:
    record = EpisodeRecord(
        env_id=0,
        episode_seq=1,
        ticks=3,
        verdict="incomplete",
        command_at_start=[0.1, 0.2],
        reset_facts={"reset_index": 0, "reset_valid": True},
        event_counts={"task.record.hit": 2},
    )

    assert record.command_at_start == (0.1, 0.2)
    with pytest.raises(TypeError):
        record.reset_facts["reset_index"] = 9
    with pytest.raises(TypeError):
        record.event_counts["task.record.hit"] = 9
    document = record.to_json()
    assert ensure_json_value(document) == document
    assert json.loads(json.dumps(document)) == document


def test_episode_record_deep_freezes_nested_reset_facts() -> None:
    source: dict[str, object] = {
        "nested": {"values": [1, 2], "flags": [True, None]},
        "name": "sample",
    }
    record = EpisodeRecord(
        env_id=0,
        episode_seq=0,
        ticks=1,
        verdict="incomplete",
        reset_facts=source,
    )

    source["nested"]["values"].append(3)

    nested = record.reset_facts["nested"]
    assert nested["values"] == (1, 2)
    assert nested["flags"] == (True, None)
    with pytest.raises(TypeError):
        nested["extra"] = 1
    with pytest.raises(AttributeError):
        nested["values"].append(3)


def test_episode_record_to_json_returns_a_private_deep_copy() -> None:
    record = EpisodeRecord(
        env_id=0,
        episode_seq=0,
        ticks=1,
        verdict="incomplete",
        reset_facts={"nested": {"values": [1, 2]}},
    )

    document = record.to_json()
    reset_facts = document["reset_facts"]
    assert isinstance(reset_facts, dict)
    nested = reset_facts["nested"]
    assert isinstance(nested, dict)
    nested["values"].append(99)
    nested["extra"] = True

    assert record.to_json()["reset_facts"] == {"nested": {"values": [1, 2]}}


def test_episode_record_rejects_non_json_reset_facts() -> None:
    with pytest.raises(TypeError, match="reset_facts"):
        EpisodeRecord(
            env_id=0,
            episode_seq=0,
            ticks=1,
            verdict="incomplete",
            reset_facts={"bad": torch.tensor(1.0)},
        )


def _write_task_document(
    tmp_path,
    *,
    schema: str = "unigym_extbench/3",
    extensions: str = "",
    scenario: str = "scenario:\n  command_pool: [[0.0], [1.0]]\n",
    retention: str = "retention:\n  retain_episodes: true\n  max_events: 3\n",
    report_extension: str | None = None,
) -> object:
    path = tmp_path / "task.yaml"
    extension_line = (
        "" if report_extension is None else f"    report_extension: {report_extension}\n"
    )
    text = f"""
schema: {schema}
name: composed
monitors:
  contact:
    builtin: predicate
    when:
      source: value:level
      op: gte
      value: 0.5
    trigger: edge
    emit:
      value: contact
{extension_line}{extensions}{retention}{scenario}"""
    path.write_text(text, encoding="utf-8")
    return path


def test_make_sink_task_document_composes_the_plan(tmp_path) -> None:
    path = _write_task_document(tmp_path)

    sink = make_sink(make_sink_context(options={"task_document": str(path)}))

    engine = sink.rewind_provider
    assert [monitor.name for monitor in engine.monitors] == ["contact"]
    assert [extension.name for extension in engine.extensions] == ["scenario_coverage"]
    assert engine.retention == RetentionSpec(retain_episodes=True, max_events=3, event_log=True)
    summary = sink.on_run_end()["BenchmarkSink"]
    assert summary["tasks"] == {
        "name": "sink-context",
        "monitors": ["contact"],
        "extensions": ["scenario_coverage"],
    }
    assert summary["execution"]["retention"] == {
        "retain_episodes": True,
        "max_events": 3,
        "event_log": True,
    }


def test_make_sink_task_document_resolves_a_relative_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_task_document(tmp_path)
    monkeypatch.chdir(tmp_path)

    sink = make_sink(make_sink_context(options={"task_document": "task.yaml"}))

    assert [monitor.name for monitor in sink.rewind_provider.monitors] == ["contact"]


def test_make_sink_task_document_conflicts_with_factories(tmp_path) -> None:
    path = _write_task_document(tmp_path)

    with pytest.raises(ConfigError, match="task_document conflicts"):
        make_sink(
            make_sink_context(
                options={
                    "task_document": str(path),
                    "monitor_factory": _FACTORY,
                }
            )
        )
    with pytest.raises(ConfigError, match="task_document conflicts"):
        make_sink(
            make_sink_context(
                options={
                    "task_document": str(path),
                    "extension_factory": "_support:make_report_extensions",
                }
            )
        )


@pytest.mark.parametrize(
    ("task_document", "fragment"),
    [
        (123, "expected a non-empty path string"),
        ("", "expected a non-empty path string"),
        ("   ", "expected a non-empty path string"),
        ("/no/such/task-document.yaml", "cannot read task document"),
    ],
)
def test_make_sink_rejects_bad_task_document_paths(task_document: object, fragment: str) -> None:
    with pytest.raises(ConfigError, match="task_document") as excinfo:
        make_sink(make_sink_context(options={"task_document": task_document}))

    assert fragment in str(excinfo.value)


def test_make_sink_rejects_the_old_schema_in_a_task_document(tmp_path) -> None:
    path = _write_task_document(tmp_path, schema="unigym_extbench/2")

    with pytest.raises(ConfigError, match="retired") as excinfo:
        make_sink(make_sink_context(options={"task_document": str(path)}))

    assert "task_document" in str(excinfo.value)
    assert "unigym_extbench/3" in str(excinfo.value)


def test_make_sink_rejects_duplicate_yaml_keys_in_a_task_document(tmp_path) -> None:
    path = tmp_path / "task.yaml"
    path.write_text(
        """
name: duplicate
monitors:
  same:
    builtin: predicate
    when: {source: value:x, op: gt, value: 0.0}
    trigger: edge
  same:
    builtin: predicate
    when: {source: value:y, op: gt, value: 0.0}
    trigger: edge
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="duplicate mapping key 'same'"):
        make_sink(make_sink_context(options={"task_document": str(path)}))


def test_make_sink_task_document_accepts_a_matching_monitor_extensions_option(
    tmp_path,
) -> None:
    path = _write_task_document(
        tmp_path,
        extensions=("extensions:\n" "  coverage:\n" "    builtin: scenario_coverage\n"),
        report_extension="coverage",
    )

    sink = make_sink(
        make_sink_context(
            options={
                "task_document": str(path),
                "monitor_extensions": {"contact": "coverage"},
                "write_report": False,
            }
        )
    )

    assert [extension.name for extension in sink.rewind_provider.extensions] == ["coverage"]
    summary = sink.on_run_end()["BenchmarkSink"]
    assert summary["tasks"]["extensions"] == ["coverage"]
    assert "contact" not in summary["monitors"]


def test_make_sink_task_document_rejects_a_conflicting_monitor_extensions_option(
    tmp_path,
) -> None:
    path = _write_task_document(
        tmp_path,
        extensions=(
            "extensions:\n"
            "  coverage:\n"
            "    builtin: scenario_coverage\n"
            "  counts:\n"
            "    module: unigym_bench.report:ScenarioCoverageExtension\n"
            "    params:\n"
            "      name: counts\n"
            "      commands: [[0.0]]\n"
        ),
        report_extension="coverage",
    )

    with pytest.raises(ConfigError, match="conflicts with the task document"):
        make_sink(
            make_sink_context(
                options={
                    "task_document": str(path),
                    "monitor_extensions": {"contact": "counts"},
                }
            )
        )


def test_make_sink_task_document_rejects_a_query_source_at_composition(tmp_path) -> None:
    path = tmp_path / "task.yaml"
    path.write_text(
        """
name: query
monitors:
  m:
    builtin: predicate
    when: {source: "query:height", op: gt, value: 0.0}
    trigger: edge
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="query source"):
        make_sink(make_sink_context(options={"task_document": str(path)}))
