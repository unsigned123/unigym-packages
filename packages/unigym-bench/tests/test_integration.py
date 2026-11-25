"""P9 integration: real tiny-arm MuJoCo run, zero-episode and fake conformance.

The real-run tests drive the core runner tick engine (``ExternalRunner`` over
``SyncRunner``, created by ``Run.runner``) over a self-authored two-joint MuJoCo
fixture and use the pack's ``make_sink`` factory.
The fake tests feed real runner-shaped bundles into the real sink/engine; they
do not simulate the pack, only the runner transport.

This module requires the D0 MuJoCo lane.  A missing backend fails here rather
than silently skipping.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from unigym.core.episode_termination import (
    EpisodeOutcome,
    EpisodeTermination,
    Termination,
    TerminationSource,
)
from unigym.run.sink import SinkSpec

from unigym_bench.events import Event
from unigym_bench.report import REPORT_FILENAME, REPORT_SCHEMA
from unigym_bench.sink import BenchmarkSink
from unigym_bench.specs import RetentionSpec
from _support import (
    BundleRecordingSink,
    ScriptedMonitor,
    TickScript,
    make_bundle,
    make_termination,
)
from _tiny_arm import make_run, zero_actions

if TYPE_CHECKING:
    from unigym.run.api import Run
    from unigym.run.instance import RunInstance


def _make_run(tmp_path: Path, *, steps: int, episode_length_s: float) -> Run:
    return make_run(
        tmp_path,
        steps=steps,
        episode_length_s=episode_length_s,
        sink=SinkSpec(
            factory="_support:make_observed_sink",
            options={
                "monitor_factory": "_support:make_integration_monitors",
                "extension_factory": "_support:make_coverage_extensions",
                "write_report": True,
            },
        ),
        benchmark={"tasks": [], "min_episodes": 1, "max_steps": max(steps, 1)},
    )


def _walk_sinks(node: object) -> Iterator[object]:
    yield node
    for child in getattr(node, "sinks", ()) or ():
        yield from _walk_sinks(child)


def _observed_sink(instance: RunInstance) -> BundleRecordingSink:
    for node in _walk_sinks(getattr(instance, "sink", None)):
        if isinstance(node, BundleRecordingSink):
            return node
    raise AssertionError("the observed pack sink is not mounted in the run tree")


def _search_pack_report(document: object) -> dict[str, object] | None:
    if isinstance(document, dict):
        if document.get("schema") == REPORT_SCHEMA:
            return document
        for value in document.values():
            found = _search_pack_report(value)
            if found is not None:
                return found
    elif isinstance(document, (list, tuple)):
        for value in document:
            found = _search_pack_report(value)
            if found is not None:
                return found
    return None


def _find_pack_report(document: object) -> dict[str, object]:
    found = _search_pack_report(document)
    if found is None:
        raise AssertionError("no external benchmark report in the runner summary")
    return found


def _core_termination_counter(probe: BundleRecordingSink) -> Counter[str]:
    counter: Counter[str] = Counter()
    for facts in probe.episode_terminations:
        for entry in facts.values():
            if isinstance(entry, EpisodeTermination):
                for term in entry.all:
                    counter[term.outcome.value] += 1
    return counter


def test_real_tiny_arm_run_writes_the_pack_report_not_core_benchmark_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real tiny-arm benchmark episode writes the pack report, not core benchmarks."""
    # MuJoCo emits its process warning log into the current working directory;
    # run from the test's temp dir so the repository stays clean.
    monkeypatch.chdir(tmp_path)
    run = _make_run(tmp_path, steps=20, episode_length_s=0.04)
    try:
        instance = run.build()
        probe = _observed_sink(instance)
        runner = run.runner("external", action_fn=zero_actions)
        summary = runner.run(stop_when=instance.stop_when)
    finally:
        run.close()

    raw_out_dir = instance.out_dir
    assert raw_out_dir is not None
    out_dir = Path(raw_out_dir)
    report_path = out_dir / REPORT_FILENAME
    assert report_path.is_file()
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written == _find_pack_report(summary)
    assert written["schema"] == REPORT_SCHEMA
    assert written["run"]["out_dir"] == str(out_dir)
    assert written["run"]["num_envs"] == 1

    episodes = written["episodes"]
    assert isinstance(episodes, dict)
    assert episodes["count"] >= 1
    assert episodes["count"] == len(probe.episode_terminations)
    assert written["terminations"]["outcome"] == dict(_core_termination_counter(probe))
    assert set(written["terminations"]["outcome"]) <= {outcome.value for outcome in EpisodeOutcome}
    assert written["terminations"]["code"] == {"runtime.episode_horizon": episodes["count"]}
    assert written["terminations"]["source"] == {"runtime": episodes["count"]}
    coverage = written["extensions"]["coverage"]
    assert coverage["episodes"] == episodes["count"]
    assert sum(coverage["episode_counts"]) + coverage["unmatched_episodes"] == coverage["episodes"]
    assert written["tasks"]["extensions"] == ["coverage"]

    # The runner bundle no longer exposes a live env object.
    assert probe.step_bundle_keys
    assert all("env" not in keys for keys in probe.step_bundle_keys)
    assert all("view" in keys for keys in probe.step_bundle_keys)

    # No core benchmark artifact is written (other core run state such as
    # events.jsonl may legitimately live in out_dir).
    assert not (out_dir / "benchmark_report.json").exists()
    assert not (out_dir / "results.json").exists()
    assert not (out_dir / "metrics.npz").exists()

    # The pack sink (through its engine provider) was mounted for rewind.
    providers = {getattr(provider, "name", None) for provider in instance.rewind_providers}
    assert "unigym_bench" in providers


def test_real_zero_episode_run_writes_an_honest_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real tiny-arm run stopped before its first tick reports zero episodes."""
    monkeypatch.chdir(tmp_path)
    run = _make_run(tmp_path, steps=20, episode_length_s=0.04)
    try:
        instance = run.build()
        runner = run.runner("external", action_fn=zero_actions)
        summary = runner.run(stop_when=lambda: True)
    finally:
        run.close()

    raw_out_dir = instance.out_dir
    assert raw_out_dir is not None
    report_path = Path(raw_out_dir) / REPORT_FILENAME
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written == _find_pack_report(summary)
    assert written["episodes"]["count"] == 0
    assert written["episodes"]["complete"] == 0
    assert written["episodes"]["incomplete"] == 0
    assert written["episodes"]["retained"] == 0
    assert written["terminations"]["total"] == 0
    assert written["execution"]["event_total"] == 0
    assert written["execution"]["dropped_events"] == 0


def test_zero_episode_fake_sink_report_is_honest() -> None:
    """The fake-bundle sink path also reports zero episodes honestly."""
    sink = BenchmarkSink([])

    report = sink.on_run_end()["BenchmarkSink"]

    assert isinstance(report, dict)
    assert report["episodes"]["count"] == 0
    assert report["terminations"]["total"] == 0
    assert report["execution"]["event_total"] == 0


def test_multi_env_fake_conformance_keeps_episode_facts_per_env() -> None:
    """A two-env fake run reports per-env counts and first-terminal verdicts."""
    terminal = make_termination("custom.task.hit")
    monitor = ScriptedMonitor(
        "task",
        {
            0: TickScript(
                signals={"live": torch.tensor([1.0, 2.0])},
                events=(Event(name="hit", code="task.record.hit", env_ids=(0,)),),
                terminals={0: terminal},
            ),
            1: TickScript(terminals={1: terminal}),
        },
    )
    sink = BenchmarkSink([monitor], retention=RetentionSpec(retain_episodes=True))

    for t in range(2):
        sink.on_step(t, make_bundle(t=t, num_envs=2))
        due = sink.drain_terminations()
        assert sorted(due) == [t]
        sink.on_episode_end([t], make_bundle(t=t, num_envs=2))
        sink.on_step_end(t, make_bundle(t=t, num_envs=2))

    report = sink.on_run_end()["BenchmarkSink"]
    assert isinstance(report, dict)
    # Env 0 ends at tick 0, env 1 at tick 1; env 0's reopened episode stays open
    # and is finalized honestly as incomplete with two observed ticks.
    assert report["episodes"]["count"] == 3
    assert report["episodes"]["complete"] == 2
    assert report["episodes"]["incomplete"] == 1
    assert report["episodes"]["verdicts"] == {"incomplete": 1, "terminal": 2}
    assert report["episodes"]["by_env_counts"] == {"0": 2, "1": 1}
    assert report["episodes"]["by_env"]["0"]["verdicts"] == {
        "incomplete": 1,
        "terminal": 1,
    }
    assert report["episodes"]["by_env"]["1"]["verdicts"] == {"terminal": 1}
    assert report["terminations"] == {
        "total": 2,
        "outcome": {"success": 2},
        "code": {"custom.task.hit": 2},
        "source": {"custom": 2},
    }
    assert set(report["monitors"]) == {"task"}
    assert report["execution"]["terminal_conflicts"] == 0


def test_report_termination_vocabulary_matches_core_facts() -> None:
    """A core deadline fact maps one-to-one into the pack's report vocabulary."""
    fact = EpisodeTermination(
        primary=Termination(
            outcome=EpisodeOutcome.DEADLINE,
            code="runtime.episode_horizon",
            source=TerminationSource.RUNTIME,
        )
    )
    sink = BenchmarkSink([])
    sink.on_step(0, make_bundle(t=0, num_envs=1))
    sink.on_episode_end([0], make_bundle(t=0, num_envs=1, terminations={0: fact}))

    report = sink.on_run_end()["BenchmarkSink"]
    assert isinstance(report, dict)
    assert report["episodes"]["verdicts"] == {"deadline": 1}
    assert report["terminations"] == {
        "total": 1,
        "outcome": {"deadline": 1},
        "code": {"runtime.episode_horizon": 1},
        "source": {"runtime": 1},
    }
