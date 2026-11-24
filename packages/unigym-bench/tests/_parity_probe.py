"""Subprocess probe: same-config tiny-arm fact parity vs the core built-in run.

This script is test support executed in a fresh interpreter.  It must never be
imported by the pack runtime: constructing the core built-in benchmark requires
importing ``unigym.benchmarks`` and opening its ``ACTIVATE`` registration
window, which is a test-only lane.  Running it in a subprocess keeps that global
registration state out of the main test process.

It prints one ``PARITY_JSON <document>`` line; the parent test asserts the
same-config episode/termination facts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from unigym.pack_api import discover
from unigym.registration import RegistrationPhase, enter_phase
from unigym.run.sink import SinkSpec

from _tiny_arm import make_run, zero_actions

STEPS = 2
EPISODE_LENGTH_S = 0.04


def _prepare_builtin_tasks() -> None:
    """Open the core ACTIVATE window and import the task-provider catalog."""
    enter_phase(RegistrationPhase.ACTIVATE)
    from unigym.benchmarks import build_tasks

    # An empty section list only imports the catalog; registrations run here,
    # inside the legal window, instead of later at BACKEND_BUILD.
    build_tasks([], {}, discover([]))


def _run_builtin(tmp: Path) -> dict[str, Any]:
    _prepare_builtin_tasks()
    run = make_run(
        tmp,
        steps=STEPS,
        episode_length_s=EPISODE_LENGTH_S,
        sink=None,
        benchmark={
            "tasks": [
                {
                    "name": "stand",
                    "articulation": "tiny",
                    "height_range": [-10.0, 10.0],
                }
            ],
            "min_episodes": 1,
            "max_steps": STEPS,
        },
        pack_runtime=discover([]),
    )
    try:
        instance = run.build()
        runner = run.runner("external", action_fn=zero_actions)
        runner.run(stop_when=instance.stop_when)
        out_dir = Path(instance.out_dir or "")
        report = json.loads((out_dir / "benchmark_report.json").read_text(encoding="utf-8"))
    finally:
        run.close()
    return {
        "episodes_total": report["episodes_total"],
        "steps_total": report["steps_total"],
        "terminal_outcome_counts": report["terminal_outcome_counts"],
        "terminal_code_counts": report["terminal_code_counts"],
        "terminal_source_counts": report["terminal_source_counts"],
    }


def _run_pack(tmp: Path) -> dict[str, Any]:
    run = make_run(
        tmp,
        steps=STEPS,
        episode_length_s=EPISODE_LENGTH_S,
        sink=SinkSpec(
            factory="unigym_bench.sink:make_sink",
            options={"write_report": True},
        ),
        benchmark={"tasks": [], "min_episodes": 1, "max_steps": STEPS},
    )
    try:
        instance = run.build()
        runner = run.runner("external", action_fn=zero_actions)
        runner.run(stop_when=instance.stop_when)
        out_dir = Path(instance.out_dir or "")
        report = json.loads(
            (out_dir / "external_benchmark_report.json").read_text(encoding="utf-8")
        )
    finally:
        run.close()
    return {
        "episodes_count": report["episodes"]["count"],
        "total_ticks": report["episodes"]["total_ticks"],
        "terminations": report["terminations"],
    }


def main(root: str) -> None:
    """Run both lanes under ``root`` and print the comparable facts."""
    base = Path(root)
    (base / "builtin").mkdir(parents=True, exist_ok=True)
    (base / "pack").mkdir(parents=True, exist_ok=True)
    document = {
        "builtin": _run_builtin(base / "builtin"),
        "pack": _run_pack(base / "pack"),
    }
    print("PARITY_JSON " + json.dumps(document, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
