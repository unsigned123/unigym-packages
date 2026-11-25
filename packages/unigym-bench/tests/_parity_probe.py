"""Subprocess probe: one explicit-lifecycle tiny-arm pack run.

The old live built-in benchmark leg was removed with the post-deletion core.
The historical built-in facts now live in ``tests/_frozen_builtin_facts.json``;
this probe runs only the pack lane through the explicit ``runner.terminate``
boundary and prints the facts the parent test compares against that fixture.

It prints one ``PARITY_JSON <document>`` line; the parent test owns the
comparison.  No old benchmark scaffolding is imported here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from unigym.run.sink import SinkSpec

from _tiny_arm import end_episode, make_run, zero_actions

STEPS = 2


def _run_pack(tmp: Path) -> dict[str, Any]:
    run = make_run(
        tmp,
        steps=STEPS,
        sink=SinkSpec(
            factory="unigym_bench.sink:make_sink",
            options={"write_report": True},
        ),
    )
    try:
        instance = run.build()
        runner = run.runner("external", action_fn=zero_actions)
        for _ in range(STEPS):
            runner.step(zero_actions(0))
        end_episode(runner, reset=True)
        runner.run(stop_when=lambda: True)
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
    """Run the pack lane under ``root`` and print its facts."""
    base = Path(root)
    (base / "pack").mkdir(parents=True, exist_ok=True)
    document = {"pack": _run_pack(base / "pack")}
    print("PARITY_JSON " + json.dumps(document, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
