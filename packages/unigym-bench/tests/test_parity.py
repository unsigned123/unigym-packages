"""Parity against the frozen built-in benchmark facts.

The post-deletion core no longer owns a built-in benchmark, an episode clock or
``EvaluationSession``.  The historical built-in facts are frozen in
``tests/_frozen_builtin_facts.json``; this module never imports the removed
benchmark scaffolding.  It proves that the pack's explicit-lifecycle run keeps
the frozen episode/ticks/outcome facts, while documenting the namespace change
that follows from explicit ``custom.*`` termination facts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import unigym_bench

_SRC_ROOT = str(Path(unigym_bench.__file__).resolve().parents[1])
_PROBE = Path(__file__).parent / "_parity_probe.py"
_FIXTURE = Path(__file__).parent / "_frozen_builtin_facts.json"


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = _SRC_ROOT if not existing else os.pathsep.join([_SRC_ROOT, existing])
    return env


def _parity_document(stdout: str) -> dict[str, Any]:
    for line in stdout.splitlines():
        if line.startswith("PARITY_JSON "):
            document = json.loads(line[len("PARITY_JSON ") :])
            assert isinstance(document, dict)
            return document
    raise AssertionError(f"probe printed no PARITY_JSON line:\n{stdout}")


def _frozen_facts() -> dict[str, Any]:
    document = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def test_frozen_fixture_is_the_historical_builtin_anchor() -> None:
    """The fixture keeps the old built-in and pack facts as inert data."""
    fixture = _frozen_facts()
    builtin = fixture["builtin"]
    pack = fixture["pack"]
    assert fixture["schema"] == "unigym_bench.frozen_builtin_facts/1"
    assert builtin["episodes_total"] == 1
    assert builtin["steps_total"] == 2
    assert pack["episodes_count"] == builtin["episodes_total"]
    assert pack["total_ticks"] == builtin["steps_total"]
    assert pack["terminations"]["outcome"] == builtin["terminal_outcome_counts"]


def test_pack_explicit_lifecycle_matches_frozen_episode_and_tick_facts(
    tmp_path: Path,
) -> None:
    """One explicit episode keeps the frozen counts and deadline outcome."""
    result = subprocess.run(
        [sys.executable, str(_PROBE), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(tmp_path),
        env=_subprocess_env(),
    )
    assert result.returncode == 0, result.stderr or result.stdout
    document = _parity_document(result.stdout)

    builtin = _frozen_facts()["builtin"]
    pack = document["pack"]
    assert pack["episodes_count"] == builtin["episodes_total"] == 1
    assert pack["total_ticks"] == builtin["steps_total"] == 2
    assert pack["terminations"]["total"] == builtin["episodes_total"]
    assert pack["terminations"]["outcome"] == builtin["terminal_outcome_counts"]
    # The old core-owned horizon used the runtime namespace.  The pack now ends
    # the episode explicitly, so the boundary fact is custom-owned even though
    # the outcome stays ``deadline``.
    assert pack["terminations"]["code"] == {"custom.episode_horizon": 1}
    assert pack["terminations"]["source"] == {"custom": 1}
