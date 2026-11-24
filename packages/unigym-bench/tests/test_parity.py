"""P9 parity: same-config tiny-arm facts against the core built-in benchmark.

The built-in run is constructed in a fresh subprocess because it needs the core
benchmark task catalog imported inside the ``ACTIVATE`` registration window.
Keeping that in a child process preserves the pack's runtime import fence and
leaves the parent test process's registration gate untouched.

This is fact parity, not report/schema parity: it proves that for one identical
tiny-arm config the pack's episode count, total ticks and termination
outcome/code/source counters equal the built-in benchmark's.  Built-in metric
scoring has no pack equivalent and is documented as such in the README.
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


def test_same_config_facts_match_the_core_builtin_benchmark(tmp_path: Path) -> None:
    """One tiny-arm config yields the same episode/termination facts in both."""
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

    builtin = document["builtin"]
    pack = document["pack"]
    assert builtin["episodes_total"] == pack["episodes_count"] == 1
    assert builtin["steps_total"] == pack["total_ticks"] == 2
    assert pack["terminations"] == {
        "total": builtin["episodes_total"],
        "outcome": builtin["terminal_outcome_counts"],
        "code": builtin["terminal_code_counts"],
        "source": builtin["terminal_source_counts"],
    }
