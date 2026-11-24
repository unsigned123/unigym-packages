"""Import fence: the pack imports no core benchmark scaffolding."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from unigym.core import episode_termination

import unigym_bench
from unigym_bench import assert_core_contract, compat

_SRC_ROOT = str(Path(unigym_bench.__file__).resolve().parents[1])

_BLOCKED_PREFIXES = (
    "unigym.benchmarks",
    "unigym.runner.evaluation",
    "unigym.runner.episodes",
)

_TERMINATION_SURFACES = (
    "EpisodeOutcome",
    "Termination",
    "TerminationExtension",
    "TerminationSource",
    "validate_code",
)

_FENCE_SCRIPT = f"""
import sys

import unigym_bench as pack

pack.assert_core_contract()
prefixes = {_BLOCKED_PREFIXES!r}
blocked = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
)
if blocked:
    raise SystemExit("forbidden core modules imported: " + ", ".join(blocked))
print(pack.__version__)
"""


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = _SRC_ROOT if not existing else os.pathsep.join([_SRC_ROOT, existing])
    return env


def test_pack_import_and_compat_import_no_benchmark_scaffolding() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _FENCE_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip()


def test_core_contract_passes_on_current_core() -> None:
    assert assert_core_contract() is None


def test_core_contract_names_a_missing_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        compat,
        "_REQUIRED_SURFACES",
        (("unigym", ("__version__", "SurfaceThatDoesNotExist")),),
    )
    with pytest.raises(RuntimeError) as excinfo:
        compat.assert_core_contract()
    message = str(excinfo.value)
    assert "unigym.SurfaceThatDoesNotExist" in message
    assert "is not satisfied" in message


def test_core_contract_rejects_an_older_core(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compat, "REQUIRED_CORE_VERSION", "999.0.0")
    with pytest.raises(RuntimeError) as excinfo:
        compat.assert_core_contract()
    message = str(excinfo.value)
    assert "999.0.0" in message
    assert "is older than required" in message


@pytest.mark.parametrize("attribute", _TERMINATION_SURFACES)
def test_core_contract_requires_each_termination_surface(
    monkeypatch: pytest.MonkeyPatch, attribute: str
) -> None:
    monkeypatch.delattr(episode_termination, attribute)
    with pytest.raises(RuntimeError) as excinfo:
        compat.assert_core_contract()
    assert f"unigym.core.episode_termination.{attribute}" in str(excinfo.value)
