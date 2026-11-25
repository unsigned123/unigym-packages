"""``unigym-rl``: the L2 task-environment layer over ``unigym`` core.

:class:`TaskEnv` is the base task contract; :class:`EnvSpec` builds a
task-owned ``Run`` programmatically and :class:`RingBuffer` stores per-env
history.  The optional Gymnasium vector face is
:class:`unigym_rl.gym.GymTaskEnv`.  The package deliberately does not import
``gymnasium`` at import time: the Gym wrapper is exported lazily, so a base
``TaskEnv`` user needs no RL framework installed.  The optional
:class:`RewardManager` / :class:`TerminationManager` helpers depend only on
``torch`` and can be ignored entirely.

The minimal task loop is::

    run = Run(..., termination_mode="external")
    env = MyTask(run)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(actions)

    # optional, requires the ``gymnasium`` extra
    from unigym_rl import GymTaskEnv
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

from unigym_rl.buffers import RingBuffer
from unigym_rl.context import (
    EpisodeEnd,
    TaskContext,
    TaskState,
    Transition,
)
from unigym_rl.errors import TaskConfigError, TaskContractError, TaskEnvError
from unigym_rl.managers import (
    RewardManager,
    RewardTerm,
    TerminationManager,
    TerminationResult,
    TerminationTerm,
)
from unigym_rl.spec import EnvSpec
from unigym_rl.task import TaskEnv

__all__ = [
    "EnvSpec",
    "EpisodeEnd",
    "GymTaskEnv",
    "RingBuffer",
    "RewardManager",
    "RewardTerm",
    "TaskConfigError",
    "TaskContext",
    "TaskContractError",
    "TaskEnv",
    "TaskEnvError",
    "TaskState",
    "TerminationManager",
    "TerminationResult",
    "TerminationTerm",
    "Transition",
]


def _resolve_version() -> str:
    """The installed distribution version, with a checkout fallback.

    Reads ``unigym-rl`` distribution metadata when the package is installed;
    otherwise falls back to the adjacent ``pyproject.toml`` so a source checkout
    on ``PYTHONPATH`` still reports its real version.  Returns ``"unknown"``
    only when neither source is readable.
    """
    try:
        return metadata.version("unigym-rl")
    except metadata.PackageNotFoundError:
        pass
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if entry.startswith("version"):
                _, _, rhs = entry.partition("=")
                value = rhs.strip().strip('"').strip("'")
                if value:
                    return value
    except OSError:
        pass
    return "unknown"


__version__ = _resolve_version()

_LAZY_EXPORTS = {"GymTaskEnv": "unigym_rl.gym"}


def __getattr__(name: str) -> object:
    """Resolve optional exports without importing their dependencies eagerly."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
