"""``unigym-rl``: the L2 task-environment layer over ``unigym`` core.

:class:`TaskEnv` is the base task contract; :class:`EnvSpec` builds a
task-owned ``Run`` programmatically and :class:`RingBuffer` stores per-env
history.  The optional Gymnasium vector face is
:class:`unigym_rl.gym.GymTaskEnv`.  The package deliberately does not import
``gymnasium`` at import time: the Gym wrapper is exported lazily, so a base
``TaskEnv`` user needs no RL framework installed::

    run = Run(..., termination_mode="external")
    env = MyTask(run)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(actions)

    # optional, requires the ``gymnasium`` extra
    from unigym_rl import GymTaskEnv
"""

from __future__ import annotations

from unigym_rl.buffers import RingBuffer
from unigym_rl.context import EpisodeEnd, TaskContext, TaskState, Transition
from unigym_rl.errors import TaskConfigError, TaskContractError, TaskEnvError
from unigym_rl.spec import EnvSpec
from unigym_rl.task import TaskEnv

__all__ = [
    "EnvSpec",
    "EpisodeEnd",
    "GymTaskEnv",
    "RingBuffer",
    "TaskConfigError",
    "TaskContext",
    "TaskContractError",
    "TaskEnv",
    "TaskEnvError",
    "TaskState",
    "Transition",
]

__version__ = "0.0.0"

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
