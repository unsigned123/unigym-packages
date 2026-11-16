"""``unigym-rl``: the L2 task-environment layer over ``unigym`` core.

R0 exposes the task contract only -- no training loop, no RL framework
integration, no terrain and no real backend.  Build a :class:`TaskEnv`
subclass on top of an externally-terminated core run::

    run = Run(..., termination_mode="external")
    env = MyTask(run)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(actions)
"""

from __future__ import annotations

from unigym_rl.context import EpisodeEnd, TaskContext, TaskState, Transition
from unigym_rl.errors import TaskConfigError, TaskContractError, TaskEnvError
from unigym_rl.task import TaskEnv

__all__ = [
    "EpisodeEnd",
    "TaskConfigError",
    "TaskContext",
    "TaskContractError",
    "TaskEnv",
    "TaskEnvError",
    "TaskState",
    "Transition",
]

__version__ = "0.0.0"
