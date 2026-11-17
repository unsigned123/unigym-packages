"""Teaching errors raised by the ``unigym-rl`` task layer.

The hierarchy is intentionally small: configuration mistakes (wrong run
mode), contract violations (wrong tensor shape/dtype) and general task-env
usage errors are all failures the user must fix, never silent fallbacks.
"""

from __future__ import annotations

__all__ = ["TaskConfigError", "TaskContractError", "TaskEnvError"]


class TaskEnvError(Exception):
    """Base class for every ``unigym-rl`` task-environment error."""


class TaskConfigError(TaskEnvError):
    """The run or task configuration is incompatible with ``TaskEnv``.

    Typical causes: the run is not in ``termination_mode: external``, the
    runner auto-reset flags are not caller-owned, or no control period can be
    resolved.
    """


class TaskContractError(TaskEnvError):
    """A task hook violated the (N, ...) tensor contract.

    Typical causes: a reward without shape ``(num_envs,)``, a termination mask
    that is not a boolean ``(num_envs,)`` tensor, or core terminations firing
    in external termination mode.
    """
