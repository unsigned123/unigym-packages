"""Optional Gymnasium ``VectorEnv`` face over :class:`~unigym_rl.task.TaskEnv`.

This module is the only place in ``unigym_rl`` that imports ``gymnasium``; the
package ``__init__`` exposes :class:`GymTaskEnv` lazily so importing
``unigym_rl`` never requires the optional dependency.

The wrapper is intentionally thin: it derives the spaces from the task's
declared ``obs_dim`` / ``action_dim``, forwards ``reset`` / ``step`` to the
task and converts the five-tuple leading arrays to NumPy.  ``info`` is passed
through untouched so the task-layer evidence (``final_observation``,
``final_info``, ``episode_end``, ``termination_reasons`` and ``state``) is not
lost.  Done rows are already reset by the task before the observation is
returned, so the declared autoreset mode is ``AutoresetMode.SAME_STEP``.
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
import torch

from unigym_rl.errors import TaskConfigError
from unigym_rl.task import TaskEnv

__all__ = ["GymTaskEnv"]


class GymTaskEnv(gymnasium.vector.VectorEnv):
    """Gymnasium vector environment wrapping one :class:`TaskEnv`.

    ``single_*_space`` describe one sub-environment, while
    ``observation_space`` / ``action_space`` are the batched ``(num_envs, ...)``
    spaces.  The torch device of the wrapped task is preserved as
    :attr:`device` so framework adapters (skrl) keep tensors on the run device.
    """

    metadata = {"autoreset_mode": gymnasium.vector.AutoresetMode.SAME_STEP}

    def __init__(self, task: TaskEnv) -> None:
        if not isinstance(task, TaskEnv):
            raise TaskConfigError(
                "GymTaskEnv expects a unigym_rl.task.TaskEnv instance, "
                f"got {type(task).__name__}"
            )
        if task.obs_dim is None:
            raise TaskConfigError(
                "GymTaskEnv requires TaskEnv.obs_dim to be known at construction "
                f"time; {type(task).__name__}.obs_dim is None | hint: declare "
                "obs_dim = <int> on the task class (or its base) before wrapping"
            )

        super().__init__()

        self.task = task
        self.num_envs = int(task.num_envs)
        self.device = task.device

        obs_dim = int(task.obs_dim)
        action_dim = int(task.action_dim)
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_envs, obs_dim), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_envs, action_dim), dtype=np.float32
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset every sub-environment and return ``(numpy obs, task info)``."""
        obs, info = self.task.reset(seed=seed, options=options)
        return self._to_numpy(obs), info

    def step(
        self, actions: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Step the wrapped task with a NumPy or torch action batch.

        The leading four values are NumPy; ``info`` is the task's dictionary
        unchanged so final-observation / episode-end evidence survives.
        """
        tensor = self._action_tensor(actions)
        obs, reward, terminated, truncated, info = self.task.step(tensor)
        return (
            self._to_numpy(obs),
            self._to_numpy(reward),
            self._to_numpy(terminated),
            self._to_numpy(truncated),
            info,
        )

    def close(self, **kwargs: Any) -> None:
        """Close the wrapped task once; the borrowed core Run stays open."""
        del kwargs
        if self.closed:
            return
        self.task.close()
        super().close()

    def render(self, *args: Any, **kwargs: Any) -> Any:
        """Rendering is not implemented by the task-env face."""
        del args, kwargs
        raise NotImplementedError(
            "GymTaskEnv does not implement render(); use the core run/scene "
            "viewers (or an explicit camera) for visuals"
        )

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _action_tensor(self, actions: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(actions, torch.Tensor):
            return actions.to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self.device)

    @staticmethod
    def _to_numpy(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy()
