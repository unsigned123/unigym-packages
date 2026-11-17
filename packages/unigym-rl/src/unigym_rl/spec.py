"""Programmatic, RL-owned run construction (R2.5).

``EnvSpec`` turns plain dataclass fields into a fresh core ``Run`` without a
yaml file.  Termination ownership is not configurable here: every run is built
with ``termination_mode="external"`` and both runner auto-reset flags false, so
the task layer stays the single owner of reset policy.  Construction stays
lazy: ``make_run`` only declares intent and never builds the backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from unigym_rl.errors import TaskConfigError

if TYPE_CHECKING:
    from unigym.run.api import Run
    from unigym.world import World

    from unigym_rl.gym import GymTaskEnv
    from unigym_rl.task import TaskEnv

__all__ = ["EnvSpec"]


@dataclass(frozen=True)
class EnvSpec:
    """Declarative input for an externally-terminated RL ``Run``.

    ``world`` and ``robot`` use the core programmatic declaration shape; this
    class is not a robot-profile builder.  ``make_run`` returns a fresh, lazy
    ``Run`` on every call, and the caller owns its lifecycle.
    """

    world: World
    robot: Mapping[str, object] | str
    num_envs: int = 1
    device: str | None = None
    backend: str = "mujoco"
    adapter: str | Mapping[str, object] = "manual"
    seed: int = 0
    steps: int | None = None
    episode_length_s: float | None = None
    no_artifacts: bool = True
    inspect: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.num_envs, bool) or not isinstance(self.num_envs, int):
            raise TaskConfigError(
                f"EnvSpec: num_envs must be an int, got {type(self.num_envs).__name__}"
            )
        if self.num_envs < 1:
            raise TaskConfigError(f"EnvSpec: num_envs must be >= 1, got {self.num_envs}")

    def make_run(self) -> Run:
        """Build a fresh, lazy ``Run`` with task-owned termination and resets."""
        from unigym.run.api import Run

        return Run(
            world=self.world,
            robot=self.robot,
            backend=self.backend,
            num_envs=self.num_envs,
            device=self.device,
            adapter=self.adapter,
            seed=self.seed,
            steps=self.steps,
            episode_length_s=self.episode_length_s,
            no_artifacts=self.no_artifacts,
            inspect=self.inspect,
            termination_mode="external",
            runner={"auto_reset": False, "reset_on_termination": False},
        )

    def make_task(self, task_cls: type[TaskEnv], **task_kwargs: object) -> TaskEnv:
        """Construct *task_cls* over a fresh run, validating the class first."""
        from unigym_rl.task import TaskEnv

        if not isinstance(task_cls, type) or not issubclass(task_cls, TaskEnv):
            raise TaskConfigError(f"EnvSpec.make_task expects a TaskEnv subclass, got {task_cls!r}")
        return task_cls(self.make_run(), **task_kwargs)

    def make_gym(self, task_cls: type[TaskEnv], **task_kwargs: object) -> GymTaskEnv:
        """Wrap a fresh task in ``GymTaskEnv``, importing gymnasium lazily."""
        from unigym_rl.gym import GymTaskEnv

        return GymTaskEnv(self.make_task(task_cls, **task_kwargs))
