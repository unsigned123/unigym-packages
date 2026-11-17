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
    from unigym.robots.base import RobotProfile
    from unigym.run.api import Run
    from unigym.runtime.actions import OrderedActionSpec, PolicyBinding
    from unigym.world import World

    from unigym_rl.gym import GymTaskEnv
    from unigym_rl.task import TaskEnv

__all__ = ["EnvSpec"]


@dataclass(frozen=True)
class EnvSpec:
    """Declarative input for an externally-terminated RL ``Run``.

    ``world``, ``robot``, ``action_spec`` and ``policies`` are explicit: a
    complete core :class:`~unigym.robots.base.RobotProfile`, an
    :class:`~unigym.runtime.actions.OrderedActionSpec` and a non-empty tuple of
    :class:`~unigym.runtime.actions.PolicyBinding`.  This class never builds a
    World and never infers the action layout from the ``body`` joint set.
    ``make_run`` returns a fresh, lazy ``Run`` on every call, and the caller
    owns its lifecycle.  ``episode_length_s=None`` (the default) disables the
    core episode clock: task code, normally ``max_steps``, is the only time
    authority.
    """

    world: World
    robot: RobotProfile
    action_spec: OrderedActionSpec
    policies: tuple[PolicyBinding, ...]
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
        from unigym.robots.base import RobotProfile
        from unigym.runtime.actions import OrderedActionSpec, PolicyBinding

        if not isinstance(self.robot, RobotProfile):
            raise TaskConfigError(
                f"EnvSpec: robot must be a complete RobotProfile, got "
                f"{type(self.robot).__name__}"
            )
        if not isinstance(self.action_spec, OrderedActionSpec):
            raise TaskConfigError(
                f"EnvSpec: action_spec must be an OrderedActionSpec, got "
                f"{type(self.action_spec).__name__} | hint: declare the action "
                f"groups explicitly; they are never inferred from the body set"
            )
        if not isinstance(self.policies, tuple) or not self.policies:
            raise TaskConfigError("EnvSpec: policies must be a non-empty tuple of PolicyBinding")
        for index, binding in enumerate(self.policies):
            if not isinstance(binding, PolicyBinding):
                raise TaskConfigError(
                    f"EnvSpec: policies[{index}] must be a PolicyBinding, got "
                    f"{type(binding).__name__}"
                )
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
            action_spec=self.action_spec,
            policies=self.policies,
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
