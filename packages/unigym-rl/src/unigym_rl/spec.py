"""Programmatic, RL-owned run construction (R2.5).

``EnvSpec`` turns plain dataclass fields into a fresh core ``Run`` without a
yaml file.  Termination ownership is not configurable here: every run is built
with ``termination_mode="external"`` and both runner auto-reset flags false, so
the task layer stays the single owner of reset policy.  Construction stays
lazy: ``make_run`` only declares intent and never builds the backend.
"""

from __future__ import annotations

import copy
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
    :class:`~unigym.world.World` (for example ``World.flat()``), a complete
    core :class:`~unigym.robots.base.RobotProfile`, an
    :class:`~unigym.runtime.actions.OrderedActionSpec` and a non-empty tuple of
    :class:`~unigym.runtime.actions.PolicyBinding`.  ``spawn`` is the
    run-owned robot root spawn (``{"pos": (x, y, z), "yaw": radians?}``).
    This class never builds a World, never asks the World for a robot, and
    never infers the action layout from the ``body`` joint set.
    ``make_run`` returns a fresh, lazy ``Run`` on every call, and the caller
    owns its lifecycle.  ``episode_length_s=None`` (the default) disables the
    core episode clock: task code, normally ``max_steps``, is the only time
    authority.

    ``observations`` is the run's top-level ``observations:`` declaration
    (``{"robot": ["go2"], "contact": ["feet"], "command": True, ...}``).  Its
    shape is validated here; names are resolved by the core builder against
    the declared articulations/contacts/props/cameras/sites.

    ``contacts`` is an explicit tuple of run-level contact group mappings.  It
    is passed through unchanged to the core ``Run`` top-level ``contacts:``
    schema (``list(self.contacts)``), which declares the contact groups a
    ``observations.contact`` selection can name.  An empty tuple (the default)
    declares no contact groups.
    """

    world: World
    robot: RobotProfile
    action_spec: OrderedActionSpec
    policies: tuple[PolicyBinding, ...]
    observations: Mapping[str, object]
    spawn: Mapping[str, object] | None = None
    contacts: tuple[Mapping[str, object], ...] = ()
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
        object.__setattr__(self, "observations", self._validate_observations())
        if self.spawn is not None and not isinstance(self.spawn, Mapping):
            raise TaskConfigError(
                "EnvSpec: spawn must be a mapping ({pos, yaw?}) or None, got "
                f"{type(self.spawn).__name__} | hint: the run owns the robot "
                "spawn; the world never declares one"
            )
        object.__setattr__(self, "contacts", self._validate_contacts())
        if isinstance(self.num_envs, bool) or not isinstance(self.num_envs, int):
            raise TaskConfigError(
                f"EnvSpec: num_envs must be an int, got {type(self.num_envs).__name__}"
            )
        if self.num_envs < 1:
            raise TaskConfigError(f"EnvSpec: num_envs must be >= 1, got {self.num_envs}")

    def _validate_observations(self) -> dict[str, object]:
        """Validate the ``observations:`` declaration shape, not its names.

        Name resolution is the core run builder's job: it knows the declared
        articulations, contacts, props, cameras and sites.
        """
        from unigym.env.obs import ObsConfigError, ObservationConfig

        if not isinstance(self.observations, Mapping):
            raise TaskConfigError(
                "EnvSpec: observations must be a mapping such as "
                f"{{'robot': ['go2']}}, got {type(self.observations).__name__}"
            )
        try:
            config = ObservationConfig.from_mapping(self.observations)
        except ObsConfigError as exc:
            raise TaskConfigError(f"EnvSpec: invalid observations declaration: {exc}") from exc
        return config.to_mapping()

    def _validate_contacts(self) -> tuple[dict[str, object], ...]:
        """Validate the contact declaration against the core contact schema.

        The outer container must be a tuple (no silent list coercion) and every
        entry must be a non-empty mapping.  Entries with plain body lists are
        parsed by the core ``ContactGroupSpec`` immediately; entries still in
        the declarative reference form (``{contact_group: ...}`` /
        ``{prop: ...}``) are only shape-checked here because they can only be
        resolved against the built run's body inventory.
        """
        from unigym.contacts.base import ContactGroupSpec

        if not isinstance(self.contacts, tuple):
            raise TaskConfigError(
                "EnvSpec: contacts must be a tuple of mappings, got "
                f"{type(self.contacts).__name__} | hint: this is the run "
                "top-level `contacts:` list (one mapping per contact group)"
            )
        normalized: list[dict[str, object]] = []
        for index, entry in enumerate(self.contacts):
            if not isinstance(entry, Mapping):
                raise TaskConfigError(
                    f"EnvSpec: contacts[{index}] must be a mapping (one run "
                    f"top-level `contacts:` entry), got {type(entry).__name__}"
                )
            if not entry:
                raise TaskConfigError(
                    f"EnvSpec: contacts[{index}] must not be empty | hint: a "
                    "contact group needs at least name, sensor_bodies and frame"
                )
            payload = copy.deepcopy(dict(entry))
            reference_form = any(
                isinstance(payload.get(key), Mapping) for key in ("sensor_bodies", "filter_bodies")
            )
            if not reference_form:
                try:
                    ContactGroupSpec.from_yaml(payload)
                except (TypeError, ValueError) as exc:
                    raise TaskConfigError(
                        f"EnvSpec: contacts[{index}] is not a valid run "
                        f"top-level `contacts:` entry: {exc}"
                    ) from exc
            normalized.append(payload)
        return tuple(normalized)

    def make_run(self) -> Run:
        """Build a fresh, lazy ``Run`` with task-owned termination and resets."""
        from unigym.run.api import Run

        return Run(
            world=self.world,
            robot=self.robot,
            spawn=self.spawn,
            observations=self.observations,
            contacts=list(self.contacts) if self.contacts else None,
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
