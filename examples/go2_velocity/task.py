"""Flat-ground Go2 velocity-tracking smoke over the ``unigym_rl`` task layer.

The robot profile comes from the installed ``unigym-robots`` pack
(``release:go2-mjcf``), and the flat-ground world is the Menagerie
``scene.xml`` shipped beside the downloaded ``go2.xml``.  The world owns the
Go2 asset, so the task profile copied for the run carries ``assets=None``;
otherwise the same MJCF would be attached twice.

The MuJoCo backend requires an explicit ``world_base``.  A bare
``World().flat_ground()`` therefore cannot build (and this core checkout also
lacks the default inline material texture), so this smoke resolves the
downloaded Menagerie scene instead.  No asset is downloaded here.

This module is a demo, not package API: ``examples`` is repo-only and is never
installed into the ``unigym-rl`` wheel.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from unigym.packs import require
from unigym.runtime.actions import OrderedActionGroup, OrderedActionSpec, PolicyBinding
from unigym.world import World

from unigym_rl import EnvSpec, TaskConfigError, TaskContext, TaskEnv, TaskState, Transition

if TYPE_CHECKING:
    from unigym.robots.base import RobotProfile
    from unigym.run.api import Run

    from unigym_rl.gym import GymTaskEnv

__all__ = [
    "ACTION_JOINTS",
    "COMMAND_RANGES",
    "GO2_ACTION_GROUP",
    "GO2_ACTION_SPEC",
    "GO2_ARTICULATION",
    "GO2_DEFAULT_JOINT_POS",
    "GO2_JOINTS",
    "GO2_POLICIES",
    "GO2_PROFILE_ID",
    "LOW_HEIGHT_LIMIT",
    "OBS_DIM",
    "TILT_GRAVITY_Z",
    "Go2VelocityTask",
    "go2_profile",
    "go2_scene_path",
    "make_go2_velocity_gym",
    "make_go2_velocity_spec",
    "make_go2_velocity_task",
]

GO2_PROFILE_ID = "go2"
GO2_ARTICULATION = "go2"
GO2_ACTION_GROUP = "legs"

GO2_JOINTS = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)

GO2_DEFAULT_JOINT_POS = (
    0.0,
    0.8,
    -1.5,
    0.0,
    0.8,
    -1.5,
    0.0,
    0.8,
    -1.5,
    0.0,
    0.8,
    -1.5,
)

OBS_DIM = 48

#: ``(low, high)`` for ``[vx, vy, yaw_rate]`` sampled on every reset.
COMMAND_RANGES = ((-0.6, 0.8), (-0.4, 0.4), (-0.8, 0.8))

TILT_GRAVITY_Z = -0.5
LOW_HEIGHT_LIMIT = 0.18

LIN_TRACKING_WEIGHT = 1.5
YAW_TRACKING_WEIGHT = 0.5
ALIVE_WEIGHT = 0.5
ACTION_RATE_WEIGHT = 0.01
LIN_TRACKING_SIGMA = 0.25
YAW_TRACKING_SIGMA = 0.25

# The action group keeps its physical name (``legs``) while the articulation is
# ``go2``: recorded state keys are namespaced by the articulation
# (``articulation/go2/...``), never by the action-group name.
ACTION_JOINTS = GO2_JOINTS

GO2_ACTION_SPEC = OrderedActionSpec(
    (
        OrderedActionGroup(
            name=GO2_ACTION_GROUP,
            articulation=GO2_ARTICULATION,
            joints=GO2_JOINTS,
            target="position",
            role="body",
        ),
    )
)

GO2_POLICIES = (PolicyBinding(adapter="manual", groups=((GO2_ARTICULATION, GO2_ACTION_GROUP),)),)


def go2_profile() -> RobotProfile:
    """Load the installed ``unigym-robots`` Go2 profile without downloading."""
    packs = require("unigym_robots")
    return packs.pack("unigym_robots").robot(GO2_PROFILE_ID)


def go2_scene_path() -> Path:
    """Return the downloaded Menagerie flat-ground scene beside ``go2.xml``."""
    try:
        from unigym_robots import asset_path
    except ImportError as exc:  # pragma: no cover - ``go2_profile`` runs first
        raise TaskConfigError(
            "Go2 assets need the unigym-robots package; install it and run "
            "`unigym-robots download --set mjcf-v0`"
        ) from exc
    scene = Path(asset_path("go2-mjcf")).with_name("scene.xml")
    if not scene.is_file():
        raise TaskConfigError(
            f"Go2 scene asset is missing at {scene}; download it with "
            "`unigym-robots download --set mjcf-v0` (never auto-downloaded here)"
        )
    return scene


def _world_owned_profile(profile: RobotProfile) -> RobotProfile:
    """Drop the profile asset because the world scene already owns it."""
    return dataclasses.replace(profile, assets=None)


def make_go2_velocity_spec(
    *,
    num_envs: int,
    device: str | None = None,
    seed: int = 0,
) -> EnvSpec:
    """Build the flat-ground Go2 ``EnvSpec`` from the installed pack and assets."""
    profile = go2_profile()
    scene = go2_scene_path()
    return EnvSpec(
        world=World.base(mjcf=scene),
        robot=_world_owned_profile(profile),
        action_spec=GO2_ACTION_SPEC,
        policies=GO2_POLICIES,
        num_envs=num_envs,
        device=device,
        seed=seed,
        # external mode: no core episode clock; the task's max_steps owns
        # truncation.
        episode_length_s=None,
        no_artifacts=True,
        inspect=True,
    )


class Go2VelocityTask(TaskEnv):
    """Flat-ground Go2 velocity tracking with task-owned episode state.

    Observation (48 columns, all read from the core ``state.extra`` record):

    - base linear velocity in the body frame (3)
    - base angular velocity in the body frame (3)
    - projected gravity in the body frame (3)
    - joint positions relative to the profile standby pose (12)
    - joint velocities (12)
    - previous action (12)
    - command ``[vx, vy, yaw_rate]`` (3)

    Reward is an exponential linear-velocity tracker plus an exponential
    yaw-rate tracker, a constant alive bonus and an action-rate penalty.
    ``terminated`` fires on a gravity tilt or a low base height; ``truncated``
    fires when this env's own step counter reaches ``max_steps``.  The runner
    never auto-resets; ``TaskEnv`` resets only the done rows.
    """

    obs_dim = OBS_DIM

    def __init__(
        self,
        run: Run,
        *,
        max_steps: int = 500,
        seed: int = 0,
    ) -> None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise TaskConfigError(
                f"Go2VelocityTask: max_steps must be an int >= 1, got {max_steps!r}"
            )
        self._max_steps = int(max_steps)
        self._rng = torch.Generator().manual_seed(int(seed))
        self._default_joint_pos: torch.Tensor | None = None
        super().__init__(run)
        self._command = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._prev_action = torch.zeros(
            (self.num_envs, len(GO2_JOINTS)), dtype=torch.float32, device=self.device
        )
        self._action_rate = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._step_count = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

    # ------------------------------------------------------------------ #
    # read-only task state (tests/debugging)
    # ------------------------------------------------------------------ #
    @property
    def command(self) -> torch.Tensor:
        """The current per-env ``[vx, vy, yaw_rate]`` command buffer."""
        return self._command

    @property
    def step_count(self) -> torch.Tensor:
        """The env-local truncation counters."""
        return self._step_count

    # ------------------------------------------------------------------ #
    # hooks
    # ------------------------------------------------------------------ #
    def on_bind(self, context: TaskContext) -> None:
        """Resolve the Go2 joints and standby pose on the live model/profile."""
        joints = context.instance.live.joints
        missing = [joint for joint in GO2_JOINTS if joint not in joints]
        if missing:
            raise TaskConfigError(
                f"Go2VelocityTask: compiled articulation is missing joints {missing}"
            )
        profile = context.instance.profile
        if profile is None:
            raise TaskConfigError("Go2VelocityTask: the run has no robot profile")
        full_joint_names = tuple(profile.full_joint_names)
        if full_joint_names != GO2_JOINTS:
            raise TaskConfigError(
                "Go2VelocityTask: the profile full joint order no longer matches "
                f"GO2_JOINTS: {full_joint_names}"
            )
        self._default_joint_pos = torch.tensor(
            [float(profile.default_joint_pos[joint]) for joint in GO2_JOINTS],
            dtype=torch.float32,
            device=self.device,
        )

    def _extra_column(self, state: TaskState, leaf: str) -> torch.Tensor:
        """Read one ``articulation/go2/<leaf>`` tensor from the core extra record."""
        key = f"articulation/{GO2_ARTICULATION}/{leaf}"
        value = state.extra.get(key)
        if not isinstance(value, torch.Tensor):
            raise TaskConfigError(
                f"Go2VelocityTask: state extra key {key!r} is missing or not a tensor"
            )
        value = value.to(device=self.device, dtype=torch.float32)
        if value.ndim != 2 or value.shape[0] != self.num_envs:
            raise TaskConfigError(
                f"Go2VelocityTask: state extra key {key!r} must be (N, ...), "
                f"got shape {tuple(value.shape)} for N={self.num_envs}"
            )
        return value

    def _width(self, value: torch.Tensor, width: int, leaf: str) -> torch.Tensor:
        if value.shape[1] != width:
            raise TaskConfigError(
                f"Go2VelocityTask: state extra {leaf!r} must have {width} columns, "
                f"got {value.shape[1]}"
            )
        return value

    def observe(self, state: TaskState) -> torch.Tensor:
        """Return the 48-column observation described in the class docstring."""
        base_lin_vel = self._width(self._extra_column(state, "base_lin_vel_b"), 3, "base_lin_vel_b")
        base_ang_vel = self._width(self._extra_column(state, "base_ang_vel_b"), 3, "base_ang_vel_b")
        gravity = self._width(self._extra_column(state, "gravity_b"), 3, "gravity_b")
        joint_pos = self._width(
            self._extra_column(state, "joint_pos"), len(GO2_JOINTS), "joint_pos"
        )
        joint_vel = self._width(
            self._extra_column(state, "joint_vel"), len(GO2_JOINTS), "joint_vel"
        )
        if self._default_joint_pos is None:  # pragma: no cover - on_bind always runs
            raise TaskConfigError("Go2VelocityTask: standby pose was never resolved")
        return torch.cat(
            (
                base_lin_vel,
                base_ang_vel,
                gravity,
                joint_pos - self._default_joint_pos,
                joint_vel,
                self._prev_action,
                self._command,
            ),
            dim=1,
        )

    def reward(self, transition: Transition) -> torch.Tensor:
        """Velocity/yaw tracking, alive bonus and action-rate penalty."""
        base_lin_vel = self._extra_column(transition.info, "base_lin_vel_b")
        base_ang_vel = self._extra_column(transition.info, "base_ang_vel_b")
        linear_error = ((base_lin_vel[:, :2] - self._command[:, :2]) ** 2).sum(dim=1)
        yaw_error = (base_ang_vel[:, 2] - self._command[:, 2]) ** 2
        return (
            LIN_TRACKING_WEIGHT * torch.exp(-linear_error / LIN_TRACKING_SIGMA)
            + YAW_TRACKING_WEIGHT * torch.exp(-yaw_error / YAW_TRACKING_SIGMA)
            + ALIVE_WEIGHT
            - ACTION_RATE_WEIGHT * self._action_rate
        )

    def terminated(self, transition: Transition) -> torch.Tensor:
        """Terminate on a gravity tilt or a low base height."""
        return self._tilt_mask(transition) | self._low_height_mask(transition)

    def truncated(self, transition: Transition) -> torch.Tensor:
        """Truncate when this env's own step counter reaches ``max_steps``."""
        del transition
        return self._step_count >= self._max_steps

    def termination_reasons(
        self,
        transition: Transition,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Named masks for tilt, low height and the time limit."""
        return {
            "tilt": self._tilt_mask(transition) & terminated,
            "low_height": self._low_height_mask(transition) & terminated,
            "time_limit": truncated,
        }

    def on_step(self, action: torch.Tensor, state: TaskState) -> None:
        """Record the action-rate penalty, previous action and step clock."""
        del state
        delta = action - self._prev_action
        self._action_rate = (delta * delta).sum(dim=1)
        self._prev_action = action.detach().clone().to(self.device, torch.float32)
        self._step_count += 1

    def on_reset(self, env_ids: torch.Tensor, state: TaskState) -> None:
        """Sample a new command and clear this env's task-owned episode state."""
        del state
        rows = int(env_ids.numel())
        if rows:
            unit = torch.rand((rows, len(COMMAND_RANGES)), generator=self._rng, dtype=torch.float32)
            command = torch.stack(
                [
                    low + (high - low) * unit[:, index]
                    for index, (low, high) in enumerate(COMMAND_RANGES)
                ],
                dim=1,
            )
            self._command[env_ids] = command.to(self.device)
            self._prev_action[env_ids] = 0.0
            self._action_rate[env_ids] = 0.0
            self._step_count[env_ids] = 0

    # ------------------------------------------------------------------ #
    # masks
    # ------------------------------------------------------------------ #
    def _tilt_mask(self, transition: Transition) -> torch.Tensor:
        return self._extra_column(transition.info, "gravity_b")[:, 2] > TILT_GRAVITY_Z

    def _low_height_mask(self, transition: Transition) -> torch.Tensor:
        return self._extra_column(transition.info, "base_pos_w")[:, 2] < LOW_HEIGHT_LIMIT


def make_go2_velocity_task(
    *,
    num_envs: int,
    max_steps: int,
    device: str | None = None,
    seed: int = 0,
) -> Go2VelocityTask:
    """Build a fresh real-MuJoCo :class:`Go2VelocityTask`."""
    spec = make_go2_velocity_spec(num_envs=num_envs, device=device, seed=seed)
    task = spec.make_task(Go2VelocityTask, max_steps=max_steps, seed=seed)
    assert isinstance(task, Go2VelocityTask)
    return task


def make_go2_velocity_gym(
    *,
    num_envs: int,
    max_steps: int,
    device: str | None = None,
    seed: int = 0,
) -> GymTaskEnv:
    """Wrap the demo task in the optional Gymnasium face (lazy import)."""
    task = make_go2_velocity_task(
        num_envs=num_envs,
        max_steps=max_steps,
        device=device,
        seed=seed,
    )
    try:
        from unigym_rl.gym import GymTaskEnv
    except Exception:
        task.close()
        task.run.close()
        raise
    return GymTaskEnv(task)
