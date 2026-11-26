"""Flat-ground Go2 velocity-tracking smoke over the ``unigym_rl`` task layer.

The robot profile comes from the installed ``unigym-robots`` pack
(``release:go2-mjcf``) and owns the Go2 asset.  The world is the
backend-neutral :meth:`unigym.world.World.flat` ground: it owns no robot and
no explicit ``world_base``.  The run layer is the single robot owner
(profile + spawn + explicit action layout + policy binding); the backend only
selects the profile's asset format.

This module is a demo, not package API: ``examples`` is repo-only and is never
installed into the ``unigym-rl`` wheel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from unigym.packs import PackLoader, require
from unigym.runtime.actions import OrderedActionGroup, OrderedActionSpec, PolicyBinding
from unigym.world import World

from unigym_rl import (
    EnvSpec,
    TaskConfigError,
    TaskContext,
    TaskEnv,
    TaskState,
    Transition,
)

if TYPE_CHECKING:
    from unigym.env.obs import RobotObs
    from unigym.robots.base import RobotProfile
    from unigym.run.api import Run

    from unigym_rl.gym import GymTaskEnv

__all__ = [
    "ACTION_JOINTS",
    "COMMAND_RANGES",
    "FEET_AIR_TIME_THRESHOLD",
    "FEET_AIR_TIME_WEIGHT",
    "GO2_ACTION_GROUP",
    "GO2_ACTION_SPEC",
    "GO2_ARTICULATION",
    "GO2_CONTACTS",
    "GO2_DEFAULT_JOINT_POS",
    "GO2_FEET_BODIES",
    "GO2_JOINTS",
    "GO2_POLICIES",
    "GO2_PROFILE_ID",
    "LOW_HEIGHT_LIMIT",
    "OBS_DIM",
    "TILT_GRAVITY_Z",
    "GO2_SPAWN",
    "Go2VelocityTask",
    "go2_profile",
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

#: Canonical Go2 foot bodies: the four calf links (``profile.foot_bodies``).
GO2_FEET_BODIES = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")

#: Explicit run-level contact groups.  Every group owns the ``go2``
#: articulation, reports boolean contact in the world frame against a 1 N
#: threshold, and publishes ``contact/feet`` (N, 4) / ``contact/base`` (N, 1).
GO2_CONTACTS = (
    {
        "name": "feet",
        "sensor_refs": [
            {"owner": "robot", "articulation": GO2_ARTICULATION, "body": body}
            for body in GO2_FEET_BODIES
        ],
        "frame": "world",
        "format": "contact",
        "threshold": 1.0,
    },
    {
        "name": "base",
        "sensor_refs": [{"owner": "robot", "articulation": GO2_ARTICULATION, "body": "base"}],
        "frame": "world",
        "format": "contact",
        "threshold": 1.0,
    },
)

#: Run-owned robot root spawn (metres); the world never declares one.
GO2_SPAWN = {"pos": (0.0, 0.0, 0.4)}

LIN_TRACKING_WEIGHT = 1.5
YAW_TRACKING_WEIGHT = 0.5
ALIVE_WEIGHT = 0.5
ACTION_RATE_WEIGHT = 0.01
LIN_TRACKING_SIGMA = 0.25
YAW_TRACKING_SIGMA = 0.25

#: Official-style air-time shaping: one bonus per first touchdown, only while
#: the planar command is non-trivial.
FEET_AIR_TIME_WEIGHT = 0.25
FEET_AIR_TIME_THRESHOLD = 0.5
COMMAND_XY_EPS = 0.1

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
    loader = PackLoader.from_pack_set(packs)
    return loader.robot_profile("unigym_robots", GO2_PROFILE_ID)


def make_go2_velocity_spec(
    *,
    num_envs: int,
    device: str | None = None,
    seed: int = 0,
    backend: str = "mujoco",
) -> EnvSpec:
    """Build the flat-ground Go2 ``EnvSpec`` from the installed pack.

    The backend is passed only to ``EnvSpec``: it selects the profile asset
    format and the backend build lane.  Task code stays backend-agnostic.
    """
    return EnvSpec(
        world=World.flat(),
        robot=go2_profile(),
        spawn=GO2_SPAWN,
        action_spec=GO2_ACTION_SPEC,
        policies=GO2_POLICIES,
        observations={
            "robot": [GO2_ARTICULATION],
            "contact": ["feet", "base"],
            "command": True,
            "prev_action": True,
        },
        contacts=GO2_CONTACTS,
        num_envs=num_envs,
        device=device,
        backend=backend,
        seed=seed,
        # external mode: no core episode clock; the task's max_steps owns
        # truncation and TaskEnv ends done rows through runner.terminate.
        no_artifacts=True,
        inspect=False,
    )


class Go2VelocityTask(TaskEnv):
    """Flat-ground Go2 velocity tracking with task-owned episode state.

    Observation (48 columns, all read from the typed ``state.obs`` views):

    - base linear velocity in the body frame (3)
    - base angular velocity in the body frame (3)
    - projected gravity in the body frame (3)
    - joint positions relative to the profile standby pose (12)
    - joint velocities (12)
    - previous action (12)
    - command ``[vx, vy, yaw_rate]`` (3)

    Reward is an exponential linear-velocity tracker plus an exponential
    yaw-rate tracker, a constant alive bonus, an action-rate penalty and the
    official-style ``feet_air_time`` first-touchdown bonus (weight ``0.25``,
    threshold ``0.5`` s, zero while ``||command_xy|| <= 0.1``).
    ``terminated`` fires on a gravity tilt, a low base height or any ``base``
    contact; ``truncated`` fires when this env's own step counter reaches
    ``max_steps``.  The runner never auto-resets; ``TaskEnv`` resets only the
    done rows.  This is not a convergence or official-parity claim.
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
        self._init_contact_state()

    def _init_contact_state(self) -> None:
        """Allocate the task-local per-foot contact timers (seconds)."""
        width = len(GO2_FEET_BODIES)
        self._air_time = torch.zeros(
            (self.num_envs, width), dtype=torch.float32, device=self.device
        )
        self._contact_time = torch.zeros(
            (self.num_envs, width), dtype=torch.float32, device=self.device
        )
        self._last_air_time = torch.zeros(
            (self.num_envs, width), dtype=torch.float32, device=self.device
        )
        self._prev_contact = torch.zeros(
            (self.num_envs, width), dtype=torch.bool, device=self.device
        )
        self._first_contact = torch.zeros(
            (self.num_envs, width), dtype=torch.bool, device=self.device
        )

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

    @property
    def air_time(self) -> torch.Tensor:
        """Seconds each foot has been airborne (``(num_envs, 4)``)."""
        return self._air_time

    @property
    def contact_time(self) -> torch.Tensor:
        """Seconds each foot has been in contact (``(num_envs, 4)``)."""
        return self._contact_time

    @property
    def last_air_time(self) -> torch.Tensor:
        """Air time of each foot's most recent completed flight phase."""
        return self._last_air_time

    @property
    def first_contact(self) -> torch.Tensor:
        """Per-foot mask of the touchdowns observed in the last step."""
        return self._first_contact

    # ------------------------------------------------------------------ #
    # hooks
    # ------------------------------------------------------------------ #
    def on_bind(self, context: TaskContext) -> None:
        """Resolve the Go2 joints and standby pose from the profile/live model.

        The live-model joint check is backend-specific: ``mujoco``/``mjwarp``
        expose it, while ``isaaclab2`` publishes the compiled articulation
        only through the typed state record.  On the latter the profile joint
        order (validated immediately below) is the source of truth.
        """
        profile = context.instance.profile
        if profile is None:
            raise TaskConfigError("Go2VelocityTask: the run has no robot profile")
        full_joint_names = tuple(profile.full_joint_names)
        if full_joint_names != GO2_JOINTS:
            raise TaskConfigError(
                "Go2VelocityTask: the profile full joint order no longer matches "
                f"GO2_JOINTS: {full_joint_names}"
            )
        try:
            joints = context.instance.live.joints
        except NotImplementedError:
            joints = full_joint_names
        missing = [joint for joint in GO2_JOINTS if joint not in joints]
        if missing:
            raise TaskConfigError(
                f"Go2VelocityTask: compiled articulation is missing joints {missing}"
            )
        self._default_joint_pos = torch.tensor(
            [float(profile.default_joint_pos[joint]) for joint in GO2_JOINTS],
            dtype=torch.float32,
            device=self.device,
        )

    def _joints(self, robot: RobotObs, leaf: str) -> torch.Tensor:
        """One joint leaf, checked against the Go2 joint count and moved to device."""
        value = getattr(robot, leaf)
        if value.shape[1] != len(GO2_JOINTS):
            raise TaskConfigError(
                f"Go2VelocityTask: articulation {GO2_ARTICULATION!r} {leaf!r} "
                f"must have {len(GO2_JOINTS)} columns, got {value.shape[1]}"
            )
        return value.to(device=self.device, dtype=torch.float32)

    def _vector(self, robot: RobotObs, leaf: str) -> torch.Tensor:
        """One base/gravity leaf moved to the task device."""
        return getattr(robot, leaf).to(device=self.device, dtype=torch.float32)

    def _slot(self, slot, name: str) -> torch.Tensor:
        """One typed runtime slot moved to the task device."""
        value = slot.value
        if value is None:
            raise TaskConfigError(
                f"Go2VelocityTask: observation slot {name!r} was not published | hint: "
                "declare it in EnvSpec(observations=...) and publish it from the task"
            )
        return value.to(device=self.device, dtype=torch.float32)

    def observe(self, state: TaskState) -> torch.Tensor:
        """Return the 48-column observation described in the class docstring."""
        robot = state.robot(GO2_ARTICULATION)
        base_lin_vel = self._vector(robot, "base_lin_vel_b")
        base_ang_vel = self._vector(robot, "base_ang_vel_b")
        gravity = self._vector(robot, "gravity_b")
        joint_pos = self._joints(robot, "joint_pos")
        joint_vel = self._joints(robot, "joint_vel")
        prev_action = self._slot(state.obs.prev_action, "prev_action")
        command = self._slot(state.obs.command, "command")
        if self._default_joint_pos is None:  # pragma: no cover - on_bind always runs
            raise TaskConfigError("Go2VelocityTask: standby pose was never resolved")
        return torch.cat(
            (
                base_lin_vel,
                base_ang_vel,
                gravity,
                joint_pos - self._default_joint_pos,
                joint_vel,
                prev_action,
                command,
            ),
            dim=1,
        )

    def reward(self, transition: Transition) -> torch.Tensor:
        """Velocity/yaw tracking, alive bonus, action-rate and air-time terms."""
        robot = transition.info.robot(GO2_ARTICULATION)
        base_lin_vel = self._vector(robot, "base_lin_vel_b")
        base_ang_vel = self._vector(robot, "base_ang_vel_b")
        linear_error = ((base_lin_vel[:, :2] - self._command[:, :2]) ** 2).sum(dim=1)
        yaw_error = (base_ang_vel[:, 2] - self._command[:, 2]) ** 2
        return (
            LIN_TRACKING_WEIGHT * torch.exp(-linear_error / LIN_TRACKING_SIGMA)
            + YAW_TRACKING_WEIGHT * torch.exp(-yaw_error / YAW_TRACKING_SIGMA)
            + ALIVE_WEIGHT
            - ACTION_RATE_WEIGHT * self._action_rate
            + FEET_AIR_TIME_WEIGHT * self._feet_air_time_reward()
        )

    def _feet_air_time_reward(self) -> torch.Tensor:
        """Official-style first-touchdown air-time bonus, zero on zero command.

        Reads the per-step ``_last_air_time`` / ``_first_contact`` state
        computed in :meth:`on_step`; it never re-reads the simulator.
        """
        reward = (self._last_air_time - FEET_AIR_TIME_THRESHOLD) * self._first_contact
        reward = reward.sum(dim=1)
        moving = self._command[:, :2].norm(dim=1) > COMMAND_XY_EPS
        return reward * moving.to(reward.dtype)

    def terminated(self, transition: Transition) -> torch.Tensor:
        """Terminate on a gravity tilt, a low base or any base contact."""
        return (
            self._tilt_mask(transition)
            | self._low_height_mask(transition)
            | (self._base_contact_mask(transition))
        )

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
        """Named masks for tilt, low height, base contact and the time limit."""
        return {
            "tilt": self._tilt_mask(transition) & terminated,
            "low_height": self._low_height_mask(transition) & terminated,
            "base_contact": self._base_contact_mask(transition) & terminated,
            "time_limit": truncated,
        }

    def on_step(self, action: torch.Tensor, state: TaskState) -> None:
        """Record action rate, previous action, step clock and foot timers."""
        delta = action - self._prev_action
        self._action_rate = (delta * delta).sum(dim=1)
        self._prev_action = action.detach().clone().to(self.device, torch.float32)
        self._step_count += 1
        self.env.obs.set_command(self._command)
        self.env.obs.set_prev_action(self._prev_action)
        contact = self._contact_tensor(state, "feet", len(GO2_FEET_BODIES))
        self._update_contact_timers(contact)

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
            self._air_time[env_ids] = 0.0
            self._contact_time[env_ids] = 0.0
            self._last_air_time[env_ids] = 0.0
            self._prev_contact[env_ids] = False
            self._first_contact[env_ids] = False
        self.env.obs.set_command(self._command)
        self.env.obs.set_prev_action(self._prev_action)

    # ------------------------------------------------------------------ #
    # contact timers
    # ------------------------------------------------------------------ #
    def _contact_tensor(self, state: TaskState, name: str, width: int) -> torch.Tensor:
        """Read one declared contact group as a bool ``(N, width)``."""
        from unigym.env.obs import ObsContractError

        try:
            value = state.obs.contact(name).value
        except ObsContractError as exc:
            raise TaskConfigError(
                f"Go2VelocityTask: contact/{name} is not "
                "declared/published | hint: declare the matching group through "
                "EnvSpec(contacts=...) and EnvSpec(observations={'contact': [...]})"
            ) from exc
        if value.dtype is not torch.bool:
            raise TaskConfigError(
                f"Go2VelocityTask: contact/{name} must be bool (contact "
                f"format), got {value.dtype}"
            )
        expected = (self.num_envs, width)
        if tuple(value.shape) != expected:
            raise TaskConfigError(
                f"Go2VelocityTask: contact/{name} must have shape {expected}, "
                f"got {tuple(value.shape)}"
            )
        return value

    def _update_contact_timers(self, contact: torch.Tensor) -> None:
        """Advance the air/contact timers from this step's foot contact mask."""
        first_contact = contact & ~self._prev_contact
        dt = self.control_dt
        self._last_air_time[first_contact] = self._air_time[first_contact] + dt
        self._air_time = torch.where(contact, torch.zeros_like(self._air_time), self._air_time + dt)
        self._contact_time = torch.where(
            contact, self._contact_time + dt, torch.zeros_like(self._contact_time)
        )
        self._prev_contact = contact
        self._first_contact = first_contact

    # ------------------------------------------------------------------ #
    # masks
    # ------------------------------------------------------------------ #
    def _tilt_mask(self, transition: Transition) -> torch.Tensor:
        robot = transition.info.robot(GO2_ARTICULATION)
        return self._vector(robot, "gravity_b")[:, 2] > TILT_GRAVITY_Z

    def _low_height_mask(self, transition: Transition) -> torch.Tensor:
        robot = transition.info.robot(GO2_ARTICULATION)
        return self._vector(robot, "base_pos_w")[:, 2] < LOW_HEIGHT_LIMIT

    def _base_contact_mask(self, transition: Transition) -> torch.Tensor:
        contact = self._contact_tensor(transition.info, "base", 1)
        return contact.any(dim=1)


def make_go2_velocity_task(
    *,
    num_envs: int,
    max_steps: int,
    device: str | None = None,
    seed: int = 0,
    backend: str = "mujoco",
) -> Go2VelocityTask:
    """Build a fresh :class:`Go2VelocityTask` on the requested backend."""
    spec = make_go2_velocity_spec(
        num_envs=num_envs,
        device=device,
        seed=seed,
        backend=backend,
    )
    task = spec.make_task(Go2VelocityTask, max_steps=max_steps, seed=seed)
    assert isinstance(task, Go2VelocityTask)
    return task


def make_go2_velocity_gym(
    *,
    num_envs: int,
    max_steps: int,
    device: str | None = None,
    seed: int = 0,
    backend: str = "mujoco",
) -> GymTaskEnv:
    """Wrap the demo task in the optional Gymnasium face (lazy import)."""
    task = make_go2_velocity_task(
        num_envs=num_envs,
        max_steps=max_steps,
        device=device,
        seed=seed,
        backend=backend,
    )
    try:
        from unigym_rl.gym import GymTaskEnv
    except Exception:
        task.close()
        task.run.close()
        raise
    return GymTaskEnv(task)
