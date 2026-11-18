"""Fixed-base MuJoCo CartPole demo over the ``unigym_rl`` task layer.

The demo builds one local :class:`~unigym.robots.base.RobotProfile` for the
self-contained MuJoCo Playground ``cartpole.xml`` asset.  The cart slider is the
only action column (a ``standard_pd`` position target); the pole hinge is
passive and observed.  The action layout and policy binding are explicit
module constants (``CARTPOLE_ACTION_SPEC`` / ``CARTPOLE_POLICIES``); nothing is
inferred from the ``body`` joint set.  Observations are read from the live core model by joint
name, because the policy-order ``articulation/.../joint_pos`` slice only covers
the actuated ``body`` joint set and the passive pole is not part of it.

This module is a demo, not package API: ``examples`` is repo-only and is never
installed into the ``unigym-rl`` wheel.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from unigym.actuators.ir import ActuationGroup
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.runtime.actions import OrderedActionSpec, PolicyBinding
from unigym.world import World

from unigym_rl import EnvSpec, TaskConfigError, TaskContext, TaskEnv, TaskState, Transition

if TYPE_CHECKING:
    from unigym.run.api import Run

    from unigym_rl.gym import GymTaskEnv

__all__ = [
    "ACTION_JOINT",
    "ACTION_LIMIT",
    "CARTPOLE_ACTION_SPEC",
    "CARTPOLE_ACTUATION",
    "CARTPOLE_ARTICULATION",
    "CARTPOLE_JOINTS",
    "CARTPOLE_POLICIES",
    "CARTPOLE_PROFILE",
    "CARTPOLE_XML",
    "CART_LIMIT",
    "CartPoleTask",
    "OBS_DIM",
    "POLE_JOINT",
    "POLE_LIMIT",
    "make_cartpole_gym",
    "make_cartpole_profile",
    "make_cartpole_spec",
    "make_cartpole_task",
]

CARTPOLE_ARTICULATION = "cartpole"
CARTPOLE_PROFILE = "cartpole"
CARTPOLE_JOINTS = ("slider", "hinge_1")
ACTION_JOINT = "slider"
POLE_JOINT = "hinge_1"
OBS_DIM = 4
CART_LIMIT = 1.5
POLE_LIMIT = math.radians(12.0)
ACTION_LIMIT = 1.8
CARTPOLE_XML = Path(__file__).resolve().with_name("cartpole.xml")

CARTPOLE_ACTUATION = (
    ActuationGroup(
        name="body",
        law="standard_pd",
        joint_set="body",
        gains={"kp": 50.0, "kd": 5.0},
    ),
)

CARTPOLE_ACTION_SPEC = OrderedActionSpec.from_mapping(
    [
        {
            "name": CARTPOLE_ARTICULATION,
            "articulation": CARTPOLE_ARTICULATION,
            "joints": [ACTION_JOINT],
            "target": "position",
            "role": "body",
        }
    ]
)

CARTPOLE_POLICIES = (
    PolicyBinding.from_mapping(
        {"adapter": "manual", "groups": [[CARTPOLE_ARTICULATION, CARTPOLE_ARTICULATION]]}
    ),
)


def make_cartpole_profile() -> RobotProfile:
    """The demo profile matching ``cartpole.xml`` (fixed base, one slider).

    The slider is the only actuated joint; the pole hinge is passive and is
    observed by name.  Assets stay ``None`` because the bundled world owns the
    MJCF.
    """
    return RobotProfile(
        name=CARTPOLE_PROFILE,
        joint_sets={"body": (ACTION_JOINT,)},
        default_joint_pos={ACTION_JOINT: 0.0, POLE_JOINT: 0.0},
        assets=None,
        extra_action_sets={},
        base_body="cart",
        foot_bodies=(),
        body_sets={},
        body_bindings={"cart": BodyBinding("cart", {"mjcf": "cart"})},
        actuation=CARTPOLE_ACTUATION,
    )


class CartPoleTask(TaskEnv):
    """Fixed-base CartPole: one cart action column, pole passive and observed.

    Observation is ``[cart_x, pole_theta, cart_dx, pole_dtheta]`` in joint-name
    order.  Reward is the alive bonus plus bounded shaping::

        reward = 1 - 0.5 * (pole_theta / pole_limit)**2
                   - 0.1 * (cart_x / cart_limit)**2

    ``terminated`` fires when ``|cart_x| > cart_limit`` or
    ``|pole_theta| > pole_limit``; ``truncated`` fires when the env-local
    counter reaches ``max_steps``.  Randomized resets (default off) write the
    two joint positions through :meth:`unigym.run.api.Run.set_joint_state` and
    call :meth:`unigym.run.api.Run.settle`; no root state is written because
    this robot is fixed-base.
    """

    obs_dim = OBS_DIM

    def __init__(
        self,
        run: Run,
        *,
        max_steps: int = 500,
        cart_limit: float = CART_LIMIT,
        pole_limit: float = POLE_LIMIT,
        randomize_reset: bool = False,
        reset_position_scale: float = 0.1,
        seed: int = 0,
    ) -> None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise TaskConfigError(f"CartPoleTask: max_steps must be an int >= 1, got {max_steps!r}")
        self._max_steps = int(max_steps)
        self._cart_limit = self._positive_limit(cart_limit, "cart_limit")
        self._pole_limit = self._positive_limit(pole_limit, "pole_limit")
        self._randomize_reset = bool(randomize_reset)
        self._reset_scale = self._positive_limit(reset_position_scale, "reset_position_scale")
        self._rng = torch.Generator().manual_seed(int(seed))
        self._cart_joint: Any = None
        self._pole_joint: Any = None
        super().__init__(run)
        self._step_count = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

    @staticmethod
    def _positive_limit(value: float, name: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise TaskConfigError(f"CartPoleTask: {name} must be a number, got {value!r}") from exc
        if not math.isfinite(number) or number <= 0.0:
            raise TaskConfigError(f"CartPoleTask: {name} must be finite and > 0, got {number!r}")
        return number

    def on_bind(self, context: TaskContext) -> None:
        """Resolve the cart and pole joints by name on the live core model."""
        del context
        joints = self.instance.live.joints
        try:
            self._cart_joint = joints[ACTION_JOINT]
            self._pole_joint = joints[POLE_JOINT]
        except KeyError as exc:
            raise TaskConfigError(
                f"CartPoleTask: joint {exc.args[0]!r} is not in the compiled model "
                f"of articulation {CARTPOLE_ARTICULATION!r}"
            ) from exc

    def _joint_columns(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Live ``(cart_x, pole_theta, cart_dx, pole_dtheta)`` columns."""
        cart = torch.tensor(self._cart_joint.qpos(), dtype=torch.float32, device=self.device)
        pole = torch.tensor(self._pole_joint.qpos(), dtype=torch.float32, device=self.device)
        cart_dx = torch.tensor(self._cart_joint.qvel(), dtype=torch.float32, device=self.device)
        pole_dtheta = torch.tensor(self._pole_joint.qvel(), dtype=torch.float32, device=self.device)
        return cart, pole, cart_dx, pole_dtheta

    def observe(self, state: TaskState) -> torch.Tensor:
        """Return the live ``(N, 4)`` cart/pole position and velocity block."""
        del state
        return torch.cat(self._joint_columns(), dim=1)

    def reward(self, transition: Transition) -> torch.Tensor:
        """Bounded alive reward; the exact formula is in the class docstring."""
        cart_x = transition.next_obs[:, 0]
        pole_theta = transition.next_obs[:, 1]
        return (
            1.0
            - 0.5 * (pole_theta / self._pole_limit) ** 2
            - 0.1 * (cart_x / self._cart_limit) ** 2
        )

    def terminated(self, transition: Transition) -> torch.Tensor:
        """Terminate on the cart travel limit or the pole angle limit."""
        return self._cart_limit_mask(transition) | self._pole_limit_mask(transition)

    def truncated(self, transition: Transition) -> torch.Tensor:
        """Truncate when this env's episode counter reaches ``max_steps``."""
        del transition
        return self._step_count >= self._max_steps

    def termination_reasons(
        self,
        transition: Transition,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Named masks for the cart limit, pole limit and time limit."""
        return {
            "cart_limit": self._cart_limit_mask(transition) & terminated,
            "pole_limit": self._pole_limit_mask(transition) & terminated,
            "time_limit": truncated,
        }

    def on_step(self, action: torch.Tensor, state: TaskState) -> None:
        """Advance the env-local truncation clock once per policy tick."""
        del action, state
        self._step_count += 1

    def on_reset(self, env_ids: torch.Tensor, state: TaskState) -> None:
        """Reset counters and optionally randomize the joint state."""
        del state
        self._step_count[env_ids] = 0
        if not self._randomize_reset or not env_ids.numel():
            return
        rows = int(env_ids.numel())
        columns = len(CARTPOLE_JOINTS)
        unit = torch.rand((rows, columns), generator=self._rng, dtype=torch.float32)
        position = ((unit * 2.0) - 1.0) * self._reset_scale
        velocity = torch.zeros_like(position)
        self.run.set_joint_state(
            env_ids,
            articulation=CARTPOLE_ARTICULATION,
            joint_names=CARTPOLE_JOINTS,
            position=position.to(self.device),
            velocity=velocity.to(self.device),
        )
        self.run.settle(env_ids)

    def _cart_limit_mask(self, transition: Transition) -> torch.Tensor:
        return transition.next_obs[:, 0].abs() > self._cart_limit

    def _pole_limit_mask(self, transition: Transition) -> torch.Tensor:
        return transition.next_obs[:, 1].abs() > self._pole_limit


def make_cartpole_spec(
    *, num_envs: int, device: str | None = None, seed: int = 0, backend: str = "mujoco"
) -> EnvSpec:
    """Build the demo ``EnvSpec`` over the bundled absolute MJCF path.

    CartPole is MuJoCo-only; other backends are passed through so the core
    layer can reject them with its own teaching error.
    """
    return EnvSpec(
        world=World.base(mjcf=CARTPOLE_XML),
        robot=make_cartpole_profile(),
        action_spec=CARTPOLE_ACTION_SPEC,
        policies=CARTPOLE_POLICIES,
        num_envs=num_envs,
        device=device,
        backend=backend,
        seed=seed,
        inspect=True,
    )


def make_cartpole_task(
    *,
    num_envs: int,
    max_steps: int,
    device: str | None = None,
    seed: int = 0,
    backend: str = "mujoco",
    **task_kwargs: object,
) -> CartPoleTask:
    """Build a fresh real-MuJoCo :class:`CartPoleTask`."""
    spec = make_cartpole_spec(num_envs=num_envs, device=device, seed=seed, backend=backend)
    task = spec.make_task(CartPoleTask, max_steps=max_steps, seed=seed, **task_kwargs)
    assert isinstance(task, CartPoleTask)
    return task


def make_cartpole_gym(
    *,
    num_envs: int,
    max_steps: int,
    device: str | None = None,
    seed: int = 0,
    backend: str = "mujoco",
    **task_kwargs: object,
) -> GymTaskEnv:
    """Wrap the demo task in the optional Gymnasium face (lazy import)."""
    task = make_cartpole_task(
        num_envs=num_envs,
        max_steps=max_steps,
        device=device,
        seed=seed,
        backend=backend,
        **task_kwargs,
    )
    try:
        from unigym_rl.gym import GymTaskEnv
    except Exception:
        task.close()
        task.run.close()
        raise
    return GymTaskEnv(task)
