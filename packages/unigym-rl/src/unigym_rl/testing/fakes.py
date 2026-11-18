"""Test doubles for the R0 ``TaskEnv`` contract.

These fakes implement only the surface ``TaskEnv`` consumes:
``run.spec`` / ``run.build()`` / ``run.external()``, an env exposing
``num_envs`` / ``device`` / ``obs`` / ``record()`` / ``controls`` /
``reset_envs``, and a runner returning core ``StepResult`` objects.  The
observation face is a real core ``ObservationRegistry`` with fake values, so
the typed views and snapshot types are identical to a backend's.  They are
not a backend and must never be used outside tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
from unigym.env.obs import (
    ObservationConfig,
    ObservationRegistry,
    build_observation_schema,
)
from unigym.runner.base import StepResult
from unigym.runtime.actions import OrderedActionGroup, OrderedActionSpec

__all__ = [
    "FakeActionAdapter",
    "FakeEnv",
    "FakeInstance",
    "FakeObsRegistry",
    "FakeRun",
    "FakeRunner",
    "make_fake_action_spec",
    "make_fake_obs_registry",
]


def make_fake_action_spec(
    articulation: str = "robot",
    joints: tuple[str, ...] = ("j0", "j1"),
    target: str = "position",
) -> OrderedActionSpec:
    """One-group flat action spec with explicit joint names."""
    return OrderedActionSpec(
        groups=(
            OrderedActionGroup(
                name="main",
                articulation=articulation,
                joints=joints,
                target=target,
                role="body",
            ),
        )
    )


class FakeObsRegistry:
    """Core ``ObservationRegistry`` with a tiny, fully published schema."""

    def __init__(
        self,
        *,
        num_envs: int,
        joints: Sequence[str] = ("j0", "j1", "j2"),
        robot: str = "robot",
        contacts: Mapping[str, tuple[tuple[int, ...], str]] | None = None,
        command: bool = False,
        prev_action: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        self.robot_name = str(robot)
        self.joints = tuple(str(name) for name in joints)
        contact_table = dict(contacts or {})
        self.config = ObservationConfig.from_mapping(
            {
                "robot": [self.robot_name],
                "contact": list(contact_table),
                "command": bool(command),
                "prev_action": bool(prev_action),
            }
        )
        self.schema = build_observation_schema(
            self.config,
            full_joint_names={self.robot_name: self.joints},
            contacts=contact_table,
        )
        self._registry = ObservationRegistry(
            self.config,
            self.schema,
            num_envs=int(num_envs),
            device=device,
        )

    @property
    def t(self) -> int:
        return self._registry.t

    @t.setter
    def t(self, value: int) -> None:
        self._registry.t = int(value)

    def set(self, key: str, value: torch.Tensor) -> None:
        self._registry.set(key, value)

    def set_many(self, values: Mapping[str, torch.Tensor]) -> None:
        self._registry.set_many(values)

    def set_robot(self, name: str, leaf: str, value: torch.Tensor) -> None:
        self._registry.set_robot(name, leaf, value)

    def set_contact(self, name: str, value: torch.Tensor) -> None:
        self._registry.set_contact(name, value)

    def set_command(self, value: torch.Tensor | None) -> None:
        self._registry.set_command(value)

    def set_prev_action(self, value: torch.Tensor | None) -> None:
        self._registry.set_prev_action(value)

    def robot(self, name: str):
        return self._registry.robot(name)

    def contact(self, name: str):
        return self._registry.contact(name)

    @property
    def command(self):
        return self._registry.command

    @property
    def prev_action(self):
        return self._registry.prev_action

    def snapshot(self):
        return self._registry.snapshot()

    def record(self) -> dict[str, torch.Tensor]:
        return self._registry.record()


def make_fake_obs_registry(
    *,
    num_envs: int,
    joints: Sequence[str] = ("j0", "j1", "j2"),
    robot: str = "robot",
    contacts: Mapping[str, tuple[tuple[int, ...], str]] | None = None,
    command: bool = False,
    prev_action: bool = False,
    device: str | torch.device = "cpu",
) -> FakeObsRegistry:
    """Build a fake typed observation registry for hand-made snapshots."""
    return FakeObsRegistry(
        num_envs=num_envs,
        joints=joints,
        robot=robot,
        contacts=contacts,
        command=command,
        prev_action=prev_action,
        device=device,
    )


class FakeEnv:
    """Deterministic vector env double with a per-env episode clock."""

    def __init__(
        self,
        *,
        num_envs: int = 2,
        obs_dim: int = 3,
        device: str | torch.device = "cpu",
        due_every: int = 0,
        log: list[str] | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.device = torch.device(device)
        self.due_every = int(due_every)
        self.log = log
        self.controls: dict = {}
        self.t = 0
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.episodes = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.obs = FakeObsRegistry(
            num_envs=self.num_envs,
            joints=tuple(f"j{index}" for index in range(self.obs_dim)),
            device=self.device,
        )
        self._publish()
        self.step_actions: list[tuple] = []
        self.reset_calls: list[torch.Tensor] = []

    # backend-neutral state face
    def record(self) -> dict[str, torch.Tensor]:
        due = self.due_ids()
        return {
            "episode_id": self.episodes.clone(),
            "time_in_episode": self.episode_steps.to(torch.float32) * 0.02,
            "reset_env_ids": due.clone(),
            "should_reset": bool(due.numel()),
        }

    # runner-facing operations
    def advance(self, actions: tuple) -> None:
        if self.log is not None:
            self.log.append("env.advance")
        self.t += 1
        self.episode_steps += 1
        self.step_actions.append(actions)
        self._publish()

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        if self.log is not None:
            self.log.append("env.reset_envs")
        ids = env_ids.to(device=self.device, dtype=torch.int64)
        self.episodes[ids] += 1
        self.episode_steps[ids] = 0
        values = self.obs.robot("robot").joint_pos.clone()
        values[ids] = self._reset_values(ids)
        self._publish_values(values)
        self.reset_calls.append(ids.clone())

    def due_ids(self) -> torch.Tensor:
        if self.due_every <= 0:
            return torch.zeros(0, dtype=torch.int64, device=self.device)
        return torch.nonzero(self.episode_steps >= self.due_every, as_tuple=False).squeeze(-1)

    # deterministic observations
    def _values(self) -> torch.Tensor:
        base = (self.episodes * 100 + self.episode_steps).to(torch.float32).unsqueeze(-1)
        offsets = torch.arange(self.obs_dim, dtype=torch.float32, device=self.device)
        return base + offsets

    def _reset_values(self, env_ids: torch.Tensor) -> torch.Tensor:
        base = (1000 + self.episodes[env_ids] * 10).to(torch.float32).unsqueeze(-1)
        offsets = torch.arange(self.obs_dim, dtype=torch.float32, device=self.device)
        return base + offsets

    def _publish(self) -> None:
        self._publish_values(self._values())

    def _publish_values(self, values: torch.Tensor) -> None:
        zeros = torch.zeros_like(values)
        zero_vec = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        zero_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        zero_quat[:, 0] = 1.0
        gravity = zero_vec.clone()
        gravity[:, 2] = -1.0
        self.obs.set_robot("robot", "joint_pos", values)
        self.obs.set_robot("robot", "joint_pos_rel", values)
        self.obs.set_robot("robot", "joint_vel", zeros)
        self.obs.set_robot("robot", "applied_torque", zeros)
        for leaf, value in (
            ("gravity_b", gravity),
            ("base_pos_w", zero_vec),
            ("base_quat_w", zero_quat),
            ("base_lin_vel_w", zero_vec),
            ("base_ang_vel_w", zero_vec),
            ("base_lin_vel_b", zero_vec),
            ("base_ang_vel_b", zero_vec),
        ):
            self.obs.set_robot("robot", leaf, value)
        self.obs.t = int(self.t)


class FakeRunner:
    """Minimal ``ExternalRunner`` double (step/reset returning StepResult)."""

    def __init__(self, env: FakeEnv, log: list[str] | None = None) -> None:
        self.env = env
        self.log = log
        self.calls: list[tuple[str, object]] = []
        self._ticks = 0

    @property
    def ticks_done(self) -> int:
        return int(self._ticks)

    def step(self, actions: tuple) -> StepResult:
        if self.log is not None:
            self.log.append("runner.step")
        self.calls.append(("step", actions))
        t = int(self._ticks)
        self.env.advance(actions)
        self._ticks += 1
        due = self.env.due_ids()
        return StepResult(
            t=t,
            obs=self.env.obs.snapshot(),
            record=self.env.record(),
            reset_env_ids=due,
            terminated=torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device),
        )

    def reset(self, env_ids=None) -> StepResult:
        if self.log is not None:
            self.log.append("runner.reset")
        if env_ids is None:
            ids = torch.arange(self.env.num_envs, dtype=torch.int64, device=self.env.device)
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.int64, device=self.env.device)
        self.calls.append(("reset", ids.clone()))
        self.env.reset_envs(ids)
        return StepResult(
            t=int(self._ticks),
            obs=self.env.obs.snapshot(),
            record=self.env.record(),
            reset_env_ids=ids,
            terminated=None,
        )


@dataclass
class FakeActionAdapter:
    """Adapter double carrying the only timing fact ``TaskEnv`` needs."""

    control_dt: float | None = 0.02
    _unigym_name: str = "fake"


@dataclass
class FakeInstance:
    """Live-instance double shared by ``FakeRun.build()``."""

    env: FakeEnv
    spec: SimpleNamespace
    adapter: FakeActionAdapter
    action_groups: tuple = field(default_factory=tuple)

    @property
    def controls(self) -> dict:
        return self.env.controls


class FakeRun:
    """Run double exposing ``spec`` / ``build()`` / ``external()``."""

    def __init__(
        self,
        *,
        num_envs: int = 2,
        obs_dim: int = 3,
        due_every: int = 0,
        device: str | torch.device = "cpu",
        action_spec: OrderedActionSpec | None = None,
        control_dt: float | None = 0.02,
        termination_mode: str = "external",
        auto_reset: bool = False,
        reset_on_termination: bool = False,
        log: list[str] | None = None,
    ) -> None:
        self.log = log
        self.env = FakeEnv(
            num_envs=num_envs,
            obs_dim=obs_dim,
            device=device,
            due_every=due_every,
            log=log,
        )
        self.action_spec = action_spec or make_fake_action_spec()
        self.spec = SimpleNamespace(
            termination_mode=termination_mode,
            runner={
                "auto_reset": auto_reset,
                "reset_on_termination": reset_on_termination,
            },
            observations=self.env.obs.config.to_mapping(),
            action_spec=self.action_spec,
            policies=(),
            adapter_name="fake",
        )
        self.instance = FakeInstance(
            env=self.env,
            spec=self.spec,
            adapter=FakeActionAdapter(control_dt=control_dt),
        )
        self.runner = FakeRunner(self.env, log=log)

    def build(self) -> FakeInstance:
        return self.instance

    def external(self) -> FakeRunner:
        return self.runner
