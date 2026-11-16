"""Test doubles for the R0 ``TaskEnv`` contract.

These fakes implement only the surface ``TaskEnv`` consumes:
``run.spec`` / ``run.build()`` / ``run.external()``, an env exposing
``num_envs`` / ``device`` / ``obs_cache`` / ``controls`` / ``_extra`` /
``reset_envs``, and a runner returning core ``StepResult`` objects.  They are
not a backend and must never be used outside tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
from unigym.runner.base import StepResult
from unigym.runtime.actions import OrderedActionGroup, OrderedActionSpec

__all__ = [
    "FakeActionAdapter",
    "FakeEnv",
    "FakeInstance",
    "FakeRun",
    "FakeRunner",
    "make_fake_action_spec",
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
        self.obs_cache = {"proprio": self._values()}
        self.step_actions: list[tuple] = []
        self.reset_calls: list[torch.Tensor] = []

    # backend-neutral state face
    def _extra(self) -> dict[str, object]:
        return {
            "step": self.episode_steps.clone(),
            "episode": self.episodes.clone(),
            "clock": torch.full(
                (self.num_envs,), float(self.t), dtype=torch.float32, device=self.device
            ),
        }

    # runner-facing operations
    def advance(self, actions: tuple) -> None:
        if self.log is not None:
            self.log.append("env.advance")
        self.t += 1
        self.episode_steps += 1
        self.step_actions.append(actions)
        self.obs_cache = {"proprio": self._values()}

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        if self.log is not None:
            self.log.append("env.reset_envs")
        ids = env_ids.to(device=self.device, dtype=torch.int64)
        self.episodes[ids] += 1
        self.episode_steps[ids] = 0
        values = self.obs_cache["proprio"].clone()
        values[ids] = self._reset_values(ids)
        self.obs_cache = {"proprio": values}
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
        self.env.advance(actions)
        self._ticks += 1
        return StepResult(
            t=self._ticks,
            obs={},
            extra=self.env._extra(),
            reset_env_ids=self.env.due_ids(),
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
        return StepResult(t=int(self._ticks), obs={}, extra={}, reset_env_ids=ids, terminated=None)


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
