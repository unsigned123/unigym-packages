"""R2: short skrl PPO smoke on the real tiny-arm env via ``wrap_env``.

skrl 2.1 ships a ``PPO_CFG`` dataclass rather than a ``PPO_DEFAULT_CONFIG``
dict, so this test builds an explicit tiny config.  The smoke uses a manual
``act`` / ``record_transition`` / ``update`` loop (no trainer, no checkpointing,
no logging writers) to keep the lane deterministic and bounded.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
import torch

pytest.importorskip("skrl")
pytest.importorskip("gymnasium")

import torch.nn as nn  # noqa: E402
from skrl.agents.torch.ppo import PPO  # noqa: E402
from skrl.envs.wrappers.torch import wrap_env  # noqa: E402
from skrl.memories.torch import RandomMemory  # noqa: E402
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model  # noqa: E402
from test_task_env_mujoco import (  # noqa: E402
    ACTION_DIM,
    OBS_DIM,
    TinyArmTask,
    _make_run,
    _tiny_profile,
)
from unigym.robots import (  # noqa: E402
    register_robot,
    restore_robot_registry,
    snapshot_robot_registry,
)

from unigym_rl.gym import GymTaskEnv  # noqa: E402

PPO_DEFAULT_CONFIG = {
    "rollouts": 4,
    "learning_epochs": 2,
    "mini_batches": 2,
    "discount_factor": 0.99,
    "gae_lambda": 0.95,
    "learning_rate": 1e-3,
    "grad_norm_clip": 0.5,
    "ratio_clip": 0.2,
    "value_clip": 0.2,
    "entropy_loss_scale": 0.0,
    "value_loss_scale": 2.5,
    "kl_threshold": 0.0,
    "time_limit_bootstrap": False,
    "rewards_shaper": None,
    "mixed_precision": False,
}


class TinyPolicy(GaussianMixin, Model):
    """One-hidden-layer Gaussian policy (stochastic model)."""

    def __init__(self, observation_space, action_space, device) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(self, clip_actions=False, role="policy")
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 32),
            nn.Tanh(),
            nn.Linear(32, self.num_actions),
        )
        self.log_std = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        del role
        mean_actions = self.net(inputs["observations"])
        return mean_actions, {"log_std": self.log_std.expand_as(mean_actions)}


class TinyValue(DeterministicMixin, Model):
    """One-hidden-layer deterministic value model."""

    def __init__(self, observation_space, action_space, device) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False, role="value")
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def compute(self, inputs, role):
        del role
        return self.net(inputs["observations"]), {}


@pytest.fixture(autouse=True)
def _registered_tiny_robot() -> Iterator[None]:
    """Register the test-only tiny robot and restore the registry afterwards."""
    snapshot = snapshot_robot_registry()
    register_robot("tiny", _tiny_profile)
    try:
        yield
    finally:
        restore_robot_registry(snapshot)


def test_skrl_ppo_smoke_on_tiny_arm() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    run = _make_run(1)
    task = TinyArmTask(run, shoulder_limit=100.0)
    gym_env = GymTaskEnv(task)
    env = wrap_env(gym_env, wrapper="gymnasium", verbose=False)
    try:
        observation_space = env.observation_space
        action_space = env.action_space
        device = torch.device("cpu")

        policy = TinyPolicy(observation_space, action_space, device)
        value = TinyValue(observation_space, action_space, device)

        timesteps = int(PPO_DEFAULT_CONFIG["rollouts"])
        memory = RandomMemory(memory_size=timesteps, num_envs=env.num_envs, device=device)
        agent = PPO(
            models={"policy": policy, "value": value},
            memory=memory,
            observation_space=observation_space,
            state_space=None,
            action_space=action_space,
            device=device,
            cfg=dict(PPO_DEFAULT_CONFIG),
        )
        agent.init()
        agent.enable_training_mode(True)
        agent.enable_models_training_mode(True)

        before = [parameter.detach().clone() for parameter in policy.parameters()]

        observations, _ = env.reset()
        assert observations.shape == (env.num_envs, OBS_DIM)
        assert torch.isfinite(observations).all()

        for timestep in range(timesteps):
            with torch.no_grad():
                actions, _ = agent.act(
                    observations,
                    states=None,
                    timestep=timestep,
                    timesteps=timesteps,
                )
            assert actions.shape == (env.num_envs, ACTION_DIM)

            next_observations, rewards, terminated, truncated, infos = env.step(actions)
            assert rewards.shape == (env.num_envs, 1)
            assert terminated.shape == (env.num_envs, 1)
            assert truncated.shape == (env.num_envs, 1)
            assert torch.isfinite(rewards).all()

            agent.record_transition(
                observations=observations,
                states=None,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                next_states=None,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=timestep,
                timesteps=timesteps,
            )
            observations = next_observations

        assert len(memory) == timesteps * env.num_envs
        agent.update(timestep=timesteps - 1, timesteps=timesteps)

        returns = memory.get_tensor_by_name("returns")
        assert torch.isfinite(returns).all()

        after = [parameter.detach().clone() for parameter in policy.parameters()]
        assert any(not torch.equal(old, new) for old, new in zip(before, after, strict=True))
    finally:
        env.close()
        run.close()
