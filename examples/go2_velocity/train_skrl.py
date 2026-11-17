"""Manual skrl PPO smoke loop for the flat-ground Go2 velocity task.

This script is deliberately not a trainer: it parses the shared CLI options,
builds the real MuJoCo task through :func:`task.make_go2_velocity_gym`, wraps it
with ``skrl.envs.wrappers.torch.wrap_env``, and runs a hand-written
``act`` / ``record_transition`` / ``update`` loop with the same small PPO
policy/value nets as the CartPole smoke.  There is no checkpointing, logging
framework or convergence promise.

Action caveat: the Gaussian policy samples unbounded raw actions.  The env
receives ``standby_pose + 0.25 * clip(raw, -1, 1)``, while
``agent.record_transition`` stores the raw sample.  The stored log-probability
therefore does not exactly match the position target that moved the plant.
This is the same deliberate demo shortcut as the CartPole smoke; a real trainer
would use a bounded/normalized action space or correct the log-probability.

Run from the repo root with the core venv (assets must already be downloaded
with ``unigym-robots download --set mjcf-v0``)::

    PYTHONPATH=/data1/code/dlx/unigym-core:/data1/code/dlx/unigym-rl/src \\
      /data1/code/dlx/unigym-core/.venv/bin/python \\
      examples/go2_velocity/train_skrl.py --num-envs 8 --max-iterations 1 \\
      --rollout-steps 4
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from skrl.agents.torch.ppo import PPO
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from unigym_rl.cli import parse_args

_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from task import GO2_DEFAULT_JOINT_POS, make_go2_velocity_gym  # noqa: E402

ACTION_LIMIT = 1.0
ACTION_SCALE = 0.25

PPO_CONFIG = {
    "rollouts": 16,
    "learning_epochs": 4,
    "mini_batches": 4,
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


class Go2Policy(GaussianMixin, Model):
    """One-hidden-layer Gaussian policy; ``clip_actions`` stays off here."""

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


class Go2Value(DeterministicMixin, Model):
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


def main(argv: list[str] | None = None) -> int:
    """Run the bounded PPO smoke and return a process exit code."""
    options = parse_args(argv)
    max_iterations = options.train.max_iterations
    rollout_steps = options.train.rollout_steps
    device = torch.device(options.env.device)

    torch.manual_seed(options.env.seed)
    gym_env = make_go2_velocity_gym(
        num_envs=options.env.num_envs,
        max_steps=options.env.max_episode_steps,
        device=options.env.device,
        seed=options.env.seed,
    )
    env = wrap_env(gym_env, wrapper="gymnasium", verbose=False)
    standby = torch.tensor(GO2_DEFAULT_JOINT_POS, dtype=torch.float32, device=device)
    try:
        observation_space = env.observation_space
        action_space = env.action_space
        policy = Go2Policy(observation_space, action_space, device)
        value = Go2Value(observation_space, action_space, device)

        batch = rollout_steps * env.num_envs
        config = dict(
            PPO_CONFIG,
            rollouts=rollout_steps,
            mini_batches=max(1, min(PPO_CONFIG["mini_batches"], batch)),
        )
        memory = RandomMemory(memory_size=rollout_steps, num_envs=env.num_envs, device=device)
        agent = PPO(
            models={"policy": policy, "value": value},
            memory=memory,
            observation_space=observation_space,
            state_space=None,
            action_space=action_space,
            device=device,
            cfg=config,
        )
        agent.init()
        agent.enable_training_mode(True)
        agent.enable_models_training_mode(True)

        observations, _ = env.reset()
        log_every = max(1, max_iterations // 10)
        for iteration in range(1, max_iterations + 1):
            step_reward = 0.0
            episodes = 0
            env_steps = 0
            for timestep in range(rollout_steps):
                with torch.no_grad():
                    actions, _ = agent.act(
                        observations,
                        states=None,
                        timestep=timestep,
                        timesteps=rollout_steps,
                    )
                stepped_actions = standby + ACTION_SCALE * actions.clamp(
                    -ACTION_LIMIT, ACTION_LIMIT
                )
                next_observations, rewards, terminated, truncated, infos = env.step(stepped_actions)
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
                    timesteps=rollout_steps,
                )
                step_reward += float(rewards.sum().item())
                episodes += int((terminated | truncated).sum().item())
                env_steps += env.num_envs
                observations = next_observations

            agent.update(timestep=rollout_steps - 1, timesteps=rollout_steps)
            memory.reset()
            if iteration % log_every == 0 or iteration == max_iterations:
                mean_reward = step_reward / max(env_steps, 1)
                print(
                    f"iteration {iteration}/{max_iterations} "
                    f"mean_reward={mean_reward:.4f} episodes={episodes}",
                    flush=True,
                )
    finally:
        env.close()
        gym_env.task.run.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
