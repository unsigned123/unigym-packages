"""Tests for the R0 fake run/runner/env fixtures (no real backend)."""

from __future__ import annotations

import pytest
import torch
from unigym.runner.base import StepResult
from unigym.runner.state import read_extra

from unigym_rl import TaskConfigError, TaskEnv
from unigym_rl.testing.fakes import (
    FakeEnv,
    FakeRun,
    FakeRunner,
    make_fake_action_spec,
)


class EchoTask(TaskEnv):
    def observe(self, state):
        return state.obs_cache["proprio"].clone()

    def reward(self, transition):
        return torch.zeros(self.num_envs, device=self.device)


def test_fake_action_spec_is_one_explicit_group() -> None:
    spec = make_fake_action_spec(joints=("a", "b", "c"))
    assert spec.dim == 3
    assert spec.groups[0].articulation == "robot"
    assert spec.groups[0].joints == ("a", "b", "c")


def test_fake_env_extra_has_read_extra_defaults() -> None:
    env = FakeEnv(num_envs=2, obs_dim=3)
    extra = read_extra(env)
    assert extra["reset_env_ids"].shape == (0,)
    assert extra["should_reset"] is False
    assert extra["step"].tolist() == [0, 0]


def test_fake_env_due_clock_and_reset_calls() -> None:
    env = FakeEnv(num_envs=2, due_every=2)
    runner = FakeRunner(env)
    actions = make_fake_action_spec().unflatten(torch.zeros(2, 2))

    first = runner.step(actions)
    assert isinstance(first, StepResult)
    assert first.reset_env_ids.tolist() == []
    assert first.terminated is not None and not bool(first.terminated.any())

    runner.step(actions)
    assert env.due_ids().tolist() == [0, 1]
    runner.reset(env.due_ids())
    assert env.reset_calls[-1].tolist() == [0, 1]
    assert env.episode_steps.tolist() == [0, 0]


def test_fake_runner_reset_none_resets_all() -> None:
    env = FakeEnv(num_envs=3)
    runner = FakeRunner(env)
    result = runner.reset(None)
    assert result.reset_env_ids.tolist() == [0, 1, 2]
    assert runner.ticks_done == 0
    assert len(env.reset_calls) == 1


def test_fake_run_surface_and_flags() -> None:
    run = FakeRun(num_envs=2)
    assert run.spec.termination_mode == "external"
    assert run.spec.runner["auto_reset"] is False
    assert run.spec.runner["reset_on_termination"] is False
    assert run.build() is run.instance
    assert run.external() is run.runner
    assert run.instance.env is run.env


def test_fake_run_drives_a_task_env() -> None:
    run = FakeRun(num_envs=2, obs_dim=3, due_every=1)
    task = EchoTask(run)
    obs, info = task.reset()
    assert obs.shape == (2, 3)
    assert info["state"].extra["episode"].tolist() == [1, 1]

    obs, reward, terminated, truncated, info = task.step(torch.zeros(2, 2))
    assert obs.shape == (2, 3)
    assert reward.tolist() == [0.0, 0.0]
    assert terminated.tolist() == [False, False]
    assert truncated.tolist() == [True, True]
    assert info["episode_end"].env_ids.tolist() == [0, 1]


def test_missing_control_dt_is_rejected() -> None:
    run = FakeRun(num_envs=1, control_dt=None)
    with pytest.raises(TaskConfigError, match="control_dt"):
        EchoTask(run)
