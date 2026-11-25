"""Tests for the R0 fake run/runner/env fixtures (no real backend)."""

from __future__ import annotations

import pytest
import torch
from unigym.runner.base import StepResult

from unigym_rl import TaskConfigError, TaskEnv
from unigym_rl.testing.fakes import (
    FakeEnv,
    FakeRun,
    FakeRunner,
    make_fake_action_spec,
)


class EchoTask(TaskEnv):
    def observe(self, state):
        return state.obs.robot("robot").joint_pos.clone()

    def reward(self, transition):
        return torch.zeros(self.num_envs, device=self.device)


def test_fake_action_spec_is_one_explicit_group() -> None:
    spec = make_fake_action_spec(joints=("a", "b", "c"))
    assert spec.dim == 3
    assert spec.groups[0].articulation == "robot"
    assert spec.groups[0].joints == ("a", "b", "c")


def test_fake_env_record_has_core_bookkeeping_defaults() -> None:
    env = FakeEnv(num_envs=2, obs_dim=3)
    record = env.record()
    assert record["episode_id"].tolist() == [0, 0]
    assert record["time_in_episode"].tolist() == [0.0, 0.0]
    assert record["reset_env_ids"].shape == (0,)
    assert record["should_reset"] is False
    assert env.obs.robot("robot").joint_pos.shape == (2, 3)


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


def test_fake_runner_step_t_is_pre_increment() -> None:
    env = FakeEnv(num_envs=2)
    runner = FakeRunner(env)
    actions = make_fake_action_spec().unflatten(torch.zeros(2, 2))

    first = runner.step(actions)
    second = runner.step(actions)

    assert first.t == 0
    assert second.t == 1
    assert runner.ticks_done == 2


def test_step_state_carries_reset_control_fields() -> None:
    run = FakeRun(num_envs=2, obs_dim=3, due_every=0)
    task = EchoTask(run)
    task.reset()

    _, _, _, _, info = task.step(torch.zeros(2, 2))
    state = info["state"]

    assert state.reset_env_ids.dtype is torch.int64
    assert state.reset_env_ids.shape == (0,)
    assert state.should_reset is False


def test_due_step_state_reports_typed_due_ids() -> None:
    run = FakeRun(num_envs=2, obs_dim=3, due_every=1)
    task = EchoTask(run)
    task.reset()

    _, _, _, _, info = task.step(torch.zeros(2, 2))
    state = info["episode_end"].state

    assert state.reset_env_ids.dtype is torch.int64
    assert state.reset_env_ids.tolist() == [0, 1]
    assert state.should_reset is True


def test_fake_runner_reset_none_resets_all() -> None:
    env = FakeEnv(num_envs=3)
    runner = FakeRunner(env)
    result = runner.reset(None)
    assert result.reset_env_ids.tolist() == [0, 1, 2]
    assert runner.ticks_done == 0
    assert len(env.reset_calls) == 1


def test_fake_runner_terminate_resets_and_reports_the_ended_ids() -> None:
    from unigym.core.episode_termination import Termination, TerminationSource

    env = FakeEnv(num_envs=2)
    runner = FakeRunner(env)
    facts = {
        1: Termination(
            outcome="failure",
            code="custom.task_terminated",
            source=TerminationSource.CUSTOM,
        )
    }
    result = runner.terminate([1], facts, reset=True)

    assert result.reset_env_ids.tolist() == [1]
    assert result.terminated is not None
    assert result.terminated.tolist() == [False, True]
    assert env.reset_calls[-1].tolist() == [1]
    assert runner.calls[-1][0] == "terminate"


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
    assert info["state"].episode_id.tolist() == [1, 1]

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
