"""R0 contract tests for :class:`unigym_rl.task.TaskEnv` (fake env only)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch
from unigym.runtime.actions import NamedAction

from unigym_rl import (
    EpisodeEnd,
    TaskConfigError,
    TaskContractError,
    TaskEnv,
    TaskEnvError,
    TaskState,
)
from unigym_rl.testing.fakes import FakeRun, make_fake_action_spec


class MinimalTask(TaskEnv):
    """Task with only the two required hooks, plus an optional event log."""

    def __init__(self, run, *, log: list[str] | None = None, **kwargs) -> None:
        self.log = log
        self.observed: list[TaskState] = []
        self.transitions = []
        super().__init__(run, **kwargs)

    def observe(self, state):
        self._log("task.observe")
        self.observed.append(state)
        return state.obs_cache["proprio"].clone()

    def reward(self, transition):
        self._log("task.reward")
        self.transitions.append(transition)
        return torch.ones(self.num_envs, device=self.device)

    def on_reset(self, env_ids, state):
        self._log("task.on_reset")

    def on_episode_end(self, env_ids, state, transition):
        self._log("task.on_episode_end")

    def _log(self, event: str) -> None:
        if self.log is not None:
            self.log.append(event)


class QueuedTask(MinimalTask):
    """Task whose termination masks are popped from preloaded queues."""

    def __init__(self, run, *, masks=(), trunc_masks=(), **kwargs) -> None:
        self.term_masks = list(masks)
        self.trunc_masks = list(trunc_masks)
        self.reasons_fn = None
        super().__init__(run, **kwargs)

    def terminated(self, transition):
        if not self.term_masks:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self.term_masks.pop(0)

    def truncated(self, transition):
        if not self.trunc_masks:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self.trunc_masks.pop(0)

    def termination_reasons(self, transition, terminated, truncated):
        if self.reasons_fn is None:
            return None
        return self.reasons_fn(terminated, truncated)


class ObserveOnly(TaskEnv):
    def observe(self, state):
        return state.obs_cache["proprio"].clone()


class RewardOnly(TaskEnv):
    def reward(self, transition):
        return torch.ones(self.num_envs, device=self.device)


def _mask(values: list[bool]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.bool)


def test_reset_and_step_contract() -> None:
    run = FakeRun(num_envs=2, obs_dim=4)
    task = MinimalTask(run)

    obs, info = task.reset()
    assert obs.shape == (2, 4)
    assert isinstance(info["state"], TaskState)
    assert task.num_envs == 2
    assert task.action_dim == run.action_spec.dim
    assert task.obs_dim == 4
    assert task.observation_shape == (4,)
    assert task.device == run.env.device

    obs, reward, terminated, truncated, info = task.step(torch.zeros(2, 2))

    assert obs.shape == (2, 4)
    assert reward.shape == (2,)
    assert terminated.dtype is torch.bool and terminated.shape == (2,)
    assert truncated.dtype is torch.bool and truncated.shape == (2,)
    assert not bool(terminated.any())
    assert not bool(truncated.any())
    assert info["termination_reasons"] == {}
    assert info["final_observation"] is None
    assert info["episode_end"] is None
    assert info["final_info"] == (None, None)
    assert isinstance(info["state"], TaskState)


@pytest.mark.parametrize("num_envs", [1, 2])
def test_batch_sizes(num_envs: int) -> None:
    run = FakeRun(num_envs=num_envs, obs_dim=2)
    task = MinimalTask(run)
    obs, _ = task.reset()
    assert obs.shape == (num_envs, 2)
    obs, reward, terminated, truncated, _ = task.step(torch.zeros(num_envs, 2))
    assert obs.shape == (num_envs, 2)
    assert reward.shape == (num_envs,)
    assert terminated.shape == (num_envs,)
    assert truncated.shape == (num_envs,)


def test_default_terminated_truncated_are_false() -> None:
    run = FakeRun(num_envs=2)
    task = MinimalTask(run)
    task.reset()
    _, _, terminated, truncated, _ = task.step(torch.zeros(2, 2))
    assert terminated.tolist() == [False, False]
    assert truncated.tolist() == [False, False]


def test_observe_and_reward_called_once_per_step() -> None:
    run = FakeRun(num_envs=2)
    task = MinimalTask(run)
    task.reset()
    observed_before, rewards_before = len(task.observed), len(task.transitions)
    task.step(torch.zeros(2, 2))
    assert len(task.observed) == observed_before + 1
    assert len(task.transitions) == rewards_before + 1


def test_ordering_reward_termination_and_final_obs_before_reset() -> None:
    log: list[str] = []
    run = FakeRun(num_envs=2, log=log)
    task = QueuedTask(run, log=log)
    task.term_masks = [_mask([True, False])]
    task.reset()
    log.clear()

    observed_before = len(task.observed)
    obs, reward, terminated, truncated, info = task.step(torch.zeros(2, 2))

    assert terminated.tolist() == [True, False]
    assert truncated.tolist() == [False, False]
    assert reward.tolist() == [1.0, 1.0]
    assert len(task.observed) == observed_before + 2

    pre_reset_state = task.observed[observed_before]
    post_reset_state = task.observed[observed_before + 1]
    pre_reset_obs = pre_reset_state.obs_cache["proprio"]
    post_reset_obs = post_reset_state.obs_cache["proprio"]
    assert info["final_observation"].shape == (1, 3)
    assert torch.equal(info["final_observation"], pre_reset_obs[[0]])
    assert torch.equal(obs[0], post_reset_obs[0])
    assert torch.equal(obs[1], pre_reset_obs[1])
    assert not torch.equal(obs[0], pre_reset_obs[0])

    order = [
        "runner.step",
        "env.advance",
        "task.observe",
        "task.reward",
        "task.on_episode_end",
        "runner.reset",
        "env.reset_envs",
        "task.on_reset",
        "task.observe",
    ]
    assert log == order
    assert run.runner.calls[-1][0] == "reset"
    assert run.runner.calls[-1][1].tolist() == [0]


def test_task_truncation_resets_only_done_ids() -> None:
    run = FakeRun(num_envs=2, due_every=0)
    task = QueuedTask(run)
    task.reset()
    task.trunc_masks = [_mask([False, True])]

    obs, _, terminated, truncated, info = task.step(torch.zeros(2, 2))

    assert terminated.tolist() == [False, False]
    assert truncated.tolist() == [False, True]
    assert run.env.reset_calls[-1].tolist() == [1]
    assert info["episode_end"].env_ids.tolist() == [1]
    assert run.env.episode_steps.tolist() == [1, 0]
    assert obs[1, 0].item() == pytest.approx(1020.0)
    assert obs[0, 0].item() != pytest.approx(1020.0)


def test_core_due_becomes_truncation() -> None:
    run = FakeRun(num_envs=2, due_every=1)
    task = MinimalTask(run)
    task.reset()

    obs, reward, terminated, truncated, info = task.step(torch.zeros(2, 2))

    assert not bool(terminated.any())
    assert truncated.tolist() == [True, True]
    assert info["episode_end"] is not None
    assert info["episode_end"].truncated.tolist() == [True, True]
    assert run.env.reset_calls[-1].tolist() == [0, 1]
    assert reward.tolist() == [1.0, 1.0]


def test_due_and_task_termination_combination() -> None:
    run = FakeRun(num_envs=2, due_every=1)
    task = QueuedTask(run)
    task.reset()
    task.term_masks = [_mask([True, False])]

    _, _, terminated, truncated, info = task.step(torch.zeros(2, 2))

    assert terminated.tolist() == [True, False]
    assert truncated.tolist() == [False, True]
    assert info["episode_end"].env_ids.tolist() == [0, 1]
    assert info["episode_end"].terminated.tolist() == [True, False]
    assert info["episode_end"].truncated.tolist() == [False, True]


def test_termination_reasons_and_final_info() -> None:
    run = FakeRun(num_envs=2)
    task = QueuedTask(run)
    task.reasons_fn = lambda terminated, truncated: {
        "fell": terminated.clone(),
        "timeout": truncated.clone(),
    }
    task.reset()
    task.term_masks = [_mask([True, False])]
    task.trunc_masks = [_mask([False, True])]

    _, _, terminated, truncated, info = task.step(torch.zeros(2, 2))

    reasons = info["termination_reasons"]
    assert set(reasons) == {"fell", "timeout"}
    assert reasons["fell"].tolist() == [True, False]
    assert reasons["timeout"].tolist() == [False, True]

    episode_end = info["episode_end"]
    assert isinstance(episode_end, EpisodeEnd)
    assert episode_end.env_ids.tolist() == [0, 1]
    assert episode_end.reasons["fell"].tolist() == [True, False]
    assert episode_end.reasons["timeout"].tolist() == [False, True]
    assert isinstance(episode_end.state, TaskState)

    final_info = info["final_info"]
    assert len(final_info) == 2
    assert all(isinstance(entry, EpisodeEnd) for entry in final_info)
    assert final_info[0].reasons["fell"].tolist() == [True]


def test_final_info_non_done_entries_are_none() -> None:
    run = FakeRun(num_envs=2)
    task = QueuedTask(run)
    task.reset()
    task.term_masks = [_mask([True, False])]

    _, _, _, _, info = task.step(torch.zeros(2, 2))

    final_info = info["final_info"]
    assert isinstance(final_info[0], EpisodeEnd)
    assert final_info[1] is None
    assert info["episode_end"].env_ids.tolist() == [0]


def test_flat_actions_unflatten_to_named_actions() -> None:
    run = FakeRun(num_envs=2)
    task = MinimalTask(run)
    task.reset()
    flat = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    task.step(flat)

    _, named = next(call for call in run.runner.calls if call[0] == "step")
    assert len(named) == 1
    action = named[0]
    assert isinstance(action, NamedAction)
    assert action.articulation == "robot"
    assert action.joint_names == ("j0", "j1")
    assert action.kind == "position"
    assert torch.equal(action.values, flat)
    assert not hasattr(action, "group")


def test_action_spec_override() -> None:
    run = FakeRun(num_envs=1)
    override = make_fake_action_spec(articulation="base", joints=("a", "b", "c"))
    task = MinimalTask(run, action_spec=override)
    assert task.action_dim == 3
    assert task.action_spec is override


def test_termination_mode_must_be_external() -> None:
    run = FakeRun(num_envs=1, termination_mode="core")
    with pytest.raises(TaskConfigError, match="termination_mode"):
        MinimalTask(run)


def test_runner_auto_reset_must_be_caller_owned() -> None:
    run = FakeRun(num_envs=1, auto_reset=True)
    with pytest.raises(TaskConfigError, match="auto_reset"):
        MinimalTask(run)


def test_runner_reset_on_termination_must_be_caller_owned() -> None:
    run = FakeRun(num_envs=1, reset_on_termination=True)
    with pytest.raises(TaskConfigError, match="reset_on_termination"):
        MinimalTask(run)


def test_step_state_t_and_extra_keys_match_core_semantics() -> None:
    run = FakeRun(num_envs=2, obs_dim=3)
    task = MinimalTask(run)
    task.reset()
    assert task.observed[-1].t == 0

    _, _, _, _, info = task.step(torch.zeros(2, 2))
    state = info["state"]
    assert state.t == 0
    assert {"step", "episode", "clock", "reset_env_ids", "should_reset"} <= set(state.extra)
    assert state.extra["step"].shape == (2,)
    assert state.extra["episode"].shape == (2,)
    assert state.extra["clock"].shape == (2,)
    assert state.extra["reset_env_ids"].dtype is torch.int64
    assert state.extra["reset_env_ids"].shape == (0,)
    assert state.extra["should_reset"] is False

    _, _, _, _, info = task.step(torch.zeros(2, 2))
    assert info["state"].t == 1


def test_missing_observe_raises() -> None:
    run = FakeRun(num_envs=1)
    task = RewardOnly(run)
    with pytest.raises(NotImplementedError, match="observe"):
        task.reset()


def test_missing_reward_raises() -> None:
    run = FakeRun(num_envs=1)
    task = ObserveOnly(run)
    task.reset()
    with pytest.raises(NotImplementedError, match="reward"):
        task.step(torch.zeros(1, 2))


def test_bad_reward_shape_is_a_contract_error() -> None:
    class BadReward(ObserveOnly):
        def reward(self, transition):
            return torch.ones(3)

    run = FakeRun(num_envs=2)
    task = BadReward(run)
    task.reset()
    with pytest.raises(TaskContractError, match="reward"):
        task.step(torch.zeros(2, 2))


def test_core_terminated_must_be_all_false() -> None:
    run = FakeRun(num_envs=2)
    task = MinimalTask(run)
    original = run.runner.step

    def bad_step(actions):
        result = original(actions)
        result.terminated = torch.ones(2, dtype=torch.bool)
        return result

    run.runner.step = bad_step
    task.reset()
    with pytest.raises(TaskContractError, match="core"):
        task.step(torch.zeros(2, 2))


def test_state_extra_and_obs_cache_are_observed() -> None:
    run = FakeRun(num_envs=2, obs_dim=3)
    task = MinimalTask(run)
    task.reset()
    task.step(torch.zeros(2, 2))

    first = task.observed[0]
    assert first.extra["step"].shape == (2,)
    assert first.extra["episode"].tolist() == [1, 1]
    assert first.obs_cache["proprio"].shape == (2, 3)
    assert first.t == 0
    with pytest.raises(FrozenInstanceError):
        first.t = 5  # type: ignore[misc]


def test_step_before_reset_is_rejected() -> None:
    run = FakeRun(num_envs=1)
    task = MinimalTask(run)
    with pytest.raises(TaskEnvError, match="reset"):
        task.step(torch.zeros(1, 2))


def test_close_calls_on_close_once_and_keeps_run_open() -> None:
    calls: list[str] = []

    class Closing(MinimalTask):
        def on_close(self):
            calls.append("close")

    run = FakeRun(num_envs=1)
    task = Closing(run)
    task.close()
    task.close()
    assert calls == ["close"]
    assert run.instance.env is run.env
