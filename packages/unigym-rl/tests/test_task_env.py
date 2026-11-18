"""R0 contract tests for :class:`unigym_rl.task.TaskEnv` (fake env only)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch
from unigym.runtime.actions import NamedAction

from unigym_rl import (
    EpisodeEnd,
    RingBuffer,
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
        return state.obs.robot("robot").joint_pos.clone()

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
        return state.obs.robot("robot").joint_pos.clone()


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
    pre_reset_obs = pre_reset_state.obs.robot("robot").joint_pos
    post_reset_obs = post_reset_state.obs.robot("robot").joint_pos
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


def test_step_state_clock_and_typed_obs_match_core_semantics() -> None:
    run = FakeRun(num_envs=2, obs_dim=3)
    task = MinimalTask(run)
    task.reset()
    assert task.observed[-1].t == 0

    _, _, _, _, info = task.step(torch.zeros(2, 2))
    state = info["state"]
    assert state.t == 0
    assert state.episode_id.tolist() == [1, 1]
    assert state.time_in_episode.shape == (2,)
    assert state.reset_env_ids.dtype is torch.int64
    assert state.reset_env_ids.shape == (0,)
    assert state.should_reset is False
    assert state.obs.robot("robot").joint_pos.shape == (2, 3)
    assert state.robot("robot").joint_pos is state.obs.robot("robot").joint_pos

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


def test_typed_state_and_obs_are_observed() -> None:
    run = FakeRun(num_envs=2, obs_dim=3)
    task = MinimalTask(run)
    task.reset()
    task.step(torch.zeros(2, 2))

    first = task.observed[0]
    assert first.episode_id.tolist() == [1, 1]
    assert first.time_in_episode.shape == (2,)
    assert first.obs.robot("robot").joint_pos.shape == (2, 3)
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


class StepQueuedTask(QueuedTask):
    """Task recording every ``on_step`` call before the next observation."""

    def __init__(self, run, **kwargs) -> None:
        self.step_calls: list[tuple[torch.Tensor, TaskState]] = []
        super().__init__(run, **kwargs)

    def on_step(self, action, state):
        self._log("task.on_step")
        self.step_calls.append((action.clone(), state))


class RecordingBuffer:
    """Buffer double logging the ids it is asked to reset."""

    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.reset_calls: list[list[int] | None] = []

    def reset(self, env_ids=None):
        self.log.append("buffer.reset")
        self.reset_calls.append(None if env_ids is None else env_ids.tolist())


class BufferTask(MinimalTask):
    """Task registering the buffer supplied by the test in ``on_bind``."""

    def __init__(self, run, *, buffer, **kwargs) -> None:
        self._buffer = buffer
        super().__init__(run, **kwargs)

    def on_bind(self, context):
        self.register_buffer(self._buffer)


class HistoryTask(QueuedTask):
    """Task whose observation is the current action-history count and last row."""

    obs_dim = 3

    def on_bind(self, context):
        self.actions = self.register_buffer(
            RingBuffer(context.num_envs, capacity=4, shape=(2,), device=context.device)
        )

    def on_step(self, action, state):
        self.actions.push(action)

    def observe(self, state):
        self._log("task.observe")
        self.observed.append(state)
        count = self.actions.count.to(torch.float32).unsqueeze(-1)
        last = self.actions.latest(1)[:, 0]
        return torch.cat([count, last], dim=-1)


def test_on_step_runs_once_before_observe_with_pre_reset_state() -> None:
    log: list[str] = []
    run = FakeRun(num_envs=2, log=log)
    task = StepQueuedTask(run, log=log)

    task.reset()
    assert task.step_calls == []
    assert "task.on_step" not in log
    log.clear()

    flat = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    task.step(flat)

    assert len(task.step_calls) == 1
    action, state = task.step_calls[0]
    assert torch.equal(action, flat)
    assert torch.equal(
        state.obs.robot("robot").joint_pos,
        task.observed[-1].obs.robot("robot").joint_pos,
    )
    assert log == ["runner.step", "env.advance", "task.on_step", "task.observe", "task.reward"]


def test_on_step_is_not_called_on_reset_and_precedes_post_reset_observe() -> None:
    log: list[str] = []
    run = FakeRun(num_envs=2, log=log)
    task = StepQueuedTask(run, log=log)
    task.term_masks = [_mask([True, False])]

    task.reset()
    assert task.step_calls == []
    log.clear()
    observed_before = len(task.observed)

    task.step(torch.zeros(2, 2))

    assert len(task.step_calls) == 1
    _, state = task.step_calls[0]
    assert torch.equal(
        state.obs.robot("robot").joint_pos,
        task.observed[observed_before].obs.robot("robot").joint_pos,
    )
    assert len(task.observed) == observed_before + 2
    assert log == [
        "runner.step",
        "env.advance",
        "task.on_step",
        "task.observe",
        "task.reward",
        "task.on_episode_end",
        "runner.reset",
        "env.reset_envs",
        "task.on_reset",
        "task.observe",
    ]


def test_registered_buffer_resets_before_the_initial_observation() -> None:
    log: list[str] = []
    run = FakeRun(num_envs=2, log=log)
    buffer = RecordingBuffer(log)
    task = BufferTask(run, buffer=buffer, log=log)

    task.reset()

    assert buffer.reset_calls == [[0, 1]]
    assert log == [
        "runner.reset",
        "env.reset_envs",
        "buffer.reset",
        "task.on_reset",
        "task.observe",
    ]


def test_register_buffer_is_idempotent_and_requires_a_callable_reset() -> None:
    run = FakeRun(num_envs=2)
    task = MinimalTask(run)
    log: list[str] = []
    buffer = RecordingBuffer(log)

    assert task.register_buffer(buffer) is buffer
    assert task.register_buffer(buffer) is buffer
    task.reset()
    assert buffer.reset_calls == [[0, 1]]

    with pytest.raises(TaskConfigError, match="reset"):
        task.register_buffer(object())


def test_done_only_buffer_reset_preserves_live_rows_and_post_reset_obs() -> None:
    run = FakeRun(num_envs=2)
    task = HistoryTask(run)
    task.term_masks = [_mask([True, False])]

    obs, _ = task.reset()
    assert torch.equal(obs, torch.zeros(2, 3))
    assert task.actions.count.tolist() == [0, 0]

    observed_before = len(task.observed)
    actions = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    obs, _, terminated, truncated, info = task.step(actions)

    assert terminated.tolist() == [True, False]
    assert truncated.tolist() == [False, False]
    assert len(task.observed) == observed_before + 2

    pre_state = task.observed[observed_before]
    post_state = task.observed[observed_before + 1]
    assert pre_state is not post_state
    assert torch.equal(info["final_observation"], torch.tensor([[1.0, 1.0, 2.0]]))

    assert torch.equal(obs[0], torch.zeros(3))
    assert torch.equal(obs[1], torch.tensor([1.0, 3.0, 4.0]))
    assert task.actions.count.tolist() == [0, 1]
    assert torch.equal(task.actions.latest(1)[0, 0], torch.zeros(2))
    assert torch.equal(task.actions.latest(1)[1, 0], torch.tensor([3.0, 4.0]))


def test_registering_a_buffer_without_pushing_changes_nothing() -> None:
    log: list[str] = []
    run = FakeRun(num_envs=2, log=log)
    buffer = RingBuffer(num_envs=2, capacity=3, shape=(2,))
    task = BufferTask(run, buffer=buffer, log=log)

    obs, _ = task.reset()
    assert torch.equal(obs, run.env.obs.robot("robot").joint_pos)
    log.clear()

    task.step(torch.zeros(2, 2))

    assert buffer.count.tolist() == [0, 0]
    assert torch.equal(buffer.history(), torch.zeros(2, 3, 2))
    assert log == ["runner.step", "env.advance", "task.observe", "task.reward"]
