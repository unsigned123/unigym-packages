"""R1: a real MuJoCo robot driven through ``TaskEnv`` with no RL framework.

The test fixture is a self-contained copy of the MIT-clean two-hinge
``tiny_arm.xml`` (``tests/fixtures/tiny_arm.xml``); no core test path is
imported.  One registered robot profile drives the real ``mujoco`` backend,
``TaskEnv`` owns reward/termination/truncation, and the runner only ever
resets the ids the task marked done.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from unigym.robots import register_robot, restore_robot_registry, snapshot_robot_registry
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.run.api import Run
from unigym.world import World

from unigym_rl import TaskEnv, TaskState, Transition

FIXTURE_MJCF = Path(__file__).resolve().parent / "fixtures" / "tiny_arm.xml"
ARTICULATION = "tiny"
ACTION_DIM = 2
OBS_DIM = 4
SHOULDER_LIMIT = 0.3

ACTUATION = [
    {
        "name": "body",
        "law": "standard_pd",
        "joint_set": "body",
        "gains": {"kp": 20.0, "kd": 1.0},
    }
]


def _tiny_profile() -> RobotProfile:
    """The two-hinge profile matching ``tests/fixtures/tiny_arm.xml``."""
    return RobotProfile(
        name="tiny",
        joint_sets={"body": ("shoulder", "elbow")},
        default_joint_pos={"shoulder": 0.0, "elbow": 0.0},
        assets=None,
        extra_action_sets={},
        base_body="base",
        foot_bodies=(),
        body_sets={},
        body_bindings={"base": BodyBinding("base", {"mjcf": "base"})},
    )


@pytest.fixture(autouse=True)
def _registered_tiny_robot() -> Iterator[None]:
    """Register the test-only robot and restore the registry afterwards."""
    snapshot = snapshot_robot_registry()
    register_robot("tiny", _tiny_profile)
    try:
        yield
    finally:
        restore_robot_registry(snapshot)


def _make_run(num_envs: int) -> Run:
    """One real MuJoCo run with task-owned (external) termination."""
    return Run(
        world=World.base(mjcf=FIXTURE_MJCF),
        robot={"name": ARTICULATION, "actuation": ACTUATION},
        observations={"robot": [ARTICULATION]},
        backend="mujoco",
        num_envs=num_envs,
        termination_mode="external",
        runner={"auto_reset": False, "reset_on_termination": False},
        adapter="manual",
        no_artifacts=True,
        inspect=True,
    )


class TinyArmTask(TaskEnv):
    """Deterministic task over the real tiny-arm articulation.

    ``observe`` concatenates the policy-ordered joint position and velocity
    rows from ``state.obs``.  ``terminated`` fires when the shoulder angle
    crosses ``shoulder_limit``; ``truncated`` fires only when a test declares
    a per-env tick limit.
    """

    obs_dim = OBS_DIM

    def __init__(
        self,
        run: Run,
        *,
        shoulder_limit: float = SHOULDER_LIMIT,
        truncate_env: int | None = None,
        truncate_tick: int | None = None,
    ) -> None:
        self.log: list[str] = []
        self._shoulder_limit = float(shoulder_limit)
        self._truncate_env = truncate_env
        self._truncate_tick = truncate_tick
        super().__init__(run)

    def observe(self, state: TaskState) -> torch.Tensor:
        """Flat ``(N, 4)`` = joint positions then joint velocities."""
        robot = state.robot(ARTICULATION)
        return torch.cat([robot.joint_pos, robot.joint_vel], dim=-1)

    def reward(self, transition: Transition) -> torch.Tensor:
        """Deterministic cost: the negative absolute joint positions."""
        return -transition.next_obs[:, :ACTION_DIM].abs().sum(dim=1)

    def terminated(self, transition: Transition) -> torch.Tensor:
        """Terminate when the shoulder leaves the configured band."""
        shoulder = transition.info.robot(ARTICULATION).joint_pos[:, 0]
        return shoulder.abs() > self._shoulder_limit

    def truncated(self, transition: Transition) -> torch.Tensor:
        """Optionally truncate one env once the clock reaches a tick."""
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._truncate_env is not None and self._truncate_tick is not None:
            mask[self._truncate_env] = bool(transition.info.t >= self._truncate_tick)
        return mask

    def termination_reasons(
        self,
        transition: Transition,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Two named masks: the joint limit and the time limit."""
        del transition
        return {"shoulder_limit": terminated, "time_limit": truncated}

    def on_reset(self, env_ids: torch.Tensor, state: TaskState) -> None:
        del state
        self.log.append(f"on_reset:{env_ids.tolist()}")

    def on_episode_end(
        self,
        env_ids: torch.Tensor,
        state: TaskState,
        transition: Transition,
    ) -> None:
        del state, transition
        self.log.append(f"on_episode_end:{env_ids.tolist()}")


def _spy_reset(env: TinyArmTask) -> list[list[int] | None]:
    """Record every ``runner.reset`` id list, in order with the task log."""
    calls: list[list[int] | None] = []
    original = env.runner.reset

    def reset(env_ids: object = None) -> object:
        ids = None if env_ids is None else torch.as_tensor(env_ids, dtype=torch.int64).tolist()
        calls.append(ids)
        env.log.append(f"runner.reset:{ids}")
        return original(env_ids)

    env.runner.reset = reset
    return calls


def test_manual_rollout_single_env() -> None:
    """A one-env rollout returns finite shapes and never fires early."""
    run = _make_run(1)
    env = TinyArmTask(run)
    try:
        assert env.num_envs == 1
        assert env.action_dim == ACTION_DIM
        assert env.control_dt == pytest.approx(0.02)

        obs, info = env.reset()
        assert obs.shape == (1, OBS_DIM)
        assert torch.isfinite(obs).all()
        assert info["state"].t == 0

        actions = torch.zeros(1, ACTION_DIM)
        for _ in range(4):
            obs, reward, terminated, truncated, info = env.step(actions)
            assert obs.shape == (1, OBS_DIM)
            assert torch.isfinite(obs).all()
            assert reward.shape == (1,)
            assert torch.isfinite(reward).all()
            assert terminated.shape == (1,) and terminated.dtype is torch.bool
            assert truncated.shape == (1,) and truncated.dtype is torch.bool
            assert not bool(terminated.any())
            assert not bool(truncated.any())
            assert info["final_observation"] is None
            assert info["final_info"] == (None,)
            assert info["episode_end"] is None
            assert set(info["termination_reasons"]) == {"shoulder_limit", "time_limit"}
            for mask in info["termination_reasons"].values():
                assert mask.shape == (1,)
                assert mask.dtype is torch.bool
    finally:
        env.close()
        run.close()


def test_manual_rollout_two_envs_termination_resets_done_only() -> None:
    """Env 0 terminates on a real joint excursion; only it is reset."""
    run = _make_run(2)
    env = TinyArmTask(run)
    try:
        env.reset()
        reset_calls = _spy_reset(env)
        env.log.clear()

        zeros = torch.zeros(2, ACTION_DIM)
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(zeros)
            assert obs.shape == (2, OBS_DIM)
            assert reward.shape == (2,)
            assert not bool(terminated.any()) and not bool(truncated.any())
            assert info["episode_end"] is None

        assert reset_calls == []
        assert env.log == []

        drive = torch.tensor([[1.5, 0.0], [0.0, 0.0]])
        obs, reward, terminated, truncated, info = env.step(drive)

        assert terminated.tolist() == [True, False]
        assert truncated.tolist() == [False, False]
        assert reward.shape == (2,)
        assert torch.isfinite(reward).all()

        reasons = info["termination_reasons"]
        assert reasons["shoulder_limit"].tolist() == [True, False]
        assert reasons["time_limit"].tolist() == [False, False]

        episode_end = info["episode_end"]
        assert episode_end is not None
        assert episode_end.env_ids.tolist() == [0]
        assert episode_end.terminated.tolist() == [True]
        assert episode_end.truncated.tolist() == [False]

        final_info = info["final_info"]
        assert len(final_info) == 2
        assert final_info[0] is not None
        assert final_info[0].env_ids.tolist() == [0]
        assert final_info[1] is None

        # final_observation is the PRE-reset next observation of the done row.
        pre_obs = env.observe(episode_end.state)
        final_obs = info["final_observation"]
        assert final_obs is not None
        assert final_obs.shape == (1, OBS_DIM)
        torch.testing.assert_close(final_obs, pre_obs[[0]])
        assert abs(float(pre_obs[0, 0])) > SHOULDER_LIMIT

        # reward was computed on that same pre-reset next_obs.
        torch.testing.assert_close(
            reward[0], -pre_obs[0, :ACTION_DIM].abs().sum(), atol=1e-6, rtol=0
        )

        # The returned obs is POST-reset: env 0 spawned back at 0/0, env 1 is
        # untouched and still equals its own pre-reset observation.
        assert torch.allclose(obs[0], torch.zeros(OBS_DIM), atol=1e-6)
        torch.testing.assert_close(obs[1], pre_obs[1])

        # Ordering: on_episode_end -> runner.reset(done ids) -> on_reset.
        assert env.log == [
            "on_episode_end:[0]",
            "runner.reset:[0]",
            "on_reset:[0]",
        ]
        assert reset_calls == [[0]]

        # Env 1 keeps living while env 0 starts a fresh episode.
        obs, reward, terminated, truncated, info = env.step(zeros)
        assert not bool(terminated.any()) and not bool(truncated.any())
        assert info["episode_end"] is None
    finally:
        env.close()
        run.close()


def test_task_truncation_is_separate_from_core_clock() -> None:
    """A task-owned truncation marks one env done without a termination."""
    run = _make_run(2)
    env = TinyArmTask(run, truncate_env=1, truncate_tick=1)
    try:
        env.reset()
        reset_calls = _spy_reset(env)
        env.log.clear()
        zeros = torch.zeros(2, ACTION_DIM)

        obs, reward, terminated, truncated, info = env.step(zeros)
        assert not bool(terminated.any()) and not bool(truncated.any())
        assert info["episode_end"] is None

        obs, reward, terminated, truncated, info = env.step(zeros)
        assert terminated.tolist() == [False, False]
        assert truncated.tolist() == [False, True]

        episode_end = info["episode_end"]
        assert episode_end is not None
        assert episode_end.env_ids.tolist() == [1]

        reasons = info["termination_reasons"]
        assert reasons["shoulder_limit"].tolist() == [False, False]
        assert reasons["time_limit"].tolist() == [False, True]

        assert info["final_info"][0] is None
        assert info["final_info"][1] is not None
        assert env.log == [
            "on_episode_end:[1]",
            "runner.reset:[1]",
            "on_reset:[1]",
        ]
        assert reset_calls == [[1]]

        # Env 0 was not reset and keeps its (zero-action) observation.
        pre_obs = env.observe(episode_end.state)
        torch.testing.assert_close(obs[0], pre_obs[0])
        assert torch.allclose(obs[1], torch.zeros(OBS_DIM), atol=1e-6)
    finally:
        env.close()
        run.close()


def test_core_external_termination_is_empty_and_all_false() -> None:
    """Core owns no termination source and never fires in external mode."""
    run = _make_run(2)
    try:
        instance = run.build()

        assert run.spec.termination_mode == "external"
        assert not run.spec.terminations
        term_group = instance.term_group
        assert len(term_group) == 0
        assert term_group.names == []

        runner = run.external()
        runner.reset(None)
        result = runner.step(instance.spec.action_spec.unflatten(torch.zeros(2, ACTION_DIM)))
        assert result.terminated is not None
        assert result.terminated.shape == (2,)
        assert not bool(result.terminated.any())
        assert result.reset_env_ids.numel() == 0
    finally:
        run.close()
