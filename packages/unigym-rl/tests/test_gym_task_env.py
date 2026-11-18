"""R2: Gymnasium ``VectorEnv`` face over the real R1 tiny-arm ``TaskEnv``.

The wrapper is exercised on the same self-contained MuJoCo two-hinge fixture
used by the R1 manual-rollout lane.  All leading five-tuple values are NumPy;
the task ``info`` dictionary is preserved verbatim so final-observation and
episode-end evidence survive the Gym boundary.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest
import torch

gymnasium = pytest.importorskip("gymnasium")

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

from unigym_rl import TaskConfigError, TaskEnv  # noqa: E402
from unigym_rl.gym import GymTaskEnv  # noqa: E402


@pytest.fixture(autouse=True)
def _registered_tiny_robot() -> Iterator[None]:
    """Register the test-only tiny robot and restore the registry afterwards."""
    snapshot = snapshot_robot_registry()
    register_robot("tiny", _tiny_profile)
    try:
        yield
    finally:
        restore_robot_registry(snapshot)


@contextmanager
def _gym_env(num_envs: int, **task_kwargs: object) -> Iterator[GymTaskEnv]:
    run = _make_run(num_envs)
    task = TinyArmTask(run, **task_kwargs)
    env = GymTaskEnv(task)
    try:
        yield env
    finally:
        env.close()
        run.close()


def _assert_box(space: gymnasium.Space, shape: tuple[int, ...]) -> None:
    assert isinstance(space, gymnasium.spaces.Box)
    assert space.shape == shape
    assert space.dtype == np.float32


@pytest.mark.parametrize("num_envs", [1, 2])
def test_vector_env_identity_and_spaces(num_envs: int) -> None:
    with _gym_env(num_envs) as env:
        assert isinstance(env, gymnasium.vector.VectorEnv)
        assert env.num_envs == num_envs
        assert env.metadata["autoreset_mode"] is gymnasium.vector.AutoresetMode.SAME_STEP
        assert env.device == torch.device("cpu")

        _assert_box(env.single_observation_space, (OBS_DIM,))
        _assert_box(env.observation_space, (num_envs, OBS_DIM))
        _assert_box(env.single_action_space, (ACTION_DIM,))
        _assert_box(env.action_space, (num_envs, ACTION_DIM))


@pytest.mark.parametrize("num_envs", [1, 2])
def test_reset_and_step_return_numpy(num_envs: int) -> None:
    with _gym_env(num_envs) as env:
        obs, info = env.reset(seed=0)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (num_envs, OBS_DIM)
        assert obs.dtype == np.float32
        assert "state" in info

        actions = np.zeros((num_envs, ACTION_DIM), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(actions)

        assert isinstance(obs, np.ndarray) and obs.shape == (num_envs, OBS_DIM)
        assert obs.dtype == np.float32
        assert isinstance(reward, np.ndarray) and reward.shape == (num_envs,)
        assert isinstance(terminated, np.ndarray) and terminated.shape == (num_envs,)
        assert terminated.dtype == np.bool_
        assert isinstance(truncated, np.ndarray) and truncated.shape == (num_envs,)
        assert truncated.dtype == np.bool_

        for key in (
            "state",
            "termination_reasons",
            "final_observation",
            "final_info",
            "episode_end",
        ):
            assert key in info
        assert info["final_observation"] is None
        assert info["episode_end"] is None


def test_step_accepts_torch_and_preserves_episode_evidence() -> None:
    with _gym_env(2) as env:
        env.reset()
        actions = torch.tensor([[1.5, 0.0], [0.0, 0.0]])
        obs, reward, terminated, truncated, info = env.step(actions)

        assert obs.shape == (2, OBS_DIM)
        assert terminated.tolist() == [True, False]
        assert truncated.tolist() == [False, False]

        episode_end = info["episode_end"]
        assert episode_end is not None
        assert episode_end.env_ids.tolist() == [0]
        assert episode_end.terminated.tolist() == [True]

        final_observation = info["final_observation"]
        assert final_observation is not None
        assert final_observation.shape == (1, OBS_DIM)
        assert isinstance(final_observation, torch.Tensor)

        final_info = info["final_info"]
        assert len(final_info) == 2
        assert final_info[0] is not None
        assert final_info[1] is None

        reasons = info["termination_reasons"]
        assert set(reasons) == {"shoulder_limit", "time_limit"}
        assert reasons["shoulder_limit"].tolist() == [True, False]


@pytest.mark.parametrize("num_envs", [1, 2])
def test_skrl_gymnasium_wrapper_torch_shapes(num_envs: int) -> None:
    skrl_wrappers = pytest.importorskip("skrl.envs.wrappers.torch")
    wrap_env = skrl_wrappers.wrap_env

    with _gym_env(num_envs) as env:
        wrapped = wrap_env(env, wrapper="gymnasium", verbose=False)

        assert wrapped.num_envs == num_envs
        assert wrapped.observation_space.shape == (OBS_DIM,)
        assert wrapped.action_space.shape == (ACTION_DIM,)

        obs, _ = wrapped.reset()
        assert torch.is_tensor(obs)
        assert obs.shape == (num_envs, OBS_DIM)

        actions = torch.zeros(num_envs, ACTION_DIM)
        obs, reward, terminated, truncated, _ = wrapped.step(actions)

        assert torch.is_tensor(obs) and obs.shape == (num_envs, OBS_DIM)
        assert reward.shape == (num_envs, 1)
        assert terminated.shape == (num_envs, 1)
        assert truncated.shape == (num_envs, 1)
        assert terminated.dtype is torch.bool
        assert truncated.dtype is torch.bool


class _NoObsDimTask(TaskEnv):
    """Task that never declares ``obs_dim`` (only infers it on first reset)."""

    def observe(self, state):
        return state.obs_cache["proprio"].clone()

    def reward(self, transition):
        return torch.zeros(self.num_envs, device=self.device)


def test_missing_obs_dim_is_a_teaching_error() -> None:
    from unigym_rl.testing.fakes import FakeRun

    task = _NoObsDimTask(FakeRun(num_envs=1))
    with pytest.raises(TaskConfigError, match="obs_dim"):
        GymTaskEnv(task)


def test_top_level_export_is_lazy_and_same_class() -> None:
    from unigym_rl import GymTaskEnv as LazyGymTaskEnv

    assert LazyGymTaskEnv is GymTaskEnv


def test_render_is_not_implemented() -> None:
    with _gym_env(1) as env:
        with pytest.raises(NotImplementedError, match="render"):
            env.render()


def test_close_delegates_to_task_once() -> None:
    run = _make_run(1)
    task = TinyArmTask(run)
    env = GymTaskEnv(task)
    calls: list[str] = []
    original = task.close

    def close() -> None:
        calls.append("task.close")
        original()

    task.close = close  # type: ignore[method-assign]
    try:
        env.close()
        env.close()
        assert calls == ["task.close"]
        assert env.closed is True
    finally:
        run.close()


def test_package_import_does_not_load_gymnasium() -> None:
    code = (
        "import sys\n"
        "import unigym_rl\n"
        "assert hasattr(unigym_rl, 'TaskEnv')\n"
        "assert 'gymnasium' not in sys.modules, "
        "sorted(name for name in sys.modules if 'gym' in name)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
