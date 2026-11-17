"""R2.5 tests for programmatic :class:`unigym_rl.spec.EnvSpec` construction."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from unigym.robots import register_robot, restore_robot_registry, snapshot_robot_registry
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.world import World

from unigym_rl import EnvSpec, TaskConfigError, TaskEnv, TaskState, Transition

FIXTURE_MJCF = Path(__file__).resolve().parent / "fixtures" / "tiny_arm.xml"

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
    snapshot = snapshot_robot_registry()
    register_robot("tiny", _tiny_profile)
    try:
        yield
    finally:
        restore_robot_registry(snapshot)


def _spec(*, num_envs: int = 1) -> EnvSpec:
    return EnvSpec(
        world=World.base(mjcf=FIXTURE_MJCF),
        robot={"name": "tiny", "actuation": ACTUATION},
        num_envs=num_envs,
        inspect=True,
    )


class TinySpecTask(TaskEnv):
    """Minimal real task over the tiny-arm joint positions."""

    obs_dim = 2

    def observe(self, state: TaskState) -> torch.Tensor:
        return state.extra["articulation/tiny/joint_pos"]

    def reward(self, transition: Transition) -> torch.Tensor:
        return -transition.next_obs[:, 0].abs()


def test_make_run_is_lazy_with_task_owned_termination_and_no_yaml() -> None:
    spec = _spec(num_envs=3)
    run = spec.make_run()
    try:
        assert run.built is False
        assert run.spec.num_envs == 3
        assert run.spec.termination_mode == "external"
        assert run.spec.runner["auto_reset"] is False
        assert run.spec.runner["reset_on_termination"] is False
        assert run.spec.config_path == "<memory>"
    finally:
        run.close()


def test_make_run_returns_a_fresh_lazy_run_each_call() -> None:
    spec = _spec()
    first = spec.make_run()
    second = spec.make_run()
    try:
        assert first is not second
        assert first.built is False
        assert second.built is False
    finally:
        first.close()
        second.close()


def test_make_task_builds_the_real_tiny_arm_without_a_yaml() -> None:
    spec = _spec(num_envs=2)
    task = spec.make_task(TinySpecTask)
    try:
        assert task.num_envs == 2
        assert task.action_dim == 2
        assert task.run.spec.termination_mode == "external"
        assert task.run.spec.config_path == "<memory>"

        obs, _ = task.reset()
        assert obs.shape == (2, 2)
        obs, reward, terminated, truncated, info = task.step(torch.zeros(2, 2))
        assert obs.shape == (2, 2)
        assert reward.shape == (2,)
        assert terminated.tolist() == [False, False]
        assert truncated.tolist() == [False, False]
        assert info["episode_end"] is None
    finally:
        task.close()
        task.run.close()


def test_make_task_rejects_a_non_taskenv_class() -> None:
    spec = _spec()
    with pytest.raises(TaskConfigError, match="TaskEnv"):
        spec.make_task(object)  # type: ignore[arg-type]


@pytest.mark.parametrize("num_envs", [0, -1])
def test_num_envs_below_one_is_rejected(num_envs: int) -> None:
    with pytest.raises(TaskConfigError, match="num_envs"):
        EnvSpec(
            world=World.base(mjcf=FIXTURE_MJCF),
            robot={"name": "tiny", "actuation": ACTUATION},
            num_envs=num_envs,
        )


def test_non_integer_num_envs_is_rejected() -> None:
    with pytest.raises(TaskConfigError, match="num_envs"):
        EnvSpec(
            world=World.base(mjcf=FIXTURE_MJCF),
            robot={"name": "tiny", "actuation": ACTUATION},
            num_envs=True,  # type: ignore[arg-type]
        )


def test_make_gym_wraps_a_task() -> None:
    pytest.importorskip("gymnasium")
    spec = _spec()
    env = spec.make_gym(TinySpecTask)
    try:
        obs, _ = env.reset()
        assert obs.shape == (1, 2)
    finally:
        env.close()
        env.task.run.close()


def test_importing_unigym_rl_does_not_import_gymnasium() -> None:
    code = "import sys, unigym_rl; print('gymnasium' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"
