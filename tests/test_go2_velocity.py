"""Real-MuJoCo tests for the repo-only ``examples/go2_velocity`` smoke.

The profile/spec tests need only the installed ``unigym-robots`` pack.  The
real-env tests skip honestly when the pinned Go2 MJCF assets have not been
downloaded (or the pack is not installed); the final smoke in the task report
runs them with the downloaded assets.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from unigym.runtime.actions import OrderedActionSpec, PolicyBinding

_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "go2_velocity"


def _load_module(name: str, filename: str) -> Any:
    """Load one example module under a unique name (``task`` would collide)."""
    path = _EXAMPLE_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


go2_task = _load_module("go2_velocity_task", "task.py")

GO2_JOINTS = go2_task.GO2_JOINTS
GO2_DEFAULT_JOINT_POS = go2_task.GO2_DEFAULT_JOINT_POS
GO2_ARTICULATION = go2_task.GO2_ARTICULATION


def _require_go2_assets() -> None:
    """Skip when the pack/entry point or the downloaded assets are absent."""
    pytest.importorskip("unigym_robots")
    try:
        go2_task.go2_profile()
        go2_task.go2_scene_path()
    except Exception as exc:
        pytest.skip(f"Go2 pack/assets unavailable: {exc}")


def _close(env: Any) -> None:
    env.close()
    env.run.close()


def _standby(num_envs: int) -> torch.Tensor:
    return torch.tensor(GO2_DEFAULT_JOINT_POS, dtype=torch.float32).repeat(num_envs, 1)


@pytest.fixture
def one_env() -> Iterator[Any]:
    """One real Go2 env; closed (task + borrowed run) after the test."""
    _require_go2_assets()
    env = go2_task.make_go2_velocity_task(num_envs=1, max_steps=50)
    try:
        yield env
    finally:
        _close(env)


def test_pack_profile_uses_real_go2_calf_bodies() -> None:
    """The profile loads and binds the four Menagerie calf bodies."""
    pytest.importorskip("unigym_robots")
    try:
        profile = go2_task.go2_profile()
    except Exception as exc:
        pytest.skip(f"unigym-robots pack entry point unavailable: {exc}")

    assert profile.name == go2_task.GO2_PROFILE_ID
    assert profile.foot_bodies == ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
    assert "FL_foot" not in profile.body_bindings
    for body in profile.foot_bodies:
        binding = profile.body_bindings[body]
        assert binding.names["mjcf"] == body
        assert binding.names["usd"] == body
    assert profile.base_body == "base"
    assert tuple(profile.joint_order) == GO2_JOINTS


def test_action_spec_is_explicit_twelve_joint_body_group() -> None:
    """The action layout is explicit, 12 columns, and never inferred."""
    assert len(GO2_JOINTS) == 12
    assert isinstance(go2_task.GO2_ACTION_SPEC, OrderedActionSpec)
    assert go2_task.GO2_ACTION_SPEC.dim == 12
    assert go2_task.GO2_ACTION_GROUP == "legs"
    group = go2_task.GO2_ACTION_SPEC.groups[0]
    assert group.name == go2_task.GO2_ACTION_GROUP
    assert group.articulation == GO2_ARTICULATION
    assert group.joints == GO2_JOINTS
    assert group.target == "position"
    assert group.role == "body"
    assert all(isinstance(binding, PolicyBinding) for binding in go2_task.GO2_POLICIES)
    assert go2_task.GO2_POLICIES[0].groups == ((GO2_ARTICULATION, go2_task.GO2_ACTION_GROUP),)


def test_one_env_reset_and_step_are_finite(one_env: Any) -> None:
    """One env returns finite ``(1, 48)`` obs and ``(1,)`` reward."""
    obs, info = one_env.reset()
    assert obs.shape == (1, go2_task.OBS_DIM)
    assert torch.isfinite(obs).all()
    assert info["state"].t == 0
    torch.testing.assert_close(obs[:, -3:], one_env.command)

    for _ in range(3):
        obs, reward, terminated, truncated, info = one_env.step(_standby(1))
        assert obs.shape == (1, go2_task.OBS_DIM)
        assert torch.isfinite(obs).all()
        assert reward.shape == (1,)
        assert torch.isfinite(reward).all()
        assert terminated.shape == (1,) and terminated.dtype is torch.bool
        assert truncated.shape == (1,) and truncated.dtype is torch.bool
        assert set(info["termination_reasons"]) == {"tilt", "low_height", "time_limit"}


def test_command_is_resampled_and_visible_on_reset(one_env: Any) -> None:
    """Each reset samples a new in-range command and exposes it in the obs."""
    obs, _ = one_env.reset()
    first = one_env.command.clone()
    assert first.shape == (1, 3)
    for index, (low, high) in enumerate(go2_task.COMMAND_RANGES):
        assert bool(((first[:, index] >= low) & (first[:, index] <= high)).all())
    torch.testing.assert_close(obs[:, -3:], first)

    obs, _ = one_env.reset()
    second = one_env.command.clone()
    assert not torch.equal(first, second)
    torch.testing.assert_close(obs[:, -3:], second)


def test_tilt_and_low_height_fire_named_termination_reasons() -> None:
    """A forced roll terminates tilt; a forced low base terminates low_height."""
    _require_go2_assets()
    env = go2_task.make_go2_velocity_task(num_envs=2, max_steps=100)
    try:
        env.reset()
        ids = torch.arange(2, dtype=torch.int64)
        half_turn = math.sqrt(0.5)
        position = torch.tensor([[0.0, 0.0, 0.3], [0.0, 0.0, 0.05]], dtype=torch.float32)
        quat = torch.tensor(
            [[half_turn, half_turn, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        )
        zeros = torch.zeros((2, 3), dtype=torch.float32)
        env.run.set_root_state(
            ids,
            name=GO2_ARTICULATION,
            pos_w=position,
            quat_wxyz=quat,
            lin_vel_w=zeros,
            ang_vel_w=zeros,
        )
        env.run.settle(ids)

        _, reward, terminated, truncated, info = env.step(_standby(2))
        assert torch.isfinite(reward).all()
        assert terminated.tolist() == [True, True]
        assert not bool(truncated.any())

        reasons = info["termination_reasons"]
        assert reasons["tilt"].tolist() == [True, False]
        assert reasons["low_height"].tolist() == [False, True]
        assert reasons["time_limit"].tolist() == [False, False]
    finally:
        _close(env)


def test_env_local_step_budget_truncates_with_time_limit_reason() -> None:
    """The per-env counter truncates at ``max_steps`` with ``time_limit``."""
    _require_go2_assets()
    env = go2_task.make_go2_velocity_task(num_envs=2, max_steps=2)
    try:
        env.reset()
        _, _, terminated, truncated, _ = env.step(_standby(2))
        assert terminated.tolist() == [False, False]
        assert truncated.tolist() == [False, False]

        _, _, terminated, truncated, info = env.step(_standby(2))
        assert terminated.tolist() == [False, False]
        assert truncated.tolist() == [True, True]
        assert info["termination_reasons"]["time_limit"].tolist() == [True, True]
        assert info["termination_reasons"]["tilt"].tolist() == [False, False]
        assert info["termination_reasons"]["low_height"].tolist() == [False, False]

        # Done rows were reset, so their own counter starts over.
        _, _, terminated, truncated, _ = env.step(_standby(2))
        assert terminated.tolist() == [False, False]
        assert truncated.tolist() == [False, False]
    finally:
        _close(env)


def test_eight_env_construction_and_one_step() -> None:
    """Eight envs build, reset and step with finite batched tensors."""
    _require_go2_assets()
    env = go2_task.make_go2_velocity_task(num_envs=8, max_steps=50, seed=2)
    try:
        obs, _ = env.reset()
        assert env.command.shape == (8, 3)
        obs, reward, terminated, truncated, info = env.step(_standby(8))
        assert obs.shape == (8, go2_task.OBS_DIM)
        assert torch.isfinite(obs).all()
        assert reward.shape == (8,)
        assert torch.isfinite(reward).all()
        assert terminated.shape == (8,) and terminated.dtype is torch.bool
        assert truncated.shape == (8,) and truncated.dtype is torch.bool
        assert set(info["termination_reasons"]) == {"tilt", "low_height", "time_limit"}
    finally:
        _close(env)


def test_skrl_eight_env_smoke() -> None:
    """One manual-PPO iteration on eight Go2 envs when skrl is installed."""
    pytest.importorskip("skrl")
    pytest.importorskip("gymnasium")
    _require_go2_assets()

    previous_task = sys.modules.get("task")
    sys.modules["task"] = go2_task
    try:
        train_skrl = _load_module("go2_velocity_train_skrl", "train_skrl.py")
        code = train_skrl.main(
            [
                "--num-envs",
                "8",
                "--max-episode-steps",
                "50",
                "--max-iterations",
                "1",
                "--rollout-steps",
                "2",
            ]
        )
    finally:
        if previous_task is None:
            sys.modules.pop("task", None)
        else:
            sys.modules["task"] = previous_task
    assert code == 0
