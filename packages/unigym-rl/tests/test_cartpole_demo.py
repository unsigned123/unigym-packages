"""Real-MuJoCo tests for the repo-only ``examples/cartpole`` demo.

The base tests need only the demo XML and the core ``unigym`` checkout; the
optional skrl smoke is skipped when ``skrl``/``gymnasium`` are absent.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from unigym.robots import ROBOT_REGISTRY
from unigym.robots.base import RobotProfile
from unigym.runtime.actions import OrderedActionSpec, PolicyBinding

_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "cartpole"
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from task import (  # noqa: E402
    CARTPOLE_ACTION_SPEC,
    CARTPOLE_ACTUATION,
    CARTPOLE_ARTICULATION,
    CARTPOLE_JOINTS,
    CARTPOLE_POLICIES,
    CARTPOLE_PROFILE,
    CARTPOLE_XML,
    CartPoleTask,
    make_cartpole_profile,
    make_cartpole_spec,
    make_cartpole_task,
)


@pytest.fixture
def task() -> Iterator[object]:
    """One real two-env task; closed (task + borrowed run) after the test."""
    env = make_cartpole_task(num_envs=2, max_steps=10)
    try:
        yield env
    finally:
        env.close()
        env.run.close()


def _close(env: CartPoleTask) -> None:
    env.close()
    env.run.close()


def test_profile_spec_and_one_action_column() -> None:
    """The demo profile builds an external-termination one-action task."""
    assert CARTPOLE_XML.is_file()
    profile = make_cartpole_profile()
    assert isinstance(profile, RobotProfile)
    assert profile.name == CARTPOLE_PROFILE
    assert profile.actuation == CARTPOLE_ACTUATION
    assert isinstance(CARTPOLE_ACTION_SPEC, OrderedActionSpec)
    assert all(isinstance(binding, PolicyBinding) for binding in CARTPOLE_POLICIES)
    assert CARTPOLE_PROFILE not in ROBOT_REGISTRY

    spec = make_cartpole_spec(num_envs=2)
    assert spec.robot.name == CARTPOLE_PROFILE
    assert spec.action_spec is CARTPOLE_ACTION_SPEC
    assert spec.policies == CARTPOLE_POLICIES
    run = spec.make_run()
    try:
        assert run.built is False
        assert run.spec.termination_mode == "external"
        assert run.spec.runner["auto_reset"] is False
        assert run.spec.runner["reset_on_termination"] is False
        assert run.spec.config_path == "<memory>"
    finally:
        run.close()

    env = spec.make_task(CartPoleTask, max_steps=10)
    try:
        assert env.num_envs == 2
        assert env.action_dim == 1
        assert env.obs_dim == 4
        assert env.action_spec.groups[0].joints == (CARTPOLE_JOINTS[0],)
        assert env.control_dt == pytest.approx(0.02)
        assert env.termination_reasons is not None
    finally:
        _close(env)


def test_reset_and_step_shapes_are_finite(task: CartPoleTask) -> None:
    """Reset and a short zero-action rollout stay finite ``(N, 4)`` / ``(N,)``."""
    obs, info = task.reset()
    assert obs.shape == (2, 4)
    assert torch.isfinite(obs).all()
    assert info["state"].t == 0

    for _ in range(3):
        obs, reward, terminated, truncated, info = task.step(torch.zeros(2, 1))
        assert obs.shape == (2, 4)
        assert torch.isfinite(obs).all()
        assert reward.shape == (2,)
        assert torch.isfinite(reward).all()
        assert terminated.shape == (2,) and terminated.dtype is torch.bool
        assert truncated.shape == (2,) and truncated.dtype is torch.bool
        assert set(info["termination_reasons"]) == {"cart_limit", "pole_limit", "time_limit"}


def test_observation_columns_are_named_joint_state(task: CartPoleTask) -> None:
    """The four columns are cart/pole position then velocity, by joint name."""
    _, info = task.reset()
    ids = torch.arange(task.num_envs, dtype=torch.int64)
    position = torch.tensor([[0.31, 0.12], [-0.25, -0.08]], dtype=torch.float32)
    velocity = torch.tensor([[0.7, -0.4], [-0.5, 0.3]], dtype=torch.float32)
    task.run.set_joint_state(
        ids,
        articulation=CARTPOLE_ARTICULATION,
        joint_names=CARTPOLE_JOINTS,
        position=position,
        velocity=velocity,
    )
    task.run.settle(ids)

    expected = torch.cat([position, velocity], dim=1)
    state = task._snapshot(t=info["state"].t, obs=task.env.obs.snapshot())
    obs = task.observe(state)
    torch.testing.assert_close(obs, expected, atol=1e-5, rtol=0)


def test_termination_masks_and_named_reasons() -> None:
    """Cart and pole limits terminate their own envs with matching reasons."""
    env = make_cartpole_task(
        num_envs=3,
        max_steps=100,
        cart_limit=1.5,
        pole_limit=math.radians(12.0),
    )
    try:
        _, info = env.reset()
        ids = torch.arange(3, dtype=torch.int64)
        position = torch.tensor([[1.6, 0.0], [0.0, 0.5], [0.0, 0.0]], dtype=torch.float32)
        env.run.set_joint_state(
            ids,
            articulation=CARTPOLE_ARTICULATION,
            joint_names=CARTPOLE_JOINTS,
            position=position,
        )
        env.run.settle(ids)

        _, reward, terminated, truncated, step_info = env.step(torch.zeros(3, 1))
        assert terminated.tolist() == [True, True, False]
        assert truncated.tolist() == [False, False, False]
        assert torch.isfinite(reward).all()

        reasons = step_info["termination_reasons"]
        assert reasons["cart_limit"].tolist() == [True, False, False]
        assert reasons["pole_limit"].tolist() == [False, True, False]
        assert reasons["time_limit"].tolist() == [False, False, False]

        episode_end = step_info["episode_end"]
        assert episode_end is not None
        assert episode_end.env_ids.tolist() == [0, 1]
        assert episode_end.terminated.tolist() == [True, True]
    finally:
        _close(env)


def test_per_env_step_budget_truncates() -> None:
    """The env-local counter truncates at ``max_steps`` with a time reason."""
    env = make_cartpole_task(num_envs=2, max_steps=2, cart_limit=100.0, pole_limit=100.0)
    try:
        env.reset()
        _, _, terminated, truncated, _ = env.step(torch.zeros(2, 1))
        assert terminated.tolist() == [False, False]
        assert truncated.tolist() == [False, False]

        _, _, terminated, truncated, info = env.step(torch.zeros(2, 1))
        assert terminated.tolist() == [False, False]
        assert truncated.tolist() == [True, True]
        assert info["termination_reasons"]["time_limit"].tolist() == [True, True]
        assert info["termination_reasons"]["cart_limit"].tolist() == [False, False]
        assert info["termination_reasons"]["pole_limit"].tolist() == [False, False]

        # The done rows were reset: the counter starts over.
        _, _, terminated, truncated, _ = env.step(torch.zeros(2, 1))
        assert terminated.tolist() == [False, False]
        assert truncated.tolist() == [False, False]
    finally:
        _close(env)


def test_default_reset_is_zero_and_randomized_reset_is_visible() -> None:
    """Default reset writes nothing; the optional write shows up in the obs."""
    plain = make_cartpole_task(num_envs=2, max_steps=10)
    try:
        obs, _ = plain.reset()
        assert torch.equal(obs[:, :2], torch.zeros(2, 2))
    finally:
        _close(plain)

    randomized = make_cartpole_task(
        num_envs=8,
        max_steps=10,
        randomize_reset=True,
        reset_position_scale=0.2,
        seed=3,
    )
    try:
        obs, _ = randomized.reset()
        assert torch.isfinite(obs).all()
        assert (obs[:, :2].abs() <= 0.2 + 1e-6).all()
        assert bool((obs[:, :2].abs() > 0.0).any())
    finally:
        _close(randomized)


def test_skrl_demo_smoke() -> None:
    """One bounded ``train_skrl.main`` iteration when skrl is installed."""
    pytest.importorskip("skrl")
    pytest.importorskip("gymnasium")

    import train_skrl

    code = train_skrl.main(
        [
            "--num-envs",
            "1",
            "--max-episode-steps",
            "20",
            "--max-iterations",
            "1",
            "--rollout-steps",
            "2",
        ]
    )
    assert code == 0
