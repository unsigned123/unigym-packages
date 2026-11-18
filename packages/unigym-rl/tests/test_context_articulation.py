"""Typed ``TaskState.robot(name)`` tests (no engine required)."""

from __future__ import annotations

import pytest
import torch
from unigym.env.obs import ObsContractError, RobotObs

from unigym_rl import TaskContractError, TaskState
from unigym_rl.testing.fakes import make_fake_obs_registry

NUM_ENVS = 2
JOINTS = ("j0", "j1", "j2")
LEAVES = (
    "joint_pos",
    "joint_pos_rel",
    "joint_vel",
    "applied_torque",
    "gravity_b",
    "base_pos_w",
    "base_quat_w",
    "base_lin_vel_w",
    "base_ang_vel_w",
    "base_lin_vel_b",
    "base_ang_vel_b",
)
WIDTHS = {
    "joint_pos": len(JOINTS),
    "joint_pos_rel": len(JOINTS),
    "joint_vel": len(JOINTS),
    "applied_torque": len(JOINTS),
    "gravity_b": 3,
    "base_pos_w": 3,
    "base_quat_w": 4,
    "base_lin_vel_w": 3,
    "base_ang_vel_w": 3,
    "base_lin_vel_b": 3,
    "base_ang_vel_b": 3,
}


def _registry(*, name: str = "go2", leaves: tuple[str, ...] = LEAVES):
    registry = make_fake_obs_registry(num_envs=NUM_ENVS, robot=name, joints=JOINTS)
    for leaf in leaves:
        value = torch.arange(NUM_ENVS * WIDTHS[leaf], dtype=torch.float32).reshape(
            NUM_ENVS, WIDTHS[leaf]
        )
        registry.set_robot(name, leaf, value)
    return registry


def _state(registry=None) -> TaskState:
    registry = registry or _registry()
    return TaskState(t=0, num_envs=NUM_ENVS, obs=registry.snapshot())


def test_accessor_returns_the_published_values_with_shapes() -> None:
    registry = _registry()
    state = _state(registry)

    robot = state.robot("go2")

    assert isinstance(robot, RobotObs)
    assert state.robot("go2").key == "articulation/go2"
    for leaf in LEAVES:
        returned = getattr(robot, leaf)
        torch.testing.assert_close(returned, registry.snapshot().value(f"articulation/go2/{leaf}"))
        assert returned.shape == (NUM_ENVS, WIDTHS[leaf])


def test_joint_gather_uses_the_served_order() -> None:
    state = _state()

    selected = state.robot("go2").joints(("j2", "j0"), leaf="joint_pos")

    expected = state.obs.value("articulation/go2/joint_pos")[:, [2, 0]]
    torch.testing.assert_close(selected, expected)


def test_missing_articulation_is_a_teaching_error() -> None:
    state = _state()

    with pytest.raises(ObsContractError, match="robot observation 'other'") as excinfo:
        state.robot("other")

    assert "go2" in str(excinfo.value)


def test_missing_leaf_is_a_teaching_error() -> None:
    registry = _registry(leaves=tuple(leaf for leaf in LEAVES if leaf != "gravity_b"))
    state = _state(registry)

    with pytest.raises(ObsContractError, match="gravity_b"):
        _ = state.robot("go2").gravity_b


def test_unknown_joint_name_is_a_teaching_error() -> None:
    state = _state()

    with pytest.raises(ObsContractError, match="missing"):
        state.robot("go2").joints(("missing",), leaf="joint_pos")


def test_task_state_requires_an_obs_snapshot() -> None:
    with pytest.raises(TaskContractError, match="ObsSnapshot"):
        TaskState(t=0, num_envs=NUM_ENVS, obs={})  # type: ignore[arg-type]


def test_task_state_validates_bookkeeping_tensors() -> None:
    registry = _registry()
    with pytest.raises(TaskContractError, match="episode_id"):
        TaskState(t=0, num_envs=NUM_ENVS, obs=registry.snapshot(), episode_id=torch.zeros(3))
    with pytest.raises(TaskContractError, match="reset_env_ids"):
        TaskState(t=0, num_envs=NUM_ENVS, obs=registry.snapshot(), reset_env_ids=torch.tensor([9]))
