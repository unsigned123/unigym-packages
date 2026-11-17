"""Typed ``TaskState.articulation(name)`` tests (no engine required)."""

from __future__ import annotations

import pytest
import torch

from unigym_rl import ArticulationState, TaskContractError, TaskState

NUM_ENVS = 2
JOINTS = 3
LEAVES = (
    "joint_pos",
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
    "joint_pos": JOINTS,
    "joint_vel": JOINTS,
    "applied_torque": JOINTS,
    "gravity_b": 3,
    "base_pos_w": 3,
    "base_quat_w": 4,
    "base_lin_vel_w": 3,
    "base_ang_vel_w": 3,
    "base_lin_vel_b": 3,
    "base_ang_vel_b": 3,
}


def _extra(name: str = "go2") -> dict[str, torch.Tensor]:
    """One complete namespaced articulation record."""
    return {
        f"articulation/{name}/{leaf}": torch.arange(
            NUM_ENVS * WIDTHS[leaf], dtype=torch.float32
        ).reshape(NUM_ENVS, WIDTHS[leaf])
        for leaf in LEAVES
    }


def _state(extra: dict[str, object]) -> TaskState:
    return TaskState(t=0, num_envs=NUM_ENVS, extra=extra, obs_cache={})


def test_accessor_returns_the_published_values_with_shapes() -> None:
    extra = _extra()
    state = _state(extra)

    articulation = state.articulation("go2")

    assert isinstance(articulation, ArticulationState)
    for leaf in LEAVES:
        returned = getattr(articulation, leaf)
        assert returned is extra[f"articulation/go2/{leaf}"]
        assert returned.shape == (NUM_ENVS, WIDTHS[leaf])


def test_missing_articulation_is_a_teaching_error() -> None:
    extra = _extra("other")
    state = _state(extra)

    with pytest.raises(TaskContractError, match="missing extra") as excinfo:
        state.articulation("go2")

    message = str(excinfo.value)
    assert "articulation/go2/joint_pos" in message
    assert "other" in message


def test_missing_leaf_is_a_teaching_error() -> None:
    extra = _extra()
    del extra["articulation/go2/gravity_b"]
    state = _state(extra)

    with pytest.raises(TaskContractError, match="gravity_b"):
        state.articulation("go2")


@pytest.mark.parametrize(
    ("leaf", "value", "match"),
    [
        ("joint_pos", torch.zeros(NUM_ENVS, JOINTS, dtype=torch.int64), "floating point"),
        ("joint_pos", torch.zeros(NUM_ENVS, JOINTS, 1), "must be \\(num_envs, width\\)"),
        ("joint_pos", torch.zeros(NUM_ENVS + 1, JOINTS), "must have 2 rows"),
        ("joint_vel", torch.zeros(NUM_ENVS, JOINTS + 1), "must have width 3"),
        ("base_quat_w", torch.zeros(NUM_ENVS, 3), "must have width 4"),
        ("gravity_b", torch.zeros(NUM_ENVS, 3), None),  # control: valid
    ],
)
def test_malformed_leaf_is_a_teaching_error(
    leaf: str,
    value: torch.Tensor,
    match: str | None,
) -> None:
    extra = _extra()
    extra[f"articulation/go2/{leaf}"] = value
    state = _state(extra)

    if match is None:
        state.articulation("go2")
        return
    with pytest.raises(TaskContractError, match=match):
        state.articulation("go2")


def test_non_tensor_leaf_is_rejected() -> None:
    extra: dict[str, object] = dict(_extra())
    extra["articulation/go2/applied_torque"] = [[0.0] * JOINTS] * NUM_ENVS
    state = _state(extra)

    with pytest.raises(TaskContractError, match="torch.Tensor"):
        state.articulation("go2")


def test_empty_articulation_name_is_rejected() -> None:
    state = _state(_extra())

    with pytest.raises(TaskContractError, match="non-empty string"):
        state.articulation("")
