"""Frozen records exchanged between :class:`~unigym_rl.task.TaskEnv` and hooks.

All batch tensors are ``(num_envs, ...)`` on the task device.  The records are
deliberately small and immutable: a hook may read them, never mutate them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from unigym_rl.errors import TaskContractError

if TYPE_CHECKING:
    from unigym.env.backends.base import EnvBackend
    from unigym.run.api import Run
    from unigym.run.instance import RunInstance
    from unigym.runtime.actions import OrderedActionSpec
    from unigym.runtime.controllers import ControlRuntime

__all__ = [
    "ArticulationState",
    "EpisodeEnd",
    "TaskContext",
    "TaskState",
    "Transition",
]

#: fixed-width articulation leaves: leaf -> expected trailing dimension.
_ARTICULATION_WIDTHS = {
    "base_pos_w": 3,
    "base_quat_w": 4,
    "base_lin_vel_w": 3,
    "base_ang_vel_w": 3,
    "base_lin_vel_b": 3,
    "base_ang_vel_b": 3,
    "gravity_b": 3,
}
#: joint leaves: their width must agree with each other, not a constant.
_ARTICULATION_JOINTS = ("joint_pos", "joint_vel", "applied_torque")
_ARTICULATION_LEAVES = (
    "joint_pos",
    "joint_vel",
    "applied_torque",
    "base_pos_w",
    "base_quat_w",
    "base_lin_vel_w",
    "base_ang_vel_w",
    "base_lin_vel_b",
    "base_ang_vel_b",
    "gravity_b",
)


def _require_int(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TaskContractError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise TaskContractError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_mapping(value: object, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise TaskContractError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _require_mask(value: object, name: str, num_envs: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TaskContractError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.dtype is not torch.bool:
        raise TaskContractError(f"{name} must be bool, got {value.dtype}")
    if value.shape != (num_envs,):
        raise TaskContractError(f"{name} must have shape ({num_envs},), got {tuple(value.shape)}")
    return value


@dataclass(frozen=True)
class TaskContext:
    """Immutable binding of one run to one task instance.

    Attributes:
        run: The user-facing core ``Run`` facade.
        instance: The live core ``RunInstance`` built by ``run.build()``.
        env: The backend env (``instance.env``).
        controls: The physical control table keyed by
            ``(articulation, control name)``.
        action_spec: The declaration-ordered flat action layout.
        num_envs: Batch size ``N``.
        device: Torch device of the run.
        control_dt: Policy period in seconds.
    """

    run: Run
    instance: RunInstance
    env: EnvBackend
    controls: Mapping[tuple[str, str], ControlRuntime]
    action_spec: OrderedActionSpec
    num_envs: int
    device: torch.device
    control_dt: float

    def __post_init__(self) -> None:
        _require_int(self.num_envs, "TaskContext.num_envs", minimum=1)
        _require_mapping(self.controls, "TaskContext.controls")
        if self.action_spec is None:
            raise TaskContractError("TaskContext.action_spec must not be None")
        if not isinstance(self.device, torch.device):
            raise TaskContractError(
                f"TaskContext.device must be a torch.device, got {type(self.device).__name__}"
            )
        if isinstance(self.control_dt, bool) or not isinstance(self.control_dt, int | float):
            raise TaskContractError(
                f"TaskContext.control_dt must be a number, got {type(self.control_dt).__name__}"
            )
        if not math.isfinite(float(self.control_dt)) or float(self.control_dt) <= 0.0:
            raise TaskContractError(
                f"TaskContext.control_dt must be finite and > 0, got {self.control_dt!r}"
            )
        object.__setattr__(self, "control_dt", float(self.control_dt))


@dataclass(frozen=True)
class ArticulationState:
    """The typed, per-articulation projection of one state snapshot.

    Every tensor is ``(num_envs, width)`` on the backend device, in policy
    joint order for the joint leaves.  Build one through
    :meth:`TaskState.articulation`; the accessor validates the published
    ``articulation/<name>/...`` keys instead of asking task code to parse
    string paths.
    """

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    applied_torque: torch.Tensor
    base_pos_w: torch.Tensor
    base_quat_w: torch.Tensor
    base_lin_vel_w: torch.Tensor
    base_ang_vel_w: torch.Tensor
    base_lin_vel_b: torch.Tensor
    base_ang_vel_b: torch.Tensor
    gravity_b: torch.Tensor


def _articulation_names(extra: Mapping[str, object]) -> list[str]:
    """Articulation names present in the namespaced extra record."""
    names: set[str] = set()
    for key in extra:
        if not isinstance(key, str) or not key.startswith("articulation/"):
            continue
        parts = key.split("/")
        if len(parts) >= 3 and parts[1]:
            names.add(parts[1])
    return sorted(names)


def _require_articulation_tensor(
    value: object,
    key: str,
    num_envs: int,
    *,
    width: int | None,
) -> torch.Tensor:
    """Validate one published articulation tensor and return it unchanged."""
    if not isinstance(value, torch.Tensor):
        raise TaskContractError(
            f"TaskState.articulation: extra[{key!r}] must be a torch.Tensor, "
            f"got {type(value).__name__}"
        )
    if not value.is_floating_point():
        raise TaskContractError(
            f"TaskState.articulation: extra[{key!r}] must be floating point, " f"got {value.dtype}"
        )
    if value.ndim != 2:
        raise TaskContractError(
            f"TaskState.articulation: extra[{key!r}] must be (num_envs, width), "
            f"got shape {tuple(value.shape)}"
        )
    if value.shape[0] != num_envs:
        raise TaskContractError(
            f"TaskState.articulation: extra[{key!r}] must have {num_envs} rows, "
            f"got {value.shape[0]}"
        )
    if value.shape[1] == 0:
        raise TaskContractError(
            f"TaskState.articulation: extra[{key!r}] must not have an empty " "feature dimension"
        )
    if width is not None and value.shape[1] != width:
        raise TaskContractError(
            f"TaskState.articulation: extra[{key!r}] must have width {width}, "
            f"got {value.shape[1]}"
        )
    return value


@dataclass(frozen=True)
class TaskState:
    """One engine-state snapshot, pre-reset unless documented otherwise.

    Attributes:
        t: Policy tick index the snapshot belongs to (non-negative).
        num_envs: Batch size ``N``.
        extra: The backend ``extra`` record (including ``reset_env_ids`` /
            ``should_reset`` defaults).
        obs_cache: The backend observation cache after the step/reset.
    """

    t: int
    num_envs: int
    extra: Mapping[str, object]
    obs_cache: Mapping[str, object]

    def __post_init__(self) -> None:
        _require_int(self.t, "TaskState.t", minimum=0)
        _require_int(self.num_envs, "TaskState.num_envs", minimum=1)
        extra = _require_mapping(self.extra, "TaskState.extra")
        obs_cache = _require_mapping(self.obs_cache, "TaskState.obs_cache")
        object.__setattr__(self, "extra", dict(extra))
        object.__setattr__(self, "obs_cache", dict(obs_cache))

    def articulation(self, name: str) -> ArticulationState:
        """Return the typed state of one articulation.

        Args:
            name: The core articulation name (the ``articulation/<name>/...``
                key namespace).

        Returns:
            An :class:`ArticulationState` carrying the ten published leaves.

        Raises:
            TaskContractError: The name is malformed, the articulation is not
                published, or a leaf is missing/malformed (wrong type, not
                floating point, not 2-D, wrong row count or width).
        """
        if not isinstance(name, str) or not name.strip():
            raise TaskContractError(
                "TaskState.articulation: name must be a non-empty string, " f"got {name!r}"
            )
        prefix = f"articulation/{name}/"
        available = _articulation_names(self.extra)
        values: dict[str, torch.Tensor] = {}
        joint_width: int | None = None
        for leaf in _ARTICULATION_LEAVES:
            key = f"{prefix}{leaf}"
            if key not in self.extra:
                hint = (
                    f" | available articulations: {available}"
                    if available
                    else " | hint: the run publishes these keys only for a " "declared articulation"
                )
                raise TaskContractError(
                    f"TaskState.articulation({name!r}): missing extra[{key!r}]" f"{hint}"
                )
            width = _ARTICULATION_WIDTHS.get(leaf)
            if leaf in _ARTICULATION_JOINTS:
                if joint_width is None:
                    value = _require_articulation_tensor(
                        self.extra[key], key, self.num_envs, width=None
                    )
                    joint_width = int(value.shape[1])
                else:
                    value = _require_articulation_tensor(
                        self.extra[key], key, self.num_envs, width=joint_width
                    )
            else:
                value = _require_articulation_tensor(
                    self.extra[key], key, self.num_envs, width=width
                )
            values[leaf] = value
        return ArticulationState(**values)


@dataclass(frozen=True)
class Transition:
    """One policy tick as seen by the task hooks.

    ``obs`` is the observation the actions were chosen from, ``next_obs`` the
    observation after the step.  ``terminated`` / ``truncated`` are the final
    task masks: they are all-False in the transition passed to the
    ``terminated`` / ``truncated`` hooks themselves (the masks are being
    computed there) and filled in for ``reward``, ``termination_reasons`` and
    ``on_episode_end``.
    """

    obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: TaskState

    def __post_init__(self) -> None:
        if not isinstance(self.info, TaskState):
            raise TaskContractError(
                f"Transition.info must be a TaskState, got {type(self.info).__name__}"
            )
        num_envs = self.info.num_envs
        for name in ("obs", "action", "next_obs"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TaskContractError(
                    f"Transition.{name} must be a torch.Tensor, got {type(value).__name__}"
                )
            if value.ndim != 2:
                raise TaskContractError(
                    f"Transition.{name} must be 2-D, got shape {tuple(value.shape)}"
                )
            if value.shape[0] != num_envs:
                raise TaskContractError(
                    f"Transition.{name} must have {num_envs} rows, got {value.shape[0]}"
                )
        if self.obs.shape[1] != self.next_obs.shape[1]:
            raise TaskContractError(
                "Transition.obs and Transition.next_obs must have the same feature "
                f"dimension, got {self.obs.shape[1]} and {self.next_obs.shape[1]}"
            )
        _require_mask(self.terminated, "Transition.terminated", num_envs)
        _require_mask(self.truncated, "Transition.truncated", num_envs)


@dataclass(frozen=True)
class EpisodeEnd:
    """The done-row record of one step (batch of finished episodes).

    ``env_ids`` and every mask are aligned over the done rows only (length
    ``M``), not the full ``num_envs`` batch.  ``state`` is the pre-reset
    snapshot.
    """

    env_ids: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    reasons: Mapping[str, torch.Tensor]
    state: TaskState

    def __post_init__(self) -> None:
        if not isinstance(self.state, TaskState):
            raise TaskContractError(
                f"EpisodeEnd.state must be a TaskState, got {type(self.state).__name__}"
            )
        if not isinstance(self.env_ids, torch.Tensor):
            raise TaskContractError(
                f"EpisodeEnd.env_ids must be a torch.Tensor, got {type(self.env_ids).__name__}"
            )
        if self.env_ids.ndim != 1 or self.env_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TaskContractError(
                "EpisodeEnd.env_ids must be a 1-D integer tensor, got "
                f"{self.env_ids.dtype} shape {tuple(self.env_ids.shape)}"
            )
        count = int(self.env_ids.numel())
        if count:
            if int(self.env_ids.min()) < 0 or int(self.env_ids.max()) >= self.state.num_envs:
                raise TaskContractError(
                    f"EpisodeEnd.env_ids must be in [0, {self.state.num_envs}), "
                    f"got {self.env_ids.tolist()}"
                )
            if int(torch.unique(self.env_ids).numel()) != count:
                raise TaskContractError(
                    f"EpisodeEnd.env_ids must be unique, got {self.env_ids.tolist()}"
                )
        _require_mask(self.terminated, "EpisodeEnd.terminated", count)
        _require_mask(self.truncated, "EpisodeEnd.truncated", count)
        reasons = _require_mapping(self.reasons, "EpisodeEnd.reasons")
        for name, mask in reasons.items():
            if not isinstance(name, str):
                raise TaskContractError(
                    f"EpisodeEnd.reasons keys must be str, got {type(name).__name__}"
                )
            _require_mask(mask, f"EpisodeEnd.reasons[{name!r}]", count)
        object.__setattr__(self, "reasons", dict(reasons))
