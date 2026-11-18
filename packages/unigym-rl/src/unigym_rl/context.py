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
    from unigym.env.obs import ObsSnapshot, RobotObs
    from unigym.run.api import Run
    from unigym.run.instance import RunInstance
    from unigym.runtime.actions import OrderedActionSpec
    from unigym.runtime.controllers import ControlRuntime

__all__ = [
    "EpisodeEnd",
    "TaskContext",
    "TaskState",
    "Transition",
]


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


def _require_row_tensor(
    value: object,
    name: str,
    num_envs: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TaskContractError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.ndim != 1 or value.shape[0] != num_envs:
        raise TaskContractError(f"{name} must have shape ({num_envs},), got {tuple(value.shape)}")
    return value


def _require_ids(value: object, name: str, num_envs: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TaskContractError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.ndim != 1 or value.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TaskContractError(
            f"{name} must be a 1-D integer tensor, got {value.dtype} shape {tuple(value.shape)}"
        )
    if value.numel():
        if int(value.min()) < 0 or int(value.max()) >= num_envs:
            raise TaskContractError(f"{name} must be in [0, {num_envs}), got {value.tolist()}")
        if int(torch.unique(value).numel()) != int(value.numel()):
            raise TaskContractError(f"{name} must be unique, got {value.tolist()}")
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
class TaskState:
    """One engine-state snapshot, pre-reset unless documented otherwise.

    Attributes:
        t: Policy tick index the snapshot belongs to (non-negative).
        num_envs: Batch size ``N``.
        obs: The typed observation snapshot published by this tick.
        episode_id: Per-env episode counter (``(N,)`` int64), when published.
        time_in_episode: Per-env episode time in seconds (``(N,)``), when
            published.
        reset_env_ids: Env ids the core marked due for reset (``(K,)`` int64).
        should_reset: Whether ``reset_env_ids`` is non-empty.
    """

    t: int
    num_envs: int
    obs: ObsSnapshot
    episode_id: torch.Tensor | None = None
    time_in_episode: torch.Tensor | None = None
    reset_env_ids: torch.Tensor | None = None
    should_reset: bool = False

    def __post_init__(self) -> None:
        from unigym.env.obs import ObsSnapshot

        _require_int(self.t, "TaskState.t", minimum=0)
        _require_int(self.num_envs, "TaskState.num_envs", minimum=1)
        if not isinstance(self.obs, ObsSnapshot):
            raise TaskContractError(
                f"TaskState.obs must be an ObsSnapshot, got {type(self.obs).__name__}"
            )
        if self.obs.num_envs != self.num_envs:
            raise TaskContractError(
                f"TaskState.obs has {self.obs.num_envs} rows, expected {self.num_envs}"
            )
        if self.episode_id is not None:
            _require_row_tensor(self.episode_id, "TaskState.episode_id", self.num_envs)
        if self.time_in_episode is not None:
            _require_row_tensor(self.time_in_episode, "TaskState.time_in_episode", self.num_envs)
        if self.reset_env_ids is None:
            reset_ids = torch.zeros(0, dtype=torch.int64, device=self.obs.device)
            object.__setattr__(self, "reset_env_ids", reset_ids)
        else:
            _require_ids(self.reset_env_ids, "TaskState.reset_env_ids", self.num_envs)
        if not isinstance(self.should_reset, bool):
            raise TaskContractError(
                f"TaskState.should_reset must be a bool, got {type(self.should_reset).__name__}"
            )
        object.__setattr__(self, "should_reset", bool(self.should_reset))

    def robot(self, name: str) -> RobotObs:
        """Return the typed observation view of one articulation."""
        return self.obs.robot(name)


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
