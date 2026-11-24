"""Pack-owned immutable per-tick frame passed to monitors.

A frame is the only data a monitor receives on a tick.  The streaming engine
builds a :class:`TickFrame` and borrows it for one callback; a monitor must not
retain the frame, the view it carries, or a tensor read from it.

The engine copies the small bookkeeping mapping at construction, so a frame
never aliases the runner bundle's mutable mappings.  Tensor fields are borrowed
from the boundary and carry the same rule: read them, do not keep them.

Episode finalization intentionally takes only the due env ids in this revision.
Richer per-episode records belong to the report-extension layer, not to the
monitor boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .json import JsonValue, ensure_json_value

if TYPE_CHECKING:
    from unigym.evaluation import EnvView

__all__ = ["TickFrame"]


@dataclass(frozen=True)
class TickFrame:
    """One tick's read-only facts for a monitor.

    Attributes:
        view: The boundary-scoped evaluation view.  Borrowed for this callback.
        t: Policy tick index.
        step_index: Per-env episode-local step indices, shape ``(num_envs,)``.
        time_s: Optional per-env episode time, shape ``(num_envs,)``.
        cmd: Optional observed command, leading dimension ``num_envs``.
        reset_state: Per-env reset facts, copied and JSON-validated.
        num_envs: Environment count for the whole run.
        device: Device every tensor in this frame lives on.
    """

    view: EnvView
    t: int
    step_index: Tensor
    time_s: Tensor | None
    cmd: Tensor | None
    reset_state: Mapping[int, JsonValue]
    num_envs: int
    device: torch.device

    def __post_init__(self) -> None:
        """Validate the frame and copy its bookkeeping mapping."""
        if self.view is None:
            raise TypeError("tick frame.view must be an EnvView, got None")
        if isinstance(self.num_envs, bool) or not isinstance(self.num_envs, int):
            raise TypeError(f"tick frame.num_envs must be an int, got {self.num_envs!r}")
        if self.num_envs <= 0:
            raise ValueError(f"tick frame.num_envs must be positive, got {self.num_envs}")
        if isinstance(self.t, bool) or not isinstance(self.t, int):
            raise TypeError(f"tick frame.t must be an int, got {self.t!r}")
        if self.t < 0:
            raise ValueError(f"tick frame.t must be >= 0, got {self.t}")
        if not isinstance(self.device, torch.device):
            raise TypeError(
                f"tick frame.device must be a torch.device, got {type(self.device).__name__}"
            )
        _require_tensor_vector(self.step_index, self, "step_index", kind="integer")
        if self.time_s is not None:
            _require_tensor_vector(self.time_s, self, "time_s", kind="floating")
        if self.cmd is not None:
            if not isinstance(self.cmd, Tensor):
                raise TypeError(f"tick frame.cmd must be a tensor or None, got {self.cmd!r}")
            if self.cmd.ndim < 1 or self.cmd.shape[0] != self.num_envs:
                raise ValueError(
                    "tick frame.cmd must have leading dimension "
                    f"num_envs={self.num_envs}, got shape {tuple(self.cmd.shape)}"
                )
            if self.cmd.device != self.device:
                raise ValueError(
                    f"tick frame.cmd device {self.cmd.device} does not match frame "
                    f"device {self.device}"
                )
        if not isinstance(self.reset_state, Mapping):
            raise TypeError(
                "tick frame.reset_state must be a mapping, "
                f"got {type(self.reset_state).__name__}"
            )
        reset_state: dict[int, JsonValue] = {}
        for key, value in self.reset_state.items():
            if isinstance(key, bool) or not isinstance(key, int):
                raise TypeError(f"tick frame.reset_state keys must be ints, got {key!r}")
            if key < 0 or key >= self.num_envs:
                raise ValueError(
                    f"tick frame.reset_state key {key} is outside [0, num_envs={self.num_envs})"
                )
            reset_state[key] = ensure_json_value(value, where=f"tick frame.reset_state[{key}]")
        object.__setattr__(self, "reset_state", MappingProxyType(reset_state))


def _require_tensor_vector(
    value: object,
    frame: TickFrame,
    field_name: str,
    *,
    kind: str,
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"tick frame.{field_name} must be a tensor, got {type(value).__name__}")
    if value.ndim != 1 or value.shape[0] != frame.num_envs:
        raise ValueError(
            f"tick frame.{field_name} must have shape (num_envs={frame.num_envs},), "
            f"got {tuple(value.shape)}"
        )
    if kind == "integer":
        if value.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                f"tick frame.{field_name} has dtype {value.dtype}, "
                "expected torch.int32 or torch.int64"
            )
    elif not value.is_floating_point():
        raise TypeError(
            f"tick frame.{field_name} has dtype {value.dtype}, expected a floating dtype"
        )
    if value.device != frame.device:
        raise ValueError(
            f"tick frame.{field_name} device {value.device} does not match frame "
            f"device {frame.device}"
        )
