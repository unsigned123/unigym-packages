"""Per-env fixed-capacity history buffers (R2.5).

``RingBuffer`` stores one chronological history per env.  Each env owns its
write pointer and valid count, so a reset of one row never disturbs another.
``latest`` / ``history`` are oldest -> newest and zero-fill slots that have not
been written yet.  The buffer never pushes anything implicitly and has no
dependency on ``TaskEnv``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

__all__ = ["RingBuffer"]

_INT_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)


def _require_int(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


class RingBuffer:
    """A ``(num_envs, capacity, *shape)`` per-env chronological ring."""

    def __init__(
        self,
        num_envs: int,
        capacity: int,
        shape: tuple[int, ...] = (),
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device | None = None,
    ) -> None:
        self.num_envs = _require_int(num_envs, "RingBuffer.num_envs", minimum=1)
        self.capacity = _require_int(capacity, "RingBuffer.capacity", minimum=1)
        if not isinstance(shape, tuple):
            raise TypeError(f"RingBuffer.shape must be a tuple, got {type(shape).__name__}")
        self.shape = tuple(
            _require_int(dim, f"RingBuffer.shape[{index}]", minimum=0)
            for index, dim in enumerate(shape)
        )
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"RingBuffer.dtype must be a torch.dtype, got {type(dtype).__name__}")
        try:
            self.device = torch.device("cpu") if device is None else torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise TypeError(f"RingBuffer.device is not a valid device: {device!r}") from exc
        self.dtype = dtype

        self.data = torch.zeros(
            (self.num_envs, self.capacity, *self.shape),
            dtype=dtype,
            device=self.device,
        )
        self._ptr = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._count = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

    @property
    def count(self) -> torch.Tensor:
        """Per-env number of valid slots, capped at ``capacity``."""
        return self._count.clone()

    def push(self, values: torch.Tensor) -> None:
        """Append one ``(num_envs, *shape)`` value per env and advance pointers."""
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"RingBuffer.push expects a torch.Tensor, got {type(values).__name__}")
        expected = (self.num_envs, *self.shape)
        if tuple(values.shape) != expected:
            raise ValueError(f"RingBuffer.push expects shape {expected}, got {tuple(values.shape)}")
        values = values.to(device=self.device, dtype=self.dtype)
        rows = torch.arange(self.num_envs, device=self.device)
        self.data[rows, self._ptr] = values
        self._ptr = (self._ptr + 1) % self.capacity
        self._count = torch.clamp(self._count + 1, max=self.capacity)

    def latest(self, n: int = 1) -> torch.Tensor:
        """Return the newest *n* slots per env in chronological order."""
        n = _require_int(n, "RingBuffer.latest(n)", minimum=1)
        if n > self.capacity:
            raise ValueError(f"RingBuffer.latest(n) must be <= capacity, got {n} > {self.capacity}")
        steps_back = torch.arange(n - 1, -1, -1, dtype=torch.int64, device=self.device)
        rows = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(self.num_envs, n)
        indices = (self._ptr.unsqueeze(1) - 1 - steps_back) % self.capacity
        values = self.data[rows, indices]
        valid = steps_back.unsqueeze(0) < self._count.unsqueeze(1)
        mask = valid.reshape(self.num_envs, n, *([1] * len(self.shape)))
        return values * mask

    def history(self) -> torch.Tensor:
        """Return the whole capacity per env in chronological order."""
        return self.latest(self.capacity)

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        """Clear data, pointers and counts for *env_ids* (all when ``None``)."""
        if env_ids is None:
            ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        else:
            ids = self._validate_ids(env_ids)
        if not ids.numel():
            return
        self.data[ids] = 0
        self._ptr[ids] = 0
        self._count[ids] = 0

    def _validate_ids(self, env_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            ids = env_ids
        else:
            if isinstance(env_ids, str | bytes) or not isinstance(env_ids, Sequence):
                raise TypeError(
                    "RingBuffer.reset(env_ids) expects a tensor or a sequence of ints, "
                    f"got {type(env_ids).__name__}"
                )
            try:
                ids = torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"RingBuffer.reset(env_ids) could not read ids from {env_ids!r}"
                ) from exc
        if ids.ndim != 1 or ids.dtype not in _INT_DTYPES:
            raise TypeError(
                "RingBuffer.reset(env_ids) expects a 1-D integer tensor, got "
                f"{ids.dtype} shape {tuple(ids.shape)}"
            )
        ids = ids.to(device=self.device, dtype=torch.int64)
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= self.num_envs):
            raise ValueError(
                f"RingBuffer.reset(env_ids) ids must be in [0, {self.num_envs}), "
                f"got {ids.tolist()}"
            )
        return ids
