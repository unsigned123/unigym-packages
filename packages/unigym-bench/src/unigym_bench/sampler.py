"""Scenario command sampler: per-env cursor, dwell and rewind state.

A :class:`ScenarioSampler` applies one command from a validated
:class:`~unigym_bench.scenario.Scenario` to every environment.  The pool
command normally changes only at an episode boundary; the optional
closed-loop path below can additionally apply a published desired command at
the next policy update.  The sampler owns all mutable runtime state:

- the applied buffer ``cmd`` of shape ``(num_envs, dim)`` and dtype
  ``float32``;
- one cursor per environment into the scenario's cyclic order;
- one dwell countdown per environment, in episodes;
- a private seeded CPU RNG used once to draw each environment's starting
  cursor.

Semantics are episode-granular and explicit:

- the traversal order is ``scenario.plan`` when declared, otherwise the pool
  order ``0..len(pool)-1``; an environment follows that order cyclically;
- each environment starts at a seeded random offset in that order, so
  ``seed`` controls the first command but every entry of ``order`` is still
  visited once per cycle;
- ``reset(env_ids)`` writes the environment's current command and never
  advances the cursor;
- ``update`` changes a command only when the bound signal view carries a
  valid ``scenario/desired_cmd``; otherwise the command clock is episodes,
  not seconds;
- ``on_episode_end`` decrements the dwell counter and advances the cursor only
  when the dwell block is exhausted; the runner then calls ``reset`` to apply
  the next command.

Closed-loop convention
----------------------

A monitor named ``scenario`` publishes two local level signals; the pack
engine prefixes its name and republishes them as ``scenario/desired_cmd`` and
the optional ``scenario/command_valid``.  This module owns the consumer side:

- ``scenario/desired_cmd`` is a ``float32`` tensor of shape
  ``(num_envs, dim)`` on the sampler device; every value must be finite;
- ``scenario/command_valid`` is an optional ``bool`` tensor of shape
  ``(num_envs,)``; when present, only the rows that are ``True`` are applied
  and every ``False`` row keeps its previous command;
- a missing ``desired_cmd`` leaves ``cmd`` unchanged, so the pool/cursor
  behavior of :meth:`ScenarioSampler.reset` and
  :meth:`ScenarioSampler.on_episode_end` is untouched;
- an invalid signal (wrong tensor type, shape, dtype or device) raises a
  teaching error instead of silently falling back.

The sampler never scans the signal namespace, never infers a command from
another monitor and never duplicates monitor detection: only the two names
above are read.  Applying a desired command changes ``cmd`` alone; it never
advances a cursor, consumes dwell or draws from the RNG.  Episode boundaries
still behave as before: ``on_episode_end`` advances the cursor and ``reset``
then writes the pool command, which can overwrite a desired command for the
reset environments until the next ``update`` re-applies the still-published
level signal.  A signal never drives the episode clock.  ``cmd`` is part of
:meth:`ScenarioSampler.rewind_state`, so it restores with the sampler while
the signal store itself rides the core ``SignalBus`` rewind provider.

Timing is the core runner order: ``sink.on_step(t)`` publishes after tick
``t``'s physics and observation snapshot, ``update(t+1)`` is the first read,
and the command it returns is used by ``env.step(t+2)``.  In-episode closed
loop therefore has two policy-tick latency from the observed view to the
acting command; this sampler cannot shorten it.

The sampler is rewind-capable.  Its state is JSON-safe and contains the
concrete buffer, cursors, dwell counters and RNG state; the pack streaming
engine never duplicates any of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from unigym.commands import CommandSampler

from .json import JsonValue, ensure_json_value
from .scenario import Scenario, command_key
from .specs import ScenarioSpec, SpecError

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from unigym.core.episode_result import EpisodeResultView
    from unigym.env.backends.base import EnvBackend
    from unigym.signals import SignalView

__all__ = [
    "COMMAND_VALID_SIGNAL",
    "DESIRED_CMD_SIGNAL",
    "SAMPLER_REWIND_SCHEMA",
    "ScenarioSampler",
    "make_sampler",
]

#: Schema id of :meth:`ScenarioSampler.rewind_state` payloads.
SAMPLER_REWIND_SCHEMA = "unigym_bench.sampler/1"

#: Namespaced level signal carrying the next closed-loop command.
DESIRED_CMD_SIGNAL = "scenario/desired_cmd"
#: Optional per-env validity mask gating :data:`DESIRED_CMD_SIGNAL`.
COMMAND_VALID_SIGNAL = "scenario/command_valid"

_STATE_KEYS = frozenset(
    {
        "schema",
        "dim",
        "num_envs",
        "dwell",
        "order",
        "cursor",
        "remaining",
        "rng",
        "cmd",
    }
)

_MISSING = object()


def make_sampler(
    *,
    dim: int,
    device: str | torch.device,
    num_envs: int,
    **params: object,
) -> CommandSampler:
    """Build the pack's scenario sampler from a command-section factory call.

    ``unigym`` resolves a ``command.module`` reference as
    ``factory(dim=..., device=..., num_envs=..., **params)``.  This factory
    accepts three named parameters:

    - ``scenario`` (required): a :class:`ScenarioSpec` or a prebuilt
      :class:`Scenario`;
    - ``seed`` (required): an integer seeding the private RNG;
    - ``dwell`` (optional, default ``1``): episodes per command.

    Args:
        dim: Command width declared by the run.
        device: Tensor device for the applied command buffer.
        num_envs: Number of environments.
        **params: The three parameters above; any other name is rejected.

    Returns:
        A rewind-capable :class:`ScenarioSampler`.

    Raises:
        TypeError: A required parameter is missing, an unknown parameter is
            supplied, or a parameter has the wrong type.
        SpecError: The scenario is invalid or ``dim`` does not match it.
    """
    scenario = params.pop("scenario", _MISSING)
    seed = params.pop("seed", _MISSING)
    dwell = params.pop("dwell", 1)
    if params:
        raise TypeError(
            "make_sampler: unknown parameter(s) "
            f"{sorted(params)}; expected 'scenario', 'seed' and optional 'dwell'"
        )
    if scenario is _MISSING:
        raise TypeError(
            "make_sampler: missing required parameter 'scenario' " "(a ScenarioSpec or Scenario)"
        )
    if seed is _MISSING:
        raise TypeError("make_sampler: missing required parameter 'seed'")
    return ScenarioSampler(
        scenario=_coerce_scenario(scenario),
        dim=dim,
        device=device,
        num_envs=num_envs,
        seed=_require_int(seed, "make_sampler.seed"),
        dwell=_require_int(dwell, "make_sampler.dwell", minimum=1),
    )


class ScenarioSampler(CommandSampler):
    """Episode-granular sampler over a validated :class:`Scenario`.

    Args:
        scenario: The command pool and traversal order.
        dim: Command width; must equal ``scenario.dim``.
        device: Tensor device for ``cmd``.
        num_envs: Number of environments.
        seed: Integer seeding the private CPU RNG.
        dwell: Episodes to keep a command before advancing (default ``1``).

    Raises:
        TypeError: A parameter has the wrong type.
        SpecError: ``dim`` does not match the scenario.
    """

    rewind_capable = True

    def __init__(
        self,
        *,
        scenario: Scenario,
        dim: int,
        device: str | torch.device,
        num_envs: int,
        seed: int,
        dwell: int = 1,
    ) -> None:
        if not isinstance(scenario, Scenario):
            raise TypeError(
                "ScenarioSampler.scenario: expected a Scenario, " f"got {type(scenario).__name__}"
            )
        _require_positive_int(dim, "ScenarioSampler.dim")
        _require_positive_int(num_envs, "ScenarioSampler.num_envs")
        _require_int(seed, "ScenarioSampler.seed")
        _require_int(dwell, "ScenarioSampler.dwell", minimum=1)
        if scenario.dim != dim:
            raise SpecError(
                "ScenarioSampler.dim: expected the scenario width " f"{scenario.dim}, got {dim}"
            )
        if not isinstance(device, (str, torch.device)):
            raise TypeError(
                "ScenarioSampler.device: expected a str or torch.device, "
                f"got {type(device).__name__}"
            )
        super().__init__(dim, device, num_envs)
        self.scenario = scenario
        self.dwell = dwell
        self._order: tuple[int, ...] = scenario.order
        self._rng = torch.Generator(device="cpu")
        try:
            self._rng.manual_seed(seed)
        except RuntimeError as exc:
            raise ValueError(f"ScenarioSampler.seed: {exc}") from exc
        offsets = torch.randint(
            0,
            len(self._order),
            (num_envs,),
            generator=self._rng,
            dtype=torch.int64,
        )
        self._cursor: list[int] = [int(value) for value in offsets.tolist()]
        self._remaining: list[int] = [dwell] * num_envs
        self._commands = torch.tensor(scenario.commands, dtype=torch.float32, device=self.device)
        self._signals: SignalView | None = None
        self._write(torch.arange(num_envs, device=self.device))

    def reset(self, env_ids=None) -> torch.Tensor:
        """Apply each environment's current command; never advance a cursor."""
        self._write(self._resolve_ids(env_ids))
        return self.cmd

    def update(self, step_dt: float, cmd: torch.Tensor) -> torch.Tensor:
        """Apply the next valid closed-loop command, if one is published.

        The call reads ``scenario/desired_cmd`` and the optional
        ``scenario/command_valid`` from the bound signal view.  It changes
        ``cmd`` only; cursor, dwell and RNG stay untouched.  Without a bound
        view, or without a published desired command, the pool command is
        returned unchanged.

        Args:
            step_dt: Policy step duration (unused by this sampler).
            cmd: The runner's current command buffer (unused; ``self.cmd`` is
                the owned buffer).

        Returns:
            The sampler's applied command buffer.

        Raises:
            TypeError: A published signal is not a tensor or has the wrong
                dtype.
            ValueError: A published signal has the wrong shape or device, or
                a desired command is not finite.
        """
        del step_dt, cmd
        self._apply_desired_signal()
        return self.cmd

    def _apply_desired_signal(self) -> None:
        """Copy a valid published command into the owned buffer."""
        signals = self._signals
        if signals is None:
            return
        desired = signals.get(DESIRED_CMD_SIGNAL)
        if desired is None:
            return
        checked = self._validate_desired_cmd(desired)
        valid = signals.get(COMMAND_VALID_SIGNAL)
        if valid is None:
            self.cmd.copy_(checked)
            return
        mask = self._validate_command_valid(valid)
        if bool(mask.any()):
            self.cmd[mask] = checked[mask]

    def _validate_desired_cmd(self, value: object) -> torch.Tensor:
        """Validate one desired-command tensor and return it narrowed."""
        where = f"ScenarioSampler signal {DESIRED_CMD_SIGNAL!r}"
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{where}: expected a torch.Tensor, got {type(value).__name__}")
        expected_shape = (self.num_envs, self.dim)
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"{where}: expected shape {expected_shape}, got {tuple(value.shape)}")
        if value.dtype is not torch.float32:
            raise TypeError(f"{where}: expected dtype torch.float32, got {value.dtype}")
        device = torch.device(self.device)
        if value.device != device:
            raise ValueError(f"{where}: expected device {device}, got {value.device}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{where}: expected every value to be finite")
        return value

    def _validate_command_valid(self, value: object) -> torch.Tensor:
        """Validate one per-env validity mask and return it narrowed."""
        where = f"ScenarioSampler signal {COMMAND_VALID_SIGNAL!r}"
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{where}: expected a torch.Tensor, got {type(value).__name__}")
        expected_shape = (self.num_envs,)
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"{where}: expected shape {expected_shape}, got {tuple(value.shape)}")
        if value.dtype is not torch.bool:
            raise TypeError(f"{where}: expected dtype torch.bool, got {value.dtype}")
        device = torch.device(self.device)
        if value.device != device:
            raise ValueError(f"{where}: expected device {device}, got {value.device}")
        return value

    def on_episode_end(
        self,
        env_ids: torch.Tensor,
        results: Mapping[int, tuple[EpisodeResultView, ...]],
    ) -> None:
        """Consume one dwell episode and advance the cursor when it expires.

        The callback runs before ``reset``; ``reset`` applies the command the
        advanced cursor points at.  ``results`` is ignored in this card.
        """
        del results
        for env_id in self._resolve_ids(env_ids).tolist():
            index = int(env_id)
            self._remaining[index] -= 1
            if self._remaining[index] == 0:
                self._cursor[index] = (self._cursor[index] + 1) % len(self._order)
                self._remaining[index] = self.dwell

    def bind_env(self, env: EnvBackend) -> None:
        """Validate the live environment's shape at the build seam.

        The sampler keeps no reference to the environment: a restored sampler
        must not depend on having been rebound.
        """
        bound = getattr(env, "num_envs", None)
        if isinstance(bound, bool) or not isinstance(bound, int):
            raise TypeError(
                "ScenarioSampler.bind_env: env.num_envs must be an int, " f"got {bound!r}"
            )
        if bound != self.num_envs:
            raise ValueError(
                "ScenarioSampler.bind_env: expected env.num_envs " f"{self.num_envs}, got {bound}"
            )

    def bind_signals(self, signals: SignalView) -> None:
        """Store the run's read-only signal view for closed-loop reads.

        The full :class:`~unigym.signals.SignalView` protocol (``get`` and
        ``snapshot``) is required even though only ``get`` is used, so a
        partial object cannot masquerade as a view.  The view is not
        serialized: signal values ride the core ``SignalBus`` rewind provider,
        while :meth:`rewind_state` carries the applied ``cmd`` buffer.  Only
        :meth:`update` reads it.
        """
        if not callable(getattr(signals, "get", None)) or not callable(
            getattr(signals, "snapshot", None)
        ):
            raise TypeError(
                "ScenarioSampler.bind_signals: expected a SignalView with callable "
                f"get(name) and snapshot(), got {type(signals).__name__}"
            )
        self._signals = signals

    def rewind_state(self) -> dict[str, JsonValue]:
        """Return a JSON-safe private copy of cursor, dwell, RNG and commands."""
        state: dict[str, JsonValue] = {
            "schema": SAMPLER_REWIND_SCHEMA,
            "dim": self.dim,
            "num_envs": self.num_envs,
            "dwell": self.dwell,
            "order": list(self._order),
            "cursor": list(self._cursor),
            "remaining": list(self._remaining),
            "rng": [int(value) for value in self._rng.get_state().tolist()],
            "cmd": self.cmd.detach().to("cpu", copy=True).tolist(),
        }
        normalized = ensure_json_value(state, where="ScenarioSampler rewind state")
        if not isinstance(normalized, dict):
            raise TypeError("ScenarioSampler rewind state: expected an object")
        return normalized

    def apply_rewind_state(self, state: dict[str, JsonValue]) -> None:
        """Restore a state produced by :meth:`rewind_state` in place.

        The payload is fully decoded and validated before any live state is
        touched, so an invalid checkpoint leaves the sampler unchanged.
        """
        decoded = _decode_state(state, self)
        self._rng.set_state(decoded.rng_state)
        self._cursor = decoded.cursor
        self._remaining = decoded.remaining
        self.cmd.copy_(decoded.cmd)

    def _write(self, env_ids: torch.Tensor) -> None:
        ids = [int(value) for value in env_ids.tolist()]
        if not ids:
            return
        indices = [self._order[self._cursor[env_id]] for env_id in ids]
        self.cmd[env_ids] = self._commands[
            torch.tensor(indices, dtype=torch.long, device=self.device)
        ]


class _DecodedState:
    def __init__(
        self,
        *,
        cursor: list[int],
        remaining: list[int],
        rng_state: torch.Tensor,
        cmd: torch.Tensor,
    ) -> None:
        self.cursor = cursor
        self.remaining = remaining
        self.rng_state = rng_state
        self.cmd = cmd


def _decode_state(state: object, sampler: ScenarioSampler) -> _DecodedState:
    document = ensure_json_value(state, where="ScenarioSampler rewind state")
    if not isinstance(document, dict):
        raise TypeError(
            "ScenarioSampler rewind state: expected an object, " f"got {type(document).__name__}"
        )
    unknown = sorted(set(document) - _STATE_KEYS)
    if unknown:
        raise ValueError(
            "ScenarioSampler rewind state: unknown key(s) "
            f"{unknown}; expected only {sorted(_STATE_KEYS)}"
        )
    missing = sorted(_STATE_KEYS - set(document))
    if missing:
        raise ValueError(f"ScenarioSampler rewind state: missing key(s) {missing}")
    if document["schema"] != SAMPLER_REWIND_SCHEMA:
        raise ValueError(
            "ScenarioSampler rewind state.schema: expected "
            f"{SAMPLER_REWIND_SCHEMA!r}, got {document['schema']!r}"
        )
    _require_state_int(document["dim"], "ScenarioSampler rewind state.dim", sampler.dim)
    _require_state_int(
        document["num_envs"], "ScenarioSampler rewind state.num_envs", sampler.num_envs
    )
    _require_state_int(document["dwell"], "ScenarioSampler rewind state.dwell", sampler.dwell)
    raw_order = document["order"]
    if not isinstance(raw_order, list) or tuple(raw_order) != sampler._order:
        raise ValueError(
            "ScenarioSampler rewind state.order: expected "
            f"{list(sampler._order)}, got {raw_order!r}"
        )
    cursor = _decode_ints(
        document["cursor"],
        "ScenarioSampler rewind state.cursor",
        count=sampler.num_envs,
        minimum=0,
        maximum=len(sampler._order) - 1,
    )
    remaining = _decode_ints(
        document["remaining"],
        "ScenarioSampler rewind state.remaining",
        count=sampler.num_envs,
        minimum=1,
        maximum=sampler.dwell,
    )
    raw_rng = document["rng"]
    if not isinstance(raw_rng, list) or not raw_rng:
        raise ValueError(
            "ScenarioSampler rewind state.rng: expected a non-empty list of "
            f"bytes, got {raw_rng!r}"
        )
    rng = _decode_ints(
        raw_rng,
        "ScenarioSampler rewind state.rng",
        count=len(raw_rng),
        minimum=0,
        maximum=255,
    )
    rng_state = torch.tensor(rng, dtype=torch.uint8)
    try:
        torch.Generator(device="cpu").set_state(rng_state)
    except RuntimeError as exc:
        raise ValueError(
            "ScenarioSampler rewind state.rng: invalid generator state " f"({exc})"
        ) from exc
    raw_cmd = document["cmd"]
    if not isinstance(raw_cmd, list) or len(raw_cmd) != sampler.num_envs:
        raise ValueError(
            "ScenarioSampler rewind state.cmd: expected "
            f"{sampler.num_envs} command rows, got {raw_cmd!r}"
        )
    cmd: list[list[float]] = []
    for env_id, row in enumerate(raw_cmd):
        where = f"ScenarioSampler rewind state.cmd[{env_id}]"
        if not isinstance(row, list):
            raise ValueError(f"{where}: expected a command list, got {row!r}")
        key = command_key(row)
        if len(key) != sampler.dim:
            raise ValueError(f"{where}: expected {sampler.dim} values, got {len(key)}")
        if tuple(float(value) for value in row) != key:
            raise ValueError(f"{where}: command is not float32-exact; expected {list(key)}")
        # A closed-loop command is applied from ``scenario/desired_cmd`` and
        # need not belong to the scenario pool; the checkpoint restores it as
        # written.  Dimension, finiteness and float32 exactness still hold.
        cmd.append(list(key))
    return _DecodedState(
        cursor=cursor,
        remaining=remaining,
        rng_state=rng_state,
        cmd=torch.tensor(cmd, dtype=torch.float32, device=sampler.device),
    )


def _decode_ints(
    raw: object,
    where: str,
    *,
    count: int,
    minimum: int,
    maximum: int,
) -> list[int]:
    if not isinstance(raw, list) or len(raw) != count:
        raise ValueError(f"{where}: expected a list of {count} ints, got {raw!r}")
    values: list[int] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where}[{index}]: expected an int, got {value!r}")
        if not minimum <= value <= maximum:
            raise ValueError(
                f"{where}[{index}]: expected an int in [{minimum}, {maximum}], " f"got {value}"
            )
        values.append(value)
    return values


def _require_state_int(raw: object, where: str, expected: int) -> None:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw != expected:
        raise ValueError(f"{where}: expected {expected}, got {raw!r}")


def _coerce_scenario(value: object) -> Scenario:
    if isinstance(value, Scenario):
        return value
    if isinstance(value, ScenarioSpec):
        return Scenario.from_spec(value)
    raise TypeError(
        "make_sampler.scenario: expected a ScenarioSpec or Scenario, " f"got {type(value).__name__}"
    )


def _require_int(value: object, where: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where}: expected an int, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{where}: expected an int >= {minimum}, got {value}")
    return value


def _require_positive_int(value: object, where: str) -> None:
    _require_int(value, where, minimum=1)
