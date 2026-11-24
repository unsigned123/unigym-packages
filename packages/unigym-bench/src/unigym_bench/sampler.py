"""Scenario command sampler: per-env cursor, dwell and rewind state.

A :class:`ScenarioSampler` applies one command from a validated
:class:`~unigym_bench.scenario.Scenario` to every environment and changes that
command only at an episode boundary.  The sampler owns all mutable runtime
state:

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
- ``update`` never changes a command (the command clock is episodes, not
  seconds);
- ``on_episode_end`` decrements the dwell counter and advances the cursor only
  when the dwell block is exhausted; the runner then calls ``reset`` to apply
  the next command.

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
    "SAMPLER_REWIND_SCHEMA",
    "ScenarioSampler",
    "make_sampler",
]

#: Schema id of :meth:`ScenarioSampler.rewind_state` payloads.
SAMPLER_REWIND_SCHEMA = "unigym_bench.sampler/1"

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
        """Return the applied buffer unchanged (commands are episode-granular)."""
        del step_dt, cmd
        return self.cmd

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
        """Store the read-only signal view; this card defines no signal reads."""
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
        if key not in sampler.scenario.commands:
            raise ValueError(f"{where}: command {list(key)} is not in the sampler scenario pool")
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
