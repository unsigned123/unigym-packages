"""Command-pool scenario values and observed-command grouping.

A :class:`~unigym_bench.specs.ScenarioSpec` is the authoring document: an
ordered command pool plus an optional plan.  This module turns that document
into a :class:`Scenario` whose command vectors are quantized to exact
``float32`` values and checked pairwise distinct, so two commands can never
silently collapse into one report bucket.

The scenario is device-neutral and JSON-oriented: commands are Python tuples of
``float`` values that are exactly representable in ``float32``.  A sampler
copies them onto its device; a report groups episodes by the same quantization.

Grouping uses the command observed on an episode's first tick
(:attr:`unigym_bench.events.EpisodeRecord.command_at_start`), never an episode
id.  Process-level episode ids are not a boundary key.

Every failure raises :class:`~unigym_bench.specs.SpecError` (a ``ValueError``)
naming the field and the expected/actual value.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .events import EpisodeRecord
from .specs import ScenarioSpec, SpecError

__all__ = [
    "Scenario",
    "command_key",
    "group_records_by_command",
]

_FLOAT32 = struct.Struct(">f")


@dataclass(frozen=True)
class Scenario:
    """A command pool and its cyclic traversal order.

    Args:
        commands: Non-empty pool of equal-length, finite command vectors.
            Values are quantized to ``float32`` at construction.
        plan: Optional pool indices defining the cyclic traversal order.  When
            omitted, the pool order (``0..len(commands)-1``) is used.

    The sampler owns every mutable runtime fact (cursor, dwell, RNG and the
    applied command buffer); this value is immutable and shareable.
    """

    commands: tuple[tuple[float, ...], ...]
    plan: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Quantize the pool, reject duplicates and validate the plan."""
        commands = _validate_commands(self.commands)
        object.__setattr__(self, "commands", commands)
        if self.plan is not None:
            object.__setattr__(self, "plan", _validate_plan(self.plan, len(commands)))

    @property
    def dim(self) -> int:
        """Command width (the number of values in every vector)."""
        return len(self.commands[0])

    @property
    def order(self) -> tuple[int, ...]:
        """The cyclic traversal order over :attr:`commands`."""
        if self.plan is not None:
            return self.plan
        return tuple(range(len(self.commands)))

    @classmethod
    def from_spec(cls, spec: ScenarioSpec) -> Scenario:
        """Build a validated scenario from an authoring :class:`ScenarioSpec`.

        Args:
            spec: A validated task-document scenario.

        Returns:
            A scenario whose commands are float32-exact and pairwise distinct.

        Raises:
            SpecError: ``spec`` is not a :class:`ScenarioSpec`, the pool is
                ragged or empty, or two commands collapse to one float32
                vector.
        """
        if not isinstance(spec, ScenarioSpec):
            raise SpecError("scenario: expected a ScenarioSpec, " f"got {type(spec).__name__}")
        return cls(commands=spec.command_pool, plan=spec.plan)


def command_key(command: Sequence[float]) -> tuple[float, ...]:
    """Return the float32-exact grouping key for an observed command.

    Args:
        command: A per-env command vector, for example an
            :attr:`EpisodeRecord.command_at_start` tuple.

    Returns:
        A hashable tuple whose values are exactly representable in
        ``float32``.  Negative zero is normalized to ``0.0`` so keys match
        numeric equality.

    Raises:
        SpecError: ``command`` is not a non-empty numeric sequence, a value is
            not a finite number, or a value is out of float32 range.
    """
    if not _is_sequence(command):
        raise SpecError(f"command: expected a command vector, got {type(command).__name__}")
    values = list(command)
    if not values:
        raise SpecError("command: expected a non-empty command vector")
    return tuple(_quantize(value, f"command[{index}]") for index, value in enumerate(values))


def group_records_by_command(
    records: Iterable[EpisodeRecord],
) -> dict[tuple[float, ...], tuple[EpisodeRecord, ...]]:
    """Group finalized episodes by their observed start command.

    Records are keyed by the float32-exact
    :attr:`EpisodeRecord.command_at_start`; an episode id is never used as a
    key.  Groups preserve first-observation order and their records preserve
    the input order, so a report can derive per-command and per-env counts
    without a hidden ordering.

    Args:
        records: Finalized episode records.

    Returns:
        A mapping from command key to the records that started with it.

    Raises:
        TypeError: An item is not an :class:`EpisodeRecord`.
        SpecError: A record has no observed start command (for example a
            collect-only run), so it cannot be attributed to a command.
    """
    grouped: dict[tuple[float, ...], list[EpisodeRecord]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, EpisodeRecord):
            raise TypeError(
                f"records[{index}]: expected an EpisodeRecord, " f"got {type(record).__name__}"
            )
        command = record.command_at_start
        if command is None:
            raise SpecError(
                f"records[{index}] (env {record.env_id}): no command_at_start "
                "was observed; cannot group the episode by command"
            )
        grouped.setdefault(command_key(command), []).append(record)
    return {key: tuple(values) for key, values in grouped.items()}


def _validate_commands(raw: object) -> tuple[tuple[float, ...], ...]:
    if not _is_sequence(raw):
        raise SpecError(
            "scenario.commands: expected a non-empty sequence of command "
            f"vectors, got {type(raw).__name__}"
        )
    rows = list(raw)
    if not rows:
        raise SpecError("scenario.commands: expected at least one command vector")
    commands: list[tuple[float, ...]] = []
    dimension: int | None = None
    for row_index, row in enumerate(rows):
        where = f"scenario.commands[{row_index}]"
        if not _is_sequence(row):
            raise SpecError(f"{where}: expected a command vector, got {type(row).__name__}")
        values = list(row)
        if not values:
            raise SpecError(f"{where}: command vector must not be empty")
        if dimension is None:
            dimension = len(values)
        elif len(values) != dimension:
            raise SpecError(
                f"{where}: expected {dimension} values to match "
                f"scenario.commands[0], got {len(values)}"
            )
        commands.append(
            tuple(
                _quantize(value, f"{where}[{value_index}]")
                for value_index, value in enumerate(values)
            )
        )
    seen: dict[tuple[float, ...], int] = {}
    for index, command in enumerate(commands):
        previous = seen.get(command)
        if previous is not None:
            raise SpecError(
                "scenario.commands: duplicate float32 command vector at "
                f"indices {previous} and {index}: {list(command)}"
            )
        seen[command] = index
    return tuple(commands)


def _validate_plan(raw: object, pool_size: int) -> tuple[int, ...]:
    if not _is_sequence(raw):
        raise SpecError(
            "scenario.plan: expected a non-empty sequence of command-pool "
            f"indices, got {type(raw).__name__}"
        )
    values = list(raw)
    if not values:
        raise SpecError("scenario.plan: expected at least one index")
    plan: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise SpecError(f"scenario.plan[{index}]: expected an int index, got {value!r}")
        if not 0 <= value < pool_size:
            raise SpecError(
                f"scenario.plan[{index}]: expected an index in [0, {pool_size}), " f"got {value}"
            )
        plan.append(value)
    return tuple(plan)


def _quantize(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(f"{where}: expected a finite number, got {value!r}")
    try:
        number = float(value)
    except OverflowError as exc:
        raise SpecError(f"{where}: expected a finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise SpecError(f"{where}: expected a finite number, got {value!r}")
    try:
        quantized = _FLOAT32.unpack(_FLOAT32.pack(number))[0]
    except OverflowError as exc:
        raise SpecError(f"{where}: {value!r} is not representable as a float32 command") from exc
    return 0.0 if quantized == 0.0 else quantized


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
