"""Pack-owned condition DSL over resolved step sources.

A condition is a pure, per-env boolean predicate.  It reads one or more
:class:`~unigym_bench.sources.SourceRef` values through a
:class:`~unigym_bench.sources.SourceResolver`, compares or combines them, and
returns a ``(num_envs,)`` ``torch.bool`` tensor on the step device.  It owns no
history, window, trigger state, event or terminal.

Vocabulary:

- :class:`Comparison` compares one tensor source with a finite scalar using
  ``gt``, ``lt``, ``gte``, ``lte``, ``eq`` or an ``abs_*`` variant.  Without
  ``dim`` the source must be a per-env 1-D tensor.  With ``dim`` it must be a
  ``(num_envs, width)`` tensor and ``dim`` selects an index from the trailing
  (last) axis, matching the core height/tilt condition convention.
- :class:`All` is a per-env logical AND.  It is not a temporal quantifier and
  keeps no state across ticks.
- :func:`parse_condition` builds a condition tree from a closed, JSON-safe
  mapping.  Unknown keys and operators are rejected.  Temporal operators
  (``always``, ``never``, ``fraction_true``, ``hold_until_end``) are reserved:
  the parser fails with a teaching error instead of guessing semantics.

JSON sources are transport values, not comparison operands.  A future typed
scalar condition may compare them; :class:`Comparison` rejects them fast.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, cast, runtime_checkable

import torch
from torch import Tensor
from unigym.evaluation import EnvStepContext

from .json import JsonValue
from .sources import SourceError, SourceRef, SourceResolver

__all__ = [
    "All",
    "Comparison",
    "ComparisonOp",
    "Condition",
    "ConditionError",
    "ConditionParseError",
    "TemporalOperatorError",
    "parse_condition",
]

#: Comparison operators accepted by :class:`Comparison`.
ComparisonOp: TypeAlias = Literal[
    "gt",
    "lt",
    "gte",
    "lte",
    "eq",
    "abs_gt",
    "abs_lt",
    "abs_gte",
    "abs_lte",
]

_OPS: tuple[ComparisonOp, ...] = (
    "gt",
    "lt",
    "gte",
    "lte",
    "eq",
    "abs_gt",
    "abs_lt",
    "abs_gte",
    "abs_lte",
)
_OP_SET: frozenset[str] = frozenset(_OPS)

_LEAF_KEYS: frozenset[str] = frozenset({"source", "op", "value", "dim"})
_ALL_KEYS: frozenset[str] = frozenset({"all"})
_RESERVED_TEMPORAL: tuple[str, ...] = (
    "always",
    "never",
    "fraction_true",
    "hold_until_end",
)


class ConditionError(ValueError):
    """A condition could not evaluate its declared source."""


class ConditionParseError(ConditionError):
    """A condition mapping does not satisfy the closed authoring schema."""


class TemporalOperatorError(ConditionParseError, NotImplementedError):
    """A reserved temporal operator was requested before it is implemented."""


@runtime_checkable
class Condition(Protocol):
    """A pure per-env boolean predicate over one step boundary.

    Implementations return a ``(step.num_envs,)`` ``torch.bool`` tensor on
    ``step.device``.  They never mutate the step, retain a borrowed tensor or
    keep cross-tick state.
    """

    def evaluate(self, step: EnvStepContext, resolver: SourceResolver) -> Tensor:
        """Return the per-env bool mask for ``step``."""
        ...


@dataclass(frozen=True, slots=True)
class Comparison:
    """Compare one tensor source against a finite scalar.

    Boolean, integer and floating sources are accepted and converted to
    ``float64`` for the comparison; complex sources are rejected.

    Args:
        ref: Source reference producing a per-env tensor.
        op: One of the :data:`ComparisonOp` operators.
        value: Finite scalar threshold.
        dim: Optional trailing-axis index.  ``None`` requires a 1-D per-env
            tensor; an integer requires a ``(num_envs, width)`` tensor and
            selects ``tensor[..., dim]`` (negative indices are accepted).
    """

    ref: SourceRef
    op: ComparisonOp
    value: float
    dim: int | None = None

    def __post_init__(self) -> None:
        """Validate the reference, operator, threshold and axis index."""
        if not isinstance(self.ref, SourceRef):
            raise TypeError(
                "Comparison.ref must be a SourceRef, "
                f"got {type(self.ref).__name__}; use SourceRef.parse('kind:key')"
            )
        if not isinstance(self.op, str) or self.op not in _OP_SET:
            raise ConditionError(f"Comparison.op must be one of {list(_OPS)}, got {self.op!r}")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError(
                "Comparison.value must be a finite int or float, "
                f"got {type(self.value).__name__}"
            )
        try:
            number = float(self.value)
        except (OverflowError, ValueError) as exc:
            raise ConditionError(
                f"Comparison.value must be a finite number representable as a "
                f"float, got {self.value!r}"
            ) from exc
        if not math.isfinite(number):
            raise ConditionError(f"Comparison.value must be finite, got {number!r}")
        object.__setattr__(self, "value", number)
        if self.dim is not None and (isinstance(self.dim, bool) or not isinstance(self.dim, int)):
            raise TypeError(f"Comparison.dim must be an int or None, got {type(self.dim).__name__}")

    def evaluate(self, step: EnvStepContext, resolver: SourceResolver) -> Tensor:
        """Return the per-env comparison mask.

        Raises:
            ConditionError: The source resolves to JSON, has an incompatible
                shape, is complex, or lives on another device.
        """
        source = resolver.resolve(step, self.ref)
        if not isinstance(source, Tensor):
            raise ConditionError(
                f"comparison source {self.ref} resolved to a JSON value "
                f"({type(source).__name__}); JSON sources are not comparable "
                "without a future typed scalar condition"
            )
        values = _read_per_env(source, step, self.ref, self.dim)
        if values.device != step.device:
            raise ConditionError(
                f"comparison source {self.ref} lives on device {values.device}, "
                f"expected the step device {step.device}"
            )
        if values.is_complex():
            raise ConditionError(
                f"comparison source {self.ref} must be boolean, integer or "
                f"floating point, got complex dtype {values.dtype}"
            )
        comparable = values.to(dtype=torch.float64)
        finite = torch.isfinite(comparable)
        if not bool(finite.all()):
            bad = int((~finite).nonzero(as_tuple=False)[0].item())
            raise ConditionError(
                f"comparison source {self.ref} has non-finite value "
                f"{comparable[bad].item()!r} at env {bad}; condition sources must "
                "be finite"
            )
        return _compare(comparable, self.op, self.value)

    def to_mapping(self) -> dict[str, JsonValue]:
        """Return the closed, JSON-safe leaf mapping accepted by the parser.

        ``source`` stays in the authoring ``kind:key`` string form.  The
        mapping form of :class:`unigym_bench.sources.SourceRef` exists only for
        explicit source construction and is not a second condition grammar.
        """
        mapping: dict[str, JsonValue] = {
            "source": str(self.ref),
            "op": self.op,
            "value": self.value,
        }
        if self.dim is not None:
            mapping["dim"] = self.dim
        return mapping


@dataclass(frozen=True, slots=True)
class All:
    """Per-env logical AND of a non-empty sequence of conditions.

    ``All`` is evaluated at one boundary.  It is not a temporal "for all
    frames" quantifier and keeps no history or window state.
    """

    conditions: Sequence[Condition]

    def __post_init__(self) -> None:
        """Validate and freeze the child conditions."""
        raw = self.conditions
        if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
            raise TypeError(
                f"All.conditions must be a sequence of Condition, got {type(raw).__name__}"
            )
        conditions = tuple(raw)
        if not conditions:
            raise ConditionError(
                "All.conditions must contain at least one condition; "
                "an empty conjunction is not meaningful"
            )
        for index, child in enumerate(conditions):
            if not isinstance(child, Condition):
                raise TypeError(
                    f"All.conditions[{index}] must satisfy Condition "
                    f"(an object with evaluate), got {type(child).__name__}"
                )
        object.__setattr__(self, "conditions", conditions)

    def evaluate(self, step: EnvStepContext, resolver: SourceResolver) -> Tensor:
        """Return the per-env AND of every child condition.

        Raises:
            ConditionError: A child returns a non-tensor, a non-bool, a wrong
                shape or a tensor on another device.
        """
        combined: Tensor | None = None
        for index, condition in enumerate(self.conditions):
            current = condition.evaluate(step, resolver)
            _check_mask(current, step, f"All.conditions[{index}]")
            combined = current if combined is None else combined & current
        assert combined is not None
        return combined

    def to_mapping(self) -> dict[str, JsonValue]:
        """Return the closed, JSON-safe conjunction mapping.

        Every child must expose ``to_mapping``.  A custom condition without that
        JSON-safe form cannot be used in a monitor config fingerprint and fails
        here with a teaching error.

        Raises:
            ConditionError: A child has no ``to_mapping`` method.
        """
        children: list[JsonValue] = []
        for index, condition in enumerate(self.conditions):
            encode = getattr(condition, "to_mapping", None)
            if not callable(encode):
                raise ConditionError(
                    f"All.conditions[{index}] of type {type(condition).__name__} has no "
                    "'to_mapping' method; a condition used by a predicate monitor must "
                    "have a JSON-safe config form (use Comparison/All or parse_condition)"
                )
            children.append(encode())
        return {"all": children}


def parse_condition(document: object) -> Condition:
    """Build a condition tree from a closed JSON-safe mapping.

    Leaf form::

        {"source": "record:foo", "op": "gt", "value": 1.0, "dim": 2}

    ``dim`` is optional.  Conjunction form::

        {"all": [<condition>, <condition>, ...]}

    Args:
        document: A raw mapping (normally decoded JSON/YAML data).

    Returns:
        A :class:`Comparison` or :class:`All`.

    Raises:
        ConditionParseError: The mapping is malformed, has unknown keys or
            operators, or uses a bad source reference.
        TemporalOperatorError: A reserved temporal operator key is present.
    """
    if not isinstance(document, Mapping):
        raise ConditionParseError(f"condition must be a mapping, got {type(document).__name__}")
    reserved = [key for key in _RESERVED_TEMPORAL if key in document]
    if reserved:
        raise TemporalOperatorError(
            f"temporal operator {reserved[0]!r} is reserved and not implemented "
            "in pack v1; a condition is evaluated at one boundary only and "
            "'all' is a logical AND, not a temporal quantifier"
        )
    if "all" in document:
        _check_keys(document, _ALL_KEYS, "all condition")
        raw = document["all"]
        if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
            raise ConditionParseError(
                f"'all' must be a non-empty list of conditions, got {type(raw).__name__}"
            )
        if not raw:
            raise ConditionParseError("'all' must contain at least one condition")
        return All([parse_condition(child) for child in raw])
    _check_keys(document, _LEAF_KEYS, "leaf condition")
    missing = [key for key in ("source", "op", "value") if key not in document]
    if missing:
        raise ConditionParseError(
            f"leaf condition is missing required keys {missing}; "
            f"expected keys {sorted(_LEAF_KEYS)}"
        )
    source = document["source"]
    if not isinstance(source, str):
        raise ConditionParseError(
            "leaf condition 'source' must be a string like 'record:<key>', "
            f"got {type(source).__name__}"
        )
    try:
        ref = SourceRef.parse(source)
    except SourceError as exc:
        raise ConditionParseError(f"leaf condition 'source': {exc}") from exc
    op = document["op"]
    if not isinstance(op, str):
        raise ConditionParseError(f"leaf condition 'op' must be a string, got {type(op).__name__}")
    if op not in _OP_SET:
        raise ConditionParseError(
            f"leaf condition 'op' {op!r} is not supported; expected one of {list(_OPS)}"
        )
    value = document["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConditionParseError(
            "leaf condition 'value' must be a finite number, " f"got {type(value).__name__}"
        )
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ConditionParseError(
            "leaf condition 'value' must be a finite number representable as a "
            f"float, got {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ConditionParseError(f"leaf condition 'value' must be finite, got {number!r}")
    dim: int | None = None
    if "dim" in document:
        raw_dim = document["dim"]
        if isinstance(raw_dim, bool) or not isinstance(raw_dim, int):
            raise ConditionParseError(
                "leaf condition 'dim' must be an integer trailing-axis index, " f"got {raw_dim!r}"
            )
        dim = raw_dim
    # ``op`` was validated against _OP_SET above.
    return Comparison(ref=ref, op=cast(ComparisonOp, op), value=number, dim=dim)


def _read_per_env(
    source: Tensor,
    step: EnvStepContext,
    ref: SourceRef,
    dim: int | None,
) -> Tensor:
    """Validate one source tensor and return its per-env comparison values.

    The v1 contract is deliberately narrow: without ``dim`` the source is 1-D,
    and with ``dim`` it is exactly 2-D ``(num_envs, width)``.  Higher-rank
    sources are rejected instead of silently reducing an ambiguous axis.
    """
    if source.ndim == 0:
        raise ConditionError(f"comparison source {ref} must be a per-env tensor, got a scalar")
    if source.shape[0] != step.num_envs:
        raise ConditionError(
            f"comparison source {ref} must have leading dimension "
            f"num_envs={step.num_envs}, got shape {tuple(source.shape)}"
        )
    if dim is None:
        if source.ndim != 1:
            raise ConditionError(
                f"comparison source {ref} must be 1-D when dim is None, "
                f"got shape {tuple(source.shape)}; pass dim=<trailing index>"
            )
        return source
    if source.ndim != 2:
        raise ConditionError(
            f"comparison source {ref} must be 2-D (num_envs, width) when "
            f"dim is set, got shape {tuple(source.shape)}"
        )
    width = int(source.shape[-1])
    if not (-width <= dim < width):
        raise ConditionError(
            f"comparison source {ref} dim={dim} is outside the trailing axis " f"of size {width}"
        )
    return source[..., dim]


def _compare(values: Tensor, op: ComparisonOp, threshold: float) -> Tensor:
    """Apply one comparison operator to per-env values."""
    if op == "gt":
        return values > threshold
    if op == "lt":
        return values < threshold
    if op == "gte":
        return values >= threshold
    if op == "lte":
        return values <= threshold
    if op == "eq":
        return values == threshold
    magnitude = torch.abs(values)
    if op == "abs_gt":
        return magnitude > threshold
    if op == "abs_lt":
        return magnitude < threshold
    if op == "abs_gte":
        return magnitude >= threshold
    return magnitude <= threshold


def _check_mask(value: object, step: EnvStepContext, where: str) -> None:
    """Validate one child condition result."""
    if not isinstance(value, Tensor):
        raise ConditionError(f"{where} returned {type(value).__name__}, expected a torch.Tensor")
    expected = (step.num_envs,)
    if tuple(value.shape) != expected:
        raise ConditionError(f"{where} returned shape {tuple(value.shape)}, expected {expected}")
    if value.dtype != torch.bool:
        raise ConditionError(f"{where} returned dtype {value.dtype}, expected torch.bool")
    if value.device != step.device:
        raise ConditionError(f"{where} returned device {value.device}, expected {step.device}")


def _check_keys(document: Mapping[str, object], allowed: frozenset[str], where: str) -> None:
    """Reject unknown keys in one closed condition mapping."""
    unknown = sorted(repr(key) for key in document if key not in allowed)
    if unknown:
        raise ConditionParseError(
            f"{where} has unknown keys {unknown}; allowed keys {sorted(allowed)}"
        )
