"""Optional, pure-Python reward and termination managers (R3).

The managers are sugar: users may import them or ignore them and implement
``TaskEnv.reward`` / ``terminated`` / ``truncated`` directly.  They are not a
new environment type, they never reset or advance an episode, and they keep no
per-step state.  Every call is a pure function of the ``Transition`` it is
given.

The managers deliberately do not know ``num_envs``: term outputs are validated
for shape/dtype (``(N,)`` floating for rewards, ``(N,)`` bool for masks) and for
agreement between terms, while the final ``N`` check stays in
:class:`unigym_rl.task.TaskEnv`.

This module depends only on ``torch`` and ``unigym_rl`` itself (no
``gymnasium``, no ``skrl``, no core imports).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from unigym_rl.context import Transition
from unigym_rl.errors import TaskConfigError, TaskContractError

__all__ = [
    "RewardManager",
    "RewardTerm",
    "TerminationManager",
    "TerminationResult",
    "TerminationTerm",
]

_TERMINATION_MODES = ("terminated", "truncated")


@dataclass(frozen=True)
class RewardTerm:
    """One named reward contribution.

    ``fn`` is called with the step ``Transition`` and must return a 1-D
    floating tensor.  ``weight`` scales that tensor; disabled terms are skipped
    entirely.
    """

    name: str
    fn: Callable[[Transition], torch.Tensor]
    weight: float = 1.0
    enabled: bool = True


@dataclass(frozen=True)
class TerminationTerm:
    """One named termination condition.

    ``fn`` returns a 1-D bool mask.  ``mode`` decides whether a True row ends
    the episode as ``terminated`` or ``truncated``.
    """

    name: str
    fn: Callable[[Transition], torch.Tensor]
    mode: Literal["terminated", "truncated"] = "terminated"


@dataclass(frozen=True)
class TerminationResult:
    """The masks and per-term reason masks produced by a ``TerminationManager``.

    ``reasons`` maps every declared term name to its raw (unmerged) mask;
    ``terminated`` / ``truncated`` are the OR over the terms of each mode.
    ``TaskEnv`` still merges core episode-clock due ids into ``truncated``.
    """

    terminated: torch.Tensor
    truncated: torch.Tensor
    reasons: Mapping[str, torch.Tensor]


def _require_name(name: object, kind: str) -> str:
    if not isinstance(name, str):
        raise TaskConfigError(f"{kind}.name must be a str, got {type(name).__name__}")
    if not name.strip():
        raise TaskConfigError(f"{kind}.name must be a non-empty string, got {name!r}")
    return name


def _require_callable(fn: object, name: str, kind: str) -> Callable[[Transition], torch.Tensor]:
    if not callable(fn):
        raise TaskConfigError(f"{kind} {name!r}.fn must be callable, got {type(fn).__name__}")
    return fn


def _require_mode(mode: object, name: str) -> str:
    if not isinstance(mode, str) or mode not in _TERMINATION_MODES:
        raise TaskConfigError(
            f"TerminationTerm {name!r}.mode must be one of {_TERMINATION_MODES}, got {mode!r}"
        )
    return mode


def _require_weight(weight: object, name: str) -> float:
    if isinstance(weight, bool) or not isinstance(weight, int | float):
        raise TaskConfigError(
            f"RewardTerm {name!r}.weight must be a number, got {type(weight).__name__}"
        )
    value = float(weight)
    if not math.isfinite(value):
        raise TaskConfigError(f"RewardTerm {name!r}.weight must be finite, got {value!r}")
    return value


def _require_clip(clip: object) -> tuple[float, float] | None:
    if clip is None:
        return None
    if not isinstance(clip, tuple) or len(clip) != 2:
        raise TaskConfigError(
            "RewardManager clip must be a (low, high) tuple or None, "
            f"got {clip!r} ({type(clip).__name__})"
        )
    bounds: list[float] = []
    for label, bound in zip(("low", "high"), clip, strict=True):
        if isinstance(bound, bool) or not isinstance(bound, int | float):
            raise TaskConfigError(
                f"RewardManager clip {label} must be a number, got {type(bound).__name__}"
            )
        value = float(bound)
        if not math.isfinite(value):
            raise TaskConfigError(f"RewardManager clip {label} must be finite, got {value!r}")
        bounds.append(value)
    if bounds[0] > bounds[1]:
        raise TaskConfigError(
            f"RewardManager clip low must be <= high, got ({bounds[0]!r}, {bounds[1]!r})"
        )
    return (bounds[0], bounds[1])


def _require_terms(terms: object, kind: str, term_type: type) -> tuple:
    if isinstance(terms, str | bytes) or not isinstance(terms, Sequence):
        raise TaskConfigError(f"{kind} expects a sequence of terms, got {type(terms).__name__}")
    resolved = tuple(terms)
    if not resolved:
        raise TaskConfigError(
            f"{kind} requires at least one {term_type.__name__}; an empty manager "
            "cannot produce a reward/termination mask"
        )
    for term in resolved:
        if not isinstance(term, term_type):
            raise TaskConfigError(
                f"{kind} expects {term_type.__name__} entries, got {type(term).__name__}"
            )
    return resolved


def _require_unique_names(names: Sequence[str], kind: str) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise TaskConfigError(f"{kind} has a duplicate term name {name!r}")
        seen.add(name)


def _reward_value(term: RewardTerm, transition: Transition) -> torch.Tensor:
    value = term.fn(transition)
    if not isinstance(value, torch.Tensor):
        raise TaskContractError(
            f"reward term {term.name!r} must return a torch.Tensor, got {type(value).__name__}"
        )
    if value.ndim != 1:
        raise TaskContractError(
            f"reward term {term.name!r} must return a 1-D (num_envs,) tensor, "
            f"got shape {tuple(value.shape)}"
        )
    if not value.is_floating_point():
        raise TaskContractError(
            f"reward term {term.name!r} must return a floating tensor, got {value.dtype}"
        )
    return value


def _termination_mask(term: TerminationTerm, transition: Transition) -> torch.Tensor:
    value = term.fn(transition)
    if not isinstance(value, torch.Tensor):
        raise TaskContractError(
            f"termination term {term.name!r} must return a torch.Tensor, "
            f"got {type(value).__name__}"
        )
    if value.ndim != 1:
        raise TaskContractError(
            f"termination term {term.name!r} must return a 1-D (num_envs,) bool tensor, "
            f"got shape {tuple(value.shape)}"
        )
    if value.dtype is not torch.bool:
        raise TaskContractError(
            f"termination term {term.name!r} must return bool, got {value.dtype}"
        )
    return value


class RewardManager:
    """Sum weighted reward terms, with an optional clip on the total.

    ``terms`` must be a non-empty sequence of :class:`RewardTerm` with unique
    names.  Disabled terms are not evaluated at all.  At least one term must be
    enabled.  Construction validates weights and ``clip``; each call validates
    the term outputs.  The manager holds no episode or step state.
    """

    def __init__(
        self,
        terms: Sequence[RewardTerm],
        *,
        clip: tuple[float, float] | None = None,
    ) -> None:
        resolved = _require_terms(terms, "RewardManager", RewardTerm)
        for term in resolved:
            _require_name(term.name, "RewardTerm")
            _require_callable(term.fn, term.name, "RewardTerm")
            _require_weight(term.weight, term.name)
        names = [term.name for term in resolved]
        _require_unique_names(names, "RewardManager")
        if not any(term.enabled for term in resolved):
            raise TaskConfigError(
                "RewardManager requires at least one enabled term; all declared terms "
                "are disabled"
            )
        self.terms = resolved
        self.clip = _require_clip(clip)

    @property
    def names(self) -> tuple[str, ...]:
        """Every declared term name, in declaration order."""
        return tuple(term.name for term in self.terms)

    def __call__(self, transition: Transition) -> torch.Tensor:
        """Return the optional-clipped total of the enabled weighted terms."""
        total: torch.Tensor | None = None
        for term in self.terms:
            if not term.enabled:
                continue
            value = _reward_value(term, transition)
            if total is None:
                total = term.weight * value
                continue
            if value.shape != total.shape:
                raise TaskContractError(
                    f"reward term {term.name!r} returned shape {tuple(value.shape)}, "
                    f"expected {tuple(total.shape)} to match the other terms"
                )
            total = total + term.weight * value
        assert total is not None  # guaranteed by construction
        if self.clip is not None:
            total = torch.clamp(total, min=self.clip[0], max=self.clip[1])
        return total

    def breakdown(self, transition: Transition) -> dict[str, torch.Tensor]:
        """Return the weighted tensor of every enabled term, keyed by name."""
        result: dict[str, torch.Tensor] = {}
        shape: tuple[int, ...] | None = None
        for term in self.terms:
            if not term.enabled:
                continue
            value = _reward_value(term, transition)
            if shape is None:
                shape = tuple(value.shape)
            elif tuple(value.shape) != shape:
                raise TaskContractError(
                    f"reward term {term.name!r} returned shape {tuple(value.shape)}, "
                    f"expected {shape} to match the other terms"
                )
            result[term.name] = term.weight * value
        return result


class TerminationManager:
    """OR termination terms into ``terminated`` / ``truncated`` masks.

    ``terms`` must be a non-empty sequence of :class:`TerminationTerm` with
    unique names.  Every term's raw mask is reported in
    :attr:`TerminationResult.reasons`; the manager does not merge core
    episode-clock due ids (``TaskEnv`` still owns that merge).  The manager
    holds no episode or step state.
    """

    def __init__(self, terms: Sequence[TerminationTerm]) -> None:
        resolved = _require_terms(terms, "TerminationManager", TerminationTerm)
        for term in resolved:
            _require_name(term.name, "TerminationTerm")
            _require_callable(term.fn, term.name, "TerminationTerm")
            _require_mode(term.mode, term.name)
        names = [term.name for term in resolved]
        _require_unique_names(names, "TerminationManager")
        self.terms = resolved

    @property
    def names(self) -> tuple[str, ...]:
        """Every declared term name, in declaration order."""
        return tuple(term.name for term in self.terms)

    def __call__(self, transition: Transition) -> TerminationResult:
        """Return per-mode OR masks plus every raw per-term reason mask."""
        terminated: torch.Tensor | None = None
        truncated: torch.Tensor | None = None
        reasons: dict[str, torch.Tensor] = {}
        shape: tuple[int, ...] | None = None
        for term in self.terms:
            mask = _termination_mask(term, transition)
            if shape is None:
                shape = tuple(mask.shape)
            elif tuple(mask.shape) != shape:
                raise TaskContractError(
                    f"termination term {term.name!r} returned shape {tuple(mask.shape)}, "
                    f"expected {shape} to match the other terms"
                )
            reasons[term.name] = mask
            if term.mode == "terminated":
                terminated = mask if terminated is None else terminated | mask
            else:
                truncated = mask if truncated is None else truncated | mask
        assert terminated is not None or truncated is not None  # non-empty terms
        if terminated is None:
            terminated = torch.zeros_like(truncated)
        if truncated is None:
            truncated = torch.zeros_like(terminated)
        return TerminationResult(terminated=terminated, truncated=truncated, reasons=reasons)
