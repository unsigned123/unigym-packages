"""Contract tests for the optional reward / termination managers (R3)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from unigym_rl import (
    RewardManager,
    RewardTerm,
    TaskConfigError,
    TaskContractError,
    TaskState,
    TerminationManager,
    TerminationResult,
    TerminationTerm,
    Transition,
)
from unigym_rl.testing.fakes import make_fake_obs_registry


def _transition(rows: tuple[float, ...]) -> Transition:
    num_envs = len(rows)
    values = torch.tensor(rows, dtype=torch.float32).unsqueeze(1)
    obs = torch.cat([values, torch.zeros(num_envs, 1)], dim=1)
    registry = make_fake_obs_registry(num_envs=num_envs)
    state = TaskState(t=0, num_envs=num_envs, obs=registry.snapshot())
    mask = torch.zeros(num_envs, dtype=torch.bool)
    return Transition(
        obs=obs,
        action=obs,
        next_obs=obs,
        terminated=mask,
        truncated=mask.clone(),
        info=state,
    )


def _row_reward(transition: Transition) -> torch.Tensor:
    return transition.obs[:, 0]


def _constant(value: float):
    def fn(transition: Transition) -> torch.Tensor:
        return torch.full((transition.info.num_envs,), value)

    return fn


def _mask(pattern: tuple[bool, ...], dtype: torch.dtype = torch.bool):
    def fn(transition: Transition) -> torch.Tensor:
        del transition
        return torch.tensor(pattern, dtype=dtype)

    return fn


# --------------------------------------------------------------------------- #
# RewardManager
# --------------------------------------------------------------------------- #
def test_reward_sums_weighted_terms() -> None:
    manager = RewardManager(
        (
            RewardTerm("row", _row_reward, weight=2.0),
            RewardTerm("bias", _constant(1.0), weight=3.0),
        )
    )

    total = manager(_transition((0.5, 1.0, -1.5)))

    assert manager.names == ("row", "bias")
    assert torch.allclose(total, torch.tensor([4.0, 5.0, -0.0]))


def test_reward_disabled_terms_are_skipped() -> None:
    manager = RewardManager(
        (
            RewardTerm("row", _row_reward),
            RewardTerm("bias", _constant(100.0), enabled=False),
        )
    )

    total = manager(_transition((1.0, 2.0)))
    breakdown = manager.breakdown(_transition((1.0, 2.0)))

    assert torch.allclose(total, torch.tensor([1.0, 2.0]))
    assert set(breakdown) == {"row"}
    assert manager.names == ("row", "bias")


def test_reward_breakdown_returns_weighted_terms() -> None:
    manager = RewardManager(
        (
            RewardTerm("row", _row_reward, weight=2.0),
            RewardTerm("bias", _constant(1.0), weight=-0.5),
        )
    )

    breakdown = manager.breakdown(_transition((1.0, 3.0)))

    assert breakdown["row"].tolist() == [2.0, 6.0]
    assert breakdown["bias"].tolist() == [-0.5, -0.5]


def test_reward_clip_applies_to_total_only() -> None:
    manager = RewardManager(
        (RewardTerm("bias", _constant(5.0), weight=2.0),),
        clip=(-1.0, 1.0),
    )

    total = manager(_transition((0.0, 0.0)))
    breakdown = manager.breakdown(_transition((0.0, 0.0)))

    assert total.tolist() == [1.0, 1.0]
    assert breakdown["bias"].tolist() == [10.0, 10.0]


def test_reward_manager_has_no_hidden_state() -> None:
    manager = RewardManager((RewardTerm("row", _row_reward),))
    low = _transition((1.0, 1.0))
    high = _transition((7.0, 8.0))

    assert torch.allclose(manager(low), torch.tensor([1.0, 1.0]))
    assert torch.allclose(manager(high), torch.tensor([7.0, 8.0]))
    assert torch.allclose(manager(low), torch.tensor([1.0, 1.0]))


def test_reward_manager_rejects_empty_or_disabled_terms() -> None:
    with pytest.raises(TaskConfigError):
        RewardManager(())
    with pytest.raises(TaskConfigError):
        RewardManager(terms=[])  # type: ignore[arg-type]
    with pytest.raises(TaskConfigError):
        RewardManager((RewardTerm("row", _row_reward, enabled=False),))


def test_reward_manager_rejects_duplicate_names() -> None:
    with pytest.raises(TaskConfigError, match="duplicate"):
        RewardManager((RewardTerm("same", _row_reward), RewardTerm("same", _constant(0.0))))


def test_reward_manager_rejects_bad_terms_weights_and_clip() -> None:
    with pytest.raises(TaskConfigError):
        RewardManager((RewardTerm("row", None),))  # type: ignore[arg-type]
    with pytest.raises(TaskConfigError):
        RewardManager((RewardTerm("", _row_reward),))
    for weight in (float("nan"), float("inf"), True, "1.0"):
        with pytest.raises(TaskConfigError):
            RewardManager((RewardTerm("row", _row_reward, weight=weight),))  # type: ignore[arg-type]
    for clip in ((1.0,), (2.0, 1.0), (0.0, float("inf")), [0.0, 1.0], 1.0):
        with pytest.raises(TaskConfigError):
            RewardManager((RewardTerm("row", _row_reward),), clip=clip)  # type: ignore[arg-type]


def test_reward_manager_validates_term_outputs() -> None:
    manager = RewardManager((RewardTerm("bad", lambda transition: torch.ones(2, 1)),))
    with pytest.raises(TaskContractError, match="1-D"):
        manager(_transition((0.0, 0.0)))

    manager = RewardManager((RewardTerm("bad", _mask((True, False))),))
    with pytest.raises(TaskContractError, match="floating"):
        manager(_transition((0.0, 0.0)))

    manager = RewardManager((RewardTerm("bad", lambda transition: 1.0),))  # type: ignore[arg-type, return-value]
    with pytest.raises(TaskContractError, match="torch.Tensor"):
        manager(_transition((0.0, 0.0)))

    manager = RewardManager(
        (
            RewardTerm("row", _row_reward),
            RewardTerm("bad", lambda transition: torch.ones(1)),
        )
    )
    with pytest.raises(TaskContractError, match="match the other terms"):
        manager(_transition((0.0, 0.0, 0.0)))


# --------------------------------------------------------------------------- #
# TerminationManager
# --------------------------------------------------------------------------- #
def test_termination_modes_and_reasons() -> None:
    manager = TerminationManager(
        (
            TerminationTerm("fell", _mask((True, False, True))),
            TerminationTerm("timeout", _mask((False, True, False)), mode="truncated"),
            TerminationTerm("tilted", _mask((False, True, True))),
        )
    )

    result = manager(_transition((0.0, 0.0, 0.0)))

    assert manager.names == ("fell", "timeout", "tilted")
    assert result.terminated.tolist() == [True, True, True]
    assert result.truncated.tolist() == [False, True, False]
    assert result.reasons["fell"].tolist() == [True, False, True]
    assert result.reasons["timeout"].tolist() == [False, True, False]
    assert result.reasons["tilted"].tolist() == [False, True, True]
    assert set(result.reasons) == set(manager.names)


def test_termination_untouched_mode_is_all_false() -> None:
    manager = TerminationManager(
        (TerminationTerm("timeout", _mask((True, False)), mode="truncated"),)
    )

    result = manager(_transition((0.0, 0.0)))

    assert result.truncated.tolist() == [True, False]
    assert result.terminated.tolist() == [False, False]
    assert result.terminated.dtype is torch.bool
    assert result.terminated.shape == (2,)


def test_termination_manager_has_no_hidden_state() -> None:
    manager = TerminationManager(
        (TerminationTerm("fell", lambda transition: transition.obs[:, 0] > 0.5),)
    )
    low = _transition((0.0, 0.2))
    high = _transition((1.0, 0.9))

    assert manager(low).terminated.tolist() == [False, False]
    assert manager(high).terminated.tolist() == [True, True]
    assert manager(low).terminated.tolist() == [False, False]


def test_termination_manager_rejects_empty_duplicate_and_bad_terms() -> None:
    with pytest.raises(TaskConfigError):
        TerminationManager(())
    with pytest.raises(TaskConfigError, match="duplicate"):
        TerminationManager(
            (
                TerminationTerm("same", _mask((True,))),
                TerminationTerm("same", _mask((False,))),
            )
        )
    with pytest.raises(TaskConfigError, match="callable"):
        TerminationManager((TerminationTerm("bad", None),))  # type: ignore[arg-type]
    with pytest.raises(TaskConfigError, match="mode"):
        TerminationManager((TerminationTerm("bad", _mask((True,)), mode="done"),))  # type: ignore[arg-type]


def test_termination_manager_validates_term_outputs() -> None:
    manager = TerminationManager((TerminationTerm("bad", _mask((1, 0), dtype=torch.int64)),))
    with pytest.raises(TaskContractError, match="bool"):
        manager(_transition((0.0, 0.0)))

    manager = TerminationManager((TerminationTerm("bad", lambda transition: torch.ones(2, 1)),))
    with pytest.raises(TaskContractError, match="1-D"):
        manager(_transition((0.0, 0.0)))

    manager = TerminationManager((TerminationTerm("bad", lambda transition: True),))  # type: ignore[arg-type, return-value]
    with pytest.raises(TaskContractError, match="torch.Tensor"):
        manager(_transition((0.0, 0.0)))

    manager = TerminationManager(
        (
            TerminationTerm("ok", _mask((True, False))),
            TerminationTerm("bad", _mask((True, False, True))),
        )
    )
    with pytest.raises(TaskContractError, match="match the other terms"):
        manager(_transition((0.0, 0.0, 0.0)))


def test_termination_result_is_frozen() -> None:
    result = TerminationResult(
        terminated=torch.zeros(1, dtype=torch.bool),
        truncated=torch.zeros(1, dtype=torch.bool),
        reasons={},
    )
    with pytest.raises(FrozenInstanceError):
        result.terminated = torch.ones(1, dtype=torch.bool)  # type: ignore[misc]
