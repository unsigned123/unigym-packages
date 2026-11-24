"""Scenario sampler lifecycle, per-env state, factory and rewind contract."""

from __future__ import annotations

import copy
import inspect
import json

import pytest
import torch
from unigym.commands import CommandSampler

from unigym_bench.engine import StreamingEngine
from unigym_bench.sampler import (
    SAMPLER_REWIND_SCHEMA,
    ScenarioSampler,
    make_sampler,
)
from unigym_bench.specs import ScenarioSpec, SpecError
from _support import SamplerEnv, SamplerSignals


def _spec(*, count: int = 10, plan: list[int] | None = None) -> ScenarioSpec:
    pool = [[float(index), float(index) + 0.5] for index in range(count)]
    return ScenarioSpec(command_pool=pool, plan=plan)


def _sampler(
    *,
    num_envs: int = 1,
    seed: int = 0,
    dwell: int = 1,
    spec: ScenarioSpec | None = None,
) -> ScenarioSampler:
    sampler = make_sampler(
        dim=2,
        device="cpu",
        num_envs=num_envs,
        scenario=_spec() if spec is None else spec,
        seed=seed,
        dwell=dwell,
    )
    assert isinstance(sampler, ScenarioSampler)
    return sampler


def _command(sampler: ScenarioSampler, env_id: int = 0) -> tuple[float, ...]:
    return tuple(float(value) for value in sampler.cmd[env_id].tolist())


def _expected_command(sampler: ScenarioSampler, cursor: int) -> tuple[float, ...]:
    index = sampler.scenario.order[cursor]
    return sampler.scenario.commands[index]


def test_factory_signature_is_the_contract_keyword_only_shape() -> None:
    signature = inspect.signature(make_sampler)

    assert list(signature.parameters) == ["dim", "device", "num_envs", "params"]
    for name in ("dim", "device", "num_envs"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["params"].kind is inspect.Parameter.VAR_KEYWORD


def test_factory_returns_a_float32_command_buffer() -> None:
    sampler = _sampler(num_envs=3)

    assert isinstance(sampler, CommandSampler)
    assert sampler.cmd.dtype is torch.float32
    assert tuple(sampler.cmd.shape) == (3, 2)
    assert sampler.rewind_capable is True
    assert sampler.rewind_inert is False


def test_factory_rejects_unknown_parameters() -> None:
    with pytest.raises(TypeError) as excinfo:
        make_sampler(
            dim=2,
            device="cpu",
            num_envs=1,
            scenario=_spec(),
            seed=0,
            surprise=1,
        )

    assert "surprise" in str(excinfo.value)


def test_factory_requires_scenario_and_seed() -> None:
    with pytest.raises(TypeError) as excinfo:
        make_sampler(dim=2, device="cpu", num_envs=1, seed=0)
    assert "scenario" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        make_sampler(dim=2, device="cpu", num_envs=1, scenario=_spec())
    assert "seed" in str(excinfo.value)


def test_factory_validates_dim_against_the_scenario() -> None:
    with pytest.raises(SpecError) as excinfo:
        make_sampler(
            dim=3,
            device="cpu",
            num_envs=1,
            scenario=_spec(),
            seed=0,
        )

    assert "scenario width 2" in str(excinfo.value)


def test_factory_seed_is_explicit_and_does_not_use_ambient_rng() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        first = _sampler(num_envs=4, seed=7)
        torch.manual_seed(999)
        second = _sampler(num_envs=4, seed=7)

    assert torch.equal(first.cmd, second.cmd)


def test_reset_applies_the_current_command_and_update_is_a_no_op() -> None:
    sampler = _sampler(num_envs=2, seed=4)
    before = sampler.cmd.clone()

    returned = sampler.update(0.02, torch.zeros_like(sampler.cmd))
    assert returned is sampler.cmd
    assert torch.equal(sampler.cmd, before)

    sampler.reset(torch.tensor([1], dtype=torch.int64))
    assert torch.equal(sampler.cmd[0], before[0])
    assert torch.equal(sampler.cmd[1], before[1])


def test_dwell_one_advances_the_cursor_at_each_episode_boundary() -> None:
    sampler = _sampler(seed=11)
    scenario = sampler.scenario
    state = sampler.rewind_state()

    sampler.on_episode_end(torch.tensor([0], dtype=torch.int64), {})
    sampler.reset(torch.tensor([0], dtype=torch.int64))

    next_cursor = (state["cursor"][0] + 1) % len(scenario.order)
    assert sampler.rewind_state()["cursor"] == [next_cursor]
    assert _command(sampler) == _expected_command(sampler, next_cursor)


def test_dwell_two_repeats_a_command_before_advancing() -> None:
    sampler = _sampler(seed=5, dwell=2)
    scenario = sampler.scenario
    state = sampler.rewind_state()
    first_command = _command(sampler)

    sampler.on_episode_end(torch.tensor([0], dtype=torch.int64), {})
    sampler.reset(torch.tensor([0], dtype=torch.int64))

    assert sampler.rewind_state()["cursor"] == state["cursor"]
    assert _command(sampler) == first_command

    sampler.on_episode_end(torch.tensor([0], dtype=torch.int64), {})
    sampler.reset(torch.tensor([0], dtype=torch.int64))

    next_cursor = (state["cursor"][0] + 1) % len(scenario.order)
    assert sampler.rewind_state()["cursor"] == [next_cursor]
    assert _command(sampler) == _expected_command(sampler, next_cursor)


def test_ten_distinct_commands_are_observed_across_ten_episodes() -> None:
    sampler = _sampler(seed=3)
    scenario = sampler.scenario
    seen: list[tuple[float, ...]] = []

    for _ in range(10):
        seen.append(_command(sampler))
        sampler.on_episode_end(torch.tensor([0], dtype=torch.int64), {})
        sampler.reset(torch.tensor([0], dtype=torch.int64))

    assert len(set(seen)) == 10
    assert set(seen) == set(scenario.commands)


def test_per_env_cursor_and_dwell_are_independent_for_num_envs_above_one() -> None:
    sampler = _sampler(num_envs=3, seed=9, dwell=2)
    scenario = sampler.scenario
    state = sampler.rewind_state()
    before = sampler.cmd.clone()

    sampler.on_episode_end(torch.tensor([1], dtype=torch.int64), {})
    sampler.reset(torch.tensor([1], dtype=torch.int64))

    after = sampler.rewind_state()["cursor"]
    assert after[0] == state["cursor"][0]
    assert after[1] == state["cursor"][1]
    assert after[2] == state["cursor"][2]
    assert torch.equal(sampler.cmd[0], before[0])
    assert torch.equal(sampler.cmd[1], before[1])
    assert torch.equal(sampler.cmd[2], before[2])

    sampler.on_episode_end(torch.tensor([1], dtype=torch.int64), {})
    sampler.reset(torch.tensor([1], dtype=torch.int64))

    advanced = sampler.rewind_state()["cursor"]
    assert advanced[0] == state["cursor"][0]
    assert advanced[1] == (state["cursor"][1] + 1) % len(scenario.order)
    assert advanced[2] == state["cursor"][2]
    assert _command(sampler, 1) == _expected_command(sampler, advanced[1])


def test_plan_order_is_used_for_advancing() -> None:
    sampler = _sampler(spec=_spec(plan=[3, 1, 0]), seed=2)
    state = sampler.rewind_state()

    sampler.on_episode_end(torch.tensor([0], dtype=torch.int64), {})
    sampler.reset(torch.tensor([0], dtype=torch.int64))

    next_cursor = (state["cursor"][0] + 1) % 3
    assert sampler.rewind_state()["cursor"] == [next_cursor]
    assert _command(sampler) == _expected_command(sampler, next_cursor)


def test_on_episode_end_ignores_results_and_validates_bound_env() -> None:
    sampler = _sampler(num_envs=2, seed=1)

    sampler.on_episode_end(torch.tensor([0], dtype=torch.int64), {5: ()})

    sampler.bind_env(SamplerEnv(num_envs=2))
    sampler.bind_signals(SamplerSignals())

    with pytest.raises(ValueError):
        sampler.bind_env(SamplerEnv(num_envs=1))


def test_rewind_roundtrip_restores_cursor_dwell_rng_and_commands() -> None:
    sampler = _sampler(num_envs=2, seed=13, dwell=2)
    snapshot = sampler.rewind_state()
    assert snapshot["schema"] == SAMPLER_REWIND_SCHEMA
    assert "rng" in snapshot
    assert "cursor" in snapshot
    assert "cmd" in snapshot

    sampler.on_episode_end(torch.tensor([0, 1], dtype=torch.int64), {})
    sampler.reset(torch.tensor([0, 1], dtype=torch.int64))
    sampler.on_episode_end(torch.tensor([0], dtype=torch.int64), {})
    sampler.reset(torch.tensor([0], dtype=torch.int64))
    assert sampler.rewind_state() != snapshot

    sampler.apply_rewind_state(snapshot)

    assert sampler.rewind_state() == snapshot
    assert _command(sampler, 0) == tuple(snapshot["cmd"][0])
    assert _command(sampler, 1) == tuple(snapshot["cmd"][1])


def test_rewind_state_is_a_private_json_safe_copy() -> None:
    sampler = _sampler(num_envs=2, seed=15)
    snapshot = sampler.rewind_state()

    assert json.loads(json.dumps(snapshot)) == snapshot
    pristine = copy.deepcopy(snapshot)

    snapshot["cursor"][0] = 999
    assert sampler.rewind_state()["cursor"][0] != 999

    sampler.apply_rewind_state(pristine)
    pristine["cursor"][0] = 42
    assert sampler.rewind_state()["cursor"][0] != 42


def test_invalid_rewind_state_leaves_the_sampler_unchanged() -> None:
    sampler = _sampler(num_envs=2, seed=17)
    snapshot = sampler.rewind_state()

    invalid = copy.deepcopy(snapshot)
    invalid["cursor"][0] = 999
    with pytest.raises(ValueError):
        sampler.apply_rewind_state(invalid)
    assert sampler.rewind_state() == snapshot

    invalid = copy.deepcopy(snapshot)
    invalid["dim"] = 3
    with pytest.raises(ValueError):
        sampler.apply_rewind_state(invalid)
    assert sampler.rewind_state() == snapshot


def test_wrong_length_rng_state_is_rejected_without_partial_mutation() -> None:
    sampler = _sampler(num_envs=2, seed=19)
    snapshot = sampler.rewind_state()
    sampler.on_episode_end(torch.tensor([0, 1], dtype=torch.int64), {})
    sampler.reset(torch.tensor([0, 1], dtype=torch.int64))
    current = sampler.rewind_state()
    assert current != snapshot

    invalid = copy.deepcopy(current)
    invalid["rng"] = invalid["rng"][:16]
    with pytest.raises(ValueError) as excinfo:
        sampler.apply_rewind_state(invalid)

    assert "rng" in str(excinfo.value)
    assert sampler.rewind_state() == current


def test_pack_rewind_provider_contains_no_sampler_state() -> None:
    engine = StreamingEngine([])

    state = engine.rewind_state()

    assert set(state) == {"schema", "engine", "monitors"}
    document = json.dumps(state)
    assert "sampler" not in document
    assert "cursor" not in document
    assert '"rng"' not in document
