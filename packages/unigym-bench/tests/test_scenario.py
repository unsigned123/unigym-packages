"""Scenario quantization, distinctness, plan validation and command grouping."""

from __future__ import annotations

import struct

import pytest

from unigym_bench.events import EpisodeRecord
from unigym_bench.scenario import Scenario, command_key, group_records_by_command
from unigym_bench.specs import ScenarioSpec, SpecError


def _float32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", value))[0]


def _pool(count: int = 10) -> list[list[float]]:
    return [[float(index), float(index) + 0.5] for index in range(count)]


def _record(
    *,
    env_id: int = 0,
    episode_seq: int = 0,
    command: tuple[float, ...] | None = (0.0, 0.5),
) -> EpisodeRecord:
    return EpisodeRecord(
        env_id=env_id,
        episode_seq=episode_seq,
        ticks=1,
        verdict="incomplete",
        command_at_start=command,
    )


def test_scenario_quantizes_commands_to_float32() -> None:
    spec = ScenarioSpec(command_pool=[[0.1, 1.0 / 3.0]])

    scenario = Scenario.from_spec(spec)

    assert scenario.commands == ((_float32(0.1), _float32(1.0 / 3.0)),)
    assert scenario.dim == 2
    assert scenario.order == (0,)


def test_scenario_default_order_is_the_pool_order() -> None:
    scenario = Scenario(commands=tuple(tuple(row) for row in _pool()))

    assert scenario.order == tuple(range(10))
    assert len(scenario.commands) == 10


def test_scenario_uses_an_explicit_plan_order() -> None:
    spec = ScenarioSpec(command_pool=_pool(), plan=[3, 1, 0])

    scenario = Scenario.from_spec(spec)

    assert scenario.order == (3, 1, 0)


def test_ten_distinct_commands_cover_all_pool_entries() -> None:
    scenario = Scenario.from_spec(ScenarioSpec(command_pool=_pool()))

    assert len(set(scenario.commands)) == 10
    assert {scenario.commands[index] for index in scenario.order} == set(scenario.commands)


def test_duplicate_commands_fail_naming_both_indices() -> None:
    spec = ScenarioSpec(command_pool=[[0.0, 1.0], [2.0, 3.0], [0.0, 1.0]])

    with pytest.raises(SpecError) as excinfo:
        Scenario.from_spec(spec)

    message = str(excinfo.value)
    assert "indices 0 and 2" in message


def test_commands_duplicate_after_float32_quantization_fail() -> None:
    spec = ScenarioSpec(command_pool=[[0.0], [1e-50]])

    with pytest.raises(SpecError) as excinfo:
        Scenario.from_spec(spec)

    assert "duplicate" in str(excinfo.value)


def test_non_float32_representable_values_fail() -> None:
    spec = ScenarioSpec(command_pool=[[1e300]])

    with pytest.raises(SpecError) as excinfo:
        Scenario.from_spec(spec)

    assert "float32" in str(excinfo.value)


def test_dimension_mismatch_fails_in_the_authoring_spec() -> None:
    with pytest.raises(SpecError) as excinfo:
        ScenarioSpec(command_pool=[[0.0, 1.0], [2.0]])

    assert "expected 2 values" in str(excinfo.value)


def test_dimension_mismatch_fails_for_a_direct_scenario() -> None:
    with pytest.raises(SpecError) as excinfo:
        Scenario(commands=((0.0, 1.0), (2.0,)))

    assert "expected 2 values" in str(excinfo.value)


def test_plan_index_out_of_range_fails() -> None:
    with pytest.raises(SpecError) as excinfo:
        ScenarioSpec(command_pool=_pool(2), plan=[0, 2])

    assert "[0, 2)" in str(excinfo.value)


def test_command_key_is_float32_exact_and_normalizes_negative_zero() -> None:
    assert command_key([0.1, 1.0 / 3.0]) == (_float32(0.1), _float32(1.0 / 3.0))
    assert command_key([-0.0, 0.0]) == (0.0, 0.0)


def test_group_records_by_observed_start_command_not_by_episode_id() -> None:
    first = _record(env_id=0, episode_seq=7, command=(0.0, 0.5))
    second = _record(env_id=1, episode_seq=7, command=(0.0, 0.5))
    third = _record(env_id=0, episode_seq=8, command=(1.0, 1.5))

    groups = group_records_by_command([first, second, third])

    assert list(groups) == [(0.0, 0.5), (1.0, 1.5)]
    assert groups[(0.0, 0.5)] == (first, second)
    assert groups[(1.0, 1.5)] == (third,)


def test_grouping_collapses_observed_values_with_the_same_float32_key() -> None:
    exact = _float32(0.1)
    first = _record(env_id=0, command=(exact,))
    second = _record(env_id=1, command=(exact + 1e-10,))

    groups = group_records_by_command([first, second])

    assert list(groups) == [(exact,)]
    assert groups[(exact,)] == (first, second)


def test_grouping_requires_an_observed_start_command() -> None:
    with pytest.raises(SpecError) as excinfo:
        group_records_by_command([_record(command=None)])

    assert "no command_at_start" in str(excinfo.value)


def test_grouping_rejects_non_records() -> None:
    with pytest.raises(TypeError) as excinfo:
        group_records_by_command([object()])

    assert "EpisodeRecord" in str(excinfo.value)
