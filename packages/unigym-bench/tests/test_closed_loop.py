"""Closed-loop convention: engine publication, sampler reads and latency."""

from __future__ import annotations

import pytest
import torch
from unigym.values import ValueStore

from unigym_bench.engine import StreamingEngine
from unigym_bench.sampler import (
    COMMAND_VALID_SIGNAL,
    DESIRED_CMD_SIGNAL,
    ScenarioSampler,
    make_sampler,
)
from unigym_bench.specs import ScenarioSpec
from _support import SamplerValues, ScriptedMonitor, TickScript, make_bundle


def _sampler(*, num_envs: int = 2, seed: int = 0) -> ScenarioSampler:
    spec = ScenarioSpec(command_pool=[[float(index), float(index) + 0.5] for index in range(10)])
    sampler = make_sampler(dim=2, device="cpu", num_envs=num_envs, scenario=spec, seed=seed)
    assert isinstance(sampler, ScenarioSampler)
    return sampler


def _desired(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


def _scenario_monitor(script: dict[int, TickScript]) -> ScriptedMonitor:
    return ScriptedMonitor("scenario", script)


def test_monitor_desired_command_is_applied_at_the_next_update() -> None:
    desired = _desired([[9.0, -1.0], [2.5, 3.5]])
    monitor = _scenario_monitor({0: TickScript(values={"desired_cmd": desired})})
    store = ValueStore(num_envs=2, device="cpu")
    engine = StreamingEngine([monitor], publisher=store.publisher("sink"))
    sampler = _sampler()
    sampler.bind_values(store)
    before = sampler.cmd.clone()

    engine.on_step(make_bundle(num_envs=2, t=0))
    assert torch.equal(sampler.cmd, before)

    returned = sampler.update(0.02, sampler.cmd)

    assert returned is sampler.cmd
    assert torch.equal(sampler.cmd, desired)


def test_applying_a_desired_command_leaves_cursor_dwell_and_rng_untouched() -> None:
    desired = _desired([[9.0, -1.0], [2.5, 3.5]])
    monitor = _scenario_monitor({0: TickScript(values={"desired_cmd": desired})})
    store = ValueStore(num_envs=2, device="cpu")
    engine = StreamingEngine([monitor], publisher=store.publisher("sink"))
    sampler = _sampler()
    sampler.bind_values(store)
    before = sampler.rewind_state()

    engine.on_step(make_bundle(num_envs=2, t=0))
    sampler.update(0.02, sampler.cmd)

    after = sampler.rewind_state()
    assert after["cmd"] != before["cmd"]
    assert after["cursor"] == before["cursor"]
    assert after["remaining"] == before["remaining"]
    assert after["rng"] == before["rng"]


def test_command_valid_false_holds_the_previous_command() -> None:
    first = _desired([[7.0, 7.5], [8.0, 8.5]])
    second = _desired([[-1.0, -2.0], [-3.0, -4.0]])
    third = _desired([[4.0, 4.5], [5.0, 5.5]])
    monitor = _scenario_monitor(
        {
            0: TickScript(values={"desired_cmd": first}),
            1: TickScript(
                values={
                    "desired_cmd": second,
                    "command_valid": torch.tensor([False, False]),
                }
            ),
            2: TickScript(
                values={
                    "desired_cmd": second,
                    "command_valid": torch.tensor([True, True]),
                }
            ),
            3: TickScript(
                values={
                    "desired_cmd": third,
                    "command_valid": torch.tensor([True, False]),
                }
            ),
        }
    )
    store = ValueStore(num_envs=2, device="cpu")
    engine = StreamingEngine([monitor], publisher=store.publisher("sink"))
    sampler = _sampler()
    sampler.bind_values(store)

    engine.on_step(make_bundle(num_envs=2, t=0))
    sampler.update(0.02, sampler.cmd)
    assert torch.equal(sampler.cmd, first)

    engine.on_step(make_bundle(num_envs=2, t=1))
    sampler.update(0.02, sampler.cmd)
    assert torch.equal(sampler.cmd, first)

    engine.on_step(make_bundle(num_envs=2, t=2))
    sampler.update(0.02, sampler.cmd)
    assert torch.equal(sampler.cmd, second)

    engine.on_step(make_bundle(num_envs=2, t=3))
    sampler.update(0.02, sampler.cmd)
    assert torch.equal(sampler.cmd[0], third[0])
    assert torch.equal(sampler.cmd[1], second[1])


def test_desired_command_shape_is_validated() -> None:
    view = SamplerValues({DESIRED_CMD_SIGNAL: torch.zeros(2, 3, dtype=torch.float32)})
    sampler = _sampler()
    sampler.bind_values(view)

    with pytest.raises(ValueError, match=r"expected shape \(2, 2\), got \(2, 3\)"):
        sampler.update(0.02, sampler.cmd)


def test_desired_command_type_is_validated() -> None:
    view = SamplerValues({DESIRED_CMD_SIGNAL: [[0.0, 0.0], [0.0, 0.0]]})
    sampler = _sampler()
    sampler.bind_values(view)

    with pytest.raises(TypeError, match="expected a torch.Tensor"):
        sampler.update(0.02, sampler.cmd)


def test_desired_command_dtype_is_validated() -> None:
    view = SamplerValues({DESIRED_CMD_SIGNAL: torch.zeros(2, 2, dtype=torch.int64)})
    sampler = _sampler()
    sampler.bind_values(view)

    with pytest.raises(TypeError, match="expected dtype torch.float32"):
        sampler.update(0.02, sampler.cmd)


def test_desired_command_device_is_validated() -> None:
    view = SamplerValues(
        {DESIRED_CMD_SIGNAL: torch.empty(2, 2, dtype=torch.float32, device="meta")}
    )
    sampler = _sampler()
    sampler.bind_values(view)

    with pytest.raises(ValueError, match="expected device cpu"):
        sampler.update(0.02, sampler.cmd)


def test_desired_command_must_be_finite() -> None:
    view = SamplerValues({DESIRED_CMD_SIGNAL: torch.tensor([[float("nan"), 0.0], [0.0, 0.0]])})
    sampler = _sampler()
    sampler.bind_values(view)

    with pytest.raises(ValueError, match="finite"):
        sampler.update(0.02, sampler.cmd)


def test_command_valid_shape_and_dtype_are_validated() -> None:
    desired = _desired([[1.0, 2.0], [3.0, 4.0]])
    view = SamplerValues(
        {DESIRED_CMD_SIGNAL: desired, COMMAND_VALID_SIGNAL: torch.zeros(2, 2, dtype=torch.bool)}
    )
    sampler = _sampler()
    sampler.bind_values(view)

    with pytest.raises(ValueError, match=r"expected shape \(2,\), got \(2, 2\)"):
        sampler.update(0.02, sampler.cmd)

    view.values[COMMAND_VALID_SIGNAL] = torch.zeros(2, dtype=torch.int64)
    with pytest.raises(TypeError, match="expected dtype torch.bool"):
        sampler.update(0.02, sampler.cmd)


def test_closed_loop_has_two_policy_tick_latency() -> None:
    commands = {
        0: _desired([[1.0, 2.0], [3.0, 4.0]]),
        1: _desired([[5.0, 6.0], [7.0, 8.0]]),
        2: _desired([[9.0, 10.0], [11.0, 12.0]]),
    }
    monitor = _scenario_monitor(
        {tick: TickScript(values={"desired_cmd": value}) for tick, value in commands.items()}
    )
    store = ValueStore(num_envs=2, device="cpu")
    engine = StreamingEngine([monitor], publisher=store.publisher("sink"))
    sampler = _sampler()
    sampler.bind_values(store)
    pool = sampler.cmd.clone()

    used: dict[int, torch.Tensor] = {}
    returned: dict[int, torch.Tensor] = {}
    for tick in range(3):
        used[tick] = sampler.cmd.clone()
        returned[tick] = sampler.update(0.02, sampler.cmd).clone()
        engine.on_step(make_bundle(num_envs=2, t=tick))

    assert torch.equal(used[0], pool)
    assert torch.equal(used[1], pool)
    assert torch.equal(returned[1], commands[0])
    assert torch.equal(used[2], commands[0])
    assert not torch.equal(used[1], commands[0])
    assert torch.equal(returned[0], pool)


def test_rewind_restores_sampler_buffer_and_values_store_together() -> None:
    first = _desired([[1.0, 2.0], [3.0, 4.0]])
    second = _desired([[5.0, 6.0], [7.0, 8.0]])
    monitor = _scenario_monitor(
        {
            0: TickScript(values={"desired_cmd": first}),
            1: TickScript(values={"desired_cmd": second}),
        }
    )
    store = ValueStore(num_envs=2, device="cpu")
    engine = StreamingEngine([monitor], publisher=store.publisher("sink"))
    sampler = _sampler()
    sampler.bind_values(store)

    engine.on_step(make_bundle(num_envs=2, t=0))
    sampler.update(0.02, sampler.cmd)
    assert torch.equal(sampler.cmd, first)
    sampler_state = sampler.rewind_state()
    values_state = store.rewind_state()

    engine.on_step(make_bundle(num_envs=2, t=1))
    sampler.update(0.02, sampler.cmd)
    sampler.on_episode_end(torch.tensor([0], dtype=torch.int64), {})
    sampler.reset(torch.tensor([0], dtype=torch.int64))
    assert sampler.rewind_state() != sampler_state
    assert not torch.equal(store.snapshot()[DESIRED_CMD_SIGNAL], first)

    store.apply_rewind_state(values_state)
    sampler.apply_rewind_state(sampler_state)

    assert sampler.rewind_state() == sampler_state
    restored = store.snapshot()
    assert set(restored) == set(values_state["values"])
    assert all(torch.equal(restored[name], value) for name, value in values_state["values"].items())
    assert torch.equal(sampler.update(0.02, sampler.cmd), first)


def test_update_reads_only_the_documented_closed_loop_values() -> None:
    view = SamplerValues(record=True)
    sampler = _sampler()
    sampler.bind_values(view)
    before = sampler.cmd.clone()

    sampler.update(0.02, sampler.cmd)

    assert torch.equal(sampler.cmd, before)
    assert view.requests == [DESIRED_CMD_SIGNAL]
    assert view.snapshot_calls == 0


def test_foreign_monitor_command_named_value_is_ignored() -> None:
    foreign = _desired([[42.0, 43.0], [44.0, 45.0]])
    monitor = ScriptedMonitor("contact", {0: TickScript(values={"desired_cmd": foreign})})
    store = ValueStore(num_envs=2, device="cpu")
    engine = StreamingEngine([monitor], publisher=store.publisher("sink"))
    sampler = _sampler()
    sampler.bind_values(store)
    before = sampler.cmd.clone()

    engine.on_step(make_bundle(num_envs=2, t=0))
    sampler.update(0.02, sampler.cmd)

    assert torch.equal(sampler.cmd, before)
    assert "contact/desired_cmd" in store.snapshot()
