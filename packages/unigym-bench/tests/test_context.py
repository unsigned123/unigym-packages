"""Benchmark step context: construction, validation and engine adoption."""

from __future__ import annotations

import dataclasses
import importlib.util
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

import unigym_bench
from unigym.core.episode_termination import (
    EpisodeOutcome,
    Termination,
    TerminationSource,
)
from unigym.evaluation import EnvStepContext, EnvView
from unigym.values import StepValues
from unigym_bench.context import BenchmarkStepContext
from unigym_bench.engine import EngineStateError, StreamingEngine
from unigym_bench.events import Event


def _view(num_envs: int = 2) -> EnvView:
    return EnvView(record={}, obs=None, env_id=None, num_envs=num_envs, device="cpu")


def _context(
    num_envs: int = 2,
    *,
    events: object = (),
    terminals: object = None,
) -> BenchmarkStepContext:
    return BenchmarkStepContext(
        view=_view(num_envs),
        values=StepValues({}),
        t=0,
        step_index=torch.zeros(num_envs, dtype=torch.int64),
        time_s=None,
        cmd=None,
        reset_state={},
        events=events,  # type: ignore[arg-type]
        terminals={} if terminals is None else terminals,  # type: ignore[arg-type]
    )


def _event(env_id: int = 0) -> Event:
    return Event(name="hit", code="task.contact.hit", env_ids=(env_id,))


def _termination() -> Termination:
    return Termination(
        outcome=EpisodeOutcome.SUCCESS,
        code="custom.test.reason",
        source=TerminationSource.CUSTOM,
    )


def test_subclass_construction_and_defaults() -> None:
    context = _context()

    assert isinstance(context, EnvStepContext)
    assert isinstance(context, BenchmarkStepContext)
    assert context.events == ()
    assert dict(context.terminals) == {}
    assert context.num_envs == 2
    assert context.t == 0


def test_frozen_and_slots() -> None:
    context = _context()

    with pytest.raises(FrozenInstanceError):
        context.t = 1  # type: ignore[misc]
    assert not hasattr(context, "__dict__")
    assert hasattr(BenchmarkStepContext, "__slots__")


def test_events_are_copied_to_an_immutable_tuple() -> None:
    event = _event()
    source = [event]
    context = _context(events=source)

    assert context.events == (event,)
    assert isinstance(context.events, tuple)
    source.append(_event(1))
    assert context.events == (event,)

    with pytest.raises(TypeError, match="events\\[1\\]"):
        _context(events=(event, object()))
    with pytest.raises(TypeError, match="events must be a sequence"):
        _context(events="not events")


def test_terminals_are_copied_to_a_read_only_mapping() -> None:
    termination = _termination()
    source = {0: termination}
    context = _context(terminals=source)

    assert dict(context.terminals) == {0: termination}
    assert context.terminals is not source
    source[1] = termination
    assert dict(context.terminals) == {0: termination}

    with pytest.raises(TypeError):
        context.terminals[1] = termination  # type: ignore[index]


def test_terminals_validate_env_id_range_and_types() -> None:
    termination = _termination()

    with pytest.raises(ValueError, match="outside"):
        _context(terminals={2: termination})
    with pytest.raises(ValueError, match="outside"):
        _context(terminals={-1: termination})
    with pytest.raises(TypeError, match="int env ids"):
        _context(terminals={True: termination})
    with pytest.raises(TypeError, match="int env ids"):
        _context(terminals={"0": termination})
    with pytest.raises(TypeError, match="must be a Termination"):
        _context(terminals={0: object()})
    with pytest.raises(TypeError, match="must be a mapping"):
        _context(terminals=[termination])


def test_replace_preserves_pack_fields() -> None:
    event = _event()
    termination = _termination()
    context = _context(events=(event,), terminals={0: termination})

    replaced = replace(context, t=3)

    assert replaced is not context
    assert replaced.t == 3
    assert replaced.events == (event,)
    assert dict(replaced.terminals) == {0: termination}
    assert replace(context, events=()).events == ()
    assert dict(replace(context, terminals={1: termination}).terminals) == {1: termination}


def test_from_env_step_wraps_without_a_bundle() -> None:
    step = EnvStepContext(
        view=_view(2),
        values=StepValues({"level": torch.tensor([1.0, 2.0])}),
        t=4,
        step_index=torch.zeros(2, dtype=torch.int64),
        time_s=None,
        cmd=None,
        reset_state={0: {"reset": True}},
    )

    context = BenchmarkStepContext.from_env_step(step)

    assert isinstance(context, BenchmarkStepContext)
    assert context.view is step.view
    assert context.values is step.values
    assert context.t == step.t
    assert context.step_index is step.step_index
    assert dict(context.reset_state) == dict(step.reset_state)
    assert context.events == ()
    assert dict(context.terminals) == {}

    with pytest.raises(TypeError, match="expected an EnvStepContext"):
        BenchmarkStepContext.from_env_step(object())  # type: ignore[arg-type]


def test_no_pack_frame_type_remains() -> None:
    assert importlib.util.find_spec("unigym_bench.frames") is None
    assert not hasattr(unigym_bench, "TickFrame")
    assert dataclasses.is_dataclass(BenchmarkStepContext)


def test_engine_rejects_missing_step_context() -> None:
    engine = StreamingEngine([])

    with pytest.raises(EngineStateError, match="bundle.step"):
        engine.on_step({"view": _view(1), "t": 0})
    with pytest.raises(EngineStateError, match="bundle.step"):
        engine.on_step({"step": None, "step_index": torch.zeros(1, dtype=torch.int64)})
    with pytest.raises(TypeError, match="expected an EnvStepContext"):
        engine.on_step({"step": object()})


def test_engine_reads_only_bundle_step() -> None:
    engine = StreamingEngine([])
    context = _context(num_envs=1)

    bundle = {
        "step": context,
        "view": object(),
        "t": "not a tick",
        "step_index": object(),
        "time_s": object(),
        "cmd": object(),
        "reset_state": object(),
    }
    engine.on_step(bundle)

    assert engine.num_envs == 1
    assert engine.open_envs == (0,)
