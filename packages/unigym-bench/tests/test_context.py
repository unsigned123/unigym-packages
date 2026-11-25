"""Benchmark step context: construction, validation and engine adoption."""

from __future__ import annotations

import dataclasses
import importlib.util
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

import unigym_bench
from unigym.evaluation import EnvStepContext, EnvView
from unigym.values import StepValues
from unigym_bench.context import BenchmarkStepContext
from unigym_bench.engine import EngineStateError, StreamingEngine
from unigym_bench.query import TickQuery


def _view(num_envs: int = 2) -> EnvView:
    return EnvView(record={}, obs=None, env_id=None, num_envs=num_envs, device="cpu")


def _context(
    num_envs: int = 2,
    *,
    query: object = None,
) -> BenchmarkStepContext:
    return BenchmarkStepContext(
        view=_view(num_envs),
        values=StepValues({}),
        t=0,
        step_index=torch.zeros(num_envs, dtype=torch.int64),
        time_s=None,
        cmd=None,
        reset_state={},
        query=query,  # type: ignore[arg-type]
    )


def test_subclass_construction_and_defaults() -> None:
    context = _context()

    assert isinstance(context, EnvStepContext)
    assert isinstance(context, BenchmarkStepContext)
    assert context.query is None
    assert context.num_envs == 2
    assert context.t == 0


def test_frozen_and_slots() -> None:
    context = _context()

    with pytest.raises(FrozenInstanceError):
        context.t = 1  # type: ignore[misc]
    assert not hasattr(context, "__dict__")
    assert hasattr(BenchmarkStepContext, "__slots__")


def test_replace_preserves_the_pack_query_field() -> None:
    context = _context()

    replaced = replace(context, t=3)

    assert replaced is not context
    assert replaced.t == 3
    assert replaced.query is None


def test_query_field_requires_a_tick_query_or_none() -> None:
    view = _view(2)
    tick = TickQuery(view)

    assert _context(query=tick).query is tick
    with pytest.raises(TypeError, match="TickQuery or None"):
        _context(query=object())


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
    assert context.query is None

    with pytest.raises(TypeError, match="expected an EnvStepContext"):
        BenchmarkStepContext.from_env_step(object())  # type: ignore[arg-type]


def test_from_env_step_does_not_inherit_a_tick_snapshot() -> None:
    source = _context(query=TickQuery(_view(2)))
    explicit = TickQuery(_view(2))

    wrapped = BenchmarkStepContext.from_env_step(source)
    attached = BenchmarkStepContext.from_env_step(source, query=explicit)

    assert wrapped.query is None
    assert attached.query is explicit


def test_replace_preserves_an_attached_tick_snapshot() -> None:
    tick = TickQuery(_view(2))
    context = _context(query=tick)

    assert replace(context, t=5).query is tick


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
