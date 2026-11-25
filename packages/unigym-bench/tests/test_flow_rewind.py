"""Stateful flow checkpoints through the single pack rewind provider."""

from __future__ import annotations

import json

import pytest
import torch

from unigym_bench.engine import REWIND_SCHEMA, StreamingEngine
from unigym_bench.flow import (
    BenchmarkTick,
    DefaultBenchmarkFlow,
    EpisodeBatch,
    EpisodeEvidence,
    FlowDecision,
    FlowRun,
)
from unigym_bench.json import JsonValue
from unigym_bench.monitors import threshold_config
from unigym_bench.sink import BenchmarkSink
from unigym_bench.specs import RetentionSpec
from _support import CountingReportExtension, make_bundle, make_values


class CountingFlow:
    """Stateful flow whose ticks/episodes counters prove a round trip."""

    name = "counted"
    stateful = True

    def __init__(self) -> None:
        self.ticks = 0
        self.episodes = 0

    def on_tick(self, tick: BenchmarkTick) -> FlowDecision:
        """Count one observed tick."""
        del tick
        self.ticks += 1
        return FlowDecision()

    def on_episode_end(self, batch: EpisodeBatch) -> EpisodeEvidence | None:
        """Count each due environment and invoke the declared extensions."""
        self.episodes += len(batch.due)
        for extension in batch.extensions:
            extension.on_env_end(batch.due, batch.records)
        return None

    def on_run_end(self, run: FlowRun) -> None:
        """Delegate the report to the sink's default fallback."""
        del run

    def snapshot(self) -> dict[str, int]:
        """Return the JSON-safe counter state."""
        return {"ticks": self.ticks, "episodes": self.episodes}

    def restore(self, state: JsonValue) -> None:
        """Restore the counter state."""
        assert isinstance(state, dict)
        self.ticks = state["ticks"]  # type: ignore[assignment]
        self.episodes = state["episodes"]  # type: ignore[assignment]

    def validate_restore(self, state: JsonValue) -> None:
        """Refuse a structurally valid but policy-invalid counter state."""
        if not isinstance(state, dict):
            raise TypeError("counted flow state: expected an object")
        if set(state) != {"ticks", "episodes"}:
            raise ValueError(
                f"counted flow state: expected keys ['episodes', 'ticks'], " f"got {sorted(state)}"
            )
        for name in ("ticks", "episodes"):
            value = state[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"counted flow state.{name}: expected a non-negative int, got {value!r}"
                )


class ExplodingFlow(CountingFlow):
    """Stateful flow that fails after partially mutating during restore."""

    def validate_restore(self, state: JsonValue) -> None:
        """Skip pre-flight so the failure happens inside ``restore``."""

    def restore(self, state: JsonValue) -> None:
        """Apply the poisoned state, then raise after mutating."""
        assert isinstance(state, dict)
        self.ticks = state["ticks"]  # type: ignore[assignment]
        self.episodes = state["episodes"]  # type: ignore[assignment]
        if state.get("explode"):
            self.ticks = -1
            raise RuntimeError("flow restore boom")


class TensorFlow(CountingFlow):
    """Stateful flow whose snapshot is not JSON-safe."""

    name = "tensor"

    def snapshot(self) -> dict[str, object]:
        """Return a live tensor, which the provider must refuse."""
        return {"ticks": torch.tensor(self.ticks)}


def _sink(
    flow: object,
    *,
    extension: CountingReportExtension | None = None,
) -> tuple[BenchmarkSink, StreamingEngine]:
    monitor = threshold_config("contact", "value:level", threshold=0.5)
    engine = StreamingEngine(
        [monitor],
        retention=RetentionSpec(retain_episodes=True),
        extensions=() if extension is None else (extension,),
    )
    return BenchmarkSink(engine=engine, flow=flow), engine


def _bundle(t: int, row: list[float]) -> dict[str, object]:
    return make_bundle(num_envs=1, t=t, values=make_values(row))


def _step(sink: BenchmarkSink, t: int, row: list[float]) -> dict[str, object]:
    bundle = _bundle(t, row)
    sink.on_step(t, bundle)
    return bundle


def _due(sink: BenchmarkSink, t: int, row: list[float]) -> None:
    bundle = _step(sink, t, row)
    sink.drain_terminations()
    sink.on_episode_end([0], bundle)
    sink.on_reset([0])


def test_stateless_default_flow_adds_no_flow_key() -> None:
    sink, engine = _sink(DefaultBenchmarkFlow())

    _step(sink, 0, [0.0])
    state = sink.rewind_provider.rewind_state()

    assert sink.flow.stateful is False
    assert sink.rewind_provider is engine
    assert sink.rewind_provider.name == "unigym_bench"
    assert state["schema"] == REWIND_SCHEMA
    assert set(state) == {"schema", "engine", "monitors"}
    assert "flow" not in state


def test_stateful_flow_state_round_trips_after_a_due_episode() -> None:
    flow = CountingFlow()
    sink, engine = _sink(flow)
    assert sink.rewind_provider is engine

    _due(sink, 0, [1.0])
    snapshot = sink.rewind_provider.rewind_state()
    assert snapshot["flow"] == {
        "name": "counted",
        "state": {"ticks": 1, "episodes": 1},
    }

    _due(sink, 1, [1.0])
    assert flow.ticks == 2
    assert flow.episodes == 2
    assert engine.summary.episodes == 2

    sink.rewind_provider.apply_rewind_state(snapshot)

    assert flow.snapshot() == {"ticks": 1, "episodes": 1}
    assert engine.summary.episodes == 1
    assert [record.episode_seq for record in engine.records] == [0]

    _due(sink, 1, [1.0])

    assert flow.snapshot() == {"ticks": 2, "episodes": 2}
    assert engine.summary.episodes == 2
    assert [record.episode_seq for record in engine.records] == [0, 1]


def test_stop_requested_is_snapshotted_and_restored() -> None:
    sink, engine = _sink(CountingFlow())

    _step(sink, 0, [0.0])
    clear = sink.rewind_provider.rewind_state()
    assert clear["engine"]["stop_requested"] is False

    tick = engine.begin_tick(_bundle(1, [0.0]))
    engine.apply_decision(tick, FlowDecision(stop=True))
    assert sink.stop_when() is True
    stopped = sink.rewind_provider.rewind_state()
    assert stopped["engine"]["stop_requested"] is True

    sink.rewind_provider.apply_rewind_state(clear)
    assert sink.stop_when() is False

    sink.rewind_provider.apply_rewind_state(stopped)
    assert sink.stop_when() is True
    sink.on_rewind({})
    assert sink.stop_when() is True


def test_flow_state_is_json_safe_private_and_does_not_duplicate_evidence() -> None:
    sink, _engine = _sink(CountingFlow())
    _due(sink, 0, [1.0])

    state = sink.rewind_provider.rewind_state()
    assert json.loads(json.dumps(state)) == state
    assert set(state["flow"]) == {"name", "state"}
    assert "records" not in state["flow"]["state"]
    assert "episode_evidence" not in state["flow"]["state"]

    state["flow"]["state"]["ticks"] = 999
    assert sink.rewind_provider.rewind_state()["flow"]["state"]["ticks"] == 1


def test_malformed_or_non_json_flow_payload_is_rejected_with_zero_change() -> None:
    sink, _engine = _sink(CountingFlow())
    _due(sink, 0, [1.0])
    before = sink.rewind_provider.rewind_state()

    invalid_counter = json.loads(json.dumps(before))
    invalid_counter["flow"]["state"]["ticks"] = -1
    with pytest.raises(ValueError, match="ticks"):
        sink.rewind_provider.validate_rewind_state(invalid_counter)
    with pytest.raises(ValueError, match="ticks"):
        sink.rewind_provider.apply_rewind_state(invalid_counter)
    assert sink.rewind_provider.rewind_state() == before

    live_tensor = json.loads(json.dumps(before))
    live_tensor["flow"]["state"]["ticks"] = torch.tensor(1)
    with pytest.raises(TypeError, match="expected a JSON value"):
        sink.rewind_provider.apply_rewind_state(live_tensor)
    assert sink.rewind_provider.rewind_state() == before


def test_missing_extra_and_mismatched_flow_payloads_are_rejected() -> None:
    stateful_sink, _engine = _sink(CountingFlow())
    _due(stateful_sink, 0, [1.0])
    stateful = stateful_sink.rewind_provider.rewind_state()

    missing = json.loads(json.dumps(stateful))
    del missing["flow"]
    with pytest.raises(ValueError, match="requires a flow snapshot"):
        stateful_sink.rewind_provider.validate_rewind_state(missing)
    with pytest.raises(ValueError, match="requires a flow snapshot"):
        stateful_sink.rewind_provider.apply_rewind_state(missing)

    mismatched = json.loads(json.dumps(stateful))
    mismatched["flow"]["name"] = "other"
    with pytest.raises(ValueError, match="does not match live flow"):
        stateful_sink.rewind_provider.apply_rewind_state(mismatched)

    unknown_key = json.loads(json.dumps(stateful))
    unknown_key["flow"]["extra"] = 1
    with pytest.raises(ValueError, match="unknown"):
        stateful_sink.rewind_provider.apply_rewind_state(unknown_key)

    stateless_sink, _engine = _sink(DefaultBenchmarkFlow())
    _step(stateless_sink, 0, [0.0])
    stateless = stateless_sink.rewind_provider.rewind_state()
    stateless["flow"] = {"name": "default", "state": {}}
    with pytest.raises(ValueError, match="live flow is stateless"):
        stateless_sink.rewind_provider.apply_rewind_state(stateless)


def test_old_and_pre_f5_payloads_are_rejected_for_a_stateful_flow() -> None:
    sink, _engine = _sink(CountingFlow())
    _due(sink, 0, [1.0])
    state = sink.rewind_provider.rewind_state()

    old_rewind = json.loads(json.dumps(state))
    old_rewind["schema"] = "unigym_bench.rewind/4"
    with pytest.raises(ValueError, match=r"expected 'unigym_bench.rewind/5'"):
        sink.rewind_provider.validate_rewind_state(old_rewind)

    pre_f5 = json.loads(json.dumps(state))
    del pre_f5["engine"]["stop_requested"]
    with pytest.raises(ValueError, match="stop_requested"):
        sink.rewind_provider.apply_rewind_state(pre_f5)

    pre_flow = json.loads(json.dumps(state))
    del pre_flow["flow"]
    with pytest.raises(ValueError, match="requires a flow snapshot"):
        sink.rewind_provider.apply_rewind_state(pre_flow)


def test_atomic_rollback_on_flow_restore_failure() -> None:
    extension = CountingReportExtension()
    flow = ExplodingFlow()
    sink, engine = _sink(flow, extension=extension)

    _due(sink, 0, [1.0])
    before = sink.rewind_provider.rewind_state()
    assert before["flow"]["state"] == {"ticks": 1, "episodes": 1}
    assert extension.episodes == 1

    poisoned = json.loads(json.dumps(before))
    poisoned["flow"]["state"] = {"ticks": 1, "episodes": 1, "explode": True}

    with pytest.raises(RuntimeError, match="flow restore boom"):
        sink.rewind_provider.apply_rewind_state(poisoned)

    assert sink.rewind_provider.rewind_state() == before
    assert flow.snapshot() == {"ticks": 1, "episodes": 1}
    assert extension.episodes == 1
    assert engine.summary.episodes == 1


def test_non_json_flow_snapshot_is_rejected() -> None:
    sink, _engine = _sink(TensorFlow())
    _step(sink, 0, [0.0])

    with pytest.raises(TypeError, match="rewind snapshot"):
        sink.rewind_provider.rewind_state()
