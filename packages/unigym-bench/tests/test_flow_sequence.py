"""FlowSequence: single terminal owner, report owner, evidence and checkpoints."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import pytest
from unigym.core.episode_termination import Termination

from unigym_bench.engine import StreamingEngine
from unigym_bench.events import EpisodeSummary, Event
from unigym_bench.flow import (
    BenchmarkTick,
    DefaultBenchmarkFlow,
    EpisodeBatch,
    EpisodeEvidence,
    FlowDecision,
    FlowRun,
    FlowSequence,
    validate_flow,
)
from unigym_bench.json import JsonValue
from unigym_bench.report import REPORT_SCHEMA, RunMetadata
from unigym_bench.sink import BenchmarkSink
from unigym_bench.specs import RetentionSpec
from _support import ScriptedMonitor, TickScript, make_bundle, make_context, make_termination


def _event(code: str) -> Event:
    return Event(name=code.rsplit(".", 1)[-1], code=code, env_ids=(0,))


def _flow_run() -> FlowRun:
    return FlowRun(
        metadata=RunMetadata(),
        summary=EpisodeSummary(),
        records=(),
        monitor_views={},
    )


class ChildFlow:
    """Configurable stateless child that records every callback."""

    stateful = False

    def __init__(
        self,
        name: str,
        *,
        events: tuple[Event, ...] = (),
        terminations: Mapping[int, Termination] | None = None,
        evidence: Mapping[int, Mapping[str, JsonValue]] | None = None,
        report: Mapping[str, JsonValue] | None = None,
        stop: bool = False,
    ) -> None:
        self.name = name
        self._events = events
        self._terminations = {} if terminations is None else dict(terminations)
        self._evidence = evidence
        self._report = report
        self._stop = stop
        self.ticks: list[BenchmarkTick] = []
        self.batches: list[EpisodeBatch] = []
        self.runs: list[FlowRun] = []

    def on_tick(self, tick: BenchmarkTick) -> FlowDecision:
        self.ticks.append(tick)
        return FlowDecision(
            events=self._events,
            terminations=self._terminations,
            stop=self._stop,
        )

    def on_episode_end(self, batch: EpisodeBatch) -> EpisodeEvidence | None:
        self.batches.append(batch)
        if self._evidence is None:
            return None
        return EpisodeEvidence(self._evidence)

    def on_run_end(self, run: FlowRun) -> Mapping[str, JsonValue] | None:
        self.runs.append(run)
        return self._report


class BoundaryOnlyFlow:
    """Valid child without an ``on_tick`` callback."""

    stateful = False

    def __init__(self, name: str) -> None:
        self.name = name
        self.batches: list[EpisodeBatch] = []
        self.runs: list[FlowRun] = []

    def on_episode_end(self, batch: EpisodeBatch) -> None:
        self.batches.append(batch)

    def on_run_end(self, run: FlowRun) -> None:
        self.runs.append(run)


class ResetChild(BoundaryOnlyFlow):
    """Child that records reset notifications and can fail on demand."""

    def __init__(self, name: str, calls: list[tuple[str, tuple[int, ...]]], *, fail: bool = False):
        super().__init__(name)
        self.calls = calls
        self.fail = fail

    def on_reset(self, env_ids: Sequence[int]) -> None:
        """Record the ordered call, then raise when configured."""
        self.calls.append((self.name, tuple(int(env_id) for env_id in env_ids)))
        if self.fail:
            raise RuntimeError(f"{self.name} reset boom")


class StatefulChild:
    """Stateful child whose cursor must survive a checkpoint round trip."""

    stateful = True

    def __init__(self, name: str, cursor: int = 0) -> None:
        self.name = name
        self.cursor = cursor

    def on_tick(self, tick: BenchmarkTick) -> FlowDecision:
        del tick
        self.cursor += 1
        return FlowDecision()

    def on_episode_end(self, batch: EpisodeBatch) -> None:
        del batch

    def on_run_end(self, run: FlowRun) -> None:
        del run

    def snapshot(self) -> dict[str, int]:
        return {"cursor": self.cursor}

    def restore(self, state: JsonValue) -> None:
        assert isinstance(state, dict)
        self.cursor = state["cursor"]  # type: ignore[assignment]

    def validate_restore(self, state: JsonValue) -> None:
        if not isinstance(state, dict) or "cursor" not in state:
            raise ValueError("stateful child state: expected a cursor")
        cursor = state["cursor"]
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError(
                f"stateful child state.cursor: expected a non-negative int, got {cursor!r}"
            )


class RestoreBoomChild:
    """Stateful child that mutates and then fails on a poisoned restore."""

    stateful = True
    name = "boom"

    def __init__(self, cursor: int = 0) -> None:
        self.cursor = cursor

    def on_tick(self, tick: BenchmarkTick) -> FlowDecision:
        del tick
        return FlowDecision()

    def on_episode_end(self, batch: EpisodeBatch) -> None:
        del batch

    def on_run_end(self, run: FlowRun) -> None:
        del run

    def snapshot(self) -> dict[str, int]:
        return {"cursor": self.cursor}

    def restore(self, state: JsonValue) -> None:
        assert isinstance(state, dict)
        self.cursor = state["cursor"]  # type: ignore[assignment]
        if state.get("explode"):
            raise RuntimeError("boom restore")


def _sink(flow: FlowSequence) -> tuple[BenchmarkSink, StreamingEngine]:
    monitor = ScriptedMonitor("contact", {0: TickScript()})
    engine = StreamingEngine([monitor], retention=RetentionSpec(retain_episodes=True))
    return BenchmarkSink(engine=engine, flow=flow), engine


def test_duplicate_child_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate child flow names"):
        FlowSequence(
            (ChildFlow("dup"), ChildFlow("dup")),
            terminal_owner="dup",
            report_owner="dup",
        )


def test_empty_child_name_and_empty_sequence_are_rejected() -> None:
    with pytest.raises(TypeError, match="non-empty"):
        FlowSequence((ChildFlow(""),), terminal_owner="", report_owner="")

    with pytest.raises(ValueError, match="at least one child flow"):
        FlowSequence((), terminal_owner="a", report_owner="a")


@pytest.mark.parametrize("field", ["terminal_owner", "report_owner"])
def test_owner_must_name_a_declared_child(field: str) -> None:
    owners = {"terminal_owner": "a", "report_owner": "a", field: "missing"}

    with pytest.raises(ValueError, match="not a declared child"):
        FlowSequence((ChildFlow("a"),), **owners)


@pytest.mark.parametrize("field", ["terminal_owner", "report_owner"])
def test_owner_must_be_a_non_empty_string(field: str) -> None:
    owners = {"terminal_owner": "a", "report_owner": "a", field: ""}

    with pytest.raises(TypeError, match="non-empty"):
        FlowSequence((ChildFlow("a"),), **owners)


def test_every_child_receives_the_same_immutable_tick() -> None:
    first = ChildFlow("first")
    second = ChildFlow("second")
    boundary = BoundaryOnlyFlow("boundary")
    sequence = FlowSequence(
        (first, second, boundary),
        terminal_owner="first",
        report_owner="first",
    )
    tick = BenchmarkTick(step=make_context(num_envs=1))

    decision = sequence.on_tick(tick)

    assert decision == FlowDecision()
    assert first.ticks == [tick]
    assert second.ticks == [tick]
    assert first.ticks[0] is tick
    assert second.ticks[0] is tick


def test_owner_terminations_events_and_stop_combine_in_flow_order() -> None:
    first = ChildFlow("first", events=(_event("custom.first.a"),))
    owner = ChildFlow(
        "owner",
        events=(_event("custom.owner.b"),),
        terminations={1: make_termination("custom.owner.done")},
        stop=True,
    )
    sequence = FlowSequence((first, owner), terminal_owner="owner", report_owner="owner")

    decision = sequence.on_tick(BenchmarkTick(step=make_context(num_envs=2)))

    assert [event.code for event in decision.events] == ["custom.first.a", "custom.owner.b"]
    assert set(decision.terminations) == {1}
    assert decision.terminations[1].code == "custom.owner.done"
    assert decision.stop is True


def test_non_owner_terminations_are_a_teaching_error() -> None:
    outsider = ChildFlow(
        "outsider",
        terminations={0: make_termination("custom.outsider.hit")},
    )
    owner = ChildFlow("owner")
    sequence = FlowSequence((outsider, owner), terminal_owner="owner", report_owner="owner")

    with pytest.raises(ValueError, match="only the terminal owner 'owner'") as error:
        sequence.on_tick(BenchmarkTick(step=make_context(num_envs=1)))

    assert "outsider" in str(error.value)


def test_non_owner_stop_is_a_teaching_error() -> None:
    outsider = ChildFlow("outsider", stop=True)
    owner = ChildFlow("owner")
    sequence = FlowSequence((outsider, owner), terminal_owner="owner", report_owner="owner")

    with pytest.raises(ValueError, match="only the terminal owner 'owner' may stop") as error:
        sequence.on_tick(BenchmarkTick(step=make_context(num_envs=1)))

    assert "outsider" in str(error.value)
    assert "events and evidence only" in str(error.value)


def test_non_owner_report_is_a_teaching_error() -> None:
    outsider = ChildFlow("outsider", report={"score": 1})
    owner = ChildFlow("owner")
    sequence = FlowSequence((outsider, owner), terminal_owner="owner", report_owner="owner")

    with pytest.raises(ValueError, match="only the report owner 'owner'") as error:
        sequence.on_run_end(_flow_run())

    assert "outsider" in str(error.value)


def test_report_owner_is_the_only_report_source() -> None:
    silent = ChildFlow("silent")
    owner = ChildFlow("owner", report={"score": 1})
    sequence = FlowSequence((silent, owner), terminal_owner="silent", report_owner="owner")

    assert sequence.on_run_end(_flow_run()) == {"score": 1}


def test_default_flow_can_be_a_child_and_report_owner() -> None:
    observer = BoundaryOnlyFlow("observer")
    sequence = FlowSequence(
        (observer, DefaultBenchmarkFlow()),
        terminal_owner="default",
        report_owner="default",
    )

    report = sequence.on_run_end(_flow_run())

    assert report is not None
    assert report["schema"] == REPORT_SCHEMA


def test_every_child_observes_the_boundary_and_shared_keys_are_namespaced() -> None:
    # Both children use the same local "score" key.  Duplicate flow names are
    # impossible by construction, so the namespaces keep both values instead of
    # one child overwriting the other.
    first = ChildFlow("first", evidence={0: {"score": 1, "only_first": True}})
    second = ChildFlow("second", evidence={0: {"score": 2}, 1: {"score": 3}})
    silent = ChildFlow("silent")
    sequence = FlowSequence(
        (first, second, silent),
        terminal_owner="first",
        report_owner="first",
    )
    batch = EpisodeBatch(due=(0, 1), final_step=make_context(num_envs=2))

    evidence = sequence.on_episode_end(batch)

    assert evidence is not None
    assert set(evidence.evidence[0]) == {"first", "second"}
    assert evidence.evidence[0] == {
        "first": {"score": 1, "only_first": True},
        "second": {"score": 2},
    }
    assert evidence.evidence[0]["first"]["score"] != evidence.evidence[0]["second"]["score"]
    assert evidence.evidence[1] == {"second": {"score": 3}}
    assert first.batches == [batch]
    assert second.batches == [batch]
    assert silent.batches == [batch]


def test_on_reset_forwards_to_every_callable_child_in_declaration_order() -> None:
    calls: list[tuple[str, tuple[int, ...]]] = []
    first = ResetChild("first", calls)
    second = BoundaryOnlyFlow("second")
    third = ResetChild("third", calls)
    sequence = FlowSequence(
        (first, second, third),
        terminal_owner="first",
        report_owner="first",
    )

    sequence.on_reset([2, 0])

    assert calls == [("first", (2, 0)), ("third", (2, 0))]
    assert not hasattr(second, "on_reset")


def test_on_reset_reaches_non_owner_children_and_stops_on_exception() -> None:
    calls: list[tuple[str, tuple[int, ...]]] = []
    first = ResetChild("first", calls)
    boom = ResetChild("boom", calls, fail=True)
    later = ResetChild("later", calls)
    sequence = FlowSequence(
        (first, boom, later),
        terminal_owner="first",
        report_owner="first",
    )

    with pytest.raises(RuntimeError, match="boom reset boom"):
        sequence.on_reset([0])

    assert calls == [("first", (0,)), ("boom", (0,))]


def test_stateful_child_state_round_trips_as_a_flow_name_map() -> None:
    first = StatefulChild("first", cursor=3)
    second = StatefulChild("second", cursor=7)
    plain = ChildFlow("plain")
    sequence = FlowSequence(
        (first, second, plain),
        terminal_owner="first",
        report_owner="first",
    )

    assert sequence.stateful is True
    assert validate_flow(sequence) is sequence
    snapshot = sequence.snapshot()
    assert snapshot == {"first": {"cursor": 3}, "second": {"cursor": 7}}
    assert json.loads(json.dumps(snapshot)) == snapshot

    first.cursor = 10
    second.cursor = 20
    sequence.restore(snapshot)

    assert (first.cursor, second.cursor) == (3, 7)


def test_restore_preflights_child_state_before_mutating() -> None:
    first = StatefulChild("first", cursor=3)
    second = StatefulChild("second", cursor=7)
    sequence = FlowSequence((first, second), terminal_owner="first", report_owner="first")
    poisoned = {"first": {"cursor": -1}, "second": {"cursor": 7}}

    with pytest.raises(ValueError, match="cursor"):
        sequence.validate_restore(poisoned)
    with pytest.raises(ValueError, match="cursor"):
        sequence.restore(poisoned)

    assert (first.cursor, second.cursor) == (3, 7)

    with pytest.raises(ValueError, match="do not match"):
        sequence.restore({"first": {"cursor": 1}})

    assert (first.cursor, second.cursor) == (3, 7)


def test_restore_failure_rolls_every_child_back() -> None:
    # Rollback is best-effort: both children have a working restore for the
    # rollback state.  A broken child whose restore also fails cannot be
    # rolled back; the engine aborts the rewind in that case.
    first = StatefulChild("first", cursor=3)
    boom = RestoreBoomChild(cursor=5)
    sequence = FlowSequence((first, boom), terminal_owner="first", report_owner="first")
    poisoned = {"first": {"cursor": 11}, "boom": {"cursor": 13, "explode": True}}

    with pytest.raises(RuntimeError, match="boom restore"):
        sequence.restore(poisoned)

    assert first.cursor == 3
    assert boom.cursor == 5


def test_stateless_sequence_advertises_no_checkpoint_face() -> None:
    sequence = FlowSequence(
        (ChildFlow("a"), BoundaryOnlyFlow("b")),
        terminal_owner="a",
        report_owner="a",
    )

    assert sequence.stateful is False
    assert not hasattr(sequence, "snapshot")
    assert not hasattr(sequence, "restore")
    assert validate_flow(sequence) is sequence


def test_stateful_sequence_joins_the_engine_checkpoint() -> None:
    first = StatefulChild("first")
    sequence = FlowSequence((first,), terminal_owner="first", report_owner="first")
    sink, engine = _sink(sequence)
    assert sink.rewind_provider is engine

    sink.on_step(0, make_bundle(num_envs=1, t=0))
    state = sink.rewind_provider.rewind_state()
    assert state["flow"] == {"name": "sequence", "state": {"first": {"cursor": 1}}}

    sink.on_step(1, make_bundle(num_envs=1, t=1))
    assert first.cursor == 2
    sink.rewind_provider.apply_rewind_state(state)

    assert first.cursor == 1


def test_stateless_sequence_adds_no_flow_key_to_the_checkpoint() -> None:
    sequence = FlowSequence(
        (ChildFlow("a"), BoundaryOnlyFlow("b")),
        terminal_owner="a",
        report_owner="a",
    )
    sink, _engine = _sink(sequence)

    sink.on_step(0, make_bundle(num_envs=1, t=0))

    assert "flow" not in sink.rewind_provider.rewind_state()


def test_namespaced_evidence_reaches_the_pack_report_through_the_engine() -> None:
    first = ChildFlow("first", evidence={0: {"score": 1}})
    second = ChildFlow("second", evidence={0: {"score": 2}})
    sequence = FlowSequence((first, second), terminal_owner="first", report_owner="first")
    sink, _engine = _sink(sequence)
    bundle = make_bundle(num_envs=1, t=0)

    sink.on_step(0, bundle)
    sink.drain_terminations()
    sink.on_episode_end([0], bundle)
    report = sink.on_run_end()["BenchmarkSink"]

    assert report["episodes"]["evidence"] == {
        "0/0": {"first": {"score": 1}, "second": {"score": 2}}
    }
