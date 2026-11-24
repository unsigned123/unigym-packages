"""Construction helpers for monitor tests: real frames and simple probes.

These are test inputs, not stand-ins for pack components: every frame is built
from the real :class:`unigym_bench.frames.TickFrame`, and every probe is a
small callable that returns a real tensor for the frame's device.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import torch

from unigym.core.episode_termination import (
    EpisodeOutcome,
    Termination,
    TerminationSource,
)
from unigym.evaluation import EnvView
from unigym.run import SinkContext
from unigym.run.spec import build_spec
from unigym.signals import SignalPublisher
from unigym_bench.events import Event
from unigym_bench.frames import TickFrame
from unigym_bench.json import JsonValue
from unigym_bench.monitors import BaseMonitor, Monitor, ThresholdMonitor


def make_view(num_envs: int = 2, *, device: str | torch.device = "cpu") -> EnvView:
    """Build a real, empty vector view with no observation snapshot."""
    return EnvView(record={}, obs=None, env_id=None, num_envs=num_envs, device=device)


def make_frame(
    *,
    num_envs: int = 2,
    t: int = 0,
    cmd: torch.Tensor | None = None,
    view: EnvView | None = None,
    reset_state: Mapping[int, JsonValue] | None = None,
    step_index: torch.Tensor | None = None,
    time_s: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> TickFrame:
    """Build a frame with a real view and default per-env bookkeeping."""
    target = torch.device("cpu") if device is None else device
    if view is None:
        view = make_view(num_envs, device=target)
    if step_index is None:
        step_index = torch.zeros(num_envs, dtype=torch.int64, device=target)
    return TickFrame(
        view=view,
        t=t,
        step_index=step_index,
        time_s=time_s,
        cmd=cmd,
        reset_state={} if reset_state is None else reset_state,
        num_envs=num_envs,
        device=target,
    )


def make_bundle(
    *,
    num_envs: int = 2,
    t: int = 0,
    cmd: torch.Tensor | None = None,
    view: EnvView | None = None,
    reset_state: Mapping[int, JsonValue] | None = None,
    step_index: torch.Tensor | None = None,
    time_s: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
    terminations: Mapping[int, object] | None = None,
) -> dict[str, object]:
    """Build a real runner-shaped bundle for an engine tick."""
    target = torch.device(device)
    if view is None:
        view = make_view(num_envs, device=target)
    if step_index is None:
        step_index = torch.zeros(num_envs, dtype=torch.int64, device=target)
    bundle: dict[str, object] = {
        "t": t,
        "step_index": step_index,
        "time_s": time_s,
        "cmd": cmd,
        "reset_state": {} if reset_state is None else reset_state,
        "view": view,
    }
    if terminations is not None:
        bundle["terminations"] = terminations
    return bundle


def make_termination(
    code: str = "custom.test.reason",
    *,
    outcome: EpisodeOutcome = EpisodeOutcome.SUCCESS,
    source: TerminationSource = TerminationSource.CUSTOM,
    metrics: Mapping[str, float] | None = None,
) -> Termination:
    """Build a real core termination for engine tests."""
    return Termination(
        outcome=outcome,
        code=code,
        source=source,
        metrics={} if metrics is None else metrics,
    )


@dataclass(frozen=True)
class TickScript:
    """One tick's scripted monitor outputs."""

    signals: Mapping[str, torch.Tensor] = field(default_factory=dict)
    events: tuple[Event, ...] = ()
    terminals: Mapping[int, Termination] = field(default_factory=dict)


class ScriptedMonitor(BaseMonitor):
    """Test monitor that emits exactly the outputs scripted for each tick."""

    def __init__(self, name: str, script: Mapping[int, TickScript]) -> None:
        super().__init__(name)
        self._script = dict(script)
        self._signals: dict[str, torch.Tensor] = {}
        self._events: list[Event] = []
        self._terminals: dict[int, Termination] = {}
        self.episode_ends: list[tuple[int, ...]] = []
        self.resets: list[tuple[int, ...]] = []

    def on_tick(self, frame: TickFrame) -> None:
        step = self._script.get(frame.t)
        if step is None:
            self._signals = {}
            return
        self._signals = {name: value.to(frame.device) for name, value in step.signals.items()}
        self._events = list(step.events)
        self._terminals = dict(step.terminals)

    def signals(self) -> Mapping[str, torch.Tensor]:
        return dict(self._signals)

    def drain_events(self) -> Sequence[Event]:
        events = tuple(self._events)
        self._events.clear()
        return events

    def drain_terminals(self) -> Mapping[int, Termination]:
        terminals = dict(self._terminals)
        self._terminals.clear()
        return terminals

    def on_episode_end(self, env_ids: Sequence[int]) -> None:
        self.episode_ends.append(tuple(env_ids))

    def reset(self, env_ids: Sequence[int]) -> None:
        self.resets.append(tuple(env_ids))


def constant_probe(
    values: Sequence[float], *, dtype: torch.dtype = torch.float32
) -> Callable[[TickFrame], torch.Tensor]:
    """Return a probe that always reports the same per-env row."""
    row = [float(value) for value in values]

    def probe(frame: TickFrame) -> torch.Tensor:
        return torch.tensor(row, dtype=dtype, device=frame.device)

    return probe


def bool_probe(values: Sequence[bool]) -> Callable[[TickFrame], torch.Tensor]:
    """Return a boolean probe that always reports the same per-env row."""
    row = [bool(value) for value in values]

    def probe(frame: TickFrame) -> torch.Tensor:
        return torch.tensor(row, dtype=torch.bool, device=frame.device)

    return probe


def series_probe(
    rows: Sequence[Sequence[float]], *, dtype: torch.dtype = torch.float32
) -> Callable[[TickFrame], torch.Tensor]:
    """Return a probe that advances one row per call."""
    remaining = [[float(value) for value in row] for row in rows]
    state = {"index": 0}

    def probe(frame: TickFrame) -> torch.Tensor:
        index = state["index"]
        if index >= len(remaining):
            raise AssertionError(f"series probe exhausted after {len(remaining)} calls")
        state["index"] = index + 1
        return torch.tensor(remaining[index], dtype=dtype, device=frame.device)

    return probe


def table_probe(
    rows: Sequence[Sequence[float]], *, dtype: torch.dtype = torch.float32
) -> Callable[[TickFrame], torch.Tensor]:
    """Return a probe that selects a row by the frame's policy tick."""
    table = [[float(value) for value in row] for row in rows]

    def probe(frame: TickFrame) -> torch.Tensor:
        if frame.t >= len(table):
            raise AssertionError(f"table probe has no row for tick {frame.t}")
        return torch.tensor(table[frame.t], dtype=dtype, device=frame.device)

    return probe


_SINK_CONTEXT_CFG: dict[str, JsonValue] = {
    "run_name": "sink-context",
    "seed": 0,
    "env": {"backend": "mujoco", "num_envs": 1},
    "world_base": {"mjcf": "model.xml"},
    "collect": {"steps": 1},
    "articulations": [{"name": "arm", "joints": [{"name": "shoulder"}]}],
    "action_spec": [
        {
            "name": "arm",
            "articulation": "arm",
            "joints": ["shoulder"],
            "target": "position",
            "role": "body",
        }
    ],
    "policies": [{"adapter": "manual", "groups": [["arm", "arm"]]}],
    "observations": {"robot": ["arm"]},
}


def make_sink_context(
    *,
    options: Mapping[str, JsonValue] | None = None,
    signals: SignalPublisher | None = None,
    auto_reset: bool | None = None,
) -> SinkContext:
    """Build a real SinkContext over a real, backend-free RunSpec."""
    spec = build_spec(_SINK_CONTEXT_CFG, Namespace(config="run.yaml"))
    if auto_reset is not None:
        spec = replace(spec, runner={**spec.runner, "auto_reset": auto_reset})
    return SinkContext(
        spec=spec,
        mode=spec.mode,
        out_dir="data_out/sink-context",
        num_envs=1,
        policy_dt=0.02,
        seed=0,
        action_groups=(),
        run_identity=None,
        options={} if options is None else options,
        signals=signals,
    )


def make_threshold_monitors() -> tuple[Monitor, ...]:
    """Importable zero-argument monitor factory used by make_sink tests."""
    return (
        ThresholdMonitor(
            "contact",
            constant_probe([0.0]),
            threshold=0.5,
            terminal_outcome="success",
        ),
    )


class StaticReportExtension:
    """Stateless extension that returns a fixed block and keeps no state."""

    stateful = False

    def __init__(self, name: str, block: Mapping[str, JsonValue]) -> None:
        self.name = name
        self._block = block

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, object]) -> None:
        """No accumulator."""
        del env_ids, records

    def on_run_end(self, summary: object) -> Mapping[str, JsonValue]:
        """Return the fixed block."""
        del summary
        return dict(self._block)


class ViewProbeReportExtension:
    """Stateless extension proving fallback views reach ``on_run_end``."""

    stateful = False

    def __init__(self, name: str = "contact_report") -> None:
        self.name = name

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, object]) -> None:
        """No accumulator."""
        del env_ids, records

    def on_run_end(self, summary: object) -> Mapping[str, JsonValue]:
        """Report whether the covered monitor's fallback view is visible."""
        views = getattr(summary, "monitor_views", {})
        return {"saw_view": "contact" in views}


class CaptureReportExtension:
    """Stateless extension that records boundary calls into a caller-owned list.

    The extension itself owns no accumulator; the test supplies the list, so
    the explicit ``stateful = False`` declaration stays true.
    """

    stateful = False

    def __init__(
        self, calls: list[tuple[tuple[int, ...], object]], *, name: str = "capture"
    ) -> None:
        self.name = name
        self._calls = calls

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, object]) -> None:
        """Record the due ids and the read-only mapping in the caller's list."""
        self._calls.append((tuple(env_ids), records))

    def on_run_end(self, summary: object) -> Mapping[str, JsonValue]:
        """Return the number of observed boundaries."""
        del summary
        return {"boundaries": len(self._calls)}


class CountingReportExtension:
    """Stateful extension whose accumulator must survive a rewind."""

    stateful = True

    def __init__(self, name: str = "counts") -> None:
        self.name = name
        self.episodes = 0
        self.envs: list[int] = []
        self.run_end_calls = 0

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, object]) -> None:
        """Accumulate one count per due environment."""
        del records
        ids = [int(env_id) for env_id in env_ids]
        self.episodes += len(ids)
        self.envs.extend(ids)

    def on_run_end(self, summary: object) -> Mapping[str, JsonValue]:
        """Count the call and return the accumulated state."""
        del summary
        self.run_end_calls += 1
        return {
            "episodes": self.episodes,
            "envs": list(self.envs),
            "run_end_calls": self.run_end_calls,
        }

    def snapshot(self) -> JsonValue:
        """Return a private JSON-safe copy of the accumulator."""
        return {
            "episodes": self.episodes,
            "envs": list(self.envs),
            "run_end_calls": self.run_end_calls,
        }

    def restore(self, state: JsonValue) -> None:
        """Replace the accumulator from a snapshot's private copy."""
        assert isinstance(state, dict)
        self.episodes = int(state["episodes"])
        envs = state["envs"]
        assert isinstance(envs, list)
        self.envs = [int(value) for value in envs]
        self.run_end_calls = int(state["run_end_calls"])


class NonJsonReportExtension:
    """Stateless extension that leaks a tensor into its block."""

    stateful = False
    name = "bad"

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, object]) -> None:
        """No accumulator."""
        del env_ids, records

    def on_run_end(self, summary: object) -> Mapping[str, object]:
        """Return a non-JSON value."""
        del summary
        return {"tensor": torch.tensor(1.0)}


class ListReportExtension:
    """Stateless extension that returns a non-mapping block."""

    stateful = False
    name = "listy"

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, object]) -> None:
        """No accumulator."""
        del env_ids, records

    def on_run_end(self, summary: object) -> object:
        """Return a list, which is not a report block."""
        del summary
        return [1, 2, 3]


class MissingSnapshotExtension:
    """Invalid double: declares stateful without snapshot/restore."""

    stateful = True
    name = "no_snapshot"

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, object]) -> None:
        """No accumulator."""
        del env_ids, records

    def on_run_end(self, summary: object) -> Mapping[str, JsonValue]:
        """Return an empty block."""
        del summary
        return {}


class StatelessWithSnapshotExtension:
    """Invalid double: declares stateless but implements snapshot/restore."""

    stateful = False
    name = "contradiction"

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, object]) -> None:
        """No accumulator."""
        del env_ids, records

    def on_run_end(self, summary: object) -> Mapping[str, JsonValue]:
        """Return an empty block."""
        del summary
        return {}

    def snapshot(self) -> JsonValue:
        """Return an empty snapshot."""
        return {}

    def restore(self, state: JsonValue) -> None:
        """Accept an empty snapshot."""
        del state


class UndeclaredExtension:
    """Invalid double: does not declare ``stateful``."""

    name = "undeclared"

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, object]) -> None:
        """No accumulator."""
        del env_ids, records

    def on_run_end(self, summary: object) -> Mapping[str, JsonValue]:
        """Return an empty block."""
        del summary
        return {}


class SummaryFactsReportExtension:
    """Stateless extension that projects facts from ``RunSummary`` only."""

    stateful = False
    name = "summary_facts"

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, object]) -> None:
        """No accumulator."""
        del env_ids, records

    def on_run_end(self, summary: object) -> Mapping[str, JsonValue]:
        """Return retained-episode facts visible through the summary."""
        episode_summary = getattr(summary, "episode_summary")
        retention = getattr(summary, "retention")
        return {
            "episodes": len(getattr(summary, "episodes")),
            "verdicts": dict(episode_summary.verdict_counts),
            "retain_episodes": bool(retention.retain_episodes),
        }


class LeakyReportViewMonitor(BaseMonitor):
    """Monitor whose fallback view leaks a tensor."""

    def report_view(self) -> Mapping[str, JsonValue]:
        """Return a non-JSON view."""
        return {"bad": torch.tensor(1.0)}  # type: ignore[return-value]


class RecordingReportExtension:
    """Stateful extension used by the sink construction-seam tests."""

    stateful = True

    def __init__(self, name: str = "execution") -> None:
        self.name = name
        self.env_ends: list[tuple[int, ...]] = []

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, object]) -> None:
        """Record the due env ids without retaining the records."""
        del records
        self.env_ends.append(tuple(env_ids))

    def on_run_end(self, summary: object) -> Mapping[str, JsonValue]:
        """Return a small JSON-safe block counting retained episodes and ends."""
        episodes = len(getattr(summary, "episodes"))
        return {"episodes": episodes, "env_ends": len(self.env_ends)}

    def snapshot(self) -> JsonValue:
        """Return a private JSON-safe copy of the boundary history."""
        return {"env_ends": [list(ids) for ids in self.env_ends]}

    def restore(self, state: JsonValue) -> None:
        """Replace the boundary history from a snapshot's private copy."""
        assert isinstance(state, dict)
        env_ends = state["env_ends"]
        assert isinstance(env_ends, list)
        self.env_ends = [tuple(int(value) for value in ids) for ids in env_ends]


def make_report_extensions() -> tuple[RecordingReportExtension, ...]:
    """Importable zero-argument report-extension factory used by make_sink tests."""
    return (RecordingReportExtension(),)
