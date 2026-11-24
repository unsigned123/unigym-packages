"""Construction helpers for monitor tests: real frames and simple probes.

These are test inputs, not stand-ins for pack components: every frame is built
from the real :class:`unigym_bench.frames.TickFrame`, and every probe is a
small callable that returns a real tensor for the frame's device.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import torch

from unigym.core.episode_termination import (
    EpisodeOutcome,
    Termination,
    TerminationSource,
)
from unigym.evaluation import EnvView
from unigym_bench.events import Event
from unigym_bench.frames import TickFrame
from unigym_bench.json import JsonValue
from unigym_bench.monitors import BaseMonitor


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
