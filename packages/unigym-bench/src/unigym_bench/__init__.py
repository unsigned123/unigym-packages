"""External benchmark pack for unigym.

P1 provides version metadata, the core-contract import fence, strict JSON
helpers and the closed task-spec loader in :mod:`unigym_bench.specs`.  P2 adds
the monitor protocol and the built-in monitors in
:mod:`unigym_bench.monitors`.  P3 adds the per-run streaming engine in
:mod:`unigym_bench.engine`.  P4 adds the core sink shell in
:mod:`unigym_bench.sink` and the full immutable episode record in
:mod:`unigym_bench.events`.  P5 adds the ``unigym_bench`` rewind provider on
the streaming engine.  P6 adds the report builder, the report-extension
protocol and the monitor fallback in :mod:`unigym_bench.report`.  P7 adds the
validated command scenario in :mod:`unigym_bench.scenario` and the
rewind-capable scenario sampler in :mod:`unigym_bench.sampler`. P8 adds the
closed-loop convention: a ``scenario`` monitor publishes ``scenario/*`` level
signals through the engine, and the sampler applies a valid desired command at
the next update.  P9 adds the real tiny-arm integration, bounded-retention and
built-in fact-parity tests, plus opt-in report persistence (``write_report``)
to the pack-owned ``external_benchmark_report.json``.  P16 adds the validated
task-document composition layer in :mod:`unigym_bench.compose`: the closed
builtin registry, source/trigger/emit predicate composition, the
scenario-derived coverage extension and the ``task_document`` sink option.
P17 finalized the 0.2.0 release documentation and release evidence. The 0.3.0
release locks the pack to the post-deletion core: no run mode, no episode
clock, no built-in benchmark/``EvaluationSession``, explicit
``runner.terminate``/sink terminals and core ``on_reset`` reopening.
"""

from __future__ import annotations

from .compat import assert_core_contract
from .compose import ComposeError, TaskPlan, compose_task
from .flow import (
    BenchmarkFlow,
    BenchmarkTick,
    DefaultBenchmarkFlow,
    EpisodeBatch,
    EpisodeEvidence,
    EventRule,
    FlowDecision,
    FlowRun,
    RewindableFlow,
    TerminationRule,
    validate_flow,
)
from .json import JsonValue, ensure_json_value
from .version import CORE_SURFACE_VERSION, REQUIRED_CORE_VERSION, __version__

__all__ = [
    "BenchmarkFlow",
    "BenchmarkTick",
    "CORE_SURFACE_VERSION",
    "ComposeError",
    "DefaultBenchmarkFlow",
    "EpisodeBatch",
    "EpisodeEvidence",
    "EventRule",
    "FlowDecision",
    "FlowRun",
    "RewindableFlow",
    "TerminationRule",
    "JsonValue",
    "REQUIRED_CORE_VERSION",
    "TaskPlan",
    "__version__",
    "assert_core_contract",
    "compose_task",
    "ensure_json_value",
    "validate_flow",
]
