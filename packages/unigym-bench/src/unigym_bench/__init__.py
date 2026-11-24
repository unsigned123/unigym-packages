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
to the pack-owned ``external_benchmark_report.json``.
"""

from __future__ import annotations

from .compat import assert_core_contract
from .json import JsonValue, ensure_json_value
from .version import CORE_SURFACE_VERSION, REQUIRED_CORE_VERSION, __version__

__all__ = [
    "CORE_SURFACE_VERSION",
    "JsonValue",
    "REQUIRED_CORE_VERSION",
    "__version__",
    "assert_core_contract",
    "ensure_json_value",
]
