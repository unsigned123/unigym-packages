"""External benchmark pack for unigym.

P1 provides version metadata, the core-contract import fence, strict JSON
helpers and the closed task-spec loader in :mod:`unigym_bench.specs`.  No
benchmark engine, monitors, sink, report or sampler code exists yet.
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
