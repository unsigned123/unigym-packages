"""Core-contract fence for the external benchmark pack.

The pack builds on the public sink, run, evaluation, value, identity and
termination surfaces of ``unigym``.  It must never reach for the benchmark
scaffolding it replaces (``unigym.benchmarks``, ``unigym.runner.evaluation``,
``unigym.runner.episodes``), so this module checks only the allowed surfaces
and reports every missing one in a single teaching error.

The required surface is the post-deletion core: explicit sink selection
(``sinks:`` / ``CollectSinkSpec`` / ``SinkDeclaration``), the reset-aware sink
capability, and value/evaluation contexts.  There is no core ``mode``, no
episode clock, no ``Wait`` and no ``EvaluationSession`` to depend on.

``unigym.run`` is imported only to verify the public construction face; the P0
plan requires that surface even though importing ``unigym.run`` also loads
torch.  No benchmark scaffolding is imported.
"""

from __future__ import annotations

import importlib

from .version import CORE_SURFACE_VERSION, REQUIRED_CORE_VERSION

__all__ = ["assert_core_contract"]

_REQUIRED_SURFACES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("unigym", ("__version__",)),
    (
        "unigym.sink",
        ("Sink", "TerminalSink", "StoppingSink", "RewindAwareSink", "ResetAwareSink"),
    ),
    (
        "unigym.run",
        (
            "SinkContext",
            "SinkSpec",
            "SinkFactory",
            "CollectSinkSpec",
            "SinkDeclaration",
        ),
    ),
    ("unigym.evaluation", ("EnvView", "EnvStepContext")),
    ("unigym.values", ("ValueStore", "ValueView", "ValuePublisher", "StepValues")),
    ("unigym.core.identity", ("RunIdentity",)),
    (
        "unigym.core.episode_termination",
        (
            "EpisodeOutcome",
            "Termination",
            "TerminationExtension",
            "TerminationSource",
            "validate_code",
        ),
    ),
)


def assert_core_contract() -> None:
    """Fail fast unless the installed ``unigym`` satisfies the pack contract.

    The check verifies the required public surfaces and that the installed core
    version is at least ``REQUIRED_CORE_VERSION``.

    Raises:
        RuntimeError: naming every missing surface and any version mismatch.
            There is no fallback import and no partial success.
    """
    problems: list[str] = []
    core_version: str | None = None
    for module_name, attribute_names in _REQUIRED_SURFACES:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            problems.append(f"{module_name} (not importable: {exc})")
            continue
        if module_name == "unigym":
            version = getattr(module, "__version__", None)
            if isinstance(version, str):
                core_version = version
            elif hasattr(module, "__version__"):
                problems.append(
                    f"unigym.__version__ must be a string, got {type(version).__name__}"
                )
        problems.extend(
            f"{module_name}.{attribute_name}"
            for attribute_name in attribute_names
            if not hasattr(module, attribute_name)
        )
    required_version = _version_tuple(REQUIRED_CORE_VERSION)
    if core_version is not None and required_version is not None:
        installed_version = _version_tuple(core_version)
        if installed_version is None or installed_version < required_version:
            problems.append(
                f"unigym version {core_version} is older than required " f"{REQUIRED_CORE_VERSION}"
            )
    if not problems:
        return
    required = "; ".join(
        f"{module_name}: {', '.join(attribute_names)}"
        for module_name, attribute_names in _REQUIRED_SURFACES
    )
    raise RuntimeError(
        f"unigym-bench core surface contract v{CORE_SURFACE_VERSION} is not "
        f"satisfied: {'; '.join(problems)}. The pack requires unigym "
        f">={REQUIRED_CORE_VERSION} with the public surfaces [{required}]. "
        "Install a compatible unigym; no fallback import or benchmark "
        "scaffolding is used."
    )


def _version_tuple(text: str) -> tuple[int, ...] | None:
    base = text.split("+", 1)[0].split("-", 1)[0]
    parts: list[int] = []
    for part in base.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            return None
    return tuple(parts)
