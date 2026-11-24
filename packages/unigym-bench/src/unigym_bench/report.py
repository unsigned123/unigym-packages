"""Pack-owned run report: base blocks, monitor fallback and extensions.

The report is assembled at run end from facts the pack already owns: the run
metadata supplied at construction, the streaming engine's finalized
:class:`~unigym_bench.events.EpisodeRecord` values and aggregate
:class:`~unigym_bench.events.EpisodeSummary`, each monitor's run-scoped
``report_view`` and each report extension's ``on_run_end`` block.

Layout (fixed top-level blocks, in this order)::

    {
      "schema": "external_benchmark_report/1",
      "run": {...},          # identity + construction metadata
      "tasks": {...},        # task name, monitor names, extension names
      "episodes": {...},     # retention-independent counts + retained detail
      "terminations": {...}, # counters by outcome/code/source
      "execution": {...},    # event total/dropped, conflicts, retention caps
      "monitors": {...},     # fallback: monitor.report_view()
      "extensions": {...}    # custom: <extension>.on_run_end(summary)
    }

Namespacing rules:

- dynamic blocks are namespaced (``monitors/<name>``, ``extensions/<name>``), so
  a dynamic name cannot shadow a base block;
- monitor and extension names are unique within their own namespace; a
  duplicate is a construction error;
- a monitor that declares ``report_extension`` is *not* also emitted under
  ``monitors``.  The declared extension owns that block; there is no silent
  duplicate;
- ``coverage`` belongs to a scenario extension, never to a base block.

Every assembled document passes :func:`unigym_bench.json.ensure_json_value`, so
a tensor, numpy scalar, callable or other non-JSON value fails with the exact
field path instead of being coerced.  Dynamic mapping keys are sorted so the
same inputs produce the same key order.

The report is a pack artifact.  Nothing in this module writes the core
``benchmark_report.json`` or any other core artifact.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, cast, runtime_checkable

from unigym.core.identity import RunIdentity

from .events import EpisodeRecord, EpisodeSummary
from .json import JsonValue, ensure_json_value
from .monitors import Monitor
from .specs import RetentionSpec
from .version import __version__

__all__ = [
    "REPORT_SCHEMA",
    "ReportBuilder",
    "ReportExtension",
    "RunMetadata",
    "RunSummary",
    "validate_report_extensions",
]

#: Schema id of the pack-owned external benchmark report.
REPORT_SCHEMA = "external_benchmark_report/1"

#: The mandatory top-level blocks, in assembly order.
_BASE_BLOCKS: tuple[str, ...] = (
    "schema",
    "run",
    "tasks",
    "episodes",
    "terminations",
    "execution",
    "monitors",
    "extensions",
)

#: Categories aggregated in the ``terminations`` base block.
_TERMINATION_CATEGORIES: tuple[str, ...] = ("outcome", "code", "source")


@runtime_checkable
class ReportExtension(Protocol):
    """Run-scoped projection that contributes one ``extensions/<name>`` block.

    An extension observes finalized episode records through
    :meth:`on_env_end` and returns its report block from :meth:`on_run_end`.
    It never receives a tick frame, an ``EnvView`` or a tensor, so it cannot
    read live boundary data or mutate another environment.

    A run-scoped accumulator is legitimate.  An extension that keeps one must
    declare ``stateful = True`` and implement ``snapshot()``/``restore()`` so
    the sink rewind provider can restore it; an extension with no state must
    declare ``stateful = False``.  The declaration is explicit because a
    forgotten snapshot must not silently drop abandoned-branch state.
    """

    name: str
    stateful: bool

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, EpisodeRecord]) -> None:
        """Observe the due environments and their finalized records.

        The callback is due-only: it fires for the environments that reached a
        runner boundary, never for a still-open episode.  Episodes that remain
        open when the run ends are finalized as ``incomplete`` (or ``rewound``)
        and are visible to :meth:`on_run_end` through
        :attr:`RunSummary.episodes` / :attr:`RunSummary.episode_summary`.

        Args:
            env_ids: The due environment ids for this episode boundary.
            records: A read-only mapping containing exactly ``env_ids``.  The
                records are frozen; reading is allowed, mutation is not.
        """

    def on_run_end(self, summary: RunSummary) -> Mapping[str, JsonValue] | None:
        """Return this extension's JSON-safe report block, or ``None``.

        The block is emitted under ``extensions/<name>``; ``None`` contributes
        no block.  The summary is frozen and JSON-oriented; it carries no views
        or tensors.
        """


@dataclass(frozen=True)
class RunMetadata:
    """Construction-time run facts used by the report's ``run`` block.

    Every field is a plain JSON scalar except ``run_identity``.  A direct
    programmatic caller may leave the text fields empty and ``num_envs`` at
    ``0``; those values are reported honestly rather than inferred.  The pack
    sink fills them from its :class:`unigym.run.SinkContext`.
    """

    run_identity: RunIdentity | None = None
    run_name: str = ""
    task_name: str = ""
    mode: str = ""
    out_dir: str = ""
    seed: int = 0
    num_envs: int = 0
    policy_dt: float = 0.0
    loop_mode: str = "pool"
    pack_version: str = __version__

    def __post_init__(self) -> None:
        """Validate the metadata types and ranges."""
        if self.run_identity is not None and not isinstance(self.run_identity, RunIdentity):
            raise TypeError(
                "run metadata.run_identity: expected a RunIdentity or None, "
                f"got {type(self.run_identity).__name__}"
            )
        for name in ("run_name", "task_name", "mode", "out_dir", "loop_mode", "pack_version"):
            _require_str(getattr(self, name), f"run metadata.{name}")
        _require_int(self.seed, "run metadata.seed")
        _require_int(self.num_envs, "run metadata.num_envs", minimum=0)
        _require_float(self.policy_dt, "run metadata.policy_dt", minimum=0.0)


@dataclass(frozen=True)
class RunSummary:
    """Frozen, JSON-oriented snapshot handed to extension ``on_run_end``.

    The summary exposes run identity/metadata, the retention-independent
    aggregate counters, the retained episode records, every monitor's fallback
    view and the aggregated event/termination facts.  Open episodes that were
    finalized at run end are present in ``episodes``/``episode_summary`` even
    though they never triggered ``on_env_end``.  It holds no views, tensors or
    backend handles.
    """

    metadata: RunMetadata
    episodes: tuple[EpisodeRecord, ...]
    episode_summary: EpisodeSummary
    monitor_views: Mapping[str, Mapping[str, JsonValue]]
    retention: RetentionSpec = field(default_factory=RetentionSpec)
    event_counts: Mapping[str, int] | None = None
    termination_counts: Mapping[str, Mapping[str, int]] | None = None

    def __post_init__(self) -> None:
        """Validate the inputs and freeze the derived aggregates."""
        if not isinstance(self.metadata, RunMetadata):
            raise TypeError(
                "run summary.metadata: expected a RunMetadata, "
                f"got {type(self.metadata).__name__}"
            )
        episodes = tuple(self.episodes)
        for index, record in enumerate(episodes):
            if not isinstance(record, EpisodeRecord):
                raise TypeError(
                    f"run summary.episodes[{index}]: expected an EpisodeRecord, "
                    f"got {type(record).__name__}"
                )
        object.__setattr__(self, "episodes", episodes)
        if not isinstance(self.episode_summary, EpisodeSummary):
            raise TypeError(
                "run summary.episode_summary: expected an EpisodeSummary, "
                f"got {type(self.episode_summary).__name__}"
            )
        if not isinstance(self.retention, RetentionSpec):
            raise TypeError(
                "run summary.retention: expected a RetentionSpec, "
                f"got {type(self.retention).__name__}"
            )
        object.__setattr__(self, "monitor_views", _freeze_monitor_views(self.monitor_views))
        event_counts = (
            _aggregate_event_counts(episodes)
            if self.event_counts is None
            else _validate_count_mapping(self.event_counts, "run summary.event_counts")
        )
        object.__setattr__(
            self, "event_counts", MappingProxyType(dict(sorted(event_counts.items())))
        )
        termination_counts = (
            _termination_counts_from_aggregate(self.episode_summary)
            if self.termination_counts is None
            else _validate_termination_counts(self.termination_counts)
        )
        object.__setattr__(self, "termination_counts", termination_counts)

    @property
    def run_identity(self) -> RunIdentity | None:
        """The run identity recorded at construction, or ``None``."""
        return self.metadata.run_identity

    @property
    def run_name(self) -> str:
        """The configured run name."""
        return self.metadata.run_name

    @property
    def task_name(self) -> str:
        """The task name reported for this run."""
        return self.metadata.task_name

    @property
    def num_envs(self) -> int:
        """The number of environments declared for the run."""
        return self.metadata.num_envs

    @property
    def loop_mode(self) -> str:
        """The pack closed-loop mode selected at construction."""
        return self.metadata.loop_mode


class ReportBuilder:
    """Assemble the pack report from a summary, monitors and extensions.

    The builder owns layout, namespacing, collision detection, JSON validation
    and deterministic ordering.  It never invokes lifecycle callbacks: the sink
    drives extension ``on_env_end``/``on_run_end`` at the correct point and
    passes the returned blocks to :meth:`assemble`.

    Args:
        monitors: Monitor instances in task declaration order.
        extensions: Report extensions in task declaration order.  Names must be
            unique and a stateful extension must implement snapshot/restore.
        monitor_extensions: Optional ``monitor name -> extension name`` map from
            a monitor's declared ``report_extension``.  A declared monitor is
            omitted from the ``monitors`` fallback block.

    Raises:
        TypeError: A monitor/extension is malformed or the stateful contract is
            inconsistent.
        ValueError: A name is duplicated or a declared reference is missing.
    """

    def __init__(
        self,
        *,
        monitors: Sequence[Monitor],
        extensions: Sequence[ReportExtension] = (),
        monitor_extensions: Mapping[str, str] | None = None,
    ) -> None:
        self._monitors = _validate_monitors(monitors)
        self._extensions = validate_report_extensions(extensions)
        self._monitor_extensions = _validate_monitor_extensions(
            monitor_extensions,
            monitor_names=tuple(monitor.name for monitor in self._monitors),
            extension_names=tuple(extension.name for extension in self._extensions),
        )

    def assemble(
        self,
        summary: RunSummary,
        extension_blocks: Mapping[str, Mapping[str, JsonValue] | None],
    ) -> dict[str, JsonValue]:
        """Build and validate the external benchmark report.

        Args:
            summary: The frozen run summary handed to extensions.
            extension_blocks: Each extension's ``on_run_end`` result, keyed by
                extension name.  A ``None`` value contributes no block.

        Returns:
            A fresh JSON-safe report document with a deterministic key order.

        Raises:
            TypeError: The summary or a block has the wrong type.
            ValueError: Extension blocks are missing/extra, a duplicate name is
                present, or the assembled report is not JSON-safe.
        """
        if not isinstance(summary, RunSummary):
            raise TypeError(
                "report builder.assemble: expected a RunSummary, " f"got {type(summary).__name__}"
            )
        blocks = _validate_extension_blocks(extension_blocks, self._extensions)
        covered = set(self._monitor_extensions)
        report: dict[str, JsonValue] = {
            "schema": REPORT_SCHEMA,
            "run": _run_block(summary.metadata),
            "tasks": _tasks_block(summary.metadata, self._monitors, self._extensions),
            "episodes": _episodes_block(summary),
            "terminations": _terminations_block(summary),
            "execution": _execution_block(summary),
            "monitors": _monitors_block(summary, self._monitors, covered),
            "extensions": _extensions_block(blocks, self._extensions),
        }
        if tuple(report) != _BASE_BLOCKS:
            raise ValueError(
                "external benchmark report: internal block order is inconsistent; "
                f"expected {list(_BASE_BLOCKS)}, got {list(report)}"
            )
        normalized = ensure_json_value(report, where="external benchmark report")
        if not isinstance(normalized, dict):
            raise TypeError("external benchmark report: expected an object")
        return normalized


def validate_report_extensions(extensions: object) -> tuple[ReportExtension, ...]:
    """Validate a report-extension sequence and return it as a tuple.

    A valid extension has a non-empty unique ``name``, an explicit boolean
    ``stateful`` declaration and callable ``on_env_end``/``on_run_end``.  A
    stateful extension must implement both ``snapshot()`` and ``restore()``; a
    stateless extension must not implement them, so the declaration and the
    implementation cannot disagree.  The stateless-with-snapshot case is
    rejected rather than ignored: accepting it would let a future stateful
    accumulator silently opt out of the rewind provider.

    Args:
        extensions: The candidate sequence.

    Returns:
        The validated extensions in declaration order.

    Raises:
        TypeError: The sequence or one extension's contract is malformed.
        ValueError: Two extensions share a name.
    """
    if isinstance(extensions, (str, bytes, bytearray)) or not isinstance(extensions, Sequence):
        raise TypeError(
            "report extensions: expected a sequence of ReportExtension instances, "
            f"got {type(extensions).__name__}"
        )
    checked: list[ReportExtension] = []
    names: set[str] = set()
    for extension in extensions:
        name = getattr(extension, "name", None)
        if not isinstance(name, str) or not name:
            raise TypeError(f"report extension.name: expected a non-empty string, got {name!r}")
        if name in names:
            raise ValueError(f"report extensions: duplicate extension name {name!r}")
        names.add(name)
        stateful = getattr(extension, "stateful", None)
        if not isinstance(stateful, bool):
            raise TypeError(
                f"report extension {name!r}.stateful: expected an explicit bool "
                f"declaration, got {stateful!r}"
            )
        for callback in ("on_env_end", "on_run_end"):
            if not callable(getattr(extension, callback, None)):
                raise TypeError(f"report extension {name!r}: missing callable {callback}()")
        has_snapshot = callable(getattr(extension, "snapshot", None))
        has_restore = callable(getattr(extension, "restore", None))
        if stateful:
            if not (has_snapshot and has_restore):
                raise TypeError(
                    f"report extension {name!r}: stateful=True requires callable "
                    "snapshot() and restore()"
                )
        elif has_snapshot or has_restore:
            raise TypeError(
                f"report extension {name!r}: declares stateful=False but implements "
                "snapshot/restore; declare stateful=True or remove the capability"
            )
        # The structural checks above establish the protocol shape; the name and
        # stateful attribute are validated as values, not merely as attributes.
        checked.append(cast(ReportExtension, extension))
    return tuple(checked)


def _validate_monitors(monitors: object) -> tuple[Monitor, ...]:
    if isinstance(monitors, (str, bytes, bytearray)) or not isinstance(monitors, Sequence):
        raise TypeError(
            f"report builder monitors: expected a sequence, got {type(monitors).__name__}"
        )
    checked: list[Monitor] = []
    names: set[str] = set()
    for monitor in monitors:
        name = getattr(monitor, "name", None)
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"report builder monitor.name: expected a non-empty string, got {name!r}"
            )
        if name in names:
            raise ValueError(f"report builder monitors: duplicate monitor name {name!r}")
        names.add(name)
        if not callable(getattr(monitor, "report_view", None)):
            raise TypeError(f"report builder monitor {name!r}: missing callable report_view()")
        checked.append(cast(Monitor, monitor))
    return tuple(checked)


def _validate_monitor_extensions(
    mapping: object,
    *,
    monitor_names: tuple[str, ...],
    extension_names: tuple[str, ...],
) -> Mapping[str, str]:
    if mapping is None:
        return MappingProxyType({})
    if not isinstance(mapping, Mapping):
        raise TypeError(
            "report builder monitor_extensions: expected a mapping of monitor name "
            f"to extension name, got {type(mapping).__name__}"
        )
    known_monitors = set(monitor_names)
    known_extensions = set(extension_names)
    validated: dict[str, str] = {}
    for monitor_name, extension_name in mapping.items():
        if not isinstance(monitor_name, str) or monitor_name not in known_monitors:
            raise ValueError(
                f"report builder monitor_extensions: unknown monitor {monitor_name!r}; "
                f"expected one of {sorted(known_monitors)}"
            )
        if not isinstance(extension_name, str) or extension_name not in known_extensions:
            raise ValueError(
                f"report builder monitor_extensions[{monitor_name!r}]: unknown extension "
                f"{extension_name!r}; expected one of {sorted(known_extensions)}"
            )
        validated[monitor_name] = extension_name
    return MappingProxyType(dict(sorted(validated.items())))


def _validate_extension_blocks(
    blocks: object,
    extensions: tuple[ReportExtension, ...],
) -> dict[str, Mapping[str, JsonValue] | None]:
    if not isinstance(blocks, Mapping):
        raise TypeError(
            "report builder extension_blocks: expected a mapping keyed by extension "
            f"name, got {type(blocks).__name__}"
        )
    expected = {extension.name for extension in extensions}
    provided = set(blocks)
    missing = sorted(expected - provided)
    extra = sorted(provided - expected)
    if missing or extra:
        raise ValueError(
            "report builder extension_blocks: expected exactly "
            f"{sorted(expected)}, missing {missing}, extra {extra}"
        )
    validated: dict[str, Mapping[str, JsonValue] | None] = {}
    for name, block in blocks.items():
        if block is None:
            validated[name] = None
            continue
        if not isinstance(block, Mapping):
            raise TypeError(
                f"report extension {name!r}.on_run_end: expected a mapping or None, "
                f"got {type(block).__name__}"
            )
        validated[name] = block
    return validated


def _run_block(metadata: RunMetadata) -> dict[str, JsonValue]:
    identity = None if metadata.run_identity is None else metadata.run_identity.to_json()
    return {
        "identity": identity,
        "run_name": metadata.run_name,
        "task_name": metadata.task_name,
        "mode": metadata.mode,
        "out_dir": metadata.out_dir,
        "seed": metadata.seed,
        "num_envs": metadata.num_envs,
        "policy_dt": metadata.policy_dt,
        "loop_mode": metadata.loop_mode,
        "pack_version": metadata.pack_version,
    }


def _tasks_block(
    metadata: RunMetadata,
    monitors: tuple[Monitor, ...],
    extensions: tuple[ReportExtension, ...],
) -> dict[str, JsonValue]:
    return {
        "name": metadata.task_name,
        "monitors": [monitor.name for monitor in monitors],
        "extensions": [extension.name for extension in extensions],
    }


def _episodes_block(summary: RunSummary) -> dict[str, JsonValue]:
    aggregate = summary.episode_summary
    retention = summary.retention
    verdicts = dict(sorted(aggregate.verdict_counts.items()))
    incomplete = verdicts.get("incomplete", 0)
    rewound = aggregate.rewound
    complete = aggregate.episodes - incomplete - rewound
    block: dict[str, JsonValue] = {
        "count": aggregate.episodes,
        "complete": complete,
        "incomplete": incomplete,
        "rewound": rewound,
        "retained": len(summary.episodes),
        "retain_episodes": retention.retain_episodes,
        "total_ticks": aggregate.total_ticks,
        "verdicts": verdicts,
        "by_env_counts": {
            str(env_id): count for env_id, count in sorted(aggregate.episodes_by_env.items())
        },
    }
    if retention.retain_episodes:
        # Per-record detail is available only when the run retained episodes; a
        # report must not fabricate an empty length/by_env block for a run that
        # kept only counters.
        ticks = [record.ticks for record in summary.episodes]
        block["length"] = {
            "min": min(ticks) if ticks else None,
            "max": max(ticks) if ticks else None,
            "mean": (sum(ticks) / len(ticks)) if ticks else None,
        }
        block["by_env"] = _episodes_by_env(summary.episodes)
    return block


def _episodes_by_env(records: tuple[EpisodeRecord, ...]) -> dict[str, JsonValue]:
    grouped: dict[int, list[EpisodeRecord]] = {}
    for record in records:
        grouped.setdefault(record.env_id, []).append(record)
    by_env: dict[str, JsonValue] = {}
    for env_id in sorted(grouped):
        env_records = grouped[env_id]
        verdicts: dict[str, int] = {}
        for record in env_records:
            verdicts[record.verdict] = verdicts.get(record.verdict, 0) + 1
        by_env[str(env_id)] = {
            "count": len(env_records),
            "total_ticks": sum(record.ticks for record in env_records),
            "incomplete": sum(1 for record in env_records if record.verdict == "incomplete"),
            "rewound": sum(1 for record in env_records if record.rewound),
            "verdicts": dict(sorted(verdicts.items())),
        }
    return by_env


def _terminations_block(summary: RunSummary) -> dict[str, JsonValue]:
    counts = summary.termination_counts or {}
    outcome = dict(sorted(counts.get("outcome", {}).items()))
    code = dict(sorted(counts.get("code", {}).items()))
    source = dict(sorted(counts.get("source", {}).items()))
    return {
        "total": sum(outcome.values()),
        "outcome": outcome,
        "code": code,
        "source": source,
    }


def _execution_block(summary: RunSummary) -> dict[str, JsonValue]:
    aggregate = summary.episode_summary
    retention = summary.retention
    return {
        "event_total": aggregate.event_count,
        "dropped_events": aggregate.dropped_events,
        "terminal_conflicts": aggregate.conflicts,
        "retention": {
            "retain_episodes": retention.retain_episodes,
            "max_events": retention.max_events,
            "event_log": retention.event_log,
        },
    }


def _monitors_block(
    summary: RunSummary,
    monitors: tuple[Monitor, ...],
    covered: set[str],
) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for monitor in monitors:
        if monitor.name in covered:
            continue
        view = summary.monitor_views.get(monitor.name)
        if view is None:
            raise ValueError(
                f"external benchmark report.monitors: run summary has no report_view for "
                f"monitor {monitor.name!r}"
            )
        normalized = ensure_json_value(dict(view), where=f"monitors.{monitor.name}")
        if not isinstance(normalized, dict):
            raise TypeError(
                f"external benchmark report.monitors[{monitor.name!r}]: expected an object"
            )
        result[monitor.name] = normalized
    return dict(sorted(result.items()))


def _extensions_block(
    blocks: Mapping[str, Mapping[str, JsonValue] | None],
    extensions: tuple[ReportExtension, ...],
) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for extension in extensions:
        block = blocks[extension.name]
        if block is None:
            continue
        normalized = ensure_json_value(dict(block), where=f"extensions.{extension.name}")
        if not isinstance(normalized, dict):
            raise TypeError(
                f"external benchmark report.extensions[{extension.name!r}]: " "expected an object"
            )
        result[extension.name] = normalized
    return dict(sorted(result.items()))


def _freeze_monitor_views(
    views: object,
) -> Mapping[str, Mapping[str, JsonValue]]:
    if not isinstance(views, Mapping):
        raise TypeError(
            "run summary.monitor_views: expected a mapping, " f"got {type(views).__name__}"
        )
    frozen: dict[str, Mapping[str, JsonValue]] = {}
    for name, view in views.items():
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"run summary.monitor_views key must be a non-empty string, got {name!r}"
            )
        if not isinstance(view, Mapping):
            raise TypeError(
                f"run summary.monitor_views[{name!r}]: expected a mapping, "
                f"got {type(view).__name__}"
            )
        normalized = ensure_json_value(dict(view), where=f"run summary.monitor_views[{name!r}]")
        if not isinstance(normalized, dict):
            raise TypeError(f"run summary.monitor_views[{name!r}]: expected an object")
        frozen[name] = MappingProxyType(normalized)
    return MappingProxyType(dict(sorted(frozen.items())))


def _aggregate_event_counts(episodes: tuple[EpisodeRecord, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in episodes:
        for code, count in record.event_counts.items():
            counts[code] = counts.get(code, 0) + count
    return counts


def _termination_counts_from_aggregate(
    aggregate: EpisodeSummary,
) -> Mapping[str, Mapping[str, int]]:
    """Build termination counts from the engine's retention-independent counters."""
    return MappingProxyType(
        {
            "outcome": MappingProxyType(dict(sorted(aggregate.termination_outcome_counts.items()))),
            "code": MappingProxyType(dict(sorted(aggregate.termination_code_counts.items()))),
            "source": MappingProxyType(dict(sorted(aggregate.termination_source_counts.items()))),
        }
    )


def _validate_count_mapping(value: object, where: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{where}: expected a mapping, got {type(value).__name__}")
    counts: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError(f"{where}: keys must be non-empty strings, got {key!r}")
        _require_int(count, f"{where}[{key!r}]", minimum=0)
        counts[key] = count
    return counts


def _validate_termination_counts(
    value: object,
) -> Mapping[str, Mapping[str, int]]:
    if not isinstance(value, Mapping):
        raise TypeError(
            "run summary.termination_counts: expected a mapping, " f"got {type(value).__name__}"
        )
    unknown = sorted(set(value) - set(_TERMINATION_CATEGORIES))
    if unknown:
        raise ValueError(
            "run summary.termination_counts: unknown categories "
            f"{unknown}; expected only {sorted(_TERMINATION_CATEGORIES)}"
        )
    validated: dict[str, Mapping[str, int]] = {}
    for category in _TERMINATION_CATEGORIES:
        counts = _validate_count_mapping(
            value.get(category, {}), f"run summary.termination_counts[{category!r}]"
        )
        validated[category] = MappingProxyType(dict(sorted(counts.items())))
    return MappingProxyType(validated)


def _require_str(value: object, where: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{where}: expected a string, got {value!r}")


def _require_int(value: object, where: str, *, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where}: expected an int, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{where}: expected an int >= {minimum}, got {value}")


def _require_float(value: object, where: str, *, minimum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{where}: expected a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{where}: expected a finite number, got {number!r}")
    if number < minimum:
        raise ValueError(f"{where}: expected a number >= {minimum}, got {number}")
