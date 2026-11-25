"""Pack-owned run report: base blocks, monitor fallback and extensions.

The report is assembled at run end from facts the pack already owns: the run
metadata supplied at construction, the streaming engine's finalized
:class:`~unigym_bench.events.EpisodeRecord` values and aggregate
:class:`~unigym_bench.events.EpisodeSummary`, each monitor's run-scoped
``report_view`` and each report extension's ``on_run_end`` block.

Layout (fixed top-level blocks, in this order)::

    {
      "schema": "external_benchmark_report/3",
      "run": {...},          # identity + construction metadata
      "tasks": {...},        # task name, monitor names, extension names
      "episodes": {...},     # counts, retained detail and episode evidence
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
import re
import struct
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
    "REPORT_BLOCKS",
    "REPORT_FILENAME",
    "REPORT_SCHEMA",
    "ReportBuilder",
    "ReportExtension",
    "RunMetadata",
    "RunSummary",
    "ScenarioCoverageExtension",
    "validate_report_configuration",
    "validate_report_extensions",
]

#: Schema id of the pack-owned external benchmark report.
#:
#: Schema 3 moves evidence under ``episodes.evidence`` and makes monitor report
#: views flow-selected.  The obsolete ``run.mode`` field stays removed: the
#: post-deletion core has no run mode, and the pack must not fabricate one.
#: ``run.loop_mode`` remains the pack's own closed-loop selection.
REPORT_SCHEMA = "external_benchmark_report/3"

#: Pack-owned report filename written under a run's ``out_dir``.
#: It can never collide with the core ``benchmark_report.json`` / ``results.json``
#: artifacts because the name is fixed and pack-specific.
REPORT_FILENAME = "external_benchmark_report.json"

#: The mandatory top-level blocks, in assembly order.
REPORT_BLOCKS: tuple[str, ...] = (
    "schema",
    "run",
    "tasks",
    "episodes",
    "terminations",
    "execution",
    "monitors",
    "extensions",
)

_BASE_BLOCKS = REPORT_BLOCKS

#: Categories aggregated in the ``terminations`` base block.
_TERMINATION_CATEGORIES: tuple[str, ...] = ("outcome", "code", "source")

#: Schema id of :class:`ScenarioCoverageExtension`'s private rewind snapshot.
_SCENARIO_COVERAGE_SNAPSHOT_SCHEMA = "unigym_bench.scenario_coverage/1"

#: Extension names follow the monitor-name grammar.
_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")


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
        for name in ("run_name", "task_name", "out_dir", "loop_mode", "pack_version"):
            _require_str(getattr(self, name), f"run metadata.{name}")
        _require_int(self.seed, "run metadata.seed")
        _require_int(self.num_envs, "run metadata.num_envs", minimum=0)
        _require_float(self.policy_dt, "run metadata.policy_dt", minimum=0.0)


@dataclass(frozen=True)
class RunSummary:
    """Frozen, JSON-oriented snapshot handed to extension ``on_run_end``.

    The summary exposes run identity/metadata, the retention-independent
    aggregate counters, the retained episode records, every monitor's fallback
    view, finalized episode evidence and the aggregated event/termination
    facts.  Open episodes that were finalized at run end are present in
    ``episodes``/``episode_summary`` even though they never triggered
    ``on_env_end``.  It holds no views, tensors or backend handles.
    """

    metadata: RunMetadata
    episodes: tuple[EpisodeRecord, ...]
    episode_summary: EpisodeSummary
    monitor_views: Mapping[str, Mapping[str, JsonValue]]
    evidence: Mapping[tuple[int, int], Mapping[str, JsonValue]] = field(
        default_factory=lambda: MappingProxyType({})
    )
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
        object.__setattr__(self, "evidence", _freeze_episode_evidence(self.evidence))
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


class ScenarioCoverageExtension:
    """Stateful report extension that tracks scenario command coverage.

    The extension observes each finalized episode's
    :attr:`~unigym_bench.events.EpisodeRecord.command_at_start` through
    :meth:`on_env_end`; it never receives a tick frame, view or tensor.  The
    command is matched against a validated command pool with exact ``float32``
    quantization, so the sampler remains the only owner of command
    quantization.  The accumulator is JSON-safe and participates in the pack
    rewind provider through :meth:`snapshot`/:meth:`restore`.

    Args:
        name: Extension name matching ``[a-z][a-z0-9_]*``.
        commands: Non-empty sequence of equal-length finite command vectors.

    Raises:
        TypeError: ``name`` or ``commands`` has the wrong type.
        ValueError: The name is malformed, the pool is empty, a row has a
            different length, a value is non-finite after ``float32``
            quantization, or two rows are duplicates.
    """

    stateful = True

    def __init__(self, name: str, *, commands: Sequence[Sequence[float]]) -> None:
        self.name = _require_extension_name(name)
        self._commands = _require_command_pool(commands, f"report extension {name!r}.commands")
        self._dim = len(self._commands[0])
        self._index_by_row = {row: index for index, row in enumerate(self._commands)}
        self._covered = [False] * len(self._commands)
        self._episode_counts = [0] * len(self._commands)
        self._unmatched = 0
        self._episodes = 0

    def on_env_end(self, env_ids: Sequence[int], records: Mapping[int, EpisodeRecord]) -> None:
        """Match each due environment's start command against the pool.

        Args:
            env_ids: The due environment ids.
            records: A read-only mapping containing exactly ``env_ids``.

        Raises:
            TypeError: ``records`` is not a mapping, a record is not an
                ``EpisodeRecord``, or a record is missing for a due env.
            ValueError: An env id is invalid, the mapping has extra ids, or a
                start command is not representable as a finite float32 row.
        """
        ids = _validate_env_ids(env_ids)
        if not isinstance(records, Mapping):
            raise TypeError(
                "scenario coverage.on_env_end.records: expected a mapping, "
                f"got {type(records).__name__}"
            )
        missing = [env_id for env_id in ids if env_id not in records]
        extra = sorted(set(records) - set(ids))
        if missing or extra:
            raise ValueError(
                "scenario coverage.on_env_end.records: expected exactly "
                f"{list(ids)}, missing {missing}, extra {extra}"
            )
        for env_id in ids:
            record = records[env_id]
            if not isinstance(record, EpisodeRecord):
                raise TypeError(
                    f"scenario coverage.on_env_end.records[{env_id}]: expected an "
                    f"EpisodeRecord, got {type(record).__name__}"
                )
            self._episodes += 1
            command = record.command_at_start
            if command is None or len(command) != self._dim:
                self._unmatched += 1
                continue
            row = _quantize_float32_row(command, f"scenario coverage episode {env_id}")
            index = self._index_by_row.get(row)
            if index is None:
                self._unmatched += 1
                continue
            self._covered[index] = True
            self._episode_counts[index] += 1

    def on_run_end(self, summary: RunSummary) -> Mapping[str, JsonValue]:
        """Return the same coverage facts the retired monitor reported."""
        del summary
        covered = sum(self._covered)
        pool_size = len(self._commands)
        return {
            "name": self.name,
            "kind": "scenario_coverage",
            "pool_size": pool_size,
            "covered": covered,
            "coverage_ratio": covered / pool_size,
            "episode_counts": list(self._episode_counts),
            "unmatched_episodes": self._unmatched,
            "episodes": self._episodes,
        }

    def snapshot(self) -> JsonValue:
        """Return a JSON-safe private copy of the coverage accumulator."""
        return {
            "schema": _SCENARIO_COVERAGE_SNAPSHOT_SCHEMA,
            "name": self.name,
            "commands": [list(row) for row in self._commands],
            "covered": list(self._covered),
            "episode_counts": list(self._episode_counts),
            "unmatched": self._unmatched,
            "episodes": self._episodes,
        }

    def validate_restore(self, state: JsonValue) -> None:
        """Pre-flight a snapshot without mutating the accumulator."""
        self._decode_snapshot(state)

    def restore(self, state: JsonValue) -> None:
        """Replace the accumulator from a snapshot's private copy."""
        decoded = self._decode_snapshot(state)
        covered, episode_counts, unmatched, episodes = decoded
        self._covered = covered
        self._episode_counts = episode_counts
        self._unmatched = unmatched
        self._episodes = episodes

    def _decode_snapshot(self, state: JsonValue) -> tuple[list[bool], list[int], int, int]:
        payload = ensure_json_value(state, where=f"scenario coverage {self.name!r} restore")
        if not isinstance(payload, dict):
            raise TypeError(
                f"scenario coverage {self.name!r} restore: expected an object, "
                f"got {type(payload).__name__}"
            )
        expected = {
            "schema",
            "name",
            "commands",
            "covered",
            "episode_counts",
            "unmatched",
            "episodes",
        }
        missing = sorted(expected - set(payload))
        unknown = sorted(set(payload) - expected)
        if missing or unknown:
            raise ValueError(
                f"scenario coverage {self.name!r} restore: expected keys "
                f"{sorted(expected)}, missing {missing}, unknown {unknown}"
            )
        if payload["schema"] != _SCENARIO_COVERAGE_SNAPSHOT_SCHEMA:
            raise ValueError(
                f"scenario coverage {self.name!r} restore: expected schema "
                f"{_SCENARIO_COVERAGE_SNAPSHOT_SCHEMA!r}, got {payload['schema']!r}"
            )
        if payload["name"] != self.name:
            raise ValueError(
                f"scenario coverage {self.name!r} restore: snapshot names "
                f"extension {payload['name']!r}"
            )
        pool_size = len(self._commands)
        if payload["commands"] != [list(row) for row in self._commands]:
            raise ValueError(
                f"scenario coverage {self.name!r} restore: snapshot command pool "
                "does not match the extension configuration"
            )
        covered = _decode_bool_list(payload["covered"], pool_size, "covered")
        episode_counts = _decode_int_list(payload["episode_counts"], pool_size, "episode_counts")
        unmatched = _decode_nonnegative_int(payload["unmatched"], "unmatched")
        episodes = _decode_nonnegative_int(payload["episodes"], "episodes")
        return covered, episode_counts, unmatched, episodes


class ReportBuilder:
    """Assemble the pack report from a summary, monitors and extensions.

    The builder owns layout, namespacing, collision detection, JSON validation
    and deterministic ordering.  It never invokes lifecycle callbacks: the flow
    drives extension ``on_env_end``/``on_run_end`` at the correct point and
    passes the returned blocks to :meth:`assemble`.

    Args:
        monitors: Selected report monitor sources in declaration order.  A
            monitor view absent from the run summary is omitted, not an error.
        extensions: Selected report extensions in declaration order.  Names must
            be unique and a stateful extension must implement snapshot/restore.
        monitor_extensions: Optional ``monitor name -> extension name`` map from
            a monitor's declared ``report_extension``.  A declared monitor is
            omitted from the ``monitors`` fallback block.
        task_monitors: All task monitor names for the ``tasks.monitors`` block.
            Defaults to the selected monitor names.
        task_extensions: All declared extension names for the
            ``tasks.extensions`` block.  Defaults to the selected extension names.

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
        task_monitors: Sequence[str] | None = None,
        task_extensions: Sequence[str] | None = None,
    ) -> None:
        self._monitors = _validate_monitors(monitors)
        self._extensions = validate_report_extensions(extensions)
        monitor_names = tuple(monitor.name for monitor in self._monitors)
        extension_names = tuple(extension.name for extension in self._extensions)
        self._task_monitors = _validate_task_names(
            task_monitors,
            default=monitor_names,
            where="report builder task_monitors",
        )
        self._task_extensions = _validate_task_names(
            task_extensions,
            default=extension_names,
            where="report builder task_extensions",
        )
        unknown_monitors = sorted(set(monitor_names) - set(self._task_monitors))
        if unknown_monitors:
            raise ValueError(
                "report builder monitors: selected monitor(s) are not declared in "
                f"task_monitors: {unknown_monitors}"
            )
        unknown_extensions = sorted(set(extension_names) - set(self._task_extensions))
        if unknown_extensions:
            raise ValueError(
                "report builder extensions: selected extension(s) are not declared in "
                f"task_extensions: {unknown_extensions}"
            )
        self._monitor_extensions = _validate_monitor_extensions(
            monitor_extensions,
            monitor_names=self._task_monitors,
            extension_names=self._task_extensions,
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
            "tasks": _tasks_block(
                summary.metadata,
                self._task_monitors,
                self._task_extensions,
            ),
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


def validate_report_configuration(
    *,
    monitors: Sequence[Monitor],
    extensions: Sequence[ReportExtension] = (),
    monitor_extensions: Mapping[str, str] | None = None,
) -> None:
    """Validate report composition without retaining live monitor objects.

    Args:
        monitors: Monitor instances in task declaration order.
        extensions: Report extensions in task declaration order.
        monitor_extensions: Optional monitor-name to extension-name map.

    Raises:
        TypeError: A monitor/extension is malformed or the stateful contract is
            inconsistent.
        ValueError: A name is duplicated or a declared reference is missing.
    """
    ReportBuilder(
        monitors=monitors,
        extensions=extensions,
        monitor_extensions=monitor_extensions,
    )


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
        "out_dir": metadata.out_dir,
        "seed": metadata.seed,
        "num_envs": metadata.num_envs,
        "policy_dt": metadata.policy_dt,
        "loop_mode": metadata.loop_mode,
        "pack_version": metadata.pack_version,
    }


def _tasks_block(
    metadata: RunMetadata,
    monitor_names: tuple[str, ...],
    extension_names: tuple[str, ...],
) -> dict[str, JsonValue]:
    return {
        "name": metadata.task_name,
        "monitors": list(monitor_names),
        "extensions": list(extension_names),
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
    evidence = {
        f"{env_id}/{episode_seq}": dict(payload)
        for (env_id, episode_seq), payload in sorted(summary.evidence.items())
        if payload
    }
    if evidence:
        block["evidence"] = evidence
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
            # The flow owns view retention: an omitted view is absent, not an
            # error.  Unknown selected names are rejected by the flow selection.
            continue
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


def _validate_task_names(
    value: Sequence[str] | None,
    *,
    default: tuple[str, ...],
    where: str,
) -> tuple[str, ...]:
    """Validate the full task declaration names for a report block."""
    if value is None:
        return default
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(
            f"{where}: expected a sequence of names or None, got {type(value).__name__}"
        )
    names: list[str] = []
    for name in value:
        if not isinstance(name, str) or not name:
            raise TypeError(f"{where}: expected non-empty strings, got {name!r}")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError(f"{where}: duplicate names {names!r}")
    return tuple(names)


def _freeze_episode_evidence(
    evidence: object,
) -> Mapping[tuple[int, int], Mapping[str, JsonValue]]:
    """Validate and freeze episode evidence keyed by ``(env_id, episode_seq)``."""
    if not isinstance(evidence, Mapping):
        raise TypeError(f"run summary.evidence: expected a mapping, got {type(evidence).__name__}")
    frozen: dict[tuple[int, int], Mapping[str, JsonValue]] = {}
    for key, payload in evidence.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or any(isinstance(part, bool) or not isinstance(part, int) or part < 0 for part in key)
        ):
            raise TypeError(
                "run summary.evidence: keys must be (env_id, episode_seq) non-negative "
                f"int tuples, got {key!r}"
            )
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"run summary.evidence[{key[0]}/{key[1]}]: expected a mapping, "
                f"got {type(payload).__name__}"
            )
        normalized = ensure_json_value(
            dict(payload), where=f"run summary.evidence[{key[0]}/{key[1]}]"
        )
        if not isinstance(normalized, dict):
            raise TypeError(f"run summary.evidence[{key[0]}/{key[1]}]: expected an object")
        frozen[(key[0], key[1])] = MappingProxyType(normalized)
    return MappingProxyType(dict(sorted(frozen.items())))


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


def _require_extension_name(value: object) -> str:
    if not isinstance(value, str) or not _NAME_PATTERN.fullmatch(value):
        raise ValueError(f"scenario coverage.name: expected [a-z][a-z0-9_]*, got {value!r}")
    return value


def _require_command_pool(value: object, where: str) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(
            f"{where}: expected a sequence of command vectors, got {type(value).__name__}"
        )
    rows = list(value)
    if not rows:
        raise ValueError(f"{where}: expected at least one command vector")
    commands: list[tuple[float, ...]] = []
    dimension: int | None = None
    for row_index, row in enumerate(rows):
        row_where = f"{where}[{row_index}]"
        if isinstance(row, (str, bytes, bytearray)) or not isinstance(row, Sequence):
            raise TypeError(f"{row_where}: expected a command vector, got {type(row).__name__}")
        numbers: list[float] = []
        for item_index, item in enumerate(row):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError(
                    f"{row_where}[{item_index}]: expected a finite number, got {item!r}"
                )
            number = float(item)
            if not math.isfinite(number):
                raise ValueError(
                    f"{row_where}[{item_index}]: expected a finite number, got {item!r}"
                )
            numbers.append(number)
        if dimension is None:
            dimension = len(numbers)
            if dimension == 0:
                raise ValueError(f"{row_where}: command vector must not be empty")
        elif len(numbers) != dimension:
            raise ValueError(
                f"{row_where}: expected {dimension} values to match {where}[0], "
                f"got {len(numbers)}"
            )
        commands.append(_quantize_float32_row(numbers, row_where))
    if len(set(commands)) != len(commands):
        raise ValueError(f"{where}: expected unique command vectors, got duplicates")
    return tuple(commands)


def _quantize_float32_row(values: Sequence[float], where: str) -> tuple[float, ...]:
    """Quantize one command row to exact float32 Python values."""
    row: list[float] = []
    for index, value in enumerate(values):
        number = _quantize_float32(float(value), f"{where}[{index}]")
        row.append(number)
    return tuple(row)


def _quantize_float32(number: float, where: str) -> float:
    """Return the float32-quantized value without importing a tensor library."""
    try:
        packed = struct.pack("<f", number)
    except (OverflowError, struct.error) as exc:
        raise ValueError(f"{where}: value {number!r} is not representable as float32") from exc
    quantized = struct.unpack("<f", packed)[0]
    if not math.isfinite(quantized):
        raise ValueError(f"{where}: value {number!r} is not finite after float32 quantization")
    return quantized


def _validate_env_ids(env_ids: object) -> tuple[int, ...]:
    if isinstance(env_ids, (str, bytes, bytearray)) or not isinstance(env_ids, Sequence):
        raise TypeError(
            f"scenario coverage.on_env_end.env_ids: expected a sequence of ints, "
            f"got {type(env_ids).__name__}"
        )
    ids: list[int] = []
    for index, env_id in enumerate(env_ids):
        if isinstance(env_id, bool) or not isinstance(env_id, int):
            raise TypeError(
                f"scenario coverage.on_env_end.env_ids[{index}]: expected an int, "
                f"got {env_id!r}"
            )
        if env_id < 0:
            raise ValueError(
                f"scenario coverage.on_env_end.env_ids[{index}]: expected an id >= 0, "
                f"got {env_id}"
            )
        ids.append(env_id)
    if len(set(ids)) != len(ids):
        raise ValueError(
            "scenario coverage.on_env_end.env_ids: expected unique ids, " f"got {tuple(ids)}"
        )
    return tuple(ids)


def _decode_bool_list(value: JsonValue, expected: int, field: str) -> list[bool]:
    items = _decode_state_list(value, field)
    if len(items) != expected:
        raise ValueError(f"{field}: expected {expected} values, got {len(items)}")
    result: list[bool] = []
    for index, item in enumerate(items):
        if type(item) is not bool:
            raise TypeError(f"{field}[{index}]: expected a bool, got {item!r}")
        result.append(item)
    return result


def _decode_int_list(value: JsonValue, expected: int, field: str) -> list[int]:
    items = _decode_state_list(value, field)
    if len(items) != expected:
        raise ValueError(f"{field}: expected {expected} values, got {len(items)}")
    result: list[int] = []
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{field}[{index}]: expected an int, got {item!r}")
        if item < 0:
            raise ValueError(f"{field}[{index}]: expected an int >= 0, got {item}")
        result.append(item)
    return result


def _decode_nonnegative_int(value: JsonValue, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field}: expected an int, got {value!r}")
    if value < 0:
        raise ValueError(f"{field}: expected an int >= 0, got {value}")
    return value


def _decode_state_list(value: JsonValue, field: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise TypeError(f"{field}: expected a list, got {type(value).__name__}")
    return value


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
