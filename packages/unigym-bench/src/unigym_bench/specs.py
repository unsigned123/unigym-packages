"""Closed task-document schema for the external benchmark pack.

A task document is a single YAML task: a task name, a mapping of monitors, a
mapping of report extensions, the run retention policy, an optional scenario and
the task flow rules.  This module owns the authoring surface.  It validates a document at
the boundary where it enters the pack and returns frozen, read-only specs; it
does not import monitors, extensions, samplers or the streaming engine.

The schema is closed:

- unknown keys are rejected at every level, and an explicit ``null`` for an
  object field is rejected rather than treated as omission; omit the key to
  use the documented default;
- ``schema`` is optional for authoring; when present it must be
  ``unigym_extbench/3``.  Absent and explicit null are distinct; the retired
  ``unigym_extbench/1`` and ``/2`` forms are rejected with a migration error
  (there is no dual-schema shim);
- monitor and extension names are the mapping keys and must match
  ``[a-z][a-z0-9_]*``; duplicate YAML keys are rejected, not silently merged;
- a monitor declares exactly one of ``builtin``, ``module`` or an inline
  ``factory`` callable; an extension declares exactly one of ``builtin`` or
  ``module``;
- a ``builtin: predicate`` monitor composes its condition from ``when`` and
  its trigger/value specs from ``trigger``/``emit``; ``emit`` accepts only the
  local ``value`` name.  A ``builtin: phase`` monitor reads its explicit
  ``source`` and boundary params.  Predicate fields are rejected on
  module/factory declarations and on builtins that do not consume them;
- the optional ``flow`` section owns terminal and event decisions.  Its
  ``termination_rules`` name a monitor boolean value, an outcome and a
  ``custom.*`` code; its ``events`` name a monitor boolean value and an event.
  A document without ``flow`` has no automatic termination and no automatic
  events;
- ``params`` is a JSON object; it is validated with
  :func:`unigym_bench.json.ensure_json_value` and deep-frozen;
- ``module`` references use the ``pkg.mod:attr`` form and are checked for
  syntax only (imports happen later, at the composition boundary);
- relative resource paths resolve against the task document directory, never
  the process working directory.

Every failure raises :class:`SpecError` naming the field and the
expected/actual value.  :func:`load_task_document` prefixes the document path.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias, cast

import yaml
from unigym.core.episode_termination import EpisodeOutcome, validate_code

from .conditions import ConditionError, parse_condition
from .events import SEVERITIES, Severity
from .json import JsonScalar, JsonValue, ensure_json_value
from .predicates import EmitSpec, TriggerSpec
from .sources import SourceError, SourceRef

__all__ = [
    "SCHEMA",
    "EventRuleSpec",
    "ExtensionSpec",
    "FlowSpec",
    "FrozenJson",
    "MonitorSpec",
    "RetentionSpec",
    "ScenarioSpec",
    "SpecError",
    "TaskDocument",
    "TaskSpec",
    "TerminationRuleSpec",
    "load_task_document",
    "parse_task_document",
]

#: Document schema understood by this loader.
SCHEMA = "unigym_extbench/3"
_RETIRED_SCHEMAS: frozenset[str] = frozenset({"unigym_extbench/1", "unigym_extbench/2"})
_MONITOR_OUTCOMES: frozenset[EpisodeOutcome] = frozenset(
    {EpisodeOutcome.SUCCESS, EpisodeOutcome.FAILURE, EpisodeOutcome.DEADLINE}
)

#: Read-only counterpart of :data:`unigym_bench.json.JsonValue`.  Mappings
#: become ``MappingProxyType`` and lists become tuples so a parsed spec cannot
#: be mutated by a consumer or aliased to the source document.
FrozenJson: TypeAlias = JsonScalar | tuple["FrozenJson", ...] | Mapping[str, "FrozenJson"]

_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_MODULE_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*"
)

_MISSING = object()

_DOCUMENT_KEYS = frozenset(
    {"schema", "name", "monitors", "extensions", "retention", "scenario", "flow"}
)
_MONITOR_KEYS = frozenset(
    {
        "builtin",
        "module",
        "factory",
        "params",
        "report_extension",
        "resources",
        "when",
        "trigger",
        "emit",
        "source",
    }
)
_EXTENSION_KEYS = frozenset({"builtin", "module", "params", "resources"})
_RETENTION_KEYS = frozenset({"retain_episodes", "max_events", "event_log"})
_SCENARIO_KEYS = frozenset({"command_pool", "plan"})
_FLOW_KEYS = frozenset({"termination_rules", "events"})
_TERMINATION_RULE_KEYS = frozenset({"monitor", "value", "outcome", "code"})
_EVENT_RULE_KEYS = frozenset({"monitor", "value", "name", "severity", "code", "data"})


class SpecError(ValueError):
    """A task document does not satisfy the closed authoring schema.

    The message names the field and the expected/actual value.  When raised by
    :func:`load_task_document` it also names the document path.
    """


@dataclass(frozen=True)
class RetentionSpec:
    """How much episode history and event data a run retains."""

    retain_episodes: bool = False
    max_events: int = 200_000
    event_log: bool = True

    def __post_init__(self) -> None:
        """Validate the retention policy."""
        _require_bool(self.retain_episodes, "retention.retain_episodes")
        _require_int(self.max_events, "retention.max_events", minimum=0)
        _require_bool(self.event_log, "retention.event_log")


@dataclass(frozen=True)
class MonitorSpec:
    """Declaration of one monitor.

    Exactly one of ``builtin``, ``module`` or ``factory`` selects the monitor.
    ``factory`` is an inline callable for programmatic composition; a YAML
    document cannot express it and must use ``builtin`` or ``module``.

    A ``builtin: predicate`` declaration carries the closed predicate fields
    ``when`` (condition mapping), ``trigger`` (kind string or closed mapping)
    and optional ``emit`` (closed mapping).  A ``builtin: phase`` declaration
    carries the explicit ``source`` reference and its boundaries through
    ``params``.  Predicate fields are not accepted on a module/factory
    declaration or on a builtin that does not consume them.
    """

    name: str
    builtin: str | None = None
    module: str | None = None
    factory: Callable[..., object] | None = None
    params: Mapping[str, FrozenJson] = field(default_factory=lambda: MappingProxyType({}))
    report_extension: str | None = None
    resources: Mapping[str, Path] = field(default_factory=lambda: MappingProxyType({}))
    when: Mapping[str, FrozenJson] | None = None
    trigger: str | Mapping[str, FrozenJson] | None = None
    emit: Mapping[str, FrozenJson] | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        """Validate the monitor declaration."""
        where = f"monitor {self.name!r}"
        _require_name(self.name, "monitor name")
        selected = [
            label
            for label, value in (
                ("builtin", self.builtin),
                ("module", self.module),
                ("factory", self.factory),
            )
            if value is not None
        ]
        if len(selected) != 1:
            actual = ", ".join(selected) if selected else "none"
            raise SpecError(
                f"{where}: expected exactly one of 'builtin', 'module' or 'factory', "
                f"got {actual}"
            )
        if self.builtin is not None:
            _require_str(self.builtin, f"{where}.builtin")
        if self.module is not None:
            _require_module(self.module, f"{where}.module")
        if self.factory is not None and not callable(self.factory):
            raise SpecError(
                f"{where}.factory: expected a callable factory, got "
                f"{type(self.factory).__name__}; use 'module' for an import path"
            )
        if self.report_extension is not None:
            _require_name(self.report_extension, f"{where}.report_extension")
        object.__setattr__(self, "params", _validate_params(self.params, f"{where}.params"))
        object.__setattr__(
            self, "resources", _validate_resources(self.resources, f"{where}.resources")
        )
        self._validate_predicate_fields(where)

    def _validate_predicate_fields(self, where: str) -> None:
        """Validate and freeze the closed predicate fields."""
        when = None
        if self.when is not None:
            when = _validate_json_object(self.when, f"{where}.when")
            try:
                parse_condition(dict(when))
            except (ConditionError, TypeError) as exc:
                raise SpecError(f"{where}.when: {exc}") from exc
        trigger = None
        if self.trigger is not None:
            if isinstance(self.trigger, str):
                try:
                    TriggerSpec.from_mapping({"kind": self.trigger})
                except (TypeError, ValueError) as exc:
                    raise SpecError(f"{where}.trigger: {exc}") from exc
                trigger = self.trigger
            elif isinstance(self.trigger, Mapping):
                raw_trigger = _validate_json_object(self.trigger, f"{where}.trigger")
                try:
                    TriggerSpec.from_mapping(raw_trigger)
                except (TypeError, ValueError) as exc:
                    raise SpecError(f"{where}.trigger: {exc}") from exc
                trigger = raw_trigger
            else:
                raise SpecError(
                    f"{where}.trigger: expected a trigger kind string or a mapping, "
                    f"got {type(self.trigger).__name__}"
                )
        emit = None
        if self.emit is not None:
            emit = _validate_json_object(self.emit, f"{where}.emit")
            try:
                EmitSpec.from_mapping(emit)
            except (TypeError, ValueError) as exc:
                raise SpecError(f"{where}.emit: {exc}") from exc
        source = None
        if self.source is not None:
            if not isinstance(self.source, str):
                raise SpecError(
                    f"{where}.source: expected a 'kind:key' string, "
                    f"got {type(self.source).__name__}"
                )
            try:
                SourceRef.parse(self.source)
            except SourceError as exc:
                raise SpecError(f"{where}.source: {exc}") from exc
            source = self.source
        object.__setattr__(self, "when", when)
        object.__setattr__(self, "trigger", trigger)
        object.__setattr__(self, "emit", emit)
        object.__setattr__(self, "source", source)
        has_predicate = when is not None or trigger is not None or emit is not None
        if self.module is not None or self.factory is not None:
            if has_predicate or source is not None:
                raise SpecError(
                    f"{where}: 'when'/'trigger'/'emit'/'source' require a builtin "
                    "that consumes them ('predicate' or 'phase'), not a "
                    "module/factory declaration"
                )
            return
        if self.builtin == "predicate":
            if when is None:
                raise SpecError(f"{where}.when: required for builtin 'predicate'")
            if trigger is None:
                raise SpecError(f"{where}.trigger: required for builtin 'predicate'")
            if source is not None:
                raise SpecError(
                    f"{where}.source: builtin 'predicate' reads sources through "
                    "'when'; remove 'source' or declare builtin 'phase'"
                )
            if self.params:
                raise SpecError(
                    f"{where}.params: builtin 'predicate' takes no params; put the "
                    "condition in 'when' and the outputs in 'trigger'/'emit'"
                )
            return
        if self.builtin == "phase":
            if source is None:
                raise SpecError(f"{where}.source: required for builtin 'phase'")
            if has_predicate:
                raise SpecError(
                    f"{where}: builtin 'phase' accepts 'source' and params, not "
                    "'when'/'trigger'/'emit'"
                )
            return
        # An unknown builtin is rejected by the composition registry; the
        # predicate fields are left untouched so the registry error names the
        # available builtins instead of hiding behind a field-shape error.


@dataclass(frozen=True)
class ExtensionSpec:
    """Declaration of one report extension.

    Exactly one of ``builtin`` or ``module`` selects the extension.
    """

    name: str
    builtin: str | None = None
    module: str | None = None
    params: Mapping[str, FrozenJson] = field(default_factory=lambda: MappingProxyType({}))
    resources: Mapping[str, Path] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """Validate the extension declaration."""
        where = f"extension {self.name!r}"
        _require_name(self.name, "extension name")
        if (self.builtin is None) == (self.module is None):
            raise SpecError(
                f"{where}: expected exactly one of 'builtin' or 'module', "
                f"got builtin={self.builtin!r}, module={self.module!r}"
            )
        if self.builtin is not None:
            _require_str(self.builtin, f"{where}.builtin")
        if self.module is not None:
            _require_module(self.module, f"{where}.module")
        object.__setattr__(self, "params", _validate_params(self.params, f"{where}.params"))
        object.__setattr__(
            self, "resources", _validate_resources(self.resources, f"{where}.resources")
        )


@dataclass(frozen=True)
class ScenarioSpec:
    """Command pool and optional execution plan owned by a task document.

    The sampler owns the cursor, dwell and RNG; this spec only states which
    command vectors exist and, optionally, the order in which to apply them.
    """

    command_pool: tuple[tuple[float, ...], ...]
    plan: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the command pool and plan."""
        pool = _validate_command_pool(self.command_pool, "scenario.command_pool")
        object.__setattr__(self, "command_pool", pool)
        if self.plan is not None:
            object.__setattr__(self, "plan", _validate_plan(self.plan, len(pool)))


@dataclass(frozen=True)
class TerminationRuleSpec:
    """One explicit value-to-terminal rule in a task document."""

    monitor: str
    value: str
    outcome: str
    code: str

    def __post_init__(self) -> None:
        """Validate the rule fields at the document boundary."""
        _require_name(self.monitor, "flow.termination_rules.monitor")
        _require_name(self.value, "flow.termination_rules.value")
        outcome = _require_flow_outcome(self.outcome, "flow.termination_rules.outcome")
        object.__setattr__(self, "outcome", outcome.value)
        if not isinstance(self.code, str) or not self.code.startswith("custom."):
            raise SpecError(
                "flow.termination_rules.code: expected a 'custom.*' code, " f"got {self.code!r}"
            )
        try:
            validate_code(self.code)
        except (TypeError, ValueError) as exc:
            raise SpecError(f"flow.termination_rules.code: {exc}") from exc


@dataclass(frozen=True)
class EventRuleSpec:
    """One explicit value-to-event rule in a task document."""

    monitor: str
    value: str
    name: str
    severity: Severity = "info"
    code: str | None = None
    data: Mapping[str, FrozenJson] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """Validate the event rule fields at the document boundary."""
        _require_name(self.monitor, "flow.events.monitor")
        _require_name(self.value, "flow.events.value")
        _require_name(self.name, "flow.events.name")
        if not isinstance(self.severity, str) or self.severity not in SEVERITIES:
            raise SpecError(
                f"flow.events.severity: expected one of {sorted(SEVERITIES)}, "
                f"got {self.severity!r}"
            )
        if self.code is not None:
            if not isinstance(self.code, str):
                raise SpecError(f"flow.events.code: expected a string, got {self.code!r}")
            try:
                validate_code(self.code)
            except (TypeError, ValueError) as exc:
                raise SpecError(f"flow.events.code: {exc}") from exc
        object.__setattr__(self, "data", _validate_params(self.data, "flow.events.data"))


@dataclass(frozen=True)
class FlowSpec:
    """Explicit decision rules owned by the task document."""

    termination_rules: tuple[TerminationRuleSpec, ...] = ()
    events: tuple[EventRuleSpec, ...] = ()

    def __post_init__(self) -> None:
        """Validate and freeze the flow rule tuples."""
        rules = tuple(self.termination_rules)
        seen: set[tuple[str, str]] = set()
        for index, rule in enumerate(rules):
            if not isinstance(rule, TerminationRuleSpec):
                raise SpecError(
                    f"flow.termination_rules[{index}]: expected a TerminationRuleSpec, "
                    f"got {type(rule).__name__}"
                )
            key = (rule.monitor, rule.value)
            if key in seen:
                raise SpecError(
                    "flow.termination_rules: duplicate "
                    f"(monitor={rule.monitor!r}, value={rule.value!r}) rule; the later "
                    "rule is dead policy"
                )
            seen.add(key)
        events = tuple(self.events)
        for index, rule in enumerate(events):
            if not isinstance(rule, EventRuleSpec):
                raise SpecError(
                    f"flow.events[{index}]: expected an EventRuleSpec, "
                    f"got {type(rule).__name__}"
                )
        object.__setattr__(self, "termination_rules", rules)
        object.__setattr__(self, "events", events)


@dataclass(frozen=True)
class TaskSpec:
    """A validated task: monitors, report extensions and retention.

    A task document may omit ``extensions``; the field then defaults to an empty
    mapping.  This matches the plan: report extensions are optional even though
    they are part of the task model.
    """

    name: str
    monitors: Mapping[str, MonitorSpec]
    extensions: Mapping[str, ExtensionSpec] = field(default_factory=lambda: MappingProxyType({}))
    retention: RetentionSpec = field(default_factory=RetentionSpec)
    flow: FlowSpec = field(default_factory=FlowSpec)

    def __post_init__(self) -> None:
        """Validate names, namespaces, extension references and flow monitors."""
        _require_str(self.name, "task.name")
        monitors = _validate_monitor_mapping(self.monitors)
        extensions = _validate_extension_mapping(self.extensions)
        object.__setattr__(self, "monitors", monitors)
        object.__setattr__(self, "extensions", extensions)
        if not isinstance(self.retention, RetentionSpec):
            raise SpecError(
                "task.retention: expected a RetentionSpec, " f"got {type(self.retention).__name__}"
            )
        if not isinstance(self.flow, FlowSpec):
            raise SpecError(f"task.flow: expected a FlowSpec, got {type(self.flow).__name__}")
        available = sorted(extensions)
        for monitor_name, monitor in monitors.items():
            reference = monitor.report_extension
            if reference is not None and reference not in extensions:
                raise SpecError(
                    f"task.monitors.{monitor_name}.report_extension: expected an "
                    f"existing extension name from {available}, got {reference!r}"
                )
        monitor_names = set(monitors)
        for rule in self.flow.termination_rules:
            if rule.monitor not in monitor_names:
                raise SpecError(
                    f"flow.termination_rules.monitor: unknown monitor {rule.monitor!r}; "
                    f"declared monitors are {sorted(monitor_names)}"
                )
        for rule in self.flow.events:
            if rule.monitor not in monitor_names:
                raise SpecError(
                    f"flow.events.monitor: unknown monitor {rule.monitor!r}; "
                    f"declared monitors are {sorted(monitor_names)}"
                )


@dataclass(frozen=True)
class TaskDocument:
    """The loader wrapper around one :class:`TaskSpec`.

    ``task`` holds the validated task.  ``scenario``, ``schema``, ``base_dir``
    and ``source`` belong to the document, not to the task: a scenario is
    document-level data shared by the task, and provenance is supplied by
    :func:`load_task_document`.  A task document contains exactly one task.
    """

    task: TaskSpec
    scenario: ScenarioSpec | None = None
    schema: str | None = None
    base_dir: Path | None = None
    source: Path | None = None

    def __post_init__(self) -> None:
        """Validate document provenance."""
        if not isinstance(self.task, TaskSpec):
            raise SpecError(f"task document: expected a TaskSpec, got {type(self.task).__name__}")
        if self.scenario is not None and not isinstance(self.scenario, ScenarioSpec):
            raise SpecError(
                "task document.scenario: expected a ScenarioSpec or None, "
                f"got {type(self.scenario).__name__}"
            )
        if self.schema is not None and self.schema != SCHEMA:
            raise SpecError(_schema_error(self.schema))
        if self.base_dir is not None and not isinstance(self.base_dir, Path):
            raise SpecError(
                "task document.base_dir: expected a Path or None, "
                f"got {type(self.base_dir).__name__}"
            )
        if self.source is not None and not isinstance(self.source, Path):
            raise SpecError(
                "task document.source: expected a Path or None, "
                f"got {type(self.source).__name__}"
            )


def load_task_document(path: str | Path) -> TaskDocument:
    """Load and validate a YAML task document from ``path``.

    The document directory becomes the resolution base for relative resource
    paths.  Relative documents are resolved for reading only; resource paths
    never resolve against the process working directory.

    Args:
        path: Path to a YAML task document.

    Returns:
        A frozen :class:`TaskDocument` whose ``source`` is the document path.

    Raises:
        SpecError: The file cannot be read as UTF-8 text, is not valid YAML,
            contains a duplicate mapping key, or violates the task schema.
    """
    document_path = Path(path).expanduser()
    if not document_path.is_absolute():
        document_path = document_path.resolve()
    try:
        text = document_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SpecError(f"{document_path}: cannot read task document: {exc}") from exc
    try:
        raw = yaml.load(text, Loader=_UniqueKeyLoader)
    except SpecError as exc:
        raise SpecError(f"{document_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SpecError(f"{document_path}: invalid YAML: {exc}") from exc
    try:
        document = parse_task_document(raw, base_dir=document_path.parent)
    except SpecError as exc:
        raise SpecError(f"{document_path}: {exc}") from exc
    return replace(document, source=document_path)


def parse_task_document(mapping: object, base_dir: Path | str | None = None) -> TaskDocument:
    """Validate an in-memory task document mapping.

    Args:
        mapping: The raw document mapping, usually the result of a YAML parse.
        base_dir: Absolute directory used to resolve relative resource paths.
            It is required when a relative resource path is declared; absolute
            paths do not need it.  It is never inferred from the working
            directory.

    Optional document keys may be omitted.  Omitting ``schema`` or ``scenario``
    means "not declared", and omitting ``extensions`` or ``retention`` applies
    the documented default; an explicit ``null`` is a configuration error, not
    an alias for omission.

    Returns:
        A frozen :class:`TaskDocument` with ``source=None``.

    Raises:
        SpecError: The mapping is not a task document or violates the schema.
    """
    base = _validate_base_dir(base_dir)
    if not isinstance(mapping, Mapping):
        raise SpecError("task document: expected a mapping, " f"got {type(mapping).__name__}")
    _check_keys(mapping, _DOCUMENT_KEYS, "task document")
    if "schema" in mapping:
        schema = mapping["schema"]
        if schema is None:
            raise SpecError(
                "task document.schema: explicit null is not allowed; omit the key "
                f"or use {SCHEMA!r}"
            )
        if schema != SCHEMA:
            raise SpecError(_schema_error(schema))
    else:
        schema = None
    name = mapping.get("name")
    _require_str(name, "task document.name")
    if "monitors" not in mapping or mapping["monitors"] is None:
        raise SpecError(
            f"task {name!r}: field 'monitors' is required (mapping of monitor name "
            "to declaration)"
        )
    monitors = _parse_monitors(mapping["monitors"], name, base)
    extensions = _parse_extensions(
        mapping["extensions"] if "extensions" in mapping else _MISSING, name, base
    )
    retention = _parse_retention(mapping["retention"] if "retention" in mapping else _MISSING)
    scenario = _parse_scenario(mapping["scenario"] if "scenario" in mapping else _MISSING, name)
    flow = _parse_flow(mapping["flow"] if "flow" in mapping else _MISSING, name)
    task = TaskSpec(
        name=name,
        monitors=monitors,
        extensions=extensions,
        retention=retention,
        flow=flow,
    )
    return TaskDocument(task=task, scenario=scenario, schema=schema, base_dir=base)


def _validate_base_dir(base_dir: Path | str | None) -> Path | None:
    if base_dir is None:
        return None
    base = Path(base_dir)
    if not base.is_absolute():
        raise SpecError("task document base_dir: expected an absolute path, " f"got {str(base)!r}")
    return base.resolve()


def _parse_monitors(
    raw: object, task_name: str, base_dir: Path | None
) -> Mapping[str, MonitorSpec]:
    if not isinstance(raw, Mapping):
        raise SpecError(
            f"task {task_name!r}.monitors: expected a mapping of name to declaration, "
            f"got {type(raw).__name__}"
        )
    parsed: dict[str, MonitorSpec] = {}
    for key, entry in raw.items():
        where = f"task {task_name!r}.monitors.{key}"
        _require_name(key, f"task {task_name!r}.monitors name")
        if not isinstance(entry, Mapping):
            raise SpecError(
                f"{where}: expected a mapping with 'builtin', 'module' or 'factory', "
                f"got {type(entry).__name__}"
            )
        _reject_inline_name(entry, where, key)
        _check_keys(entry, _MONITOR_KEYS, where)
        for field_name in ("when", "trigger", "emit", "source"):
            if field_name in entry and entry[field_name] is None:
                raise SpecError(
                    f"{where}.{field_name}: explicit null is not allowed; omit the "
                    "key when the field does not apply"
                )
        parsed[key] = MonitorSpec(
            name=key,
            builtin=entry.get("builtin"),
            module=entry.get("module"),
            factory=entry.get("factory"),
            params=entry.get("params", MappingProxyType({})),
            report_extension=entry.get("report_extension"),
            resources=_resolve_resources(
                entry["resources"] if "resources" in entry else _MISSING,
                f"{where}.resources",
                base_dir,
            ),
            when=entry.get("when"),
            trigger=entry.get("trigger"),
            emit=entry.get("emit"),
            source=entry.get("source"),
        )
    return MappingProxyType(parsed)


def _parse_extensions(
    raw: object, task_name: str, base_dir: Path | None
) -> Mapping[str, ExtensionSpec]:
    if raw is _MISSING:
        return MappingProxyType({})
    if raw is None:
        raise SpecError(
            f"task {task_name!r}.extensions: explicit null is not allowed; omit the "
            "key to declare no extensions"
        )
    if not isinstance(raw, Mapping):
        raise SpecError(
            f"task {task_name!r}.extensions: expected a mapping of name to declaration, "
            f"got {type(raw).__name__}"
        )
    parsed: dict[str, ExtensionSpec] = {}
    for key, entry in raw.items():
        where = f"task {task_name!r}.extensions.{key}"
        _require_name(key, f"task {task_name!r}.extensions name")
        if not isinstance(entry, Mapping):
            raise SpecError(
                f"{where}: expected a mapping with 'builtin' or 'module', "
                f"got {type(entry).__name__}"
            )
        _reject_inline_name(entry, where, key)
        _check_keys(entry, _EXTENSION_KEYS, where)
        parsed[key] = ExtensionSpec(
            name=key,
            builtin=entry.get("builtin"),
            module=entry.get("module"),
            params=entry.get("params", MappingProxyType({})),
            resources=_resolve_resources(
                entry["resources"] if "resources" in entry else _MISSING,
                f"{where}.resources",
                base_dir,
            ),
        )
    return MappingProxyType(parsed)


def _parse_retention(raw: object) -> RetentionSpec:
    if raw is _MISSING:
        return RetentionSpec()
    if raw is None:
        raise SpecError(
            "task.retention: explicit null is not allowed; omit the key to use the " "defaults"
        )
    if not isinstance(raw, Mapping):
        raise SpecError("task.retention: expected a mapping, " f"got {type(raw).__name__}")
    _check_keys(raw, _RETENTION_KEYS, "task.retention")
    values = {key: raw[key] for key in _RETENTION_KEYS if key in raw}
    return RetentionSpec(**values)


def _parse_scenario(raw: object, task_name: str) -> ScenarioSpec | None:
    where = f"task {task_name!r}.scenario"
    if raw is _MISSING:
        return None
    if raw is None:
        raise SpecError(
            f"{where}: explicit null is not allowed; omit the key to declare no " "scenario"
        )
    if not isinstance(raw, Mapping):
        raise SpecError(f"{where}: expected a mapping, got {type(raw).__name__}")
    _check_keys(raw, _SCENARIO_KEYS, where)
    if "command_pool" not in raw:
        raise SpecError(
            f"{where}.command_pool: required (a non-empty list of equal-length "
            "numeric command vectors)"
        )
    return ScenarioSpec(command_pool=raw["command_pool"], plan=raw.get("plan"))


def _parse_flow(raw: object, task_name: str) -> FlowSpec:
    where = f"task {task_name!r}.flow"
    if raw is _MISSING:
        return FlowSpec()
    if raw is None:
        raise SpecError(
            f"{where}: explicit null is not allowed; omit the key to declare no "
            "automatic termination or events"
        )
    if not isinstance(raw, Mapping):
        raise SpecError(f"{where}: expected a mapping, got {type(raw).__name__}")
    _check_keys(raw, _FLOW_KEYS, where)
    termination_rules = _parse_termination_rules(
        raw["termination_rules"] if "termination_rules" in raw else _MISSING, where
    )
    events = _parse_event_rules(raw["events"] if "events" in raw else _MISSING, where)
    return FlowSpec(termination_rules=termination_rules, events=events)


def _parse_termination_rules(raw: object, where: str) -> tuple[TerminationRuleSpec, ...]:
    if raw is _MISSING:
        return ()
    if raw is None:
        raise SpecError(
            f"{where}.termination_rules: explicit null is not allowed; omit the key "
            "to declare no automatic termination"
        )
    if not _is_sequence(raw):
        raise SpecError(
            f"{where}.termination_rules: expected a list of rules, " f"got {type(raw).__name__}"
        )
    rules: list[TerminationRuleSpec] = []
    for index, item in enumerate(raw):
        item_where = f"{where}.termination_rules[{index}]"
        if not isinstance(item, Mapping):
            raise SpecError(f"{item_where}: expected a mapping, got {type(item).__name__}")
        _check_keys(item, _TERMINATION_RULE_KEYS, item_where)
        missing = sorted(_TERMINATION_RULE_KEYS - set(item))
        if missing:
            raise SpecError(f"{item_where}: missing required keys {missing}")
        rules.append(
            TerminationRuleSpec(
                monitor=cast(str, item["monitor"]),
                value=cast(str, item["value"]),
                outcome=cast(str, item["outcome"]),
                code=cast(str, item["code"]),
            )
        )
    return tuple(rules)


def _parse_event_rules(raw: object, where: str) -> tuple[EventRuleSpec, ...]:
    if raw is _MISSING:
        return ()
    if raw is None:
        raise SpecError(
            f"{where}.events: explicit null is not allowed; omit the key to declare "
            "no automatic events"
        )
    if not _is_sequence(raw):
        raise SpecError(f"{where}.events: expected a list of rules, got {type(raw).__name__}")
    rules: list[EventRuleSpec] = []
    for index, item in enumerate(raw):
        item_where = f"{where}.events[{index}]"
        if not isinstance(item, Mapping):
            raise SpecError(f"{item_where}: expected a mapping, got {type(item).__name__}")
        _check_keys(item, _EVENT_RULE_KEYS, item_where)
        for required in ("monitor", "value", "name"):
            if required not in item:
                raise SpecError(f"{item_where}: missing required key {required!r}")
        data = item.get("data", MappingProxyType({}))
        rules.append(
            EventRuleSpec(
                monitor=cast(str, item["monitor"]),
                value=cast(str, item["value"]),
                name=cast(str, item["name"]),
                severity=cast(Severity, item.get("severity", "info")),
                code=cast(str | None, item.get("code")),
                data=cast(Mapping[str, object], data),
            )
        )
    return tuple(rules)


def _require_flow_outcome(value: object, where: str) -> EpisodeOutcome:
    if isinstance(value, EpisodeOutcome):
        outcome = value
    elif isinstance(value, str):
        try:
            outcome = EpisodeOutcome(value)
        except ValueError as exc:
            raise SpecError(
                f"{where}: unknown episode outcome {value!r}; "
                f"allowed: {sorted(item.value for item in _MONITOR_OUTCOMES)}"
            ) from exc
    else:
        raise SpecError(f"{where}: expected an EpisodeOutcome or str, got {value!r}")
    if outcome not in _MONITOR_OUTCOMES:
        raise SpecError(
            f"{where}: {outcome.value!r} is not a flow-owned terminal outcome; "
            f"allowed: {sorted(item.value for item in _MONITOR_OUTCOMES)}"
        )
    return outcome


def _schema_error(schema: object) -> str:
    if schema in _RETIRED_SCHEMAS:
        return (
            f"task document.schema: {schema!r} is retired; migrate to {SCHEMA!r}. "
            "Monitor 'emit' now accepts only 'value' (monitors publish values only); "
            "move event/terminal output to flow.termination_rules / flow.events. "
            "Monitor snapshots are unigym_bench.monitor/4 and rewind payloads are "
            "unigym_bench.rewind/5; old checkpoints are rejected."
        )
    return f"task document.schema: expected {SCHEMA!r}, got {schema!r}"


def _resolve_resources(raw: object, where: str, base_dir: Path | None) -> Mapping[str, Path]:
    if raw is _MISSING:
        return MappingProxyType({})
    if raw is None:
        raise SpecError(
            f"{where}: explicit null is not allowed; omit the key to declare no " "resources"
        )
    if not isinstance(raw, Mapping):
        raise SpecError(
            f"{where}: expected a mapping of resource name to path, " f"got {type(raw).__name__}"
        )
    resolved: dict[str, Path] = {}
    for key, value in raw.items():
        _require_name(key, f"{where} name")
        if not isinstance(value, str) or not value.strip():
            raise SpecError(f"{where}.{key}: expected a non-empty path string, got {value!r}")
        path = Path(value).expanduser()
        if not path.is_absolute():
            if base_dir is None:
                raise SpecError(
                    f"{where}.{key}: relative path {value!r} requires the task "
                    "document base directory; use an absolute path or load the "
                    "document from a file"
                )
            path = base_dir / path
        resolved[key] = path.resolve()
    return MappingProxyType(resolved)


def _validate_monitor_mapping(raw: object) -> Mapping[str, MonitorSpec]:
    if not isinstance(raw, Mapping):
        raise SpecError(
            "task.monitors: expected a mapping of name to MonitorSpec, " f"got {type(raw).__name__}"
        )
    parsed: dict[str, MonitorSpec] = {}
    for key, spec in raw.items():
        _require_name(key, "task.monitors name")
        if not isinstance(spec, MonitorSpec):
            raise SpecError(
                f"task.monitors.{key}: expected a MonitorSpec, " f"got {type(spec).__name__}"
            )
        if spec.name != key:
            raise SpecError(
                f"task.monitors.{key}: the declaration names {spec.name!r}; "
                "the mapping key is the monitor name"
            )
        parsed[key] = spec
    return MappingProxyType(parsed)


def _validate_extension_mapping(raw: object) -> Mapping[str, ExtensionSpec]:
    if not isinstance(raw, Mapping):
        raise SpecError(
            "task.extensions: expected a mapping of name to ExtensionSpec, "
            f"got {type(raw).__name__}"
        )
    parsed: dict[str, ExtensionSpec] = {}
    for key, spec in raw.items():
        _require_name(key, "task.extensions name")
        if not isinstance(spec, ExtensionSpec):
            raise SpecError(
                f"task.extensions.{key}: expected an ExtensionSpec, " f"got {type(spec).__name__}"
            )
        if spec.name != key:
            raise SpecError(
                f"task.extensions.{key}: the declaration names {spec.name!r}; "
                "the mapping key is the extension name"
            )
        parsed[key] = spec
    return MappingProxyType(parsed)


def _validate_params(raw: object, where: str) -> Mapping[str, FrozenJson]:
    normalized = _normalize_json_object(raw, where)
    return MappingProxyType({key: _freeze(value) for key, value in normalized.items()})


def _validate_json_object(raw: object, where: str) -> Mapping[str, FrozenJson]:
    normalized = _normalize_json_object(raw, where)
    return MappingProxyType({key: _freeze(value) for key, value in normalized.items()})


def _normalize_json_object(raw: object, where: str) -> dict[str, JsonValue]:
    try:
        normalized = ensure_json_value(raw, where=where)
    except (TypeError, ValueError) as exc:
        raise SpecError(str(exc)) from exc
    if not isinstance(normalized, dict):
        raise SpecError(
            f"{where}: expected a JSON object (mapping), " f"got {type(normalized).__name__}"
        )
    return normalized


def _freeze(value: JsonValue) -> FrozenJson:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _validate_resources(raw: object, where: str) -> Mapping[str, Path]:
    if not isinstance(raw, Mapping):
        raise SpecError(
            f"{where}: expected a mapping of resource name to Path, " f"got {type(raw).__name__}"
        )
    validated: dict[str, Path] = {}
    for key, value in raw.items():
        _require_name(key, f"{where} name")
        if not isinstance(value, Path):
            raise SpecError(f"{where}.{key}: expected a Path, got {type(value).__name__}")
        if not value.is_absolute():
            raise SpecError(
                f"{where}.{key}: expected an absolute path resolved against the "
                f"task document, got {str(value)!r}"
            )
        validated[key] = value
    return MappingProxyType(validated)


def _validate_command_pool(raw: object, where: str) -> tuple[tuple[float, ...], ...]:
    if not _is_sequence(raw):
        raise SpecError(
            f"{where}: expected a non-empty list of command vectors, " f"got {type(raw).__name__}"
        )
    rows = list(raw)
    if not rows:
        raise SpecError(f"{where}: expected at least one command vector, got an empty list")
    commands: list[tuple[float, ...]] = []
    dimension: int | None = None
    for row_index, row in enumerate(rows):
        row_where = f"{where}[{row_index}]"
        if not _is_sequence(row):
            raise SpecError(
                f"{row_where}: expected a command vector (list of numbers), "
                f"got {type(row).__name__}"
            )
        values = list(row)
        if dimension is None:
            dimension = len(values)
            if dimension == 0:
                raise SpecError(f"{row_where}: command vector must not be empty")
        elif len(values) != dimension:
            raise SpecError(
                f"{row_where}: expected {dimension} values to match "
                f"{where}[0], got {len(values)}"
            )
        commands.append(
            tuple(
                _command_value(value, f"{row_where}[{value_index}]")
                for value_index, value in enumerate(values)
            )
        )
    return tuple(commands)


def _command_value(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(f"{where}: expected a finite number, got {value!r}")
    try:
        number = float(value)
    except OverflowError as exc:
        raise SpecError(f"{where}: number is out of range, got {value!r}") from exc
    if not math.isfinite(number):
        raise SpecError(f"{where}: expected a finite number, got {value!r}")
    return number


def _validate_plan(raw: object, pool_size: int) -> tuple[int, ...]:
    where = "scenario.plan"
    if not _is_sequence(raw):
        raise SpecError(
            f"{where}: expected a non-empty list of command-pool indices, "
            f"got {type(raw).__name__}"
        )
    values = list(raw)
    if not values:
        raise SpecError(f"{where}: expected at least one index, got an empty list")
    plan: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise SpecError(f"{where}[{index}]: expected an integer index, got {value!r}")
        if not 0 <= value < pool_size:
            raise SpecError(
                f"{where}[{index}]: expected an index in [0, {pool_size}), " f"got {value}"
            )
        plan.append(value)
    return tuple(plan)


def _check_keys(raw: Mapping[object, object], allowed: frozenset[str], where: str) -> None:
    unknown = [key for key in raw if key not in allowed]
    if not unknown:
        return
    formatted = ", ".join(sorted(repr(key) for key in unknown))
    raise SpecError(f"{where}: unknown key(s) {formatted}; expected only {sorted(allowed)}")


def _reject_inline_name(entry: Mapping[object, object], where: str, key: object) -> None:
    if "name" in entry:
        raise SpecError(
            f"{where}: do not declare 'name' inside the entry; "
            f"the mapping key {key!r} is the name"
        )


def _require_str(value: object, where: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SpecError(f"{where}: expected a non-empty string, got {value!r}")


def _require_name(value: object, where: str) -> None:
    if not isinstance(value, str) or not _NAME_PATTERN.fullmatch(value):
        raise SpecError(f"{where}: expected a name matching [a-z][a-z0-9_]*, got {value!r}")


def _require_module(value: object, where: str) -> None:
    if not isinstance(value, str) or not _MODULE_PATTERN.fullmatch(value):
        raise SpecError(f"{where}: expected a module path 'pkg.mod:attr', got {value!r}")


def _require_bool(value: object, where: str) -> None:
    if not isinstance(value, bool):
        raise SpecError(f"{where}: expected a bool, got {value!r}")


def _require_int(value: object, where: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecError(f"{where}: expected an int, got {value!r}")
    if value < minimum:
        raise SpecError(f"{where}: expected an int >= {minimum}, got {value}")


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_mapping(
    loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise SpecError(
                f"YAML mapping key at line {key_node.start_mark.line + 1} is not "
                f"hashable: {key!r}"
            ) from exc
        if duplicate:
            raise SpecError(f"duplicate mapping key {key!r} at line {key_node.start_mark.line + 1}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)
