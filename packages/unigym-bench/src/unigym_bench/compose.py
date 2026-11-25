"""Compose a validated task document into live pack components.

The composition boundary is the only place where an authoring document becomes
live monitors, report extensions, a retention policy and the task flow.  It owns
the closed builtin registry, module/factory resolution, duplicate-name checks,
the scenario-derived coverage extension and the explicit flow rule conversion.
It never accepts YAML callables: a document factory is an API-only Python
callable, and a ``module`` reference is a syntax-checked import path imported
only here.

The closed registries are intentionally small:

``monitors``
    ``predicate`` (a :class:`~unigym_bench.monitors.PredicateMonitor` built from
    ``when``/``trigger``/``emit``) and ``phase`` (a
    :class:`~unigym_bench.monitors.PhaseMonitor` built from ``source`` and
    boundary params).
``extensions``
    ``scenario_coverage`` (a
    :class:`~unigym_bench.report.ScenarioCoverageExtension` built from the
    document command pool).

An unknown builtin fails with the registry names.  The retired typed monitor
builtins are not aliases and are rejected.

Scenario policy
---------------

A declared ``scenario`` and a ``scenario_coverage`` extension are coupled: a
declared scenario is automatically paired with a coverage extension when the
document does not already declare one, using the documented name
``scenario_coverage``.  If that name is already taken by a different
extension, composition fails and asks the author to rename it or declare the
coverage extension explicitly.  A ``scenario_coverage`` extension without a
document scenario is a configuration error because the command pool is the
document scenario.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from .conditions import ConditionError, parse_condition
from .flow import DefaultBenchmarkFlow, EventRule, TerminationRule
from .monitors import Monitor, PhaseMonitor, PredicateMonitor
from .predicates import EmitSpec, TriggerSpec
from .report import ReportExtension, ScenarioCoverageExtension, validate_report_extensions
from .scenario import Scenario
from .sources import SourceError, SourceRef, SourceUnavailableError
from .specs import (
    ExtensionSpec,
    FlowSpec,
    MonitorSpec,
    RetentionSpec,
    ScenarioSpec,
    SpecError,
    TaskDocument,
)

__all__ = [
    "ComposeError",
    "EXTENSION_BUILTINS",
    "MONITOR_BUILTINS",
    "SCENARIO_COVERAGE_NAME",
    "TaskPlan",
    "compose_task",
]

#: Monitor builtin names understood by :func:`compose_task`.
MONITOR_BUILTINS: tuple[str, ...] = ("phase", "predicate")

#: Extension builtin names understood by :func:`compose_task`.
EXTENSION_BUILTINS: tuple[str, ...] = ("scenario_coverage",)

#: Extension name used when a declared scenario auto-composes coverage.
SCENARIO_COVERAGE_NAME = "scenario_coverage"


class ComposeError(ValueError):
    """A task document cannot be composed into live pack components."""


@dataclass(frozen=True)
class TaskPlan:
    """The live composition result of one task document.

    ``monitors`` and ``extensions`` are in task declaration order, with an
    automatic scenario coverage extension appended when applicable.
    ``monitor_extensions`` maps a monitor name to the extension name declared
    through ``report_extension``.
    """

    monitors: tuple[Monitor, ...]
    extensions: tuple[ReportExtension, ...]
    retention: RetentionSpec
    scenario: ScenarioSpec | None
    monitor_extensions: Mapping[str, str]
    flow: DefaultBenchmarkFlow


def compose_task(document: TaskDocument) -> TaskPlan:
    """Compose a validated task document into live pack components.

    Args:
        document: A :class:`~unigym_bench.specs.TaskDocument` returned by the
            task-document loader or parser.  Raw mappings are rejected.

    Returns:
        A frozen :class:`TaskPlan` with live monitors/extensions, the document
        retention policy, the declared scenario and the declared
        ``report_extension`` references.

    Raises:
        TypeError: ``document`` is not a :class:`TaskDocument`.
        ComposeError: A builtin is unknown, a declaration cannot be
            instantiated, a name is duplicated, a ``report_extension``
            reference is missing, or scenario coverage cannot be composed.
        SourceUnavailableError: A predicate or phase source uses the reserved
            ``query:`` namespace; this fails at composition, not at evaluation.
    """
    if not isinstance(document, TaskDocument):
        raise TypeError(
            "compose_task: expected a TaskDocument, "
            f"got {type(document).__name__}; use load_task_document/parse_task_document"
        )
    monitors = tuple(_compose_monitor(spec) for spec in document.task.monitors.values())
    extensions = _compose_extensions(document)
    _require_unique_names([monitor.name for monitor in monitors], "monitor", "live monitor")
    _require_unique_names(
        [extension.name for extension in extensions],
        "extension",
        "report extension",
    )
    _validate_declared_names(list(document.task.monitors.values()), monitors, "monitor")
    _validate_declared_names(
        list(document.task.extensions.values()),
        extensions[: len(document.task.extensions)],
        "extension",
    )
    extension_names = {extension.name for extension in extensions}
    monitor_extensions: dict[str, str] = {}
    for spec in document.task.monitors.values():
        reference = spec.report_extension
        if reference is None:
            continue
        if reference not in extension_names:
            raise ComposeError(
                f"monitor {spec.name!r}.report_extension: expected an extension "
                f"name from {sorted(extension_names)}, got {reference!r}"
            )
        monitor_extensions[spec.name] = reference
    flow = _compose_flow(document.task.flow)
    return TaskPlan(
        monitors=monitors,
        extensions=extensions,
        retention=document.task.retention,
        scenario=document.scenario,
        monitor_extensions=MappingProxyType(dict(sorted(monitor_extensions.items()))),
        flow=flow,
    )


def _compose_flow(spec: FlowSpec) -> DefaultBenchmarkFlow:
    """Convert the document flow section into the live default flow."""
    rules = tuple(
        TerminationRule(
            monitor=rule.monitor,
            value=rule.value,
            outcome=rule.outcome,
            code=rule.code,
        )
        for rule in spec.termination_rules
    )
    events = tuple(
        EventRule(
            monitor=rule.monitor,
            value=rule.value,
            name=rule.name,
            severity=rule.severity,
            code=rule.code,
            data=rule.data,
        )
        for rule in spec.events
    )
    try:
        return DefaultBenchmarkFlow(rules, events=events)
    except (TypeError, ValueError) as exc:
        raise ComposeError(f"flow: {exc}") from exc


def _compose_monitor(spec: MonitorSpec) -> Monitor:
    if spec.builtin is not None:
        if spec.builtin == "predicate":
            return _compose_predicate(spec)
        if spec.builtin == "phase":
            return _compose_phase(spec)
        raise ComposeError(
            f"monitor {spec.name!r}.builtin: unknown builtin {spec.builtin!r}; "
            f"known monitor builtins are {list(MONITOR_BUILTINS)}"
        )
    if spec.module is not None:
        return _compose_module_monitor(spec)
    return _compose_factory_monitor(spec)


def _compose_predicate(spec: MonitorSpec) -> PredicateMonitor:
    where = f"monitor {spec.name!r}"
    if spec.when is None or spec.trigger is None:
        raise ComposeError(f"{where}: builtin 'predicate' requires 'when' and 'trigger'")
    if spec.params:
        raise ComposeError(f"{where}.params: builtin 'predicate' takes no params")
    if spec.resources:
        raise ComposeError(f"{where}.resources: builtin 'predicate' does not consume resources")
    _reject_query_sources(spec.when, f"{where}.when")
    try:
        condition = parse_condition(dict(spec.when))
    except (ConditionError, TypeError) as exc:
        raise ComposeError(f"{where}.when: {exc}") from exc
    trigger = _parse_trigger(spec.trigger, f"{where}.trigger")
    emit = _parse_emit(spec.emit, f"{where}.emit")
    return PredicateMonitor(spec.name, condition, trigger, emit)


def _compose_phase(spec: MonitorSpec) -> PhaseMonitor:
    where = f"monitor {spec.name!r}"
    if spec.source is None:
        raise ComposeError(f"{where}.source: builtin 'phase' requires a source")
    if spec.resources:
        raise ComposeError(f"{where}.resources: builtin 'phase' does not consume resources")
    _reject_query_source(spec.source, f"{where}.source")
    allowed = {"boundaries", "phase_names"}
    unknown = sorted(set(spec.params) - allowed)
    if unknown:
        raise ComposeError(
            f"{where}.params: unknown key(s) {unknown}; expected only {sorted(allowed)}"
        )
    if "boundaries" not in spec.params:
        raise ComposeError(f"{where}.params.boundaries: required for builtin 'phase'")
    try:
        return PhaseMonitor(
            spec.name,
            spec.source,
            boundaries=cast(Sequence[float], spec.params["boundaries"]),
            phase_names=cast(Sequence[str] | None, spec.params.get("phase_names")),
        )
    except (TypeError, ValueError) as exc:
        raise ComposeError(f"{where}: {exc}") from exc


def _compose_module_monitor(spec: MonitorSpec) -> Monitor:
    where = f"monitor {spec.name!r}"
    _reject_module_resources(spec, where)
    target = _import_callable(spec.module, f"{where}.module")
    return _call_and_validate_monitor(target, spec, where)


def _compose_factory_monitor(spec: MonitorSpec) -> Monitor:
    where = f"monitor {spec.name!r}"
    if spec.factory is None:
        raise ComposeError(f"{where}: expected a factory callable")
    _reject_module_resources(spec, where)
    return _call_and_validate_monitor(spec.factory, spec, where)


def _reject_module_resources(spec: MonitorSpec, where: str) -> None:
    """Reject unconsumed resources before any module import or factory call."""
    if spec.resources:
        raise ComposeError(
            f"{where}.resources: module/factory declarations do not consume resources"
        )


def _call_and_validate_monitor(
    factory: Callable[..., object], spec: MonitorSpec, where: str
) -> Monitor:
    result = _call_factory(factory, spec, where)
    if not isinstance(result, Monitor):
        raise ComposeError(f"{where}: factory returned {type(result).__name__}, expected a Monitor")
    return result


def _compose_extensions(document: TaskDocument) -> tuple[ReportExtension, ...]:
    declared: list[ReportExtension] = []
    coverage_declared = False
    task_extensions = document.task.extensions
    for spec in task_extensions.values():
        if spec.builtin is not None:
            if spec.builtin == "scenario_coverage":
                declared.append(_compose_scenario_coverage(spec, document))
                coverage_declared = True
                continue
            raise ComposeError(
                f"extension {spec.name!r}.builtin: unknown builtin {spec.builtin!r}; "
                f"known extension builtins are {list(EXTENSION_BUILTINS)}"
            )
        declared.append(_compose_module_extension(spec))
    if document.scenario is not None and not coverage_declared:
        if SCENARIO_COVERAGE_NAME in task_extensions:
            raise ComposeError(
                f"extension {SCENARIO_COVERAGE_NAME!r} is declared with a different "
                "builtin but a document scenario auto-composes scenario coverage; "
                f"rename that extension or declare builtin 'scenario_coverage' "
                f"under {SCENARIO_COVERAGE_NAME!r}"
            )
        declared.append(_build_scenario_coverage(SCENARIO_COVERAGE_NAME, document.scenario))
    try:
        return validate_report_extensions(declared)
    except (TypeError, ValueError) as exc:
        raise ComposeError(f"report extension composition: {exc}") from exc


def _compose_scenario_coverage(
    spec: ExtensionSpec, document: TaskDocument
) -> ScenarioCoverageExtension:
    where = f"extension {spec.name!r}"
    if spec.params or spec.resources:
        raise ComposeError(
            f"{where}: builtin 'scenario_coverage' takes its command pool from "
            "the document scenario; remove params/resources"
        )
    if document.scenario is None:
        raise ComposeError(
            f"{where}.builtin: 'scenario_coverage' requires a document scenario "
            "with a command_pool"
        )
    return _build_scenario_coverage(spec.name, document.scenario)


def _build_scenario_coverage(name: str, scenario: ScenarioSpec) -> ScenarioCoverageExtension:
    try:
        validated = Scenario.from_spec(scenario)
        return ScenarioCoverageExtension(name, commands=validated.commands)
    except (TypeError, ValueError, SpecError) as exc:
        raise ComposeError(
            f"scenario coverage {name!r}: cannot compose from document scenario: {exc}"
        ) from exc


def _compose_module_extension(spec: ExtensionSpec) -> ReportExtension:
    where = f"extension {spec.name!r}"
    if spec.module is None:
        raise ComposeError(f"{where}: expected a module path")
    if spec.resources:
        raise ComposeError(f"{where}.resources: module declarations do not consume resources")
    target = _import_callable(spec.module, f"{where}.module")
    result = _call_factory(target, spec, where)
    if not isinstance(result, ReportExtension):
        raise ComposeError(
            f"{where}: factory returned {type(result).__name__}, expected a " "ReportExtension"
        )
    return result


def _call_factory(
    factory: Callable[..., object], spec: MonitorSpec | ExtensionSpec, where: str
) -> object:
    params = dict(spec.params)
    try:
        return factory(**params)
    except Exception as exc:
        raise ComposeError(
            f"{where}: factory call failed with {type(exc).__name__}: {exc}"
        ) from exc


def _import_callable(path: str | None, where: str) -> Callable[..., object]:
    if not isinstance(path, str):
        raise ComposeError(f"{where}: expected a 'pkg.mod:attr' path, got {path!r}")
    module_name, _, attribute_path = path.partition(":")
    try:
        target: object = importlib.import_module(module_name)
    except Exception as exc:
        raise ComposeError(f"{where}={path!r} could not be imported: {exc}") from exc
    for attribute in attribute_path.split("."):
        try:
            target = getattr(target, attribute)
        except AttributeError as exc:
            raise ComposeError(
                f"{where}={path!r} could not be resolved: module {module_name!r} "
                f"has no attribute {attribute!r}"
            ) from exc
    if not callable(target):
        raise ComposeError(f"{where}={path!r} is not callable")
    return cast(Callable[..., object], target)


def _parse_trigger(raw: str | Mapping[str, object], where: str) -> TriggerSpec:
    try:
        if isinstance(raw, str):
            return TriggerSpec.from_mapping({"kind": raw})
        return TriggerSpec.from_mapping(raw)
    except (TypeError, ValueError) as exc:
        raise ComposeError(f"{where}: {exc}") from exc


def _parse_emit(raw: Mapping[str, object] | None, where: str) -> EmitSpec:
    try:
        return EmitSpec.from_mapping({} if raw is None else raw)
    except (TypeError, ValueError) as exc:
        raise ComposeError(f"{where}: {exc}") from exc


def _reject_query_sources(document: Mapping[str, object], where: str) -> None:
    for key, value in document.items():
        if key == "source" and isinstance(value, str):
            _reject_query_source(value, f"{where}.source")
        elif isinstance(value, Mapping):
            _reject_query_sources(cast(Mapping[str, object], value), f"{where}.{key}")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, item in enumerate(value):
                if isinstance(item, Mapping):
                    _reject_query_sources(
                        cast(Mapping[str, object], item), f"{where}.{key}[{index}]"
                    )


def _reject_query_source(text: str, where: str) -> None:
    try:
        ref = SourceRef.parse(text)
    except SourceError as exc:
        raise ComposeError(f"{where}: {exc}") from exc
    if ref.kind == "query":
        raise SourceUnavailableError(
            f"{where}: query source {ref.key!r} is unavailable at composition: "
            "use record:, obs: or value: instead"
        )


def _require_unique_names(names: Sequence[str], kind: str, what: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ComposeError(
            f"{what}: duplicate {kind} name(s) {sorted(duplicates)}; task names "
            "must be unique within their namespace"
        )


def _validate_declared_names(
    specs: Sequence[MonitorSpec | ExtensionSpec],
    objects: Sequence[Monitor | ReportExtension],
    kind: str,
) -> None:
    for spec, live in zip(specs, objects):
        name = getattr(live, "name", None)
        if name != spec.name:
            raise ComposeError(
                f"{kind} {spec.name!r}: factory returned {kind} named {name!r}; "
                f"the declaration name is {spec.name!r}"
            )
