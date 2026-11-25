"""Task-document composition: registry closure, live objects and scenario policy."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
import yaml
from unigym_bench.compose import (
    EXTENSION_BUILTINS,
    MONITOR_BUILTINS,
    SCENARIO_COVERAGE_NAME,
    ComposeError,
    TaskPlan,
    compose_task,
)
from unigym_bench.monitors import PhaseMonitor, PredicateMonitor
from unigym_bench.report import ScenarioCoverageExtension
from unigym_bench.sources import SourceUnavailableError
from unigym_bench.specs import (
    ExtensionSpec,
    MonitorSpec,
    RetentionSpec,
    SpecError,
    parse_task_document,
)


def _make_monitor() -> PredicateMonitor:
    return PredicateMonitor(
        "contact",
        {"source": "value:force", "op": "gte", "value": 1.0},
        "edge",
    )


def _make_named_monitor(name: str) -> PredicateMonitor:
    return PredicateMonitor(
        name,
        {"source": "value:force", "op": "gte", "value": 1.0},
        "edge",
    )


def _document(
    *,
    monitors: object | None = None,
    extensions: object | None = None,
    scenario: object | None = None,
    schema: object | None = None,
) -> dict[str, object]:
    document: dict[str, object] = {
        "name": "reach",
        "monitors": (
            {
                "contact": {
                    "builtin": "predicate",
                    "when": {"source": "value:force", "op": "gte", "value": 1.0},
                    "trigger": "edge",
                }
            }
            if monitors is None
            else monitors
        ),
    }
    if extensions is not None:
        document["extensions"] = extensions
    if scenario is not None:
        document["scenario"] = scenario
    if schema is not None:
        document["schema"] = schema
    return document


def _scenario() -> dict[str, object]:
    return {"command_pool": [[0.0], [1.0]]}


def test_compose_predicate_document_to_a_task_plan() -> None:
    document = parse_task_document(_document())

    plan = compose_task(document)

    assert isinstance(plan, TaskPlan)
    assert [type(monitor) for monitor in plan.monitors] == [PredicateMonitor]
    assert plan.monitors[0].name == "contact"
    assert plan.extensions == ()
    assert plan.retention == RetentionSpec()
    assert plan.scenario is None
    assert dict(plan.monitor_extensions) == {}


def test_compose_rejects_a_raw_mapping() -> None:
    with pytest.raises(TypeError, match="expected a TaskDocument"):
        compose_task(_document())  # type: ignore[arg-type]


def test_compose_phase_monitor_from_source_and_boundaries() -> None:
    document = parse_task_document(
        _document(
            monitors={
                "height_phase": {
                    "builtin": "phase",
                    "source": "value:height",
                    "params": {"boundaries": [0.0, 1.0]},
                }
            }
        )
    )

    plan = compose_task(document)

    assert isinstance(plan.monitors[0], PhaseMonitor)
    assert plan.monitors[0].name == "height_phase"


def test_compose_module_monitor_imports_at_composition_and_passes_params() -> None:
    document = parse_task_document(
        _document(
            monitors={
                "contact": {
                    "module": "unigym_bench.monitors:threshold_config",
                    "params": {
                        "name": "contact",
                        "source": "value:force",
                        "threshold": 2.0,
                    },
                }
            }
        )
    )

    plan = compose_task(document)

    assert isinstance(plan.monitors[0], PredicateMonitor)
    assert plan.monitors[0].report_view()["condition"] == {
        "source": "value:force",
        "op": "gte",
        "value": 2.0,
    }


def test_compose_api_factory_and_report_extension_reference() -> None:
    document = parse_task_document(
        _document(
            monitors={
                "contact": {
                    "factory": _make_monitor,
                    "report_extension": "contact_report",
                }
            },
            extensions={
                "contact_report": {
                    "module": "unigym_bench.report:ScenarioCoverageExtension",
                    "params": {"name": "contact_report", "commands": [[0.0]]},
                }
            },
        )
    )

    plan = compose_task(document)

    assert isinstance(plan.monitors[0], PredicateMonitor)
    assert isinstance(plan.extensions[0], ScenarioCoverageExtension)
    assert dict(plan.monitor_extensions) == {"contact": "contact_report"}


def test_compose_factory_receives_json_params_as_keywords() -> None:
    document = parse_task_document(
        _document(
            monitors={
                "contact": {
                    "factory": _make_named_monitor,
                    "params": {"name": "contact"},
                }
            }
        )
    )

    plan = compose_task(document)

    assert plan.monitors[0].name == "contact"


def test_compose_scenario_auto_composes_coverage_without_an_extension() -> None:
    document = parse_task_document(_document(scenario=_scenario()))

    plan = compose_task(document)

    assert len(plan.extensions) == 1
    extension = plan.extensions[0]
    assert isinstance(extension, ScenarioCoverageExtension)
    assert extension.name == SCENARIO_COVERAGE_NAME
    assert plan.scenario == document.scenario
    block = extension.on_run_end(_empty_summary())
    assert block["pool_size"] == 2
    assert block["covered"] == 0


def test_compose_explicit_scenario_coverage_extension_uses_its_declared_name() -> None:
    document = parse_task_document(
        _document(
            extensions={"coverage": {"builtin": "scenario_coverage"}},
            scenario=_scenario(),
        )
    )

    plan = compose_task(document)

    assert [extension.name for extension in plan.extensions] == ["coverage"]
    assert SCENARIO_COVERAGE_NAME not in [extension.name for extension in plan.extensions]


def test_compose_scenario_coverage_requires_a_document_scenario() -> None:
    document = parse_task_document(
        _document(extensions={"coverage": {"builtin": "scenario_coverage"}})
    )

    with pytest.raises(ComposeError, match="requires a document scenario"):
        compose_task(document)


def test_compose_scenario_name_collision_fails_fast() -> None:
    document = parse_task_document(
        _document(
            extensions={
                SCENARIO_COVERAGE_NAME: {
                    "module": "unigym_bench.report:ScenarioCoverageExtension",
                    "params": {
                        "name": SCENARIO_COVERAGE_NAME,
                        "commands": [[0.0]],
                    },
                }
            },
            scenario=_scenario(),
        )
    )

    with pytest.raises(ComposeError, match="auto-composes scenario coverage"):
        compose_task(document)


def test_compose_rejects_scenario_coverage_params() -> None:
    document = parse_task_document(
        _document(
            extensions={
                "coverage": {
                    "builtin": "scenario_coverage",
                    "params": {"commands": [[0.0]]},
                }
            },
            scenario=_scenario(),
        )
    )

    with pytest.raises(ComposeError, match="takes its command pool"):
        compose_task(document)


@pytest.mark.parametrize(
    ("kind", "declaration", "fragment"),
    [
        (
            "monitor",
            {"builtin": "threshold"},
            str(list(MONITOR_BUILTINS)),
        ),
        (
            "extension",
            {"builtin": "contact"},
            str(list(EXTENSION_BUILTINS)),
        ),
    ],
)
def test_compose_unknown_builtin_names_the_registry(
    kind: str, declaration: dict[str, object], fragment: str
) -> None:
    if kind == "monitor":
        document = parse_task_document(_document(monitors={"m": declaration}))
    else:
        document = parse_task_document(_document(extensions={"e": declaration}))

    with pytest.raises(ComposeError) as excinfo:
        compose_task(document)

    assert "unknown builtin" in str(excinfo.value)
    assert fragment in str(excinfo.value)


def test_compose_rejects_query_sources_at_composition() -> None:
    predicate_doc = parse_task_document(
        _document(
            monitors={
                "m": {
                    "builtin": "predicate",
                    "when": {"source": "query:height", "op": "gt", "value": 0.0},
                    "trigger": "edge",
                }
            }
        )
    )
    phase_doc = parse_task_document(
        _document(
            monitors={
                "m": {
                    "builtin": "phase",
                    "source": "query:height",
                    "params": {"boundaries": [0.0]},
                }
            }
        )
    )

    with pytest.raises(SourceUnavailableError, match="query source"):
        compose_task(predicate_doc)
    with pytest.raises(SourceUnavailableError, match="query source"):
        compose_task(phase_doc)


def test_compose_rejects_live_name_mismatch_and_duplicate_extensions() -> None:
    mismatched = parse_task_document(
        _document(monitors={"contact": {"factory": lambda: _make_named_monitor("other")}})
    )
    with pytest.raises(ComposeError, match="declaration name"):
        compose_task(mismatched)

    duplicate = parse_task_document(
        _document(
            extensions={
                "one": {
                    "module": "unigym_bench.report:ScenarioCoverageExtension",
                    "params": {"name": "one", "commands": [[0.0]]},
                },
                "two": {
                    "module": "unigym_bench.report:ScenarioCoverageExtension",
                    "params": {"name": "one", "commands": [[0.0]]},
                },
            }
        )
    )
    with pytest.raises(ComposeError, match="duplicate extension name"):
        compose_task(duplicate)


def test_compose_module_errors_are_teaching_errors() -> None:
    document = parse_task_document(
        _document(monitors={"m": {"module": "no_such_pack_module:make"}})
    )

    with pytest.raises(ComposeError, match="could not be imported"):
        compose_task(document)


def test_compose_keeps_the_document_retention_policy() -> None:
    document = parse_task_document(
        {
            **_document(scenario=_scenario()),
            "retention": {"retain_episodes": True, "max_events": 5, "event_log": False},
        }
    )

    plan = compose_task(document)

    assert plan.retention == RetentionSpec(retain_episodes=True, max_events=5, event_log=False)


def test_yaml_cannot_declare_a_callable_factory(tmp_path: Path) -> None:
    path = tmp_path / "task.yaml"
    path.write_text(
        yaml.safe_dump(_document(monitors={"m": {"factory": "pkg.mod:make"}})),
        encoding="utf-8",
    )

    with pytest.raises(SpecError, match="expected a callable factory"):
        parse_task_document(yaml.safe_load(path.read_text(encoding="utf-8")))

    tagged = tmp_path / "tagged.yaml"
    tagged.write_text(
        """
name: tagged
monitors:
  m:
    factory: !!python/name:os.system
""",
        encoding="utf-8",
    )
    from unigym_bench.specs import load_task_document

    with pytest.raises(SpecError, match="invalid YAML"):
        load_task_document(tagged)


def _empty_summary() -> object:
    from unigym_bench.events import EpisodeSummary
    from unigym_bench.report import RunMetadata, RunSummary

    return RunSummary(
        metadata=RunMetadata(),
        episodes=(),
        episode_summary=EpisodeSummary(),
        monitor_views={},
    )


def test_compose_does_not_accept_extension_specs_without_a_builtin_or_module() -> None:
    with pytest.raises(SpecError, match="expected exactly one"):
        ExtensionSpec(name="coverage")

    with pytest.raises(SpecError, match="expected exactly one"):
        MonitorSpec(name="m")


@pytest.mark.parametrize(
    "declaration",
    [
        {
            "builtin": "predicate",
            "when": {"source": "value:force", "op": "gte", "value": 1.0},
            "trigger": "edge",
            "resources": {"levels": "data/levels.yaml"},
        },
        {
            "builtin": "phase",
            "source": "value:height",
            "params": {"boundaries": [0.0]},
            "resources": {"levels": "data/levels.yaml"},
        },
    ],
)
def test_compose_rejects_resources_on_builtin_monitors(declaration, tmp_path) -> None:
    document = parse_task_document(_document(monitors={"m": declaration}), base_dir=tmp_path)

    with pytest.raises(ComposeError, match="does not consume resources"):
        compose_task(document)


def test_compose_rejects_module_resources_before_import(tmp_path, monkeypatch) -> None:
    marker = tmp_path / "imported.marker"
    module_name = f"p16_side_effect_{uuid.uuid4().hex}"
    (tmp_path / f"{module_name}.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported')\n"
        "def make():\n"
        "    return None\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    document = parse_task_document(
        _document(
            monitors={
                "m": {
                    "module": f"{module_name}:make",
                    "resources": {"levels": "data/levels.yaml"},
                }
            }
        ),
        base_dir=tmp_path,
    )

    with pytest.raises(ComposeError, match="do not consume resources"):
        compose_task(document)

    assert not marker.exists()


def test_compose_rejects_factory_resources_before_calling_the_factory(tmp_path) -> None:
    calls: list[str] = []

    def factory() -> PredicateMonitor:
        calls.append("called")
        return _make_named_monitor("m")

    document = parse_task_document(
        _document(
            monitors={
                "m": {
                    "factory": factory,
                    "resources": {"levels": "data/levels.yaml"},
                }
            }
        ),
        base_dir=tmp_path,
    )

    with pytest.raises(ComposeError, match="do not consume resources"):
        compose_task(document)

    assert calls == []
