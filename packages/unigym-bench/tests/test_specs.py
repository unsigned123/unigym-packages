"""Task-document schema: valid documents, fail-fast errors and read-only data."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path
from types import MappingProxyType

import pytest
import yaml

from unigym_bench.specs import (
    SCHEMA,
    ExtensionSpec,
    MonitorSpec,
    RetentionSpec,
    ScenarioSpec,
    SpecError,
    TaskDocument,
    TaskSpec,
    load_task_document,
    parse_task_document,
)


def _monitor_declaration() -> dict[str, object]:
    return {
        "builtin": "predicate",
        "when": {"source": "value:force", "op": "gte", "value": 1.0},
        "trigger": "edge",
    }


def _monitor_spec(name: str, **overrides: object) -> MonitorSpec:
    values: dict[str, object] = {"name": name, **_monitor_declaration()}
    values.update(overrides)
    return MonitorSpec(**values)


def _document(**overrides: object) -> dict[str, object]:
    document: dict[str, object] = {
        "name": "reach",
        "monitors": {"grasp": _monitor_declaration()},
    }
    document.update(overrides)
    return document


def _write(tmp_path: Path, text: str, name: str = "task.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_minimal_document() -> None:
    document = parse_task_document(_document())

    assert isinstance(document, TaskDocument)
    assert isinstance(document.task, TaskSpec)
    assert document.task.name == "reach"
    assert list(document.task.monitors) == ["grasp"]
    monitor = document.task.monitors["grasp"]
    assert monitor.name == "grasp"
    assert monitor.builtin == "predicate"
    assert monitor.module is None
    assert monitor.factory is None
    assert monitor.report_extension is None
    assert monitor.params == {}
    assert monitor.resources == {}
    assert monitor.when == {"source": "value:force", "op": "gte", "value": 1.0}
    assert monitor.trigger == "edge"
    assert monitor.emit is None
    assert monitor.source is None
    assert document.task.extensions == {}
    assert document.task.retention == RetentionSpec()
    assert document.scenario is None
    assert document.schema is None
    assert document.base_dir is None
    assert document.source is None
    assert isinstance(monitor.params, MappingProxyType)
    assert isinstance(document.task.monitors, MappingProxyType)


def test_parse_full_document() -> None:
    document = parse_task_document(
        {
            "schema": SCHEMA,
            "name": "grasp",
            "monitors": {
                "contact": {
                    "builtin": "predicate",
                    "when": {"source": "value:force", "op": "gte", "value": 0.5},
                    "trigger": {"kind": "hold", "steps": 3},
                    "emit": {
                        "event": "contact",
                        "severity": "warning",
                        "terminal_reason": "hit",
                        "terminal_outcome": "failure",
                    },
                    "report_extension": "contact_report",
                },
                "custom": {"module": "my_pkg.monitors:make", "params": {}},
            },
            "extensions": {
                "contact_report": {
                    "module": "my_pkg.report:make_contact",
                    "params": {"bins": [1, 2]},
                },
                "summary": {"module": "my_pkg.report:make_summary"},
            },
            "retention": {"retain_episodes": True, "max_events": 10, "event_log": False},
            "scenario": {
                "command_pool": [[0.1, 0.2], [0.3, 0.4]],
                "plan": [1, 0, 1],
            },
        }
    )

    assert document.schema == SCHEMA
    assert document.task.retention == RetentionSpec(
        retain_episodes=True, max_events=10, event_log=False
    )
    contact = document.task.monitors["contact"]
    assert contact.report_extension == "contact_report"
    assert contact.when["value"] == 0.5
    assert contact.trigger["kind"] == "hold"
    assert contact.trigger["steps"] == 3
    assert contact.emit["terminal_outcome"] == "failure"
    assert document.task.extensions["contact_report"].params["bins"] == (1, 2)
    assert document.scenario == ScenarioSpec(command_pool=((0.1, 0.2), (0.3, 0.4)), plan=(1, 0, 1))


def test_load_document_records_provenance(tmp_path: Path) -> None:
    path = _write(tmp_path, yaml.safe_dump(_document()))

    document = load_task_document(path)

    assert document.source == path
    assert document.base_dir == tmp_path
    assert document.task.name == "reach"


@pytest.mark.parametrize(
    "document",
    [
        _document(),
        _document(extensions={"report": {"builtin": "summary"}}),
        _document(extensions={"report": {"module": "pkg.ext:make"}}),
        _document(monitors={"custom": {"module": "pkg.monitors:make"}}),
        _document(monitors={"custom": {"factory": len}}),
        _document(monitors={"m": {"module": "pkg.monitors:make", "params": {"a": [1, 2]}}}),
        _document(retention={"retain_episodes": True}),
        _document(scenario={"command_pool": [[0.0], [1.0]]}),
        _document(scenario={"command_pool": [[0, 1]], "plan": [0]}),
    ],
)
def test_valid_documents(document: dict[str, object]) -> None:
    parsed = parse_task_document(document)

    assert parsed.task.name == document["name"]


@pytest.mark.parametrize(
    ("document", "fragment"),
    [
        (None, "expected a mapping"),
        ([], "expected a mapping"),
        ("text", "expected a mapping"),
        ({"name": "x", "monitors": {}, "extra": 1}, "unknown key(s) 'extra'"),
        ({"name": "x"}, "field 'monitors' is required"),
        ({"name": "x", "monitors": None}, "field 'monitors' is required"),
        ({"monitors": {}}, "task document.name"),
        ({"name": "", "monitors": {}}, "task document.name"),
        ({"name": 123, "monitors": {}}, "task document.name"),
        ({"name": "x", "monitors": []}, "expected a mapping of name"),
        ({"name": "x", "monitors": {"Bad": {"builtin": "b"}}}, "expected a name matching"),
        ({"name": "x", "monitors": {"m": []}}, "expected a mapping with"),
        ({"name": "x", "monitors": {"m": {}}}, "expected exactly one"),
        (
            {"name": "x", "monitors": {"m": {"builtin": "b", "module": "pkg.m:f"}}},
            "expected exactly one of 'builtin', 'module' or 'factory'",
        ),
        (
            {"name": "x", "monitors": {"m": {"builtin": "b", "factory": len}}},
            "expected exactly one of 'builtin', 'module' or 'factory'",
        ),
        (
            {"name": "x", "monitors": {"m": {"builtin": "b", "extra": 1}}},
            "unknown key(s) 'extra'",
        ),
        (
            {"name": "x", "monitors": {"m": {"name": "other", "builtin": "b"}}},
            "the mapping key 'm' is the name",
        ),
        (
            {"name": "x", "monitors": {"m": {"builtin": "b"}}, "extensions": {"e": {}}},
            "expected exactly one of 'builtin' or 'module'",
        ),
        (
            {
                "name": "x",
                "monitors": {"m": {"builtin": "b"}},
                "extensions": {"e": {"builtin": "b", "module": "pkg.e:f"}},
            },
            "expected exactly one of 'builtin' or 'module'",
        ),
        (
            {
                "name": "x",
                "monitors": {"m": {"builtin": "b"}},
                "extensions": {"e": {"builtin": "b", "factory": len}},
            },
            "unknown key(s) 'factory'",
        ),
        (
            {
                "name": "x",
                "monitors": {"m": {"builtin": "b", "report_extension": "missing"}},
            },
            "expected an existing extension name",
        ),
        (
            {
                "name": "x",
                "monitors": {"m": {"builtin": "b", "report_extension": 5}},
            },
            "expected a name matching",
        ),
        (
            {"name": "x", "monitors": {"m": {"builtin": "b"}}, "schema": "other/1"},
            "task document.schema",
        ),
        (
            {"name": "x", "monitors": {"m": {"builtin": "b"}}, "retention": []},
            "task.retention",
        ),
        (
            {
                "name": "x",
                "monitors": {"m": {"builtin": "b"}},
                "retention": {"unknown": 1},
            },
            "unknown key(s) 'unknown'",
        ),
        (
            {
                "name": "x",
                "monitors": {"m": {"builtin": "b"}},
                "scenario": {"plan": [0]},
            },
            "command_pool: required",
        ),
        (
            {
                "name": "x",
                "monitors": {"m": {"builtin": "b"}},
                "scenario": {"command_pool": [[0.0]], "seed": 1},
            },
            "unknown key(s) 'seed'",
        ),
    ],
)
def test_invalid_documents(document: object, fragment: str) -> None:
    with pytest.raises(SpecError) as excinfo:
        parse_task_document(document)

    assert fragment in str(excinfo.value)


@pytest.mark.parametrize(
    "name",
    ["Bad", "bad-name", "1bad", "bad name", "_bad", "", "bad\n"],
)
def test_rejects_bad_monitor_names(name: str) -> None:
    with pytest.raises(SpecError) as excinfo:
        parse_task_document(_document(monitors={name: _monitor_declaration()}))

    assert "expected a name matching [a-z][a-z0-9_]*" in str(excinfo.value)


@pytest.mark.parametrize(
    "reference",
    [
        "pkg.mod",
        "pkg..mod:attr",
        ":attr",
        "pkg.mod:",
        "pkg.mod:attr:extra",
        "1pkg:attr",
        "pkg.mod:1attr",
        "pkg-mod:attr",
        "",
        123,
    ],
)
def test_rejects_bad_module_paths(reference: object) -> None:
    with pytest.raises(SpecError) as excinfo:
        parse_task_document(_document(monitors={"m": {"module": reference}}))

    assert "expected a module path 'pkg.mod:attr'" in str(excinfo.value)


@pytest.mark.parametrize(
    ("params", "fragment"),
    [
        (None, "expected a JSON object (mapping)"),
        ([], "expected a JSON object (mapping)"),
        ("text", "expected a JSON object (mapping)"),
        ({"x": object()}, "expected a JSON value"),
        ({1: "x"}, "mapping keys must be strings"),
        ({"x": math.nan}, "float must be finite"),
        ({"x": math.inf}, "float must be finite"),
        ({"x": {"y": object()}}, "expected a JSON value"),
    ],
)
def test_rejects_non_json_params(params: object, fragment: str) -> None:
    with pytest.raises(SpecError) as excinfo:
        parse_task_document(_document(monitors={"m": {"builtin": "b", "params": params}}))

    message = str(excinfo.value)
    assert "params" in message
    assert fragment in message


@pytest.mark.parametrize(
    ("retention", "fragment"),
    [
        ({"retain_episodes": "yes"}, "expected a bool"),
        ({"retain_episodes": 1}, "expected a bool"),
        ({"max_events": -1}, "expected an int >= 0"),
        ({"max_events": 1.5}, "expected an int"),
        ({"max_events": True}, "expected an int"),
        ({"event_log": 0}, "expected a bool"),
        ({"event_log": "false"}, "expected a bool"),
    ],
)
def test_rejects_bad_retention(retention: object, fragment: str) -> None:
    with pytest.raises(SpecError) as excinfo:
        parse_task_document(_document(retention=retention))

    assert "retention." in str(excinfo.value)
    assert fragment in str(excinfo.value)


@pytest.mark.parametrize(
    ("scenario", "fragment"),
    [
        ([], "expected a mapping"),
        ({"command_pool": []}, "at least one command vector"),
        ({"command_pool": [[]]}, "must not be empty"),
        ({"command_pool": [["a"]]}, "expected a finite number"),
        ({"command_pool": [[True]]}, "expected a finite number"),
        ({"command_pool": [[math.inf]]}, "expected a finite number"),
        ({"command_pool": [[0.0], [1.0, 2.0]]}, "expected 1 values"),
        ({"command_pool": [[0.0]], "plan": []}, "at least one index"),
        ({"command_pool": [[0.0]], "plan": [1]}, "expected an index in [0, 1)"),
        ({"command_pool": [[0.0]], "plan": [1.5]}, "expected an integer index"),
        ({"command_pool": [[0.0]], "plan": [True]}, "expected an integer index"),
    ],
)
def test_rejects_bad_scenario(scenario: object, fragment: str) -> None:
    with pytest.raises(SpecError) as excinfo:
        parse_task_document(_document(scenario=scenario))

    assert "scenario" in str(excinfo.value)
    assert fragment in str(excinfo.value)


def test_scenario_normalizes_numbers() -> None:
    parsed = parse_task_document(_document(scenario={"command_pool": [[0, 1.5]]}))

    assert parsed.scenario == ScenarioSpec(command_pool=((0.0, 1.5),))


def test_load_rejects_duplicate_monitor_names(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
name: duplicate
monitors:
  same:
    builtin: predicate
    when: {source: value:x, op: gt, value: 0.0}
    trigger: edge
  same:
    builtin: predicate
    when: {source: value:y, op: gt, value: 0.0}
    trigger: edge
""",
    )

    with pytest.raises(SpecError) as excinfo:
        load_task_document(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "duplicate mapping key 'same'" in message


def test_load_rejects_duplicate_extension_names(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
name: duplicate
monitors:
  m:
    builtin: predicate
    when: {source: value:x, op: gt, value: 0.0}
    trigger: edge
extensions:
  same:
    builtin: one
  same:
    builtin: two
""",
    )

    with pytest.raises(SpecError) as excinfo:
        load_task_document(path)

    assert "duplicate mapping key 'same'" in str(excinfo.value)


def test_load_rejects_duplicate_top_level_keys(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
name: first
name: second
monitors:
  m:
    builtin: predicate
    when: {source: value:x, op: gt, value: 0.0}
    trigger: edge
""",
    )

    with pytest.raises(SpecError) as excinfo:
        load_task_document(path)

    assert "duplicate mapping key 'name'" in str(excinfo.value)


def test_load_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "task.yaml"
    path.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(SpecError) as excinfo:
        load_task_document(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "cannot read task document" in message


@pytest.mark.parametrize(
    ("document", "fragment"),
    [
        (_document(schema=None), "task document.schema: explicit null"),
        (_document(retention=None), "task.retention: explicit null"),
        (_document(scenario=None), "scenario: explicit null"),
        (_document(extensions=None), "extensions: explicit null"),
    ],
)
def test_rejects_explicit_null_document_fields(document: dict[str, object], fragment: str) -> None:
    with pytest.raises(SpecError) as excinfo:
        parse_task_document(document)

    assert fragment in str(excinfo.value)


def test_rejects_explicit_null_resources(tmp_path: Path) -> None:
    document = _document()
    document["monitors"]["grasp"]["resources"] = None

    with pytest.raises(SpecError) as excinfo:
        parse_task_document(document, base_dir=tmp_path)

    assert ".resources: explicit null is not allowed" in str(excinfo.value)


def test_load_rejects_invalid_yaml(tmp_path: Path) -> None:
    path = _write(tmp_path, "name: [unclosed")

    with pytest.raises(SpecError) as excinfo:
        load_task_document(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "invalid YAML" in message


def test_load_rejects_empty_document(tmp_path: Path) -> None:
    path = _write(tmp_path, "")

    with pytest.raises(SpecError) as excinfo:
        load_task_document(path)

    assert str(path) in str(excinfo.value)
    assert "expected a mapping" in str(excinfo.value)


def test_load_rejects_non_finite_yaml_number(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
name: non_finite
monitors:
  m:
    builtin: predicate
    when:
      source: value:x
      op: gt
      value: .nan
    trigger: edge
""",
    )

    with pytest.raises(SpecError) as excinfo:
        load_task_document(path)

    assert "float must be finite" in str(excinfo.value)


def test_load_error_names_file_field_expected_and_actual(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
name: bad
monitors:
  grasp:
    builtin: predicate
    when: {source: value:x, op: gt, value: 0.0}
    trigger: edge
    module: pkg.mod:make
""",
    )

    with pytest.raises(SpecError) as excinfo:
        load_task_document(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "monitor 'grasp'" in message
    assert "expected exactly one" in message
    assert "got builtin, module" in message


def test_relative_resource_resolves_against_document_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks_dir = tmp_path / "tasks"
    data_dir = tasks_dir / "data"
    data_dir.mkdir(parents=True)
    resource = data_dir / "levels.yaml"
    resource.write_text("levels: [1]\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    document = _document()
    document["monitors"]["grasp"]["resources"] = {"levels": "data/levels.yaml"}
    path = tasks_dir / "task.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    monkeypatch.chdir(elsewhere)

    loaded = load_task_document(path)
    parsed = parse_task_document(document, base_dir=tasks_dir)

    expected = resource.resolve()
    assert loaded.task.monitors["grasp"].resources["levels"] == expected
    assert parsed.task.monitors["grasp"].resources["levels"] == expected
    assert not str(expected).startswith(str(elsewhere))


def test_parse_requires_base_dir_for_relative_resource() -> None:
    document = _document()
    document["monitors"]["grasp"]["resources"] = {"levels": "data/levels.yaml"}

    with pytest.raises(SpecError) as excinfo:
        parse_task_document(document)

    assert "requires the task document base directory" in str(excinfo.value)


def test_absolute_resource_is_preserved(tmp_path: Path) -> None:
    absolute = tmp_path / "levels.yaml"
    document = _document()
    document["monitors"]["grasp"]["resources"] = {"levels": str(absolute)}

    parsed = parse_task_document(document, base_dir=tmp_path)

    assert parsed.task.monitors["grasp"].resources["levels"] == absolute.resolve()


def test_parse_rejects_relative_base_dir() -> None:
    with pytest.raises(SpecError) as excinfo:
        parse_task_document(_document(), base_dir="relative")

    assert "expected an absolute path" in str(excinfo.value)


@pytest.mark.parametrize(
    ("resources", "fragment"),
    [
        ("data", "expected a mapping of resource name to path"),
        ({"Bad": "data/x"}, "expected a name matching"),
        ({"levels": 5}, "expected a non-empty path string"),
        ({"levels": ""}, "expected a non-empty path string"),
    ],
)
def test_rejects_bad_resources(resources: object, fragment: str, tmp_path: Path) -> None:
    document = _document()
    document["monitors"]["grasp"]["resources"] = resources

    with pytest.raises(SpecError) as excinfo:
        parse_task_document(document, base_dir=tmp_path)

    assert fragment in str(excinfo.value)


def test_specs_are_frozen_and_read_only(tmp_path: Path) -> None:
    document = parse_task_document(
        _document(
            monitors={
                "m": {
                    "module": "pkg.monitors:make",
                    "params": {"nested": {"levels": [1, 2]}},
                    "resources": {"levels": "data/levels.yaml"},
                }
            }
        ),
        base_dir=tmp_path,
    )
    task = document.task
    monitor = task.monitors["m"]

    assert is_dataclass(document)
    assert is_dataclass(task)
    assert is_dataclass(monitor)
    with pytest.raises(FrozenInstanceError):
        task.name = "changed"
    with pytest.raises(FrozenInstanceError):
        monitor.module = "pkg.monitors:other"
    with pytest.raises(TypeError):
        task.monitors["m"] = monitor
    with pytest.raises(TypeError):
        monitor.params["new"] = 1
    with pytest.raises(TypeError):
        monitor.resources["new"] = tmp_path / "new"
    assert isinstance(monitor.params, MappingProxyType)
    assert isinstance(monitor.params["nested"], MappingProxyType)
    assert monitor.params["nested"]["levels"] == (1, 2)


def test_parse_does_not_alias_the_source_document() -> None:
    source = _document()
    source["monitors"]["grasp"]["when"] = {"all": [{"source": "value:x", "op": "gt", "value": 1.0}]}
    parsed = parse_task_document(source)

    source["name"] = "changed"
    source["monitors"]["grasp"]["builtin"] = "phase"
    source["monitors"]["grasp"]["when"]["all"].append(
        {"source": "value:y", "op": "lt", "value": 2.0}
    )

    monitor = parsed.task.monitors["grasp"]
    assert parsed.task.name == "reach"
    assert monitor.builtin == "predicate"
    assert len(monitor.when["all"]) == 1


def test_programmatic_specs_validate_invariants() -> None:
    with pytest.raises(SpecError) as excinfo:
        _monitor_spec("Bad")
    assert "expected a name matching" in str(excinfo.value)

    with pytest.raises(SpecError) as excinfo:
        _monitor_spec("ok", module="pkg.mod:make")
    assert "expected exactly one" in str(excinfo.value)

    with pytest.raises(SpecError) as excinfo:
        MonitorSpec(name="ok")
    assert "expected exactly one" in str(excinfo.value)

    with pytest.raises(SpecError) as excinfo:
        MonitorSpec(name="ok", factory="pkg.mod:make")
    assert "use 'module' for an import path" in str(excinfo.value)

    with pytest.raises(SpecError) as excinfo:
        ExtensionSpec(name="ok", builtin="one", module="pkg.mod:make")
    assert "expected exactly one of 'builtin' or 'module'" in str(excinfo.value)

    assert MonitorSpec(name="ok", factory=len).factory is len


def test_programmatic_task_rejects_key_name_mismatch() -> None:
    monitor = _monitor_spec("other")
    with pytest.raises(SpecError) as excinfo:
        TaskSpec(name="task", monitors={"key": monitor})
    assert "the mapping key is the monitor name" in str(excinfo.value)

    extension = ExtensionSpec(name="other", builtin="summary")
    with pytest.raises(SpecError) as excinfo:
        TaskSpec(
            name="task",
            monitors={"m": _monitor_spec("m")},
            extensions={"key": extension},
        )
    assert "the mapping key is the extension name" in str(excinfo.value)


def test_programmatic_task_rejects_missing_extension_reference() -> None:
    monitor = _monitor_spec("m", report_extension="missing")

    with pytest.raises(SpecError) as excinfo:
        TaskSpec(name="task", monitors={"m": monitor})

    message = str(excinfo.value)
    assert "task.monitors.m.report_extension" in message
    assert "expected an existing extension name" in message


def test_programmatic_task_document_schema_is_closed() -> None:
    task = TaskSpec(name="task", monitors={"m": _monitor_spec("m")})

    assert TaskDocument(task=task, schema=SCHEMA).schema == SCHEMA
    with pytest.raises(SpecError) as excinfo:
        TaskDocument(task=task, schema="other/1")
    assert "expected 'unigym_extbench/2'" in str(excinfo.value)
    with pytest.raises(SpecError) as excinfo:
        TaskDocument(task=task, schema="unigym_extbench/1")
    assert "expected 'unigym_extbench/2'" in str(excinfo.value)


def test_schema_is_two_and_rejects_the_old_explicit_schema() -> None:
    assert SCHEMA == "unigym_extbench/2"
    assert parse_task_document(_document()).schema is None
    assert parse_task_document(_document(schema=SCHEMA)).schema == SCHEMA

    with pytest.raises(SpecError) as excinfo:
        parse_task_document(_document(schema="unigym_extbench/1"))

    assert "task document.schema" in str(excinfo.value)
    assert "unigym_extbench/2" in str(excinfo.value)


def test_predicate_fields_are_frozen_and_validate_trigger_emit() -> None:
    document = parse_task_document(
        _document(
            monitors={
                "contact": {
                    "builtin": "predicate",
                    "when": {
                        "all": [
                            {"source": "value:force", "op": "gte", "value": 1.0},
                            {"source": "record:arm/limit", "op": "lt", "value": 2.0},
                        ]
                    },
                    "trigger": {"kind": "hold", "steps": 2},
                    "emit": {"event": "held", "severity": "warning"},
                }
            }
        )
    )

    monitor = document.task.monitors["contact"]
    assert isinstance(monitor.when, MappingProxyType)
    assert monitor.trigger == {"kind": "hold", "steps": 2}
    assert monitor.emit == {"event": "held", "severity": "warning"}
    with pytest.raises(TypeError):
        monitor.when["all"] = ()


def test_phase_declaration_uses_source_and_params() -> None:
    document = parse_task_document(
        _document(
            monitors={
                "phase": {
                    "builtin": "phase",
                    "source": "value:height",
                    "params": {"boundaries": [0.0, 1.0], "phase_names": ["low", "mid", "high"]},
                }
            }
        )
    )

    monitor = document.task.monitors["phase"]
    assert monitor.source == "value:height"
    assert monitor.params["boundaries"] == (0.0, 1.0)


@pytest.mark.parametrize(
    ("monitor", "fragment"),
    [
        (
            {"builtin": "predicate", "trigger": "edge"},
            "when: required for builtin 'predicate'",
        ),
        (
            {
                "builtin": "predicate",
                "when": {"source": "value:x", "op": "gt", "value": 0.0},
            },
            "trigger: required for builtin 'predicate'",
        ),
        (
            {
                "builtin": "predicate",
                "when": {"source": "value:x", "op": "gt", "value": 0.0},
                "trigger": "level",
            },
            "not implemented in pack v1",
        ),
        (
            {
                "builtin": "predicate",
                "when": {"source": "value:x", "op": "gt", "value": 0.0},
                "trigger": "always",
            },
            "reserved",
        ),
        (
            {
                "builtin": "predicate",
                "when": {"source": "value:x", "op": "gt", "value": 0.0},
                "trigger": "edge",
                "source": "value:x",
            },
            "reads sources through 'when'",
        ),
        (
            {
                "builtin": "predicate",
                "when": {"source": "value:x", "op": "gt", "value": 0.0},
                "trigger": "edge",
                "params": {"x": 1},
            },
            "takes no params",
        ),
        (
            {
                "builtin": "predicate",
                "when": {"source": "value:x", "op": "gt", "value": 0.0},
                "trigger": "edge",
                "emit": {"unknown": 1},
            },
            "emit mapping has unknown keys",
        ),
        (
            {
                "builtin": "phase",
                "params": {"boundaries": [0.0]},
            },
            "source: required for builtin 'phase'",
        ),
        (
            {
                "builtin": "phase",
                "source": "bare-key",
                "params": {"boundaries": [0.0]},
            },
            "no namespace prefix",
        ),
        (
            {
                "module": "pkg.monitors:make",
                "when": {"source": "value:x", "op": "gt", "value": 0.0},
                "trigger": "edge",
            },
            "require a builtin that consumes them",
        ),
        (
            {"factory": len, "source": "value:x"},
            "require a builtin that consumes them",
        ),
        (
            {
                "builtin": "predicate",
                "when": {"source": "value:x", "op": "gt", "value": 0.0},
                "trigger": "edge",
                "emit": None,
            },
            "explicit null is not allowed",
        ),
    ],
)
def test_rejects_invalid_predicate_field_forms(monitor: dict[str, object], fragment: str) -> None:
    with pytest.raises(SpecError) as excinfo:
        parse_task_document(_document(monitors={"m": monitor}))

    assert fragment in str(excinfo.value)


def test_query_source_is_accepted_by_the_spec_and_deferred_to_composition() -> None:
    document = parse_task_document(
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

    assert document.task.monitors["m"].when["source"] == "query:height"
