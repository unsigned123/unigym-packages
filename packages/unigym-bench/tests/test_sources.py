"""Source references and the stateless step source resolver."""

from __future__ import annotations

import pytest
import torch

from _support import make_context
from unigym.env.obs import ObservationSchema, ObservationSpec, ObsSnapshot
from unigym.evaluation import EnvStepContext, EnvView
from unigym.values import StepValues
from unigym_bench.sources import (
    SourceError,
    SourceParseError,
    SourceRef,
    SourceResolveError,
    SourceResolver,
    SourceUnavailableError,
)


def _observation(values: dict[str, torch.Tensor], num_envs: int) -> ObsSnapshot:
    entries = {
        key: ObservationSpec(key, tuple(value.shape[1:]), "f32") for key, value in values.items()
    }
    return ObsSnapshot(
        values,
        schema=ObservationSchema(entries),
        num_envs=num_envs,
        device="cpu",
    )


def _context(
    *,
    record: dict[str, object] | None = None,
    obs: ObsSnapshot | None = None,
    values: StepValues | None = None,
    num_envs: int = 2,
) -> EnvStepContext:
    view = EnvView(
        record={} if record is None else record,
        obs=obs,
        env_id=None,
        num_envs=num_envs,
        device="cpu",
    )
    return make_context(num_envs=num_envs, view=view, values=values)


@pytest.mark.parametrize(
    ("text", "kind", "key"),
    [
        ("record:foo", "record", "foo"),
        ("obs:foo", "obs", "foo"),
        ("value:foo", "value", "foo"),
        ("query:foo", "query", "foo"),
        ("record:articulation/arm/joint_pos", "record", "articulation/arm/joint_pos"),
        ("value:a/b", "value", "a/b"),
    ],
)
def test_parse_canonical_references(text: str, kind: str, key: str) -> None:
    ref = SourceRef.parse(text)

    assert ref.kind == kind
    assert ref.key == key
    assert str(ref) == text


@pytest.mark.parametrize(
    "text",
    ["foo", "record/foo", "Record:foo", "/foo", "query/foo", "source/foo", "record-foo"],
)
def test_parse_rejects_shorthand_and_unknown_prefixes(text: str) -> None:
    with pytest.raises(SourceParseError, match="namespace"):
        SourceRef.parse(text)


@pytest.mark.parametrize("text", ["record", "obs", "value", "query", "record:", "obs:"])
def test_parse_requires_a_key(text: str) -> None:
    with pytest.raises(SourceParseError, match="missing its key"):
        SourceRef.parse(text)


def test_parse_rejects_non_string_text() -> None:
    with pytest.raises(SourceParseError, match="must be a string"):
        SourceRef.parse(3)  # type: ignore[arg-type]


def test_source_ref_constructor_validation() -> None:
    assert SourceRef("record", "foo").kind == "record"

    with pytest.raises(SourceParseError, match="source kind"):
        SourceRef("timer", "foo")  # type: ignore[arg-type]
    with pytest.raises(SourceParseError, match="source kind"):
        SourceRef(["record"], "foo")  # type: ignore[arg-type]
    with pytest.raises(SourceParseError, match="source kind"):
        SourceRef({"kind": "record"}, "foo")  # type: ignore[arg-type]
    with pytest.raises(SourceParseError, match="source kind"):
        SourceRef(None, "foo")  # type: ignore[arg-type]
    with pytest.raises(SourceParseError, match="non-empty"):
        SourceRef("record", "")
    with pytest.raises(SourceParseError, match="non-empty"):
        SourceRef("record", 3)  # type: ignore[arg-type]


def test_source_ref_from_mapping() -> None:
    assert SourceRef.from_mapping({"kind": "obs", "key": "foo"}) == SourceRef("obs", "foo")

    with pytest.raises(SourceParseError, match="unknown keys"):
        SourceRef.from_mapping({"kind": "value", "key": "foo", "extra": 1})
    with pytest.raises(SourceParseError, match="missing required keys"):
        SourceRef.from_mapping({"kind": "value"})
    with pytest.raises(SourceParseError, match="missing required keys"):
        SourceRef.from_mapping({"key": "foo"})
    with pytest.raises(SourceParseError, match="must be a mapping"):
        SourceRef.from_mapping(["value", "foo"])  # type: ignore[arg-type]
    with pytest.raises(SourceParseError, match="'kind'"):
        SourceRef.from_mapping({"kind": "timer", "key": "foo"})
    with pytest.raises(SourceParseError, match="must be a string"):
        SourceRef.from_mapping({"kind": 3, "key": "foo"})
    with pytest.raises(SourceParseError, match="non-empty string"):
        SourceRef.from_mapping({"kind": "obs", "key": 3})


def test_resolve_record_tensor_is_borrowed() -> None:
    tensor = torch.tensor([1.0, 2.0])
    step = _context(record={"level": tensor})

    resolved = SourceResolver().resolve(step, SourceRef.parse("record:level"))

    assert resolved is tensor


def test_resolve_obs_tensor() -> None:
    tensor = torch.tensor([0.0, 1.0])
    step = _context(obs=_observation({"joint_pos": tensor}, num_envs=2))

    resolved = SourceResolver().resolve(step, SourceRef.parse("obs:joint_pos"))

    assert resolved is tensor


def test_resolve_value_tensor_and_json() -> None:
    level = torch.tensor([0.0, 1.0])
    step = _context(
        values=StepValues({"level": level, "meta": {"nested": [1, 2]}}),
    )
    resolver = SourceResolver()

    resolved_level = resolver.resolve(step, SourceRef.parse("value:level"))
    assert isinstance(resolved_level, torch.Tensor)
    assert torch.equal(resolved_level, level)
    assert resolved_level is not level

    resolved_meta = resolver.resolve(step, SourceRef.parse("value:meta"))
    assert resolved_meta == {"nested": [1, 2]}
    assert isinstance(resolved_meta, dict)
    resolved_meta["nested"].append(3)
    assert resolver.resolve(step, SourceRef.parse("value:meta")) == {"nested": [1, 2]}


def test_resolve_value_none_is_a_json_null_not_an_absent_key() -> None:
    step = _context(values=StepValues({"nothing": None, "other": 1}))
    resolver = SourceResolver()

    assert resolver.resolve(step, SourceRef.parse("value:nothing")) is None
    with pytest.raises(SourceResolveError, match="not present"):
        resolver.resolve(step, SourceRef.parse("value:absent"))


def test_resolve_final_context_boundary() -> None:
    record = {"final_pos": torch.tensor([1.0, 2.0])}
    obs = _observation({"final_obs": torch.tensor([3.0, 4.0])}, num_envs=2)
    final_level = torch.tensor([5.0, 6.0])
    values = StepValues({"final_level": final_level, "final_meta": {"done": True}})
    step = make_context(
        num_envs=2,
        t=99,
        view=EnvView(record=record, obs=obs, env_id=None, num_envs=2, device="cpu"),
        values=values,
    )
    resolver = SourceResolver()

    resolved_record = resolver.resolve(step, SourceRef.parse("record:final_pos"))
    resolved_obs = resolver.resolve(step, SourceRef.parse("obs:final_obs"))
    resolved_value = resolver.resolve(step, SourceRef.parse("value:final_level"))
    assert isinstance(resolved_record, torch.Tensor)
    assert isinstance(resolved_obs, torch.Tensor)
    assert isinstance(resolved_value, torch.Tensor)
    assert torch.equal(resolved_record, record["final_pos"])
    assert torch.equal(resolved_obs, obs.values["final_obs"])
    assert torch.equal(resolved_value, final_level)
    assert resolver.resolve(step, SourceRef.parse("value:final_meta")) == {"done": True}


def test_missing_record_key_names_sorted_candidates() -> None:
    step = _context(record={"beta": torch.zeros(2), "alpha": torch.zeros(2)})

    with pytest.raises(SourceResolveError) as excinfo:
        SourceResolver().resolve(step, SourceRef.parse("record:missing"))

    message = str(excinfo.value)
    assert "record" in message
    assert "'missing'" in message
    assert "['alpha', 'beta']" in message


def test_missing_obs_snapshot_and_key_name_candidates() -> None:
    resolver = SourceResolver()
    without_obs = _context()
    with pytest.raises(SourceResolveError, match="no observation snapshot"):
        resolver.resolve(without_obs, SourceRef.parse("obs:joint_pos"))

    with_obs = _context(obs=_observation({"alpha": torch.zeros(2), "beta": torch.zeros(2)}, 2))
    with pytest.raises(SourceResolveError) as excinfo:
        resolver.resolve(with_obs, SourceRef.parse("obs:missing"))
    message = str(excinfo.value)
    assert "obs" in message
    assert "'missing'" in message
    assert "['alpha', 'beta']" in message


def test_missing_value_names_sorted_candidates() -> None:
    step = _context(values=StepValues({"z": 1, "a": 2}))

    with pytest.raises(SourceResolveError) as excinfo:
        SourceResolver().resolve(step, SourceRef.parse("value:missing"))

    message = str(excinfo.value)
    assert "value" in message
    assert "'missing'" in message
    assert "['a', 'z']" in message


def test_none_and_non_tensor_record_values_fail() -> None:
    resolver = SourceResolver()
    none_step = _context(record={"level": None})
    with pytest.raises(SourceResolveError, match="not present"):
        resolver.resolve(none_step, SourceRef.parse("record:level"))

    text_step = _context(record={"body_names": ["torso"]})
    with pytest.raises(SourceResolveError, match="not a torch.Tensor"):
        resolver.resolve(text_step, SourceRef.parse("record:body_names"))


def test_query_source_fails_fast_as_unavailable() -> None:
    step = _context()

    with pytest.raises(SourceUnavailableError) as excinfo:
        SourceResolver().resolve(step, SourceRef.parse("query:body_pose"))

    message = str(excinfo.value)
    assert "query" in message
    assert "typed" in message
    assert "not implemented" in message
    assert isinstance(excinfo.value, NotImplementedError)
    assert isinstance(excinfo.value, SourceError)


def test_resolve_validates_arguments() -> None:
    resolver = SourceResolver()
    step = _context()

    with pytest.raises(TypeError, match="EnvStepContext"):
        resolver.resolve(object(), SourceRef.parse("record:foo"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="SourceRef"):
        resolver.resolve(step, "record:foo")  # type: ignore[arg-type]


def test_resolver_is_stateless_and_does_not_cache() -> None:
    resolver = SourceResolver()
    first = torch.tensor([1.0, 1.0])
    second = torch.tensor([2.0, 2.0])
    first_step = _context(record={"level": first})
    second_step = _context(record={"level": second})
    ref = SourceRef.parse("record:level")

    assert resolver.resolve(first_step, ref) is first
    assert resolver.resolve(second_step, ref) is second
    assert resolver.resolve(first_step, ref) is first

    assert not hasattr(resolver, "__dict__")
    with pytest.raises(TypeError):
        vars(resolver)
