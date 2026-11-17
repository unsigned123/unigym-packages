from __future__ import annotations

import pytest

from unigym_test_packs.markers import (
    PackMarkerError,
    marker_requirement,
    parse_requirement,
    validate_pack_name,
)


class _Item:
    def __init__(self, *marks: pytest.Mark) -> None:
        self.nodeid = "tests/test_sample.py::test_case"
        self._marks = marks

    def iter_markers(self, name: str | None = None) -> tuple[pytest.Mark, ...]:
        if name is None:
            return self._marks
        return tuple(mark for mark in self._marks if mark.name == name)


def _mark(*args: object, **kwargs: object) -> pytest.Mark:
    return pytest.mark.requires_test_packs(*args, **kwargs).mark


def test_parse_requirement_returns_declared_order() -> None:
    assert parse_requirement(("basic", "g1"), {}, nodeid="test") == ("basic", "g1")


def test_parse_requirement_rejects_zero_arguments() -> None:
    with pytest.raises(PackMarkerError, match="requires at least one"):
        parse_requirement((), {}, nodeid="test")


def test_parse_requirement_rejects_duplicate_names() -> None:
    with pytest.raises(PackMarkerError, match="repeats pack name"):
        parse_requirement(("basic", "basic"), {}, nodeid="test")


@pytest.mark.parametrize("name", ["", "Basic", "1basic", "basic-pack", 3])
def test_parse_requirement_rejects_illegal_names(name: object) -> None:
    with pytest.raises(PackMarkerError, match="must be a string matching"):
        parse_requirement((name,), {}, nodeid="test")


def test_parse_requirement_rejects_keyword_arguments() -> None:
    with pytest.raises(PackMarkerError, match="positional pack names only"):
        parse_requirement(("basic",), {"basic": True}, nodeid="test")


def test_marker_requirement_returns_empty_without_marker() -> None:
    assert marker_requirement(_Item()) == ()


def test_marker_requirement_reads_single_marker() -> None:
    item = _Item(_mark("basic"))
    assert marker_requirement(item) == ("basic",)


def test_marker_requirement_rejects_duplicate_markers() -> None:
    item = _Item(_mark("basic"), _mark("g1"))
    with pytest.raises(PackMarkerError, match="at most once"):
        marker_requirement(item)


def test_validate_pack_name_accepts_contract_grammar() -> None:
    assert validate_pack_name("isaaclab_cartpole") == "isaaclab_cartpole"
