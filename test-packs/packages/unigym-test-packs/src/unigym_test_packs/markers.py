"""Validate the ``requires_test_packs`` pytest marker."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import pytest

MARKER_NAME = "requires_test_packs"
"""The resource marker a pack-dependent test declares."""

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class PackMarkerError(ValueError):
    """A ``requires_test_packs`` marker is malformed."""


def validate_pack_name(value: object, *, where: str = "pack name") -> str:
    """Return *value* as a legal pack name or raise a teaching error.

    Args:
        value: The candidate value.
        where: A short description of the field being validated.

    Returns:
        The validated pack name.

    Raises:
        PackMarkerError: The value is not a string matching
            ``^[a-z][a-z0-9_]*$``.
    """
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise PackMarkerError(
            f"{where} must be a string matching [a-z][a-z0-9_]*, got {value!r}"
        )
    return value


def parse_requirement(
    args: Sequence[object],
    kwargs: Mapping[str, object],
    *,
    nodeid: str,
) -> tuple[str, ...]:
    """Validate one marker's arguments and return its declared pack names.

    Args:
        args: Positional marker arguments.
        kwargs: Keyword marker arguments; any keyword is an error.
        nodeid: The test node id, used in error messages.

    Returns:
        The declared pack names in declaration order.

    Raises:
        PackMarkerError: The marker has no arguments, duplicate arguments,
            keyword arguments, or an illegal pack name.
    """
    if kwargs:
        raise PackMarkerError(
            f"{nodeid}: {MARKER_NAME} takes positional pack names only, "
            f"got keyword(s) {sorted(kwargs)}"
        )
    if not args:
        raise PackMarkerError(
            f"{nodeid}: {MARKER_NAME} requires at least one pack name"
        )

    names = tuple(validate_pack_name(arg, where=f"{nodeid}: pack name") for arg in args)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise PackMarkerError(
            f"{nodeid}: {MARKER_NAME} repeats pack name(s) {duplicates}; "
            f"declare each pack once"
        )
    return names


def marker_requirement(item: pytest.Item) -> tuple[str, ...]:
    """Return the pack names declared on *item*, or an empty tuple.

    Args:
        item: The collected pytest item.

    Returns:
        The single marker's declared pack names, or ``()`` when absent.

    Raises:
        PackMarkerError: The item carries more than one marker or the marker
            arguments are malformed.
    """
    markers = tuple(item.iter_markers(name=MARKER_NAME))
    if not markers:
        return ()
    if len(markers) != 1:
        raise PackMarkerError(
            f"{item.nodeid}: {MARKER_NAME} may appear at most once per item, "
            f"found {len(markers)}"
        )
    marker = markers[0]
    return parse_requirement(marker.args, marker.kwargs, nodeid=item.nodeid)


__all__ = [
    "MARKER_NAME",
    "PackMarkerError",
    "marker_requirement",
    "parse_requirement",
    "validate_pack_name",
]
