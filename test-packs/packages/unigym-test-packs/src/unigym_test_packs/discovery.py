"""Discover installed test packs from the ``unigym.test_packs`` entry points."""

from __future__ import annotations

import importlib.metadata
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from unigym.pack_api import load_manifest_v1
from unigym.packs import PackError

ENTRY_POINT_GROUP = "unigym.test_packs"
"""The dedicated entry-point group test-pack distributions publish."""


class PackDiscoveryError(RuntimeError):
    """Installed test-pack metadata or its manifest is broken."""


class EntryPointLike(Protocol):
    """The entry-point surface discovery consumes."""

    @property
    def name(self) -> str:
        """The entry-point name, which is the pack name."""
        ...

    def load(self) -> object:
        """Load the target object declared by the entry point."""
        ...


@dataclass(frozen=True)
class PackIndex:
    """An immutable installed pack name -> root mapping."""

    roots: Mapping[str, Path]

    def __post_init__(self) -> None:
        """Freeze the root mapping so later discovery cannot mutate it."""
        object.__setattr__(self, "roots", MappingProxyType(dict(self.roots)))

    def names(self) -> tuple[str, ...]:
        """Return installed pack names in discovery order."""
        return tuple(self.roots)

    def root(self, name: str) -> Path:
        """Return the installed root for *name* or fail with the known names."""
        try:
            return self.roots[name]
        except KeyError:
            raise PackDiscoveryError(
                f"test pack {name!r} is not installed; installed packs: {self.names()}"
            ) from None

    def missing(self, names: Iterable[str]) -> tuple[str, ...]:
        """Return the requested names that are not installed, preserving order."""
        return tuple(name for name in names if name not in self.roots)


def _resolve_root(entry: EntryPointLike) -> Path:
    """Resolve one entry point to an absolute pack root."""
    try:
        target = entry.load()
    except Exception as err:
        raise PackDiscoveryError(
            f"test pack entry point {entry.name!r} failed to load its target: {err}"
        ) from err

    if callable(target):
        try:
            target = target()
        except Exception as err:
            raise PackDiscoveryError(
                f"test pack entry point {entry.name!r} callable target failed: {err}"
            ) from err

    if not isinstance(target, (str, os.PathLike)):
        raise PackDiscoveryError(
            f"test pack entry point {entry.name!r} target must be a path string or "
            f"a zero-argument callable returning one, got {type(target).__name__}"
        )

    root = Path(os.fspath(target))
    if not root.is_absolute():
        raise PackDiscoveryError(
            f"test pack entry point {entry.name!r} target {str(root)!r} must be absolute"
        )
    resolved = root.resolve()
    if not resolved.is_dir():
        raise PackDiscoveryError(
            f"test pack entry point {entry.name!r} target {resolved} is not a directory"
        )
    return resolved


def discover_installed_packs(
    entry_points: Iterable[EntryPointLike] | None = None,
) -> PackIndex:
    """Load every installed test pack and return its validated root index.

    Args:
        entry_points: Explicit entry points for tests. ``None`` reads the
            installed ``unigym.test_packs`` group.

    Returns:
        The immutable installed pack index, ordered by entry-point name.

    Raises:
        PackDiscoveryError: An entry point fails to load, resolves to an invalid
            root, names a different pack than its manifest, or duplicates a
            pack name.
    """
    try:
        selected = (
            importlib.metadata.entry_points().select(group=ENTRY_POINT_GROUP)
            if entry_points is None
            else entry_points
        )
        ordered = sorted(selected, key=lambda entry: entry.name)
    except Exception as err:
        raise PackDiscoveryError(
            f"failed to read installed entry points in group {ENTRY_POINT_GROUP!r}: {err}"
        ) from err

    roots: dict[str, Path] = {}
    for entry in ordered:
        if entry.name in roots:
            raise PackDiscoveryError(f"duplicate test pack entry point {entry.name!r}")
        root = _resolve_root(entry)
        try:
            manifest = load_manifest_v1(root)
        except PackError as err:
            raise PackDiscoveryError(
                f"test pack entry point {entry.name!r} resolved to a broken pack at "
                f"{root}: {err}"
            ) from err
        if manifest.name != entry.name:
            raise PackDiscoveryError(
                f"test pack entry point {entry.name!r} points at manifest name "
                f"{manifest.name!r}; the entry-point name and manifest name must match"
            )
        roots[entry.name] = manifest.root

    return PackIndex(roots)


__all__ = [
    "ENTRY_POINT_GROUP",
    "EntryPointLike",
    "PackDiscoveryError",
    "PackIndex",
    "discover_installed_packs",
]
