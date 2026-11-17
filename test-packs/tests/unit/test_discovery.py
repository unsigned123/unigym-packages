from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from pathlib import Path

import pytest

from unigym_test_packs.discovery import (
    PackDiscoveryError,
    PackIndex,
    discover_installed_packs,
)


@dataclass(frozen=True)
class _Entry:
    name: str
    target: object

    def load(self) -> object:
        if isinstance(self.target, BaseException):
            raise self.target
        return self.target


def _write_pack(root: Path, *, name: str = "basic") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "unigym.pack.yaml").write_text(
        "schema: unigym.pack/1\n"
        f"name: {name}\n"
        "version: 1.0.0\n"
        "api: 1\n"
        "trust: content\n"
        "provides: {}\n",
        encoding="utf-8",
    )
    return root


def test_discovery_reads_sorted_entry_points_and_roots(tmp_path: Path) -> None:
    alpha = _write_pack(tmp_path / "alpha", name="alpha")
    beta = _write_pack(tmp_path / "beta", name="beta")

    index = discover_installed_packs(
        [_Entry("beta", beta), _Entry("alpha", lambda: alpha)]
    )

    assert index.names() == ("alpha", "beta")
    assert index.root("alpha") == alpha.resolve()
    assert index.root("beta") == beta.resolve()


def test_discovery_rejects_entry_name_manifest_mismatch(tmp_path: Path) -> None:
    root = _write_pack(tmp_path / "pack", name="different")

    with pytest.raises(PackDiscoveryError, match="entry-point name and manifest name"):
        discover_installed_packs([_Entry("basic", root)])


def test_discovery_rejects_entry_load_failure() -> None:
    with pytest.raises(PackDiscoveryError, match="failed to load its target"):
        discover_installed_packs([_Entry("basic", RuntimeError("boom"))])


def test_discovery_rejects_callable_failure(tmp_path: Path) -> None:
    def target() -> Path:
        raise RuntimeError("boom")

    with pytest.raises(PackDiscoveryError, match="callable target failed"):
        discover_installed_packs([_Entry("basic", target)])


def test_discovery_rejects_non_path_target() -> None:
    with pytest.raises(PackDiscoveryError, match="path string or"):
        discover_installed_packs([_Entry("basic", 42)])


def test_discovery_rejects_relative_target(tmp_path: Path) -> None:
    _write_pack(tmp_path / "pack")
    with pytest.raises(PackDiscoveryError, match="must be absolute"):
        discover_installed_packs([_Entry("basic", "pack")])


def test_discovery_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(PackDiscoveryError, match="is not a directory"):
        discover_installed_packs([_Entry("basic", tmp_path / "missing")])


def test_discovery_rejects_broken_manifest(tmp_path: Path) -> None:
    root = tmp_path / "pack"
    root.mkdir()
    (root / "unigym.pack.yaml").write_text("schema: wrong\n", encoding="utf-8")

    with pytest.raises(PackDiscoveryError, match="broken pack"):
        discover_installed_packs([_Entry("basic", root)])


def test_discovery_rejects_duplicate_entry_names(tmp_path: Path) -> None:
    root = _write_pack(tmp_path / "pack")
    with pytest.raises(PackDiscoveryError, match="duplicate test pack entry point"):
        discover_installed_packs([_Entry("basic", root), _Entry("basic", root)])


def test_discovery_wraps_installed_metadata_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_entry_points() -> object:
        raise TypeError("bad entry_points.txt")

    monkeypatch.setattr(importlib.metadata, "entry_points", broken_entry_points)

    with pytest.raises(PackDiscoveryError, match="failed to read installed entry points"):
        discover_installed_packs()


def test_pack_index_names_missing_packs_and_reports_unknown_roots(
    tmp_path: Path,
) -> None:
    root = _write_pack(tmp_path / "pack")
    index = PackIndex({"basic": root})

    assert index.missing(("basic", "other")) == ("other",)
    with pytest.raises(PackDiscoveryError, match="installed packs"):
        index.root("other")
