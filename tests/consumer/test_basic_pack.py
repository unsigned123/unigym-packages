from __future__ import annotations

from pathlib import Path

import pytest
from unigym.pack_api import load_manifest_v1

pytestmark = pytest.mark.requires_test_packs("basic")


def test_basic_pack_root_carries_manifest(pack_root) -> None:
    root: Path = pack_root("basic")

    manifest = load_manifest_v1(root)
    assert manifest.name == "basic"
    assert (root / "robots" / "basic_arm.yaml").is_file()
    assert (root / "scenes" / "tabletop.yaml").is_file()


def test_basic_pack_runtime_resolves_declared_resources(pack_root, pack_runtime) -> None:
    root: Path = pack_root("basic")
    runtime = pack_runtime("basic")

    assert runtime.catalog.packs() == ("basic",)
    assert runtime.resource_for("providers", "basic_arm") == root / "robots" / "basic_arm.yaml"
    assert runtime.resource_for("scenes", "basic.tabletop") == root / "scenes" / "tabletop.yaml"
    assert "name: basic_arm" in (root / "robots" / "basic_arm.yaml").read_text(encoding="utf-8")
