from __future__ import annotations

from pathlib import Path

import pytest
from unigym.pack_api import load_manifest_v1
from unigym.scene_presets import load_scene_preset

pytestmark = pytest.mark.requires_test_packs("basic")


def test_basic_pack_root_carries_manifest(pack_root) -> None:
    root: Path = pack_root("basic")

    manifest = load_manifest_v1(root)
    assert manifest.name == "basic"
    assert (root / "robots" / "basic_arm.yaml").is_file()
    assert (root / "scene_presets" / "tabletop.yaml").is_file()


def test_basic_pack_runtime_resolves_declared_resources(pack_root, pack_runtime) -> None:
    root: Path = pack_root("basic")
    runtime = pack_runtime("basic")

    assert tuple(pack.name for pack in runtime.pack_set.packs()) == ("basic",)
    assert runtime.resolve("providers", "basic_arm") == root / "robots" / "basic_arm.yaml"
    assert (
        runtime.resolve("scene_presets", "basic.tabletop")
        == root / "scene_presets" / "tabletop.yaml"
    )
    assert "name: basic_arm" in (root / "robots" / "basic_arm.yaml").read_text(encoding="utf-8")


def test_basic_pack_ships_a_loadable_scene_preset(pack_runtime) -> None:
    preset = load_scene_preset("basic.tabletop", pack_runtime("basic"))

    assert preset.name == "tabletop"
    assert preset.provider is None

