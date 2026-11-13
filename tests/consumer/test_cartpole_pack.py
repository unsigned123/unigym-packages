from __future__ import annotations

from pathlib import Path

import pytest
from unigym.pack_api import load_manifest_v1

pytestmark = pytest.mark.requires_test_packs("isaaclab_cartpole")


def test_cartpole_pack_carries_usd_and_companion(pack_root) -> None:
    root: Path = pack_root("isaaclab_cartpole")

    manifest = load_manifest_v1(root)
    assert manifest.name == "isaaclab_cartpole"
    assert (root / "assets" / "cartpole.usd").is_file()
    assert (root / "assets" / "Props" / "instanceable_meshes.usd").is_file()


def test_cartpole_pack_runtime_resolves_asset(pack_root, pack_runtime) -> None:
    root: Path = pack_root("isaaclab_cartpole")
    runtime = pack_runtime("isaaclab_cartpole")

    usd = runtime.resource_for("assets", "isaaclab_cartpole.usd")
    assert usd == root / "assets" / "cartpole.usd"
    assert b"Props" in usd.read_bytes()
    assert (usd.parent / "Props" / "instanceable_meshes.usd").is_file()
