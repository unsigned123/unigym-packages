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


def test_cartpole_pack_runtime_resolves_asset_scene_fixtures(pack_root, pack_runtime) -> None:
    """The pack declares the static/dynamic USD and URDF asset-scene fixtures."""
    root: Path = pack_root("isaaclab_cartpole")
    runtime = pack_runtime("isaaclab_cartpole")

    expected = {
        "static_cup.usda": root / "assets" / "static_cup.usda",
        "dynamic_cube.usda": root / "assets" / "dynamic_cube.usda",
        "jointless_cube.urdf": root / "assets" / "jointless_cube.urdf",
        "jointed_cube.urdf": root / "assets" / "jointed_cube.urdf",
    }
    for asset_id, path in expected.items():
        assert runtime.resource_for("assets", asset_id) == path
        assert path.is_file()

    static_stage = expected["static_cup.usda"].read_text(encoding="utf-8")
    assert "PhysicsCollisionAPI" in static_stage
    assert "PhysicsRigidBodyAPI" not in static_stage
    assert "ArticulationRootAPI" not in static_stage

    dynamic_stage = expected["dynamic_cube.usda"].read_text(encoding="utf-8")
    assert "PhysicsRigidBodyAPI" in dynamic_stage
    assert "PhysicsCollisionAPI" in dynamic_stage

    jointless = expected["jointless_cube.urdf"].read_text(encoding="utf-8")
    assert "<joint " not in jointless
    assert "<inertial>" in jointless or "<inertial " in jointless
    assert "<collision>" in jointless
    assert "<visual>" in jointless

    jointed = expected["jointed_cube.urdf"].read_text(encoding="utf-8")
    assert "<joint " in jointed
