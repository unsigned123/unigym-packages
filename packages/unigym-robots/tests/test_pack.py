"""Pack-profile tests for the shipped unigym-robots pack."""

from __future__ import annotations

import pytest
import yaml

from unigym_robots import pack_root

PACK = pack_root()
ROBOT_IDS = ("go1", "go2", "g1")
EXPECTED_ASSET_BACKENDS = {
    "go1": {"mjcf": ("mujoco", "mjwarp")},
    "go2": {"mjcf": ("mujoco", "mjwarp"), "usd": ("isaaclab2",)},
    "g1": {"mjcf": ("mujoco", "mjwarp"), "usd": ("isaaclab2",)},
}


def test_pack_root_carries_the_manifest_and_robot_profiles() -> None:
    assert PACK.is_dir()
    assert (PACK / "unigym.pack.yaml").is_file()
    for robot_id in ROBOT_IDS:
        assert (PACK / "robots" / f"{robot_id}.yaml").is_file()


def test_manifest_declares_each_robot_as_a_resource_provider() -> None:
    manifest = yaml.safe_load((PACK / "unigym.pack.yaml").read_text(encoding="utf-8"))
    assert manifest["schema"] == "unigym.pack/1"
    assert manifest["name"] == "unigym_robots"
    providers = {item["id"]: item for item in manifest["provides"]["providers"]}
    assert tuple(providers) == ROBOT_IDS
    for robot_id, item in providers.items():
        assert item["kind"] == "robot"
        assert item["resource"] == f"robots/{robot_id}.yaml"


def test_profiles_carry_assets_and_actuation() -> None:
    for robot_id in ROBOT_IDS:
        document = yaml.safe_load(
            (PACK / "robots" / f"{robot_id}.yaml").read_text(encoding="utf-8")
        )
        assert document["schema"] == "unigym.robot/1"
        assert document["name"] == robot_id
        assert document["joint_sets"]
        assert document["default_joint_pos"]
        assets = document.get("assets") or {}
        assert assets.get("mjcf", "").startswith("release:")
        actuation = document.get("actuation")
        assert isinstance(actuation, list) and actuation
        assert all(item.get("law") for item in actuation)


def test_profiles_declare_asset_backends_per_format() -> None:
    """Each shipped profile gates its declared formats to the right backends."""
    for robot_id, expected in EXPECTED_ASSET_BACKENDS.items():
        document = yaml.safe_load(
            (PACK / "robots" / f"{robot_id}.yaml").read_text(encoding="utf-8")
        )
        declared = document.get("asset_backends") or {}
        assert {fmt: tuple(names) for fmt, names in declared.items()} == expected


def test_no_robot_binaries_are_shipped() -> None:
    package_root = PACK.parent
    suffixes = {".xml", ".usd", ".usda", ".stl", ".obj", ".dae", ".png", ".jpg", ".tar", ".gz"}
    found = sorted(
        str(path.relative_to(package_root))
        for path in package_root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
    assert found == []


def test_pack_robot_returns_a_complete_profile() -> None:
    pytest.importorskip("unigym")
    from unigym.packs import discover_set
    from unigym.robots import ROBOT_REGISTRY
    from unigym.robots.base import RobotProfile

    pack_set = discover_set([PACK])
    pack = pack_set.pack("unigym_robots")
    for robot_id in ROBOT_IDS:
        profile = pack.robot(robot_id)
        assert isinstance(profile, RobotProfile)
        assert profile.name == robot_id
        assert profile.assets is not None
        assert profile.assets.mjcf == f"release:{robot_id}-mjcf"
        assert {
            asset_format: tuple(backends)
            for asset_format, backends in profile.asset_backends.items()
        } == EXPECTED_ASSET_BACKENDS[robot_id]
        assert profile.actuation
        assert pack_set.robot(robot_id) == profile
        assert robot_id not in ROBOT_REGISTRY

    go2 = pack.robot("go2")
    assert go2.assets.usd == "release:go2-usd"
    assert go2.actuation[0].joint_set == "body"
