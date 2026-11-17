from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from unigym.pack_api import load_manifest_v1

pytestmark = pytest.mark.requires_test_packs("g1")

G1_USD_SHA256 = "448e37857ce9bec3bf29c41c88dd81932cd9b0ad2a107bb307353168b33a7266"
G1_USDA_SHA256 = "d465d4c4ee6bd4863556f86e306bfb0eeaee125128a17a24bbe52d5340f3e5d7"

EXPECTED_ASSETS = {
    "g1.usd",
    "configuration/g1_29dof_with_hand_rev_1_0_base.usd",
    "configuration/g1_29dof_with_hand_rev_1_0_physics_simplified.usd",
    "configuration/g1_29dof_with_hand_rev_1_0_physics.usd",
    "configuration/g1_29dof_with_hand_rev_1_0_robot.usd",
    "configuration/g1_29dof_with_hand_rev_1_0_robot.usda",
    "configuration/g1_29dof_with_hand_rev_1_0_sensor.usd",
    "configuration/g1_Gripper_three_finger_hand_physics_right.usd",
    "configuration/g1_left_hand_inspire_left_physics.usd",
    "configuration/g1_left_hand_three_finger_hand_physics_left.usd",
    "configuration/g1_right_hand_inspire_right_physics.usd",
    "configuration/g1_right_hand_three_finger_hand_physics_right.usd",
    "configuration/three_finger_hand/three_finger_hand_base_left.usd",
    "configuration/three_finger_hand/three_finger_hand_base_right.usd",
    "configuration/three_finger_hand/three_finger_hand_physics_left.usd",
    "configuration/three_finger_hand/three_finger_hand_physics_right.usd",
    "configuration/three_finger_hand/three_fingers_left_hand_robot.usd",
    "configuration/three_finger_hand/three_fingers_left_hand.usd",
    "configuration/three_finger_hand/three_fingers_right_hand_robot.usd",
    "configuration/three_finger_hand/three_fingers_right_hand.usd",
}


def test_g1_pack_carries_the_complete_asset_tree(pack_root) -> None:
    root: Path = pack_root("g1")

    assert load_manifest_v1(root).name == "g1"
    assets = root / "assets"
    actual = {
        path.relative_to(assets).as_posix()
        for path in assets.rglob("*")
        if path.is_file()
    }
    assert actual == EXPECTED_ASSETS
    assert len(actual) == 20


def test_g1_pack_runtime_resolves_root_usd(pack_root, pack_runtime) -> None:
    root: Path = pack_root("g1")
    runtime = pack_runtime("g1")

    usd = runtime.resource_for("assets", "g1.usd")
    assert usd == root / "assets" / "g1.usd"
    assert b"g1_29dof_with_hand_rev_1_0" in usd.read_bytes()
    assert (usd.parent / "configuration").is_dir()
    base = usd.parent / "configuration" / "g1_29dof_with_hand_rev_1_0_base.usd"
    assert base.is_file()
    assert base.stat().st_size > 30_000_000
    assert hashlib.sha256(usd.read_bytes()).hexdigest() == G1_USD_SHA256
    usda = usd.parent / "configuration" / "g1_29dof_with_hand_rev_1_0_robot.usda"
    assert hashlib.sha256(usda.read_bytes()).hexdigest() == G1_USDA_SHA256
