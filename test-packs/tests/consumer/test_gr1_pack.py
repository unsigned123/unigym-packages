from __future__ import annotations

from pathlib import Path

import pytest
from unigym.pack_api import load_manifest_v1
from unigym.robots import build_robot_profile
from tests.consumer._profile_snapshot import load_profile_snapshot, profile_document

pytestmark = pytest.mark.requires_test_packs("gr1")


def test_gr1_pack_carries_the_robot_data(pack_root) -> None:
    root: Path = pack_root("gr1")

    manifest = load_manifest_v1(root)
    assert manifest.name == "gr1"
    assert (root / "robots" / "gr1.yaml").is_file()


def test_gr1_robot_provider_registers_the_neutral_profile(pack_runtime) -> None:
    runtime = pack_runtime("gr1")

    handle = runtime.activate("gr1")

    assert handle.robots == ("gr1",)
    profile = build_robot_profile({"name": "gr1"})
    assert profile.name == "gr1"
    assert len(profile.joint_sets["body"]) == 17
    assert len(profile.joint_sets["right_hand"]) == 11
    assert len(profile.joint_sets["left_hand"]) == 11
    assert len(profile.full_joint_names) == 39
    assert profile.full_joint_names[0] == "robot0_torso_waist_yaw"
    assert profile.base_body == "base"
    assert profile.foot_bodies == ()
    assert set(profile.action_faces) == {"right_hand", "left_hand"}
    assert profile.assets is None
    assert profile_document(profile) == load_profile_snapshot("gr1")
