from __future__ import annotations

import pytest
from unigym.pack_api import ItemStatus
from unigym.robots import build_robot_profile
from tests.consumer._profile_snapshot import load_profile_snapshot, profile_document

pytestmark = pytest.mark.requires_test_packs("g1")


def test_g1_robot_provider_registers_the_neutral_profile(pack_runtime) -> None:
    runtime = pack_runtime("g1")

    activated = runtime.activate("g1")
    robots = activated.items_for(face="providers", kind="robot")
    assert tuple(item.id for item in robots) == ("g1",)
    assert all(item.status is ItemStatus.REGISTERED for item in robots)
    profile = build_robot_profile({"name": "g1"})
    assert profile.name == "g1"
    assert len(profile.joint_sets["body"]) == 29
    assert len(profile.joint_sets["hand"]) == 14
    assert profile.joint_sets["body"][0] == "left_hip_pitch_joint"
    assert profile.default_joint_pos["left_hip_pitch_joint"] == pytest.approx(-0.312)
    assert profile.default_joint_pos["left_hand_index_0_joint"] == 0.0
    assert profile.base_body == "base"
    assert profile.base_height == pytest.approx(0.78)
    assert profile.extra_action_sets == {}
    assert profile.assets is None
    assert profile_document(profile) == load_profile_snapshot("g1")
