"""Normalize a RobotProfile into the frozen migration snapshot shape."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from unigym.robots.base import RobotProfile

_DATA = Path(__file__).resolve().parent / "data"


def profile_document(profile: RobotProfile) -> dict[str, Any]:
    """Return every neutral profile field in the snapshot representation."""
    return {
        "schema": "unigym.robot/1",
        "name": profile.name,
        "joint_sets": {key: list(value) for key, value in profile.joint_sets.items()},
        "default_joint_pos": {
            key: float(value) for key, value in profile.default_joint_pos.items()
        },
        "base_body": str(profile.base_body) if profile.base_body is not None else None,
        "base_height": profile.base_height,
        "assets": profile.assets.to_dict() if profile.assets is not None else None,
        "foot_bodies": [str(value) for value in profile.foot_bodies],
        "body_sets": {
            key: [str(value) for value in values]
            for key, values in profile.body_sets.items()
        },
        "body_bindings": {
            str(key): dict(value.names) for key, value in profile.body_bindings.items()
        },
        "extra_action_sets": {
            key: list(value) for key, value in profile.extra_action_sets.items()
        },
        "full_joint_order": (
            list(profile.full_joint_order) if profile.full_joint_order else None
        ),
    }


def load_profile_snapshot(name: str) -> dict[str, Any]:
    """Load one frozen profile migration snapshot by file stem."""
    path = _DATA / f"{name}_profile.json"
    return json.loads(path.read_text(encoding="utf-8"))
