"""Real tiny-arm MuJoCo run assembly shared by the P9 integration tests.

This is test support, not a pack component: it builds a real core ``Run`` over a
self-authored two-joint MJCF and returns the objects the public seam expects.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from unigym.actuators.ir import ActuationGroup
from unigym.assets.asset_ref import AssetRef
from unigym.core.episode_termination import EpisodeOutcome, Termination, TerminationSource
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.run.api import Run
from unigym.run.sink import SinkSpec
from unigym.runtime.actions import NamedAction, OrderedActionSpec, PolicyBinding
from unigym.world import World

if TYPE_CHECKING:
    from unigym.packs.loader import PackLoader
    from unigym.runner.external import ExternalRunner

__all__ = [
    "JOINTS",
    "TINY_ARM",
    "action_spec",
    "end_episode",
    "floor_world",
    "make_run",
    "policy_binding",
    "robot_profile",
    "zero_actions",
]

TINY_ARM = Path(__file__).resolve().parent / "_data" / "tiny_arm.xml"
JOINTS = ("shoulder", "elbow")

ActionFunction: TypeAlias = Callable[[int], tuple[NamedAction, ...]]


def robot_profile() -> RobotProfile:
    """Return the two-joint profile matching ``tiny_arm.xml``."""
    return RobotProfile(
        name="tiny",
        joint_sets={"body": JOINTS},
        default_joint_pos={"shoulder": 0.1, "elbow": -0.2},
        assets=AssetRef(mjcf=str(TINY_ARM)),
        extra_action_sets={},
        base_body="base",
        foot_bodies=(),
        body_sets={},
        body_bindings={"base": BodyBinding("base", {"mjcf": "base"})},
        actuation=(
            ActuationGroup(
                name="body",
                law="standard_pd",
                joint_set="body",
                gains={"kp": 20.0, "kd": 1.0},
            ),
        ),
    )


def action_spec() -> OrderedActionSpec:
    """Return the position-action layout for the two joints."""
    return OrderedActionSpec.from_mapping(
        [
            {
                "name": "tiny",
                "articulation": "tiny",
                "joints": list(JOINTS),
                "target": "position",
                "role": "body",
            }
        ]
    )


def policy_binding() -> PolicyBinding:
    """Return the manual adapter binding for the tiny articulation."""
    return PolicyBinding.from_mapping({"adapter": "manual", "groups": [["tiny", "tiny"]]})


def floor_world(tmp_path: Path) -> Path:
    """Write and return a minimal floor-only world MJCF under ``tmp_path``."""
    path = tmp_path / "world.xml"
    path.write_text(
        '<mujoco model="floor"><worldbody>'
        '<geom name="world_floor" type="plane" size="2 2 0.1"/>'
        "</worldbody></mujoco>",
        encoding="utf-8",
    )
    return path


def zero_actions(_t: int) -> tuple[NamedAction, ...]:
    """Return one zero position action for the tiny articulation."""
    return (NamedAction.zeros(1, "tiny", JOINTS, "position"),)


def make_run(
    tmp_path: Path,
    *,
    steps: int,
    sink: SinkSpec | None,
    pack_runtime: PackLoader | None = None,
) -> Run:
    """Build one real tiny-arm ``Run`` with the supplied sink.

    The post-deletion core owns no run mode, episode clock or built-in
    benchmark, so the run declares only its body, backend and steps.  Episodes
    end through :func:`end_episode` (``runner.terminate``) in the test bodies.
    """
    return Run(
        world=World.base(mjcf=floor_world(tmp_path)),
        robot=robot_profile(),
        backend="mujoco",
        observations={"robot": ["tiny"]},
        action_spec=action_spec(),
        policies=(policy_binding(),),
        steps=steps,
        output_dir=str(tmp_path / "run"),
        no_artifacts=True,
        sink=sink,
        pack_runtime=pack_runtime,
        discover_runtime=False,
    )


def end_episode(
    runner: ExternalRunner,
    *,
    env_id: int = 0,
    reset: bool,
    code: str = "custom.episode_horizon",
    outcome: EpisodeOutcome = EpisodeOutcome.DEADLINE,
) -> None:
    """End one episode through the core's explicit terminate boundary."""
    runner.terminate(
        [env_id],
        {
            env_id: Termination(
                outcome=outcome,
                code=code,
                source=TerminationSource.CUSTOM,
            )
        },
        reset=reset,
    )
