#!/usr/bin/env python3
"""Build, install and run the test-pack wheels outside the checkout."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_PROJECT = REPO_ROOT / "packages" / "unigym-test-packs"
BASIC_PROJECT = REPO_ROOT / "packages" / "unigym-test-pack-basic"
CARTPOLE_PROJECT = REPO_ROOT / "packages" / "unigym-test-pack-isaaclab-cartpole"
G1_PROJECT = REPO_ROOT / "packages" / "unigym-test-pack-g1"

G1_EXPECTED_ASSETS = {
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

G1_USD_SHA256 = "448e37857ce9bec3bf29c41c88dd81932cd9b0ad2a107bb307353168b33a7266"
G1_USDA_SHA256 = "d465d4c4ee6bd4863556f86e306bfb0eeaee125128a17a24bbe52d5340f3e5d7"

CONSUMER_TEST = '''\
import pytest
from unigym.pack_api import load_manifest_v1

pytestmark = pytest.mark.requires_test_packs("basic", "isaaclab_cartpole", "g1")


def test_installed_wheels(pack_root, pack_runtime):
    root = pack_root("basic")
    assert load_manifest_v1(root).name == "basic"
    assert (root / "robots" / "basic_arm.yaml").is_file()
    runtime = pack_runtime("basic")
    assert runtime.catalog.packs() == ("basic",)
    assert runtime.resource_for("providers", "basic_arm").is_file()
    assert runtime.resource_for("scenes", "basic.tabletop").is_file()

    cartpole_root = pack_root("isaaclab_cartpole")
    assert load_manifest_v1(cartpole_root).name == "isaaclab_cartpole"
    cartpole = pack_runtime("isaaclab_cartpole")
    usd = cartpole.resource_for("assets", "isaaclab_cartpole.usd")
    assert b"Props" in usd.read_bytes()
    assert (usd.parent / "Props" / "instanceable_meshes.usd").is_file()

    g1_root = pack_root("g1")
    assert load_manifest_v1(g1_root).name == "g1"
    g1 = pack_runtime("g1")
    g1_usd = g1.resource_for("assets", "g1.usd")
    assert (g1_usd.parent / "configuration" / "three_finger_hand").is_dir()
    assert (g1_usd.parent / "configuration" / "g1_29dof_with_hand_rev_1_0_base.usd").is_file()
'''


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _single(wheels: list[Path], pattern: str) -> Path:
    matches = sorted(wheels[0].parent.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {pattern} wheel, found {matches}")
    return matches[0]


def _entry_points_text(archive: zipfile.ZipFile) -> str:
    names = archive.namelist()
    for name in names:
        if name.endswith(".dist-info/entry_points.txt"):
            return archive.read(name).decode("utf-8")
    return ""


def _check_wheel_contents(
    plugin_wheel: Path, basic_wheel: Path, cartpole_wheel: Path, g1_wheel: Path
) -> None:
    with zipfile.ZipFile(plugin_wheel) as archive:
        names = set(archive.namelist())
        entries = _entry_points_text(archive)
    if "unigym_test_packs/plugin.py" not in names:
        raise SystemExit(f"{plugin_wheel} is missing the plugin module")
    if "pytest11" not in entries or "unigym_test_packs" not in entries:
        raise SystemExit(f"{plugin_wheel} is missing the pytest11 entry point")

    with zipfile.ZipFile(basic_wheel) as archive:
        names = set(archive.namelist())
        entries = _entry_points_text(archive)
    required = {
        "unigym_test_pack_basic/pack/unigym.pack.yaml",
        "unigym_test_pack_basic/pack/robots/basic_arm.yaml",
        "unigym_test_pack_basic/pack/scenes/tabletop.yaml",
    }
    missing = sorted(required - names)
    if missing:
        raise SystemExit(f"{basic_wheel} is missing pack data: {missing}")
    if "[unigym.test_packs]" not in entries or "basic =" not in entries:
        raise SystemExit(f"{basic_wheel} is missing the unigym.test_packs entry point")

    with zipfile.ZipFile(cartpole_wheel) as archive:
        names = set(archive.namelist())
        entries = _entry_points_text(archive)
    required = {
        "unigym_test_pack_isaaclab_cartpole/pack/unigym.pack.yaml",
        "unigym_test_pack_isaaclab_cartpole/pack/assets/cartpole.usd",
        "unigym_test_pack_isaaclab_cartpole/pack/assets/Props/instanceable_meshes.usd",
    }
    missing = sorted(required - names)
    if missing:
        raise SystemExit(f"{cartpole_wheel} is missing pack data: {missing}")
    if not any(name.endswith("THIRD_PARTY_NOTICES.md") for name in names):
        raise SystemExit(f"{cartpole_wheel} is missing THIRD_PARTY_NOTICES.md")
    if "[unigym.test_packs]" not in entries or "isaaclab_cartpole =" not in entries:
        raise SystemExit(
            f"{cartpole_wheel} is missing the unigym.test_packs entry point"
        )

    with zipfile.ZipFile(g1_wheel) as archive:
        names = set(archive.namelist())
        entries = _entry_points_text(archive)
    required = {
        "unigym_test_pack_g1/pack/unigym.pack.yaml",
        "unigym_test_pack_g1/pack/assets/g1.usd",
        "unigym_test_pack_g1/pack/assets/configuration/g1_29dof_with_hand_rev_1_0_base.usd",
        "unigym_test_pack_g1/pack/assets/configuration/g1_29dof_with_hand_rev_1_0_robot.usda",
        "unigym_test_pack_g1/pack/assets/configuration/three_finger_hand/"
        "three_finger_hand_base_left.usd",
        "unigym_test_pack_g1/pack/assets/configuration/three_finger_hand/"
        "three_finger_hand_base_right.usd",
    }
    missing = sorted(required - names)
    if missing:
        raise SystemExit(f"{g1_wheel} is missing pack data: {missing}")
    asset_prefix = "unigym_test_pack_g1/pack/assets/"
    g1_assets = {
        name[len(asset_prefix) :] for name in names if name.startswith(asset_prefix)
    }
    if g1_assets != G1_EXPECTED_ASSETS:
        missing = sorted(G1_EXPECTED_ASSETS - g1_assets)
        extra = sorted(g1_assets - G1_EXPECTED_ASSETS)
        raise SystemExit(
            f"{g1_wheel} G1 asset set differs; missing={missing}, extra={extra}"
        )
    with zipfile.ZipFile(g1_wheel) as archive:
        usd = archive.read(f"{asset_prefix}g1.usd")
        usda = archive.read(
            f"{asset_prefix}configuration/g1_29dof_with_hand_rev_1_0_robot.usda"
        )
    if hashlib.sha256(usd).hexdigest() != G1_USD_SHA256:
        raise SystemExit(f"{g1_wheel} g1.usd bytes differ")
    if hashlib.sha256(usda).hexdigest() != G1_USDA_SHA256:
        raise SystemExit(f"{g1_wheel} g1 robot.usda bytes differ")
    if not any(name.endswith("THIRD_PARTY_NOTICES.md") for name in names):
        raise SystemExit(f"{g1_wheel} is missing THIRD_PARTY_NOTICES.md")
    if "[unigym.test_packs]" not in entries or "g1 =" not in entries:
        raise SystemExit(f"{g1_wheel} is missing the unigym.test_packs entry point")


def main() -> int:
    """Build and exercise the wheel set, returning a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    core = parser.add_mutually_exclusive_group(required=True)
    core.add_argument("--core-wheel", type=Path, help="prebuilt unigym core wheel")
    core.add_argument("--core-checkout", type=Path, help="unigym core checkout to build")
    parser.add_argument("--online", action="store_true", help="allow package downloads")
    parser.add_argument("--keep", action="store_true", help="keep the temporary tree")
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="unigym-test-packs-wheel-"))
    try:
        dist = work / "dist"
        dist.mkdir()
        for project in (PLUGIN_PROJECT, BASIC_PROJECT, CARTPOLE_PROJECT, G1_PROJECT):
            _run(["uv", "build", "--wheel", "--out-dir", str(dist), str(project)], cwd=REPO_ROOT)

        if args.core_wheel is not None:
            core_wheel = args.core_wheel.resolve()
            if not core_wheel.is_file():
                raise SystemExit(f"core wheel does not exist: {core_wheel}")
        else:
            assert args.core_checkout is not None
            core_checkout = args.core_checkout.resolve()
            if not (core_checkout / "pyproject.toml").is_file():
                raise SystemExit(f"core checkout has no pyproject.toml: {core_checkout}")
            _run(
                ["uv", "build", "--wheel", "--out-dir", str(dist), str(core_checkout)],
                cwd=REPO_ROOT,
            )
            core_wheel = _single(list(dist.glob("*.whl")), "unigym-*.whl")

        wheels = list(dist.glob("*.whl"))
        plugin_wheel = _single(wheels, "unigym_test_packs-*.whl")
        basic_wheel = _single(wheels, "unigym_test_pack_basic-*.whl")
        cartpole_wheel = _single(wheels, "unigym_test_pack_isaaclab_cartpole-*.whl")
        g1_wheel = _single(wheels, "unigym_test_pack_g1-*.whl")
        _check_wheel_contents(plugin_wheel, basic_wheel, cartpole_wheel, g1_wheel)

        venv = work / "venv"
        _run(["uv", "venv", "--python", "3.11", str(venv)], cwd=REPO_ROOT)
        python = venv / "bin" / "python"
        offline = [] if args.online else ["--offline"]
        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                *offline,
                "--no-deps",
                str(core_wheel),
                str(plugin_wheel),
                str(basic_wheel),
                str(cartpole_wheel),
                str(g1_wheel),
            ],
            cwd=REPO_ROOT,
        )
        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                *offline,
                "pytest>=8,<10",
                "PyYAML>=6,<7",
            ],
            cwd=REPO_ROOT,
        )

        run_dir = work / "consumer"
        run_dir.mkdir()
        (run_dir / "test_wheel.py").write_text(CONSUMER_TEST, encoding="utf-8")
        env = dict(os.environ)
        for name in ("PYTHONPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
            env.pop(name, None)
        _run(
            [str(python), "-m", "pytest", "--strict-test-packs", "-q"],
            cwd=run_dir,
            env=env,
        )
        print("wheel check passed", flush=True)
        return 0
    finally:
        if args.keep:
            print(f"kept temporary tree: {work}", file=sys.stderr)
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
