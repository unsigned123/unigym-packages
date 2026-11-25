"""Real-MuJoCo tests for the repo-only ``examples/go2_velocity`` smoke.

The profile/spec tests need only the installed ``unigym-robots`` pack.  The
real-env tests skip honestly when the pinned Go2 MJCF assets have not been
downloaded (or the pack is not installed); the final smoke in the task report
runs them with the downloaded assets.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from unigym.env.obs import ObsSnapshot
from unigym.runtime.actions import OrderedActionSpec, PolicyBinding

from unigym_rl.testing.fakes import make_fake_obs_registry

_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "go2_velocity"


def _load_module(name: str, filename: str) -> Any:
    """Load one example module under a unique name (``task`` would collide)."""
    path = _EXAMPLE_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


go2_task = _load_module("go2_velocity_task", "task.py")

GO2_JOINTS = go2_task.GO2_JOINTS
GO2_DEFAULT_JOINT_POS = go2_task.GO2_DEFAULT_JOINT_POS
GO2_ARTICULATION = go2_task.GO2_ARTICULATION


def _require_go2_assets() -> None:
    """Skip when the pack/entry point or the downloaded assets are absent."""
    pytest.importorskip("unigym_robots")
    try:
        go2_task.go2_profile()
    except Exception as exc:
        pytest.skip(f"Go2 pack/assets unavailable: {exc}")


def _close(env: Any) -> None:
    env.close()
    env.run.close()


def _standby(num_envs: int) -> torch.Tensor:
    return torch.tensor(GO2_DEFAULT_JOINT_POS, dtype=torch.float32).repeat(num_envs, 1)


@pytest.fixture
def one_env() -> Iterator[Any]:
    """One real Go2 env; closed (task + borrowed run) after the test."""
    _require_go2_assets()
    env = go2_task.make_go2_velocity_task(num_envs=1, max_steps=50)
    try:
        yield env
    finally:
        _close(env)


def test_pack_profile_uses_real_go2_calf_bodies() -> None:
    """The profile loads and binds the four Menagerie calf bodies."""
    pytest.importorskip("unigym_robots")
    try:
        profile = go2_task.go2_profile()
    except Exception as exc:
        pytest.skip(f"unigym-robots pack entry point unavailable: {exc}")

    assert profile.name == go2_task.GO2_PROFILE_ID
    assert profile.foot_bodies == ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
    assert "FL_foot" not in profile.body_bindings
    for body in profile.foot_bodies:
        binding = profile.body_bindings[body]
        assert binding.names["mjcf"] == body
        assert binding.names["usd"] == body
    assert profile.base_body == "base"
    assert tuple(profile.joint_order) == GO2_JOINTS


def test_action_spec_is_explicit_twelve_joint_body_group() -> None:
    """The action layout is explicit, 12 columns, and never inferred."""
    assert len(GO2_JOINTS) == 12
    assert isinstance(go2_task.GO2_ACTION_SPEC, OrderedActionSpec)
    assert go2_task.GO2_ACTION_SPEC.dim == 12
    assert go2_task.GO2_ACTION_GROUP == "legs"
    group = go2_task.GO2_ACTION_SPEC.groups[0]
    assert group.name == go2_task.GO2_ACTION_GROUP
    assert group.articulation == GO2_ARTICULATION
    assert group.joints == GO2_JOINTS
    assert group.target == "position"
    assert group.role == "body"
    assert all(isinstance(binding, PolicyBinding) for binding in go2_task.GO2_POLICIES)
    assert go2_task.GO2_POLICIES[0].groups == ((GO2_ARTICULATION, go2_task.GO2_ACTION_GROUP),)


def test_spec_uses_flat_world_and_profile_assets() -> None:
    """The World owns no robot; the profile asset and spawn are run-owned."""
    _require_go2_assets()

    assert not hasattr(go2_task, "go2_scene_path")
    spec = go2_task.make_go2_velocity_spec(num_envs=1)

    assert spec.world.is_flat is True
    assert spec.world.fragment() == {}
    assert spec.spawn == go2_task.GO2_SPAWN
    assert spec.robot.assets is not None
    assert spec.robot.assets.mjcf == "release:go2-mjcf"
    assert spec.action_spec is go2_task.GO2_ACTION_SPEC
    assert spec.policies is go2_task.GO2_POLICIES
    assert spec.inspect is False


def test_spec_has_no_core_episode_clock() -> None:
    """The Go2 spec is external and clock-free; ``max_steps`` owns time."""
    _require_go2_assets()

    spec = go2_task.make_go2_velocity_spec(num_envs=1)
    assert not hasattr(spec, "episode_length_s")
    run = spec.make_run()
    try:
        assert run.spec.termination_mode == "external"
        assert "episode_length_s" not in run.spec.cfg["env"]
        assert run.spec.cfg["world_base"]["mjcf"].endswith("flat_ground.xml")
        assert [entry["name"] for entry in run.spec.cfg["articulations"]] == ["go2"]
        assert run.spec.cfg["articulations"][0]["spawn"] == {"pos": [0.0, 0.0, 0.4]}
    finally:
        run.close()


def test_flat_world_go2_parity_keeps_the_asset_contact_options(one_env: Any) -> None:
    """The flat world inherits the Go2 asset's ``<option>`` block verbatim."""
    import mujoco
    from unigym.assets.release import resolve_release_ref

    profile = go2_task.go2_profile()
    assert profile.assets is not None
    source = mujoco.MjModel.from_xml_path(resolve_release_ref(profile.assets.mjcf))
    model = one_env.env.mj_model

    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.opt.impratio == 100.0
    assert model.opt.cone == source.opt.cone
    assert model.opt.impratio == source.opt.impratio


def test_one_env_reset_and_step_are_finite(one_env: Any) -> None:
    """One env returns finite ``(1, 48)`` obs and ``(1,)`` reward."""
    obs, info = one_env.reset()
    assert obs.shape == (1, go2_task.OBS_DIM)
    assert torch.isfinite(obs).all()
    state = info["state"]
    assert state.t == 0
    robot = state.robot(GO2_ARTICULATION)
    assert robot.joint_pos.shape == (1, len(GO2_JOINTS))
    assert robot.gravity_b.shape == (1, 3)
    torch.testing.assert_close(obs[:, -3:], one_env.command)

    for _ in range(3):
        obs, reward, terminated, truncated, info = one_env.step(_standby(1))
        assert obs.shape == (1, go2_task.OBS_DIM)
        assert torch.isfinite(obs).all()
        assert reward.shape == (1,)
        assert torch.isfinite(reward).all()
        assert terminated.shape == (1,) and terminated.dtype is torch.bool
        assert truncated.shape == (1,) and truncated.dtype is torch.bool
        assert set(info["termination_reasons"]) == {
            "tilt",
            "low_height",
            "base_contact",
            "time_limit",
        }


def test_command_is_resampled_and_visible_on_reset(one_env: Any) -> None:
    """Each reset samples a new in-range command and exposes it in the obs."""
    obs, _ = one_env.reset()
    first = one_env.command.clone()
    assert first.shape == (1, 3)
    for index, (low, high) in enumerate(go2_task.COMMAND_RANGES):
        assert bool(((first[:, index] >= low) & (first[:, index] <= high)).all())
    torch.testing.assert_close(obs[:, -3:], first)

    obs, _ = one_env.reset()
    second = one_env.command.clone()
    assert not torch.equal(first, second)
    torch.testing.assert_close(obs[:, -3:], second)


def test_tilt_and_low_height_fire_named_termination_reasons() -> None:
    """A forced roll terminates tilt; a forced low base terminates low_height."""
    _require_go2_assets()
    env = go2_task.make_go2_velocity_task(num_envs=2, max_steps=100)
    try:
        env.reset()
        ids = torch.arange(2, dtype=torch.int64)
        half_turn = math.sqrt(0.5)
        position = torch.tensor([[0.0, 0.0, 0.3], [0.0, 0.0, 0.05]], dtype=torch.float32)
        quat = torch.tensor(
            [[half_turn, half_turn, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        )
        zeros = torch.zeros((2, 3), dtype=torch.float32)
        env.run.set_root_state(
            ids,
            name=GO2_ARTICULATION,
            pos_w=position,
            quat_wxyz=quat,
            lin_vel_w=zeros,
            ang_vel_w=zeros,
        )
        env.run.settle(ids)

        _, reward, terminated, truncated, info = env.step(_standby(2))
        assert torch.isfinite(reward).all()
        assert terminated.tolist() == [True, True]
        assert not bool(truncated.any())

        reasons = info["termination_reasons"]
        assert reasons["tilt"].tolist() == [True, False]
        assert reasons["low_height"].tolist() == [False, True]
        assert reasons["time_limit"].tolist() == [False, False]
    finally:
        _close(env)


def test_env_local_step_budget_truncates_with_time_limit_reason() -> None:
    """The per-env counter truncates at ``max_steps`` with ``time_limit``."""
    _require_go2_assets()
    env = go2_task.make_go2_velocity_task(num_envs=2, max_steps=2)
    try:
        env.reset()
        _, _, terminated, truncated, _ = env.step(_standby(2))
        assert terminated.tolist() == [False, False]
        assert truncated.tolist() == [False, False]

        _, _, terminated, truncated, info = env.step(_standby(2))
        assert terminated.tolist() == [False, False]
        assert truncated.tolist() == [True, True]
        assert info["termination_reasons"]["time_limit"].tolist() == [True, True]
        assert info["termination_reasons"]["tilt"].tolist() == [False, False]
        assert info["termination_reasons"]["low_height"].tolist() == [False, False]
        assert info["termination_reasons"]["base_contact"].tolist() == [False, False]

        # Done rows were reset, so their own counter starts over.
        _, _, terminated, truncated, _ = env.step(_standby(2))
        assert terminated.tolist() == [False, False]
        assert truncated.tolist() == [False, False]
    finally:
        _close(env)


def test_eight_env_construction_and_one_step() -> None:
    """Eight envs build, reset and step with finite batched tensors."""
    _require_go2_assets()
    env = go2_task.make_go2_velocity_task(num_envs=8, max_steps=50, seed=2)
    try:
        obs, _ = env.reset()
        assert env.command.shape == (8, 3)
        obs, reward, terminated, truncated, info = env.step(_standby(8))
        assert obs.shape == (8, go2_task.OBS_DIM)
        assert torch.isfinite(obs).all()
        assert reward.shape == (8,)
        assert torch.isfinite(reward).all()
        assert terminated.shape == (8,) and terminated.dtype is torch.bool
        assert truncated.shape == (8,) and truncated.dtype is torch.bool
        assert set(info["termination_reasons"]) == {
            "tilt",
            "low_height",
            "base_contact",
            "time_limit",
        }
    finally:
        _close(env)


# --------------------------------------------------------------------------- #
# contact declaration, timers, air-time reward and base termination
# --------------------------------------------------------------------------- #
GO2_FEET_BODIES = go2_task.GO2_FEET_BODIES


def _timer_task(*, num_envs: int = 2, control_dt: float = 0.1) -> Any:
    """A task shell with the task-local timers and a typed obs registry."""
    from types import SimpleNamespace

    task = object.__new__(go2_task.Go2VelocityTask)
    task.device = torch.device("cpu")
    task.num_envs = int(num_envs)
    task.control_dt = float(control_dt)
    task.env = SimpleNamespace(
        obs=make_fake_obs_registry(
            num_envs=task.num_envs,
            robot=GO2_ARTICULATION,
            joints=GO2_JOINTS,
            contacts={"feet": ((len(GO2_FEET_BODIES),), "bool"), "base": ((1,), "bool")},
            command=True,
            prev_action=True,
        )
    )
    task._command = torch.zeros((task.num_envs, 3), dtype=torch.float32)
    task._init_contact_state()
    return task


def _typed_state(registry: Any, *, values: dict[str, Any] | None = None) -> Any:
    """A TaskState over a fake typed registry (optionally raw values)."""
    base = registry.snapshot()
    snapshot = (
        base
        if values is None
        else ObsSnapshot(
            values,
            schema=registry.schema,
            num_envs=base.num_envs,
            device=base.device,
        )
    )
    return go2_task.TaskState(t=0, num_envs=snapshot.num_envs, obs=snapshot)


def _fake_transition(
    *,
    num_envs: int = 2,
    feet: torch.Tensor | None = None,
    base: torch.Tensor | None = None,
    gravity_z: float = -1.0,
    base_height: float = 0.4,
) -> Any:
    """A ``Transition`` carrying the ten articulation leaves plus contacts."""
    zeros_joint = torch.zeros((num_envs, len(GO2_JOINTS)), dtype=torch.float32)
    zeros_vec = torch.zeros((num_envs, 3), dtype=torch.float32)
    gravity = zeros_vec.clone()
    gravity[:, 2] = gravity_z
    base_pos = zeros_vec.clone()
    base_pos[:, 2] = base_height
    registry = make_fake_obs_registry(
        num_envs=num_envs,
        robot=GO2_ARTICULATION,
        joints=GO2_JOINTS,
        contacts={"feet": ((len(GO2_FEET_BODIES),), "bool"), "base": ((1,), "bool")},
    )
    registry.set_robot(GO2_ARTICULATION, "joint_pos", zeros_joint.clone())
    registry.set_robot(GO2_ARTICULATION, "joint_pos_rel", zeros_joint.clone())
    registry.set_robot(GO2_ARTICULATION, "joint_vel", zeros_joint.clone())
    registry.set_robot(GO2_ARTICULATION, "applied_torque", zeros_joint.clone())
    registry.set_robot(GO2_ARTICULATION, "base_pos_w", base_pos)
    registry.set_robot(
        GO2_ARTICULATION,
        "base_quat_w",
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32).repeat(num_envs, 1),
    )
    registry.set_robot(GO2_ARTICULATION, "base_lin_vel_w", zeros_vec.clone())
    registry.set_robot(GO2_ARTICULATION, "base_ang_vel_w", zeros_vec.clone())
    registry.set_robot(GO2_ARTICULATION, "base_lin_vel_b", zeros_vec.clone())
    registry.set_robot(GO2_ARTICULATION, "base_ang_vel_b", zeros_vec.clone())
    registry.set_robot(GO2_ARTICULATION, "gravity_b", gravity)
    if feet is not None:
        registry.set_contact("feet", feet)
    if base is not None:
        registry.set_contact("base", base)
    info = go2_task.TaskState(t=0, num_envs=num_envs, obs=registry.snapshot())
    zeros = torch.zeros(num_envs, dtype=torch.bool)
    return go2_task.Transition(
        obs=torch.zeros((num_envs, 3), dtype=torch.float32),
        action=torch.zeros((num_envs, len(GO2_JOINTS)), dtype=torch.float32),
        next_obs=torch.zeros((num_envs, 3), dtype=torch.float32),
        terminated=zeros,
        truncated=zeros.clone(),
        info=info,
    )


def test_contact_groups_are_explicit_and_reach_the_run_config() -> None:
    """Go2 declares ``feet``/``base`` with explicit calf/base body names."""
    assert [group["name"] for group in go2_task.GO2_CONTACTS] == ["feet", "base"]
    feet, base = go2_task.GO2_CONTACTS
    assert feet["sensor_refs"] == [
        {"owner": "robot", "articulation": GO2_ARTICULATION, "body": body}
        for body in GO2_FEET_BODIES
    ]
    assert base["sensor_refs"] == [
        {"owner": "robot", "articulation": GO2_ARTICULATION, "body": "base"}
    ]
    for group in go2_task.GO2_CONTACTS:
        assert group["frame"] == "world"
        assert group["format"] == "contact"
        assert group["threshold"] == 1.0

    _require_go2_assets()
    spec = go2_task.make_go2_velocity_spec(num_envs=1)
    assert spec.contacts == go2_task.GO2_CONTACTS
    run = spec.make_run()
    try:
        assert run.spec.cfg["contacts"] == [dict(group) for group in go2_task.GO2_CONTACTS]
    finally:
        run.close()


def test_contact_keys_and_shapes_on_reset_and_step(one_env: Any) -> None:
    """MuJoCo publishes bool ``contact/feet`` (1, 4) and ``contact/base`` (1, 1)."""
    _, info = one_env.reset()
    for state in (info["state"], one_env.step(_standby(1))[4]["state"]):
        feet = state.obs.contact("feet").value
        base = state.obs.contact("base").value
        assert feet.shape == (1, len(GO2_FEET_BODIES))
        assert feet.dtype is torch.bool
        assert base.shape == (1, 1)
        assert base.dtype is torch.bool
    assert one_env.air_time.shape == (1, len(GO2_FEET_BODIES))
    assert one_env.last_air_time.shape == (1, len(GO2_FEET_BODIES))


def test_air_time_marks_first_contact_once_and_keeps_contact_time() -> None:
    """Two airborne ticks then contact: one first-contact, no refresh after."""
    task = _timer_task(num_envs=2, control_dt=0.1)
    airborne = torch.zeros((2, len(GO2_FEET_BODIES)), dtype=torch.bool)
    in_contact = torch.ones_like(airborne)

    task._update_contact_timers(airborne)
    task._update_contact_timers(airborne)
    torch.testing.assert_close(task.air_time, torch.full_like(task.air_time, 0.2))

    task._update_contact_timers(in_contact)
    assert task.first_contact.all()
    # The flight phase spans the previous two ticks plus the contact tick.
    torch.testing.assert_close(task.last_air_time, torch.full_like(task.last_air_time, 0.3))
    torch.testing.assert_close(task.air_time, torch.zeros_like(task.air_time))
    torch.testing.assert_close(task.contact_time, torch.full_like(task.contact_time, 0.1))

    task._update_contact_timers(in_contact)
    assert not task.first_contact.any()
    torch.testing.assert_close(task.last_air_time, torch.full_like(task.last_air_time, 0.3))
    torch.testing.assert_close(task.contact_time, torch.full_like(task.contact_time, 0.2))


def test_air_time_on_step_reads_feet_contact_and_rejects_bad_shapes() -> None:
    """``on_step`` reads ``contact/feet`` only, failing fast on bad input."""
    task = _timer_task(num_envs=2, control_dt=0.05)
    task._prev_action = torch.zeros((2, len(GO2_JOINTS)), dtype=torch.float32)
    task._action_rate = torch.zeros(2, dtype=torch.float32)
    task._step_count = torch.zeros(2, dtype=torch.int64)
    feet = torch.ones((2, len(GO2_FEET_BODIES)), dtype=torch.bool)
    registry = make_fake_obs_registry(
        num_envs=2,
        robot=GO2_ARTICULATION,
        joints=GO2_JOINTS,
        contacts={"feet": ((len(GO2_FEET_BODIES),), "bool")},
    )
    registry.set_contact("feet", feet)
    state = _typed_state(registry)

    task.on_step(torch.zeros((2, len(GO2_JOINTS))), state)
    assert task.first_contact.all()
    torch.testing.assert_close(task.last_air_time, torch.full_like(task.last_air_time, 0.05))

    with pytest.raises(go2_task.TaskConfigError, match="contact/feet"):
        task.on_step(
            torch.zeros((2, len(GO2_JOINTS))),
            _typed_state(registry, values={}),
        )
    with pytest.raises(go2_task.TaskConfigError, match="bool"):
        task.on_step(
            torch.zeros((2, len(GO2_JOINTS))),
            _typed_state(registry, values={"contact/feet": feet.float()}),
        )
    with pytest.raises(go2_task.TaskConfigError, match="shape"):
        task.on_step(
            torch.zeros((2, len(GO2_JOINTS))),
            _typed_state(
                registry,
                values={"contact/feet": torch.ones((2, 3), dtype=torch.bool)},
            ),
        )


def test_reset_clears_only_the_selected_contact_rows() -> None:
    """``on_reset`` zeroes timers for ``env_ids`` and leaves other rows alone."""
    task = _timer_task(num_envs=2, control_dt=0.1)
    task._rng = torch.Generator().manual_seed(0)
    task._command = torch.zeros((2, 3), dtype=torch.float32)
    task._prev_action = torch.zeros((2, len(GO2_JOINTS)), dtype=torch.float32)
    task._action_rate = torch.zeros(2, dtype=torch.float32)
    task._step_count = torch.zeros(2, dtype=torch.int64)
    airborne = torch.zeros((2, len(GO2_FEET_BODIES)), dtype=torch.bool)
    for _ in range(2):
        task._update_contact_timers(airborne)

    state = _typed_state(
        make_fake_obs_registry(
            num_envs=2,
            robot=GO2_ARTICULATION,
            joints=GO2_JOINTS,
            command=True,
            prev_action=True,
        )
    )
    task.on_reset(torch.tensor([1]), state)

    torch.testing.assert_close(task.air_time[0], torch.full((4,), 0.2))
    torch.testing.assert_close(task.air_time[1], torch.zeros(4))
    torch.testing.assert_close(task.last_air_time[1], torch.zeros(4))
    assert not task.first_contact[1].any()


def test_feet_air_time_reward_signs_and_zero_command() -> None:
    """First-contact bonus is signed, one-shot and zero on a near-zero command."""
    task = _timer_task(num_envs=1, control_dt=0.1)
    task._action_rate = torch.zeros(1, dtype=torch.float32)
    task._command = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)

    task._last_air_time = torch.tensor([[0.8, 0.0, 0.0, 0.0]], dtype=torch.float32)
    task._first_contact = torch.tensor([[True, False, False, False]])
    torch.testing.assert_close(task._feet_air_time_reward(), torch.tensor([0.3]))

    task._last_air_time = torch.tensor([[0.2, 0.0, 0.0, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(task._feet_air_time_reward(), torch.tensor([-0.3]))

    task._last_air_time = torch.tensor([[0.8, 0.2, 0.0, 0.0]], dtype=torch.float32)
    task._first_contact = torch.tensor([[True, True, False, False]])
    torch.testing.assert_close(task._feet_air_time_reward(), torch.tensor([0.0]))

    task._last_air_time = torch.tensor([[0.8, 0.0, 0.0, 0.0]], dtype=torch.float32)
    task._first_contact = torch.tensor([[True, False, False, False]])
    torch.testing.assert_close(task._feet_air_time_reward(), torch.tensor([0.3]))

    task._first_contact = torch.zeros_like(task._first_contact)
    torch.testing.assert_close(task._feet_air_time_reward(), torch.zeros(1))

    task._first_contact = torch.tensor([[True, False, False, False]])
    task._command = torch.tensor([[0.05, 0.05, 0.5]], dtype=torch.float32)
    torch.testing.assert_close(task._feet_air_time_reward(), torch.zeros(1))

    reward = task.reward(_fake_transition(num_envs=1, gravity_z=-1.0))
    assert reward.shape == (1,) and torch.isfinite(reward).all()


def test_base_contact_terminates_with_named_reason() -> None:
    """Any true ``contact/base`` terminates and is reported as base_contact."""
    task = _timer_task(num_envs=2)
    task._command = torch.zeros((2, 3), dtype=torch.float32)
    base = torch.tensor([[True], [False]])
    transition = _fake_transition(num_envs=2, base=base)

    terminated = task.terminated(transition)
    assert terminated.tolist() == [True, False]
    reasons = task.termination_reasons(
        transition,
        terminated,
        torch.zeros(2, dtype=torch.bool),
    )
    assert reasons["base_contact"].tolist() == [True, False]
    assert reasons["tilt"].tolist() == [False, False]
    assert reasons["low_height"].tolist() == [False, False]


def test_missing_base_contact_fails_fast() -> None:
    """A task built without the base group must not silently ignore it."""
    task = _timer_task(num_envs=1)
    transition = _fake_transition(num_envs=1, base_height=0.4)
    with pytest.raises(go2_task.TaskConfigError, match="contact/base"):
        task.terminated(transition)


def test_train_skrl_forwards_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--backend`` reaches the task builder unchanged."""
    previous_task = sys.modules.get("task")
    sys.modules["task"] = go2_task
    try:
        train_skrl = _load_module("go2_velocity_train_skrl_backend", "train_skrl.py")
        seen: dict[str, Any] = {}

        def _capture(**kwargs: Any) -> Any:
            seen.update(kwargs)
            raise RuntimeError("stop-after-forward")

        monkeypatch.setattr(train_skrl, "make_go2_velocity_gym", _capture)
        with pytest.raises(RuntimeError, match="stop-after-forward"):
            train_skrl.main(["--backend", "isaaclab2", "--num-envs", "2"])
    finally:
        if previous_task is None:
            sys.modules.pop("task", None)
        else:
            sys.modules["task"] = previous_task
    assert seen["backend"] == "isaaclab2"
    assert seen["num_envs"] == 2


def test_skrl_eight_env_smoke() -> None:
    """One manual-PPO iteration on eight Go2 envs when skrl is installed."""
    pytest.importorskip("skrl")
    pytest.importorskip("gymnasium")
    _require_go2_assets()

    previous_task = sys.modules.get("task")
    sys.modules["task"] = go2_task
    try:
        train_skrl = _load_module("go2_velocity_train_skrl", "train_skrl.py")
        code = train_skrl.main(
            [
                "--num-envs",
                "8",
                "--max-episode-steps",
                "50",
                "--max-iterations",
                "1",
                "--rollout-steps",
                "2",
            ]
        )
    finally:
        if previous_task is None:
            sys.modules.pop("task", None)
        else:
            sys.modules["task"] = previous_task
    assert code == 0
