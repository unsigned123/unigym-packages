# Go2 velocity tracking smoke

A runnable flat-ground velocity-tracking smoke for the real Menagerie Go2 in
MuJoCo. It is repo-only: `examples` is not part of the installed `unigym-rl`
wheel and adds no package API.

What it proves:

- the installed `unigym-robots` pack profile loads (joints, standby pose,
  feet, actuation) after the Go2 foot-body fix;
- explicit `OrderedActionSpec` / `PolicyBinding` constants for the 12 leg
  joints;
- an `EnvSpec` over the backend-neutral `World.flat()` ground, with the
  profile-owned Go2 asset and a run-owned spawn;
- a task-owned command buffer, observation, reward and episode clock;
- typed `TaskState.articulation("go2")` access instead of string extra keys;
- 1-env and 8-env reset/step;
- a manual skrl PPO smoke with no trainer abstraction.

## Assets (required first)

The example never downloads anything. Install the assets package, then fetch
the pinned sets for the backends you use:

```bash
pip install unigym-robots
unigym-robots download --set mjcf-v0      # MuJoCo backend
unigym-robots download --set usd-go2-v0   # IsaacLab 2.x backend
```

`make_go2_velocity_spec()` resolves `release:go2-mjcf` from that install and
fails with a teaching error if the file is absent. It does not contact the
network.

## Environment

| Item | Value |
| --- | --- |
| Robot | `go2` pack profile, 12 leg joints, `standard_pd` kp=40 / kd=1 |
| World | `World.flat()` (backend-neutral ground, no robot) |
| Spawn | run-owned `{"pos": (0.0, 0.0, 0.4)}` |
| Action | 12 absolute position targets (radians) |
| Observation | 48 columns (below) |
| Command | `[vx, vy, yaw_rate]`, sampled per env on every reset |
| Reward | linear tracking + yaw tracking + alive - action rate |
| Terminated | projected gravity tilt or low base height |
| Truncated | env-local step counter |
| Reasons | `tilt`, `low_height`, `time_limit` |

Observation columns, all read through the typed
`state.articulation("go2")` accessor:

```text
base_lin_vel_b (3) | base_ang_vel_b (3) | gravity_b (3)
| joint_pos - standby (12) | joint_vel (12)
| previous action (12) | command (3)
```

`TaskState.articulation` reads the extra record because this smoke does not
request a core observation catalog; the extra record always publishes the
namespaced articulation keys. Joint columns follow the profile joint order
`GO2_JOINTS`; at bind time the task verifies every name against the live model
on backends that expose one (`mujoco`, `mjwarp`) and otherwise falls back to the
validated profile joint order (`isaaclab2` publishes no live proxy).

Reward terms (no `RewardManager`/`TerminationManager`; direct hooks keep the
task-owned command/action/clock state in one place):

```text
1.5 * exp(-||v_xy - cmd_xy||^2 / 0.25)
+ 0.5 * exp(-(w_z - cmd_yaw)^2 / 0.25)
+ 0.5
- 0.01 * ||action - previous_action||^2
```

`terminated` fires when `gravity_b.z > -0.5` (about 60 degrees of tilt) or the
base height is below 0.18 m. `truncated` fires when that env's own counter
reaches `max_steps`. There is no runner auto-reset: `TaskEnv` resets only the
done rows, and every episode's command/previous action/counter is task-owned.

## Ownership notes

The example follows the final ownership split:

```text
RobotProfile  -> robot identity (asset, joints, pose, bindings, actuation)
World.flat()  -> environment only (ground); no robot, no explicit world_base
EnvSpec/Run   -> robot instance + spawn + action_spec + policies
TaskState     -> typed articulation observations
```

The profile keeps its `release:go2-mjcf` / `release:go2-usd` assets; the
backend only selects the format it supports. The example no longer resolves
`scene.xml`, calls `asset_path("go2-mjcf")`, wraps the world MJCF, or drops
the profile asset with `assets=None`; the same MJCF is attached exactly once
by the run layer. The action group is named `legs` (`GO2_ACTION_GROUP`) on the
`go2` articulation, matching the profile's physical actuation group. Recorded
state keys stay namespaced by the articulation (`articulation/go2/...`), not
by the action-group name.

## Run

From the repo root with the core checkout on `PYTHONPATH`:

```bash
PYTHONPATH=/data1/code/dlx/unigym-core:/data1/code/dlx/unigym-rl/src \
  /data1/code/dlx/unigym-core/.venv/bin/python \
  examples/go2_velocity/train_skrl.py --num-envs 8 --max-iterations 1 \
  --rollout-steps 4
```

`--backend` selects the simulation backend (default `mujoco`):

```bash
# MuJoCo on CPU
... examples/go2_velocity/train_skrl.py --backend mujoco --device cpu \
  --num-envs 8 --max-iterations 1 --rollout-steps 4

# IsaacLab 2.x on GPU
... examples/go2_velocity/train_skrl.py --backend isaaclab2 --device cuda:0 \
  --num-envs 1024 --max-iterations 1 --rollout-steps 4
```

IsaacLab2 still needs its Kit/AppLauncher bootstrap before the backend is
imported. This script does not start Kit itself; launch it through the IsaacLab
environment/launcher wrapper (for example the dog `unigym-isaaclab232` entry
point). A plain `python ... --backend isaaclab2` without that bootstrap fails
with `No module named 'omni'`.

The default is 8 envs, 500 max episode steps, 1000 iterations and 64 rollout
steps. The script prints per-iteration mean reward and episode count. It does
not checkpoint, log to a service, or claim convergence.

## Known limitations

- **Action transform vs log-probability.** The Gaussian policy samples
  unbounded raw actions; the env receives
  `standby_pose + 0.25 * clip(raw, -1, 1)`. The recorded action is the raw
  sample, so its log-probability does not exactly match the position target
  that moved the plant. This is intentional and documented, not hidden.
- Flat ground only: no terrain, heightmap, parkour or command curriculum.
- No domain randomization, observation normalization, checkpoint format or
  evaluation harness.
- No trainer abstraction and no convergence guarantee.
