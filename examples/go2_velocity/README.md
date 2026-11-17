# Go2 velocity tracking smoke

A runnable flat-ground velocity-tracking smoke for the real Menagerie Go2 in
MuJoCo. It is repo-only: `examples` is not part of the installed `unigym-rl`
wheel and adds no package API.

What it proves:

- the installed `unigym-robots` pack profile loads (joints, standby pose,
  feet, actuation) after the Go2 foot-body fix;
- explicit `OrderedActionSpec` / `PolicyBinding` constants for the 12 leg
  joints;
- an `EnvSpec` over the downloaded Menagerie `scene.xml` flat ground;
- a task-owned command buffer, observation, reward and episode clock;
- 1-env and 8-env reset/step;
- a manual skrl PPO smoke with no trainer abstraction.

## Assets (required first)

The example never downloads anything. Install the assets package, then fetch
the pinned MJCF set:

```bash
pip install unigym-robots
unigym-robots download --set mjcf-v0
```

`make_go2_velocity_spec()` resolves `release:go2-mjcf` from that install and
fails with a teaching error if the file is absent. It does not contact the
network.

## Environment

| Item | Value |
| --- | --- |
| Robot | `go2` pack profile, 12 leg joints, `standard_pd` kp=40 / kd=1 |
| World | downloaded Menagerie `scene.xml` (floor + lights) |
| Action | 12 absolute position targets (radians) |
| Observation | 48 columns (below) |
| Command | `[vx, vy, yaw_rate]`, sampled per env on every reset |
| Reward | linear tracking + yaw tracking + alive - action rate |
| Terminated | projected gravity tilt or low base height |
| Truncated | env-local step counter |
| Reasons | `tilt`, `low_height`, `time_limit` |

Observation columns, all read from the core `state.extra` record:

```text
base_lin_vel_b (3) | base_ang_vel_b (3) | gravity_b (3)
| joint_pos - standby (12) | joint_vel (12)
| previous action (12) | command (3)
```

`state.extra` is used instead of `state.obs_cache` because this smoke does not
request a core observation catalog; the extra record always publishes the
namespaced articulation keys. Joint columns follow the profile joint order
`GO2_JOINTS`; the task verifies every name on the live model at bind time.

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

## World and profile notes

The MuJoCo backend requires an explicit `world_base`. A bare
`World().flat_ground()` cannot build (this checkout also lacks the default
inline material texture), so this example uses the flat ground already in the
downloaded Menagerie `scene.xml`. Because that scene owns the Go2 MJCF, the
profile passed to the run is a copy with `assets=None`; the pack profile is
still the source of all joint/body/actuation metadata. The action group is
named `go2` (same as the articulation) so the core collect sink's
per-articulation metric key stays aligned; the profile's physical actuation
group is still named `legs`.

## Run

From the repo root with the core checkout on `PYTHONPATH`:

```bash
PYTHONPATH=/data1/code/dlx/unigym-core:/data1/code/dlx/unigym-rl/src \
  /data1/code/dlx/unigym-core/.venv/bin/python \
  examples/go2_velocity/train_skrl.py --num-envs 8 --max-iterations 1 \
  --rollout-steps 4
```

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
