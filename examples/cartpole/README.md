# CartPole demo

A small, runnable end-to-end demo of `unigym-rl` on a real MuJoCo CPU model.
It is repo-only: `examples` is not part of the installed `unigym-rl` wheel and
adds no package API.

What it proves:

- a demo `RobotProfile` (with explicit actuation) for a self-contained MJCF;
- explicit `OrderedActionSpec` / `PolicyBinding` constants;
- an `EnvSpec` over `World.base(mjcf=...)` with task-owned termination;
- a fixed-base articulation with exactly one action column (`slider`);
- a passive, observed pole joint (`hinge_1`);
- a task-owned observation/reward/termination/truncation contract;
- optional randomized resets through `set_joint_state` + `settle`;
- a manual skrl PPO loop with no trainer abstraction.

## Environment

| Item | Value |
| --- | --- |
| Articulation | `cartpole` (fixed base, body `cart`) |
| Action | one column, `slider` position target, clipped by the demo to `[-1.8, 1.8]` |
| Observation | `[cart_x, pole_theta, cart_dx, pole_dtheta]` |
| Reward | `1 - 0.5*(pole_theta/pole_limit)^2 - 0.1*(cart_x/cart_limit)^2` |
| Terminated | `abs(cart_x) > cart_limit` (1.5 m) or `abs(pole_theta) > pole_limit` (12 deg) |
| Truncated | env-local step counter reaches `max_steps` |
| Reasons | `cart_limit`, `pole_limit`, `time_limit` |

`make_cartpole_spec()` passes an explicit `RobotProfile`, an explicit
`OrderedActionSpec` and explicit `PolicyBinding` policies to `EnvSpec`; the
action layout is never inferred from the `body` joint set. `EnvSpec` also
requires an explicit `World`; it never constructs one.

The policy-order `articulation/cartpole/joint_pos` slice only covers the
actuated `body` joint set, so it contains the cart but not the passive pole.
The demo therefore reads the cart and pole joints by name from the live core
model (`run.build().live.joints[...]`), which keeps the column order explicit.
No `set_root_state` is used: the CartPole base is fixed.

Randomized resets are off by default. Pass `randomize_reset=True` (and
optionally `reset_position_scale=...`, `seed=...`) to
`make_cartpole_task`/`make_cartpole_gym` to sample both joint positions and
write them through `Run.set_joint_state` followed by `Run.settle`.

## Install / run

The gymnasium and skrl extras are needed only for the training script:

```bash
pip install -e '.[gymnasium,skrl]'
```

From the repo root with the core checkout on `PYTHONPATH`:

```bash
PYTHONPATH=/data1/code/dlx/unigym-core:/data1/code/dlx/unigym-rl/src \
  /data1/code/dlx/unigym-core/.venv/bin/python \
  examples/cartpole/train_skrl.py --max-iterations 10 --rollout-steps 16
```

A one-iteration smoke:

```bash
PYTHONPATH=/data1/code/dlx/unigym-core:/data1/code/dlx/unigym-rl/src \
  /data1/code/dlx/unigym-core/.venv/bin/python \
  examples/cartpole/train_skrl.py --max-iterations 1 --rollout-steps 2
```

The script prints per-iteration mean reward and episode count. It does not
checkpoint, log to a service, or claim convergence.

## Known limitations

- **Action clipping vs log-probability.** The Gaussian policy samples
  unbounded actions; the env receives them clipped to `[-1.8, 1.8]`. The
  recorded log-probability belongs to the sampled (unclipped) action, so it is
  an approximation of the action that moved the plant. This is intentional and
  not hidden by the demo.
- No trainer, checkpoint format, evaluation harness, or convergence guarantee.
- Fixed-base, single-articulation, one-action cart only.
- Observation reads the live model rather than the policy-order `extra` slice,
  because the passive pole is absent from that slice.

## Asset provenance

`cartpole.xml` is an unmodified copy of the MuJoCo Playground CartPole asset
(commit `56420458c8d81acd6942779d9dab327382117175`, Apache-2.0, Copyright The
dm_control Authors / Google DeepMind). See `THIRD_PARTY_NOTICES.md` and
`THIRD_PARTY_LICENSES/Apache-2.0.txt`.
