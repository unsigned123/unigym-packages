# unigym-rl

L2 task-environment layer over [`unigym`](https://github.com/unsigned123/unigym)
core.

```text
core Run / ExternalRunner / reset / state
  -> unigym-rl TaskEnv
  -> user or RL framework training loop
```

`unigym-rl` owns the Gymnasium-shaped vector environment and the task hooks.
It does **not** own the training loop, optimizer, checkpoint format, reward
implementation, or policy execution/deployment (L3 stays in core).

## R0 status

R0 is the package skeleton plus the fake-env `TaskEnv` contract only:

- `TaskEnv` base class with the final `reset` / `step` / `close` driver;
- frozen `TaskContext` / `TaskState` / `Transition` / `EpisodeEnd` records;
- teaching errors (`TaskEnvError`, `TaskConfigError`, `TaskContractError`);
- fake run/runner/env fixtures under `unigym_rl.testing`;
- unit/contract tests against the fakes.

Not implemented in R0: a real backend, skrl/rsl-rl integration, a training
loop, reward normalization, terrain, multi-agent coordination, or Gymnasium
subclassing. Optional extras (`gymnasium`, `skrl`, `rsl-rl`) are declared in
`pyproject.toml` (rsl-rl remains planned, not implemented).

## R1 status

R1 proves the same `TaskEnv` contract on one **real MuJoCo robot** with a
manual rollout and no RL framework:

- a self-contained, MIT-clean two-hinge MJCF fixture at
  `tests/fixtures/tiny_arm.xml` (copied from core; no core test path is
  imported);
- a test-only robot profile registered through `unigym.robots.register_robot`
  (`tiny`, `body = shoulder, elbow`, base body `base`) and restored after
  each test;
- a real `Run` with `backend="mujoco"`, `adapter="manual"`,
  `num_envs=1` and `2`, `termination_mode="external"`, and
  `runner={auto_reset: false, reset_on_termination: false}`;
- a `TaskEnv` subclass whose observation is the flattened
  `articulation/tiny/{joint_pos,joint_vel}` state, with deterministic reward
  and task-owned `terminated` / `truncated` / named reason masks;
- manual `env.reset()` / `env.step(actions)` rollouts asserting done-only
  `runner.reset`, `on_episode_end` before reset, pre-reset
  `final_observation`, post-reset returned observation, and an empty
  all-False core termination group.

R1 adds **no** trainer, no Gymnasium/skrl/rsl-rl dependency, no terrain, no
multi-agent support, and no core changes. The Run is built from the
programmatic config shape rather than a yaml file; see
`tests/test_task_env_mujoco.py`.

## R2 status

R2 adds the optional Gymnasium `VectorEnv` face and a short skrl PPO smoke,
still with no trainer in the package:

- `unigym_rl.gym.GymTaskEnv` subclasses `gymnasium.vector.VectorEnv` and wraps
  one `TaskEnv`;
- `num_envs`, `device`, `single_observation_space` / `observation_space`
  `(num_envs, obs_dim)` and `single_action_space` / `action_space`
  `(num_envs, action_dim)` `Box` spaces;
- `metadata["autoreset_mode"] = AutoresetMode.SAME_STEP` (the task already
  returns post-reset observations for done rows);
- `reset(seed, options)` and `step(actions)` accept NumPy or torch batches and
  return NumPy obs/reward/terminated/truncated; the task `info` dictionary is
  passed through intact (`final_observation`, `final_info`, `episode_end`,
  `termination_reasons`, `state`);
- `close()` delegates to `TaskEnv`; `render()` raises an honest
  `NotImplementedError`;
- `GymTaskEnv` is exported lazily from `unigym_rl` and from `unigym_rl.gym`, so
  importing the package never imports `gymnasium`.

The optional dependencies are declared as extras (the base install stays free
of RL frameworks):

```bash
pip install 'unigym-rl[gymnasium]'   # gymnasium>=1.0,<2
pip install 'unigym-rl[skrl]'        # skrl>=2.1,<3
```

R2 adds **no** trainer, no `train.py`, no rsl-rl, no terrain and no multi-agent
support. The skrl lane uses a manual `wrap_env` + `agent.act` /
`record_transition` / `update` loop (skrl 2.1 ships `PPO_CFG`, not a
`PPO_DEFAULT_CONFIG` dict) and is deterministic under `torch.manual_seed(0)`.

## Installing / testing

R0 needs no install. Run the tests with the core checkout on `PYTHONPATH`
(adjust the interpreter for your environment):

```bash
PYTHONPATH=/data1/code/dlx/unigym-core:/data1/code/dlx/unigym-rl/src \
  /data1/code/dlx/unigym-core/.venv/bin/python -m pytest tests -q
```

Static checks:

```bash
PYTHONPATH=/data1/code/dlx/unigym-core:/data1/code/dlx/unigym-rl/src \
  /data1/code/dlx/unigym-core/.venv/bin/python -m ruff check src tests
PYTHONPATH=/data1/code/dlx/unigym-core:/data1/code/dlx/unigym-rl/src \
  /data1/code/dlx/unigym-core/.venv/bin/python -m ruff format --check src tests
```

## External termination mode

`TaskEnv` requires the run to hand termination ownership to the task layer.
The run config must declare:

```yaml
termination_mode: external
terminations: []
runner:
  auto_reset: false
  reset_on_termination: false
```

In this mode core builds an empty, never-firing termination group, never
resets a terminated row itself, and reports `StepResult.terminated` as
all-False. `TaskEnv` asserts that invariant and raises `TaskContractError` if
core ever fires. Core still reports per-env episode-clock due ids through
`StepResult.reset_env_ids`; `TaskEnv` merges those into `truncated` (a due env
that is also task-terminated stays terminated, not truncated) and resets only
the done ids.

## TaskEnv hooks

Subclass `TaskEnv` and implement `observe` and `reward`. Everything else has
a safe default.

```python
class MyTask(TaskEnv):
    def observe(self, state):            # required -> (N, obs_dim) tensor
        return state.obs_cache["proprio"]

    def reward(self, transition):        # required -> (N,) tensor
        return torch.zeros(self.num_envs)

    def terminated(self, transition):    # default: all False
        return torch.zeros(self.num_envs, dtype=torch.bool)

    def truncated(self, transition):     # default: all False
        return torch.zeros(self.num_envs, dtype=torch.bool)
```

Full hook list:

| Hook | Purpose | Default |
| --- | --- | --- |
| `on_bind(context)` | capture `context.env` / controls | no-op |
| `observe(state)` | flat observation | required |
| `reward(transition)` | per-env reward | required |
| `terminated(transition)` | terminal mask | all False |
| `truncated(transition)` | truncation mask | all False |
| `termination_reasons(transition, terminated, truncated)` | log masks | `{}` |
| `on_reset(env_ids, state)` | after every reset | no-op |
| `on_episode_end(env_ids, state, transition)` | done rows, before reset | no-op |
| `on_close()` | final cleanup | no-op |

Final driver API (do not override):

```python
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(flat_actions)
env.close()
```

`step` receives a flat `(N, action_dim)` float tensor and unflattens it with
the run's `OrderedActionSpec` into joint-addressed `NamedAction` objects (no
action-group identity). `info` carries:

- `state`: post-reset `TaskState`;
- `termination_reasons`: `{name: (N,) bool}`;
- `final_observation`: pre-reset `next_obs` rows for done envs, or `None`;
- `final_info`: length-`N` tuple, `None` for live envs and one `EpisodeEnd`
  per done env;
- `episode_end`: batch-level `EpisodeEnd` over done rows, or `None`.

Ordering guarantees: reward, termination masks, reasons and the pre-reset
final observation are all computed before reset; `on_episode_end` runs before
reset; only done ids are reset; the observation returned by `step` is
post-reset; the Run is never stopped or closed by `TaskEnv`.

## Layout

```text
src/unigym_rl/
  __init__.py       # TaskEnv exports + lazy GymTaskEnv
  context.py        # TaskContext / TaskState / Transition / EpisodeEnd
  task.py           # TaskEnv + reset/step driver
  gym.py            # optional Gymnasium VectorEnv wrapper (lazy)
  errors.py
  testing/
    __init__.py
    fakes.py        # fake run/runner/env (tests only)
tests/
  fixtures/
    tiny_arm.xml     # self-contained MIT-clean MuJoCo fixture (R1)
  test_task_env.py
  test_task_env_fake.py
  test_task_env_mujoco.py   # real MuJoCo manual rollout (R1)
  test_gym_task_env.py      # Gymnasium VectorEnv face (R2)
  test_skrl_smoke.py        # short skrl PPO smoke (R2)
```
