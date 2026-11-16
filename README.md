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

## R2.5 status

R2.5 adds construction and bookkeeping sugar only: no trainer and no new env
type.

### Programmatic `EnvSpec`

`EnvSpec` builds a fresh, lazy core `Run` from plain fields instead of a yaml
file. Termination ownership is fixed: every run declares
`termination_mode="external"` and
`runner={auto_reset: false, reset_on_termination: false}`, and `EnvSpec`
exposes no way to override either.

```python
from unigym.world import World
from unigym_rl import EnvSpec

spec = EnvSpec(
    world=World.base(mjcf="robot.xml"),
    robot={"name": "my_robot", "actuation": [...]},
    num_envs=4,
    inspect=True,
)

task = spec.make_task(MyTask)       # fresh run + task; caller closes both
# gym = spec.make_gym(MyTask)       # requires the gymnasium extra (lazy import)
```

`make_task` validates that its argument is a `TaskEnv` subclass.
`make_gym` imports `GymTaskEnv` lazily, so importing `unigym_rl` or using
`EnvSpec` never imports `gymnasium`. `make_run` returns a fresh `Run` on every
call and never builds the backend.

### Action history: `RingBuffer` + `on_step`

`RingBuffer` stores a fixed-capacity chronological history per env. Each env
has its own write pointer and count, so resetting one row never disturbs
another; unwritten slots are zero-filled. It has no dependency on `TaskEnv`
and never pushes implicitly.

```python
from unigym_rl import RingBuffer

class MyTask(TaskEnv):
    def on_bind(self, context):
        self.actions = self.register_buffer(
            RingBuffer(context.num_envs, capacity=16, shape=(context.action_spec.dim,))
        )

    def on_step(self, action, state):
        self.actions.push(action)

    def observe(self, state):
        ...
```

`register_buffer` returns the buffer, is idempotent by identity, and only
requires a callable `reset(env_ids=None)`. Registered buffers are reset before
the post-reset observation is computed (all ids on `reset`, done ids on the
`step` done path). `on_step` is called exactly once per `step`, after the
runner advances and before `observe` computes the next observation; it is not
called by `reset`. This round adds no automatic action push.

## CLI options sugar

R2.6 adds an optional `unigym_rl.cli` parser for a small set of run knobs.
It is sugar only: no trainer, no config-file format, and no new env type.
`import unigym_rl` does not import `unigym_rl.cli`; import it explicitly.

```python
from unigym_rl.cli import parse_args

opts = parse_args()  # or parse_args(["--num-envs", "4"])
spec = EnvSpec(
    world=world,
    robot=robot,
    num_envs=opts.env.num_envs,
    device=opts.env.device,
    seed=opts.env.seed,
)
task = spec.make_task(MyTask, max_steps=opts.env.max_episode_steps)
# a future trainer would read opts.train.max_iterations / rollout_steps
```

```bash
python train.py --num-envs 8 --max-episode-steps 500 --device cpu \
  --seed 0 --max-iterations 1000 --rollout-steps 64
```

Hyphenated flags are canonical; underscore aliases (`--num_envs`,
`--max_episode_steps`, `--max_iterations`, `--rollout_steps`) behave
identically. Defaults: `num_envs=8`, `max_episode_steps=500`, `device="cpu"`,
`seed=0`, `max_iterations=1000`, `rollout_steps=64`. Positive-int fields reject
values below 1 both on the CLI and in direct dataclass construction; an empty
`device` is rejected.

Parameter ownership: `num_envs` / `device` / `seed` go to `EnvSpec`;
`max_episode_steps` is the task truncation budget; `max_iterations` /
`rollout_steps` are reserved for a future trainer. The parser never constructs
an `EnvSpec`, `World`, robot, task, or trainer, and no CLI value is propagated
to `TaskEnv.reset`.

## Examples

- `examples/cartpole/` — a runnable fixed-base MuJoCo CartPole demo: a
  programmatic `EnvSpec`, a local demo `RobotProfile`, one slider action column
  with a passive observed pole, optional randomized resets via
  `set_joint_state` + `settle`, and a manual skrl PPO loop. It is repo-only and
  is not installed into the wheel; see `examples/cartpole/README.md`.

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
  /data1/code/dlx/unigym-core/.venv/bin/python -m ruff check src tests examples
PYTHONPATH=/data1/code/dlx/unigym-core:/data1/code/dlx/unigym-rl/src \
  /data1/code/dlx/unigym-core/.venv/bin/python -m ruff format --check src tests examples
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
| `register_buffer(buffer)` | register a history buffer; reset on env reset | empty |
| `on_step(action, state)` | once per step, before `observe` | no-op |
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
  __init__.py       # TaskEnv / EnvSpec / RingBuffer exports + lazy GymTaskEnv
  context.py        # TaskContext / TaskState / Transition / EpisodeEnd
  spec.py           # programmatic external-termination Run builder (R2.5)
  buffers.py        # per-env RingBuffer history (R2.5)
  cli.py            # optional argparse option dataclasses (R2.6)
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
  test_spec.py              # programmatic EnvSpec (R2.5)
  test_buffers.py           # RingBuffer history (R2.5)
  test_cli.py               # optional CLI option sugar (R2.6)
```
