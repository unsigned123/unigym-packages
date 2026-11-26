# unigym-rl

L2 task-environment layer over [`unigym`](https://github.com/unsigned123/unigym)
core. This package is developed and released in
`packages/unigym-rl` of
[`unigym-packages`](https://github.com/unsigned123/unigym-packages); core stays
in its own repository.

```text
core Run / ExternalRunner / reset / state
  -> unigym-rl TaskEnv
  -> user or RL framework training loop
```

`unigym-rl` owns the Gymnasium-shaped vector environment and the task hooks.
It does **not** own the training loop, optimizer, checkpoint format, reward
implementation, or policy execution/deployment (L3 stays in core).

## Release 0.2.2: core 0.110 pack model v2

`unigym-rl 0.2.2` is a patch release. It moves the core range to
`unigym >=0.110,<0.111` for the core pack model v2 cutover. The task/lifecycle
contract is unchanged from `0.2.1`; the Go2 velocity example now resolves its
profile through `PackLoader.from_pack_set(...).robot_profile(...)` instead of
the removed `Pack.robot(...)` helper. `unigym-rl` itself still does not use
packs, subscriptions or a trainer.

## Release 0.2.1: core 0.109 range

`unigym-rl 0.2.1` is a patch release. It moves the core range to
`unigym >=0.109,<0.110` so the lane resolves against the additive
`ValueStore.subscribe` core release, and `task.py` now reads
`runner.ticks_done` directly for the post-step tick count instead of the
`getattr(self.runner, "ticks_done", int(result.t))` fallback. The
task/lifecycle behavior is otherwise the `0.2.0` contract below; `unigym-rl`
does not use subscriptions.

## Release 0.2.0: core 0.108 lockstep

`unigym-rl 0.2.0` requires `unigym >=0.108,<0.109` and adapts to the
post-deletion core with no compatibility shims:

- `EnvSpec` no longer declares `episode_length_s`; core has no episode clock
  and the task owns every boundary.
- `TaskEnv.step` ends its task-owned done rows through
  `runner.terminate(env_ids, facts, reset=True)` (one `custom.*` fact per env)
  and resets backend-due rows; task `on_episode_end` still runs first and the
  returned observation is post-reset.
- contact groups use the breaking typed-ref schema (`sensor_refs` /
  `filter_refs`), not the removed `sensor_bodies` / `filter_bodies`.
- `termination_mode="external"`, `auto_reset: false` and
  `reset_on_termination: false` remain mandatory.

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
- manual `env.reset()` / `env.step(actions)` rollouts asserting that task-owned
  done rows end through `runner.terminate(..., reset=True)` and are then
  re-opened, `on_episode_end` before the end boundary, pre-reset
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
`runner={auto_reset: false, reset_on_termination: false}`, `EnvSpec` exposes no
way to override either, and it declares no episode duration (core has no
episode clock).

```python
from unigym.robots.base import RobotProfile
from unigym.runtime.actions import OrderedActionSpec, PolicyBinding
from unigym.world import World
from unigym_rl import EnvSpec

spec = EnvSpec(
    world=World.flat(),                      # environment only: no robot, no world_base
    robot=RobotProfile(...),                 # complete profile (joints/assets/actuation)
    spawn={"pos": (0.0, 0.0, 0.4)},          # run-owned robot root spawn
    action_spec=OrderedActionSpec(...),      # explicit; never inferred from body
    policies=(PolicyBinding(...),),          # explicit adapter ownership
    observations={"robot": ["go2"]},         # config-declared typed sources
    num_envs=4,
    inspect=True,
)

task = spec.make_task(MyTask)       # fresh run + task; caller closes both
# gym = spec.make_gym(MyTask)       # requires the gymnasium extra (lazy import)
```

`make_task` validates that its argument is a `TaskEnv` subclass.
`make_gym` imports `GymTaskEnv` lazily, so importing `unigym_rl` or using
`EnvSpec` requires an explicit `World`, `RobotProfile`, `OrderedActionSpec`,
`PolicyBinding` tuple and `observations` declaration; it never constructs a
World, infers an action layout or guesses observation sources.
`EnvSpec` never imports `gymnasium`. `make_run` returns a fresh `Run` on every
call and never builds the backend.

### Typed observation state

`TaskState.obs` is the core immutable `ObsSnapshot`; `TaskState.robot(name)` is
a convenience for `state.obs.robot(name)`. Task code reads typed views and
never parses artifact strings:

```python
robot = state.obs.robot("go2")
obs = torch.cat((robot.base_lin_vel_b, robot.base_ang_vel_b, robot.gravity_b), dim=1)
feet = state.obs.contact("feet").any()
command = state.obs.command.value
prev_action = state.obs.prev_action.value
```

The observation sources are declared once by `EnvSpec(observations=...)` (or
the core run config) and the core resolves the names. `TaskState` also carries
`episode_id`, `time_in_episode`, `reset_env_ids` and `should_reset`; there is
no raw observation cache or `extra` record.

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
    action_spec=action_spec,
    policies=policies,
    observations={"robot": [robot.name]},
    num_envs=opts.env.num_envs,
    device=opts.env.device,
    backend=opts.env.backend,
    seed=opts.env.seed,
)
task = spec.make_task(MyTask, max_steps=opts.env.max_episode_steps)
# a future trainer would read opts.train.max_iterations / rollout_steps
```

```bash
python train.py --num-envs 8 --max-episode-steps 500 --device cpu \
  --backend mujoco --seed 0 --max-iterations 1000 --rollout-steps 64
```

Hyphenated flags are canonical; underscore aliases (`--num_envs`,
`--max_episode_steps`, `--max_iterations`, `--rollout_steps`) behave
identically. Defaults: `num_envs=8`, `max_episode_steps=500`, `device="cpu"`,
`backend="mujoco"`, `seed=0`, `max_iterations=1000`, `rollout_steps=64`.
Positive-int fields reject values below 1 both on the CLI and in direct
dataclass construction; empty `device` and `backend` strings are rejected.

`--backend` selects the simulation backend (`mujoco`, `isaaclab2`,
`isaaclab3`); the core layer validates the backend name, while engine
import/availability is checked later at backend build time. IsaacLab2 still
requires its Kit/AppLauncher bootstrap before importing the backend. For
example:

```bash
--backend mujoco --device cpu
--backend isaaclab2 --device cuda:0
```

Parameter ownership: `num_envs` / `device` / `backend` / `seed` go to
`EnvSpec`; `max_episode_steps` is the task truncation budget; `max_iterations` /
`rollout_steps` are reserved for a future trainer. The parser never constructs
an `EnvSpec`, `World`, robot, task, or trainer, and no CLI value is propagated
to `TaskEnv.reset`.

## Optional task managers

R3 adds two optional, pure-Python helpers: `RewardManager` and
`TerminationManager`. They are sugar only: users may import them, or ignore
them and implement `reward` / `terminated` / `truncated` directly. There is no
new env type, no manager config or registry, no automatic reset, and no
training loop. Managers keep no per-step state; every call is a pure function
of the `Transition` it receives.

```python
from unigym_rl import RewardManager, RewardTerm, TerminationManager, TerminationTerm


class MyTask(TaskEnv):
    def on_bind(self, context):
        self.rewards = RewardManager(
            (
                RewardTerm("upright", upright_bonus, weight=1.0),
                RewardTerm("effort", effort_penalty, weight=0.01),
            ),
            clip=(-10.0, 10.0),
        )
        self.terminations = TerminationManager(
            (
                TerminationTerm("fallen", fall_mask),
                TerminationTerm("timeout", timeout_mask, mode="truncated"),
            )
        )

    def reward(self, transition):
        return self.rewards(transition)

    def terminated(self, transition):
        return self.terminations(transition).terminated

    def truncated(self, transition):
        return self.terminations(transition).truncated

    def termination_reasons(self, transition, terminated, truncated):
        return self.terminations(transition).reasons
```

- Reward terms must return a 1-D floating tensor and termination terms a 1-D
  bool tensor. The managers do not know `num_envs`; `TaskEnv` still validates
  the final `(N,)` shape.
- `breakdown(transition)` returns the weighted tensor of each enabled reward
  term for logging; `clip` affects only the total.
- `TerminationResult.reasons` keeps every term's raw mask. The manager does
  not merge backend-owned due ids; `TaskEnv` still computes
  `truncated = (backend_due | task_truncated) & ~task_terminated`.
- Names, terms, weights, `clip` and modes are validated at construction;
  `unigym_rl.managers` imports only `torch` and `unigym_rl` itself.

## Examples

- `examples/cartpole/` — a runnable fixed-base MuJoCo CartPole demo: a
  programmatic `EnvSpec`, a local demo `RobotProfile`, one slider action column
  with a passive observed pole, optional randomized resets via
  `set_joint_state` + `settle`, and a manual skrl PPO loop. It is repo-only and
  is not installed into the wheel; see `examples/cartpole/README.md`.
- `examples/go2_velocity/` — a flat-ground Go2 velocity-tracking smoke: the
  installed `unigym-robots` pack profile, a 48-column observation, per-env
  commands, tracking/alive rewards, task-owned tilt/height/time termination,
  and an 8-env manual skrl PPO loop. It requires
  `unigym-robots download --set mjcf-v0` for MuJoCo (or
  `--set usd-go2-v0` for `--backend isaaclab2`); assets are never
  auto-downloaded. See `examples/go2_velocity/README.md`.

## Installing / testing

`unigym-rl` is developed and released from
`packages/unigym-rl` in the
[`unigym-packages`](https://github.com/unsigned123/unigym-packages)
repository. Core is not on PyPI yet, so install `unigym` from a checkout and
install this package against it from `packages/unigym-rl`:

```bash
git clone https://github.com/unsigned123/unigym ../unigym-core
pip install -e ../unigym-core
pip install -e '.[gymnasium,skrl]'
```

From the **root of the `unigym-packages` repository**, run the tests with the
core checkout on `PYTHONPATH` (adjust paths for your environment):

```bash
PYTHONPATH=../unigym-core:packages/unigym-rl/src \
  python -m pytest packages/unigym-rl/tests -q
```

Static checks:

```bash
python -m ruff check packages/unigym-rl/src packages/unigym-rl/tests packages/unigym-rl/examples
python -m ruff format --check packages/unigym-rl/src packages/unigym-rl/tests packages/unigym-rl/examples
```

Releases declare a core range in `pyproject.toml`
(`unigym>=0.110,<0.111`) and are tagged `unigym-rl-v<version>`; see
`COMPATIBILITY.md` at the repository root.

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
core ever fires. Core owns no episode clock: episode boundaries come only from
the task (or a backend-owned due mask). `TaskEnv` ends its task-owned done
rows through `runner.terminate(env_ids, facts, reset=True)` with one
`custom.*` core fact per env, then re-observes the reset rows. A row the
backend already reported through `StepResult.reset_env_ids` is only reset, so
it is not ended twice. Backend due ids still merge into `truncated` (a due env
that is also task-terminated stays terminated, not truncated).

## TaskEnv hooks

Subclass `TaskEnv` and implement `observe` and `reward`. Everything else has
a safe default.

```python
class MyTask(TaskEnv):
    def observe(self, state):            # required -> (N, obs_dim) tensor
        return state.obs.robot("go2").joint_pos

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
final observation are all computed before the episode boundary;
`on_episode_end` runs before the boundary; task-owned rows end via
`runner.terminate(..., reset=True)` and backend due rows are reset, so only
done ids are touched; the observation returned by `step` is post-reset; the
Run is never stopped or closed by `TaskEnv`.

## Layout

```text
src/unigym_rl/
  __init__.py       # TaskEnv / EnvSpec / RingBuffer exports + lazy GymTaskEnv
  context.py        # TaskContext / TaskState / Transition / EpisodeEnd
  spec.py           # programmatic external-termination Run builder (R2.5)
  buffers.py        # per-env RingBuffer history (R2.5)
  managers.py       # optional RewardManager / TerminationManager (R3)
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
  test_managers.py          # optional reward/termination managers (R3)
```
