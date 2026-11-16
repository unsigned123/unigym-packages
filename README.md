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
subclassing. Optional future extras (`gymnasium`, `skrl`, `rsl-rl`) are
planned but intentionally **not** declared as dependencies yet.

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
  __init__.py
  context.py        # TaskContext / TaskState / Transition / EpisodeEnd
  task.py           # TaskEnv + reset/step driver
  errors.py
  testing/
    __init__.py
    fakes.py        # fake run/runner/env (tests only)
tests/
  test_task_env.py
  test_task_env_fake.py
```
