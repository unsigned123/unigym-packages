# unigym-bench

`unigym-bench` is an external benchmark pack for `unigym` that will build a
streaming, monitor-driven benchmarking layer on the public sink, signal and
evaluation contracts, without importing `unigym.benchmarks`,
`unigym.runner.evaluation` or `unigym.runner.episodes`.

Status: P8. The package contains version metadata, the core-contract import
fence, strict JSON helpers, the closed YAML task-spec loader
(`unigym_bench.specs`), the monitor protocol with the v1 built-in monitors
(`unigym_bench.monitors`), the streaming engine (`unigym_bench.engine`), the
core sink shell (`unigym_bench.sink`), the rewind provider, the report builder
with report extensions (`unigym_bench.report`), and the validated command
scenario with its rewind-capable sampler (`unigym_bench.scenario`,
`unigym_bench.sampler`). The sampler also consumes the pack's documented
closed-loop signals.

## Report

At run end the pack sink returns `{"BenchmarkSink": <report>}` where the report
is a JSON-safe document with the fixed blocks `schema`, `run`, `tasks`,
`episodes`, `terminations`, `execution`, `monitors` and `extensions`. Episode,
termination and per-env counters are maintained independently of episode
retention, so the aggregate report is correct even with the default
`retain_episodes: false`; per-record `length`/`by_env` detail is emitted only
when episodes are retained. `execution` records the event total, dropped events,
terminal conflicts and the retention/cap policy. Monitors without a declared
`report_extension` contribute their `report_view()` under `monitors`; a monitor
that declares an extension is emitted only through that extension's
`extensions/<name>` block. A stateful report extension declares
`stateful = True` and implements `snapshot()`/`restore()` so the
`unigym_bench` rewind provider restores its accumulator. The pack never writes
the core `benchmark_report.json` or `results.json`.

## Monitors

`unigym_bench.monitors.Monitor` is the stateful per-env contract. A monitor
observes a `TickFrame`, returns per-env level signals, drains one-shot `Event`
occurrences and core `Termination` requests, observes the due env ids through
`on_episode_end(env_ids)`, clears per-env measurement state through `reset`,
and contributes a JSON-safe `report_view`. Stateful built-ins implement the
optional `RewindableMonitor.snapshot()/restore()` capability with JSON-safe
private copies; a snapshot carries a construction/config fingerprint and
`restore` rejects a monitor with a different configuration.

`BaseMonitor` is a convenience class with empty values and no-op lifecycle.
The v1 built-ins are `ThresholdMonitor`, `HoldMonitor`,
`EventCounterMonitor`, `PhaseMonitor`, `SafetyMonitor` and
`ScenarioCoverageMonitor`. Value-driven built-ins take an explicit `Probe`
callable that returns one `(num_envs,)` tensor on the frame device; the
composition layer, not the monitor, decides where that value comes from.
Monitors never retain a frame, view or tensor.

## Closed loop

The pack engine publishes every monitor's per-env level signals under
`<monitor>/<signal>`. A monitor named `scenario` therefore publishes the
closed-loop convention:

- `scenario/desired_cmd`: a `float32` tensor of shape `(num_envs, dim)` on the
  run device; every value must be finite;
- `scenario/command_valid`: an optional `bool` tensor of shape `(num_envs,)`;
  `False` rows keep their previous command, `True` rows take the desired row.

`ScenarioSampler.bind_signals` requires the full read-only `SignalView`
(`get` and `snapshot`) and stores it; `update` reads only those two names. It
applies a valid desired command to `self.cmd` for the next env step and never
advances the sampler cursor or RNG; without a desired command the pool/cursor
behavior is unchanged. The sampler never scans the namespace, never falls back
to another monitor's signals and never duplicates monitor detection. Invalid
signal type, shape, dtype or device fails fast.

Episode boundaries remain episode-granular: `on_episode_end` advances the
cursor and `reset(env_ids)` writes the pool command, so a reset can overwrite
a desired command for those envs until the next `update` re-applies the
still-published level signal. A signal never drives the episode clock.

The runner order is `sink.on_step(t)` (publish) then `sampler.update(t+1)`
(read) then `env.step(t+2)` (act), so in-episode closed loop has two
policy-tick latency. The applied `cmd` is part of the sampler rewind state; the
signal store rides the core `SignalBus` rewind provider, and the pack does not
duplicate either.

## Task documents

`unigym_bench.specs.load_task_document` loads and validates one YAML task
document into a frozen `TaskDocument`. `TaskDocument.task` is the validated
`TaskSpec`; the optional `scenario` and the provenance fields live on the
document wrapper. A document declares a task name, a mapping of monitors, an
optional mapping of report extensions, an optional retention policy and an
optional scenario.

The schema is closed. Unknown keys are rejected, names match
`[a-z][a-z0-9_]*`, monitor/extension params must be JSON objects, module paths
use `pkg.mod:attr`, and duplicate YAML keys fail instead of being merged. An
explicit `null` for an object field (`schema`, `retention`, `scenario`,
`extensions`, `resources`) is a configuration error; omit the key to apply the
documented default. `schema` may be omitted for authoring; when present it must
be `unigym_extbench/1`. Relative resource paths resolve against the task
document directory, never the process working directory.

## Dependencies

Development requires Python >=3.11 with a compatible `unigym` (`>=0.106.0`)
installed, plus the `dev` extra. The YAML loader requires `PyYAML>=6.0,<7.0`,
which is a direct runtime dependency of the pack. Pytest adds `src` to its own
path, so tests do not need a caller-provided `PYTHONPATH`; the import-fence
subprocess also sets its own path from the imported package.

`unigym_bench.__version__` stays at `0.1.0` while the pack is a pre-release;
this revision does not bump it. `CORE_SURFACE_VERSION` is a pack-local revision
of the required `unigym` public surface (currently `2`, after adding the core
termination facts), not the distribution version.
