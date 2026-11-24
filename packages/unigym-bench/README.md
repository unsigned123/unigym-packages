# unigym-bench

`unigym-bench` is an external benchmark pack for `unigym` that will build a
streaming, monitor-driven benchmarking layer on the public sink, signal and
evaluation contracts, without importing `unigym.benchmarks`,
`unigym.runner.evaluation` or `unigym.runner.episodes`.

Status: P2. The package contains the skeleton — version metadata, the
core-contract import fence, strict JSON helpers, the closed YAML task-spec
loader (`unigym_bench.specs`), and the monitor protocol with the v1 built-in
monitors (`unigym_bench.monitors`). There is no benchmark engine, sink, report,
scenario or sampler implementation yet.

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
