# unigym-bench

`unigym-bench` is an external benchmark pack for `unigym` that builds a
streaming, monitor-driven benchmarking layer on the public sink, evaluation,
value and step-context contracts, without importing `unigym.benchmarks`,
`unigym.runner.evaluation` or `unigym.runner.episodes`.

Status: P10 (core `0.107.0` value/context baseline; no throughput claim).
P10 moves the import fence and version pin to core C13: the pack requires
`unigym>=0.107,<0.108`, `CORE_SURFACE_VERSION` is `3`, and the fence checks
`unigym.values` (`ValueStore`, `ValueView`, `ValuePublisher`, `StepValues`)
and `unigym.evaluation.EnvStepContext` instead of the removed
`unigym.signals`. Runtime migration is in progress: P11 replaces the pack
`TickFrame` with `BenchmarkStepContext` (a core `EnvStepContext` subclass), and
the streaming engine now reads only `bundle["step"]` while monitors take the
core context. The sink and sampler still use the P9 signal vocabulary, and
value publication plus final-context recording are migrated in P15; the sampler
follows in P12. The package
already contains the P0-P9 tree: version metadata, the core-contract import
fence, strict JSON helpers, the closed YAML task-spec loader
(`unigym_bench.specs`), the monitor protocol with the v1 built-in monitors
(`unigym_bench.monitors`), the streaming engine
(`unigym_bench.engine`), the core sink shell (`unigym_bench.sink`), the rewind
provider, the report builder with report extensions (`unigym_bench.report`),
and the validated command scenario with its rewind-capable sampler
(`unigym_bench.scenario`, `unigym_bench.sampler`). The P9 tree includes a real
tiny-arm MuJoCo integration test through `Run.runner` (the core SyncRunner
tick engine), zero-episode and multi-env conformance coverage, a
bounded-journal retention test, and the parity limitations documented below.

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

## Authoring a run

> **P9-era construction path on core C13.** The example and `make_sink` details
> below describe the P9 pack as it ran against core `0.106.x`. Core C13
> (`0.107.0`) exposes `SinkContext.values` backed by `ValueStore`/`ValueView`
> and has removed the signal surface. P11 migrates only the engine's step
> context and monitor signatures; the sink and sampler still use the P9 signal
> vocabulary, so `make_sink` and the P9 pack suite are expected to be red or
> unusable until P15 (sink value publication) and P12 (sampler) migrate them.

The pack is mounted as a core custom sink. A real benchmark run therefore keeps
the run config in charge of the body/backend and gives the pack only its
construction context:

```python
from unigym.run.api import Run
from unigym.run.sink import SinkSpec

run = Run(
    # ... world, robot, action_spec, policies, steps, episode_length_s ...
    mode="benchmark",
    benchmark={"tasks": [], "min_episodes": 1, "max_steps": 2000},
    # A custom sink owns benchmark control flow, so no task pack runtime is
    # needed and the built-in task catalog is not loaded.
    pack_runtime=None,
    discover_runtime=False,
    sink=SinkSpec(
        factory="unigym_bench.sink:make_sink",
        options={
            "monitor_factory": "my_pkg.monitors:make_monitors",
            "extension_factory": "my_pkg.extensions:make_extensions",  # optional
            "monitor_extensions": {"coverage": "coverage_report"},       # optional
            "loop_mode": "pool",                                         # or "sink-steered"
            "write_report": True,
        },
    ),
)
instance = run.build()
runner = run.runner("external", action_fn=policy_actions)
summary = runner.run(stop_when=instance.stop_when)
```

`monitor_factory` and `extension_factory` are zero-argument callables, given as
`module:callable` paths, that return the instances in task declaration order;
they must be importable from the run process. A factory is the only injection
seam because monitor/extension instances are not JSON-safe. In the un-migrated
P9 code, `make_sink(ctx)` fails fast on unknown options, an unknown loop mode, a
`sink-steered` run without `ctx.signals`, `runner.auto_reset: false`, or a
declared `monitor_extensions` reference that does not exist. On core C13 the
equivalent construction context is `ctx.values` (`ValueStore`/`ValueView`); the
`ctx.signals` branch is P9-era and is removed by the P11/P15 sink and engine
migration.

With `write_report: true` the sink writes the assembled document to
`<out_dir>/external_benchmark_report.json` at run end. The filename and the
`external_benchmark_report/1` schema are pack-owned; the core
`benchmark_report.json`, `results.json` and `metrics.npz` are never touched.
Writing is opt-in because it is a filesystem side effect. The sink creates
`out_dir` if needed and replaces the file atomically; a failed write raises and
is retried by a later `on_run_end()` call rather than being cached as success.
The same report is also returned through the sink's summary
(`{"BenchmarkSink": <report>}`); when
the core composes sinks behind its `FanOutSink`, the summary is nested under the
fan-out's keys, so reading the written file is the stable integration point.

## Retention and event cap

Retention is explicit and independent per mechanism. The YAML block below is the
`retention:` field of a task document (`unigym_bench.specs`), not a
`make_sink` option:

```yaml
retention:
  retain_episodes: false   # counters only; no per-record history
  max_events: 200000       # bounded in-memory event journal
  event_log: true          # false stores no journal entries at all
```

A programmatic caller passes the same values as a
`unigym_bench.specs.RetentionSpec` to `StreamingEngine`/`BenchmarkSink`.
`make_sink` has no retention option in this revision.

The engine counts every occurrence and every dropped occurrence. When the
journal reaches `max_events`, later events are not stored but still increment
`execution.event_total` and `execution.dropped_events` in the report. Episode,
termination and per-env counters are retention-independent, so the aggregate
report stays correct with `retain_episodes: false`; per-record `length` and
`by_env` detail exists only when records are retained. Events are kept in
memory only in this revision: no event log file is written.

## Monitors

`unigym_bench.monitors.Monitor` is the stateful per-env contract. A monitor
observes a `BenchmarkStepContext` (a core `EnvStepContext` subclass), returns per-env level signals, drains one-shot `Event`
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

> **P9-era closed loop, not P10 behavior.** The `SignalView`, `SignalBus`,
> `ctx.signals` and `bind_signals` statements below describe the P9 pack against
> core `0.106.x`. Core C13 (`0.107.0`) removed `unigym.signals` and exposes
> `SinkContext.values` plus `ValueStore`/`ValueView`/`ValuePublisher`/
> `StepValues`. The un-migrated sampler and sink value publication are expected
> to be red or unusable until P12 (sampler) and P15 (sink, final context); the
> command names and latency described here are P9-era, not a current claim.

In the P9 code that still ships, the pack engine published every monitor's
per-env level signals under `<monitor>/<signal>`. A monitor named `scenario`
therefore published the closed-loop convention:

- `scenario/desired_cmd`: a `float32` tensor of shape `(num_envs, dim)` on the
  run device; every value must be finite;
- `scenario/command_valid`: an optional `bool` tensor of shape `(num_envs,)`;
  `False` rows keep their previous command, `True` rows take the desired row.

`ScenarioSampler.bind_signals` (the P9 name, not a P10 API) requires the full
read-only `SignalView` (`get` and `snapshot`) and stores it; `update` reads only
those two names. It applies a valid desired command to `self.cmd` for the next
env step and never advances the sampler cursor or RNG; without a desired command
the pool/cursor behavior is unchanged. The sampler never scans the namespace,
never falls back to another monitor's signals and never duplicates monitor
detection. Invalid signal type, shape, dtype or device fails fast.

Episode boundaries remain episode-granular: `on_episode_end` advances the
cursor and `reset(env_ids)` writes the pool command, so a reset can overwrite
a desired command for those envs until the next `update` re-applies the
still-published level signal. A signal never drives the episode clock.

In P9 the runner order was `sink.on_step(t)` (publish) then
`sampler.update(t+1)` (read) then `env.step(t+2)` (act), so in-episode closed
loop had two policy-tick latency. The applied `cmd` is part of the sampler
rewind state; the signal store rode the core `SignalBus` rewind provider, and
the pack did not duplicate either. On core C13 the store is `ValueStore`/`ValueView`
and the removed `SignalBus` provider is not available until the P11/P12/P15
migration lands.

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

## Limitations and parity

Known limitations in this revision:

- **No YAML `TaskDocument` composition yet.** `load_task_document` validates a
  task document, but `make_sink` accepts factory paths, not a task document.
  Turning a document's monitor/extension specs into live instances is a later
  composition layer.
- **Events are in-memory only.** The capped journal is not persisted; there is
  no `external_benchmark_events.jsonl` writer in this revision.
- **No early-stop policy.** `BenchmarkSink.stop_when()` always returns false;
  the caller controls `runner.run(stop_when=...)`. `min_episodes`/`max_steps`
  remain core benchmark keys the pack does not consume.
- **MuJoCo integration lane.** The real tiny-arm test needs the D0 MuJoCo lane.
  A missing backend is a test failure, not a skip.

Parity against the core built-in benchmark is **documented, not claimed**:

| Aspect | Core built-in benchmark | `unigym-bench` pack | P9 status |
| --- | --- | --- | --- |
| Entry | `mode=benchmark` + `EvaluationSession` over pack-runtime tasks | `mode=benchmark` + custom sink (`make_sink`) with monitors | Different composition; both use the same core runner tick engine |
| Report artifact | `benchmark_report.json` / `results.json` / `metrics.npz` | `external_benchmark_report.json` (opt-in) | Disjoint, pack-owned filename |
| Schema | `benchmark_report` | `external_benchmark_report/1` | Not byte-compatible; mapping documented here |
| Episode end | core term group, episode horizon, benchmark limits | core facts plus pack first-terminal; verdicts `terminal`, `deadline`, `core`, `incomplete`, `rewound` | Same core boundary facts; pack adds custom terminals |
| Scoring | per-task `success` and metrics | monitor `report_view` and extension blocks; no aggregate success score | Different purpose; not interchangeable |
| Early stop | `min_episodes` / `max_steps` | caller-provided `stop_when` | Pack has no built-in stop policy |
| Retention | episode buffer retains for scoring | counters by default; capped event journal | Pack is lossier by default and reports its drops |
| Distribution | needs a benchmark task provider/pack runtime | installed sink; no pack runtime in `benchmark` mode | Pack avoids task-provider activation |
| Rewind | built-in session/buffer | pack engine provider `unigym_bench` | Same core rewind seam |

P9 includes a real same-config parity test. Constructing the core built-in
benchmark from a bare `Run` is not a normal external-pack path: the benchmark
task catalog must be imported inside the core `ACTIVATE` registration window,
while core imports it lazily during `BACKEND_BUILD`. The test therefore runs
the built-in lane in a fresh subprocess with that window opened explicitly,
then runs the pack sink over the same tiny-arm config. Both runs use two policy
ticks and one 0.04 s episode horizon, and the test asserts equal:

- episode count (1);
- total episode ticks (2);
- termination `outcome` / `code` / `source` counters (`deadline` /
  `runtime.episode_horizon` / `runtime`).

That is **fact parity**, not report or scoring parity. The built-in task
`success`/metric block has no pack equivalent, the two report schemas are not
byte-compatible, and the pack runtime never imports `unigym.benchmarks`; the
parity probe is test-only. The in-process integration test additionally checks
the same-run core `EpisodeTermination` facts against the pack report and the
report vocabulary against the core `EpisodeOutcome` / `TerminationSource`
enums. No throughput claim is made; the retention test proves the journal is
bounded by `max_events`.

## Dependencies

Development requires Python >=3.11 with a compatible `unigym` (`>=0.107.0`)
installed, plus the `dev` extra. The YAML loader requires `PyYAML>=6.0,<7.0`,
which is a direct runtime dependency of the pack. Pytest adds `src` to its own
path, so tests do not need a caller-provided `PYTHONPATH`; the import-fence
subprocess also sets its own path from the imported package.

`unigym_bench.__version__` is `0.2.0` for the core `0.107.0` baseline release.
`CORE_SURFACE_VERSION` is a pack-local revision
of the required `unigym` public surface (currently `3`, after adopting the
core value/context seams), not the distribution version.
