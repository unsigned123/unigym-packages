# unigym-bench

`unigym-bench` is an external benchmark pack for `unigym` that builds a
streaming, monitor-driven benchmarking layer on the public sink, evaluation,
value and step-context contracts, without importing `unigym.benchmarks`,
`unigym.runner.evaluation` or `unigym.runner.episodes`.

Status: P14, applied on the P15 value/context runtime baseline (core `0.107.0`;
no throughput claim).
P10 moves the import fence and version pin to core C13: the pack requires
`unigym>=0.107,<0.108`, `CORE_SURFACE_VERSION` is `3`, and the fence checks
`unigym.values` (`ValueStore`, `ValueView`, `ValuePublisher`, `StepValues`)
and `unigym.evaluation.EnvStepContext` instead of the removed
`unigym.signals`. This baseline includes the P15 runtime migration: P11
replaces the pack `TickFrame` with `BenchmarkStepContext` (a core
`EnvStepContext` subclass), P12 adds the source resolver and condition DSL, and
P13 adds `PredicateMonitor`. P15 moves the runtime onto the core value
transport:

- `make_sink` reads `ctx.values` as a `ValuePublisher`, and `sink-steered`
  fails fast when it is absent;
- the engine consumes `bundle["step"]` for ticks and `bundle["final_step"]` for
  episode-end evaluation, publishes monitor values as `<monitor>/<local>` and
  never uses a raw signal publisher;
- `PredicateMonitor.on_episode_end` evaluates final-context `episode_end`
  predicates; their events are recorded in the ending episode and their
  report/scoring terminals are counted in the aggregate/report without a
  runner drain or latch conflict;
- `ScenarioSampler.bind_values(ValueView)` replaces `bind_signals`;
- the pack rewind schema is `unigym_bench.rewind/4` and monitor snapshots are
  `unigym_bench.monitor/3`; older payloads are rejected explicitly.

P14 collapses the typed built-in monitors. `ThresholdMonitor`, `HoldMonitor`,
`SafetyMonitor` and `EventCounterMonitor` are now `PredicateMonitor`
configurations built by `threshold_config`, `hold_config`, `safety_config` and
`event_counter_config`; the `Probe` protocol and the old typed monitor
constructors/state classes are deleted. `PhaseMonitor` takes an explicit
`SourceRef` and resolves it through `SourceResolver`. Scenario coverage is a
stateful `ScenarioCoverageExtension` in `report.py`, fed by finalized
`EpisodeRecord.command_at_start` values instead of a tick monitor. P16
(TaskDocument composition) remains a later card. The package already contains
the P0-P15 tree with the P14 collapse: version metadata, the core-contract
import fence, strict JSON helpers, the closed YAML task-spec loader
(`unigym_bench.specs`), the monitor protocol with the configured predicate and
phase monitors (`unigym_bench.monitors`), the streaming engine
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
seam because monitor/extension instances are not JSON-safe. `make_sink(ctx)`
fails fast on unknown options, an unknown loop mode, a `sink-steered` run
without `ctx.values`, `runner.auto_reset: false`, or a declared
`monitor_extensions` reference that does not exist. `ctx.values` is the core
`ValuePublisher` from `SinkContext`; the pack never reads a signal surface.

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
observes a `BenchmarkStepContext` (a core `EnvStepContext` subclass), returns
per-env level values, drains one-shot `Event` occurrences and runner
`Termination` requests, and observes the due env ids plus the final step
context through `on_episode_end(env_ids, final_context)`. Episode-end
report/scoring terminals are drained through the separate optional
`drain_episode_terminals()` face, never through the runner drain. A monitor
clears per-env measurement state through `reset` and contributes a JSON-safe
`report_view`. Stateful monitors implement the optional
`RewindableMonitor.snapshot()/restore()` capability with JSON-safe private
copies; a snapshot carries a construction/config fingerprint and `restore`
rejects a monitor with a different configuration.

`BaseMonitor` is a convenience class with empty values and no-op lifecycle.
`PredicateMonitor` is the compositional monitor: one condition tree, one
`TriggerSpec` (`tick`, `edge`, `hold`, `once`, `episode_end`) and one
`EmitSpec` (local value name, optional event/severity and optional terminal).
Each condition reads explicit `SourceRef` values (`record:`, `obs:`, `value:`;
`query:` is rejected in v1) through `SourceResolver`; there is no `Probe`
callable and no fallback lookup.

The retired typed monitors are configuration helpers that return a validated
`PredicateMonitor`:

```python
from unigym_bench.monitors import (
    event_counter_config, hold_config, safety_config, threshold_config,
)

contact = threshold_config("contact", "value:force", threshold=5.0,
                           terminal_outcome="failure")
settled = hold_config("settled", "value:error", threshold=0.05, hold_steps=10)
danger = safety_config("danger", "value:tilt", threshold=0.6, severity="fatal")
bounces = event_counter_config("bounces", "value:contact")
```

`PhaseMonitor(name, source, boundaries=..., phase_names=...)` remains a
specialized monitor: it resolves one per-env floating vector from `source` on
each tick and reports phases, transitions and episode phases.

Scenario coverage is not a monitor. `ScenarioCoverageExtension` in
`unigym_bench.report` is a stateful report extension: `on_env_end` reads the
finalized `EpisodeRecord.command_at_start` of each due env and matches it
against a validated command pool with exact `float32` quantization (using no
tick frame, view or tensor). Its accumulator is snapshotted by the rewind
provider, and its `on_run_end` block is emitted under
`extensions/<name>`. Like the retired monitor it observes due episode
boundaries only; an episode finalized only at run end is not a coverage
observation.

### Report field mapping

The new predicate report is intentionally generic. Where a retired field has
no predicate equivalent, the table names the replacement and the reason.

| Old monitor field | New report field | Notes |
| --- | --- | --- |
| `kind: threshold/hold/safety/event_counter` | `kind: predicate` | The configuration is visible under `condition`, `trigger` and `emit`. |
| `threshold`, `operator`, `hold_steps` | `condition`, `trigger` | The closed condition mapping and trigger spec are the source of truth. |
| `trigger_events`, `holds`, `violation_events`, `occurrences` | `fired` | One count per fired environment row. |
| per-env trigger counters | `fired_counts` | Same per-env edge/hold counts. |
| `episodes` | `episodes` | Unchanged. |
| `terminal_outcome` | `terminal_outcome` | Unchanged; the terminal code keeps `custom.<monitor>.<reason>`. |
| `severity` | `severity` | Event severity for threshold/hold/safety; report metadata for the event counter. |
| `active_ticks`, `violation_ticks`, `max_streak` | (none) | A `PredicateMonitor` publishes the trigger mask, not a continuous level, so these watchers have no equivalent. |
| signal `level` | (none) | Only the configured emit value is published (`<monitor>/<value>`). |
| signal `active` / `violating` / `holding` / `count` | emit value | The published bit is the trigger firing mask, not the continuous condition or cumulative counter. |
| event data `value`, `threshold`, `operator`, `hold_steps` | generic trigger data | Events carry only `{"trigger": <kind>}` plus `steps` for a hold trigger. The old monitor-specific data fields are not preserved. |
| terminal metrics `value`, `threshold`, `hold_steps` | empty `metrics` | Predicate terminals have `metrics: {}`; outcome, code, source and first-terminal behavior are preserved. |
| scenario coverage fields | `extensions/<name>` | `ScenarioCoverageExtension` returns the original `pool_size`, `covered`, `coverage_ratio`, `episode_counts`, `unmatched_episodes` and `episodes` facts. |

Event grouping also differs: `PredicateMonitor` emits one event per firing tick
with the fired env ids grouped, while the retired typed monitors emitted one
event per env. Per-episode `event_counts` and the firing counts above remain
per env; `execution.event_total` counts grouped occurrences.

Event and terminal payload parity is therefore limited. Event names, codes,
severity and env ids are preserved, but event `data` is the generic trigger
description rather than the old measured value/threshold/operator payload.
Terminal outcome, code, source and per-env first-terminal behavior are
preserved, but terminal `metrics` are empty because a predicate does not retain
the measured value.

## Closed loop

The pack engine publishes every monitor's per-env level values under
`<monitor>/<local>` through the core `ValuePublisher` that `make_sink` receives
as `ctx.values`. A monitor named `scenario` therefore owns the closed-loop
convention:

- `scenario/desired_cmd`: a `float32` tensor of shape `(num_envs, dim)` on the
  run device; every value must be finite;
- `scenario/command_valid`: an optional `bool` tensor of shape `(num_envs,)`;
  `False` rows keep their previous command, `True` rows take the desired row.

`ScenarioSampler.bind_values` requires the full read-only `ValueView` (`get`
and `snapshot`) and stores it; `update` reads only those two names from the
run's core `ValueStore`. It applies a valid desired command to `self.cmd` for
the next env step and never advances the sampler cursor or RNG; without a
desired command the pool/cursor behavior is unchanged. The sampler never scans
the value namespace, never falls back to another monitor's values and never
duplicates monitor detection. Invalid value type, shape, dtype or device fails
fast.

Episode boundaries remain episode-granular: `on_episode_end` advances the
cursor and `reset(env_ids)` writes the pool command, so a reset can overwrite
a desired command for those envs until the next `update` re-applies the
still-published level value. A value never drives the episode clock.

The runner order is `sink.on_step(t)` (publish the tick context snapshot) then
`sampler.update(t+1)` (read the store) then `env.step(t+2)` (act), so in-episode
closed loop has two policy-tick latency. The applied `cmd` is part of the
sampler rewind state; the core `values` store is a separate core rewind
provider (`name="values"`), and the pack provider never duplicates either.
The tick context carries the pre-`on_step` value snapshot; the final context
carries the post-`on_step` snapshot used for episode-end predicates and their
final monitor publication.

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
