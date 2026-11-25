# unigym-bench

`unigym-bench` is an external benchmark pack for `unigym` that builds a
streaming, monitor-driven benchmarking layer on the public sink, evaluation,
value and step-context contracts, without importing `unigym.benchmarks`,
`unigym.runner.evaluation` or `unigym.runner.episodes`.

Status: `0.3.0`, the post-deletion-core release (core `0.108.x`; no throughput
claim). The pack now requires `unigym>=0.108,<0.109` and
`CORE_SURFACE_VERSION` is `4`. The post-deletion core owns no run `mode`, no
episode clock and no built-in benchmark or `EvaluationSession`; the pack
therefore ends every episode through an explicit `runner.terminate(...)`
boundary (or a pack sink terminal) and reopens envs through the core
`ResetAwareSink.on_reset` capability. The fence checks the explicit sink
selection (`CollectSinkSpec`/`SinkDeclaration`/`SinkSpec`), `ResetAwareSink`,
`unigym.values` (`ValueStore`, `ValueView`, `ValuePublisher`, `StepValues`) and
`unigym.evaluation.EnvStepContext`. This baseline includes the P15 runtime
migration: P11
replaced the pack `TickFrame` with `BenchmarkStepContext` (a core
`EnvStepContext` subclass), P12 added the source resolver and condition DSL,
and P13 added `PredicateMonitor`. P15 moved the runtime onto the core value
transport:

- `make_sink` reads `ctx.values` as a `ValuePublisher`, and `sink-steered`
  fails fast when it is absent;
- the engine consumes `bundle["step"]` for ticks and `bundle["final_step"]` for
  episode-end evaluation, publishes monitor values as `<monitor>/<local>` and
  never uses a raw signal publisher;
- `PredicateMonitor.on_episode_end` evaluates final-context `episode_end`
  predicates and refreshes their observed values only; monitors produce no
  events and no terminals;
- `ScenarioSampler.bind_values(ValueView)` replaces `bind_signals`;
- the pack rewind schema is `unigym_bench.rewind/5` and monitor snapshots are
  `unigym_bench.monitor/4`; older payloads are rejected explicitly. F4 added
  finalized episode evidence to the `/5` engine payload, so the `/5` shape
  changed in F4: pre-F4 `/5` payloads are rejected. F5 finalizes the schema with
  a required run-scoped `engine.stop_requested` flag and an optional `flow`
  block for stateful flows; pre-F5 `/5` payloads lack the flag and are
  rejected.

P14 collapses the typed built-in monitors. `ThresholdMonitor`, `HoldMonitor`,
`SafetyMonitor` and `EventCounterMonitor` are now `PredicateMonitor`
configurations built by `threshold_config`, `hold_config`, `safety_config` and
`event_counter_config`; the `Probe` protocol and the old typed monitor
constructors/state classes are deleted. `PhaseMonitor` takes an explicit
`SourceRef` and resolves it through `SourceResolver`. Scenario coverage is a
stateful `ScenarioCoverageExtension` in `report.py`, fed by finalized
`EpisodeRecord.command_at_start` values instead of a tick monitor.

P16 adds the task-document composition layer in `unigym_bench.compose`. The
schema is now `unigym_extbench/3`; the retired `/1` and `/2` forms are rejected
with a migration error and no dual-schema shim. A monitor declares either
`builtin: predicate` with `when`/`trigger`/`emit` (emit is a value name only),
`builtin: phase` with `source` and boundary params, or a `module`/API
`factory`. Predicate fields are closed and rejected on declarations that do not
consume them. The optional `flow` section owns explicit
`termination_rules`/`events`; monitors publish values only. Source references use the P12
`record:`/`obs:`/`value:`/`query:` grammar, and a `query:` source fails at
composition with `SourceUnavailableError`. Trigger and emit specs parse through
the P13 `TriggerSpec`/`EmitSpec`, so `level` and reserved temporal operators are
rejected. Extensions close over `scenario_coverage`, which reads the document
scenario command pool. A declared scenario auto-composes a coverage extension
named `scenario_coverage` unless the document already declares one; a name
collision is a teaching error. `compose_task(document)` returns a frozen
`TaskPlan(monitors, extensions, retention, scenario, monitor_extensions, flow)`, and
`make_sink` accepts a JSON-safe `task_document` path option that runs this
composition instead of the factory path. The package contains the P0-P16
implementation tree, first finalized as the `0.2.0` release and carried forward
as `0.3.0` on the post-deletion core: version metadata, the
core-contract import fence, strict JSON helpers, the closed YAML task-spec
loader (`unigym_bench.specs`), the composition layer (`unigym_bench.compose`),
the monitor protocol with the configured predicate and phase monitors
(`unigym_bench.monitors`), the streaming engine (`unigym_bench.engine`), the
core sink shell (`unigym_bench.sink`), the rewind provider, the report builder
with report extensions (`unigym_bench.report`), and the validated command
scenario with its rewind-capable sampler (`unigym_bench.scenario`,
`unigym_bench.sampler`). The pack includes a real tiny-arm MuJoCo integration
test through `Run.runner` (the core SyncRunner tick engine), zero-episode and
multi-env conformance coverage, a bounded-journal retention test, and the
parity limitations documented below.

## Release 0.3.0: post-deletion core lockstep

`unigym-bench 0.3.0` is a breaking release with no compatibility shims. It
requires core `unigym>=0.108,<0.109` and pack-local core surface revision `4`.

Changed contracts:

- The core has no run `mode`, no `episode_length_s`, no built-in benchmark and
  no `EvaluationSession`. The pack removes `ctx.mode` and `RunMetadata.mode`;
  no run mode is fabricated.
- The report schema is `external_benchmark_report/3`. The obsolete `run.mode`
  field is removed; `run.loop_mode` remains the pack's own closed-loop
  selection. The retired `/1` and `/2` report schemas are rejected; there is no
  compatibility shim.
- Episodes end only through an explicit boundary: `runner.terminate(env_ids,
  terminations, reset=True/False)` or a pack sink terminal drained by the
  runner. There is no hidden episode clock; a run that never ends an episode
  reports it honestly as `incomplete` at run end.
- `BenchmarkSink` implements the core `ResetAwareSink.on_reset` capability.
  A reset of a finalized env reopens it; a core reset of an already-open env
  restarts it in place (per-episode ticks/events/command/reset facts cleared,
  `monitor.reset` notified) without fabricating an episode record; a pre-start
  reset-all is a no-op because the engine binds on the first tick. `on_step_end`
  no longer infers a reset. The `reset_after_episode` constructor argument and
  the `runner.auto_reset: false` rejection are removed because the sink now
  observes resets directly. With `runner.auto_reset: false`, episodes must
  still end through an explicit boundary (`terminate(..., reset=True)`, or
  `reset=False` followed by `runner.reset`); a core-owned due left open is
  outside the pack contract.
- Explicit `runner.terminate` facts are custom-owned facts (`custom.*` code,
  `TerminationSource.CUSTOM`) and now appear in the pack's `terminations`
  counters. A core-owned namespace such as `runtime.episode_horizon` cannot be
  produced by the pack.
- The compat fence requires `ResetAwareSink`, `CollectSinkSpec`,
  `SinkDeclaration`, `SinkSpec`, `SinkContext`, `EnvView`/`EnvStepContext` and
  the four value surfaces; it never requires the removed benchmark scaffolding.

## Release 0.2.0: breaking API (historical)

The `0.2.0` notes are retained for history; `0.3.0` above supersedes the core
range, report schema and lifecycle contracts.

`unigym-bench 0.2.0` is a breaking release with no compatibility shims. It
requires core `unigym>=0.107,<0.108` (C13) and pack-local core surface revision
`3`. The release keeps the report schema `external_benchmark_report/2`; the
public pack seams below are new, removed or changed.

Removed from the pack:

- `TickFrame`: monitors now receive `BenchmarkStepContext`, a subclass of the
  core `unigym.evaluation.EnvStepContext`.
- The `Probe` protocol, the old typed monitors (`ThresholdMonitor`,
  `HoldMonitor`, `SafetyMonitor`, `EventCounterMonitor`) and their state
  classes. Use `PredicateMonitor` with `threshold_config`, `hold_config`,
  `safety_config` or `event_counter_config` instead; the retired names are not
  aliases.
- All pack use of the removed core signal surface (`SignalBus`, `SignalView`,
  `SignalPublisher`, `signal_bus_contract`) and `bind_signals`. Values flow
  through core `ValueStore`/`ValueView`/`ValuePublisher`. The sampler binds with
  `ScenarioSampler.bind_values(ValueView)`, and the engine publishes
  `<monitor>/<local>` through `ctx.values`.

Changed contracts:

- Task documents use `unigym_extbench/3`; the retired `/1` and `/2` forms are
  rejected with a migration error and there is no dual-schema shim. Monitors
  publish values only; the `flow` section owns `termination_rules` and `events`.
  `make_sink` accepts a `task_document` path and composes it with
  `compose_task`.
- Source references use the `record:`/`obs:`/`value:`/`query:` grammar. A
  `query:` source fails at composition with `SourceUnavailableError`.
- Episode-end evaluation uses the final step context (`bundle["final_step"]`
  and `on_episode_end(env_ids, final_context)`) and refreshes observed values
  only; it produces no event or terminal.
- The pack rewind schema is `unigym_bench.rewind/5` and monitor snapshots are
  `unigym_bench.monitor/4`; older payloads are rejected explicitly.
- The report schema is `external_benchmark_report/3`. `PredicateMonitor`
  blocks are predicate-shaped (`kind: predicate`, `condition`, `trigger`,
  `emit`, `fired`, `fired_counts`); `PhaseMonitor` blocks keep `kind: phase`,
  and `module`/API factory monitors contribute their own `report_view`. For
  predicate monitors, event `data` is the generic trigger description and
  predicate terminals carry empty `metrics`. The full field mapping is in
  [Report field mapping](#report-field-mapping).

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

## Benchmark flow

`BenchmarkSink(flow=...)` composes a `unigym_bench.flow.BenchmarkFlow` beside
the engine. Monitors are observation-only; the flow is the only event/terminal
decision owner for ticks. The stable seam is:

- `BenchmarkFlow`: `name`, `stateful`, optional `on_tick(tick)`, required
  `on_episode_end(batch)` and `on_run_end(run)`;
- `BenchmarkTick`: the tick's step context and per-monitor observed values;
- `FlowDecision`: flow-authored events, per-env terminations and a run-stop
  flag;
- `EpisodeBatch`: the due ids, final step context and final observed values;
- `FlowRun`: metadata, run summary, retained records, monitor views and the
  report-extension set;
- `DefaultBenchmarkFlow`: the stateless default that applies explicit
  `TerminationRule`/`EventRule` declarations and assembles the report.

The engine remains the validation, latch, journal, record and rewind owner.
The sink drives `begin_tick -> flow.on_tick -> apply_decision` and
`begin_episode_end -> flow.on_episode_end -> commit_episode_end`; the engine
validates every decision before it mutates run state. A flow without `on_tick`
is valid and applies no tick policy. A `stateful=True` flow must implement
callable `snapshot()`/`restore()`; its policy state joins the single
`unigym_bench` rewind payload under an optional `flow` block. A stateless flow
must not implement those methods and adds no checkpoint key.

`DefaultBenchmarkFlow` semantics are intentionally small:

- a `TerminationRule(monitor, value, outcome, code)` reads a per-env boolean
  value; rules are evaluated in declaration order and the first true rule per
  env wins;
- duplicate `(monitor, value)` rules are rejected; there is no hidden
  "monitor active => terminal" fallback and no flow-level trigger (the monitor
  trigger is the only cadence owner);
- an `EventRule(monitor, value, name, ...)` consumes the same boolean value;
  event code defaults to `task.<monitor>.<name>`;
- empty rules mean no automatic termination and no automatic events.

The stop flag is run-scoped. A flow requests it through
`FlowDecision.stop`; the engine keeps it until run end and snapshots it in the
required `engine.stop_requested` field. Rewind restores exactly the
checkpoint's flag, so a forked branch never inherits a stop from the abandoned
timeline. Episode evidence is engine-persisted and included in the engine
rewind payload; flow state is stored only in its own `flow` block and does not
duplicate records or evidence. Extension persistence and rewind stay
engine-owned while the flow owns extension invocation.

Flow callback exceptions are fail-fast with no rollback. `begin_tick` advances
the episode tick counters and reads monitor values before `flow.on_tick`, and
`begin_episode_end` refreshes final monitor values before
`flow.on_episode_end`. If a flow callback raises, the sink discards the pending
tick/boundary and closes its query snapshot, but it does not restore tick
counters or already-observed values. The exception propagates and the run must
abort; it must not silently continue on partially observed state.

### F3/F4 report and episode-end semantics

The report schema is `external_benchmark_report/3`. Because monitors no longer
produce terminal/event metadata, monitor report blocks do not carry
`terminal_reason`, `terminal_outcome` or event severity/content. Episode
evidence, when the flow returns any, lives only under `episodes.evidence` keyed
by `"<env>/<seq>"`; the key is absent for a run with no evidence. The fixed
top-level block set is unchanged and a flow cannot add a block. The sink rejects
the retired `external_benchmark_report/2` id and unknown top-level fields.

The flow owns report selection. `DefaultBenchmarkFlow(report_monitor_views=...)`
retains only the named monitor views and
`DefaultBenchmarkFlow(report_extensions=...)` invokes only the named extensions;
an omitted monitor view is absent from `monitors` but stays listed in
`tasks.monitors`, and an omitted extension produces no `extensions` block.
`None` keeps the previous default of every declared view/extension.

Episode-end monitor terminals are gone, not silently reinterpreted. A monitor's
`episode_end` trigger now only refreshes its observed value for due rows; it
never schedules a report/scoring terminal. Any termination policy must be an
explicit flow `TerminationRule` evaluated on a tick value. A document without
`flow` therefore has no automatic termination and no automatic events. The
engine no longer invokes `extension.on_env_end`; the flow does, exactly once per
due boundary and in declaration order.

## Authoring a run

The pack is mounted as a core custom sink. A real benchmark run therefore keeps
the run config in charge of the body/backend and gives the pack only its
construction context:

```python
from unigym.core.episode_termination import EpisodeOutcome, Termination, TerminationSource
from unigym.run.api import Run
from unigym.run.sink import SinkSpec

run = Run(
    # ... world, robot, action_spec, policies, steps ...
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
runner = run.runner("external")
for t in range(2000):
    runner.step(policy_actions(t))
    if episode_done(t):  # caller-owned episode semantics; no core clock
        runner.terminate(
            [0],
            {
                0: Termination(
                    outcome=EpisodeOutcome.DEADLINE,
                    code="custom.episode_horizon",
                    source=TerminationSource.CUSTOM,
                )
            },
            reset=True,  # reset=False leaves the env closed until runner.reset()
        )
summary = runner.sink.on_run_end()
```

`monitor_factory` and `extension_factory` are zero-argument callables, given as
`module:callable` paths, that return the instances in task declaration order;
they must be importable from the run process. A factory is the injection seam
for programmatic callers because monitor/extension instances are not JSON-safe.
`make_sink(ctx)` fails fast on unknown options, an unknown loop mode, a
`sink-steered` run without `ctx.values`, or a declared `monitor_extensions`
reference that does not exist. `ctx.values` is the core `ValuePublisher` from
`SinkContext`; the pack never reads a signal surface.

A YAML task document is the other, JSON-safe composition path:

```python
options={
    "task_document": "tasks/reach.yaml",
    "loop_mode": "pool",
    "write_report": True,
    # "monitor_extensions": {"contact": "coverage"},  # optional; must match
}
```

`task_document` is a non-empty path string resolved by
`unigym_bench.specs.load_task_document`; relative paths resolve against the run
process working directory. `make_sink` then calls `compose_task(document)` and
uses the plan's monitors, extensions, retention and scenario coverage. It
rejects `task_document` together with `monitor_factory` or
`extension_factory`, and rejects a `monitor_extensions` option that differs
from the document's declared `report_extension` map. Missing files, duplicate
names, invalid schema and composition failures surface as `ConfigError` at
this boundary.

With `write_report: true` the sink writes the assembled document to
`<out_dir>/external_benchmark_report.json` at run end. The filename and the
`external_benchmark_report/3` schema are pack-owned; the core
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

`unigym_bench.monitors.Monitor` is the stateful per-env observation contract.
A monitor observes a `BenchmarkStepContext` (a core `EnvStepContext`
subclass), returns per-env level values through `values()`, and observes the
due env ids plus the final step context through
`on_episode_end(env_ids, final_context)`. It produces no events and no
terminals; `DefaultBenchmarkFlow` applies explicit `TerminationRule` and
`EventRule` declarations to the observed values. A monitor clears per-env
measurement state through `reset` and contributes a JSON-safe `report_view`.
Stateful monitors implement the optional `RewindableMonitor.snapshot()/restore()`
capability with JSON-safe private copies; a snapshot carries a
construction/config fingerprint and `restore` rejects a monitor with a
different configuration.

`BaseMonitor` is a convenience class with empty values and no-op lifecycle.
`PredicateMonitor` is the compositional monitor: one condition tree, one
`TriggerSpec` (`tick`, `edge`, `hold`, `once`, `episode_end`) and one
`EmitSpec` (local value name only).
Each condition reads explicit `SourceRef` values (`record:`, `obs:`, `value:`;
`query:` is rejected in v1) through `SourceResolver`; there is no `Probe`
callable and no fallback lookup.

Every engine tick attaches one `BenchmarkStepContext.query`: a pack-private
`TickQuery` bound to that boundary's view and shared by all monitors on the
tick. Contact `scan`/`manifold` reads are memoized by normalized env selection
(and group), while `state` and `geometry` are read-only passthroughs to the core
`EvalQuery`. The snapshot is lazy, so a view with no runtime query only raises
the core `EvalQueryUnavailableError` when a consumer actually reads it. The
engine closes the snapshot when the monitor callbacks return; direct and
offline contexts leave `query` as `None`. A `state` or `geometry` subface
extracted before close stays governed by the core view lifetime; the snapshot
itself rejects new accessor reads after close.

The retired typed monitors are configuration helpers that return a validated
`PredicateMonitor`:

```python
from unigym_bench.monitors import (
    event_counter_config, hold_config, safety_config, threshold_config,
)

contact = threshold_config("contact", "value:force", threshold=5.0)
settled = hold_config("settled", "value:error", threshold=0.05, hold_steps=10)
danger = safety_config("danger", "value:tilt", threshold=0.6)
bounces = event_counter_config("bounces", "value:contact")

from unigym_bench.flow import DefaultBenchmarkFlow, TerminationRule

flow = DefaultBenchmarkFlow(
    [TerminationRule("contact", "threshold_reached", "failure", "custom.contact.threshold")]
)
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
| `terminal_outcome` | flow rule `outcome` | A monitor value never terminates; an explicit flow `TerminationRule` supplies the outcome and `custom.*` code. |
| `severity` | flow event rule | Event severity belongs to a flow `EventRule`, not a monitor. |
| `active_ticks`, `violation_ticks`, `max_streak` | (none) | A `PredicateMonitor` publishes the trigger mask, not a continuous level, so these watchers have no equivalent. |
| signal `level` | (none) | Only the configured emit value is published (`<monitor>/<value>`). |
| signal `active` / `violating` / `holding` / `count` | emit value | The published bit is the trigger firing mask, not the continuous condition or cumulative counter. |
| event data `value`, `threshold`, `operator`, `hold_steps` | flow event `data` | Events are declared by the flow; a rule may supply a static JSON-safe payload. |
| terminal metrics `value`, `threshold`, `hold_steps` | empty `metrics` | Flow terminals have `metrics: {}`; outcome, code, source and first-terminal behavior are preserved. |
| scenario coverage fields | `extensions/<name>` | `ScenarioCoverageExtension` returns the original `pool_size`, `covered`, `coverage_ratio`, `episode_counts`, `unmatched_episodes` and `episodes` facts. |

Event grouping is flow-owned: an `EventRule` emits one event per tick with the
true env ids grouped. Per-episode `event_counts` and the firing counts above
remain per env; `execution.event_total` counts grouped occurrences.

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
explicit `null` for a document object field (`schema`, `retention`, `scenario`,
`extensions`, `resources`) or a predicate field (`when`, `trigger`, `emit`,
`source`) is a configuration error; omit the key to apply the documented
default. `schema` may be omitted for authoring; omission means the current
`unigym_extbench/3`, and the retired `unigym_extbench/1`/`/2` forms are rejected
with a migration error. Relative resource paths resolve against the task
document directory, never the process working directory.

### Schema `/3` monitor and flow forms

```yaml
schema: unigym_extbench/3
name: reach
monitors:
  contact:
    builtin: predicate
    when:
      source: value:force
      op: gte
      value: 0.5
    trigger: {kind: hold, steps: 3}
    emit: {value: contact}
    report_extension: contact_report
  height_phase:
    builtin: phase
    source: value:height
    params:
      boundaries: [0.0, 1.0]
      phase_names: [low, mid, high]
flow:
  termination_rules:
    - monitor: contact
      value: contact
      outcome: failure
      code: custom.contact.threshold
  events:
    - monitor: contact
      value: contact
      name: contact
      severity: warning
extensions:
  contact_report:
    builtin: scenario_coverage
scenario:
  command_pool: [[0.0], [1.0]]
  plan: [0, 1]
retention:
  retain_episodes: true
  max_events: 10000
  event_log: true
```

The closed monitor builtins are `predicate` and `phase`; the closed extension
builtin is `scenario_coverage`. The retired typed monitor names are not aliases.
A `module` declaration imports its `pkg.mod:attr` callable only at composition
and invokes it with the declaration's JSON `params` as keyword arguments; an
API `factory` callable follows the same call convention. The returned object's
name must equal the declaration name. Builtin `predicate`/`phase` and
module/factory declarations do not consume `resources`; a non-empty
`resources` map is rejected at composition before any module import or factory
call. YAML cannot express a callable factory, and a `!!python/...` YAML tag is
rejected by the safe loader.

`compose_task(document)` is the composition entry point and returns a frozen
`TaskPlan`. It rejects an unknown builtin with the registry names, duplicate
live names, a missing `report_extension` reference and a `query:` source
(`SourceUnavailableError`). A declared scenario auto-composes a
`scenario_coverage` extension when the document does not already declare one;
declaring `builtin: scenario_coverage` under another name uses that name and
suppresses the automatic extension. A `scenario_coverage` extension without a
document scenario is a configuration error, as is an automatic-name collision.

## Limitations and parity

Known limitations in this revision:

- **Closed composition registry only.** The task-document composer supports
  only `predicate`/`phase` monitors and the `scenario_coverage` extension.
  Other builtins must be declared as `module`/API `factory` declarations.
- **Events are in-memory only.** The capped journal is not persisted; there is
  no `external_benchmark_events.jsonl` writer in this revision.
- **No built-in early-stop policy.** A flow may request a stop through
  `FlowDecision.stop`; `BenchmarkSink.stop_when()` returns that run-scoped flag
  (restored by rewind). The caller still controls `runner.run(stop_when=...)`
  or drives `runner.step`, and there are no core benchmark limit keys to
  consume.
- **No episode clock.** The pack never infers `episode_length_s`; an episode
  ends only through `runner.terminate(...)` or a pack sink terminal.
- **MuJoCo integration lane.** The real tiny-arm test needs the D0 MuJoCo lane.
  A missing backend is a test failure, not a skip.

Parity against the deleted core built-in benchmark is **frozen, not live**:

| Aspect | Frozen core built-in facts | `unigym-bench` pack | Status |
| --- | --- | --- | --- |
| Entry | `tests/_frozen_builtin_facts.json` (historical) | custom sink (`make_sink`) with monitors | Different composition; both used the same core runner tick engine |
| Report artifact | `benchmark_report.json` / `results.json` / `metrics.npz` (historical) | `external_benchmark_report.json` (opt-in) | Disjoint, pack-owned filename |
| Schema | `benchmark_report` (historical) | `external_benchmark_report/3` | Not byte-compatible; mapping documented here |
| Episode end | core term group + episode horizon (historical) | explicit `runner.terminate`/sink terminal; verdicts `terminal`, `deadline`, `core`, `incomplete`, `rewound` | Same core boundary facts; pack owns the lifecycle |
| Scoring | per-task `success` and metrics (historical) | monitor `report_view` and extension blocks; no aggregate success score | Different purpose; not interchangeable |
| Early stop | `min_episodes` / `max_steps` (historical) | flow stop flag + caller `stop_when` | Pack has no built-in limit policy |
| Retention | episode buffer retains for scoring (historical) | counters by default; capped event journal | Pack is lossier by default and reports its drops |
| Distribution | needed a benchmark task provider/pack runtime | installed sink; no task-provider activation | Pack avoids task-provider activation |
| Rewind | built-in session/buffer (historical) | pack engine provider `unigym_bench` | Same core rewind seam |

The post-deletion core no longer contains `unigym.benchmarks`,
`unigym.runner.evaluation`, `unigym.runner.episodes`, a run `mode`, an episode
clock or `EvaluationSession`; no pack test imports them. The historical
same-config facts are frozen in `tests/_frozen_builtin_facts.json`. The parity
test runs only the pack lane over the tiny-arm fixture, ends one episode after
two policy ticks with an explicit `runner.terminate`, and asserts the frozen:

- episode count (1);
- total episode ticks (2);
- termination `outcome` counter (`deadline`).

The `code` and `source` namespaces intentionally change: the old core-owned
horizon fact was `runtime.episode_horizon` / `runtime`, while an explicit
boundary is custom-owned (`custom.episode_horizon` / `custom`). That is
**frozen fact parity plus a documented namespace change**, not report or
scoring parity. The built-in task `success`/metric block has no pack
equivalent, and the two report schemas are not byte-compatible. The in-process
integration test additionally checks the same-run core `EpisodeTermination`
facts against the pack report and the report vocabulary against the core
`EpisodeOutcome` / `TerminationSource` enums. No throughput claim is made; the
retention test proves the journal is bounded by `max_events`.

### Release validation

The `0.3.0` release is validated against the post-deletion core worktree. From
`packages/unigym-bench`, using the Python from the core virtual environment:

```bash
PYTHONPATH=<core-worktree>:src python -m pytest -q tests
PYTHONPATH=<core-worktree>:src python -m pytest -q \
  tests/test_parity.py tests/test_retention.py tests/test_rewind.py \
  tests/test_integration.py
PYTHONPATH=<core-worktree>:src python -m ruff check src tests
PYTHONPATH=<core-worktree>:src python -m ruff format --check src tests
```

The full suite includes the real tiny-arm MuJoCo integration, explicit
`terminate(reset=True/False)` lifecycle coverage, zero-episode and multi-env
conformance, and the bounded-retention test; the focused lane covers parity,
retention, rewind and tiny-arm. This evidence is a CPU/MuJoCo lane only. No GPU
or Isaac Lab lane is claimed.

## Dependencies

Development requires Python >=3.11 with a compatible `unigym`
(`>=0.108,<0.109`) installed, plus the `dev` extra. The YAML loader requires
`PyYAML>=6.0,<7.0`, which is a direct runtime dependency of the pack. Pytest
adds `src` to its own path, so tests do not need a caller-provided
`PYTHONPATH`; the import-fence subprocess also sets its own path from the
imported package.

`unigym_bench.__version__` is `0.3.0` for the post-deletion core `0.108.x`
baseline release. `CORE_SURFACE_VERSION` is a pack-local revision of the
required `unigym` public surface (currently `4`, after adopting the explicit
sink selection and reset-aware lifecycle seams), not the distribution version.
