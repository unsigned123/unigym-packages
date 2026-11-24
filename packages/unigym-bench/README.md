# unigym-bench

`unigym-bench` is an external benchmark pack for `unigym` that will build a
streaming, monitor-driven benchmarking layer on the public sink, signal and
evaluation contracts, without importing `unigym.benchmarks`,
`unigym.runner.evaluation` or `unigym.runner.episodes`.

Status: P1. The package contains the skeleton — version metadata, the
core-contract import fence, strict JSON helpers and the closed YAML task-spec
loader (`unigym_bench.specs`). There is no benchmark engine, monitor, sink,
report, scenario or sampler implementation yet.

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
