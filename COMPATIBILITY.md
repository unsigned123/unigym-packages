# Compatibility

Core (`unigym`) is developed and released separately from the packages in this
repository. Each package release declares the core range it supports.

| Package | Package version | Core requirement | Notes |
| --- | --- | --- | --- |
| `unigym-rl` | `0.2.x` | `unigym >=0.110,<0.111` | L2 task-environment layer over the post-deletion core explicit reset/terminate lifecycle; no core episode clock. `0.2.2` moves the range onto the core pack model v2 cutover and resolves the Go2 example profile through `PackLoader.robot_profile`; validated against core `0.110.0`. |
| `unigym-bench` | `0.4.x` | `unigym >=0.110,<0.111` | Flow-owned external benchmark layer over the post-deletion core explicit sink, reset-aware lifecycle, value/context and termination contracts; values-only monitors, `BenchmarkFlow` decision/report ownership, task/report `/3`, rewind `/5`, monitor `/4`. `0.4.1` is a metadata/compatibility release onto the core pack model v2 cutover; the production sink is unchanged and only the tiny-arm test helper uses `PackLoader`. Validated against core `0.110.0`. |
| `unigym-robots` | `0.1.x` | none | Standalone asset downloader for pinned Unitree Go1/Go2/G1 MJCF and USD assets. Assets are GitHub Release tarballs, not wheel content. `0.1.1` documents and tests profile loading through `PackLoader.robot_profile`; there is no core dependency. |

Core `0.110.0` is published on PyPI:

```bash
pip install "unigym>=0.110,<0.111"
```

Do not widen a package's core range without running its test suite against the
new core lane.

Core `0.110.0` is the breaking pack model v2 release: `PackRuntime`, the
catalog module and the `Pack.robot` / `PackSet.robot` helpers are gone in
favour of `PackLoader`. External consumers must migrate their pack code before
widening to `>=0.110,<0.111`; see `MIGRATION.md`.

The nested `test-packs/` workspace is vendored and out of scope for these
package release ranges. `test-packs/ci/core-pin.toml` pins the upstream core
commit for its own nightly/pinned lanes and is not part of the `unigym-bench` /
`unigym-rl` release set; do not update it as part of a package release.
