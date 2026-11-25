# Compatibility

Core (`unigym`) is developed and released separately from the packages in this
repository. Each package release declares the core range it supports.

| Package | Package version | Core requirement | Notes |
| --- | --- | --- | --- |
| `unigym-rl` | `0.2.x` | `unigym >=0.108,<0.109` | L2 task-environment layer over the post-deletion core explicit reset/terminate lifecycle; no core episode clock; validated against core `0.108.0`. |
| `unigym-bench` | `0.3.x` | `unigym >=0.108,<0.109` | External benchmark layer over the post-deletion core explicit sink, reset-aware lifecycle, value/context and termination contracts; validated against core `0.108.0`. |
| `unigym-robots` | `0.1.x` | none | Standalone asset downloader for pinned Unitree Go1/Go2/G1 MJCF and USD assets. Assets are GitHub Release tarballs, not wheel content. |

Core is not on PyPI yet. Install it from a checkout until it is published:

```bash
git clone https://github.com/unsigned123/unigym ../unigym-core
pip install -e ../unigym-core
```

When core publishes to PyPI, the same version range becomes installable without
a checkout. Do not widen a package's core range without running its test suite
against the new core lane.

Release ordering: core `0.108.0` must be on core `main` before
`unigym-bench 0.3.x` and `unigym-rl 0.2.x` can resolve their dependency range
or run their CI lanes.
