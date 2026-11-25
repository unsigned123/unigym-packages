# Compatibility

Core (`unigym`) is developed and released separately from the packages in this
repository. Each package release declares the core range it supports.

| Package | Package version | Core requirement | Notes |
| --- | --- | --- | --- |
| `unigym-rl` | `0.1.x` | `unigym >=0.106,<0.107` | L2 task-environment layer; validated against core `0.106.0` and core `main` at `0d66d29`. |
| `unigym-bench` | `0.2.x` | `unigym >=0.107,<0.108` | External benchmark layer over the core value/context, sink and terminal contracts; validated against core `0.107.0` (C13 stack). |
| `unigym-robots` | `0.1.x` | none | Standalone asset downloader for pinned Unitree Go1/Go2/G1 MJCF and USD assets. Assets are GitHub Release tarballs, not wheel content. |

Core is not on PyPI yet. Install it from a checkout until it is published:

```bash
git clone https://github.com/unsigned123/unigym ../unigym-core
pip install -e ../unigym-core
```

When core publishes to PyPI, the same version range becomes installable without
a checkout. Do not widen a package's core range without running its test suite
against the new core lane.

Release ordering: core `0.107.0` (C13) must be on core `main` before
`unigym-bench 0.2.x` can resolve its dependency range or run its CI lane.
