# Migration records

## Pack model v2 (core 0.110.0)

Core `0.110.0` removes the transitional pack bridge with no compatibility shim.
The public replacement is `unigym.packs.loader.PackLoader`; `unigym.pack_api`
still exports it. External consumers move like this:

```python
from unigym.packs import PackLoader, require

packs = require("unigym_robots")
pack = packs.pack("unigym_robots")      # stored Pack declaration
loader = PackLoader.from_pack_set(packs)

profile = loader.robot_profile("unigym_robots", "go2")    # was pack.robot("go2")
path = loader.resolve("providers", "go2")                # was 2-arg resource_for
item = loader.provider("robot", "go2")                   # stored PackItem
activated = loader.activate("unigym_robots")             # returns the Pack snapshot
robots = activated.items_for(face="providers", kind="robot")  # was handle.robots
```

The mapping for the removed names:

| Removed | Replacement |
| --- | --- |
| `PackRuntime` | `unigym.packs.loader.PackLoader` |
| `unigym.packs.runtime` / `unigym.packs.catalog` | `unigym.packs.loader` / `unigym.packs.packset` |
| `PackSet.robot(id)` / `Pack.robot(id)` | `PackLoader.robot_profile(pack, id)` |
| `resource_for(face, id)` | `PackLoader.resolve(face, id)` or `PackLoader.resource_for(item)` |
| `runtime.catalog.packs()` | `tuple(pack.name for pack in loader.pack_set.packs())` |
| `ActivationHandle.robots` | `loader.activate(pack).items_for(face="providers", kind="robot")` with `ItemStatus` |
| `Pack.provides` | `Pack.items` / `Pack.items_for(...)` |

`unigym-bench 0.4.1` and `unigym-rl 0.2.2` are the coordinated releases for
this core range; `unigym-robots 0.1.1` updates its docs and tests while keeping
no core dependency. `test-packs/ci/core-pin.toml` is a separate vendored pin
and is not part of these package releases.

## Standalone `unigym-rl` into `unigym-packages`

`unigym-rl` is now developed in `packages/unigym-rl` of this repository. This
file records the history import from the standalone checkout.

| Item | Value |
| --- | --- |
| Source path | `/data1/code/dlx/unigym-rl-go2-contact` |
| Source branch | `feat/go2-contact-reward` |
| Source commit | `93c6e89e53709c470b150bd460cc3cff79743f46` |
| Target repository | `unsigned123/unigym-packages` |
| Target base | `f0118d3` (`origin/main`) |
| Target branch | `feat/monorepo-unigym-rl` |
| Target prefix | `packages/unigym-rl` |
| Date | 2026-11-18 |

The import used `git subtree`, so the standalone commit is a parent of the
import merge commit and remains in history:

```bash
git remote add unigym-rl-legacy /data1/code/dlx/unigym-rl-go2-contact
git fetch unigym-rl-legacy feat/go2-contact-reward

git rm -r packages/unigym-rl
git commit -m "chore(unigym-rl): remove stale package snapshot"

git subtree add \
  --prefix=packages/unigym-rl \
  unigym-rl-legacy feat/go2-contact-reward
```

The subtree import replaced the older `packages/unigym-rl` snapshot. The
package code, tests, and examples come from `93c6e89`; the monorepo packaging
metadata (version `0.1.0`, `unigym>=0.106,<0.107`, license, authors, project
URLs) was restored on top.

The standalone checkout is archived after verification and should receive no
further changes. Core (`unigym`) remains in its own repository
(`unsigned123/unigym`) and is not part of this monorepo. No release tag is
created by this migration.
