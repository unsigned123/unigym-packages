# Migration record: standalone `unigym-rl` into `unigym-packages`

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
