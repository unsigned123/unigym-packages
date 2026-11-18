# unigym-packages

First-party package distributions for [`unigym`](https://github.com/unsigned123/unigym).

This repository is a **uv workspace holder**. It is not the core library and it
does not replace it. Core (`unigym`) stays in its own repository
(`unsigned123/unigym`) and is a version-ranged dependency of every package here
— never vendored, copied, or path-pinned in published metadata.

## Packages

| Path | Distribution | Purpose |
| --- | --- | --- |
| `packages/unigym-rl` | `unigym-rl` | L2 task-environment layer over core: `TaskEnv`, programmatic `EnvSpec`, optional Gymnasium `VectorEnv` face and skrl smoke. |
| `packages/unigym-robots` | `unigym-robots` | Pinned Unitree Go1/Go2/G1 MJCF and USD assets, downloaded from GitHub Release tarballs. No core dependency and no binaries in the wheel. |
| `test-packs/` | repo-only | Test-pack plugin and content-pack distributions plus their USD assets. Not a workspace member and not published by default; see `test-packs/README.md`. |

`test-packs/` is kept in this repository only for co-location. It has its own
nested uv workspace and is intentionally excluded from the root workspace
because of ~75 MB of tracked USD assets. A future migration to Git LFS or
GitHub release assets is possible but is not part of this layout.

`unigym-rl` is developed in `packages/unigym-rl` in this repository. The
standalone `unigym-rl` checkout is archived; see `MIGRATION.md`. Core remains
separate.

## Core is separate

```text
unigym (core repo, version 0.106.x)  ->  dependency
  -> packages/unigym-rl
  -> user / RL framework training loop

packages/unigym-robots  -> standalone asset downloader (no core dependency)
```

Packages depend on core through a version range declared in their own
`pyproject.toml`. Do not add path dependencies to published metadata and do
not copy core source into this repo.

## Local development

Core is not on PyPI yet, so a plain `uv sync` cannot resolve `unigym` and no
lockfile is committed until it can. Use a core checkout on `PYTHONPATH`, or
install core editable into an environment first:

```bash
git clone https://github.com/unsigned123/unigym ../unigym-core
python -m venv .venv
. .venv/bin/activate
pip install -e ../unigym-core
pip install -e 'packages/unigym-rl[gymnasium,skrl]'

PYTHONPATH=../unigym-core:packages/unigym-rl/src \
  python -m pytest packages/unigym-rl/tests -q
```

Static checks:

```bash
python -m ruff check packages/unigym-rl/src packages/unigym-rl/tests packages/unigym-rl/examples
python -m ruff format --check packages/unigym-rl/src packages/unigym-rl/tests packages/unigym-rl/examples
```

`unigym-robots` needs neither core nor the network for its tests:

```bash
python -m pytest packages/unigym-robots/tests -q
python -m ruff check packages/unigym-robots/src packages/unigym-robots/tests
python -m ruff format --check packages/unigym-robots/src packages/unigym-robots/tests
```

Its robot assets are GitHub Release artifacts, not wheel content:

```bash
unigym-robots download --set mjcf-v0
```

Run the CartPole demo from the package directory:

```bash
cd packages/unigym-rl
python examples/cartpole/train_skrl.py --num-envs 1 --max-iterations 1
```

## Release and version policy

- Each package keeps its own `pyproject.toml`, version and tag.
- `unigym-rl` releases are tagged `unigym-rl-v<version>` (for example
  `unigym-rl-v0.1.0`). Tags trigger `.github/workflows/release.yml`.
- Core compatibility is recorded per release in `COMPATIBILITY.md`; the current
  lane is `unigym-rl 0.1.x` -> `unigym >=0.106,<0.107`.
- `unigym-robots` code is versioned independently. Its robot assets are attached
  to the GitHub Release `unigym-robots-assets-v0.1.0` and are never published to
  PyPI. See `packages/unigym-robots/PROVENANCE.md`.
- `test-packs/` is not published by default. Publishing requires an explicit,
  separate decision about assets and packaging.
- GitHub Actions publishes through PyPI Trusted Publishing; no API tokens are
  stored in the repository.
