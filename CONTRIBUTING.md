# Contributing

## Branches and pull requests

- Branch from `main`; use a short descriptive name (`feat/...`, `fix/...`,
  `docs/...`).
- Open a pull request against `main`. Keep PRs focused on one package or one
  concern.
- CI must pass before merge. Maintainers squash or rebase as appropriate; do
  not force-push shared branches and do not rewrite `main`.
- Releases are cut from `main` by tagging, not by long-lived release branches.

## Layout rules

- `packages/*` are the only uv workspace members.
- `test-packs/` is repo-only: keep it out of the root workspace and out of
  published metadata.
- Each package owns its `pyproject.toml`, version and changelog/README.
- Do not vendor core (`unigym`) into this repository.
- Do not add `path` dependencies to published package metadata. Use version
  ranges (`unigym>=0.106,<0.107`) and install core from a checkout in CI and
  local development.
- Keep large assets, local venvs, build outputs and caches out of git unless
  they are intentional tracked test-pack assets.

## Development workflow

Core is not on PyPI yet, so use a core checkout:

```bash
git clone https://github.com/unsigned123/unigym ../unigym-core
pip install -e ../unigym-core
pip install -e 'packages/unigym-rl[gymnasium,skrl]'
pip install -e packages/unigym-bench
```

Tests and static checks for `unigym-rl`:

```bash
PYTHONPATH=../unigym-core:packages/unigym-rl/src \
  python -m pytest packages/unigym-rl/tests -q

python -m ruff check packages/unigym-rl/src packages/unigym-rl/tests packages/unigym-rl/examples
python -m ruff format --check packages/unigym-rl/src packages/unigym-rl/tests packages/unigym-rl/examples
```

Tests and static checks for `unigym-bench`:

```bash
PYTHONPATH=../unigym-core:packages/unigym-bench/src \
  python -m pytest packages/unigym-bench/tests -q

python -m ruff check packages/unigym-bench/src packages/unigym-bench/tests
python -m ruff format --check packages/unigym-bench/src packages/unigym-bench/tests
```

`test-packs/` is tested from inside that directory with its own nested
workspace (`test-packs/README.md`). It is not part of the root CI lane.

## Release workflow

1. Bump the package version in `packages/unigym-rl/pyproject.toml`.
2. Update `COMPATIBILITY.md` if the core range changes.
3. Merge to `main`, then tag `unigym-rl-v<version>` and push the tag.
4. `release.yml` builds the sdist/wheel and publishes via PyPI Trusted
   Publishing. A manual workflow dispatch is available for TestPyPI-style
   dry runs; verify the build artifact before publishing to PyPI.
5. Never publish `test-packs` as a side effect of a package release.
