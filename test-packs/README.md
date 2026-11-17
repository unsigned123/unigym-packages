# unigym-test-packs

Independent pytest-only pack distributions for `unigym`.

This repository is **not** a submodule of `unigym-core`. It ships the pytest
plugin (`unigym-test-packs`) and content pack distributions. Pack discovery uses
the dedicated `unigym.test_packs` entry-point group; the plugin never reads asset
environment variables.

## M0-M3 + R2b

- `unigym-test-packs`: marker, discovery, fixtures and strict mode.
- `unigym-test-pack-basic`: first content pack (`providers` and `scene_presets`
  faces).
- `unigym-test-pack-isaaclab-cartpole`: Cartpole USD asset pack (`assets` face).
- `unigym-test-pack-g1`: complete 43-DoF G1 USD tree and data-only `g1` robot
  profile, shipped as one selective wheel.
- `unigym-test-pack-gr1`: data-only `gr1` robot profile pack.

## Commands

```bash
uv sync --all-packages
uv run --frozen pytest
uv run --frozen pytest --strict-test-packs
uv run --frozen pytest tests/consumer/test_basic_pack.py
uv run --frozen pytest tests/consumer/test_cartpole_pack.py
uv run --frozen pytest tests/consumer/test_g1_pack.py
uv run --frozen ruff check .
uv run --frozen pyright
uv run --frozen lint-imports
uv run --frozen python tools/check_wheel.py --core-checkout ../unigym-core
```

CI lanes and the core pin are documented in `docs/ci.md`.

Test code declares pack dependencies with:

```python
pytestmark = pytest.mark.requires_test_packs("basic")
```

Then uses `pack_root(name)` for the installed tree or `pack_runtime(*names)` for
the explicit core pack runtime. Activation is always explicit through
`PackRuntime.activate`.

Missing packs skip by default. The strict/full CI lanes use
`--strict-test-packs` so a missing required pack fails at test setup; the
installed-set subset lane intentionally stays non-strict to prove the skip
path.

## Status in `unigym-packages`

This tree is imported verbatim from its former standalone repository. It is
**repo-only** in `unigym-packages`: not a member of the root uv workspace, not
hit by the root release pipeline, and not published by default because of the
large tracked USD assets in `packages/unigym-test-pack-g1` and
`packages/unigym-test-pack-isaaclab-cartpole`.

Possible future handling for the assets: move them to Git LFS or attach them to
GitHub releases and resolve them at test/download time. Either change is a
separate decision; do not publish these packs implicitly from a package
release.

The nested workspace still expects a core checkout at `../unigym-core` relative
to this directory when developed from `unigym-packages`, i.e. a `unigym-core`
directory at the repository root next to `test-packs/`. Adjust the local
`[tool.uv.sources]` entry if your checkout lives elsewhere.
