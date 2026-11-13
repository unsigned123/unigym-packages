# unigym-test-packs

Independent pytest-only pack distributions for `unigym`.

This repository is **not** a submodule of `unigym-core`. It ships the pytest
plugin (`unigym-test-packs`) and content pack distributions. Pack discovery uses
the dedicated `unigym.test_packs` entry-point group; the plugin never reads asset
environment variables.

## M0-M2

- `unigym-test-packs`: marker, discovery, fixtures and strict mode.
- `unigym-test-pack-basic`: first content pack (`providers` and `scenes` faces).
- `unigym-test-pack-isaaclab-cartpole`: Cartpole USD asset pack (`assets` face).
- `unigym-test-pack-g1`: complete 43-DoF G1 USD tree (`assets` face), shipped as
  one selective wheel; runtime configuration selects declared resources.

## Commands

```bash
uv sync --all-packages
uv run --frozen pytest
uv run --frozen pytest --strict-test-packs
uv run --frozen ruff check .
uv run --frozen pyright
uv run --frozen lint-imports
uv run --frozen python tools/check_wheel.py --core-checkout ../unigym-core
```

Test code declares pack dependencies with:

```python
pytestmark = pytest.mark.requires_test_packs("basic")
```

Then uses `pack_root(name)` for the installed tree or `pack_runtime(*names)` for
the explicit core pack runtime. Activation is always explicit through
`PackRuntime.activate`.

Missing packs skip by default. CI lanes use `--strict-test-packs` so a missing
required pack fails at test setup.
