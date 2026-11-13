# CI lanes

The test-pack repository runs against a pinned core checkout in PR CI and
against core `main` nightly. `ci/core-pin.toml` owns the core repository and
the pushed commit used by the pinned lanes.

## Workspace wiring

The workspace source is:

```toml
[tool.uv.sources]
unigym = { path = "../unigym-core", editable = true }
```

CI checks out the core repository into `core/` and creates the sibling
`../unigym-core` symlink before `uv sync --all-packages`. Local development uses
the real sibling checkout directly.

## PR lanes

```text
lint            ruff, pyright, import-linter
unit            tests/unit + tests/contract
strict-full     all tests with --strict-test-packs
subset-basic    plugin + basic only; missing packs skip
strict-missing  Cartpole-marked selector must fail at setup
wheel           build/install all wheels outside the checkout
```

The wheel lane invokes `tools/check_wheel.py --core-checkout ../unigym-core
--online`; it proves entry points, nested asset data, notices and consumer
behavior in a temporary venv.

## Nightly lanes

```text
core-pinned  all tests with --strict-test-packs against ci/core-pin.toml
core-main    the same suite against unsigned123/unigym main
wheel-main   isolated wheel/consumer and large-asset gates against core main
```

Nightly drift is a failure, not a skip.

## Local equivalents

```bash
uv sync --all-packages
uv run --frozen pytest tests/unit tests/contract -q
uv run --frozen pytest --strict-test-packs -q
uv run --frozen pytest tests/consumer/test_basic_pack.py -q
uv run --frozen pytest tests/consumer/test_cartpole_pack.py -q
uv run --frozen pytest tests/consumer/test_g1_pack.py -q
uv run --frozen python tools/check_wheel.py --core-checkout ../unigym-core
```

The installed-set subset lane is CI-specific: it starts from a full workspace
sync and uninstalls the pack distributions it does not want. It is deliberately
non-strict so the missing-pack skip path is exercised. The per-pack local
targets select test paths instead; they do not change the installed set.

## Pin maintenance

Update `ci/core-pin.toml` only to a commit that exists in
`unsigned123/unigym`. A local-only hash makes the pinned lanes unreproducible.
