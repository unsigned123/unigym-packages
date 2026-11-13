.PHONY: sync test test-strict lint typecheck imports wheel

sync:
	uv sync --all-packages

test:
	uv run --frozen pytest

test-strict:
	uv run --frozen pytest --strict-test-packs

lint:
	uv run --frozen ruff check .

typecheck:
	uv run --frozen pyright

imports:
	uv run --frozen lint-imports

wheel:
	uv run --frozen python tools/check_wheel.py --core-checkout ../unigym-core
