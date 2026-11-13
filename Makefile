.PHONY: sync test test-strict test-unit test-basic test-cartpole test-g1 test-gr1 lint typecheck imports wheel

sync:
	uv sync --all-packages

test:
	uv run --frozen pytest

test-strict:
	uv run --frozen pytest --strict-test-packs

test-unit:
	uv run --frozen pytest tests/unit tests/contract -q

test-basic:
	uv run --frozen pytest tests/consumer/test_basic_pack.py -q

test-cartpole:
	uv run --frozen pytest tests/consumer/test_cartpole_pack.py -q

test-g1:
	uv run --frozen pytest tests/consumer/test_g1_pack.py tests/consumer/test_g1_robot_profile.py -q

test-gr1:
	uv run --frozen pytest tests/consumer/test_gr1_pack.py -q

lint:
	uv run --frozen ruff check .

typecheck:
	uv run --frozen pyright

imports:
	uv run --frozen lint-imports

wheel:
	uv run --frozen python tools/check_wheel.py --core-checkout ../unigym-core
