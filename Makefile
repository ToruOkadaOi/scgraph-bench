.PHONY: install install-dev lint format test typecheck check clean

PYTHON ?= python3
UV ?= uv

install:
	$(UV) pip install -e .

install-dev:
	$(UV) pip install -e ".[dev]"

lint:
	$(UV) run ruff check src tests

format:
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

test:
	$(UV) run pytest -v tests

typecheck:
	$(UV) run mypy src tests || true

check: lint typecheck test

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
