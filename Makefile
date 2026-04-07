.PHONY: install run debug lint clean test

install:
	uv sync --dev

run:
	uv run python -m src

debug:
	uv run python -m src --input data/input/function_calling_tests.json --output data/output/function_calling_results.json

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

test:
	uv run pytest -q

clean:
	python -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['.mypy_cache', '.pytest_cache', '.ruff_cache', 'data/output'] if Path(p).exists()]"