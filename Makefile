.PHONY: install test lint typecheck build doctor

install:
	python -m pip install -e ".[adaptive,analysis,data,dev]"

test:
	python -m pytest

lint:
	python -m ruff check src tests scripts examples

typecheck:
	python -m mypy src/gatekg

build:
	python -m build

doctor:
	python -m gatekg doctor
