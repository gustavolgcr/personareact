# Thin wrappers over the CLI. `personareact --help` is the real documentation.
.PHONY: help install install-all test lint fmt smoke e1 e1-dry judge analyze corpus verify clean

VENV := .venv
PY   := $(VENV)/bin/python
CLI  := $(VENV)/bin/personareact

help:
	@grep -E '^[a-z0-9-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## locked install: analysis + judge + dev, exactly the versions in uv.lock
	uv sync --extra analysis --extra judge --extra dev

install-all: ## everything, including sentence embeddings (~1.5 GB of PyTorch)
	uv sync --extra analysis --extra judge --extra embeddings --extra dev

ci: ## run locally exactly what CI runs
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts
	uv run pytest -q
	uv run personareact list

test: ## run the test suite
	$(PY) -m pytest -q

lint: ## check formatting and lint rules
	$(VENV)/bin/ruff check src tests scripts
	$(VENV)/bin/ruff format --check src tests scripts

fmt: ## apply formatting
	$(VENV)/bin/ruff format src tests scripts
	$(VENV)/bin/ruff check --fix src tests scripts

verify: ## check that the databases are present and are the expected build
	$(CLI) data verify

corpus: ## corpus counts and integrity checks
	$(CLI) corpus stats

smoke: ## one question, six approaches, one model -- a few cents
	$(CLI) run --experiment smoke

e1-dry: ## what would experiment 1 cost?
	$(CLI) run --experiment e1 --dry-run

e1: ## run experiment 1 (504 cells -- check e1-dry first)
	$(CLI) run --experiment e1

judge: ## score a run: make judge RUN=<run_id>
	$(CLI) judge --run $(RUN)

analyze: ## tables and figures: make analyze RUN=<run_id> [EXP=e1|e2]
	$(CLI) analyze --run $(RUN) --experiment $(or $(EXP),e1)

thesis-diff: ## recompute experiment 2 over the thesis answers, corrected
	$(PY) scripts/compare_with_thesis.py --csv

clean: ## remove generated tables and figures (results/ and data/ are kept)
	rm -rf figures/* tables/* .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
