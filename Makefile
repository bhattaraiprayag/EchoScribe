SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.RECIPEPREFIX := >
.DEFAULT_GOAL := help

UV ?= uv
DOCKER ?= docker
APP_MODULE ?= backend.main:app
HOST ?= 0.0.0.0
PORT ?= 8000
DOCKER_IMAGE ?= echoscribe:test
COV_ARGS ?= --cov=backend --cov-report=term-missing

.PHONY: help sync install lint format format-check pre-commit test test-target coverage smoke docker-build check clean clean-cache clean-pyc

help:
>@printf "Available targets:\n"
>@printf "  %-14s %s\n" "sync" "Install/update project dependencies with uv"
>@printf "  %-14s %s\n" "lint" "Run Ruff lint checks"
>@printf "  %-14s %s\n" "format" "Auto-format Python files with Ruff"
>@printf "  %-14s %s\n" "format-check" "Verify formatting without changing files"
>@printf "  %-14s %s\n" "pre-commit" "Run all pre-commit hooks"
>@printf "  %-14s %s\n" "test" "Run full pytest suite"
>@printf "  %-14s %s\n" "test-target" "Run selected tests (TEST=tests/test_api.py)"
>@printf "  %-14s %s\n" "coverage" "Run pytest coverage gate for backend package"
>@printf "  %-14s %s\n" "smoke" "Run startup smoke check for backend.main:app"
>@printf "  %-14s %s\n" "docker-build" "Build Docker image"
>@printf "  %-14s %s\n" "check" "Run lint + format-check + pre-commit + test + coverage + smoke"
>@printf "  %-14s %s\n" "clean" "Remove Python/test/build caches safely"

sync:
>$(UV) sync --dev

install: sync

lint:
>$(UV) run ruff check .

format:
>$(UV) run ruff format .

format-check:
>$(UV) run ruff format --check .

pre-commit:
>$(UV) run pre-commit run --all-files --show-diff-on-failure

test:
>$(UV) run pytest -q

test-target:
>@test -n "$(TEST)" || (echo "Usage: make test-target TEST=tests/test_api.py" >&2; exit 1)
>$(UV) run pytest -q $(TEST)

coverage:
>$(UV) run pytest -q $(COV_ARGS)

smoke:
>bash -c 'set -euo pipefail; log=$$(mktemp); status=0; timeout 25 $(UV) run uvicorn $(APP_MODULE) --host $(HOST) --port $(PORT) >"$$log" 2>&1 || status=$$?; if [[ "$$status" -eq 0 || "$$status" -eq 124 ]]; then grep -q "Application startup complete" "$$log" || { cat "$$log"; rm -f "$$log"; exit 1; }; else cat "$$log"; rm -f "$$log"; exit "$$status"; fi; rm -f "$$log"'

docker-build:
>$(DOCKER) build --tag $(DOCKER_IMAGE) .

check: lint format-check pre-commit test coverage smoke

clean-pyc:
>find . -type d -name "__pycache__" -prune -exec rm -rf {} +
>find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

clean-cache:
>rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist
>find . -type d -name "*.egg-info" -prune -exec rm -rf {} +

clean: clean-pyc clean-cache
