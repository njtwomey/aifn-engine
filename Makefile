# Common tasks. `make help` lists them.
.DEFAULT_GOAL := help
# Python sources (the fixture generators and the data scripts) that ruff lints and formats.
PY_SRC := packages/compute/test/fixtures packages/methods/test/fixtures scripts
PRETTIER_FILES := git ls-files -z -co --exclude-standard | xargs -0 sh -c 'for f; do [ -f "$$f" ] && printf "%s\0" "$$f"; done' _

.PHONY: help install check lint layers names format typecheck test bench fixtures fixtures-check package examples examples-check examples-build shots thumbs clean

help: ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install the Node dependencies, and the Python fixture libraries into .venv when uv is installed
	npm install
	@if command -v uv >/dev/null; then uv sync; else echo "uv not found: skipping the Python environment (only make fixtures and the Python lint need it)"; fi

check: lint typecheck test examples-check ## Everything: lints, types, tests, examples render

lint: layers names ## Lint and format-check TypeScript; lint the Python fixture generators when uv is installed
	npx oxlint
	node examples/check.ts --imports-only
	$(PRETTIER_FILES) | xargs -0 npx prettier --check --ignore-unknown
	@if command -v uv >/dev/null; then uv run ruff check $(PY_SRC) && uv run ruff format --check $(PY_SRC); else echo "uv not found: skipping the Python lint"; fi

layers: ## Check imports against the module tree (modules.json): compute tiers, the methods area DAG
	node scripts/layers.ts

names: ## Check that no two modules export different values under one name (allowlist in the test)
	npx vitest run --config packages/methods/vitest.config.ts test/names.test.ts

format: ## Format TypeScript and Python
	$(PRETTIER_FILES) | xargs -0 npx prettier --write --ignore-unknown --log-level warn
	@if command -v uv >/dev/null; then uv run ruff check --fix $(PY_SRC) && uv run ruff format $(PY_SRC); fi

typecheck: ## Type-check every package, the examples and the scripts
	npx tsc -b

test: layers names ## Run the tests of compute, methods and render
	npx vitest run --config packages/compute/vitest.config.ts
	@# The name lint (test/names.test.ts) already ran as the names prerequisite; make runs a prerequisite once.
	npx vitest run --config packages/methods/vitest.config.ts --exclude test/names.test.ts
	npx vitest run --config packages/render/vitest.config.ts

bench: ## Run the compute micro-benchmarks (reported, not gated; not part of check)
	npx vitest bench --run --config packages/compute/vitest.config.ts

fixtures: ## Regenerate golden test values from Python (FIXTURES="numerics/linalg numerics ..." for some)
	uv run python packages/compute/test/fixtures/generate.py $(FIXTURES)

fixtures-check: ## Regenerate every fixture in memory and fail if any differs from its committed file (slow; not in check)
	uv run python packages/compute/test/fixtures/generate.py --check $(FIXTURES)

package: ## Build aifn-compute as a publishable package in packages/compute/dist (ARGS="--version x.y.z")
	node scripts/package.ts $(ARGS)

examples: ## Run the AIFN Engine site (front page and examples gallery) → http://localhost:5192/
	@echo "AIFN Engine examples → http://localhost:5192/  (recipes at /<section>/<slug>)"
	npx vite --config examples/vite.config.ts

examples-check: ## Render every example recipe on the server and report any that throw
	node examples/check.ts

examples-build: ## Build the examples gallery into examples/dist (EXAMPLES_BASE=/<path>/ when not served from /)
	npx vite build --config examples/vite.config.ts
	@# GitHub Pages serves 404.html for unknown paths, which lets the single-page app open /<section>/<slug> directly.
	cp examples/dist/index.html examples/dist/404.html

shots: ## Screenshot example pages and figures into .scratch/shots (ARGS="--only lines/line-chart --theme dark ..."; needs Chrome)
	node scripts/screenshot.ts $(ARGS)

thumbs: ## Rebuild the gallery thumbnails in examples/public/thumbs (ARGS="--only lines" for a section; needs Chrome)
	node scripts/screenshot.ts --thumbs --no-sliders $(ARGS)

clean: ## Remove build output and caches
	rm -rf packages/*/dist examples/dist node_modules/.tmp node_modules/.vite .ruff_cache
