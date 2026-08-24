# supercombo-blindspot: reproducible entrypoints.
# Cache-only paths need no GPU, no CARLA, and no ONNX model.

.DEFAULT_GOAL := help
PY ?= python

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	 | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: venv
venv: ## Create .venv and install unit-test deps (CPU-only, mirrors CI)
	$(PY) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements-ci.txt

.PHONY: test
test: ## Run the unit-test suite (as CI does)
	$(PY) -m pytest -q

.PHONY: repro
repro: ## Regenerate every table + figure from committed caches (no GPU/model)
	bash scripts/repro_from_caches.sh

.PHONY: docker-build
docker-build: ## Build the reproducible test image
	docker build -t supercombo-blindspot .

.PHONY: docker-test
docker-test: docker-build ## Build the image and run the test suite inside it
	docker run --rm supercombo-blindspot

.PHONY: clean
clean: ## Remove Python caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
