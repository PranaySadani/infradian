# INFRADIAN — reproducible pipeline. Everything runs through `uv run`.
# `make reproduce` regenerates every synthetic-tier number from a clean clone,
# with NO mcPHASES access required.

.DEFAULT_GOAL := help
PY := uv run

.PHONY: help setup hooks synth features bench train eval web-data serve test lint reproduce all

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create the uv environment and install the package
	uv sync --extra api --extra llm --extra viz --extra dev
	$(MAKE) hooks

hooks: ## Install the git pre-commit privacy guard
	cp scripts/hooks/pre-commit .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit
	@echo "pre-commit privacy guard installed"

synth: ## Generate the Tier C synthetic cohort (CC-BY)
	$(PY) -m infradian.synth.generator --n 600 --days 120 --seed 7 --out data/tier_c

features: ## Build causal features from the current dataset
	$(PY) scripts/build_features.py

bench: ## Run the full benchmark (baselines + model) on Tier C
	$(PY) scripts/run_bench.py --tier C

train: ## Train the reference model on synthetic data
	$(PY) -m infradian.models.gbm --train data/tier_c --out results/models

eval: ## Evaluate and write results JSON
	$(PY) scripts/run_bench.py --tier C --write-results

web-data: ## Export static JSON + generated TS types for the frontend
	$(PY) scripts/export_web_data.py

publish-hf: ## Mirror the dataset + model to HuggingFace (needs HF_TOKEN in env or .env)
	@test -n "$(HF_ORG)" || { echo "usage: make publish-hf HF_ORG=<your-hf-username>"; exit 1; }
	$(PY) scripts/publish_hf.py --org $(HF_ORG)

publish-hf-check: ## Dry-run the HuggingFace publish (no auth, no upload)
	@test -n "$(HF_ORG)" || { echo "usage: make publish-hf-check HF_ORG=<your-hf-username>"; exit 1; }
	$(PY) scripts/publish_hf.py --org $(HF_ORG) --dry-run

serve: ## Run the FastAPI backend locally
	$(PY) -m uvicorn infradian.api.main:app --reload --port 8000

test: ## Run the test suite (leakage, causality, parity, privacy)
	$(PY) -m pytest

lint: ## Lint with ruff
	$(PY) -m ruff check src scripts tests

reproduce: setup synth features bench eval web-data ## Regenerate every synthetic-tier number from scratch
	@echo "reproduce complete — see results/"

all: reproduce test ## Full pipeline plus tests
