.PHONY: install lint test smoke deff robustness gsm-splits tiers swebench figures all

PY ?= python

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests && ruff format --check src tests && mypy

test:
	pytest

smoke:
	passk generate --config configs/generate_smoke.yaml track=false

deff:
	$(PY) scripts/analysis/deff_validation.py --gsm-split original --out results/deff_validation

gsm-splits: deff
	$(PY) scripts/analysis/deff_validation.py --gsm-split level --out outputs/deff_level
	$(PY) scripts/analysis/export_gsm_splits.py outputs/deff_level/deff_splits.csv level
	$(PY) scripts/analysis/export_gsm_splits.py results/deff_validation/deff_splits.csv original

robustness:
	$(PY) scripts/analysis/deff_robustness.py

tiers:
	$(PY) scripts/analysis/template_tiers.py --counts results/gsm_symbolic_qwen2.5-math-1.5b/counts.csv \
	  --out results/gsm_symbolic_qwen2.5-math-1.5b/tiers
	$(PY) scripts/analysis/plot_template_tiers.py --dir results/gsm_symbolic_qwen2.5-math-1.5b/tiers

swebench:
	$(PY) scripts/analysis/swebench_repos.py

figures:
	cd figures/arxiv && $(PY) make_figs.py
	cd figures/aaai && $(PY) make_fig12.py && $(PY) make_fig3.py && $(PY) make_fig4.py

all: deff gsm-splits robustness tiers swebench figures
