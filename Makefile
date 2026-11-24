.PHONY: install lint test smoke deff robustness gsm-splits tiers swebench figures all

PY ?= python

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests && ruff format --check src tests && mypy

test:
	pytest

# CPU-only check of the generation pipeline: mock model -> grading -> counts.csv (~5 s)
smoke:
	passk generate --config configs/generate_smoke.yaml track=false

# ---- paper results (CPU only, from the saved counts / resolved matrix) ----

# Table 2, Fig. 1-2: pilot-predicted vs held-out SD ratio and coverage (GSM split by original question)
deff:
	$(PY) scripts/analysis/deff_validation.py --gsm-split original --out results/deff_validation

# Fig. 4 input: the same GSM analysis with the earlier within-level split, plus the original-split rows
gsm-splits: deff
	$(PY) scripts/analysis/deff_validation.py --gsm-split level --out outputs/deff_level
	$(PY) scripts/analysis/export_gsm_splits.py outputs/deff_level/deff_splits.csv level
	$(PY) scripts/analysis/export_gsm_splits.py results/deff_validation/deff_splits.csv original

# Robustness section: pilot size, FPC on/off
robustness:
	$(PY) scripts/analysis/deff_robustness.py

# GSM-Symbolic tier table (pass@1/4/16, iid vs template bootstrap)
tiers:
	$(PY) scripts/analysis/template_tiers.py --counts results/gsm_symbolic_qwen2.5-math-1.5b/counts.csv \
	  --out results/gsm_symbolic_qwen2.5-math-1.5b/tiers
	$(PY) scripts/analysis/plot_template_tiers.py --dir results/gsm_symbolic_qwen2.5-math-1.5b/tiers

# SWE-bench Verified per-submission analysis (Fig. 3 input)
swebench:
	$(PY) scripts/analysis/swebench_repos.py

figures:
	cd figures/arxiv && $(PY) make_figs.py
	cd figures/aaai && $(PY) make_fig12.py && $(PY) make_fig3.py && $(PY) make_fig4.py

all: deff gsm-splits robustness tiers swebench figures
