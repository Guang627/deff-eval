# passk-smooth: pilot-predicted design effects for pass@k

Code and data for the short paper on **how much concentrating a benchmark on few sources inflates the spread of
pass@k, and whether a small pilot predicts it**. Benchmarks often contain many problems from the same source (a
GSM-Symbolic template, a SWE-bench repository). The rule tested is

    SD(tier) / SD(baseline) ≈ sqrt(1 + (m − 1) ρ)

with m problems per source and ρ the intraclass correlation of the per-problem pass@k estimate. ρ is always estimated
on a pilot that is disjoint from the held-out data used to measure the spread. The analyses cover:

| benchmark | sources | data in this repo |
|---|---|---|
| GSM-Symbolic (main / P1 / P2) | templates (and original GSM8K questions) | 5,000 problems × 64 attempts, Qwen2.5-Math-1.5B-Instruct |
| SWE-bench Verified | repositories | resolved matrix of public submissions × 500 tasks (129 used) |
| MATH-500 (negative control) | none; subject × level cells | 500 problems × 200 attempts, same model |

Everything below runs on a CPU from the saved per-problem counts. Only regenerating the counts needs a GPU.

## Install

```bash
pip install -e ".[dev]"     # Python ≥ 3.10
make test                   # unit tests
```

## Reproduce the paper (CPU)

| paper item | command | output | time (1 core) |
|---|---|---|---|
| Table 2, Fig. 1–2 (predicted vs measured SD ratio, coverage) | `make deff` | `results/deff_validation/` | ~30 s |
| Fig. 4 input (GSM split by original question vs by level) | `make gsm-splits` | `results/deff_validation/gsm_{original,level}_20.csv` | ~1 min |
| Robustness section (pilot size, FPC on/off) | `make robustness` | `results/deff_validation/robustness_pilot_size.csv` | ~1.5 min |
| GSM-Symbolic tier table (pass@1/4/16, iid vs template bootstrap) | `make tiers` | `results/gsm_symbolic_qwen2.5-math-1.5b/tiers/` | ~45 s |
| SWE-bench per-submission analysis (Fig. 3 input) | `make swebench` | `results/swebench_verified/` | ~3 min |
| Figures | `make figures` | `figures/arxiv/figs/`, `figures/aaai/figs/` | seconds |

`make all` runs all of them in order. All scripts are seeded, so the shipped result files are reproduced exactly
(checked for every row above before release).

The GSM pilot / held-out split is made **by original GSM8K question** by default (`--gsm-split original`), so no
original question appears on both sides. The earlier split within difficulty level is kept as
`--gsm-split level` for comparison (Fig. 4):

```bash
python scripts/analysis/deff_validation.py --gsm-split original   # default, paper results
python scripts/analysis/deff_validation.py --gsm-split level --out outputs/deff_level
```

Other options: `--splits` (pilot/held-out splits, default 20), `--reps` (draws per tier, default 1000), `--seed`.
Method details and result tables: `docs/results_deff_validation.md`, `docs/results_gsm_symbolic.md`,
`docs/results_swebench_repos.md`.

## Data

* `results/gsm_symbolic_qwen2.5-math-1.5b/counts.csv`: one row per GSM-Symbolic problem
  (`problem_id`, template, level, `n_samples` = 64, `n_correct`, `source` = original GSM8K question id).
* `results/math500_qwen2.5-math-1.5b/counts.csv`: MATH-500, 200 attempts per problem, with subject and level.
* `results/swebench_verified/resolved_matrix.csv.gz`: submission × instance resolved flags, built from the public
  [SWE-bench/experiments](https://github.com/SWE-bench/experiments) repository (`evaluation/verified/*/results/results.json`).
  To rebuild it:

  ```bash
  git clone --depth 1 --filter=blob:none --sparse https://github.com/SWE-bench/experiments exp
  git -C exp sparse-checkout set --no-cone '/evaluation/verified/*/results/results.json'
  python scripts/analysis/swebench_repos.py --build exp
  ```

## Regenerating the counts (GPU, optional)

The counts come from sampling Qwen2.5-Math-1.5B-Instruct with vLLM (T = 0.6, top-p 0.95, 2,048 new tokens) and grading
each answer by LaTeX normalisation and numeric comparison, with `math-verify` as a second opinion when installed.

```bash
pip install -e ".[gen,grading]"
passk generate --config configs/generate_gsm_symbolic_pilot.yaml   # 100 problems, schema / grading check
passk generate --config configs/generate_gsm_symbolic.yaml         # 5,000 problems × 64 attempts
passk generate --config configs/generate_qwen_math500.yaml         # 500 problems × 200 attempts
```

Runs are resumable and write `completions.jsonl`, `counts.csv` and `meta.json` to `output_dir`. Any config key can be
overridden on the command line (`passk generate --config ... n_samples=16 limit=10`). `make smoke` checks the whole
pipeline on CPU with a mock model.

## Layout

```
scripts/analysis/   deff_validation.py (main analysis), deff_robustness.py, export_gsm_splits.py,
                    template_tiers.py + plot_template_tiers.py (GSM tiers), swebench_repos.py (SWE-bench)
figures/            figure scripts for the arXiv version (arxiv/) and the workshop version (aaai/)
results/            saved counts, resolved matrix and every result file the paper reports
src/passk_smooth/   generation (vLLM / HF / mock backends), answer grading, unbiased pass@k estimator, intervals
configs/            generation configs
tests/              unit tests
```

## License

MIT (see `pyproject.toml`).
