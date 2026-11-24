# SWE-bench Verified: same task budget, fewer or more repositories (pass@1, public submissions)

No generation or grading: the per-task resolved lists of public submissions in SWE-bench/experiments
(`evaluation/verified/*/results/results.json`, 135 submissions with that file; 129 with resolve rate 0.1–0.9 used).
Each submission is one model + agent run with one attempt per task, so only pass@1 is studied.

Reproduce (CPU, ~2 min):
```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/SWE-bench/experiments /tmp/exp
git -C /tmp/exp sparse-checkout set --no-cone '/evaluation/verified/*/results/results.json'
python scripts/analysis/swebench_repos.py --build /tmp/exp   # -> results/swebench_verified/resolved_matrix.csv.gz (already saved)
python scripts/analysis/swebench_repos.py
```

## Design

* 500 tasks, 12 repositories: django 231, sympy 75, sphinx 44, matplotlib 34, scikit-learn 32, xarray 22, astropy 22,
  pytest 19, pylint 10, requests 8, seaborn 2, flask 1.
* Budget 40 tasks. Tiers (repositories × tasks): natural (40 tasks at random from the 500), 8×5, 4×10, 2×20, 1×40.
  Repositories drawn uniformly among those with at least V tasks; with 12 repositories, 1×40 can only use django,
  sympy or sphinx.
* Target fixed in advance for each submission: resolve rate over all 500 tasks (task-equal). Sensitivity:
  mean of per-repository resolve rates (repository-equal).
* Difficulty: three strata by the share of the other submissions that solved each task. "stratified" reweights the drawn
  tasks to the pool's difficulty mix (post-stratification) and bootstraps within strata.
* 300 draws per submission, 500 bootstrap resamples; numbers are averages over the 129 submissions.

## Results (pp except coverage)

| tier | SD | bias, plain mean | bias, stratified | iid cov. | Wilson cov. | stratified cov. | repo-cluster cov. |
|---|---|---|---|---|---|---|---|
| natural | 7.09 | +0.06 | +0.02 | 0.948 | 0.960 | 0.922 | 0.912 |
| 8×5 | 6.81 | −3.09 | −0.08 | 0.928 | 0.931 | 0.906 | 0.907 |
| 4×10 | 7.90 | −3.19 | −0.30 | 0.892 | 0.900 | 0.904 | 0.821 |
| 2×20 | 8.99 | −1.34 | −0.77 | 0.861 | 0.881 | 0.887 | 0.528 |
| 1×40 | 8.81 | −3.97 | −1.01 | 0.831 | — | 0.896 | 0.004 |

(SD of the stratified estimate: 5.3 / 5.1 / 5.3 / 5.6 / 5.6. Coverage of the repository-equal target is 0.03–0.12 lower;
see `table_swebench_repos.md`. Wilson 1×40: 0.856.)

Median share of outcome variance between repositories: 0.034 (IQR 0.024–0.049). On GSM-Symbolic the template share was
0.74–0.84.

## Reading

* The same direction as GSM-Symbolic, much weaker: concentrating 40 tasks on fewer repositories raises the SD from about
  7 to 9 pp and lowers iid coverage from 0.95 to 0.83. Repositories in Verified are far less homogeneous internally than
  GSM templates (variance share 0.03 vs ~0.8).
* Choosing repositories uniformly reweights them relative to the task-equal target, giving a −3 to −4 pp bias of the plain
  mean. Post-stratifying on difficulty removes most of it and lowers the SD to ~5.5 pp, so much of the repository effect
  runs through difficulty.
* The repository-cluster bootstrap is unusable with 1–4 repositories per benchmark (coverage 0.53 at 2×20, 0 at 1×40).
* Limits: only 12 repositories, the tiers draw from different eligible sets, one attempt per task. Submissions are not
  independent models (many share agents or base models); results are averaged, not pooled.
