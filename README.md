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

Everything below runs on a CPU from the saved per-problem counts. Only regenerating the counts needs a GPU (on AWS, see below).

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

## Running generation on AWS

The MATH-500 and GSM-Symbolic counts in `results/` were generated on AWS EC2 `g5.xlarge` GPU instances (1× A10G) in
`us-east-2`, provisioned with Terraform and running the public `vllm/vllm-openai:v0.10.0` image with this repo
installed from a code tarball in S3:

* MATH-500: 500 problems × 200 attempts = 100,000 generations;
* GSM-Symbolic: 5,000 problems × 64 attempts = 320,000 generations, after a 100-problem pilot.

Generation runs on EC2 spot GPUs and checkpoints to S3, so an interrupted job resumes where it stopped.

What Terraform (`infra/terraform/`) creates:

* a private, versioned, encrypted S3 bucket for code, partial results, logs and final counts;
* least-privilege IAM roles (EC2 worker limited to this bucket and to terminating instances tagged
  `Project=passk-smooth`; a SageMaker execution role);
* an ECR repository with scan-on-push (only for the optional `code_source = "ecr"` path);
* the GPU worker: Deep Learning Base AMI, IMDSv2 with hop limit 2 so the container reaches instance-role credentials,
  no inbound ports or SSH keys (Session Manager only).

The worker (`infra/terraform/user_data.sh.tftpl`) pulls earlier partial results from S3, syncs progress and the log
back every 10 minutes, runs `passk generate`, writes `EXIT_STATUS`, uploads, and terminates itself. Rerunning
`terraform apply` with the same `run_name` resumes a stopped run. `scripts/worker_install.sh` pins every package the vLLM image already ships, so installing this repo cannot
break the image's vllm / torch / transformers stack.

```bash
export AWS_PROFILE=<profile> AWS_REGION=us-east-2          # request G/VT spot quota for g5 first
terraform -chdir=infra/terraform init
terraform -chdir=infra/terraform apply -var-file=example.tfvars -var launch_generation=false   # bucket, IAM, ECR
scripts/upload_code.sh                                                                          # git HEAD -> S3

# pilot, then the full run (same run_name to resume after a spot interruption)
terraform -chdir=infra/terraform apply -var-file=example.tfvars \
  -var generate_config=configs/generate_gsm_symbolic_pilot.yaml -var run_name=gsm-symbolic-pilot
terraform -chdir=infra/terraform apply -var-file=example.tfvars \
  -var generate_config=configs/generate_gsm_symbolic.yaml -var run_name=gsm-symbolic

scripts/status.sh gsm-symbolic          # instance state, problems finished, vLLM progress, last log lines
scripts/fetch_results.sh gsm-symbolic   # S3 -> outputs/generate/gsm-symbolic/ (FULL=1 also pulls completions)
```

Other options (`infra/terraform/variables.tf`): `use_spot`, `instance_type`, `generate_overrides` (e.g.
`"limit=20 n_samples=16"`), `hf_token_ssm_parameter` for gated models, and `code_source = "ecr"` to run a prebuilt image
from `docker/Dockerfile.gpu` (`scripts/push_ecr.sh`). `infra/sagemaker/launch_generation.py` submits the same job as a
managed-spot SageMaker training job with S3 checkpointing (`--dry-run` prints the request); it was not used for the
runs above. Remove everything with `terraform -chdir=infra/terraform destroy` once the counts are saved.

## Layout

```
scripts/analysis/   deff_validation.py (main analysis), deff_robustness.py, export_gsm_splits.py,
                    template_tiers.py + plot_template_tiers.py (GSM tiers), swebench_repos.py (SWE-bench)
figures/            figure scripts for the arXiv version (arxiv/) and the workshop version (aaai/)
results/            saved counts, resolved matrix and every result file the paper reports
src/passk_smooth/   generation (vLLM / HF / mock backends), answer grading, unbiased pass@k estimator, intervals
configs/            generation configs
infra/              Terraform (S3, IAM, ECR, EC2 spot GPU worker) and a SageMaker launcher
docker/             GPU image for the ECR / SageMaker path
scripts/*.sh        upload code, run status, fetch results, worker install, push image
tests/              unit tests
```

## License

MIT (see `pyproject.toml`).
