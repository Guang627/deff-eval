#!/usr/bin/env bash
set -euo pipefail
RUN="${1:-qwen2.5-math-1.5b}"
BUCKET="$(terraform -chdir=infra/terraform output -raw bucket)"
EXTRA=(--exclude "checkpoints/*")
[ "${FULL:-0}" = "1" ] || EXTRA+=(--exclude "completions.jsonl")
aws s3 sync "s3://$BUCKET/runs/$RUN/" "outputs/generate/$RUN/" "${EXTRA[@]}"
echo "outputs/generate/$RUN/counts.csv"
