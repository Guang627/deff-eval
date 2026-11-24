#!/usr/bin/env bash
set -euo pipefail
RUN="${1:-qwen2.5-math-1.5b}"
BUCKET="$(terraform -chdir=infra/terraform output -raw bucket)"
echo "== instances tagged passk-smooth (not terminated)"
aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=passk-smooth" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query "Reservations[].Instances[].[InstanceId,InstanceType,State.Name,LaunchTime,InstanceLifecycle]" --output table
echo "== problems finished (synced every 10 min)"
if aws s3 ls "s3://$BUCKET/runs/$RUN/completions.jsonl" >/dev/null 2>&1; then
  aws s3 cp "s3://$BUCKET/runs/$RUN/completions.jsonl" - | wc -l
else
  echo "0 (nothing synced yet)"
fi
aws s3 cp "s3://$BUCKET/runs/$RUN/EXIT_STATUS" - 2>/dev/null | sed 's/^/exit status: /' || echo "still running (no EXIT_STATUS yet)"
LOG="$(mktemp)"
aws s3 cp "s3://$BUCKET/runs/$RUN/logs/latest.log" "$LOG" --only-show-errors 2>/dev/null || true
echo "== current batch (vLLM progress bar; a batch is batch_problems problems, 50 by default)"
tr '\r' '\n' < "$LOG" | grep -a "Processed prompts" | tail -n 1 || echo "(no progress bar yet: model still loading, or log not synced)"
echo "== last log lines (repeated vLLM warnings hidden)"
if [ -s "$LOG" ]; then
  tr '\r' '\n' < "$LOG" | grep -av -e "per-request generators" -e "Processed prompts" | tail -n 15 || true
else
  echo "(no log yet; first upload ~10 min after boot)"
fi
rm -f "$LOG"
