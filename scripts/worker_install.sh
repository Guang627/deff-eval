#!/bin/bash
set -euo pipefail
SRC="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PYTHON:-python3}"
CONSTRAINTS="$(mktemp)"
"$PY" -m pip list --format=freeze 2>/dev/null | grep -E '^[A-Za-z0-9_.-]+==' > "$CONSTRAINTS"
echo "worker_install: pinning $(wc -l < "$CONSTRAINTS") preinstalled packages"

"$PY" -m pip install --no-cache-dir -c "$CONSTRAINTS" "$SRC[s3]" 'datasets>=2.18'
"$PY" -m pip install --no-cache-dir -c "$CONSTRAINTS" 'math-verify>=0.7' \
  || echo "worker_install: math-verify unavailable under the image's pins; using the built-in grader"

"$PY" -m pip check || echo "worker_install: pip check reports issues (see above); continuing"
"$PY" - <<'PY'
import importlib
for m in ("vllm", "transformers", "huggingface_hub", "datasets", "boto3", "passk_smooth"):
    mod = importlib.import_module(m)
    print(f"worker_install: {m} {getattr(mod, '__version__', '?')}")
PY
