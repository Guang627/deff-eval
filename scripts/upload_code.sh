#!/usr/bin/env bash
set -euo pipefail
DEST="$(terraform -chdir=infra/terraform output -raw code_s3_uri)"
TMP="$(mktemp -d)/passk-smooth.tar.gz"
git archive --format=tar.gz -o "$TMP" HEAD
aws s3 cp "$TMP" "$DEST"
echo "uploaded $(git rev-parse --short HEAD) -> $DEST"
