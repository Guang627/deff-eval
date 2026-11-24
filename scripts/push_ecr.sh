#!/usr/bin/env bash
set -euo pipefail
TAG="${1:-gpu-latest}"
REGION="${AWS_REGION:-us-east-1}"
REPO_URL="$(terraform -chdir=infra/terraform output -raw ecr_repository_url)"
REGISTRY="${REPO_URL%%/*}"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
docker build -f docker/Dockerfile.gpu -t "$REPO_URL:$TAG" -t "$REPO_URL:$(git rev-parse --short HEAD)" .
docker push "$REPO_URL:$TAG"
docker push "$REPO_URL:$(git rev-parse --short HEAD)"
echo "pushed $REPO_URL:$TAG"
