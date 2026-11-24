"""Tiny local/S3 storage layer. Paths starting with ``s3://`` go through boto3."""

from __future__ import annotations

import os
from pathlib import Path


def is_s3(uri: str) -> bool:
    return uri.startswith("s3://")


def split_s3(uri: str) -> tuple[str, str]:
    rest = uri[len("s3://") :]
    bucket, _, key = rest.partition("/")
    return bucket, key


def _client():
    import boto3

    return boto3.client("s3", region_name=os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION")))


def upload_file(local: str | Path, uri: str) -> None:
    bucket, key = split_s3(uri)
    _client().upload_file(str(local), bucket, key)


def download_file(uri: str, local: str | Path) -> Path:
    bucket, key = split_s3(uri)
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(bucket, key, str(local))
    return Path(local)


def sync_dir_to_s3(local_dir: str | Path, uri: str) -> list[str]:
    """Upload every file under ``local_dir`` to the ``uri`` prefix. Returns uploaded keys."""
    bucket, prefix = split_s3(uri.rstrip("/") + "/")
    cli = _client()
    keys = []
    for p in sorted(Path(local_dir).rglob("*")):
        if p.is_file():
            key = prefix + p.relative_to(local_dir).as_posix()
            cli.upload_file(str(p), bucket, key)
            keys.append(key)
    return keys


def localize(uri: str, cache_dir: str | Path = ".cache/s3") -> Path:
    """Return a local path for ``uri``, downloading from S3 if needed."""
    if not is_s3(uri):
        return Path(uri)
    bucket, key = split_s3(uri)
    return download_file(uri, Path(cache_dir) / bucket / key)
