from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


class _NoOp:
    def log_params(self, *_: Any, **__: Any) -> None: ...
    def log_metrics(self, *_: Any, **__: Any) -> None: ...
    def log_artifacts(self, *_: Any, **__: Any) -> None: ...
    def log_artifact(self, *_: Any, **__: Any) -> None: ...
    def set_tags(self, *_: Any, **__: Any) -> None: ...


@contextlib.contextmanager
def run(experiment: str, run_name: str, config: dict[str, Any], enabled: bool = True) -> Iterator[Any]:
    try:
        import mlflow
    except Exception:
        mlflow = None  # type: ignore[assignment]
    if not enabled or mlflow is None or os.environ.get("PASSK_DISABLE_TRACKING"):
        yield _NoOp()
        return
    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{Path('mlflow.db').resolve()}"))
        mlflow.set_experiment(experiment)
        ctx = mlflow.start_run(run_name=run_name)
    except Exception as e:
        logging.getLogger(__name__).warning("MLflow unavailable (%s); tracking disabled", e)
        yield _NoOp()
        return
    with ctx:
        mlflow.set_tags({"git_sha": _git_sha()})
        flat = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in config.items()}
        mlflow.log_params({k: str(v)[:500] for k, v in flat.items()})

        class _Run:
            def log_params(self, p: dict[str, Any]) -> None:
                mlflow.log_params(p)

            def log_metrics(self, m: dict[str, float], step: int | None = None) -> None:
                mlflow.log_metrics({k.replace("@", "_at_"): float(v) for k, v in m.items()}, step=step)

            def log_artifacts(self, d: str | Path) -> None:
                mlflow.log_artifacts(str(d))

            def log_artifact(self, p: str | Path) -> None:
                mlflow.log_artifact(str(p))

            def set_tags(self, t: dict[str, str]) -> None:
                mlflow.set_tags(t)

        yield _Run()
