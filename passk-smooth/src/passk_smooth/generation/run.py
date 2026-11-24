"""Generate N attempts per problem, grade them, and write per-problem counts.

Outputs (in ``output_dir``):
  completions.jsonl  one line per problem: id, completions, correctness (optional)
  counts.csv         problem_id, subject, level, cluster, n_samples, n_correct
  meta.json          config, timing, model, git sha
Resumable: problems already in completions.jsonl are skipped.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from .. import storage, tracking
from ..config import GenerateConfig
from ..data.grading import grade
from ..data.problems import cluster_label, load_problems
from .backends import make_backend

log = logging.getLogger(__name__)


def _load_done(comp_path: Path) -> dict[str, dict]:
    """Read finished problems; drop a half-written line left by an interrupted run.

    The progress sync copies completions.jsonl to S3 while it is being appended to, so a spot
    interruption can leave a truncated last line. Unreadable lines are discarded (those problems
    are simply generated again) and the file is rewritten so new lines append cleanly.
    """
    done: dict[str, dict] = {}
    good: list[str] = []
    bad = 0
    for line in comp_path.read_text().splitlines():
        try:
            r = json.loads(line)
            done[r["problem_id"]] = r
            good.append(line)
        except (json.JSONDecodeError, KeyError, TypeError):
            bad += 1
    if bad:
        log.warning("dropped %d unreadable line(s) from %s (interrupted write); regenerating them", bad, comp_path)
        comp_path.write_text("".join(g + "\n" for g in good))
    return done


def run_generation(cfg: GenerateConfig) -> Path:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    comp_path = out / "completions.jsonl"
    done: dict[str, dict] = {}
    if comp_path.exists():
        done = _load_done(comp_path)
    problems = load_problems(
        cfg.dataset, cfg.split, cfg.limit, configs=cfg.dataset_configs, variants_per_template=cfg.variants_per_template
    )
    todo = [p for p in problems if p["problem_id"] not in done]
    log.info("%d problems, %d already done", len(problems), len(problems) - len(todo))

    t0 = time.time()
    with tracking.run("passk-generate", f"{cfg.backend}:{cfg.model}", cfg.model_dump(), cfg.track) as tr:
        backend = make_backend(cfg) if todo else None
        with comp_path.open("a") as fh:
            for start in range(0, len(todo), cfg.batch_problems):
                batch = todo[start : start + cfg.batch_problems]
                comps = backend.generate(batch, cfg.n_samples)  # type: ignore[union-attr]
                for p, cs in zip(batch, comps, strict=True):
                    correct = [grade(c, p["answer"], cfg.use_math_verify) for c in cs]
                    rec = {"problem_id": p["problem_id"], "correct": correct}
                    if cfg.keep_completions:
                        rec["completions"] = cs
                    fh.write(json.dumps(rec) + "\n")
                    done[p["problem_id"]] = rec
                fh.flush()
                tr.log_metrics({"problems_done": len(done)}, step=len(done))
                log.info("graded %d / %d", len(done), len(problems))

        rows = []
        for p in problems:
            c = done[p["problem_id"]]["correct"]
            rows.append(
                {
                    "problem_id": p["problem_id"],
                    "subject": p["subject"],
                    "level": p["level"],
                    "cluster": cluster_label(p, "subject_level"),
                    "n_samples": len(c),
                    "n_correct": int(sum(c)),
                    **({"source": p["source"]} if "source" in p else {}),
                }
            )
        df = pd.DataFrame(rows)
        df.to_csv(out / "counts.csv", index=False)
        meta = {
            "config": cfg.model_dump(),
            "seconds": time.time() - t0,
            "git_sha": tracking._git_sha(),
            "mean_pass1": float((df.n_correct / df.n_samples).mean()),
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        tr.log_metrics({"pass_at_1": meta["mean_pass1"], "frac_never_solved": float((df.n_correct == 0).mean())})
        tr.log_artifact(out / "counts.csv")
        tr.log_artifact(out / "meta.json")
    if cfg.s3_uri:
        storage.sync_dir_to_s3(out, cfg.s3_uri)
        log.info("uploaded to %s", cfg.s3_uri)
    return out / "counts.csv"
