from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIXTURE = Path(__file__).with_name("fixtures") / "toy_math.jsonl"


def gsm_final_answer(answer: str) -> str:
    tail = answer.rsplit("####", 1)[-1] if "####" in answer else answer
    return tail.strip().replace(",", "").replace("$", "").rstrip(".")


def _is_gsm_symbolic(r: dict[str, Any]) -> bool:
    return "question" in r and "instance" in r


def _normalise_gsm_symbolic(r: dict[str, Any], config: str) -> dict[str, Any]:
    tid, inst = int(r["id"]), int(r["instance"])
    return {
        "problem_id": f"{config}/t{tid:03d}/i{inst:02d}",
        "problem": r["question"],
        "answer": gsm_final_answer(str(r["answer"])),
        "subject": f"t{tid:03d}",
        "level": config,
        "source": str(r.get("original_id", f"t{tid:03d}")),
        "instance": inst,
    }


def _normalise_record(r: dict[str, Any], i: int, config: str = "main") -> dict[str, Any]:
    if _is_gsm_symbolic(r):
        return _normalise_gsm_symbolic(r, str(r.get("config", config)))
    return {
        "problem_id": str(r.get("unique_id") or r.get("problem_id") or r.get("id") or i),
        "problem": r["problem"],
        "answer": str(r["answer"]),
        "subject": str(r.get("subject") or r.get("type") or "unknown"),
        "level": str(r.get("level", "unknown")),
    }


def load_problems(
    source: str,
    split: str = "test",
    limit: int | None = None,
    seed: int = 0,
    configs: list[str] | None = None,
    variants_per_template: int | None = None,
) -> list[dict[str, Any]]:
    if source == "toy":
        source = str(FIXTURE)
    out: list[dict[str, Any]] = []
    if source.endswith(".jsonl") or Path(source).exists():
        with open(source) as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        out = [_normalise_record(r, i) for i, r in enumerate(rows)]
    else:  # pragma: no cover
        from datasets import load_dataset

        cfg_names: list[str | None] = list(configs) if configs else [None]
        for cfg_name in cfg_names:
            ds = load_dataset(source, cfg_name, split=split) if cfg_name else load_dataset(source, split=split)
            out += [_normalise_record(r, i, cfg_name or "main") for i, r in enumerate(ds)]
    if variants_per_template is not None:
        out = [p for p in out if "instance" not in p or p["instance"] < variants_per_template]
    if limit is not None and limit < len(out):
        import random

        rnd = random.Random(seed)
        out = sorted(rnd.sample(out, limit), key=lambda r: r["problem_id"])
    return out


def cluster_label(p: dict[str, Any], scheme: str = "subject_level") -> str:
    if scheme == "subject":
        return p["subject"]
    if scheme == "subject_level":
        return f"{p['subject']}|L{p['level']}"
    if scheme == "none":
        return p["problem_id"]
    raise ValueError(f"unknown cluster scheme {scheme!r}")
