"""Generation pipeline (mock backend), resume, and attempt subsampling."""

import json

import numpy as np
import pandas as pd
import pytest

from passk_smooth.config import GenerateConfig, load_config
from passk_smooth.experiments.common import subsample_counts, true_count_pmf, true_passk
from passk_smooth.generation.run import run_generation


def test_generation_mock_and_resume(tmp_path):
    cfg = GenerateConfig(backend="mock", dataset="toy", n_samples=40, batch_problems=5, output_dir=str(tmp_path), track=False)
    path = run_generation(cfg)
    df = pd.read_csv(path)
    assert len(df) == 12 and (df.n_samples == 40).all()
    assert df.n_correct.between(0, 40).all()
    # never-solvable mock problems get zero correct (grading path works)
    assert (df.n_correct == 0).any() and (df.n_correct > 0).any()
    lines_before = (tmp_path / "completions.jsonl").read_text().count("\n")
    run_generation(cfg)  # resume: nothing new generated
    assert (tmp_path / "completions.jsonl").read_text().count("\n") == lines_before


def test_resume_survives_truncated_last_line(tmp_path):
    cfg = GenerateConfig(backend="mock", dataset="toy", n_samples=20, batch_problems=4, output_dir=str(tmp_path), track=False)
    run_generation(cfg)
    comp = tmp_path / "completions.jsonl"
    lines = comp.read_text().splitlines()
    # simulate a spot interruption mid-write: last two problems lost, one line half-written
    comp.write_text("\n".join(lines[:-2]) + "\n" + lines[-2][: len(lines[-2]) // 2])
    df = pd.read_csv(run_generation(cfg))
    assert len(df) == len(lines) == 12
    assert all(json.loads(x) for x in comp.read_text().splitlines())


def test_hypergeometric_subsampling_keeps_chen_unbiased(rng):
    c = rng.integers(0, 201, 300)
    truth = true_passk(c, 200, np.array([3]))[0]
    w = 1 - np.array([np.prod([(5 - x - i) / (5 - i) for i in range(3)]) if x <= 2 else 0 for x in range(6)])
    est = np.mean([w[subsample_counts(rng, c, 200, 5)].mean() for _ in range(3000)])
    assert est == pytest.approx(truth, abs=2e-3)
    assert true_count_pmf(c, 200, 5).sum() == pytest.approx(1.0)


def test_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("n_samples: 5\nbackend: mock\n")
    cfg = load_config(p, GenerateConfig, ["n_samples=7", "sampling.temperature=0.3"])
    assert cfg.n_samples == 7 and cfg.sampling.temperature == 0.3
