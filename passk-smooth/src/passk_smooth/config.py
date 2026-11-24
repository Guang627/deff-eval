"""Typed, validated experiment configs loaded from YAML (see ``configs/``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class SamplingConfig(BaseModel):
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 2048
    seed: int = 0


class GenerateConfig(BaseModel):
    """Ground-truth generation: N attempts per problem, graded."""

    backend: Literal["vllm", "hf", "mock"] = "vllm"
    model: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    dataset: str = "HuggingFaceH4/MATH-500"
    split: str = "test"
    dataset_configs: list[str] | None = None  # e.g. [main, p1, p2] for apple/GSM-Symbolic
    variants_per_template: int | None = None  # GSM-Symbolic: keep instances 0..V-1 of every template
    limit: int | None = None
    n_samples: int = 200
    batch_problems: int = 50
    system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}."
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    keep_completions: bool = True
    use_math_verify: bool = True
    output_dir: str = "outputs/generate"
    s3_uri: str | None = None  # e.g. s3://my-bucket/passk/qwen1.5b
    mock_seed: int = 0
    track: bool = True


def load_config(path: str | Path, model: type[BaseModel], overrides: list[str] | None = None) -> Any:
    """Load YAML into ``model``; ``overrides`` are ``key=value`` (dotted keys ok, value parsed as YAML)."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        node = data
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return model.model_validate(data)
