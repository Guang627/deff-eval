"""Generation backends. ``vllm`` for GPU runs, ``hf`` for small CPU checks,
``mock`` for tests and CI (no model, no network)."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..config import GenerateConfig


class Backend(ABC):
    def __init__(self, cfg: GenerateConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def generate(self, problems: list[dict[str, Any]], n: int) -> list[list[str]]:
        """Return n completions for each problem."""


def build_messages(cfg: GenerateConfig, problem: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": cfg.system_prompt}, {"role": "user", "content": problem}]


class VLLMBackend(Backend):  # pragma: no cover - needs a GPU
    def __init__(self, cfg: GenerateConfig) -> None:
        from vllm import LLM, SamplingParams

        self.cfg = cfg
        self.llm = LLM(
            model=cfg.model,
            tensor_parallel_size=cfg.tensor_parallel_size,
            dtype=cfg.dtype,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            seed=cfg.sampling.seed,
        )
        self.tok = self.llm.get_tokenizer()
        self.SamplingParams = SamplingParams

    def generate(self, problems: list[dict[str, Any]], n: int) -> list[list[str]]:
        s = self.cfg.sampling
        params = self.SamplingParams(n=n, temperature=s.temperature, top_p=s.top_p, max_tokens=s.max_tokens, seed=s.seed)
        prompts = [
            self.tok.apply_chat_template(build_messages(self.cfg, p["problem"]), tokenize=False, add_generation_prompt=True)
            for p in problems
        ]
        outs = self.llm.generate(prompts, params, use_tqdm=True)
        return [[c.text for c in o.outputs] for o in outs]


class HFBackend(Backend):  # pragma: no cover - needs model weights
    """transformers.generate; slow, intended for a handful of problems on CPU."""

    def __init__(self, cfg: GenerateConfig) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg, self.torch = cfg, torch
        self.tok = AutoTokenizer.from_pretrained(cfg.model)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.float32)
        torch.manual_seed(cfg.sampling.seed)

    def generate(self, problems: list[dict[str, Any]], n: int) -> list[list[str]]:
        s, out = self.cfg.sampling, []
        for p in problems:
            msgs = build_messages(self.cfg, p["problem"])
            text = (
                self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                if self.tok.chat_template
                else p["problem"]
            )
            ids = self.tok(text, return_tensors="pt")
            gen = self.model.generate(
                **ids,
                do_sample=True,
                temperature=s.temperature,
                top_p=s.top_p,
                max_new_tokens=s.max_tokens,
                num_return_sequences=n,
                pad_token_id=self.tok.eos_token_id,
            )
            out.append(self.tok.batch_decode(gen[:, ids["input_ids"].shape[1] :], skip_special_tokens=True))
        return out


class MockBackend(Backend):
    """Pretend model: problem-specific success probability from a hash of its id.

    Emits "\\boxed{gold}" with probability p and a wrong boxed answer otherwise,
    so the grading and aggregation path is exercised end to end.
    """

    def __init__(self, cfg: GenerateConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.mock_seed)

    @staticmethod
    def latent_p(problem_id: str) -> float:
        h = int(hashlib.sha256(problem_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return 0.0 if h < 0.15 else (1.0 if h > 0.92 else h)

    def generate(self, problems: list[dict[str, Any]], n: int) -> list[list[str]]:
        out = []
        for p in problems:
            q = self.latent_p(p["problem_id"])
            ok = self.rng.random(n) < q
            out.append(
                [
                    f"Step-by-step reasoning... so the answer is \\boxed{{{p['answer']}}}."
                    if o
                    else "Some reasoning... \\boxed{\\text{wrong}}"
                    for o in ok
                ]
            )
        return out


def make_backend(cfg: GenerateConfig) -> Backend:
    backends: dict[str, type[Backend]] = {"vllm": VLLMBackend, "hf": HFBackend, "mock": MockBackend}
    return backends[cfg.backend](cfg)
