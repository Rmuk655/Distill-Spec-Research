"""
GBVAlgorithm — SpecDecAlgorithm implementation for DistillSpec + GBV.

Glues together:
  - verifiers/runner.py  (speculative decoding loop with OTLP tree verification)
  - trainer.py           (knowledge distillation training)
  - losses/              (pluggable loss objectives)
  - core/model_families  (family-specific LoRA, temperature, masking)
"""

from __future__ import annotations

import time
from typing import Optional

import torch

from algorithms.base import (
    SpecDecAlgorithm, AlgorithmMeta, GenerationResult,
    DraftSource, VerificationType, TrainingType,
)


class GBVAlgorithm(SpecDecAlgorithm):
    """
    DistillSpec + GBV speculative decoding.

    Draft model: independent small LLM (Qwen2.5-0.5B by default).
    Verification: OTLP tree (gbv | traversal | specinfer | naive | ...).
    Training: KL / EBE / JSD / L1 / reverse-KL distillation.

    Supported verifier names (pass as verifier= to generate()):
        "traversal"  — bottom-up tree traversal (best empirically)
        "gbv"        — greedy best-of-K
        "specinfer"  — SpecInfer OTLP (Leviathan et al.)
        "naive"      — baseline token-level
        "bv"         — basic tree verification
        "nss"        — non-stochastic sampling
        "spectr"     — SpecTr OT-based
        "khisti"     — canonical decomposition
    """

    meta = AlgorithmMeta(
        name              = "gbv",
        paper_title       = "Generalized Block Verification for Speculative Decoding",
        arxiv_id          = "2602.16994",
        draft_source      = DraftSource.INDEPENDENT,
        verification      = VerificationType.TREE,
        training          = TrainingType.DISTRIBUTION,
        needs_draft_model = True,
        notes             = "Thomas et al. 2026. Traversal verifier outperforms OT-based "
                            "methods by ~15% block efficiency on GSM8K.",
    )

    def setup(
        self,
        target_model,
        tokenizer,
        device: str,
        draft_model=None,
        model_family=None,
        **kwargs,
    ) -> None:
        if draft_model is None:
            raise ValueError(
                "GBV requires a draft_model (separate small LLM). "
                "Pass draft_model= to setup()."
            )
        self.target_model  = target_model
        self.draft_model   = draft_model
        self.tokenizer     = tokenizer
        self.device        = device
        self.model_family  = model_family

    def generate(
        self,
        prompt_ids,
        max_new_tokens: int = 128,
        K: int = 4,
        L: int = 8,
        temperature: float = 1.0,
        verifier: str = "traversal",   # GBV-specific: which OTLP verifier to use
        **kwargs,
    ) -> GenerationResult:
        """
        Run GBV speculative decoding with the specified OTLP verifier.

        Extra kwargs:
            verifier: "traversal" | "gbv" | "specinfer" | "naive" | ...
        """
        from .verifiers.runner import speculative_decoding_loop

        prompt = self.tokenizer.decode(prompt_ids[0], skip_special_tokens=False)

        t0 = time.perf_counter()
        result_tokens, stats = speculative_decoding_loop(
            p_model           = self.target_model,
            q_model           = self.draft_model,
            tok               = self.tokenizer,
            prompt            = prompt,
            verification_algo = verifier,
            max_new_tokens    = max_new_tokens,
            K                 = K,
            L                 = L,
            p_temp            = temperature,
            q_temp            = temperature,
        )
        elapsed = time.perf_counter() - t0

        return GenerationResult(
            token_ids         = result_tokens,
            n_draft_tokens    = stats.get("n_draft_tokens", 0),
            n_accepted_tokens = stats.get("n_accepted_tokens", 0),
            n_target_calls    = stats.get("n_target_calls", 0),
            time_draft_s      = stats.get("time_draft", 0.0),
            time_target_s     = stats.get("time_target", 0.0),
            extra             = {"verifier": verifier, "K": K, "L": L},
        )

    def train(
        self,
        dataset_path: str,
        output_dir: str,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Run KL / EBE / JSD / L1 / reverse-KL distillation training.

        Delegates to trainer.py in this directory.

        Args:
            dataset_path: JSONL file with 'prompt' field.
            output_dir:   Where to save LoRA adapter.
            config:       Dict with keys matching trainer.py parse_args()
                          (loss, steps, lr, lora_r, etc.)
        """
        from .trainer import main as _train_main
        import sys

        if config is None:
            config = {}

        # Build argv for trainer.py's argparse
        argv = [
            "--dataset",   dataset_path,
            "--output",    output_dir,
            "--loss",      config.get("loss", "forward_kl"),
            "--steps",     str(config.get("steps", 1000)),
            "--lr",        str(config.get("lr", 3e-5)),
            "--model_family", config.get("model_family", "qwen"),
        ]
        if "draft" in config:
            argv += ["--draft", config["draft"]]
        if "target" in config:
            argv += ["--target", config["target"]]
        if config.get("no_wandb"):
            argv.append("--no_wandb")

        old_argv = sys.argv
        sys.argv = ["trainer.py"] + argv
        try:
            _train_main()
        finally:
            sys.argv = old_argv
