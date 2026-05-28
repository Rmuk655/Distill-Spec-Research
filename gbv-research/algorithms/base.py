"""
Common interface for all speculative decoding algorithms.

Every algorithm in algorithms/<name>/ must implement SpecDecAlgorithm
and register an instance in ALGORITHM_REGISTRY below.

┌─────────────────────────────────────────────────────────────────┐
│  Algorithm taxonomy (from the literature)                       │
│                                                                 │
│  Draft source:                                                  │
│    INDEPENDENT  — separate small model (Leviathan, GBV)        │
│    ATTACHED     — extra heads on the target (Medusa)            │
│    FEATURE      — predict target's hidden states (EAGLE)        │
│    TREE_OPTIMAL — hardware-aware tree (Sequoia)                 │
│                                                                 │
│  Verification:                                                  │
│    TOKEN_LEVEL  — one token at a time (standard accept/reject)  │
│    TREE         — multiple draft paths verified together        │
│                                                                 │
│  Training:                                                      │
│    DISTRIBUTION — match token distribution (KL / EBE / JSD)    │
│    REGRESSION   — match hidden features (EAGLE)                 │
│    SFT_HEADS    — supervised fine-tune extra heads (Medusa)     │
│    NONE         — no training, use as-is                        │
└─────────────────────────────────────────────────────────────────┘

Adding a new algorithm (e.g. EAGLE-3):
  1. Create algorithms/eagle3/
  2. Implement SpecDecAlgorithm (see algorithms/eagle/ for an example)
  3. Register it: ALGORITHM_REGISTRY["eagle3"] = Eagle3Algorithm()
  4. See docs/ADDING_AN_ALGORITHM.md for the full checklist
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class DraftSource(Enum):
    INDEPENDENT  = auto()   # separate draft LLM
    ATTACHED     = auto()   # extra heads on target (Medusa)
    FEATURE      = auto()   # predict hidden states (EAGLE)
    TREE_OPTIMAL = auto()   # hardware-optimised tree (Sequoia)


class VerificationType(Enum):
    TOKEN_LEVEL = auto()    # token-by-token accept / reject
    TREE        = auto()    # tree of draft paths


class TrainingType(Enum):
    DISTRIBUTION = auto()   # KL / JSD / EBE — match token distribution
    REGRESSION   = auto()   # match hidden-state features (EAGLE)
    SFT_HEADS    = auto()   # supervised fine-tune extra heads (Medusa)
    NONE         = auto()   # no training required


@dataclass
class AlgorithmMeta:
    """Metadata for one algorithm implementation (used in the registry UI)."""
    name:             str               # short key, e.g. "gbv", "eagle"
    paper_title:      str               # human-readable paper name
    arxiv_id:         str               # e.g. "2401.15077"
    draft_source:     DraftSource
    verification:     VerificationType
    training:         TrainingType
    needs_draft_model: bool = True      # False for Medusa (heads live on target)
    notes:            str = ""          # any important caveats


@dataclass
class GenerationResult:
    """Unified output from any speculative decoding algorithm."""
    token_ids:          List[int]         # generated token IDs
    n_draft_tokens:     int               # total draft tokens proposed
    n_accepted_tokens:  int               # total draft tokens accepted
    n_target_calls:     int               # number of target model forward passes
    time_draft_s:       float = 0.0      # wall time inside draft model forward passes
    time_target_s:      float = 0.0      # wall time inside target model forward passes
    time_total_s:       float = 0.0      # total wall-clock time for the full generate() call
                                          # (includes sampling, tree ops, Python overhead)
                                          # Use for throughput: len(token_ids) / time_total_s
    extra:              Dict[str, Any] = field(default_factory=dict)

    @property
    def acceptance_rate(self) -> float:
        if self.n_draft_tokens == 0:
            return 0.0
        return self.n_accepted_tokens / self.n_draft_tokens

    @property
    def block_efficiency(self) -> float:
        """Average accepted tokens per target call (including the forced token)."""
        if self.n_target_calls == 0:
            return 0.0
        return (self.n_accepted_tokens + self.n_target_calls) / self.n_target_calls


class SpecDecAlgorithm(ABC):
    """
    Abstract base for all speculative decoding algorithms.

    Each algorithm directory implements this class.  The orchestration layer
    (orchestration/benchmark.py) only calls these methods — it never touches
    algorithm internals.

    Minimal implementation checklist:
      1. Set cls.meta with AlgorithmMeta
      2. Implement setup(), generate()
      3. If training is needed: implement train()
      4. If the algorithm uses a separate draft model: needs_draft_model=True
    """

    meta: AlgorithmMeta  # class-level, set by each subclass

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @abstractmethod
    def setup(
        self,
        target_model,
        tokenizer,
        device: str,
        draft_model=None,          # None for Medusa-style (heads live on target)
        model_family=None,         # ModelFamily instance for family-specific ops
        **kwargs,
    ) -> None:
        """
        Initialise the algorithm with the loaded models.

        Called once before any generate() or train() calls.
        Store references to models as instance attributes.

        Args:
            target_model:  Frozen large model (the "target" / verifier).
            tokenizer:     Shared tokenizer.
            device:        e.g. "cuda", "cpu".
            draft_model:   Small model for INDEPENDENT draft source.
                           None for ATTACHED / FEATURE source types.
            model_family:  ModelFamily instance providing family-specific
                           behaviour (LoRA modules, temperature recovery, etc.)
        """

    @abstractmethod
    def generate(
        self,
        prompt_ids,                # LongTensor [1, prompt_len]
        max_new_tokens: int = 128,
        K: int = 4,                # draft paths (tree width)
        L: int = 8,                # draft block length
        temperature: float = 1.0,
        **kwargs,
    ) -> GenerationResult:
        """
        Run speculative decoding on one prompt and return results.

        Args:
            prompt_ids:      Tokenised prompt, shape [1, prompt_len].
            max_new_tokens:  Maximum tokens to generate.
            K:               Number of independent draft paths (tree width).
                             Ignored by linear algorithms (e.g. Leviathan).
            L:               Draft block length (tokens proposed per target call).
            temperature:     Sampling temperature for the target model.

        Returns:
            GenerationResult with token_ids and profiling statistics.
        """

    # ── Training (optional) ──────────────────────────────────────────────────

    def train(
        self,
        dataset_path: str,
        output_dir: str,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Train the draft component of this algorithm.

        Override this method only if meta.training != TrainingType.NONE.
        Default implementation raises NotImplementedError for algorithms
        that declare they need training but haven't implemented it yet.

        Args:
            dataset_path:  Path to JSONL training file.
            output_dir:    Where to save the trained draft component.
            config:        Algorithm-specific hyperparameters (lr, steps, etc.)
        """
        if self.meta.training == TrainingType.NONE:
            print(f"[{self.meta.name}] No training needed for this algorithm.")
            return
        raise NotImplementedError(
            f"Algorithm '{self.meta.name}' declares training={self.meta.training} "
            "but has not implemented train(). "
            "See docs/ADDING_AN_ALGORITHM.md."
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__} "
                f"draft={self.meta.draft_source.name} "
                f"verify={self.meta.verification.name} "
                f"train={self.meta.training.name}>")
