"""
Medusa stub — Cai et al. 2024 (arXiv:2401.10774).

IMPLEMENT ME:
  Fill in setup(), generate(), and train() following the paper.
  The MedusaHead module stub is in heads.py.

Key architectural facts for implementation:
  - draft_source = ATTACHED  (heads live on the target — no draft_model arg)
  - needs_draft_model = False
  - K extra heads, each predicts 1 future token
  - One target forward pass gives K+1 distributions simultaneously
  - Build a candidate tree from heads: exponential candidates but
    one target call per tree
  - Training: freeze the target LM, train only the heads on next-token
    prediction from the target's own hidden states
"""

from __future__ import annotations

from algorithms.base import (
    SpecDecAlgorithm, AlgorithmMeta, GenerationResult,
    DraftSource, VerificationType, TrainingType,
)


class MedusaAlgorithm(SpecDecAlgorithm):

    meta = AlgorithmMeta(
        name              = "medusa",
        paper_title       = "Medusa: Simple LLM Inference via Multiple Decoding Heads",
        arxiv_id          = "2401.10774",
        draft_source      = DraftSource.ATTACHED,
        verification      = VerificationType.TREE,
        training          = TrainingType.SFT_HEADS,
        needs_draft_model = False,   # heads live on the target
        notes             = "Cai et al. 2024.  No separate draft model. "
                            "K heads attached to target LM. "
                            "Reference impl: github.com/FasterDecoding/Medusa",
    )

    def setup(self, target_model, tokenizer, device,
              draft_model=None, model_family=None, **kwargs) -> None:
        """
        TODO: Attach Medusa heads to the target model.

        Steps:
          1. Load or initialise K MedusaHead modules (see heads.py)
          2. Attach them to target_model (or load pre-trained from checkpoint)
          3. Store self.target, self.heads, self.tok, self.device
        """
        raise NotImplementedError(
            "MedusaAlgorithm.setup() not yet implemented. "
            "See algorithms/medusa/algorithm.py for the implementation guide."
        )

    def generate(self, prompt_ids, max_new_tokens=128, K=4, L=8,
                 temperature=1.0, **kwargs) -> GenerationResult:
        """
        TODO: Implement Medusa generation.

        Steps (from the paper, Section 3):
          1. Run target forward pass → get hidden state h_t
          2. Each head i produces: logits_i = head_i(h_t)  [V]
          3. For each head: sample top-T candidates → candidate tree
          4. One more target forward pass verifies the full tree
          5. Accept the longest consistent prefix
        """
        raise NotImplementedError(
            "MedusaAlgorithm.generate() not yet implemented."
        )

    def train(self, dataset_path, output_dir, config=None, **kwargs) -> None:
        """
        TODO: Train Medusa heads.

        Steps:
          1. Freeze the target LM (all params require_grad=False)
          2. Collect hidden states h_t for all tokens in dataset
          3. For each head i: minimise CE(head_i(h_t), token_{t+i+1})
          4. Save heads to output_dir/medusa_heads.pt
        """
        raise NotImplementedError(
            "MedusaAlgorithm.train() not yet implemented."
        )
