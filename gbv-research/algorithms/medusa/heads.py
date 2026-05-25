"""
MedusaHead — one extra prediction head attached to the target LM.

Each head is a shallow MLP that maps target hidden states → token logits.
K heads are created; head i predicts token at position t+i+1.

IMPLEMENT ME:
  Fill in forward() as described in the paper (Section 3, Eq. 1-2):
    h_t → [Linear(hidden, hidden, bias=False)] → SiLU → [Linear(hidden, V)]
  The default depth=1 (one residual block) matches the reference implementation.

Reference:
  https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/medusa_model.py
  Look at `MedusaModel.build_medusa_head()` for the exact architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MedusaHead(nn.Module):
    """
    A single Medusa prediction head.

    Architecture (paper default, depth=1):
        h_t  →  Linear(hidden_size → hidden_size, bias=False)
             →  SiLU
             →  Linear(hidden_size → vocab_size)

    For depth > 1, stack multiple [Linear → SiLU] blocks before the final head.

    Args:
        hidden_size: Hidden dimension of the target LM (e.g. 2048 for Qwen3-0.6B).
        vocab_size:  Vocabulary size (e.g. 151936 for Qwen).
        depth:       Number of Linear+SiLU blocks (paper uses 1).
    """

    def __init__(self, hidden_size: int, vocab_size: int, depth: int = 1):
        super().__init__()
        # TODO: replace stub with the exact architecture from 2401.10774 / reference repo.
        blocks = []
        for _ in range(depth):
            blocks.append(nn.Linear(hidden_size, hidden_size, bias=False))
            blocks.append(nn.SiLU())
        self.blocks  = nn.Sequential(*blocks)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: float tensor [batch, seq_len, hidden_size]
                          — the target LM's last-layer hidden states.

        Returns:
            logits: float tensor [batch, seq_len, vocab_size]
                    — unnormalised token log-scores for position t+i+1.

        TODO:
          return self.lm_head(self.blocks(hidden_state))
        """
        raise NotImplementedError(
            "MedusaHead.forward() not yet implemented. "
            "See algorithms/medusa/heads.py — uncomment the one-liner in the TODO."
        )
