"""
EaglePredictor — the lightweight autoregressive model at the heart of EAGLE.

IMPLEMENT ME:
  Fill in forward() per Section 2 of arXiv:2401.15077.

Architecture (from the paper):
  input  = concat([h_t, embed(token_t)], dim=-1)  # [B, 1, hidden_size * 2]
  output = h_{t+1}                                 # [B, 1, hidden_size]

  The predictor is a small transformer (1-2 layers) with:
    - Input projection: Linear(hidden_size * 2, hidden_size)
    - Self-attention layer(s) with causal mask
    - Feed-forward block
    - Output: the new hidden state estimate, NOT logits

  Logits are obtained separately by calling  target.lm_head(h_{t+1}).

Reference:
  https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/ea_model.py
  Class `EaModel` — pay attention to `ea_layer` and `fc` (the input projection).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EaglePredictor(nn.Module):
    """
    Lightweight predictor that maps [h_t || embed(token_t)] → h_{t+1}.

    The target LM head is weight-tied externally (passed in or set via
    `self.lm_head = target.lm_head`).  This module only predicts hidden states.

    Args:
        hidden_size:  Hidden dimension of the target LM.
        num_layers:   Number of transformer layers in the predictor (default 1).
        num_heads:    Number of attention heads (default same as target, or fewer).
    """

    def __init__(self, hidden_size: int, num_layers: int = 1, num_heads: int = 8):
        super().__init__()
        # TODO: replace with the exact architecture from the EAGLE reference repo.
        # The reference uses a single Transformer decoder layer on top of a
        # linear input projection that fuses h_t and embed(token_t).
        self.input_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        # Placeholder — real impl uses transformer layers from the paper
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_size, nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        hidden_state: torch.Tensor,   # [B, seq_len, hidden_size]
        token_embeds: torch.Tensor,   # [B, seq_len, hidden_size]
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_state:   Target LM's last-layer hidden states for positions 0..t.
            token_embeds:   Target LM's token embeddings for the same positions.
            attention_mask: Optional causal mask for variable-length sequences.

        Returns:
            h_pred: [B, seq_len, hidden_size] — predicted next hidden states.

        TODO:
          x = self.input_proj(torch.cat([hidden_state, token_embeds], dim=-1))
          for layer in self.layers:
              x = layer(x, x, tgt_mask=attention_mask)
          return x
        """
        raise NotImplementedError(
            "EaglePredictor.forward() not yet implemented. "
            "See algorithms/eagle/predictor.py for the implementation guide."
        )
