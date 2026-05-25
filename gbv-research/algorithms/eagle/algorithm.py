"""
EAGLE stub — Li et al. 2024 (arXiv:2401.15077).

IMPLEMENT ME:
  Fill in setup(), generate(), and train() following the paper.
  The EaglePredictor module stub is in predictor.py.

Key architectural facts for implementation:
  - draft_source = FEATURE
      The EAGLE predictor is NOT an independent language model.
      Its input is [target_h_t || target_embed(token_t)] — it consumes the
      target's own hidden states, so we need a forward-hook on the target.
  - needs_draft_model = True
      `draft_model` should be the path to a pre-trained EaglePredictor checkpoint,
      or None to initialise randomly (for training from scratch).
  - training = REGRESSION
      Objective: MSE(predictor([h_{t-1}||e_{t-1}]), h_t.detach())
      Optionally add CE loss on LM_head(predicted_h) vs token_t.
  - verification = TREE
      Build a draft token tree using the K predicted tokens / beams;
      verify with one target forward pass using tree attention masks.

Step-by-step generate() (paper Section 3 + Algorithm 1):
  1.  Run target forward on context → capture h_t from the last transformer layer.
  2.  For i = 1 … K:
        a.  predictor_input = concat([h_{t+i-1}, embed(token_{t+i-1})], dim=-1)
        b.  h_{t+i} = predictor(predictor_input)
        c.  draft_logits_i = target.lm_head(h_{t+i})
        d.  sample draft_token_i from draft_logits_i
  3.  Build token tree from (draft_token_1, …, draft_token_K) with branching.
  4.  Construct tree attention mask; run one target forward over the tree.
  5.  Walk tree accepting the longest consistent prefix (rejection sampling).
  6.  Append accepted tokens to context; repeat.
"""

from __future__ import annotations

from algorithms.base import (
    SpecDecAlgorithm, AlgorithmMeta, GenerationResult,
    DraftSource, VerificationType, TrainingType,
)


class EagleAlgorithm(SpecDecAlgorithm):

    meta = AlgorithmMeta(
        name              = "eagle",
        paper_title       = "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty",
        arxiv_id          = "2401.15077",
        draft_source      = DraftSource.FEATURE,
        verification      = VerificationType.TREE,
        training          = TrainingType.REGRESSION,
        needs_draft_model = True,   # "draft model" = EaglePredictor checkpoint
        notes             = "Li et al. 2024.  Predictor maps [h_t || embed(t)] → h_{t+1}; "
                            "target LM head shared for token sampling. "
                            "Reference impl: github.com/SafeAILab/EAGLE",
    )

    def setup(self, target_model, tokenizer, device,
              draft_model=None, model_family=None, **kwargs) -> None:
        """
        TODO: Wire EAGLE predictor to the target model.

        Steps:
          1. Load or initialise EaglePredictor from predictor.py.
          2. Share the target's LM head with the predictor (weight-tie).
          3. Register a forward hook on the target's last transformer layer
             to capture h_t during generate() without a second forward pass.
          4. Store self.target, self.predictor, self.tok, self.device.

        Args:
            draft_model: str path to eagle_predictor.pt, or an EaglePredictor
                         instance, or None (random init — use for training).
        """
        raise NotImplementedError(
            "EagleAlgorithm.setup() not yet implemented. "
            "See algorithms/eagle/algorithm.py for the implementation guide."
        )

    def generate(self, prompt_ids, max_new_tokens=128, K=4, L=8,
                 temperature=1.0, **kwargs) -> GenerationResult:
        """
        TODO: Implement EAGLE generation.

        Steps (paper Section 3 / Algorithm 1):
          1. Target forward on prompt → capture h_t via hook.
          2. Predictor autoregressively predicts K candidate hidden states.
          3. Apply target.lm_head to each candidate hidden state → draft logits.
          4. Build a token tree (L candidates at first level, fewer at deeper levels).
          5. One target forward with tree attention mask over all K*L nodes.
          6. Longest-prefix acceptance via rejection sampling.
          7. Repeat until max_new_tokens.
        """
        raise NotImplementedError(
            "EagleAlgorithm.generate() not yet implemented."
        )

    def train(self, dataset_path, output_dir, config=None, **kwargs) -> None:
        """
        TODO: Train the EAGLE predictor.

        Steps:
          1. Freeze target LM (all params require_grad=False).
          2. For each token position t in dataset:
               a. Run target → capture actual h_t via hook.
               b. predictor_input = concat([h_{t-1}, embed(token_{t-1})], dim=-1)
               c. h_t_pred = predictor(predictor_input)
               d. loss_regression = MSE(h_t_pred, h_t.detach())
               e. loss_ce = CE(target.lm_head(h_t_pred), token_t)  [optional]
               f. total_loss = loss_regression + lambda * loss_ce
          3. Optimise predictor weights only.
          4. Save to output_dir/eagle_predictor.pt
        """
        raise NotImplementedError(
            "EagleAlgorithm.train() not yet implemented."
        )
