"""
EAGLE-3 stub — Li et al. 2025 (arXiv:2503.01840).

IMPLEMENT ME:
  Fill in setup(), generate(), and train() following the paper.

Key architectural facts for implementation:
  - draft_source = FEATURE  (same paradigm as EAGLE / EAGLE-2)
  - verification = TREE     (dynamic tree, same as EAGLE-2)
  - training = REGRESSION   (regress onto intermediate — not final — hidden states)
  - needs_draft_model = True

Major change vs. EAGLE-2:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  EAGLE/2   hooks the *last*  transformer layer (layer N)            │
  │  EAGLE-3   hooks an *intermediate* layer  (layer N//2, configurable)│
  └─────────────────────────────────────────────────────────────────────┘

  Why: The last-layer hidden state encodes the *target distribution* — it is
  hard to predict.  Mid-layer features are smoother and more predictable,
  leading to higher acceptance rates and lower predictor training loss.

  Training change:
    EAGLE/2:  MSE(h_pred, h_last.detach())
    EAGLE-3:  MSE(h_pred, h_mid.detach())   [hook at layer N//2]
              + optional alignment term between h_pred and h_last

  Inference change:
    Same dynamic-tree algorithm as EAGLE-2 (Algorithm 1 in 2406.16858).
    The only difference is which layer's output is fed to the predictor.

Configurable parameters (pass to setup() via kwargs):
    mid_layer_idx (int):  Which transformer layer to hook (default = N // 2).
    tree_budget   (int):  Same as EAGLE-2 (default 60).
    top_k_per_node(int):  Candidates per node (default 10).
"""

from __future__ import annotations

from algorithms.base import (
    SpecDecAlgorithm, AlgorithmMeta, GenerationResult,
    DraftSource, VerificationType, TrainingType,
)


class Eagle3Algorithm(SpecDecAlgorithm):

    meta = AlgorithmMeta(
        name              = "eagle3",
        paper_title       = "EAGLE-3: Intra-model Alignment for Efficient LLM Inference",
        arxiv_id          = "2503.01840",
        draft_source      = DraftSource.FEATURE,
        verification      = VerificationType.TREE,
        training          = TrainingType.REGRESSION,
        needs_draft_model = True,
        notes             = "Li et al. 2025.  Hooks an intermediate target layer "
                            "(not the last) — intra-model alignment reduces predictor "
                            "training loss and improves acceptance rate. "
                            "Reference: arxiv.org/abs/2503.01840",
    )

    def setup(self, target_model, tokenizer, device,
              draft_model=None, model_family=None,
              mid_layer_idx: int | None = None,
              tree_budget: int = 60,
              **kwargs) -> None:
        """
        TODO: Wire EAGLE-3 predictor using an intermediate target layer.

        Steps:
          1.  Determine mid_layer_idx (default = num_layers // 2).
          2.  Register a forward hook on target.model.layers[mid_layer_idx]
              to capture h_mid during generate().
          3.  Load or initialise EaglePredictor (same architecture as EAGLE).
          4.  Weight-tie the predictor's LM head to the target's LM head.
          5.  Store self.target, self.predictor, self.tok, self.device,
              self.mid_layer_idx, self.tree_budget.
        """
        raise NotImplementedError(
            "Eagle3Algorithm.setup() not yet implemented. "
            "See algorithms/eagle3/algorithm.py for the implementation guide."
        )

    def generate(self, prompt_ids, max_new_tokens=128, K=4, L=8,
                 temperature=1.0, tree_budget=60, **kwargs) -> GenerationResult:
        """
        TODO: Implement EAGLE-3 generation.

        Identical to EAGLE-2 dynamic-tree generate() EXCEPT:
          - The hook captures h_mid (intermediate layer), not h_last.
          - Pass h_mid (not h_last) as predictor input.

        Refer to EAGLE-2 generate() docstring for the full algorithm.
        """
        raise NotImplementedError(
            "Eagle3Algorithm.generate() not yet implemented."
        )

    def train(self, dataset_path, output_dir, config=None, **kwargs) -> None:
        """
        TODO: Train EAGLE-3 predictor with intra-model alignment.

        Steps:
          1.  Freeze target LM.
          2.  For each token position t in dataset:
                a.  Run target → capture h_mid_t (intermediate) and h_last_t.
                b.  predictor_input = concat([h_mid_{t-1}, embed(token_{t-1})], dim=-1)
                c.  h_pred = predictor(predictor_input)
                d.  loss_mid = MSE(h_pred, h_mid_t.detach())
                e.  loss_align = MSE(h_pred, h_last_t.detach()) * lambda_align
                f.  total = loss_mid + loss_align
          3.  Optimise predictor weights.
          4.  Save to output_dir/eagle3_predictor.pt
        """
        raise NotImplementedError(
            "Eagle3Algorithm.train() not yet implemented."
        )
