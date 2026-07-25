# Hidden-State Overlap Distillation Objective

Companion proposal to `prefix_overlap_objective.pdf`. Same root-sampling setup, same estimator philosophy (sample only from the teacher, evaluate the student on the teacher's realization). The change is what gets scored at each position: instead of the discrete probability the student assigns to the teacher's own token, we score the distance between the student's and teacher's continuous hidden representations at that position.

This note also states plainly where the analogy breaks, since the original objective's unbiasedness proof depends on discreteness in a way that does not carry over.

## 1. Setup

Same as the original doc: teacher `p`, student `q_theta`, prompt `x ~ D`, teacher rollout `y_1:T ~ p(.|x)`, root positions `r in R(y)`, context `c_r = (x, y_1:r)`, teacher continuation `P_1:L ~ p(.|c_r)`.

Additionally define, at every position `t` along the continuation:

- `h^p_t` = teacher's hidden state at position `t` of the continuation, taken from a chosen layer `l_p` (a fixed intermediate layer, not the final layer feeding the LM head, since that is the head-adjacent representation "one layer below" token probability referenced in the framing of this note).
- `h^q_t` = student's hidden state at position `t`, from a chosen layer `l_q`, run in the SAME forward pass that already exists in the codebase (`compute_flat_loss` / `compute_prefix_overlap_loss` both already do `teacher(gen)` and `draft(gen)`; this only adds `output_hidden_states=True`).
- `proj: R^{d_q} -> R^{d_p}` = a trainable linear (or 2-layer MLP) adapter, needed because the draft and teacher have different hidden dimensions and no natural layer correspondence (0.6B draft vs 8B teacher). This is the one new trainable MODULE this objective introduces beyond the existing model -- for the fixed-layer/linear-adapter default this is a small parameter surface, but it is not fixed in general: moving to tuned-lens-style probes, softmin layer mixtures (Β§6), or a richer adapter expands it, so it should be tracked as a first-class design choice rather than a minor detail.

## 2. Why the exact prefix-overlap identity does not transfer

The original objective's core step is:

```
Pr(P_1:t = Q_1:t) = sum_u p(u|c) q_theta(u|c)
```

This holds because `u` ranges over a finite vocabulary, so "the two sequences are exactly equal" is a well-defined event with nonzero probability. `h^p_t` and `h^q_t` are points in continuous space (even after the linear adapter). For any continuous distribution, `Pr(h^q_t = h^p_t)` is 0 identically -- there is no discrete "longest common prefix" analog once you leave token space. Substituting hidden states for tokens directly into the LCP/indicator formula gives a degenerate objective (everywhere 0), not a generalization of it.

So this is deliberately NOT "the same formula, evaluated on hidden states." It is a different objective that reuses the same sampling machinery (root selection, teacher-only rollout, per-root minibatch estimator) and is motivated by the same intuition (successive-position agreement between student and teacher), replacing the discrete match-probability with a continuous distance/similarity score. That substitution is the standard, well-precedented way to move a discrete-matching idea into representation space (FitNets, TinyBERT, MiniLM, EAGLE's `L_reg` term all do exactly this), but it needs its own justification rather than inheriting the original doc's unbiasedness proof.

## 3. Per-root objective

Define a position-wise similarity/distance score `d(h^p_t, h^q_t)`, e.g. smooth-L1, negative cosine similarity, or MSE, applied after projection AND normalization:

```
d_t = SmoothL1( LayerNorm(proj(h^q_t)), stop_grad(LayerNorm(h^p_t)) )
```

Normalization is not optional: raw activation norms drift across depth and across model scale (0.6B vs 8B), so an unnormalized Smooth-L1 term is dominated by whichever side happens to have larger-magnitude activations at the chosen layer, independent of directional agreement. This mirrors why FitNets needed an explicit mapping layer between teacher and student hint layers rather than a raw regression -- the projection handles the dimension mismatch, normalization handles the scale mismatch, and the two are separate problems.

`stop_grad` matches the original doc's treatment of the teacher as fixed: the teacher rollout and teacher hidden states are produced under `no_grad`, exactly as `compute_flat_loss` already does for `t_out`.

Per-root objective, direct analog of the original doc's `J(theta;c) = sum_t Pr(P_1:t=Q_1:t)`, but now a sum of distances to be minimized rather than a probability to be maximized:

```
J_hidden(theta; c) = sum_{t=1}^{L} d_t
```

There is no INNER Monte Carlo estimator over hidden-state matches the way there is over token matches: `d_t` is already a deterministic function of the two hidden states realized on THIS teacher rollout, so there is no sum-over-vocabulary or `M`-sample average needed just to evaluate a single root's score. This is narrower than the earlier draft's claim of "no expectation gap" -- the overall training objective is still stochastic over the sampled teacher rollout and the sampled root positions, exactly as in the prefix-overlap pipeline. `M` roots per rollout (Β§5 of the original doc) still applies here, for the same reason it applies there: reducing variance across root/rollout draws, not within one root's score.

## 4. Loss and gradient

```
L_hidden(theta; c) = (1/L) * sum_{t=1}^{L} d_t
```

Gradient flow is ordinary supervised regression, not a REINFORCE/score-function estimator (this is the same distinction that separates this proposal from the tree_pg/NSS family that failed in this project): `d_t` is a differentiable function of `proj(h^q_t)`, `h^q_t` is a differentiable function of the student's own parameters via its normal forward pass, so `.backward()` flows into both `proj` and the student's weights exactly as `loss_fn(s_logits, t_logits).backward()` already does today for jsd.

## 5. Practical note -- stabilizer term (direct analog of Β§7)

The original doc's Β§7 flags that the pure `prob` objective can collapse from a cold start (an early low-probability token zeroes out every longer prefix term) and recommends `L_total = L_prefix + lambda * L_CE`. The hidden-state version has a different but related risk: a randomly-initialized `proj` adapter produces an arbitrary, uninformative target for the student early in training, so pinning too much weight on `L_hidden` before `proj` itself has learned anything wastes gradient signal. The direct analog recommendation:

```
L_total = L_hidden_stabilized  =  lambda_ce * L_CE  +  lambda_hidden * L_hidden
```

with `lambda_hidden` dominant (matches the earlier discussion: CE small anchor, hidden-state term dominant, since CE alone is already established as a working-but-weaker baseline in this project, and hidden-state matching is the new signal being tested). `L_CE` reuses `_aux_term(aux="ce")`, already implemented in `losses/compute.py`.

Unlike the token-level `prob` collapse (a property of the objective itself), the hidden-state version's cold-start risk is specifically an ADAPTER warm-up problem -- it can also be addressed by a short `proj`-only pretraining phase (freeze the student, fit `proj` to match already-existing hidden states for a few hundred steps) before joint training begins, in addition to or instead of a CE anchor.

## 6. Layer selection -- fixed layer by default, adaptive selection is a separate variant

The simplest, lowest-risk version of this proposal uses a single FIXED teacher layer (`layer_teacher`), chosen once (e.g. a middle layer, or the layer immediately before the final block) and never revisited during training. This is what should be tried first.

An entropy-adaptive layer picker (choose whichever teacher layer currently has lowest logit-lens entropy, per DREAM/DREAM-S) is a plausible extension but not a precedent-backed recipe -- the entropy-lens literature (Entropy-Lens, tuned-lens) uses per-layer entropy as an ANALYSIS signal, not as an established training-time hard selector for a distillation target. Two concrete risks if this is added:

- **Target jitter**: a hard argmin over candidate layers means a tiny entropy change can flip the selected layer between steps, making the regression target discontinuous step to step. If used, the selection must be on the TEACHER side only and detached (`stop_grad`) -- it is a routing decision, not part of the differentiable path -- and should be softened, e.g. a temperature-weighted mixture (softmin over entropy) across a small candidate set, rather than a hard argmin.
- **Raw logit-lens fragility**: projecting an intermediate hidden state through the model's own final unembedding (raw logit-lens) is known to be brittle (this is precisely the problem Tuned Lens was built to fix, via trained per-layer affine probes). If layer selection is entropy-driven, the entropy should come from a calibrated/tuned-lens-style readout, not the raw final LM head applied naively to an intermediate layer, or the selection signal itself may be noise.

Given both risks are stability/engineering concerns rather than conceptual blockers, the recommended path is: ship the fixed-layer version first, and only add adaptive selection as a follow-up ablation if the fixed-layer version shows signal.

## 7. Framing relative to prefix-overlap and JSD

Given the sweep history in this project -- `prefix_overlap`'s `prob` objective has never beaten the `jsd` backbone at ANY point in the hyperparameter search (best `prefix_overlap` result 5.848 vs `jsd`'s 5.994, a gap the sweep report treats as real, not noise) -- this hidden-state objective should NOT be framed or tested as a `prefix_overlap` replacement. It should be framed as an auxiliary loss added to the CURRENT BEST backbone (`jsd`), i.e.:

```
L_total = L_jsd + lambda_hidden * L_hidden
```

using `jsd`'s existing single-rollout structure (`compute_flat_loss`'s `teacher(gen)`/`draft(gen)` pair) rather than the root-sampled teacher-forced-continuation structure described in Β§1-3 above. The root-sampled `prefix_overlap`-style version (Β§1-6) is a legitimate variant worth keeping in this note since it reuses more of the existing prefix-overlap machinery and is closer to the DDTE tree-verifier alignment goal, but it should be tested and reported SEPARATELY from the `jsd`+hidden-state variant, not conflated with it -- the two have different backbones with different track records in this project, and a positive result on one should not be assumed to transfer to the other.

## 8. Pseudocode

Reference implementation, matching the structure and naming conventions of the original doc's PyTorch sketch and the existing `losses/compute.py` functions it would sit alongside.

```python
def compute_hidden_state_overlap_loss(
    draft, teacher, prompt_ids,
    layer_teacher, layer_draft,   # which intermediate layer to read from each model
    M=4,                          # roots per rollout, Sec. 5 of original doc
    L=128,                        # continuation length per root
    aux_weight=0.1,               # lambda_ce
    hidden_weight=0.9,            # lambda_hidden
    distance="smooth_l1",         # or "cosine", "mse"
):
    # 1. one teacher rollout, no_grad -- same as compute_flat_loss / compute_prefix_overlap_loss
    with torch.no_grad():
        full_rollout = teacher.generate(prompt_ids, max_new_tokens=ROLLOUT_LEN)

    roots = sample_roots(full_rollout, M)   # Sec. 5: fixed-spacing or random-offset

    total_hidden_loss = 0.0
    total_ce_loss = 0.0

    for r in roots:
        c_r = full_rollout[:, : r]                       # context up to root r
        P = full_rollout[:, r : r + L]                    # teacher's own realized continuation

        # teacher forward, no_grad, hidden states included
        with torch.no_grad():
            t_out = teacher(torch.cat([c_r, P], dim=1),
                             output_hidden_states=True, return_dict=True)
            h_p = t_out.hidden_states[layer_teacher][:, c_r.shape[1]:, :]   # (L, d_p)

        # student forward, grad-enabled, over the TEACHER's tokens (no student sampling)
        s_out = draft(torch.cat([c_r, P], dim=1),
                       output_hidden_states=True, return_dict=True)
        h_q = s_out.hidden_states[layer_draft][:, c_r.shape[1]:, :]        # (L, d_q)
        h_q_proj = proj(h_q)                                               # (L, d_p), trainable

        d_t = distance_fn(layer_norm(h_q_proj), layer_norm(h_p.detach()), kind=distance)  # per-position, (L,)
        total_hidden_loss += d_t.mean()

        # CE anchor on the same student forward pass (logits already computed)
        tok_lp = log_softmax(s_out.logits[:, c_r.shape[1]-1:-1, :], dim=-1)
        tok_lp = tok_lp.gather(-1, P.unsqueeze(-1)).squeeze(-1)
        total_ce_loss += -tok_lp.sum()

    loss = (aux_weight * total_ce_loss + hidden_weight * total_hidden_loss) / M
    return loss
```

## 9. Tree-form extension

Both `draft_tree_forward_with_grad` (`losses/compute.py:26`) and `target_tree_pass` (`verifiers/inference_util.py:83`) already perform a single tree-attention-masked forward pass over all nodes of a K-ary/L-deep tree. Adding `output_hidden_states=True` to each gives per-node hidden states for every node in one call, so `d_t` above can be computed per tree node instead of per single-rollout position, with the tree built via the same `iid_draft(K,L)` sampling the `traversal_verify` verifier itself uses at inference. This means hidden-state supervision lands exactly where the deployment verifier checks, and reuses tree machinery already in the codebase rather than requiring a new forward-pass path. This variant is only worth pursuing if the deployment path genuinely uses this tree-verifier structure -- otherwise it is a separate, lower-priority variant from the flat `jsd`+hidden-state version in Β§7.

## 10. Summary

Replace token-identity matching with hidden-state distance matching. This is not a literal generalization of the prefix-overlap estimator (the LCP/indicator formulation has no continuous analog and degenerates to zero), but a well-motivated, well-precedented (FitNets/TinyBERT/MiniLM/EAGLE) sibling objective. Recommended default configuration, in priority order:

1. Backbone is `jsd` (established winner in this project), hidden-state term is an ADDED auxiliary, not a `prefix_overlap` replacement -- given `prefix_overlap`'s `prob` objective has never beaten `jsd` in this project's sweep, there is no basis yet for expecting a `prefix_overlap`-based hidden-state variant to do better; test the `jsd`-backbone version first.
2. Fixed teacher layer, not entropy-adaptive selection -- adaptive selection is a follow-up ablation, not the initial test, since it introduces target-jitter and raw-logit-lens-fragility risks documented in Β§6.
3. Normalize (LayerNorm or unit-norm) both sides before the distance term -- unnormalized Smooth-L1 on raw activations is scale-sensitive across model sizes (0.6B vs 8B).
4. Trainable projection adapter is a first-class new component (new parameter surface: layer pair, distance metric, adapter capacity), not a minor detail -- budget a short adapter warm-up phase (freeze the student, fit `proj` alone for a few hundred steps) in addition to or instead of a CE anchor, since a randomly-initialized adapter gives an uninformative early target.
5. Root-sampled/`prefix_overlap`-style variant (Β§1-6, Β§9's tree form) is a separate, secondary experiment worth keeping on the books for the DDTE tree-verifier alignment angle, but should be reported independently from the `jsd`+hidden-state result, not conflated with it.
