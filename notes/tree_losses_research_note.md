# Tree-Structured Draft Distillation Losses
## Research Note

**Status:** Closed — negative, and at the larger capacity gap the log-space variants are actively **harmful**. At 0.6B/8B no tree loss beats flat JSD on traversal K=3 (warm log-space variants sit at the noise floor; the combined JSD+depth\_weight+naive\_tree run does **not** beat JSD at matched K — the previously cited +0.293 was a K\_eval=1 BE mislabeled as K=3; the true K\_eval=3 Δ is +0.03, within noise, see [`depth_weight_research_note.md`](depth_weight_research_note.md)). At **1.7B/32B, cold-start `traversal_log` and `nss_log` collapse** — traversal Δ −0.4 to −0.6 at K1–K3 (K4 −0.22 to −0.39), the largest regressions in the whole program. The log-space fix cures gradient *vanishing* but introduces gradient *misallocation* to unreachable deep nodes.
**Draft–Teacher:** Qwen3-0.6B / Qwen3-8B (warm) **and** Qwen3-1.7B / Qwen3-32B (cold) | **Train:** math\_hard | **Eval:** math\_eval + olympiad\_eval (100 prompts, A100, temp=0.2) | **JSD bars:** traversal K=3 = 5.870 (0.6B/8B), 5.783 (1.7B/32B)

---

## What Tree Losses Are

Flat JSD and prefix-overlap losses train on **teacher-context paths**: the teacher generates a token sequence, the draft's logits are evaluated on those tokens. The draft never sees its own rollouts during training.

Tree losses train on **student-sampled draft paths**: K paths of length L are sampled fresh from the current draft at every step. The loss is evaluated on the draft's own tree of continuations. The objective for each loss is $L_V = -E[\tau_V]$ — minimise the negative expected block efficiency under verifier V, estimated over the K draft paths.

The per-verifier acceptance formula $\alpha_V(p, q, K)$ is the differentiable surrogate for the verifier's per-node acceptance probability. The full E[τ_V] decomposes as a telescoping product:

$$E[\tau_V] = \sum_{i=1}^{L} \prod_{j=1}^{i} \alpha_V(p_j, q_j, K)$$

Two structural variants exist for how to train this:

| Variant | Loss formula | Gradient coefficient on $\nabla \alpha_i$ | Trainability issue |
|---|---|---|---|
| **Product form** (`_telescoping_loss`) | $-\sum_i \prod_{j \leq i} \alpha_j$ | $\prod_{j < i} \alpha_j$ → decays as $O(\alpha^i)$ at depth | Gradient starved at depth; at warm start ($\alpha \approx 0.7$), depth-8 weight $\approx 0.7^7 \approx 0.08$ |
| **Log-space form** (`_log_telescoping_loss`) | $-\sum_i \sum_{j \leq i} \log \alpha_j = -\sum_j (L-j+1) \log \alpha_j$ | $(L-j+1)/\alpha_j$ — triangular weight, does NOT vanish | Trainable at any depth, but still trains on student paths (same ceiling as JSD) |

---

## Loss Variants

| `--loss` flag | α formula | Code | What it optimises | Notes |
|---|---|---|---|---|
| `naive_tree` | $\alpha_{\text{naive}} = \sum_v \min(p_v, q_v) + \text{residual}$ | `tree.py:550` | $-E[\tau_{\text{naive}}]$, product form | Baseline OT verifier; residual term adds K-path acceptance |
| `traversal_tree` | $\alpha_{\text{naive}}$ (same node formula, different verifier at val) | `tree.py:181` | $-E[\tau_{\text{traversal}}]$ proxy via naive α | naive==traversal at node level; val uses traversal verifier |
| `traversal_log` | $\alpha_{\text{naive}}$, log-space | `tree.py:511` | $-\sum_j (L-j+1)\log\alpha_{\text{naive}}$ | Log-space fix for product vanishing; val = traversal verifier |
| `naive_log` | $\alpha_{\text{naive}}$, log-space | `tree.py:527` | Same formula as traversal\_log | Registered separately; val = naive verifier |
| `tree_pg` (removed) | REINFORCE, score = $E[\tau]$ | removed (was on `fix_bugs`, June 2026) | Policy gradient on block efficiency | Structurally wrong — see §Gradient Analysis |
| `op_naive_tree` / `op_naive_tree_full` | $\alpha_{\text{naive}}$, off-policy paths | `tree.py:559–560` | Product form, teacher-generated paths | Teacher paths prevent survival collapse but still train offline |

---

## Ablation Table (traversal K=3, math\_eval, n=100, temp=0.2)

| Experiment | Checkpoint | Traversal K=3 BE | Δ vs JSD | Beat JSD? |
|---|---|---|---|---|
| JSD flat (baseline) | `jsd_mathhard_s123/ckpt_best` | 5.870 | — | — |
| `naive_tree` cold (K=3, L=8, 8k steps) | `math_naivetree_s123/ckpt_best` | 5.432 | −0.438 | No — major regression |
| `op_naive_tree` (GSM8K, K=3, L=8)† | `op_naive_tree_gsm8k_train/ckpt_best` | 5.030‡ | −0.293‡ | No |
| `op_naive_tree_full` (GSM8K, K=3, L=8)† | `op_naive_tree_full_gsm8k_train/ckpt_best` | 5.027‡ | −0.296‡ | No |
| `tree_pg` REINFORCE (K=3, L=8) | abandoned — monotonic decline | 4.0–4.8 (training) | ~−1.5 | No — abandoned |
| JSD + depth\_weight\_lin + naive\_tree K=1 (40k)§ | `jsd_dw_lin_naive_mathhard_s123_K1/ckpt_best` | 6.036 | +0.166 | Borderline |
| JSD + depth\_weight\_lin + naive\_tree K=3 (40k)§ | `jsd_dw_lin_naive_mathhard_s123_K3/ckpt_best` | 5.903 | +0.033 | No — within noise (the 6.163/+0.293 previously here was the K\_eval=1 BE mislabeled as K=3) |
| `traversal_log` warm (JSD init, K=3, L=8, 8k)¶ | `traversal_log_K3_L8_lr1e5_s123/ckpt_best` | 6.017 (wandb curve) | +0.147 | No — noise floor |
| `naive_log` warm (JSD init, K=3, L=8, 8k)¶ | `naive_log_K3_L8_lr1e5_s123/ckpt_best` | 5.974 (wandb curve) | +0.104 | No — below noise floor |

†GSM8K-trained, evaluated on gsm8k\_eval; JSD bar on gsm8k\_eval traversal K=3 ≈ 5.323.  
‡Δ computed vs gsm8k\_eval JSD baseline (5.323), not math\_eval baseline.  
§Combined loss: JSD + depth\_weight\_lin + naive\_tree together, evaluated at K\_eval=3. At matched K=3 the Δ is +0.03 (within noise) — the run does **not** beat JSD; the earlier +0.293 came from reading the K\_eval=1 BE (6.163) against the K=3 baseline (5.870). See [`depth_weight_research_note.md`](depth_weight_research_note.md) for the depth\_weight attribution. Single seed, n=100.
¶No eval CSV for the two warm 0.6B/8B log-space checkpoints (`traversal_log_K3_L8_lr1e5_s123`, `naive_log_K3_L8_lr1e5_s123`) exists in `results/`; their BE values are read off wandb training curves ("wandb curve") and are **not reproducible from the committed data**.

Noise floor: SE ≈ 0.10–0.15 at n=100. Deltas below 0.15 are inconclusive.

**Raw results:** [`results/jsd_jsd_dw_naive_tree_results_math_hard_40000_steps.csv`](../results/jsd_jsd_dw_naive_tree_results_math_hard_40000_steps.csv) · [`results/MathHardDSJSDNaiveTreeResults.csv`](../results/MathHardDSJSDNaiveTreeResults.csv) · [`results/NaiveTreeOffPolicyResults.csv`](../results/NaiveTreeOffPolicyResults.csv)

---

## 1.7B/32B Cold-Start: Log-Space Losses Collapse (math\_eval + olympiad\_eval, L=8, n=100)

Cold-start (no JSD warm), single seed. Δ = traversal BE − `jsd_math_hard_s123` at matched K. Raw CSVs: [`results/per_checkpoint_sweeps_2026-07/`](../results/per_checkpoint_sweeps_2026-07/) (`*_Qwen32B-Qwen1.7B.csv`). **Backend (verified 2026-07):** both log-tree files and the baseline are **FA2 at K1–K3** (matched — the K1–K3 deltas below are backend-clean); only K4 is cross-backend (log-tree FA2 vs baseline sdpa), and the collapse is so large (≫ the ±0.37 backend range) that the conclusion is unaffected regardless.

| Loss | trav Δ K1 | K2 | K3 | K4 | olympiad K1/K2/K3 |
|---|---|---|---|---|---|
| `traversal_log` | −0.626 | −0.601 | −0.437 | −0.392 | −0.48 / −0.29 / −0.49 |
| `nss_log` | −0.628 | −0.632 | −0.587 | −0.223 | −0.33 / −0.34 / −0.38 |

These are the two worst training outcomes recorded (short of the abandoned `tree_pg`). Note the direct parallel to **PO's `traversal` objective** (see [`prefix_overlap_research_note.md`](prefix_overlap_research_note.md)), which is also the worst PO variant and for the same reason.

**Unified lesson — log-space / K-aware weighting is anti-curricular at cold start.** The triangular weight $(L-j+1)/\alpha_j$ (and PO-traversal's $K(1-\alpha)^{K-1}\alpha$) both *up-weight deep, low-α nodes* — precisely the tree positions a not-yet-trained draft cannot reach. Curing product-form vanishing (§Gradient Analysis) traded one failure for the opposite one: instead of starving the deep gradient, log-space *floods* it, pouring capacity into matching tree tails the draft will never traverse at inference. Warm-start on 0.6B/8B hid this (the draft began near α≈0.7 everywhere, so "deep" nodes were already reachable); cold-start at the wider 1.7B/32B gap exposes it. **Takeaway: a depth-reweighting scheme's benefit is entirely contingent on the draft already having non-trivial α at depth — it is not a from-scratch training signal, and `bv_log`/`gbv_log`-style extensions would inherit the same defect (plus the zero-gradient gate).**

---

## Gradient & Trainability Analysis

### Product form (naive\_tree, traversal\_tree)

At warm start ($\alpha \approx 0.7$ for BE ≈ 5.9), the survival product $\prod_{j < i} \alpha_j$ at depth $i$ decays as:

| Depth $i$ | Survival weight $0.7^{i-1}$ |
|---|---|
| 1 | 1.000 |
| 2 | 0.700 |
| 4 | 0.343 |
| 8 | 0.082 |

The gradient from deep tokens ($i = 7, 8$) is < 10% of the shallow gradient. Combined with the product form's instability (if a single token has $q[\text{token}] \ll p[\text{token}]$, $\alpha \to 0$ and the survival collapses), the cold naive\_tree run showed the worst result (5.432): the model degenerates before the shallow signal can pull it up.

### Log-space form (traversal\_log, naive\_log)

The triangular weight $(L-j+1)/\alpha_j$ does not vanish at depth — gradient norms were 98–376 throughout both runs, healthy and comparable to JSD training. The log-space fix works mechanically.

But both runs showed the same failure pattern as PO losses: early BE peak, then forgetting rises (forget = 1.3 at stop) and BE declines. Gradient norms confirm the loss is active throughout — the issue is not gradient flow.

The underlying constraint: both losses train on student-sampled paths. At any point in training, the student's α on its own paths is the same signal JSD already optimised. The tree loss reweights which depths matter, but does not inject new information. JSD already maximised token-matching in expectation; depth-reweighting is a second-order correction that cannot exceed JSD's ceiling.

### tree\_pg (REINFORCE)

The REINFORCE estimator computes a whole-tree scalar reward $r = E[\tau_V]$ and multiplies it by $\sum_{k,l} \log q_\theta(a_{k,l})$ (log-prob of all K×L sampled tokens). Three structural failures:

1. **No depth credit assignment:** the same advantage multiplies shallow decisive tokens and deep never-accepted tokens equally.
2. **Global scalar baseline:** $r$ correlates with prompt difficulty, not with which tokens are responsible — surviving signal is noise.
3. **Reward is a functional of q:** REINFORCE score-function ascent concentrates $q$, but block efficiency requires $q \to p$ (spread to match the teacher). The consistent component sharpens $q$ → reduces $\sum \min(p,q)$ → steady BE decline (4.81 → 4.0, monotonic).

This is not a bug — the estimator was mechanically correct REINFORCE. It is structurally misaligned with the objective. The `*_tree_pg` loss variants were removed from the codebase on branch `fix_bugs` (June 2026).

### Why not a cold init for log-space losses?

The cold naive\_tree result (5.432) already shows what happens without JSD pretraining: the product form collapses catastrophically. The log-space variant with cold init would avoid collapse but would still be training on low-quality student paths (draft is random at cold start → all paths rejected immediately → α ≈ 0 uniformly → loss is uniform, gradient uninformative). Warm start from JSD is the right approach. Given that warm traversal\_log and naive\_log both plateau at or below noise floor, cold init is not a useful control.

---

## Limitations

**1. Student paths only:** All on-policy tree losses use the student's own draft paths. The gradient only reaches tokens the student actually samples. Rare teacher tokens (high $p$, low $q$) are never sampled and never trained — a structural blind spot that flat JSD (which trains on all positions) does not have.

**2. K=3 only for log-space runs:** traversal\_log and naive\_log were run at K=3 only. It is possible that K=1 or K=5 would behave differently. Unrun.

**3. Single seed, 8k steps, n=100:** SE ≈ 0.10–0.15. Only the combined JSD+depth\_weight+naive\_tree K=3 result (+0.293) clearly exceeds the noise floor, and that result is confounded by depth\_weight.

**4. No pure naive\_tree warm run:** Every warm-start tree loss run used the log-space variant. The product-form naive\_tree was only run cold (and collapsed). A warm pure naive\_tree would quantify whether the product form is viable when starting from a good checkpoint. Given the log-space warm runs also show no improvement, this is low priority.

---

## Code Reference

`losses/tree.py` — all tree loss implementations.

| Symbol | Line | Role |
|---|---|---|
| `_alpha_naive(p, q, K)` | 333 | Per-node α for naive/traversal verifier |
| `_alpha_nss(p, q, K)` | 341 | Per-node α for NSS verifier |
| `_telescoping_loss(alpha_fn, ...)` | 417 | Product-form $-E[\tau]$ over K paths; `detach_survival=True` by default |
| `_log_telescoping_loss(alpha_fn, ...)` | 463 | Log-space $-\sum_j(L-j+1)\log\alpha_j$; triangular depth weight |
| `traversal_tree` | 181 | Product form, naive α, val = traversal |
| `naive_tree` | 550 | Product form, naive α, val = naive |
| `traversal_log` | 511 | Log-space, naive α, val = traversal |
| `naive_log` | 527 | Log-space, naive α, val = naive |
| `op_naive_tree` / `op_naive_tree_full` | 559–560 | Off-policy paths from teacher; `_full` un-detaches survival |
| `TREE_LOSSES` | 572 | Registry — dispatched from `train.py` by `--loss` flag |

| Flag | Default | Meaning |
|---|---|---|
| `--loss` | `jsd` | Set to any key in `TREE_LOSSES` to use a tree loss |
| `--K` | — | Number of draft paths per step; use 3 (deployment-matched) |
| `--L` | — | Draft block length; use 8 (deployment-matched) |
| `--val_temp` | — | Use 0.2 |
