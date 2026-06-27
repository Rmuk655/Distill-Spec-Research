# Prefix-Overlap Distillation
## Research Note on `prefix_overlap`

**Status:** Planned — 8 runs queued, no results yet.
**Draft–Teacher pair:** Qwen3-0.6B draft / Qwen3-8B teacher
**Training data:** math_hard | **Eval data:** math_eval (n=100, val_temp=0.2)
**Source:** Researcher doc (Rahul, June 2026) — implemented faithfully with no modifications.

---

## 1. Method

Sample M teacher continuations from the prompt; reward the student's cumulative probability on every teacher prefix.

$$
\mathcal{L} = -\frac{1}{M}\sum_{m=1}^{M}\sum_{t=1}^{L} q_\theta(P^{(m)}_{1:t} \mid c)
$$

where $q_\theta(P_{1:t} \mid c) = \exp\!\left(\sum_{i=1}^{t}\log q_\theta(P_i \mid c, P_{1:i-1})\right)$.

Gradients flow through the student probabilities; teacher tokens are fixed targets. No draft proposals, no verifier decisions in the training loop.

**Optional CE mixer (doc §7):** $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{prefix}} + \lambda \cdot \mathcal{L}_{\text{CE}}$, where CE is computed on the same sampled tokens (no extra teacher pass).

**Multi-root (doc §5):** one teacher rollout of length `rollout_len`; slide a length-$L$ window every $N$ tokens. Each tail $y_{r+1:r+L}$ is a valid M=1 sample from $p(\cdot|c_r)$. Cost: one teacher generate + one student forward for all roots.

### Code

`losses/compute.py` — `compute_prefix_overlap_loss` (single-root) and `compute_prefix_overlap_multiroot_loss` (§5 multi-root).  
`train.py` — routed via `is_prefix_overlap_loss(args.loss)`.

**Key flags:**

| Flag | Default | Meaning |
|---|---|---|
| `--prefix_M` | 4 | M teacher continuations (single-root only) |
| `--prefix_L` | 8 | continuation length |
| `--prefix_root_spacing` | 0 | 0 = single-root; >0 = multi-root §5 every N tokens |
| `--prefix_rollout_len` | 128 | teacher rollout length for multi-root |
| `--prefix_aux` | ce | secondary term: `ce` (free, same tokens) or `jsd` (+1 teacher forward) |
| `--prefix_aux_weight` | 0.0 | λ for secondary term |

---

## 2. Training Runs

All runs: `--loss prefix_overlap --steps 15000 --teacher_temp 1.0 --seed 123`

```bash
# GPU 0 — single-root, M=4, pure prefix (doc §4 baseline)
CUDA_VISIBLE_DEVICES=0 python train.py \
  --loss prefix_overlap --steps 15000 \
  --prefix_M 4 --prefix_L 8 --prefix_aux_weight 0 \
  --teacher_temp 1.0 --seed 123 \
  --output checkpoints/po_single_M4_pure_s123 \
  > logs/po_single_M4_pure_s123.log 2>&1 &

# GPU 1 — single-root, M=4, CE λ=0.001 (doc §7 mixer)
CUDA_VISIBLE_DEVICES=1 python train.py \
  --loss prefix_overlap --steps 15000 \
  --prefix_M 4 --prefix_L 8 --prefix_aux ce --prefix_aux_weight 0.001 \
  --teacher_temp 1.0 --seed 123 \
  --output checkpoints/po_single_M4_ce1e3_s123 \
  > logs/po_single_M4_ce1e3_s123.log 2>&1 &

# GPU 2 — single-root, M=4, CE λ=0.01 (higher mixer)
CUDA_VISIBLE_DEVICES=2 python train.py \
  --loss prefix_overlap --steps 15000 \
  --prefix_M 4 --prefix_L 8 --prefix_aux ce --prefix_aux_weight 0.01 \
  --teacher_temp 1.0 --seed 123 \
  --output checkpoints/po_single_M4_ce1e2_s123 \
  > logs/po_single_M4_ce1e2_s123.log 2>&1 &

# GPU 3 — single-root, M=1 (M ablation)
CUDA_VISIBLE_DEVICES=3 python train.py \
  --loss prefix_overlap --steps 15000 \
  --prefix_M 1 --prefix_L 8 --prefix_aux_weight 0 \
  --teacher_temp 1.0 --seed 123 \
  --output checkpoints/po_single_M1_pure_s123 \
  > logs/po_single_M1_pure_s123.log 2>&1 &

# GPU 4 — multi-root N=4, pure prefix (doc §5, dense roots)
CUDA_VISIBLE_DEVICES=4 python train.py \
  --loss prefix_overlap --steps 15000 \
  --prefix_root_spacing 4 --prefix_rollout_len 128 --prefix_L 8 \
  --prefix_aux_weight 0 \
  --teacher_temp 1.0 --seed 123 \
  --output checkpoints/po_multi_N4_pure_s123 \
  > logs/po_multi_N4_pure_s123.log 2>&1 &

# GPU 5 — multi-root N=4, CE λ=0.001
CUDA_VISIBLE_DEVICES=5 python train.py \
  --loss prefix_overlap --steps 15000 \
  --prefix_root_spacing 4 --prefix_rollout_len 128 --prefix_L 8 \
  --prefix_aux ce --prefix_aux_weight 0.001 \
  --teacher_temp 1.0 --seed 123 \
  --output checkpoints/po_multi_N4_ce1e3_s123 \
  > logs/po_multi_N4_ce1e3_s123.log 2>&1 &

# GPU 6 — multi-root N=8, pure prefix (sparser roots)
CUDA_VISIBLE_DEVICES=6 python train.py \
  --loss prefix_overlap --steps 15000 \
  --prefix_root_spacing 8 --prefix_rollout_len 128 --prefix_L 8 \
  --prefix_aux_weight 0 \
  --teacher_temp 1.0 --seed 123 \
  --output checkpoints/po_multi_N8_pure_s123 \
  > logs/po_multi_N8_pure_s123.log 2>&1 &

# GPU 7 — multi-root N=8, CE λ=0.001
CUDA_VISIBLE_DEVICES=7 python train.py \
  --loss prefix_overlap --steps 15000 \
  --prefix_root_spacing 8 --prefix_rollout_len 128 --prefix_L 8 \
  --prefix_aux ce --prefix_aux_weight 0.001 \
  --teacher_temp 1.0 --seed 123 \
  --output checkpoints/po_multi_N8_ce1e3_s123 \
  > logs/po_multi_N8_ce1e3_s123.log 2>&1 &
```

### What each comparison answers

| Comparison | Question |
|---|---|
| GPU 0 vs 1 vs 2 | Does CE mixer help? What λ? |
| GPU 0 vs 3 | Is M=4 worth 4× teacher sampling cost vs M=1? |
| GPU 0 vs 4 | Single-root vs multi-root — does root diversity matter? |
| GPU 4 vs 6 | N=4 (30 roots/rollout) vs N=8 (16 roots) — root density effect |
| GPU 1 vs 5 | CE mixer interacts with multi-root? |

---

## 3. Eval Plan

After training, eval best checkpoint per run:

```bash
python eval.py --checkpoint checkpoints/<name>/ckpt_best \
  --modes nss,naive,traversal,specinfer,bv --K 3 --L 8 \
  --n_prompts 100 --val_temp 0.2 --dataset math_eval
```

Primary comparison: traversal BE vs jsd_flat baseline (6.01). NSS expected to benefit most — prefix-overlap is the exact NSS BE gradient (NSS accepts on exact-match prefix, factorizes per-token).

---

## 4. Results

*(Fill in after runs complete.)*

| run | traversal K=3 | nss K=3 | naive K=3 | notes |
|---|---|---|---|---|
| jsd flat (baseline) | 6.01 | — | — | reference |
| po_single_M4_pure | | | | |
| po_single_M4_ce1e3 | | | | |
| po_single_M4_ce1e2 | | | | |
| po_single_M1_pure | | | | |
| po_multi_N4_pure | | | | |
| po_multi_N4_ce1e3 | | | | |
| po_multi_N8_pure | | | | |
| po_multi_N8_ce1e3 | | | | |
