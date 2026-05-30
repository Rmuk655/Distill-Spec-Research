# Tree-loss debugger — user manual

> A **unit test + visual debugger** for the tree-structured distillation losses
> and their matching tree verifiers. For **student learning and for debugging
> regressions**. Nothing here is imported by the training or eval pipeline.

It runs on toy order-1 Markov "models" (configurable vocabulary ≥ 6, default
V=6) so every distribution, draft tree and loss term fits on one screen — while
driving the **real** loss code (`distillspec_gbv/losses/tree_losses.py`) and the
**real** verifier code (`distillspec_gbv/verifiers/`) as the source of truth.

Because it imports the real code directly, this is also a **canary**: every time
a loss function, a verifier, or the training/scoring chain changes, re-run the
debugger. If block efficiency stops improving, the tree stops turning green, or
the tracer stops matching the real loss, you have found a regression — and the
per-node breakdown tells you *where*.

---

## 1. Contents

| file | what it is |
|------|------------|
| `tiny_models.py`    | toy teacher (frozen) + student (trainable) + tree sampling in the pipeline's **exact** dict format |
| `tree_harness.py`   | Monte-Carlo block-efficiency estimator that drives the **real** `TreeVerifier`; loss↔verifier map |
| `loss_tracer.py`    | transparent per-node re-derivation of each loss, **checked against** the real `compute_tree_loss` |
| `visual_debugger.py`| matplotlib/networkx CLI stepper + train-and-watch. **Interactive charts by default** (`--save` to also write PNGs to `out/`, which is gitignored). |
| `web_debugger.py`   | **interactive web** scrub / play / jump step debugger (Flask) — start here |
| `loss_variants.py`  | experimental loss variants that "mathematically stick" (KL anchor, full-grad, faithful clamp, IFT ρ) — selectable in the web UI (§8) |
| `variant_test.py`   | efficacy check: trains each variant a few steps and reports BE before→after |
| `smoke_test.py`     | one-command sanity check: all 12 losses traced + every verifier runs + backprop works |

---

## 2. Setup

Requires the repo's normal Python env (PyTorch). The web UI additionally needs
Flask; matplotlib/networkx are only needed for the CLI variant.

```bash
pip install flask                      # for web_debugger.py
pip install matplotlib networkx        # only for the CLI visual_debugger.py
```

No GPU, no model download, no dataset — the toy models are built in memory.

First, confirm the wiring is intact:

```bash
cd gbv-research/debug/tree_losses
python smoke_test.py          # expect: ALL OK
```

---

## 3. Quick start — the web debugger

```bash
python web_debugger.py        # -> http://127.0.0.1:5050   (use --port to change)
```

### Loss taxonomy — know what you're running

| Group | Status | Examples |
|-------|--------|---------|
| 📗 **flat baselines** | **PUBLISHED** — standard literature losses | `forward_kl`, `reverse_kl`, `jsd`, `l1` |
| 🔶 **novel flat** | **EXPERIMENTAL FLAT** — novel contributions, not yet published | `ebe`, `ebe_single` |
| ⚗ **tree losses** | **EXPERIMENTAL** — under active research | `kl_tree`, `bv_tree`, `gbv_tree`, `ebe_tree`, … |
| 🔬 **tree variants** | **EXPERIMENTAL VARIANT** — debug/research only | `bv_tree_kl`, `gbv_tree_faithful`, … |

All flat losses (published and novel) train with the sequence-level API
(`student_logits [T,V]` vs `teacher_logits [T,V]`). Block efficiency is still
evaluated with the full set of tree verifiers — so you can compare flat-baseline
BE vs tree-loss BE directly on the same chart.

### Controls

| Control | What it does |
|---------|-------------|
| **V** (vocab) | Token vocabulary size ≥ 6. Changing it auto-clamps K and L. |
| **K** | Draft paths per step. Must be < V; enforced live. |
| **L** | Draft depth. Must be < V; enforced live. |
| **T-peak** | Teacher "peakedness" — how concentrated the frozen teacher distribution is. Higher = harder student task. |
| **S-scale** | Student init scale — higher = more random starting point for the student. |
| **steps / lr** | Training steps and Adam learning rate. |
| **KL λ** | KL-anchor weight. **Greyed out and disabled** for losses that don't use it (production tree losses, `bv_tree_fullgrad`, `spectr_tree_ift`). Active for all `*_kl` / `*_faithful` variants. |
| **+khisti** | Include khisti verifier in charts (scipy LP, ~0.5 s/call — slow). |

1. Pick a **loss** — hover each option in the dropdown for a crisp tooltip description.
   A warning banner appears for any tree/variant loss (experimental), hidden for flat.
   Start with `kl_tree`, then try `bv_tree` (degrades) vs `bv_tree_kl` (fixed).
   Try `forward_kl` as the **flat published baseline** to compare.
2. Click **Train** (or **▶ Train (flat baseline)** for flat losses). Training runs in a
   background thread with a progress bar (~10–15 s for 80 steps default).
3. Use **◀ Step / ► Play / Step ▶**, the **scrub bar**, or *jump to step* → **Go**.
   The **speed** slider controls animation; node colours morph via CSS transition.

### Simulating Model Size (Capability Gap)

The four parameters below let you dial in a realistic teacher–student capability gap without changing the loss or verifier logic.

| Parameter | What it models | Real-world analogy | Web UI | CLI |
|---|---|---|---|---|
| **T-peak** | Logit concentration of the teacher's W matrix | Teacher training quality / scale: larger model → higher peak | T-peak | `--teacher-peak` |
| **T-temp** | Teacher inference sharpness | How deterministic the target model is at runtime; low τ_T = very focused = big model | T-temp | `--teacher-temp` |
| **S-scale** | Init std of the student's random W | How weak the student starts: small scale = near-uniform = very small model | S-scale | `--student-scale` |
| **S-temp** | Student draft sharpness | How diffuse the student's drafts are at runtime; high τ_S = flat = small/uncertain model | S-temp | `--student-temp` |

#### Typical large-gap setting

```
T-peak = 10,  T-temp = 0.5,   S-scale = 0.2,  S-temp = 2.0
```

Simulates e.g. a 70B teacher vs 1B student. The teacher is very peaked and sharp; the student starts near-uniform and diffuse. This maximises the capability gap and makes training signal strong — losses that improve BE under this setting are most likely to transfer to real model pairs.

**Web UI / CLI defaults** are calibrated to approximate a **Qwen 0.6B draft vs Qwen 8B target** pair:
- `T-peak = 6.0, T-temp = 1.0, S-scale = 0.3, S-temp = 1.0`
- Expected BE ≈ 2.5–3.5 for L=4, K=3 after training (consistent with published Qwen spec-dec results).

> **Note on spec-dec theory.** In real speculative decoding both models run at T=1 for the acceptance formula to be unbiased. The separate temperatures here are purely for the toy — they let you study how losses behave under different *effective* quality gaps without changing the logit structure.

### The CLI debugger

```bash
# interactive matplotlib windows (default)
python visual_debugger.py --loss kl_tree --K 3 --L 4 --steps 120
python visual_debugger.py --loss bv_tree --trace-only

# flat published baseline (trains on sequences, evaluates BE under tree verifiers)
python visual_debugger.py --loss forward_kl --steps 120

# also save PNGs to out/ (gitignored) — useful for reports
python visual_debugger.py --loss kl_tree --steps 120 --save
```

All charts open as **interactive matplotlib windows** you can zoom, pan, and hover.
`--save` additionally writes PNGs to `out/` (which is in `.gitignore` — never committed).

---

## 4. What's on screen and what to watch (web debugger)

**"What you're looking at" legend (top of page).** Dynamically populated after
each run: shows V (vocab), K, L, the pending token, teacher peak and student
init scale. Tokens are **integer IDs 0…V-1, not text**. Teacher = frozen target
model *p*; Student = trainable draft model *q*.

**Draft tree (top-left).** A frozen reference tree of `K` length-`L` paths.
Each tree level = one more drafted token. Hover a node to see its full path and
exact chain weight. Topology never changes during a run — only the **colours**
animate.

* Depth ruler on the left: `d0` = the pending token (the prompt), `d1…dL` = drafted tokens.
* Colour = **chain weight** `w = Π min(1, p/q)` down the path to that node:
  * **red** — draft over-proposes this token (target rejects it),
  * **green** — target accepts (`p ≥ q`).

*Watch:* nodes should turn from red toward green from the root down. For
`gbv_tree`, the GBV-selected path is outlined in blue.

**Block efficiency under EVERY verifier (bottom-left).** One line per verifier
(`naive, nss, bv, specinfer, spectr, traversal, gbv, max`), at the training `K`.
The matched verifier is bold.

*Watch:* a good loss lifts **all** lines, not just its own. `nss` always lags
(it samples from `p` and ignores `q` for the token choice). The vertical cursor
tracks the current step.

**Metrics + Loss (top-right).** Current loss, matched verifier, matched BE, the
loss decomposition string (e.g. `E[τ] = …`), and the loss curve with a marker.

**Per-node loss breakdown (mid-right).** The *same* quantities the real loss
computes, node by node, for the current step: per-node acceptance `α_i`, the
accumulating product / chain weight, and block-acceptance `h_i` for BV/GBV.

*Watch:* this is the "why". If BE is flat, look here — are the `α_i` moving? Is
the secondary product vanishing with depth (the BV gradient problem)?

**Flat (K=1) BE — before vs after (bottom-right).** Bars for every verifier
evaluated at **K=1** before and after training. K=1 is where the tree collapses
to a single chain and the verifiers become their **flat** counterparts:
`naive` = vanilla speculative sampling, `bv` = block verification, and
`gbv`/`traversal` collapse to `bv`.

*Watch:* this answers "does the loss help flat decoding too?" For `kl_tree` every
flat bar rises (except `nss`).

---

## 5. What you'll learn

* **The shared objects.** A draft tree is two dicts keyed by the comma-joined
  prefix (`"0,3,2"`): `q_probs_dict[prefix] -> [V]` (draft, **non-leaf nodes
  only**, as `iid_draft` stores) and `p_probs_dict[prefix] -> [V]` (target, **all
  nodes**, as `target_tree_pass` stores). The single most useful scalar is the
  chain weight `w_i = Π_{j≤i} min(1, p[t_j]/q[t_j])`.
* **Why KL-anchored losses win and pure acceptance losses are fragile** (§7).
* **How each verifier-aligned loss mirrors its verifier** — and where the mirror
  is exact vs approximate (§6, §7).
* **What block efficiency actually measures** and how it is estimated here (§9).

---

## 6. Reference — the verifiers and the losses

### 6a. The eight verifiers (source of truth)

Defined in `verifiers/otlp_registry.py` (tree walk) and `verifiers/tree.py`
(per-node OTLP solvers). Each `verify(mode)` walks the tree and returns
`(accepted_node, residual_token)`; block efficiency per call is
`accepted_node.depth + 1` (accepted prefix + 1 bonus token).

| verifier | family | one-line behaviour |
|----------|--------|--------------------|
| `naive`     | OT, single-path | accept first child w.p. `min(1,p/q)`, else residual `relu(p-q)` |
| `bv`        | non-OT, single-path | block acceptance `h_i = relu(w_i p − q).Σ / (relu(w_i p − q).Σ + 1 − w_i)`; accept the last block clearing a uniform draw |
| `nss`       | OT, multi-path | sample straight from `p`; accept prob `Σ p(1−(1−q)^K)` |
| `spectr`    | OT, multi-path | K-SEQ with division factor `ρ` found by binary search |
| `specinfer` | OT, multi-path | iterative rejection over children, residual `relu(p−q)` each step |
| `khisti`    | OT, multi-path | LP rank-tournament importance sampling (canonical decomposition) |
| `traversal` | non-OT, multi-path | DFS bottom-up; accept first leaf clearing its weight, else prune & reweight |
| `gbv`       | non-OT, multi-path | greedily walk the max-`p/q` path building a **skewed** draft `q_skew`, then run `bv` on that path with `q_skew` |

Reduction worth remembering (registry docstring): for **`K=1`** the OT multi-path
methods collapse to `naive`, and `gbv`/`traversal` collapse to `bv`. That is
precisely what the *flat (K=1)* panel measures.

### 6b. The twelve tree losses

All in `tree_losses.py`. Each takes the student's per-node distributions **with
gradient** (recomputed on the fixed sampled tree by `draft_tree_forward_with_grad`)
plus the detached teacher distributions, and returns a scalar with `grad_fn`.

* **Generic divergences — `kl_tree`, `rev_kl_tree`, `jsd_tree`.** A divergence
  between `p` and `q` averaged over **every tree node** (on-policy). `kl_tree`
  = mean `KL(p‖q)` (mode-covering); `rev_kl_tree` = mean `KL(q‖p)` (mode-seeking);
  `jsd_tree` = mean symmetric JSD. These "anchor" losses push `q → p` everywhere,
  which drives `min(1,p/q) → 1` and raises BE for **all** verifiers.
* **Verifier-aligned, non-OT — `bv_tree`, `gbv_tree`, `traversal_tree`.** A
  differentiable `−E[τ]`. `bv_tree` replicates `bv_verify`'s block acceptance;
  `gbv_tree` selects the greedy max-`p/q` path (no grad), rebuilds the GBV
  `q_skew` at each node on it (with grad, mirroring `compute_skew()`), then runs
  `bv_tree` on that path with `q_skew`; `traversal_tree` maximises mean leaf
  weight only (honest lower-bound surrogate; DFS rejection is non-differentiable).
* **Verifier-aligned, OT-based — `naive_tree`, `nss_tree`, `specinfer_tree`,
  `spectr_tree`, `khisti_tree`.** Each verifier has a closed-form per-node
  acceptance `α_V(p,q,K)` (the `*_otlp_accept` methods in `tree.py`), re-implemented
  differentiably as `_alpha_*` and fed into a shared telescoping scaffold
  `E[τ_V] = Σ_{i=1}^{L} Π_{j≤i} α_V(p_j,q_j,K)`, `L_V = −E[τ_V]`.
* **Ablation — `ebe_tree`.** Single-path naive `E[τ]=1+Σ_k Π_{i≤k} α_i` plus a
  `KL(p‖q)` regulariser (`kl_weight=0.1`).

The verifier ↔ loss correspondence is 1:1 (`tree_harness.LOSS_TO_VERIFIER`),
which is what the Phase-3 8×8 cross-pair eval tests: *train with `L_V`, evaluate
under `V`*.

---

## 7. Critical analysis — does the wiring make sense?

### ✅ Wired correctly

* **The telescoping `E[τ] = Σ_i Π_{j≤i} α`** equals the expected accepted-prefix
  length — exactly what BE measures. The `_alpha_*` formulas are faithful
  transcriptions of the `*_otlp_accept` methods (verified term-by-term).
* **`gbv_tree`'s skew transform** matches `compute_skew()` including the
  numerically-stabilised factorisation, with gradient routed through `q` and the
  discrete `argsort`/permutation kept no-grad. This is the non-trivial
  "transform to tree space" and it is done properly.
* **On-policy data**: training re-scores the student's *own* sampled tree with
  grad (`draft_tree_forward_with_grad`), fixing the off-policy mismatch that
  cripples flat EBE.

### ⚠️ Subtleties and mismatches worth knowing

1. **`bv_tree` chain-weight clamp differs from `bv_verify`.** The verifier clamps
   the **product** (`w_i = min(1, w_{i-1}·p/q)`), so a token with `p/q>1` can
   *restore* weight; the loss clamps **each factor** (`Π min(1,p/q)`), monotone
   non-increasing. Not equal (`[0.5, 3]` → `w=1.0` verifier vs `w=0.5` loss).
2. **`bv_tree` (hence `gbv_tree`) has a mostly-detached gradient.** `w` and the
   survival product `Π(1−h_j)` are detached; the only live paths are leaf weight
   `h_L=w_L` (vanishes for large `L`) and the `−q` term. With **no KL anchor**
   the signal is weak.
3. **`spectr_tree` detaches `ρ`** and **`khisti_tree` uses an LP-free softmax
   surrogate**. These are the *least faithful* OT losses — first place to look if
   `spectr`/`khisti` cross-pairs underperform.
4. **`traversal_tree` is the crudest surrogate** (mean leaf weight only), yet
   `traversal` is empirically the **strongest** verifier — the best verifier has
   the loosest matched loss (an improvement opportunity, not a bug).
5. **`nss` ignores `q` for the token choice** (samples straight from `p`); the
   draft only affects coverage via `1−(1−q)^K`. Even a perfect draft leaves `nss`
   lagging.

### 🔬 What the toy actually shows (reproducible in the web UI)

Training the toy student and measuring test BE with the real verifiers:

| loss trained | matched BE before → after | helps flat (K=1)? | verdict |
|--------------|---------------------------|-------------------|---------|
| `kl_tree`   | ~1.7 → **~4.3** | yes — every flat verifier rises (naive 1.6→4.5, bv 1.6→4.6; nss lags) | strong KL anchor drives `q→p` |
| `ebe_tree`  | ~1.9 → **~4.9** | yes | strong — acceptance **+ KL regulariser** |
| `bv_tree`   | ~1.7 → **~1.1** | no | degrades — pure block-acceptance gradient with no anchor is degenerate alone |

Headline: **KL-anchored losses converge strongly; pure verifier-aligned losses
give a weak/degenerate signal on their own** — exactly what items (1)–(2)
predict, and a strong hint that the verifier-aligned losses want the same small
`KL(p‖q)` regulariser `ebe_tree` already carries.

> Caveat: toy order-1 models, not the real Qwen/Gemma pairs. The *direction*
> (anchor helps, pure acceptance is fragile) is robust; exact magnitudes are not
> meant to transfer.

---

## 8. Experimental variants that "mathematically stick" (`loss_variants.py`)

The production losses (the source of truth) stay untouched. `loss_variants.py`
offers stronger alternatives that **reuse the real helpers** (`_gbv_select_path`,
`_compute_q_skew_along_path`, `_alpha_*`, `kl_tree_loss`) and are selectable in the
web debugger's loss dropdown under *experimental variants*. They target the three
documented weaknesses (§7 items 1–3):

| variant | what it changes vs the production loss | matched |
|---------|----------------------------------------|---------|
| `bv_tree_kl`        | REAL `bv_tree` **+ λ·KL(p‖q)** anchor (the fix that matters) | bv |
| `bv_tree_fullgrad`  | un-detach `w` and survival → full pathwise gradient (+ optional λ·KL) | bv |
| `bv_tree_faithful`  | verifier-correct **product clamp** `w_i=min(1,w_{i-1}·p/q)` + detached grad + λ·KL | bv |
| `gbv_tree_kl`       | REAL `gbv_tree` + λ·KL | gbv |
| `gbv_tree_faithful` | GBV skew → product-clamp BV + λ·KL | gbv |
| `spectr_tree_ift`   | give ρ a **real gradient via the implicit function theorem** instead of detaching it | spectr |
| `spectr_tree_kl`    | REAL `spectr_tree` + λ·KL | spectr |
| `khisti_tree_kl`    | REAL `khisti_tree` (LP-free surrogate) + λ·KL | khisti |
| `traversal_tree_kl` | REAL `traversal_tree` + λ·KL | traversal |

**The λ knob.** The header `KL λ` box sets the KL-anchor weight (the same
`KL(p‖q)` term `ebe_tree` carries). Set `λ=0` to see the raw variant; raise it to
watch a degenerate loss become a learner. This is the headline interactive lesson.

### What the toy shows (40 steps, K=3 L=4, test BE under the matched verifier)

| loss | BE before → after | note |
|------|-------------------|------|
| `bv_tree`            (production) | 1.61 → **1.08** | degrades — weak/degenerate gradient, no anchor |
| `bv_tree_kl`         (λ=0.1)      | 1.69 → **3.81** | KL anchor fixes it |
| `bv_tree_fullgrad`   (λ=0)        | 1.67 → **1.42** | un-detaching alone does **not** help |
| `bv_tree_fullgrad`   (λ=0.3)      | 1.67 → **4.48** | …but full-grad **+ anchor** is the strongest BV |
| `bv_tree_faithful`   (λ=0.1)      | 1.68 → **1.44** | product clamp keeps `w` large → drowns a small anchor |
| `bv_tree_faithful`   (λ=1.0)      | 1.53 → **4.18** | …raise λ and it sticks |
| `gbv_tree`           (production) | 1.77 → **1.30** | degrades |
| `gbv_tree_kl`        (λ=0.1)      | 1.78 → **2.92** | anchor fixes it |
| `spectr_tree`        (production) | 1.80 → **4.83** | already strong |
| `spectr_tree_ift`    (λ=0)        | 1.69 → **4.77** | IFT ρ-gradient matches detached-ρ here (ρ varies slowly, as the source claims) |
| `traversal_tree_kl`  (λ=0.1)      | 1.86 → **4.68** | anchor also lifts the strongest verifier's loss |

Two takeaways, both reproducible live: (1) **a KL anchor of the right strength is
the robust fix** for the fragile acceptance losses; (2) **un-detaching the gradient
is not free** — it only helps in combination with an anchor. Run `python
variant_test.py` to regenerate this table.

> These remain *experiments on toy models*. Before promoting any variant into the
> production `tree_losses.py`, validate it on the real Qwen/Gemma cross-pair matrix.

---

## 9. Debugging regressions — your workflow when code changes

This sandbox exists so a change to the loss/verifier/training chain is caught
immediately. When you edit any of `tree_losses.py`, `tree.py`, `otlp_registry.py`,
`tree_training.py`, or `draft_generator.py`:

1. `python smoke_test.py` — if a tracer now **mismatches** the real loss, the
   loss math changed; reconcile `loss_tracer.py` *or* fix the loss. If a verifier
   errors, the tree wiring changed.
2. `python web_debugger.py` → Train the affected loss. Symptoms → likely cause:
   * **tree stays red / BE flat** → no useful gradient (check on-policy
     re-scoring and that the loss `requires_grad`).
   * **matched line rises but others don't** → the loss over-fits its verifier;
     compare to `kl_tree` as the all-verifier baseline.
   * **flat (K=1) bars don't move** → the change broke the single-chain reduction.
   * **breakdown `α_i` frozen at a value** → a detach was added or a clamp
     saturated (see §7 items 1–2).
3. Use *jump to step 0* vs *last* and the before/after bars to quantify the delta.

Because the harness uses the **real** classes, "it works in the toy" is strong
evidence the wiring is sound; "it broke in the toy" almost always means a real
bug.

---

## 10. FAQ

**Is `kl_tree` (or any loss) verified against *all* verifiers, even flat ones?**
Yes, in the web debugger. Per step it scores all eight tree verifiers
(`naive, nss, bv, specinfer, spectr, traversal, gbv, max`) at the training `K`,
and the bottom-right panel scores every verifier at **K=1 (flat)** before vs
after — where `naive` = vanilla speculative sampling and `bv` = block
verification. (The CLI `visual_debugger.py` plots the matched verifier only; use
the web UI for the all-verifier chart. `khisti` is opt-in via the `+khisti`
checkbox because its scipy LP is slow.)

**Why does the tree topology never change during a run?** The animation freezes
one reference tree and re-scores it with the evolving student, so only colours
(chain weights) move. Training itself still samples **fresh** on-policy trees
each step — the reference tree is for visualization only.

**Why are toy models valid here?** We shrink the *models* to nothing but keep the
*algorithms* exact (real loss + real verifier code). Conclusions about wiring and
gradient behaviour transfer; absolute BE numbers do not.

**Are the new variants in the production loss file?** No. They live only in
`debug/tree_losses/loss_variants.py` and are never imported by training/eval. They
reuse the real helpers so the comparison is honest, but `tree_losses.py` (the
tested source of truth) is unchanged. Promote a variant only after it wins on the
real cross-pair matrix.

---

## 11. How block efficiency is measured here

`tree_harness.estimate_block_efficiency` builds a fresh detached tree for each
trial, **clears the `Node` OTLP caches** (class-level, keyed by `id(p)`, which
gets recycled), runs the real `TreeVerifier.verify(mode)`, and averages
`accepted_node.depth + 1`. Fresh trees per trial are mandatory because
`traversal_verify` mutates its `p`/`q` dicts in place — the same hazard the unit
tests in `tests/unit/test_verifiers.py` guard against.
