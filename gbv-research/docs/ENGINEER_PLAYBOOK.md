# Engineer Execution Playbook — SpecDist / Distill-Spec-Research

A dependency-aware, sequenced plan an engineer can follow **faithfully**, one
focused GPU run at a time. It operationalises the LOCKED iteration priority from
[`docs/RESEARCH_PLAN.md`](RESEARCH_PLAN.md) into concrete runs that each do
**EXACTLY ONE thing**, using the single-purpose profiles in
`orchestration/configs/profiles/` and the `--train_only` / `--eval_only`
orchestration flags.

> **The one rule that governs everything below:** on free-tier compute you never
> sit idle waiting for a GPU run, and every GPU run produces exactly one
> artifact. Training sessions train (no eval). Eval sessions eval (no training).
> Every long run has a parallel **no-GPU** to-do list so the human is always
> making progress.

---

## 0. Ground rules (read once, internalise)

### 0.1 LOCKED iteration priority (from RESEARCH_PLAN.md Section D)

This is a **loop, not a one-shot**. Each step **gates** the next — do not advance
until the previous EXIT GATE passes.

1. **PRIORITY 1 — flat baseline `forward_kl` FIRST.** Establish a trusted,
   reproduced published-loss baseline. Nothing downstream is interpretable until
   this lands in the expected block-efficiency (BE) band.
2. **PRIORITY 2 — tree-loss training** (`kl_tree`, then `bv_tree` / `gbv_tree` /
   `traversal_tree`, …) — the core novel contribution, measured against the
   Priority-1 baseline.
3. **PRIORITY 3 — GBV verifier** — verifier-side contribution; pursued **only
   after** a tree loss shows promise.
4. **PRIORITY 4 — online / online-tree variants** — lowest priority, **deferred
   to the very end** (online code is disabled pending a smoke-verification).

### 0.2 Free-tier reality (hard limits)

| resource | budget | role |
|---|---|---|
| **Kaggle free** | **~30 GPU-h / week, resets weekly** | ALL exploration (rank losses, kill losers, tune LR) |
| Modal | **$30 one-time credit** | **ONE A100 confirmation run, spent LAST** |
| Lightning | ~15–22 free credits/mo | backup A100 / K=5 add-on in a follow-up month |
| Colab free | exhausted | never on the critical path |

- **Kaggle hardware: P100 (single 16 GB GPU) or T4×1 — NEVER T4×2.** Quota burns
  per GPU-wall-hour; T4×2 burns **2× quota for the same work** and every model
  here (0.6B draft + 8B-NF4 teacher ≈ 7.8 GB) fits one 16 GB GPU. Select
  **"P100"** (preferred) in Kaggle notebook settings. If you see T4×2, change it
  before running anything.
- **Check the quota counter** at <https://www.kaggle.com/me/quota> at the start
  of every session. Budget each week as **≤27 GPU-h** (safety buffer under 30 for
  one crashed session). One ~9-hour session ≈ 9 quota-h ≈ **one training run.**
- **A100 (Modal) is for confirmation only** — see [`docs/MODAL.md`](MODAL.md).
  Kaggle numbers are **direction only** (1 seed, n=100); never reported.

### 0.3 Always smoke first

**Before ANY real run, every session**, run the 10-step crash check
(`profiles/smoke`). A crash 8 hours into a 9-hour session wastes irreplaceable
weekly quota; 5 minutes here protects the whole session. Smoke the *specific
loss* you are about to run (`--losses <loss>`).

### 0.4 The single-purpose profiles (what each run uses)

| profile | flag set | does | does NOT |
|---|---|---|---|
| `profiles/smoke` | (10-step) | crash-check flat + tree + verifier-tree + eval paths | converge / produce real numbers |
| `profiles/train_one_loss` | `train_only` | train ONE loss → checkpoint + loss curve | eval (no BE/alpha) |
| `profiles/tree_variant_week` | `train_only` | train ONE surviving tree variant | eval / train >1 variant |
| `profiles/eval_after_train` | `eval_only`, K=3, T=1.0, n=100 | eval already-trained checkpoints | train / merge |

`--losses` on the CLI always wins over a profile's YAML default. `forward_kl` and
`reverse_kl` are accepted as aliases for the orchestration keys `kl` / `rev_kl`.

---

## 1. Dependency graph

```mermaid
graph TD
    S0["S0 · SMOKE (every session, ~5 min)"]:::smoke

    S0 --> W1
    W1["W1 · Pipeline + baseline-eval sanity + timing calibration<br/>(eval_after_train, untrained)"]:::eval
    W1 --> P1T

    subgraph PRIORITY 1 — flat forward_kl baseline
      P1T["P1-TRAIN · train forward_kl<br/>(train_one_loss --losses forward_kl)"]:::train
      P1E["P1-EVAL · eval forward_kl<br/>(eval_after_train --losses forward_kl)"]:::eval
      P1T --> P1E
    end

    P1E -->|GATE: BE finite, sane, val plateaus| P2TA

    subgraph PRIORITY 2 — tree-loss training (weekly survivor funnel)
      P2TA["P2-TRAIN-A · train tree variant A<br/>(tree_variant_week --losses kl_tree)"]:::train
      P2EA["P2-EVAL-A · eval variant A vs baseline"]:::eval
      P2TA --> P2EA
      P2EA -->|keep/kill| P2TB["P2-TRAIN-B · next surviving variant<br/>(--losses gbv_tree / bv_tree / …)"]:::train
      P2TB --> P2EB["P2-EVAL-B · eval variant B"]:::eval
    end

    P2EA -->|GATE: a tree loss beats forward_kl direction| P3
    P2EB -->|GATE| P3

    P3["P3 · GBV verifier (eval-only)<br/>gbv vs bv on survivor checkpoint<br/>(eval_after_train --losses <survivor>)"]:::eval
    P3 -->|GATE: BE(gbv) > BE(bv)| P4

    P4["P4 · online / online-tree (LAST, deferred)"]:::defer
    P3 -->|funnel ≤2 pairings| A100["A100 confirmation (Modal $30, ONCE)<br/>3 seeds, n=1319 — the ONLY paper numbers"]:::confirm

    classDef smoke fill:#eef,stroke:#446;
    classDef train fill:#efe,stroke:#484;
    classDef eval fill:#ffe,stroke:#884;
    classDef defer fill:#fee,stroke:#844;
    classDef confirm fill:#fef,stroke:#848;
```

**Reading the graph:** an arrow is a hard prerequisite. A labelled arrow is an
EXIT GATE — a quantitative bar that must be cleared before the next node is even
attempted. Eval nodes depend on their train node's merged checkpoint; the
pipeline enforces this (an eval step is BLOCKED if the merged model is missing).

---

## 2. Sequenced steps

All commands run from `gbv-research/` on Kaggle with
`--storage_root /kaggle/working/specdist` (omitted below for brevity — **always
include it**). All Kaggle runs: 1 seed, K=3, T=1.0, n=100, `--skip_existing`.

### S0 — Smoke pre-flight `[every session]`

| field | value |
|---|---|
| **Goal** | Prove the pipeline does not crash on the path you are about to run. |
| **Command** | `python orchestration/experiment.py --config profiles/smoke --smoke --yes --losses <the_loss_you_will_run>` |
| **Hardware** | P100 (or even CPU/laptop) |
| **Duration** | ~5 min |
| **Artifact** | throwaway checkpoints + a few smoke-scoped DB rows (numbers meaningless) |
| **Prerequisite** | none |
| **EXIT GATE** | every step exits 0; no NaN; no OOM on a single 16 GB GPU |
| **While this runs (no GPU)** | re-read this playbook's next step; confirm the Kaggle Accelerator is **P100, not T4×2**; check the quota counter |

### W1 — Pipeline validation + baseline-eval sanity + timing calibration

| field | value |
|---|---|
| **Goal** | Clean end-to-end run; confirm the **untrained** baseline BE is sane; measure real P100 s/step + per-eval-cell time to calibrate later weeks. |
| **Command** | `python orchestration/experiment.py --config profiles/eval_after_train --yes --skip_existing --experiment_tag w1_baseline` (only the baseline eval step runs; loss-eval steps auto-skip — no checkpoints yet) |
| **Hardware** | P100 |
| **Duration** | ~2–5 GPU-h |
| **Artifact** | baseline `block_eff` rows in `results.db`; recorded per-cell wall-times |
| **Prerequisite** | S0 |
| **EXIT GATE** | end-to-end completes; no OOM; baseline BE finite in ≈ **[1.3, 3.0]**; `peak_vram_mb` < 16 GB; per-step + per-cell timings recorded |
| **While this runs (no GPU)** | build the timing spreadsheet (s/step, h/1000-steps for flat vs tree); skim the W&B `eval/block_eff` panel; draft the Week-2 forward_kl run config |

### P1-TRAIN — Train the flat `forward_kl` baseline `[PRIORITY 1]`

| field | value |
|---|---|
| **Goal** | Produce the trusted Priority-1 baseline checkpoint + loss curve. Train this FIRST — nothing downstream is interpretable without it. |
| **Command** | `python orchestration/experiment.py --config profiles/train_one_loss --yes --losses forward_kl --experiment_tag p1_forward_kl` |
| **Hardware** | P100 |
| **Duration** | ~1.0–1.5 h / 1000 steps (flat) — fits one session with huge headroom |
| **Artifact** | `forward_kl` LoRA `ckpt_best`/`ckpt_latest` + merged model + `train_curves` rows (NO eval) |
| **Prerequisite** | W1 |
| **EXIT GATE** | training completes; **val-loss plateaus** (watch for a bounce after ~step 150 → lower LR / lengthen warmup); no NaN |
| **While this runs (no GPU)** | analyse the live train/val curve in W&B; prep the P1-EVAL command; code-review the first tree loss (`kl_tree`) you will train next; draft the decision-log entry |

### P1-EVAL — Evaluate the `forward_kl` baseline `[PRIORITY 1]`

| field | value |
|---|---|
| **Goal** | Get the reference BE/alpha the whole project is measured against. |
| **Command** | `python orchestration/experiment.py --config profiles/eval_after_train --yes --losses forward_kl --skip_existing --experiment_tag p1_forward_kl_eval` |
| **Hardware** | P100 |
| **Duration** | ~0.2–0.4 h / checkpoint-batch |
| **Artifact** | `runs` + `per_prompt` rows: `block_eff`, `alpha_mean`, `alpha_ci95`, `throughput`, `task_score` |
| **Prerequisite** | **P1-TRAIN** (merged model must exist) |
| **EXIT GATE (gates all of Priority 2)** | `forward_kl` BE is finite, plausible, within the W1 sanity band (≈ SOTA Qwen-class order of magnitude, ~3 at K=3; theoretical max L+1=9). **If it fails, STOP and fix the pipeline — do not train any tree loss.** |
| **While this runs (no GPU)** | record the baseline BE in the decision log as the reference number; finalise the Week-3 tree-variant order; review eval results in the dashboard |

### P2-TRAIN-{A,B,…} — Train ONE surviving tree variant per week `[PRIORITY 2]`

| field | value |
|---|---|
| **Goal** | Train exactly one tree-loss variant to checkpoints + loss curve. One variant per session; defer eval. |
| **Command** | `python orchestration/experiment.py --config profiles/tree_variant_week --yes --losses <variant> --experiment_tag wkN_<variant>` (variant ∈ `kl_tree`, `gbv_tree`, `bv_tree`, `traversal_tree`, …) |
| **Hardware** | P100 |
| **Duration** | ~1.5–2.5 h / 1000 steps (tree losses: 3 model passes + K-path tree) |
| **Artifact** | `<variant>` checkpoint + merged model + `train_curves` (NO eval) |
| **Prerequisite** | **P1-EVAL gate passed** |
| **EXIT GATE** | trains cleanly; val proxy plateaus; no NaN (drop on sight any loss that NaNs or collapses the curve) |
| **While this runs (no GPU)** | analyse the previous variant's eval/curves; prep the next variant's config; code-review the next tree-loss implementation in `tree_losses.py`; write the keep/kill rationale for the previous variant |

> **Weekly cadence (the survivor funnel):** each week train up to 3 candidate
> variants if the ≤27 GPU-h budget allows (≈3 × ~2 h train), **defer all eval**
> to one `eval_after_train` batch, then **keep only the survivor** that beats the
> `forward_kl` baseline direction and **kill the losers** — never spend a second
> week on a loss that did not beat the baseline.

### P2-EVAL — Evaluate trained tree variants (batched) `[PRIORITY 2]`

| field | value |
|---|---|
| **Goal** | Rank trained variants vs the baseline; pick the survivor. |
| **Command** | `python orchestration/experiment.py --config profiles/eval_after_train --yes --losses kl_tree,gbv_tree,bv_tree --skip_existing --experiment_tag wkN_eval` (list only the variants you actually trained) |
| **Hardware** | P100 |
| **Duration** | ~0.2–0.4 h per checkpoint-batch (several fit one session) |
| **Artifact** | BE/alpha rows for each variant under matched non-OT verifiers (bv/gbv/traversal) |
| **Prerequisite** | the corresponding **P2-TRAIN-*** checkpoints |
| **EXIT GATE (gates Priority 3)** | at least one tree loss shows **ΔBE > 0 vs `forward_kl`** at n=100 (direction). If none beats the baseline, revise the loss/LR and re-enter Priority 2. |
| **While this runs (no GPU)** | draw the ΔBE-vs-`forward_kl` bar chart; update the decision log; draft the analysis; prep the GBV-verifier eval command |

### P3 — GBV verifier check (eval-only) `[PRIORITY 3]`

| field | value |
|---|---|
| **Goal** | Test whether multi-path **GBV verification beats single-path BV** for the surviving tree-loss draft (no new training). |
| **Command** | `python orchestration/experiment.py --config profiles/eval_after_train --yes --losses <survivor> --skip_existing --experiment_tag p3_gbv_vs_bv` (gbv + bv are in the profile's modes; keep K≤4 for gbv stability) |
| **Hardware** | P100 |
| **Duration** | ~0.3–0.5 h (eval only) |
| **Artifact** | BE rows for the survivor under `gbv` and `bv` |
| **Prerequisite** | **P2-EVAL gate passed** + survivor checkpoint |
| **EXIT GATE** | **BE(gbv) > BE(bv)** for the survivor (within CI), at K≤4 |
| **While this runs (no GPU)** | build the loss×verifier BE heatmap; note any `gbv` degradation at K>4 (expected — `compute_skew` instability); freeze the ≤2 loss×verifier pairings for A100 |

### P4 — Online / online-tree variants `[PRIORITY 4 — LAST, deferred]`

| field | value |
|---|---|
| **Goal** | Only if Priority 2/3 land **and** budget remains: test online adaptation. |
| **Command** | smoke first (`--losses online_kl_tree`), then `train_one_loss --losses online_kl_tree` once a clean smoke confirms no divergence |
| **Hardware** | P100 |
| **Prerequisite** | P3 gate + spare quota |
| **EXIT GATE** | online beats its best offline counterpart (ΔBE > 0, non-overlapping CIs) |
| **Note** | online code is disabled pending smoke-verification; in practice push this to a follow-up **Lightning-credit** month so it never blocks the Modal credit. |

### A100 confirmation — Modal $30, ONCE, last

After the Kaggle funnel narrows to ≤2 loss×verifier pairings, spend the $30 Modal
credit **once**, strictly in priority order (Stage A `forward_kl` baseline → Stage
B best tree loss → Stage C `gbv` vs `bv`), at **3 seeds, full GSM8K n=1319, K=3**.
These are the **only** publication numbers. See RESEARCH_PLAN.md Section D and
[`docs/MODAL.md`](MODAL.md). Stop the moment the credit is exhausted (baseline +
the single headline tree-loss result are the non-negotiable minimum).

---

## 3. Per-week cadence (Kaggle, ≤27 GPU-h/week)

| week | focus | runs (each one session) | EXIT |
|---|---|---|---|
| **W1** | pipeline + baseline-eval sanity + timing | S0; W1 baseline eval | timings recorded; baseline BE sane |
| **W2** | **Priority 1**: train + eval `forward_kl`; LR/warmup tuning | S0; P1-TRAIN; (optional 1 LR-variant retrain); P1-EVAL | baseline trains cleanly, BE sane → gates Priority 2 |
| **W3** | **Priority 2** batch 1: train ≤3 tree variants, **defer eval** | S0; P2-TRAIN-A/B/C (train_only) | checkpoints + loss curves produced |
| **W4** | eval W3 variants; keep survivor, kill losers; train next survivor | S0; P2-EVAL (batched); P2-TRAIN-next | a tree loss beats `forward_kl` direction |
| **W5** | **Priority 3**: GBV verifier (eval-only on survivor) | S0; P3 | BE(gbv) > BE(bv); ≤2 pairings frozen |
| **W6** | buffer / K-T direction pre-check / A100 prep | S0; cheap K=3-vs-single-K=5 + T peek | confirmation spec frozen; Modal scripted |

The **Phase-1-then-defer-eval** discipline (train in one session, eval in a later
batched session) is what keeps every training session 100% on training.

---

## 4. Decision log template

Track every iteration so keep/kill decisions are reproducible. One row per
trained variant (copy into a tracking sheet or `docs/` note):

```
| date | run_tag / experiment_tag | loss | train_steps | lr | curve observed (train/val plateau? bounce? NaN?) | eval BE ± CI (n=100, K=3, matched verifier) | ΔBE vs forward_kl | KEEP / KILL | reason |
|------|--------------------------|------|-------------|----|--------------------------------------------------|---------------------------------------------|-------------------|------------|--------|
|      |                          |      |             |    |                                                  |                                             |                   |            |        |
```

Rules: a variant is **KILLed** if it NaNs, collapses `task_score`, or fails to
beat `forward_kl` direction after a full train. A **KEPT** survivor advances; do
not re-train killed losers in a later week unless the algorithm itself changed.

---

## 5. Resume / checkpoint discipline (Kaggle)

`/kaggle/working` is **ephemeral** — a lost session wastes irreplaceable weekly
quota. Be religious about this:

1. **Attach the dataset BEFORE Cell 1.** Re-attach your persistent Kaggle Dataset
   (the one holding `checkpoints/` + `results.db`) at the top of the notebook so a
   resumed run sees prior progress.
2. **Always pass `--storage_root /kaggle/working/specdist`** so all checkpoints,
   `results.db`, logs, and the state file live under one root.
3. **Cell 2b backup at session end.** Mirror `ckpt_latest` / `ckpt_best` +
   `results.db` to a Kaggle Dataset (or Drive) before the session dies. The
   profiles set `save_every: 25` so at most ~25 steps are ever lost.
4. **Resume with `--from <step_id>`** to restart mid-pipeline after a crash, e.g.
   `--from train_forward_kl_gsm8k` (training auto-resumes from `ckpt_latest`).
5. **Always pass `--skip_existing` on eval runs** — every restart resumes; a cell
   already in `results.db` is never recomputed (no duplicate rows, no wasted
   quota).
6. **One crash ≤ ⅓ of the week:** plan ≤3 sessions/week so a single lost session
   never costs more than a third of the quota.

---

## 6. Quick reference — exact commands

```bash
# Always smoke first (swap in the loss you are about to run):
python orchestration/experiment.py --config profiles/smoke --smoke --yes \
  --losses forward_kl --storage_root /kaggle/working/specdist

# Priority 1: train then eval the forward_kl baseline
python orchestration/experiment.py --config profiles/train_one_loss --yes \
  --losses forward_kl --storage_root /kaggle/working/specdist --experiment_tag p1_forward_kl
python orchestration/experiment.py --config profiles/eval_after_train --yes \
  --losses forward_kl --skip_existing --storage_root /kaggle/working/specdist --experiment_tag p1_forward_kl_eval

# Priority 2: train one surviving tree variant; eval later (batched)
python orchestration/experiment.py --config profiles/tree_variant_week --yes \
  --losses gbv_tree --storage_root /kaggle/working/specdist --experiment_tag wk4_gbv_tree
python orchestration/experiment.py --config profiles/eval_after_train --yes \
  --losses kl_tree,gbv_tree,bv_tree --skip_existing \
  --storage_root /kaggle/working/specdist --experiment_tag wk4_eval

# Priority 3: GBV verifier check on the survivor (eval only)
python orchestration/experiment.py --config profiles/eval_after_train --yes \
  --losses gbv_tree --skip_existing --storage_root /kaggle/working/specdist --experiment_tag p3_gbv_vs_bv

# Inspect a plan before running (no GPU, no model load):
python orchestration/experiment.py --config profiles/train_one_loss --losses forward_kl --dry_run
```
