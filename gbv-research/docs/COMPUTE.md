# Compute Resourcing & Sequencing Plan — SpecDist

*Hardware funnel for a solo student researcher on a near-zero budget.*
*Project: speculative-decoding draft-model distillation with tree-aligned losses, GSM8K.*

This doc is the **what-to-run-where-and-when** plan. Setup instructions are in the linked guides:

- [`KAGGLE.md`](./KAGGLE.md) — Kaggle free-tier setup (T4 x2 only; P100 broken; 4-bit NF4 config).
- [`MODAL.md`](./MODAL.md) — Modal on-demand A100/T4 setup ($30/mo credit).
- [`RESEARCH_PLAN.md`](./RESEARCH_PLAN.md) — research/ablation plan + locked algo priority.
- [`ENGINEER_PLAYBOOK.md`](./ENGINEER_PLAYBOOK.md) — dependency-aware sequenced run plan with exact commands.

> **Workload size:** Qwen3-8B teacher 4-bit NF4 (~4.5 GB) + Qwen3-0.6B student (~1.5 GB)
> → **peak ~6 GB VRAM**. Fits any 16 GB GPU.  
> A T4 is sufficient for all exploration; an A100 is only needed for final paper confirmation.

---

## Section 1 — Compute sources

### hw_tier: `cpu` — CPU-only convergence analysis (free)

| Source | Specs | Purpose |
|---|---|---|
| **ATS Cloud** (Adobe internal) | 128 GB RAM, 2-socket CPU, no GPU | GPT-2 parallel convergence runs — free, unattended |

**What runs here:** GPT-2 family only (`server_gpt2` config). Proves that loss objectives converge
and method beats forward_kl **on a small scale**. Never paper results — 4.3× teacher/student gap
is too small and it's CPU-only. See `deploy/ats/README.md` for the full workflow.

hw_tier tag: `cpu`

---

### hw_tier: `laptop` — Code verification (free)

| Source | Specs | Purpose |
|---|---|---|
| **Mukund's laptop** | RTX 500 Ada, 4 GB VRAM, 64 GB RAM | Smoke tests, crash-check every loss |

**What runs here:** Qwen 0.5B→0.6B pair (`laptop` config). The 0.6B teacher is ~same capacity
as the 0.5B student — distillation signal is noise. **Never interpret laptop numbers as research
findings.** Purpose: verify every code path runs without crash before promoting to T4.

hw_tier tag: `laptop`

---

### hw_tier: `colab` — Primary research platform (free T4)

| Pool | Free allowance | GPU | Effective T4-h | Notes |
|---|---|---|---|---|
| **Kaggle** | ~30 GPU-h/week | **GPU T4 x2** (only working option) | **~15 h/wk effective** | T4 x2 burns 2× quota. P100 (sm_60) is **broken** with PyTorch 2.10+cu128 (requires sm_70+). ~1 loss per session before credits run low. |
| **Modal** | **$30/mo** credit | T4 ≈ $0.59/hr | ~50 T4-h/mo | Per-second billing. |
| **Lightning AI** | 15 credits/mo | T4 ≈ $0.60/hr | ~25 T4-h/mo | Same 15-credit pool for all GPUs — use T4 not pricier options. |
| **Google Cloud** | $300 trial, 90 days | T4 (after quota grant) | ~850 T4-h | Card for verification only; quota grant needed (request T4 first). |
| **Colab free** | unreliable | T4 | n/a | Aggressive disconnects — backup only. Nothing persists. |

> **Combined free T4-class capacity ≈ ~190+ T4-class hours/month** (Kaggle + Modal + Lightning).

**What runs here:** Qwen3-0.6B draft → Qwen3-8B teacher NF4 (`kaggle` config).
This is the **minimum valid research platform** — same 8B teacher as A100; loss rankings transfer.

**Kaggle-specific constraint:** T4 x2 burns 2× quota → ~15 effective hours/week → realistically
**1 loss training + eval per Kaggle session** before credits are tight. Use `--losses kl` for one
session, `--losses rev_kl` for the next. Nothing persists between sessions without Kaggle Datasets
backup.

hw_tier tag: `colab`

---

### hw_tier: `a100` — Publication confirmation only

Use **only** once ≤2 candidates are publication-worthy.

| Source | ~Price | When |
|---|---|---|
| **IITH cluster prepaid** | ₹80/GPU-hr (~$0.96/hr) | **PRIMARY A100** |
| **Rahul Thomas (researcher)** | access TBD | Confirm hours before scheduling |
| **Modal A100** | ~$2–5/hr | Backup; $30/mo credit covers trimmed run |
| **GCP A100** | ~$3–4/hr | Last resort backup |

**What runs here:** Qwen3-0.6B draft → Qwen3-8B teacher BF16 (full precision), full GSM8K
n=1319, ≥3 seeds, paired-bootstrap + Holm–Bonferroni. **The only numbers that go in the paper.**

hw_tier tag: `a100`

---

## Section 2 — Research strategy (the funnel)

> **Key principle: always use 8B teacher for any research decision.**
> The 4B→0.6B gap (~7×) does not reliably predict 8B→0.6B (~13×) behaviour.
> Laptop (0.6B teacher) is crash-check only. GPT-2 convergence analysis shows the method
> works in principle but is not the paper setting.

| Tier | hw_tier | Config | Teacher | Purpose | Research-valid? |
|---|---|---|---|---|---|
| Crash check | `laptop` | `laptop.yaml` | Qwen3-0.6B | Code doesn't crash | No — teacher ≈ draft capacity |
| Convergence (small scale) | `cpu` | `server_gpt2` | GPT-2-M (355M) | Prove loss objectives converge | No — too small, CPU-only |
| **Exploration** | **`colab`** | **`kaggle.yaml`** | **Qwen3-8B NF4** | **Rank losses, tune LR, kill losers** | **✓ Yes** |
| **Confirmation** | **`a100`** | **`a100.yaml`** | **Qwen3-8B BF16** | **Paper numbers, multi-seed** | **✓ Yes** |

---

## Section 3 — Per-platform workflow

### Laptop (smoke / crash-check)

One-shot: `python orchestration/experiment.py --config laptop --smoke --yes`

Exercises every code path. Not iterated. Move to T4 once smoke passes.

### CPU server / ATS Cloud (convergence analysis)

See `deploy/ats/README.md` for the full workflow. In short:
```bash
bash deploy/ats/01_setup.sh           # once
python deploy/ats/02_verify.py        # check setup
python deploy/ats/03_smoke.py         # 10-step crash test
python deploy/ats/04_baseline_eval.py # baseline (alpha only)
python deploy/ats/05_train_parallel.py # all 15 GPT-2 losses in parallel
python deploy/ats/06_eval_trained.py  # eval each trained model
```

### Kaggle (primary research platform)

**Per-session workflow** (~1 loss per session given T4 x2 quota burn):

```python
# Cell 0 — Bootstrap (run once per session)
CONFIG  = "kaggle"
LOSSES  = "kl"      # ← change per session: kl → rev_kl → jsd → bv_tree → ...
SMOKE   = False
```

Each session does: baseline (if not already done) → train one loss → merge → eval.
`--skip_existing` ensures baseline is not re-run.

**Session sequence (rough)**:
- Session 1: kl, rev_kl (2 flat losses, ~5-8 hr each)
- Session 2: bv_tree, gbv_tree, traversal_tree (tree losses)
- Session 3: specinfer_tree, nss_tree, khisti_tree
- Session 4+: remaining losses

Backup checkpoints after each session — see `KAGGLE.md` → "Persisting checkpoints".

### A100 (publication confirmation)

**Persistent server — can do full iterative per-loss workflow:**

```bash
# Train + eval one loss at a time (--skip_existing resumes correctly):
python orchestration/experiment.py --config a100 --losses kl --yes
python orchestration/experiment.py --config a100 --losses bv_tree --yes
python orchestration/experiment.py --config a100 --losses gbv_tree --yes

# Or run all in one unattended shot:
python orchestration/experiment.py --config a100 --yes
```

The `--losses <name>` flag does: baseline (once) → train → merge → eval for that loss only.
Running multiple times with different `--losses` builds up results incrementally.

---

## Section 4 — GCP $300 trial

**No upfront charge.** Card = verification only; usage draws down $300.

GPUs need:
1. Upgrade to paid Cloud Billing account (still spends $300 first)
2. Request GPU quota: Console → IAM & Admin → Quotas → NVIDIA T4 GPUs → region → request 1

Request T4 first (auto-approves in minutes–2 days). A100 quota harder on fresh accounts.

---

## Section 5 — Action checklist

1. **Smoke** on laptop → confirm every loss+verifier runs without crash
2. **Kaggle Session 1:** `forward_kl` flat baseline — train + val_loss + light BE sanity
3. **Parallel T4 sessions** across Kaggle/Modal/Lightning for tree-loss ranking
4. **Confirm Rahul Thomas A100 access** before scheduling confirmation runs
5. **Load IITH cluster balance** so A100 confirmation is not blocked
6. **ATS Cloud CPU** for GPT-2 convergence analysis alongside T4 sessions (free, parallel)
7. Always `--skip_existing` and backup checkpoints — never re-run completed work
