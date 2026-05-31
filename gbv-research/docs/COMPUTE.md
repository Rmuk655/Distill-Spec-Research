# Compute Resourcing & Sequencing Plan — SpecDist

*Hardware funnel for a solo student researcher on a near-zero budget.*
*Project: speculative-decoding draft-model distillation with tree-aligned losses, GSM8K.*

This doc is the **what-to-run-where-and-when** plan. It does **not** repeat setup
instructions — follow the linked guides for those:

- [`KAGGLE.md`](./KAGGLE.md) — Kaggle free-tier setup (P100/T4, 4-bit NF4 config).
- [`MODAL.md`](./MODAL.md) — Modal on-demand A100/T4 setup ($30/mo credit).
- [`RESEARCH_PLAN.md`](./RESEARCH_PLAN.md) — research/ablation plan + **locked algo priority**.

> **Workload size (the key fact):** Qwen3-8B teacher in 4-bit NF4 (~4.5 GB) +
> Qwen3-0.6B student (~1.5 GB) → **peak ~6 GB VRAM**. This fits *any* 16 GB GPU.
> A T4/P100 is therefore sufficient for **all** exploration; an A100 buys nothing
> for prototyping and only exists in this plan for final publication confirmation.

> **Pricing caveat:** all $/hr figures and free-tier sizes below are researched and
> current as of 2026, but providers change them often — **verify current rates** on
> each provider's pricing page before committing real spend.

---

## Section 1 — Compute sources

### Free T4-class pools (the exploration workhorses)

| Pool | Free allowance | GPU to use | Effective T4-h | Notes |
|---|---|---|---|---|
| **Kaggle** | free, **~30 GPU-h / week** (resets weekly) | **P100 ×1** (single-GPU) | ~30/wk | Shared P100/T4×2. Use P100 single-GPU — **NOT T4×2** (burns 2× quota). Check `kaggle.com/me/quota`. |
| **Modal** | **$30/mo** credit (resets monthly, **no rollover**, no card to start) | T4 ≈ $0.59/hr | **~50 T4-h/mo** | A100 ≈ $2–5/hr (see below). Per-second billing. |
| **Lightning AI** | **15 credits ($15)/mo** (1 cr = $1; phone-verify) | T4 ≈ $0.60/hr | **~25 T4-h/mo** | Free-tier GPUs = T4/L4/A10G/L40S (A100/H100 ≈ Teams-plan only). **All GPUs draw the same 15-credit pool → use T4**, not pricier GPUs. |
| **Google Cloud** | **$300** trial, **90 days**, new accounts | T4 (after quota grant) | **~850 T4-h** | Card = verification only, **not** charged upfront; usage draws down $300; account **pauses** at trial end (no auto-charge). GPUs need: (a) upgrade to paid billing [still spends the $300 first] + (b) a GPU quota-increase request (default 0). See [Section 4](#section-4--gcp-300-trial-how-to). $300 ≈ ~850 T4-h or ~80–100 A100-h. |
| **Colab free** | unreliable (~15–30 h/wk T4, unpublished) | T4 | n/a | Aggressive disconnects — **backup only**. |

> **Combined free T4-class capacity ≈ ~190+ T4-class hours/month**
> (Kaggle + Modal + Lightning) — ample for the entire exploration phase.

### A100 sources (PUBLICATION confirmation only)

Use these **only** once candidates are publication-worthy (≤2 survivors).

| Source | ~Price | Reliability | When |
|---|---|---|---|
| **IITH cluster prepaid** | **₹80/GPU-hr (~$0.96/hr)** | reliable, institutional, no application | **PRIMARY A100** (likely SLURM-based). |
| **Modal A100** | ~$2–5/hr | instant | Backup; $30/mo free covers a trimmed run. |
| **E2E Networks (India)** | A100-80GB **~$1.08/hr** | on-demand | Backup. |
| **RunPod community** | **~$1.19/hr** | on-demand | Backup. |
| **Vast.ai spot** | **$0.08–0.67/hr** | **interruptible** | Avoid for long training. |
| **GCP / AWS on-demand A100** | ~$3–4/hr | on-demand (+ quota friction) | Last resort. |

---

## Section 2 — Strategy (the funnel)

```
  free T4-class pools                          A100 (paid)
  Kaggle P100 + Modal T4 + Lightning T4   →    IITH primary (Modal/GCP backup)
  ───────────────────────────────────         ──────────────────────────────
  ALL exploration / hypothesis work            PUBLICATION confirmation ONLY
  run loss batches in PARALLEL across pools     ≤2 candidate loss×verifier pairs
  workload fits 16 GB → T4/P100 sufficient      full GSM8K, ≥3 seeds, stats
```

- **T4-class free pools = ALL exploration / hypothesis establishment.** Run
  different loss batches **in parallel** across the three pools. Because the
  workload fits in 16 GB, T4/P100 is sufficient — **A100 gives no benefit** here.
- **A100 = PUBLICATION confirmation ONLY**, after exploration narrows to **≤2**
  candidate loss×verifier pairings.
- **Always use the cheapest adequate GPU for each job.** Never use an A100 for
  prototyping — it wastes the shared credit pools for zero benefit on a 6 GB workload.

---

## Section 3 — Sequencing

Tied to the **locked algo priority** in [`RESEARCH_PLAN.md`](./RESEARCH_PLAN.md):
**(1) flat baseline `forward_kl` FIRST → (2) tree-loss training → (3) GBV verifier →
(4) online variants LAST.**

> **Smoke ALWAYS first** (any free pool, ~5 min) before any real run.

| Phase | Work | Resource | Config |
|---|---|---|---|
| **0 — Smoke** | crash-check every loss + verifier | any free pool | `--smoke`, ~5 min |
| **1 — `forward_kl` flat baseline + early tree-loss convergence** | train + `val_loss` curves + a **light BE sanity** per run | free T4-class (Kaggle/Modal/Lightning) | n≈100, K=3, T=1.0; light BE = n~100, K=3, matched verifier. **Defer heavy full-GSM8K eval.** |
| **2 — tree-loss ablation & ranking** | parallel batches, keep survivors | free T4-class (parallel) | n≈100 |
| **3 — GBV verifier** | verifier-aligned loss runs | free pools | n≈100 |
| **4 — online variants** | last, lowest priority | free pools / Lightning | n≈100 |
| **Confirmation** | surviving **≤2** candidates only | **A100 (IITH)** | full GSM8K **n=1319**, **≥3 seeds**, **paired-bootstrap + Holm–Bonferroni** — the only numbers that go in the paper. **A100 spend happens here.** |

### Which resource for which phase (decision table)

| If you are… | Use | Why |
|---|---|---|
| Smoke-testing | any free pool | ~5 min, near-zero cost |
| Running a single baseline / ablation batch | Kaggle P100 | longest sessions, 30 h/wk |
| Running 2nd/3rd parallel batch | Modal T4 + Lightning T4 | parallelize across pools |
| Out of weekly Kaggle quota | Modal/Lightning T4, or GCP T4 | spread the load |
| Confirming ≤2 publication candidates | **IITH A100** (Modal/GCP backup) | reliable, cheapest A100, full-scale stats |

### Engineer checklist (ordered)

1. **Smoke** on a free pool (`--smoke`) — confirm every loss + verifier runs.
2. **Phase 1:** `forward_kl` flat baseline on a free pool; collect train + `val_loss`
   curves + a light BE sanity (n~100, K=3, matched verifier). Add early tree-loss
   convergence checks. **Defer** the heavy full eval.
3. **Phase 2:** launch tree-loss ablation batches **in parallel** across Kaggle/Modal/
   Lightning; rank and keep survivors.
4. **Phase 3:** GBV verifier runs on free pools.
5. **Phase 4:** online variants last, on free pools / Lightning.
6. **Confirmation:** take the surviving **≤2** candidates to the **IITH A100** — full
   GSM8K (n=1319), ≥3 seeds, paired-bootstrap + Holm–Bonferroni. These are the paper numbers.

---

## Section 4 — GCP $300 trial how-to

**No upfront charge.** The card is for **verification only** — you are *not* charged
upfront. Usage draws down the **$300** credit; at trial end (90 days or credit
exhausted) the account **pauses** with **no auto-charge**. You are charged only if you
explicitly upgrade and then incur **overage** beyond the credit.

**GPUs require two things** (free trial alone has GPU quota = 0):

1. **Upgrade to a paid (Cloud Billing) account** — this still **spends the $300 first**
   before any real charge.
2. **Request a GPU quota increase:**
   - Console → **IAM & Admin → Quotas**.
   - **Filter** for **`NVIDIA T4 GPUs`** in a specific region (e.g. `us-central1`).
   - Select it → **Edit Quotas** → request **1**.
   - Justify: **"ML research"**.
   - Approval is auto/support-driven — **1×T4 often auto-approves in minutes–2 days**
     once a card is attached. **A100 quota is harder on fresh accounts.**

> **Request T4 first.** It is the workload-appropriate GPU and the fastest to approve.

---

## Section 5 — Action checklist (immediate)

1. **Keep using free pools now.** Kaggle quota is back ~**June 5**;
   **Lightning + Modal T4 are available immediately** — start there.
2. **Sign up Modal + Lightning** with your **@iith.ac.in** email.
3. *(Optional)* Grab the **GCP $300 trial** and **request a T4 quota** (Section 4).
4. **Load a small IITH prepaid balance** now so the eventual A100 confirmation isn't blocked.
5. **Pursue AWS/GCP research-credit grants** only as **non-blocking** future scaling — not a dependency.
