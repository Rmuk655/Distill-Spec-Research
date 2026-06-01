# Compute Resourcing & Sequencing Plan — SpecDist

*Hardware funnel for a solo student researcher on a near-zero budget.*
*Project: speculative-decoding draft-model distillation with tree-aligned losses, GSM8K.*

This doc is the **what-to-run-where-and-when** plan. It does **not** repeat setup
instructions — follow the linked guides for those:

- [`KAGGLE.md`](./KAGGLE.md) — Kaggle free-tier setup (T4 x2 only; P100 broken; 4-bit NF4 config).
- [`MODAL.md`](./MODAL.md) — Modal on-demand A100/T4 setup ($30/mo credit).
- [`RESEARCH_PLAN.md`](./RESEARCH_PLAN.md) — research/ablation plan + **locked algo priority**.
- [`ENGINEER_PLAYBOOK.md`](./ENGINEER_PLAYBOOK.md) — dependency-aware sequenced run plan with exact commands.

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
| **Kaggle** | free, **~30 GPU-h / week** (resets weekly; **T4 x2 burns 2× → ~15 effective h/wk**) | **GPU T4 x2** (sm_75, only working option) | ~15/wk effective | P100 (sm_60) is **broken** with PyTorch 2.10+cu128 (requires sm_70+). T4 x2 is the only viable Kaggle GPU. Check `kaggle.com/me/quota`. |
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
| **Rahul Thomas (researcher)** | — (access TBD) | institutional | Collaborator A100 — confirm available hours before scheduling confirmation runs. |
| **Modal A100** | ~$2–5/hr | instant | Backup; $30/mo free covers a trimmed run. |
| **GCP A100** | ~$3–4/hr | on-demand (+ quota friction) | Last resort backup. |

---

## Section 2 — Strategy (the funnel)

### Research-valid compute tiers

> **Key principle: always use 8B teacher for any research decision.**
> The 4B→0.6B gap (~7×) is insufficient — loss rankings from a 4B teacher do not reliably predict 8B→0.6B (~13×) behaviour.
> Laptop (0.6B teacher) and Colab (4B teacher) configs are **crash-check / code-path only**, NOT research tiers.
> Quantized 8B NF4 (Kaggle T4x2 / Modal T4) gives the same loss rankings as full 8B BF16 (A100).

| Tier | Platform | Teacher | Purpose | Research-valid? |
|---|---|---|---|---|
| Crash check | Laptop (`laptop.yaml`) | Qwen3-0.6B | code doesn't crash | No — 0.6B teacher ≈ draft capacity; distillation signal is near-zero |
| Crash check / fallback | Colab free (`colab.yaml`) | Qwen3-4B BF16 | quota-exhausted fallback only | **No** — 4B teacher ≠ paper setting (8B); rankings not transferable |
| **Exploration** | **Kaggle T4 x2 + Modal T4** | **8B NF4** | rank losses, tune LR, kill losers — interchangeable, use whichever has quota remaining | **✓ Yes** |
| **Confirmation** | **A100** | **8B BF16** | paper numbers, multi-seed | **✓ Yes** |

```
  Kaggle T4 x2 (8B NF4 teacher)               →   A100 (8B BF16 teacher)
  ──────────────────────────────────           ──────────────────────────────
  Exploration: rank losses, tune LR,               PUBLICATION confirmation ONLY
  kill losers. Same 8B teacher as A100;            ≤2 candidate loss×verifier pairs
  loss rankings transfer to A100.                  full GSM8K, ≥3 seeds, stats

  (Modal T4, Lightning T4, GCP T4                  IITH primary · Rahul Thomas A100
   = additional parallel exploration slots)        · Modal A100 backup
```

- **Kaggle T4 x2 (8B NF4) = the minimum valid platform for research decisions.**
  Uses the same 8B teacher as the A100 confirmation; loss rankings established here
  transfer reliably to the full 8B BF16 run. Run batches in **parallel** across
  additional free T4 pools (Modal, Lightning) as needed.
- **Colab (`colab.yaml`, 4B teacher) = crash check / quota fallback only.** The
  4B teacher differs from the paper setting. Rankings established with a 4B teacher
  are **not** transferable to the 8B→0.6B (~13×) gap. **Never use this config for
  research direction.** Colab is a quota-exhausted fallback, not a research tier.
- **A100 = PUBLICATION confirmation ONLY**, after Kaggle exploration narrows to **≤2**
  candidate loss×verifier pairings.
- **Always use the cheapest adequate GPU for each job.** Never use an A100 for
  prototyping — it wastes the shared credit pools for zero benefit on a 6 GB workload.

---

## Section 3 — Sequencing

Tied to the **locked algo priority** in [`RESEARCH_PLAN.md`](./RESEARCH_PLAN.md):
**(1) flat baseline `forward_kl` FIRST → (2) tree-loss training → (3) GBV verifier →
(4) online variants LAST.**

> **Smoke ALWAYS first** (Tier 1 laptop, ~30-40 min) before any real run.

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
| Running a single baseline / ablation batch | Kaggle T4 x2 | longest sessions, ~15 effective h/wk (30 GPU-h/wk ÷ 2) |
| Running 2nd/3rd parallel batch | Modal T4 + Lightning T4 | parallelize across pools |
| Out of weekly Kaggle quota | Modal/Lightning T4, or GCP T4 ($300 trial, @iith.ac.in) | spread the load |
| Confirming ≤2 publication candidates | **IITH A100** · Rahul Thomas A100 · Modal/GCP A100 backup | cheapest/closest A100, full-scale stats |

### Engineer checklist (ordered)

1. **Smoke** on a free pool — confirm every loss + verifier runs without error.
2. **Phase 1:** `forward_kl` flat baseline on a free pool. Each session = train + `val_loss` curves + a light BE sanity (n≈100, K=3, matched verifier, GSM8K only). Defer the heavy full eval.
3. **Phase 2:** launch tree-loss ablation batches **in parallel** across Kaggle/Modal/Lightning; rank and keep survivors.
4. **Phase 3:** GBV verifier runs on free pools.
5. **Phase 4:** online variants last, on free pools / Lightning.
6. **Confirmation:** take the surviving **≤2** candidates to the A100 (IITH primary) — full GSM8K (n=1319), ≥3 seeds, paired-bootstrap + Holm–Bonferroni. These are the paper numbers.

> For exact run commands, YAML config names, and session protocol, see `deploy/PROGRESSION.md`.

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

1. **Use Lightning + Modal T4 now** (available immediately, no quota wait). Check Kaggle quota at `kaggle.com/me/quota`.
2. **Sign up Modal + Lightning** with your **@iith.ac.in** email (institutional email speeds verification and grant eligibility).
3. **Grab the GCP $300 trial** using your **@iith.ac.in** account and request a T4 quota (Section 4). No AWS.
4. **Confirm Rahul Thomas A100 access** — agree on available hours before scheduling confirmation runs.
5. **Load a small IITH prepaid balance** now so the eventual A100 confirmation isn't blocked.
6. **Always use `--skip_existing`** and attach checkpoint datasets before every session — never re-run work that is already in `results.db` or a saved checkpoint. See `KAGGLE.md` and `ENGINEER_PLAYBOOK.md` for restore steps.
7. **Pursue GCP/AWS research-credit grants** only as non-blocking future scaling — not a dependency.
