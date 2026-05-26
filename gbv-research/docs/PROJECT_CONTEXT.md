# DistillSpec Research Project — Context Reference

> This file is the project-specific companion to the generic skill files.
> Check it in alongside the code. Update it as decisions are made and results come in.
> Generic principles (research rigor, logging format, testing philosophy, scope management)
> live in the skill files. Project-specific facts live here.

---

## What This Project Is

Combining two ideas into one system:
- **DistillSpec** — train a small draft model via KL distillation to better match the target
- **Tree-based verification (GBV)** — generate K draft trees and verify with one batched target call

**Novel contribution**: replace the KL training objective with an Expected Block Efficiency (EBE) loss that directly optimises for tokens-accepted-per-target-call.

---

## Team

| Person | Role | Owns |
|---|---|---|
| Rahul Thomas | Columbia University PhD student / Research Lead | Research direction, novelty, EBE math, publication |
| Krishnan R | IIT Hyderabad / Research Engineer | Implementation, experiments, benchmarking, logging |
| PM | Research PM | Tracking, scope enforcement, coordination |

**Rahul Thomas must approve all scope changes.** PM enforces no new directions after Week 2 scope lock.

Krishnan R must NOT own: publication positioning, novelty calibration, broad literature exploration.
Rahul Thomas must NOT own: primary implementation, debugging, experiment tracking.

---

## Publication Target

- **Venue**: ICLR
- **Submission deadline**: ~mid-September (exact dates not yet published)
- **Minimum bar** (workshop): robustness results stronger than EAGLE-3 under distribution shift
- **Conference bar**: beat EAGLE-3 throughput in ≥1 setting
- Beating EAGLE-3 in all settings is NOT required

---

## Codebase Architecture

Three components integrated via subprocess:

```
OSD/                          ← training + eval runner
  train_qwen3.py              ← training loop; all 6 losses; crash-safe resume via ckpt_latest/
  run_all.py                  ← eval subprocess called by pipeline.py; writes to results.db
  results_db.py               ← SQLite schema; DB lives at gbv-research/db/results.db
  viz_server.py               ← web dashboard; reads db/results.db + orchestration/pipeline_state_*.json
  merge_lora.py               ← merge LoRA adapter into base weights
  online_osd.py               ← online speculative distillation (Phase 2 ablation)

GBV/                          ← verification algorithms (novel contribution)
  verifier.py                 ← TreeVerifier; dispatches to all 6 modes
  node.py                     ← OTLP solvers: specinfer, gbv, traversal, bv, alpha, naive
  main.py                     ← eval entrypoint; called as subprocess by run_all.py

orchestration/                ← pipeline coordination
  pipeline.py                 ← crash-safe multi-step orchestrator; Phases 1-4 + optional EAGLE
  run_all.py                  ← eval subprocess wrapper
  clean_restart.py            ← wipe db/ + reset state files + optional relaunch
  wandb_config.json.example   ← template; copy to wandb_config.json (gitignored)
  pipeline_state_laptop.json  ← step states for full run (separate from smoke)
  pipeline_state_laptop_smoke.json ← step states for smoke run

core/datasets/raw/            ← data
  gsm8k_train.jsonl           ← 7,473 training prompts (gitignored)
  gsm8k_30.jsonl, gsm8k_5.jsonl ← fixed eval sets (tracked)

db/                           ← all generated outputs (gitignored)
  checkpoints/                ← LoRA adapters + merged models
  results.db                  ← SQLite eval results
  logs/                       ← pipeline_output.log, be_progress.log
  wandb/                      ← W&B local sync
```

**Integration pattern**: `pipeline.py` launches `train_qwen3.py` (training) and `run_all.py` (eval) as subprocesses. `run_all.py` calls `GBV/verifier.py` for block-efficiency measurement and writes results to `db/results.db`. No deeper integration needed.

**Data format**: all components read JSONL files with a `"prompt"` field.

**Reference codebases** (read-only — for understanding, never imported):
- OSD: https://github.com/LiuXiaoxuanPKU/OSD
- GBV: https://anonymous.4open.science/r/GBV-BED8/README.md
- AdaSpec: https://github.com/yuezhouhu/adaspec (optional ablation reference)

---

## Models

| Role | Model | Notes |
|---|---|---|
| Draft (base) | Qwen2.5-0.5B | Pre-trained; LoRA fine-tuned during KD |
| Target (laptop) | Qwen3-0.6B | Fits in 6GB VRAM |
| Target (server) | Qwen3-8B | Requires A10G / A100 / 3090 |
| Shared tokenizer | Qwen3 tokenizer | vocab_size = 151936 — SD requirement |

Architecture mismatch (0.5B Qwen2.5 vs 0.6B Qwen3) is fine — SD only requires shared vocabulary.

**Dependency**: `transformers >= 4.51` required for Qwen3 support. Both codebases confirmed working after this upgrade.

---

## Hardware Environments

### Laptop (Krishnan R's machine — RTX 500, 6GB VRAM)
- Fits: Qwen3-0.6B target + Qwen2.5-0.5B draft simultaneously (~2.4GB)
- Does NOT fit: Qwen3-8B (~16GB)
- Use for: Phase 1 smoke tests, fast iteration, pipeline verification
- Recommended config: batch_size=1, grad_accum=4, steps=200, fp16=true

### Server (to be provisioned — A10G / A100 / 3090)
- Use for: Phase 2 full runs with Qwen3-8B target
- Recommended config: batch_size=4, grad_accum=1, steps=1000, bf16=true

**Promotion criteria (laptop → server)**: training runs 200 steps without crash + BE improves vs baseline (even slightly) + W&B logging confirmed + no NaN losses.

---

## Training Setup

### Loss Functions

| Loss | Description | Phase |
|---|---|---|
| `forward_kl` | KL(target ∥ student). DistillSpec canonical (Section 3.1). Mode-covering. | Phase 1 baseline |
| `ebe_token` | Acceptance-weighted MLE: `L = -Σ min(1,q/p).detach() × log p_draft(t)` | Phase 1 novel |
| `ebe_block` | Analytic gradient through ∏αᵢ. Confirms with Rahul Thomas before implementing. | Phase 2 |
| `reinforce` | Policy gradient with BE as reward. High variance. Needs server VRAM. | Phase 2+ |

**KL direction**: forward KL — KL(target ∥ student). NOT reverse KL (mode-seeking, collapses to peaked draft).

**`wrong_token_ids`** (OSD feature): NOT used. OSD-specific hard-example signal that introduces distributional shift. DistillSpec paper does not use it.

**Training mode**: teacher-sample (offline). Target generates continuations; both models score them. Trains toward target's actual inference distribution, not a static corpus.

### Training Data

| Flag | Source | Size | Use |
|---|---|---|---|
| `--dataset diverse` | 200 built-in diverse prompts | ~16k tokens/epoch | Smoke test / proof of concept |
| `--dataset gsm8k` | `data/gsm8k_train.jsonl` | 7,473 prompts | Full training runs |

Eval always uses `data/gsm8k_30.jsonl` (30 fixed prompts).

---

## Verification Algorithms (GBV modes)

All 6 modes are run in every eval step (both smoke and full pipeline).

| Mode | Role | What it rewards in the draft |
|---|---|---|
| `alpha` | Token-level acceptance rate (chain SD floor) | Per-token α |
| `naive` | Naive chain SD (explicit implementation of alpha) | Per-token acceptance |
| `bv` | Block Verification — accept/reject entire blocks | Block-level acceptance |
| `gbv` | **Generalised BV** — optimal transport over block prefixes, strictly > BV | Probability mass on accepted prefixes |
| `traversal` | **Empirically best** — longest surviving path | Joint prob of best surviving path |
| `specinfer` | Published multi-path baseline | Multi-path coverage, joint path probability |

**Primary comparison verifiers**: `specinfer` (published baseline) + `traversal` (empirically best) + `gbv` (novel contribution).

**Deprecated / not in pipeline**: `nss`, `spectr` — removed from eval sweep; present in `GBV/node.py` for reference.

---

## Canonical Baseline Numbers

Untrained Qwen2.5-0.5B draft → Qwen3-0.6B target, `gsm8k_30`, no training:

| Verifier | K | Baseline BE |
|---|---|---|
| specinfer | 3 | 2.520 |
| specinfer | 5 | 2.512 |
| gbv | 3 | 2.797 |
| gbv | 5 | 2.786 |
| traversal | 3 | 2.954 |
| traversal | 5 | 3.107 |

Any trained model must beat `specinfer K=3 = 2.520` to show improvement. Regression = investigate immediately.

---

## Experiment Structure

Pipeline runs 4 phases. Phase 4 (multi-dataset) is skipped in smoke mode.

| Phase | Step IDs | What runs |
|---|---|---|
| **Phase 1 — Baseline** | `eval_baseline_gsm8k` | Untrained draft, all 6 verifiers, gsm8k |
| **Phase 2 — Training** | `train_kl_gsm8k` + `merge_kl_gsm8k` ... × 6 losses | 1000 steps each (50 in smoke) |
| **Phase 3 — GSM8K Eval** | `eval_kl_gsm8k`, `eval_ebe_gsm8k` ... × 6 losses | All 6 verifiers, K=3+5, temps=0.6+1.0 |
| **Phase 4 — Multi-Dataset** | `eval_*_all` × 7 models | humaneval, math500, mtbench, alpaca |

**Trained models** (Phase 2 outputs, checkpoint name → label):
- `kl-gsm8k` → `kl` (forward_kl, DistillSpec baseline)
- `ebe-gsm8k` → `ebe` (Expected Block Efficiency, **novel**)
- `rev_kl-gsm8k` → `rev_kl` (reverse KL ablation)
- `jsd-gsm8k` → `jsd` (Jensen-Shannon ablation)
- `l1-gsm8k` → `l1` (L1 / total variation ablation)
- `online-gsm8k` → `online` (online OSD adaptation)

**Paper comparison table**: baseline / kl / ebe — with rev_kl, jsd, l1 as ablation rows.

**State files** (in `orchestration/`):
- `pipeline_state_laptop.json` — full 1000-step run state
- `pipeline_state_laptop_smoke.json` — smoke 50-step run state (separate; smoke "done" never blocks full run)

---

## Evaluation Metrics

| Metric | Primary purpose | Notes |
|---|---|---|
| Block efficiency (BE) | Core SD quality — avg tokens accepted per target call | Primary metric |
| Throughput (tok/s) | Systems relevance | Not comparable across hardware |
| Walltime (ms/tok) | End-to-end latency | Not comparable across hardware |
| Task accuracy (GSM8K %) | Quality preservation / bug check | If this drops >2%, something is wrong |

**W&B project**: `specdist-gbv` (set in `orchestration/wandb_config.json` per machine).
**W&B entity**: per-researcher — see `docs/SETUP.md` § 3 for multi-user credential setup.
Runs are tagged by experiment_tag (auto-generated timestamp); filter by `draft_label` column in results.db.

---

## Key Decisions Log

| Date | Decision | Owner | Rationale |
|---|---|---|---|
| — | KL direction: forward KL | Rahul Thomas | DistillSpec paper Section 3.1; mode-covering |
| — | Do NOT use `wrong_token_ids` | Rahul Thomas | Distributional shift hurts acceptance rate |
| — | Training mode: teacher-sample offline | Rahul Thomas | Targets actual inference distribution |
| — | Phase 1 target: Qwen3-0.6B (not 8B) | — | Only model fitting 6GB laptop |
| — | Draft init: Qwen2.5-0.5B pre-trained | Rahul Thomas | Has language priors; converges faster than random |
| — | GBV integration: subprocess only | Rahul Thomas | Draft is standard HF model; no deeper integration needed |
| — | OSD online mode: keep but don't use Phase 1 | Rahul Thomas | May use for online SD in later phases |
| — | EBE Phase 1: token-level surrogate | — | Differentiable, no verifier call; block-level is Phase 2 |
| — | REINFORCE: defer to server | — | High variance; doubles VRAM; not feasible on laptop |

---

## Explicit Scope Exclusions

The project SHALL NOT include (any of these appearing = flag to Rahul Thomas):
- Diffusion drafting
- Multi-device scheduling
- MoE routing
- Hardware specialization / custom CUDA kernels
- Multimodal inference
- Large-scale expert hierarchies
- Training foundational models from scratch
- More than one primary research direction simultaneously

---

## Phase Plan

| Phase | Target Week | Gate |
|---|---|---|
| Phase 1 | End of Week 3 | DistillSpec baseline + EBE token-level running on laptop; baseline numbers confirmed |
| Phase 2 | End of Week 4 | Block-level EBE loss (Rahul Thomas provides math in Week 3 meeting); server runs with Qwen3-8B |
| Phase 3 | Week 5+ | Online SD integration; robustness sweeps; EAGLE-3 comparison |

**Scope lock**: end of Week 2. After that, no new directions without explicit Rahul Thomas decision.

---

## Risks

| Risk | Impact | Status / Mitigation |
|---|---|---|
| GBV incompatible with Qwen3 | High | ✅ Resolved — upgrade transformers to ≥4.51 |
| EBE loss non-differentiable | High | ✅ Resolved — token-level surrogate is differentiable; block-level confirmed differentiable via ∂αᵢ/∂θ |
| Training diverges on Qwen3 | Medium | Fall back to lr=1e-5; reference AdaSpec |
| EAGLE-3 already does this | Medium | Rahul Thomas to confirm novelty before Week 2 scope lock |
| Server not provisioned in time | Medium | Phase 1 (laptop) must be complete before requesting server |

---

## Reference Papers

| Paper | Link | Role |
|---|---|---|
| DistillSpec | https://arxiv.org/abs/2310.08461 | Primary training methodology |
| Online Speculative Decoding | https://arxiv.org/abs/2310.07177 | OSD codebase basis; future online phase |
| Traversal Verification | https://arxiv.org/abs/2505.12398 | Primary tree verifier |
| SpecTr | https://arxiv.org/abs/2310.15141 | Tree verifier (completeness) |
| SpecInfer | https://arxiv.org/abs/2305.09781 | Published multi-path baseline |
| Leviathan et al. | https://arxiv.org/abs/2211.17192 | Original speculative decoding |
| EAGLE-3 | https://arxiv.org/abs/2503.01840 | Primary comparison target |
| Medusa | https://arxiv.org/abs/2401.10774 | Background |
| Sequoia | https://arxiv.org/abs/2402.12374 | Background |

---

## Generic Skill Files (companion to this doc)

These live in the `skills/` directory and apply to any ML project:

| Skill | When to use |
|---|---|
| `ml-experiment-log` | Logging any run; filling the results table; comparing conditions |
| `ml-research-principles` | Hypothesis formulation; interpreting results; deciding to pivot; publication readiness |
| `ml-research-swe` | Environment setup; config management; training loop hygiene; multi-environment runs |
| `ml-research-qa` | Writing tests; validating loss correctness; regression testing; debugging wrong results |
| `ml-scope-management` | PM dashboard; scope enforcement; decision log; meeting structure; phase gates |
