# DistillSpec Research Project — Context Reference

> Project-specific reference — check it in alongside the code.
> Update it as decisions are made and results come in.
> Architecture and file layout: **Codebase Architecture** section below.
> Research hypotheses, experiment matrix, pipeline phases: **GUIDE.md**.
> Design decisions and implementation notes: **DESIGN.md**.

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

Three components integrated via subprocess. `OSD/` and `GBV/` are reference-only
codebases. All active research code lives in `gbv-research/`.

```
2026 summer/                  ← git repo root
│
├── OSD/                      ← original OSD codebase (reference-only, not imported by pipeline)
│   └── distill/specInfer/    ← alpha-eval Generator still borrowed here for evaluate.py
│
├── GBV/                      ← verifier CLI (called as subprocess during eval; pending Phase 3 revert)
│   ├── main.py               ← eval entrypoint; called by evaluate.py
│   ├── verifier.py           ← TreeVerifier: all 6 verifier modes
│   ├── node.py               ← OTLP solvers
│   └── util.py               ← model loading helpers
│
└── gbv-research/             ← research project (canonical codebase)
    ├── algorithms/
    │   ├── train_qwen3.py    ← offline distillation trainer (all 5 offline losses)
    │   ├── online_serve.py   ← online SD adaptation trainer (online + online_ebe)
    │   ├── training_scaffold.py ← shared utilities (HW setup, model load, LoRA, checkpoints, W&B)
    │   └── distillspec_gbv/
    │       ├── losses/       ← forward_kl, reverse_kl, jsd, l1, ebe (class-based, tested)
    │       ├── verifiers/    ← runner.py, tree.py, otlp_registry.py (Phase 3: replaces GBV/main.py)
    │       └── trainer.py    ← model-family-agnostic replacement for train_qwen3.py
    ├── orchestration/
    │   ├── experiment.py       ← crash-safe orchestrator; Phases 1-4
    │   ├── evaluate.py        ← eval subprocess; calls GBV/main.py; writes results.db
    │   └── clean_restart.py  ← wipe db/ + reset state
    ├── core/
    │   ├── datasets/raw/     ← JSONL eval + training sets
    │   └── model_families/   ← Qwen/Gemma tokenizer helpers
    ├── dashboard/
    │   └── training_dashboard.py  ← web UI; reads db/results.db
    └── db/                   ← all generated outputs (gitignored)
        ├── checkpoints/      ← LoRA adapters + merged models
        ├── results.db        ← SQLite eval results
        └── logs/             ← pipeline_output.log, be_progress.log
```

**Integration pattern**: `experiment.py` launches `algorithms/train_qwen3.py` (offline
training) and `algorithms/online_serve.py` (online training) as subprocesses, then
`orchestration/evaluate.py` (eval). `evaluate.py` calls `GBV/main.py` as a subprocess
for block-efficiency measurement and writes results to `db/results.db`.

**Pending Phase 3 migration**: after Phase 3 eval confirms `algorithms/distillspec_gbv/verifiers/runner.py`
produces identical results to `GBV/main.py`, switch `evaluate.py` to runner.py and
revert `GBV/` to the unmodified reference repo.

**Data format**: all components read JSONL files with a `"prompt"` field.

**Reference codebases** (read-only snapshots — in `references/`, never imported):
- `references/adaspec/` — https://github.com/yuezhouhu/adaspec (ablation reference)

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

→ See **GUIDE.md § 6** for training loss details, data sources, and training-mode decisions.

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

→ See **GUIDE.md §§ 3 and 5** for the experiment matrix, pipeline phases, trained model labels, and paper comparison table.

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

