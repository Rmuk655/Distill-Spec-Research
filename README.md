# SPEC: Knowledge Distillation for Tree-Based Speculative Decoding

**Rahul Thomas** (Columbia University) · **Krishnan R** (IIT Hyderabad)

---

## TL;DR

We jointly optimise draft-model alignment and tree-based verification for speculative decoding. Where prior work improves either the draft (DistillSpec, Online SD) or the verification algorithm (SpecInfer, Traversal, SpecTr) in isolation, this project trains the draft model with objectives that directly target acceptance under multi-path verification, and measures whether the combination improves block efficiency and robustness beyond either approach alone.

## Abstract

Speculative decoding accelerates LLM inference by having a lightweight draft model propose *K* tokens that a large frozen target model verifies in a single forward pass. Two lines of work improve this separately: *knowledge-distillation approaches* (DistillSpec, Online Speculative Decoding) train the draft to better match the target's distribution, while *tree-based verification algorithms* (SpecInfer, SpecTr, Traversal Verification) keep the draft fixed but accept more tokens by exploring multiple proposal paths simultaneously.

A natural question is whether these gains are additive — and if so, which distillation objective best complements tree verification. Standard forward-KL optimises a smooth proxy of acceptance probability but does not account for the product structure of block-level acceptance. We study five distillation objectives — forward KL, reverse KL, JSD, L1, and **Expected Block Efficiency (EBE)** — paired with six tree-verification algorithms across five datasets and two sampling temperatures. EBE directly optimises `Σ_blocks (1 + Σ_k Π_{i≤k} α_i)`, the exact quantity that block efficiency measures at inference time, and is computed from rejection events during live speculative decoding, adding zero overhead to the serving path.

We train Qwen3-0.6B as draft for Qwen3-8B targets, comparing all distillation × verifier combinations against an EAGLE baseline to characterise when each method contributes and by how much.

## Problem

| Gap | Existing approaches |
|---|---|
| Draft–target alignment methods assume a fixed simple verifier (α-sampling) | DistillSpec, Online SD |
| Tree-based verification methods assume a fixed, un-adapted draft model | SpecInfer, SpecTr, Traversal |
| Neither jointly optimises the draft *for* the verifier it will be paired with | — |

A draft model that is merely close to the target in KL sense need not produce acceptance patterns that exploit the product structure of block verification. We test whether matching the distillation objective to the verification algorithm closes this gap.

## Research Hypothesis

> Training the draft model with objectives aligned to tree-based verification (EBE, online EBE) will improve block efficiency and cross-setting robustness compared to:
> - tree-based verification with a fixed, un-trained draft model,
> - standard KL distillation paired with simple α-sampling,
> - KL distillation paired with tree-based verification,
> - EAGLE-style head baselines.
>
> The target setting is not necessarily maximum absolute throughput (EAGLE-3 is a strong ceiling) but *robustness* — consistent improvement across varying sampling temperatures, context lengths, and datasets.

## Baselines and Related Work

**Distillation / alignment**
- DistillSpec — [Arxiv 2310.08461](https://arxiv.org/abs/2310.08461)
- Online Speculative Decoding — [Arxiv 2310.07177](https://arxiv.org/abs/2310.07177)

**Tree-based verification (already implemented in this codebase)**
- Traversal Verification — [Arxiv 2505.12398](https://arxiv.org/abs/2505.12398)
- SpecTr — [Arxiv 2310.15141](https://arxiv.org/abs/2310.15141)
- SpecInfer — [Arxiv 2305.09781](https://arxiv.org/abs/2305.09781)

## Experimental Setup

| Component | Choice |
|---|---|
| Models | Qwen3-0.6B (draft) → Qwen3-8B (target, frozen) |
| GPU | T4 x2 (Kaggle free tier, 8B NF4) for exploration; A100 40 GB for paper confirmation only. P100 (sm_60) is broken with PyTorch 2.10+. CPU server (ATS Cloud) for GPT-2 convergence analysis only. |
| Distillation losses | forward KL · reverse KL · JSD · L1 · EBE · online KL · online EBE |
| Verification algorithms | GBV · Traversal · SpecInfer · BV · α-sampling · naive |
| Datasets | GSM8K · HumanEval · MATH500 · MTBench · Alpaca |
| Sampling temperatures | 0.6 · 1.0 |
| Primary metric | Block efficiency (avg accepted tokens per target call) |
| Secondary metrics | Throughput (tok/s), latency (ms/tok), task score |
| Tracking | Weights & Biases |

## Results

*Results pending — experiments in progress.*

## Quick Start — Phase-1 Iteration

Phase-1 runs a **tiered** session: **train + val_loss (inside trainer) + light BE sanity** (n=100, K=3, 1 matched verifier, GSM8K only). Heavy full-GSM8K eval is deferred to A100 confirmation.

```bash
# Smoke first (~5 min, every session):
python orchestration/experiment.py --config profiles/smoke --smoke --yes --losses forward_kl

# Phase-1 tiered baseline (train + val_loss + light BE, 1 verifier):
python orchestration/experiment.py --config profiles/train_one_loss --yes \
  --losses forward_kl --skip_existing --experiment_tag p1_forward_kl

# Phase-2 tiered tree variant (train + light BE, tree-matched verifiers):
python orchestration/experiment.py --config profiles/tree_variant_week --yes \
  --losses kl_tree --skip_existing --experiment_tag wk3_kl_tree
```

`--light_eval` (YAML: `experiment.light_eval: true`) is the flag that enables the tiered mode — it keeps train + merge + one light BE eval step and drops the full baseline + multi-dataset sweep. `--train_only` still exists as an optional flag for pure training burns (no eval at all) but is not the Phase-1 default.

See **[`gbv-research/docs/ENGINEER_PLAYBOOK.md`](gbv-research/docs/ENGINEER_PLAYBOOK.md)** for the full sequenced run plan.

## Setup and Reproduction

See **[`gbv-research/docs/SETUP.md`](gbv-research/docs/SETUP.md)** for installation, hardware requirements, WandB credentials, and first-run verification.

See **[`gbv-research/README.md`](gbv-research/README.md)** for the full pipeline, directory layout, loss descriptions, verifier modes, and results database schema.

## Repository Structure

| Directory | Contents |
|---|---|
| `gbv-research/` | Research codebase: algorithms, orchestration, tests, dashboard |
| `OSD/` | Upstream DistillSpec codebase — git submodule, unmodified (Liu et al.) |
| `GBV/` | Tree verification algorithms (Traversal, SpecInfer, GBV, BV) |
