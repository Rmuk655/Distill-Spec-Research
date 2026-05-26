# Online EBE: Optimising Speculative Decoding via Expected Block Efficiency

**Rahul Thomas** (Columbia University) · **Krishnan R** (IIT Hyderabad)

---

## TL;DR

We replace the forward-KL training signal in online speculative decoding with a loss that directly maximises the expected number of tokens accepted per target-model call, improving block efficiency without increasing inference latency.

## Abstract

Speculative decoding accelerates LLM inference by having a small draft model propose *K* tokens that a large target model verifies in a single forward pass. Acceptance rate — how many draft tokens survive verification — is the key throughput lever, yet existing distillation objectives (forward KL, DistillSpec) optimise a proxy rather than acceptance directly.

We propose **online EBE** (Expected Block Efficiency loss), which computes the acceptance product `Π α_i` at positions where the draft was rejected during live speculative decoding, and uses it as the training signal. The loss is computed entirely from tokens already generated during serving, adding zero overhead to the inference path.

We train Qwen2.5-0.5B as draft for Qwen3-0.6B and Qwen3-8B targets, comparing online EBE against five baselines (forward KL, reverse KL, JSD, L1, online KL) across six verifier modes (GBV, traversal, SpecInfer, BV, alpha, naive) on GSM8K, Alpaca, and MATH.

## Results

*Results pending.*

## Setup and Reproduction

See **[`gbv-research/docs/SETUP.md`](gbv-research/docs/SETUP.md)** for installation, WandB credentials, and first-run verification.

See **[`gbv-research/README.md`](gbv-research/README.md)** for the full pipeline, directory layout, loss descriptions, verifier modes, and results database schema.

## Repository Structure

| Directory | Contents |
|---|---|
| `gbv-research/` | Main research codebase (algorithms, orchestration, tests, dashboard) |
| `OSD/` | Upstream OSD codebase — git submodule, unmodified (Liu et al., ICML 2024) |
| `GBV/` | Tree verification algorithms — Rahul Thomas |
| `OSD_ATTRIBUTION.md` | Modification notes for the one OSD file we adapted |
