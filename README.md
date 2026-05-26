# DistillSpec Research

A focused ML systems research project exploring speculative decoding, knowledge distillation, and efficient LLM inference.

The goal is to develop and experimentally validate a novel modification to the DistillSpec framework, with emphasis on:
- draft-target alignment
- block efficiency / acceptance rate
- latency and throughput improvements
- reproducible, compute-efficient experimentation

## Team

- **Rahul Thomas** (PhD Student, Columbia University) — Research Lead: direction, novelty, EBE math, publication
- **Krishnan R** (IIT Hyderabad) — Research Engineer: implementation, experiments, benchmarking
- **Target venue**: ICLR mid-September 2026

## Current Status

Phase 1 (laptop baseline) complete. Phase 3 (GSM8K eval of all 6 distillation losses) in progress.

## Research Scope

This project is intentionally narrow:
- one primary research direction
- one primary baseline architecture (Qwen2.5-0.5B draft → Qwen3-0.6B/8B target)
- one evaluation plan (block efficiency across 6 verifier modes)
- reproducible experiments
- publication-oriented iteration

## Experimental Stack

- HuggingFace Transformers + PEFT (LoRA)
- Qwen3 draft/target models
- Weights & Biases for experiment tracking
- SQLite (`db/results.db`) + Flask dashboard for local results

## Repository Structure

- `gbv-research/` — main research codebase (orchestration, docs, dashboard, tests)
  - `orchestration/` — crash-safe pipeline, configs, state files
  - `docs/` — PROJECT_CONTEXT.md, SETUP.md, extension guides
  - `db/` — results.db, checkpoints, W&B logs (gitignored)
- `OSD/` — training + eval runner (train_qwen3.py, run_all.py, viz_server.py)
- `GBV/` — novel tree verification algorithms
- `EAGLE/`, `adaspec-main/` — reference baselines (read-only)

## Reference Areas

- DistillSpec / online speculative decoding
- tree-based speculative verification (GBV, traversal, specinfer)
- draft-target alignment objectives (KL, EBE, reverse KL, JSD, L1)
- inference efficiency metrics (block efficiency, throughput)

## Principles

- correctness first
- reproducibility first
- narrow scope
- rigorous benchmarking
- no uncontrolled exploration
