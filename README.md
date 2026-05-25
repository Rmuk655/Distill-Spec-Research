# DistillSpec Research

A focused ML systems research project exploring speculative decoding, knowledge distillation, and efficient LLM inference.

The goal is to develop and experimentally validate a novel modification to the DistillSpec framework, with emphasis on:
- draft-target alignment
- block efficiency / acceptance rate
- latency and throughput improvements
- reproducible, compute-efficient experimentation

## Current Status
Project scaffold and research documentation initialized.

## Research Scope
This project is intentionally narrow:
- one primary research direction
- one primary baseline architecture
- one evaluation plan
- reproducible experiments
- publication-oriented iteration

## Planned Experimental Stack
- HuggingFace Transformers
- Qwen3 draft/target models
- Weights & Biases for tracking
- benchmark-driven evaluation

## Reference Areas
- DistillSpec / online speculative decoding
- tree-based speculative verification
- draft-target alignment objectives
- inference efficiency metrics

## Repository Structure
- `docs/` — scope, operating principles, execution plan
- `src/` — training / evaluation code
- `experiments/` — run configs and outputs
- `scripts/` — launch helpers and analysis
- `logs/` — experiment notes and reports

## Principles
- correctness first
- reproducibility first
- narrow scope
- rigorous benchmarking
- no uncontrolled exploration
