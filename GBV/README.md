# Greedy Block Verification

This repository contains the reference implementation of greedy multi-path block verification.

The code is designed to work with standard autoregressive language models exposed in the `transformers` library (e.g., `AutoModelForCausalLM` with `DynamicCache`).

The repository implements a number of speculative decoding algorithms. Each follows the same general structure:
- **Drafting**: Construct a draft tree of nodes (contexts) via i.i.d. sampling from a draft model, conditioned on the current context.
- **Target Pass**: Perform a batched target model forward pass over the tree using a tree-based attention mask.
- **Verification**: Select a verified draft tree context and an extra residual token. Append the context and the residual to the current context.
- **Additional**: Profile metrics like KV-cache size, block efficiency, target pass cost, etc., and update caches accordingly.

Drafting, Target Pass, and Additional are constant for all speculative decoding algorithms here: the only variation is in Verification.
We implement the following 7 verification methods:
- `naive` from "Fast Inference from Transformers via Speculative Decoding"
- `nss` from "SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification"
- `specinfer` from "SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification"
- `spectr` from "SpecTr: Fast Speculative Decoding via Optimal Transport"
- `bv` from "Block Verification Accelerates Speculative Decoding"
- `traversal` from "Traversal Verification for Speculative Tree Decoding"
- `gbv` is our method, Greedy Multi-Path Block Verification

---

## Repository Structure

- **`main.py`**  
  Entry point for running speculative decoding and benchmarking experiments.  
  - Sets up tokenizer and target `p_model`, draft `q_model`.  
  - Calls the core speculative decoding loop to generate text and profile metrics (e.g., wall-clock time, block efficiency, KV-cache size).  
  - Can be adapted to run on different prompts, temperatures, tree sizes, and verification algorithms.

- **`inference_util.py`**  
  Core inference functions:
  - `iid_draft`: builds the draft tree by sampling K i.i.d. paths of length L from the draft model, while maintaining a batched KV-cache.
  - `target_tree_pass`: performs a batched forward pass of the target model over all tree prefixes using a custom attention mask that enforces tree structure.

- **`verifier.py`**  
  Implements the `TreeVerifier` class operating on the draft tree:
  - Builds a tree of `Node` objects from the drafted paths.
  - Provides multiple verification algorithms, including:
    - OT-based methods: `nss`, `naive`, `spectr`, `specinfer`.
    - Non-OT-based methods: `bv`, `gbv`, `traversal`.
  - Each verifier returns a verified node (prefix on a single drafted path) and a residual token.

- **`node.py`**  
  Defines the `Node` class used to represent tree nodes:
  - Stores token id, depth, string representation of the prefix, parent/children pointers.
  - Implements the necessary OTLP solver for the four OT-based above, since these are run at the individual Node level.

- **`util.py`**  
  General utility functions:
  - Helpers for expanding and slicing `DynamicCache` instances.
  - Loading models and data.
  
---

## Dependencies

This code relies on standard Python ML tooling and has been **tested with the following setup**:
- Python: **3.11.9** (conda-forge)
- PyTorch: **2.5.1**
- transformers: **4.55.2**
- datasets: **3.6.0**
- accelerate: **1.10.0**
- tqdm: ≥ 4.60

We recommend using **PyTorch ≥ 2.5** and **transformers ≥ 4.55**, as older versions may lack `DynamicCache` support or crash on bfloat16 causal masks (e.g., with Llama-3 / Qwen-3).

Once this is complete, you can perform a dataset run like this:
```bash
python main.py \
  --p_model "Qwen/Qwen3-32B" \
  --q_model "Qwen/Qwen3-0.6B" \
  --data "data.jsonl" \
  --L 8 \
  --K 4 \
  --max_new_tokens 128 \
  --mode "gbv" \
  --p_temp 1.0 \
  --q_temp 1.0 \
  --device "cuda" \
  --dtype fp16 \
  --seed 123

## 
