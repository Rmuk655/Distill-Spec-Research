"""
DistillSpec training — Qwen3-0.6B (draft, LoRA or full SFT) -> Qwen3-8B (target, frozen).

Laptop setup (default — fits in 4 GB VRAM):
  Draft   : Qwen/Qwen2.5-0.5B (494M, LoRA)
  Teacher : Qwen/Qwen3-0.6B (600M, frozen)
  Both share the same 151936-token vocabulary — valid for speculative decoding.

Colab / Modal setup (real experiment, needs ~18 GB):
  Draft   : Qwen/Qwen3-0.6B (fine-tuned via LoRA)
  Teacher : Qwen/Qwen3-8B (frozen, 16 GB alone)

Loss choices (--loss):
  forward_kl  : KL(target ∥ draft) — DistillSpec baseline, recommended default
  jsd         : Jensen–Shannon divergence — ablation
  l1          : L1 / total-variation distance — ablation
  reverse_kl  : KL(draft ∥ target) — mode-seeking, not recommended
  ebe         : Expected Block Efficiency surrogate, multi-token cumprod — novel
  ebe_single  : Single-token EBE = -mean(alpha) — ablation vs multi-token EBE

Laptop usage (train on GSM8K train split, eval on test split):
    python train_qwen3.py --loss forward_kl --steps 200 --dataset data/gsm8k_train.jsonl
    python train_qwen3.py --loss ebe        --steps 200 --dataset data/gsm8k_train.jsonl

Colab / Modal usage:
    python train_qwen3.py --draft Qwen/Qwen3-0.6B --target Qwen/Qwen3-8B \\
        --loss forward_kl --steps 500 --dataset data/gsm8k_train.jsonl

Merge LoRA for speculative decoding:
    python train_qwen3.py --merge_only --adapter ./checkpoints/qwen25-kl-gsm8k
"""

import argparse
import json
import math
import os
import sys
import time

# ---------------------------------------------------------------------------
# Shared training utilities — DB wiring, HW setup, LoRA, checkpointing.
# training_scaffold.py lives in the same directory (gbv-research/algorithms/).
# ---------------------------------------------------------------------------
from training_scaffold import (
    write_train_step as _write_train_step,
    setup_hw_opts    as _setup_hw_opts,
)

# Offline mode — prevents HF Hub network calls on cached / air-gapped setups.
# Default: ON on local machines (models already downloaded),
#          OFF automatically in cloud envs (Colab/Modal/Spaces) where the HF
#          cache is empty on every fresh session.
# Manual override: set TRANSFORMERS_OFFLINE=0 or =1 before running.

def _is_cloud_env() -> bool:
    """Return True when running inside Colab / Modal / HF Spaces / Kaggle."""
    _cloud_keys = (
        "COLAB_BACKEND_VERSION",   # Google Colab
        "COLAB_RELEASE_TAG",       # Google Colab (alt key)
        "MODAL_TASK_ID",           # Modal
        "SPACE_ID",                # HuggingFace Spaces
        "KAGGLE_KERNEL_RUN_TYPE",  # Kaggle
    )
    return any(k in os.environ for k in _cloud_keys)

if "TRANSFORMERS_OFFLINE" not in os.environ:
    if _is_cloud_env():
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "0"
        print(
            "[OSD] Cloud env detected — HF online mode ON. "
            "Models will be downloaded from HuggingFace Hub on first run."
        )
    else:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        print(
            "[OSD] HF offline mode ON (default). "
            "Models must already be cached locally. "
            "Set TRANSFORMERS_OFFLINE=0 before running to allow first-time downloads."
        )

# Reduce CUDA allocator fragmentation on small GPUs (T4, P100).
# Set automatically; override with PYTORCH_CUDA_ALLOC_CONF=<custom> in environment.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch
import torch.nn.functional as F
import transformers
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig, TaskType, PeftModel


def _dtype_kwargs(dtype) -> dict:
    """Return the correct dtype kwarg for AutoModelForCausalLM.from_pretrained().

    transformers >= 4.51 (required for Qwen3) supports dtype= and deprecates
    torch_dtype=.  Always use dtype= — no version check needed.
    """
    return {"dtype": dtype}


# ---------------------------------------------------------------------------
# Attention implementation — pick best available at import time
# ---------------------------------------------------------------------------

def _pick_attn_impl() -> str:
    """
    Returns the fastest attention backend available on this machine:
      flash_attention_2  — requires:  pip install flash-attn   (Ampere+ GPU: A100, H100)
                           2-4x faster attention, ~30 % less VRAM. Auto-skipped if not installed.
      sdpa               — built into PyTorch ≥ 2.0 (no extra deps). Fused kernel on any CUDA GPU.
      eager              — pure-PyTorch fallback, always works.
    The chosen value is passed to every from_pretrained() call as attn_implementation=...
    """
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        pass
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return "sdpa"
    return "eager"

_ATTN_IMPL = _pick_attn_impl()


# _setup_hw_opts is imported from training_scaffold above.


# ---------------------------------------------------------------------------
# 200-prompt dataset — diverse enough to generalise across topic distributions
# ---------------------------------------------------------------------------
PROMPTS = [
    # Math & logic
    "What is 2 + 2?", "What is the square root of 144?", "Prove that sqrt(2) is irrational.",
    "What is the derivative of sin(x)?", "Solve the equation x^2 - 5x + 6 = 0.",
    "What is Bayes' theorem?", "Explain the pigeonhole principle.",
    "What is the difference between permutations and combinations?",
    "What is Big-O notation? Give an example.", "What is the time complexity of merge sort?",
    # Python
    "Write a Python function that returns the factorial of n.",
    "Write a Python function to check if a string is a palindrome.",
    "Write a Python function to merge two sorted lists.",
    "Explain list comprehensions in Python with an example.",
    "What is a generator in Python and when would you use one?",
    "What is the difference between deepcopy and copy in Python?",
    "How does Python's garbage collector work?",
    "What is the GIL in Python?",
    "Write a Python decorator that measures execution time.",
    "What is the difference between @staticmethod and @classmethod?",
    # Data structures
    "What is a linked list and how does it differ from an array?",
    "Explain the difference between a stack and a queue.",
    "How does a hash table work?", "What is a binary search tree?",
    "What is the difference between BFS and DFS?",
    "Explain what a heap data structure is.", "What is a trie and when is it used?",
    "What is the difference between a graph and a tree?",
    "Explain dynamic programming with a simple example.",
    "What is memoization?",
    # ML / AI
    "What is gradient descent and why is it used in machine learning?",
    "What is overfitting and how do you prevent it?",
    "Explain the bias-variance tradeoff.", "What is a neural network at a high level?",
    "What is backpropagation?", "What is the softmax function?",
    "Explain what attention mechanism does in transformers.",
    "What is the difference between supervised and unsupervised learning?",
    "What is L1 vs L2 regularization?", "What is cross-entropy loss?",
    "What is a convolutional neural network used for?",
    "What is batch normalization?", "Explain dropout regularization.",
    "What is the vanishing gradient problem?", "What is transfer learning?",
    "What is the difference between precision and recall?",
    "What is an ROC curve?", "Explain k-fold cross-validation.",
    "What is a random forest?", "Explain boosting vs bagging.",
    # Systems / CS theory
    "What is the difference between a process and a thread?",
    "How does virtual memory work?", "What is a deadlock?",
    "Explain the CAP theorem.", "What is a race condition?",
    "What is the difference between stack and heap memory?",
    "What is a context switch?", "Explain what a semaphore is.",
    "What is a page fault?", "How does a CPU cache work?",
    # Networking
    "What is the difference between TCP and UDP?",
    "How does the TCP three-way handshake work?",
    "What is HTTPS and how does it differ from HTTP?",
    "What is DNS and how does it resolve a domain name?",
    "What is a load balancer?", "What is NAT?",
    "Explain what a CDN is.", "What is an API?",
    "What is REST vs GraphQL?", "What is WebSocket?",
    # Databases
    "Write a SQL query to find duplicate rows in a table.",
    "What is the difference between INNER JOIN and LEFT JOIN?",
    "What is database normalization?",
    "What is the difference between SQL and NoSQL?",
    "What is an index in a database and why is it useful?",
    "What is ACID in databases?", "What is eventual consistency?",
    "What is a foreign key?", "What is database sharding?",
    "What is the difference between a primary key and a unique key?",
    # OOP / software design
    "What are the four pillars of object-oriented programming?",
    "What is the SOLID principle?", "What is the Singleton pattern?",
    "What is the Observer pattern?", "What is dependency injection?",
    "What is the difference between composition and inheritance?",
    "What is a factory method pattern?", "What is polymorphism?",
    "What is encapsulation?", "What is an interface vs abstract class?",
    # General CS / concepts
    "Explain what recursion is with a simple example.",
    "What is the difference between compiled and interpreted languages?",
    "What is a virtual machine vs a container?",
    "What is continuous integration?", "What is a REST API?",
    "What is version control and why is it important?",
    "What is the difference between static and dynamic typing?",
    "What is functional programming?",
    "Explain what a closure is in programming.",
    "What is tail recursion?",
    # Science / general knowledge
    "What is Newton's second law of motion?",
    "Explain what a linked list is in one sentence.",
    "What is the capital of France?", "What is the capital of Japan?",
    "How does gradient descent work?",
    "What is the difference between RAM and ROM?",
    "Summarize the French Revolution in two sentences.",
    "What is quantum entanglement in simple terms?",
    "How does a transistor work?",
    "What is Moore's law?",
    # More ML/LLM topics (relevant to this research)
    "What is speculative decoding?",
    "What is knowledge distillation in deep learning?",
    "What is a language model?",
    "What is beam search?", "What is temperature in language model sampling?",
    "What is top-k sampling?", "What is nucleus sampling?",
    "What is RLHF?", "What is fine-tuning a language model?",
    "What is a tokenizer?",
    "What is perplexity in language modeling?",
    "What is the difference between encoder-only and decoder-only transformers?",
    "What is a key-value cache in transformers?",
    "What is LoRA fine-tuning?", "What is quantization in deep learning?",
    "What is flash attention?", "What is gradient checkpointing?",
    "What is mixed precision training?", "What is the Adam optimizer?",
    "What is learning rate scheduling?",
    # More coding
    "Write a function to check if a number is prime.",
    "Write a function to reverse a string.",
    "Write a function to find the maximum subarray sum.",
    "Write a function to check if a binary tree is balanced.",
    "Write a function to find all permutations of a string.",
    "Write a function to do binary search on a sorted array.",
    "Write a function to flatten a nested list.",
    "Write a function to find the first non-repeating character.",
    "Write a function to implement LRU cache.",
    "Write a function to compute Fibonacci numbers efficiently.",
    # Extra general
    "What is the Pythagorean theorem?",
    "What is Ohm's law?", "What is photosynthesis?",
    "What is the difference between mitosis and meiosis?",
    "What is a chemical bond?", "What is entropy in thermodynamics?",
    "What is the speed of light?", "What is a black hole?",
    "What is the central limit theorem?",
    "What is a p-value in statistics?",
    "What is standard deviation?",
    "What is the difference between correlation and causation?",
    "What is a confidence interval?",
    "What is A/B testing?", "What is linear regression?",
    "What is logistic regression?",
    "What is the difference between mean, median, and mode?",
    "What is a normal distribution?",
    "What is a Markov chain?",
    "What is reinforcement learning?",
    "What is Q-learning?", "What is a reward function?",
    "What is the explore-exploit tradeoff?",
    "What is multi-armed bandit problem?",
    "What is Monte Carlo sampling?",
    "What is importance sampling?",
    "What is variational inference?",
    "What is a generative adversarial network?",
    "What is a diffusion model?",
    "What is the difference between a VAE and a GAN?",
]
assert len(PROMPTS) > 0


# ---------------------------------------------------------------------------
# GPU utilisation helper (used by W&B logging — soft dependency on pynvml)
# ---------------------------------------------------------------------------

def _gpu_util_percent() -> "float | None":
    """Return current GPU utilisation (0-100) using pynvml, or None if unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--draft",  default="Qwen/Qwen2.5-0.5B",
                   help="Draft model to fine-tune. Laptop default: Qwen2.5-0.5B. "
                        "Colab/server: Qwen/Qwen3-0.6B (fine-tuned from 0.6B → 8B target).")
    p.add_argument("--target", default="Qwen/Qwen3-0.6B",
                   help="Frozen teacher. Laptop default: Qwen3-0.6B (fits in 4 GB). "
                        "Colab/server: Qwen/Qwen3-8B.")
    p.add_argument("--steps",  type=int,   default=1000,
                   help="Training steps. 200 for quick test; 1000 for overnight laptop run; "
                        "5000+ for Colab/server.")
    p.add_argument("--lr",     type=float, default=3e-5)
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--output", default="./checkpoints/qwen3-distill")
    p.add_argument("--loss",   default="forward_kl",
                   choices=["forward_kl", "reverse_kl", "jsd", "l1", "ebe", "ebe_single"],
                   help="Distillation loss. ebe = block-level EBE (novel). "
                        "ebe_single = single-token EBE ablation (-mean(alpha), no cumprod).")
    p.add_argument("--lora_r",     type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--log_every",  type=int, default=10)
    p.add_argument("--merge_only", action="store_true")
    p.add_argument("--adapter",    default=None)
    p.add_argument("--no_lora",    action="store_true",
                   help="Full SFT instead of LoRA. Needs ~8 GB+ VRAM for 0.6B draft.")
    p.add_argument("--dataset",    default=None,
                   help="Path to JSONL file with training prompts (field: 'prompt'). "
                        "Falls back to built-in PROMPTS list if not provided.")
    p.add_argument("--no_shuffle", action="store_true",
                   help="Disable dataset shuffling (default: shuffle each epoch).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=100,
                   help="Save a crash-safe checkpoint every N steps (overwrites ckpt_latest/). "
                        "0 to disable. On Colab/Modal: point --output to a persistent path.")
    p.add_argument("--milestone_every", type=int, default=200,
                   help="Save a permanent numbered checkpoint every N steps "
                        "(0 to disable). Unlike --save_every which overwrites ckpt_latest/, "
                        "milestone checkpoints are kept as ckpt_step_200/, ckpt_step_400/, etc. "
                        "Use these for offline model quality analysis across training.")
    p.add_argument("--max_checkpoints", type=int, default=5,
                   help="Max number of milestone checkpoints to keep on disk "
                        "(0 = keep all). Oldest are pruned when the limit is exceeded.")
    p.add_argument("--val_split", type=float, default=0.1,
                   help="Fraction of prompts held out as validation set (default 0.1 = 10%%). "
                        "Ignored if --val_dataset is provided.")
    p.add_argument("--val_dataset", default=None,
                   help="Separate JSONL file of validation prompts. If set, overrides --val_split.")
    p.add_argument("--val_every", type=int, default=50,
                   help="Compute validation loss every N steps (0 to disable). "
                        "A rising val loss while train loss falls is the primary red flag "
                        "for overfitting — the dashboard shows both curves.")
    p.add_argument("--load_in_4bit", action="store_true",
                   help="Load the target (teacher) model in 4-bit NF4 (QLoRA mode). "
                        "Reduces teacher VRAM from ~16 GB to ~5 GB so an 8B target "
                        "fits on a free Colab T4 (15 GB). Requires bitsandbytes. "
                        "ACCURACY NOTE: 4-bit is safe for training (teacher is frozen). "
                        "For final published eval numbers use fp16 teacher to match baselines.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile() the draft model (PyTorch 2.0+). "
                        "Gives 10-30%% speedup after a one-time ~60s compile overhead. "
                        "Not recommended on Windows (triton support is limited).")
    # ── Health checks ────────────────────────────────────────────────────────
    p.add_argument("--nan_action", default="stop",
                   choices=["stop", "skip", "warn"],
                   help="What to do when a NaN/Inf loss is detected. "
                        "stop=save checkpoint and abort (default), "
                        "skip=discard the step and continue, "
                        "warn=log and continue (NaN may propagate to weights).")
    p.add_argument("--health_every", type=int, default=100,
                   help="Print a bordered health-check summary every N steps (0 to disable). "
                        "Shows: loss trend (first-10 vs last-10 avg), NaN count, "
                        "val loss status, PPL ratio vs baseline.")
    p.add_argument("--ppl_check_every", type=int, default=200,
                   help="Measure draft PPL vs the pre-training baseline every N steps "
                        "(0 to disable).  Warns if PPL > baseline × --ppl_threshold. "
                        "Pass condition: PPL within 20%% of baseline (--ppl_threshold 1.20).")
    p.add_argument("--ppl_threshold", type=float, default=1.25,
                   help="PPL warning multiplier relative to pre-training baseline "
                        "(default 1.25 = warn at 25%% above). "
                        "Set to 1.20 to match the researcher health-check table.")
    p.add_argument("--early_stop_patience", type=int, default=0,
                   help="Stop training if val loss has not improved for N consecutive "
                        "val checks (0 = disabled, default). "
                        "Useful for overnight server runs to avoid wasting GPU time "
                        "on a model that is already overfitting.")
    # ── W&B ──────────────────────────────────────────────────────────────────
    p.add_argument("--wandb_project", default="distillspec",
                   help="Weights & Biases project name (default: distillspec). "
                        "All training runs go here; use --wandb_group to bucket "
                        "sweep runs together.")
    p.add_argument("--wandb_entity", default=None,
                   help="W&B entity (team or username). "
                        "Defaults to the account you're logged in as (wandb login). "
                        "Useful when logging to a shared team workspace.")
    p.add_argument("--wandb_group", default=None,
                   help="W&B run group — set to a sweep name so all LR-sweep runs "
                        "appear under one collapsible group in the UI. "
                        "E.g. --wandb_group lr_sweep_gsm8k")
    p.add_argument("--no_wandb", action="store_true",
                   help="Disable W&B logging for this run (even if wandb is installed). "
                        "Useful for quick debugging runs or air-gapped servers.")
    # ── Sweep-friendly hyperparameters ───────────────────────────────────────
    # These replace the previously hardcoded values so W&B sweep agents can
    # vary them without editing source code.
    p.add_argument("--teacher_temp", type=float, default=0.8,
                   help="Teacher sampling temperature (default: 0.8). "
                        "Controls both the generate() call and logit rescaling: "
                        "output_scores = raw_logits/T, so we multiply by T to recover "
                        "raw_logits before computing the distillation loss. "
                        "Higher T = softer distribution = easier distillation target.")
    p.add_argument("--ebe_kl_weight", type=float, default=0.1,
                   help="KL regularizer weight in EBE loss (default: 0.1). "
                        "L_total = L_EBE + ebe_kl_weight * L_KL. "
                        "0.0 = pure EBE (unstable). 1.0 = mostly KL. Sweep: 0.01–0.5.")
    p.add_argument("--jsd_alpha", type=float, default=0.5,
                   help="Interpolation weight for JSD loss (default: 0.5 = symmetric). "
                        "JSD = alpha*KL(p_s||M) + (1-alpha)*KL(p_t||M), M=alpha*p_s+(1-alpha)*p_t. "
                        "0 = reverse KL only, 1 = forward KL only. Sweep: 0.1–0.9.")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Gradient norm clipping threshold (default: 1.0). "
                        "0 = disabled. Helps stabilise EBE and reverse-KL losses.")
    p.add_argument("--lora_dropout", type=float, default=0.05,
                   help="LoRA adapter dropout probability (default: 0.05). "
                        "Sweep: 0.0–0.2. 0 = no dropout (good for small datasets).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def forward_kl_loss(student_logits, teacher_logits):
    """KL(target ∥ student) — DistillSpec Section 3.1."""
    log_s = F.log_softmax(student_logits, dim=-1)
    p_t   = F.softmax(teacher_logits, dim=-1)
    return -(p_t * log_s).sum(dim=-1).mean()


def reverse_kl_loss(student_logits, teacher_logits):
    """KL(draft ∥ target) — mode-seeking, numerically stable.

    Bug: Qwen3 applies a −inf logit bias to forbidden tokens (pad_token, etc.).
      After log_softmax those tokens have log_t = −inf.
      Then  log_s − log_t = finite − (−inf) = +inf.
      Forward: p_s * inf = NaN (when p_s==0 due to float32 underflow) or inf.
      Backward (torch.where approach): dL/dp_s = 0 * inf = NaN — corrupts
        LoRA weights on first update, making ALL subsequent steps NaN too.

    Fix: clamp log_t BEFORE the subtraction so neither the forward value
    nor its gradient ever involves inf.  Tokens the teacher forbids get
    log_t = -100 (prob ≈ exp(-100) ≈ 0), so their log-ratio contribution
    is bounded and the gradient flows cleanly.
    """
    log_t = F.log_softmax(teacher_logits, dim=-1).clamp(min=-100.0)
    log_s = F.log_softmax(student_logits, dim=-1)
    p_s   = F.softmax(student_logits, dim=-1)
    return (p_s * (log_s - log_t)).sum(dim=-1).mean()


def jsd_loss(student_logits, teacher_logits, alpha=0.5):
    p_s = F.softmax(student_logits, dim=-1)
    p_t = F.softmax(teacher_logits, dim=-1)
    m   = alpha * p_s + (1 - alpha) * p_t
    log_m = m.log().clamp(min=-1e9)
    kl_s = (p_s * (p_s.log().clamp(min=-1e9) - log_m)).sum(-1).mean()
    kl_t = (p_t * (p_t.log().clamp(min=-1e9) - log_m)).sum(-1).mean()
    return alpha * kl_s + (1 - alpha) * kl_t


def l1_loss(student_logits, teacher_logits):
    """L1 / total-variation distance between student and teacher distributions.
    TV(p, q) = 0.5 * sum_v |p(v) - q(v)|, averaged over positions.
    Simple to implement; weaker distillation signal than KL/JSD in practice.
    Use as ablation, not primary objective.
    """
    p_s = F.softmax(student_logits, dim=-1)
    p_t = F.softmax(teacher_logits, dim=-1)
    return (p_s - p_t).abs().sum(dim=-1).mean() * 0.5


_EBE_BLOCK_LEN = 8   # must match GBV/main.py --L default (inference block length)


def ebe_loss(student_logits, teacher_logits, token_ids, kl_weight=0.1,
             block_len: int = _EBE_BLOCK_LEN):
    """
    Block-level Expected Block Efficiency surrogate + KL regulariser.

    Training objective mirrors the inference metric exactly:
        L_EBE = -Σ_{b=1}^{T/L} (1 + Σ_{k=1}^{L} Π_{i=1}^{k} α_{bL+i})  / n_blocks

    Each block of L=8 tokens gets its own cumprod — matching the L=8 speculative
    window used at inference.  This avoids the "giant single block" distortion where
    early tokens in a long sequence accumulate an unrealistically large gradient
    multiplier (Π of 128 alphas instead of Π of 8 alphas).

    Why L=8 specifically:
    - Inference calls the draft model with block_len=L=8 (evaluate.py default --L 8)
    - The accepted token count per target call is τ+1 ∈ [1, L+1]
    - Training with the same L ensures the gradient scale matches the inference scale

    Numerical safety:
    - alpha is clamped to ≥ 1e-6 before cumprod to prevent NaN gradients.
      torch.cumprod backward divides by alpha[j] internally; if alpha[j]=0 exactly
      (i.e. q_target≈0 for that token) this produces NaN.  The clamp fixes it.

    KL regulariser (λ ≈ 0.1): EBE gradient vanishes for under-estimated tokens
    (α = 1 already).  KL supplies gradient there and prevents PPL collapse.
    """
    log_s = F.log_softmax(student_logits, dim=-1)          # [T, V]
    log_t = F.log_softmax(teacher_logits, dim=-1).detach() # [T, V]  teacher fixed

    # Per-token log probs of the actual generated tokens
    log_p = log_s.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)  # [T]
    log_q = log_t.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)  # [T]

    # Acceptance probability — gradient flows through log_p (NOT detached)
    # Clamp ≥ 1e-6: prevents NaN in cumprod backward when alpha≈0
    alpha = torch.exp(torch.clamp(log_q - log_p, max=0.0)).clamp(min=1e-6)  # [T]

    # Per-block EBE: split sequence into blocks of block_len, sum within each block
    # This matches inference where each target call sees exactly block_len draft tokens.
    T = alpha.shape[0]
    n_blocks = max(1, T // block_len)
    ebe = torch.zeros((), device=alpha.device, dtype=alpha.dtype)  # 0-dim scalar
    for b in range(n_blocks):
        blk = alpha[b * block_len : (b + 1) * block_len]   # [≤ block_len]
        ebe = ebe - (1.0 + torch.cumprod(blk, dim=0).sum())
    ebe = ebe / n_blocks   # average over blocks → stable scale regardless of seq len

    # KL regulariser: forward KL keeps language quality stable
    kl = F.kl_div(log_s, log_t.exp(), reduction="batchmean")

    loss = ebe + kl_weight * kl

    # Diagnostic: return a richer dict so the training loop can log the full
    # α distribution, not just the mean.  Callers that only want a scalar use [0].
    with torch.no_grad():
        a = alpha.detach()
        frac_below_95  = (a < 0.95).float().mean().item()   # tokens with real EBE gradient
        frac_below_80  = (a < 0.80).float().mean().item()   # strongly rejected tokens
        alpha_min      = a.min().item()
        alpha_std      = a.std().item()

    return loss, {
        "mean":         a.mean().item(),
        "std":          alpha_std,
        "min":          alpha_min,
        "frac_lt_0.95": frac_below_95,   # nonzero → EBE gradient is flowing
        "frac_lt_0.80": frac_below_80,
    }


def ebe_single_loss(student_logits, teacher_logits, token_ids):
    """Single-token EBE surrogate — Loss = −mean(α) per token.

    Simplified ablation of ebe_loss: no block_len, no cumprod, no KL regulariser.

        L = −mean_{t=1}^{T}( α_t )      where  α_t = min(1, q_t / p_t)

    Direct interpretation: maximise the average per-token acceptance rate.

    Why this matters as an ablation:
    - Multi-token EBE (ebe_loss) chains 8 α's with cumprod.  If that chain
      is the source of gradient instability or vanishing, single-token EBE
      will train stably and the comparison tells us exactly which component
      was responsible.
    - No λ means a single, clean objective — no two losses fighting each other.
    - Dense gradient: every token contributes equally (no block-boundary bias).

    Returns (loss, alpha_stats_dict) — same shape as ebe_loss so callers can
    use the same logging code.
    """
    log_s = F.log_softmax(student_logits, dim=-1)          # [T, V]
    log_t = F.log_softmax(teacher_logits, dim=-1).detach() # [T, V]  teacher fixed

    log_p = log_s.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)  # [T]
    log_q = log_t.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)  # [T]

    # α = min(1, q/p) = exp(min(0, log_q − log_p)).  Clamp ≥ 1e-6 for grad safety.
    alpha = torch.exp(torch.clamp(log_q - log_p, max=0.0)).clamp(min=1e-6)  # [T]

    loss = -alpha.mean()

    with torch.no_grad():
        a = alpha.detach()
        frac_below_95 = (a < 0.95).float().mean().item()
        frac_below_80 = (a < 0.80).float().mean().item()

    return loss, {
        "mean":          a.mean().item(),
        "std":           a.std().item(),
        "min":           a.min().item(),
        "frac_lt_0.95":  frac_below_95,
        "frac_lt_0.80":  frac_below_80,
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_prompts(dataset_path=None):
    """Load prompts from JSONL file, or fall back to built-in PROMPTS list."""
    if dataset_path is None:
        return PROMPTS
    import json
    prompts = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            p = obj.get("prompt") or obj.get("question") or obj.get("instruction") or ""
            if p:
                prompts.append(p.strip())
    if not prompts:
        raise ValueError(f"No prompts found in {dataset_path}")
    print(f"Loaded {len(prompts)} training prompts from {dataset_path}")
    return prompts


# ---------------------------------------------------------------------------
# Merge utility
# ---------------------------------------------------------------------------

def merge_and_save(args):
    print(f"Loading base: {args.draft}")
    base = transformers.AutoModelForCausalLM.from_pretrained(
        args.draft, **_dtype_kwargs(torch.bfloat16))
    print(f"Loading LoRA: {args.adapter}")
    model  = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()
    out = args.adapter + "_merged"
    merged.save_pretrained(out)
    transformers.AutoTokenizer.from_pretrained(args.draft).save_pretrained(out)
    print(f"Merged -> {out}")


# ---------------------------------------------------------------------------
# Crash-safe checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(draft_model, optimizer, step: int, args,
                     losses: list, accept_weights: list, milestone: bool = False,
                     best_val_loss: float = None, val_no_improve_count: int = None):
    """
    Save a crash-safe checkpoint to {output}/ckpt_latest/ (always overwrites).
    If milestone=True also saves a permanent numbered copy to {output}/ckpt_step_{step}/.

    Milestone checkpoints are never overwritten — use them for offline model
    quality analysis (load with from_pretrained, run speculative decoding eval).
    --max_checkpoints controls how many milestones are kept; oldest are pruned first.

    On Google Colab / Modal: set --output to a persistent volume path so
    checkpoints survive session death:
        Colab:  --output /content/drive/MyDrive/OSD/checkpoints/kl-run
        Modal:  --output /vol/checkpoints/kl-run
    """
    import shutil

    # ── Crash-safe latest checkpoint (always overwrites) ────────────────────
    ckpt_dir = os.path.join(args.output, "ckpt_latest")
    os.makedirs(ckpt_dir, exist_ok=True)
    if args.no_lora:
        torch.save(draft_model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
    else:
        draft_model.save_pretrained(ckpt_dir)
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))

    state = {
        "step": step,
        "loss_history": losses[-500:],
        "accept_weight_history": accept_weights[-500:],
    }
    # Persist early-stop state so patience counter survives crash-resume
    if best_val_loss is not None and math.isfinite(best_val_loss):
        state["best_val_loss"] = best_val_loss
    if val_no_improve_count is not None:
        state["val_no_improve_count"] = val_no_improve_count
    # Atomic write: write to .tmp then os.replace() — prevents a partial file
    # if the process is killed mid-write (Colab session death, Modal timeout, OOM).
    _state_path = os.path.join(args.output, "training_state.json")
    _state_tmp  = _state_path + ".tmp"
    with open(_state_tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(_state_tmp, _state_path)   # atomic on POSIX; near-atomic on Windows
    print(f"  [ckpt] step {step}/{args.steps} → {ckpt_dir} (crash-safe)")

    # ── Milestone checkpoint (permanent numbered copy) ───────────────────────
    if not milestone:
        return

    milestone_dir = os.path.join(args.output, f"ckpt_step_{step:05d}")
    shutil.copytree(ckpt_dir, milestone_dir, dirs_exist_ok=True)
    # Copy training_state.json alongside the milestone for provenance
    shutil.copy(
        os.path.join(args.output, "training_state.json"),
        os.path.join(milestone_dir, "training_state_at_save.json"),
    )
    print(f"  [milestone] step {step} → {milestone_dir}")

    # ── Prune old milestones if max_checkpoints is set ───────────────────────
    max_ckpt = getattr(args, "max_checkpoints", 5)
    if max_ckpt > 0:
        import glob as _glob
        all_milestones = sorted(
            _glob.glob(os.path.join(args.output, "ckpt_step_*")),
            key=lambda d: int(os.path.basename(d).split("_")[-1])
        )
        while len(all_milestones) > max_ckpt:
            oldest = all_milestones.pop(0)
            shutil.rmtree(oldest, ignore_errors=True)
            print(f"  [milestone] pruned oldest: {os.path.basename(oldest)}")


# ---------------------------------------------------------------------------
# Validation loss
# ---------------------------------------------------------------------------

def _compute_val_loss(draft_model, target_model, val_prompts, tok_cache,
                      tokenizer, args, device, max_prompts: int = 10):
    """
    Compute validation loss on a held-out set without updating weights.

    Returns (mean_val_loss, mean_accept_weight_or_None).

    The val set is never shuffled — same prompts each call so loss curves
    are comparable across steps.  We cap at max_prompts for speed; the
    val set itself may be larger (configured by --val_split / --val_dataset).

    Red flag: if val_loss rises by > 10% after reaching its minimum while
    train_loss is still falling, the model is overfitting.  The dashboard
    highlights this automatically.
    """
    _TEACHER_TEMP = getattr(args, "teacher_temp", 0.8)
    sample = val_prompts[:max_prompts]
    val_losses, val_aws = [], []

    draft_model.eval()
    with torch.no_grad():
        for prompt in sample:
            # Tokenise on-the-fly for val prompts not yet in cache
            if prompt not in tok_cache:
                _msgs = [{"role": "user", "content": prompt}]
                _txt = tokenizer.apply_chat_template(
                    _msgs, tokenize=False, add_generation_prompt=True)
                tok_cache[prompt] = tokenizer(_txt, return_tensors="pt").input_ids

            prompt_ids = tok_cache[prompt].to(device)
            plen = prompt_ids.shape[1]

            with torch.no_grad():
                gen_out = target_model.generate(
                    prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=_TEACHER_TEMP,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            full_ids = gen_out.sequences.clone()  # break inference-tensor chain: generate() uses @inference_mode internally

            # Guard: skip val prompt if target generated 0 tokens (same fix as train loop)
            if not gen_out.scores:
                continue

            t_log = torch.stack(gen_out.scores, dim=0).squeeze(1).float() * _TEACHER_TEMP

            student_logits = draft_model(full_ids).logits[:, :-1, :].float()
            s_log   = student_logits[:, plen-1:, :].squeeze(0)
            gen_ids = full_ids[0, plen:]

            if args.loss == "forward_kl":
                v_loss = forward_kl_loss(s_log, t_log)
                v_aw = None
            elif args.loss == "reverse_kl":
                v_loss = reverse_kl_loss(s_log, t_log)
                v_aw = None
            elif args.loss == "jsd":
                v_loss = jsd_loss(s_log, t_log,
                                  alpha=getattr(args, "jsd_alpha", 0.5))
                v_aw = None
            elif args.loss == "l1":
                v_loss = l1_loss(s_log, t_log)
                v_aw = None
            elif args.loss == "ebe":
                v_loss, v_aw_dict = ebe_loss(s_log, t_log, gen_ids,
                                             kl_weight=getattr(args, "ebe_kl_weight", 0.1))
                v_aw = v_aw_dict["mean"] if isinstance(v_aw_dict, dict) else v_aw_dict
                if v_aw is not None:
                    val_aws.append(v_aw)
            elif args.loss == "ebe_single":
                v_loss, v_aw_dict = ebe_single_loss(s_log, t_log, gen_ids)
                v_aw = v_aw_dict["mean"] if isinstance(v_aw_dict, dict) else v_aw_dict
                if v_aw is not None:
                    val_aws.append(v_aw)
            else:
                v_loss = forward_kl_loss(s_log, t_log)
                v_aw = None

            val_losses.append(v_loss.item())

    draft_model.train()
    mean_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")
    mean_aw   = sum(val_aws)   / len(val_aws)    if val_aws    else None
    return mean_loss, mean_aw


# ---------------------------------------------------------------------------
# PPL health check — teacher-free, measures draft language-model quality
# ---------------------------------------------------------------------------

def _compute_ppl(draft_model, prompts_sample, tok_cache, tokenizer, device,
                 max_prompts: int = 5) -> float:
    """
    Compute autoregressive perplexity of the draft model on a small prompt sample.

        PPL = exp(mean cross-entropy over all tokens)

    Used for the health-check pass condition:
        PPL within --ppl_threshold × baseline (default 1.25×) → training is healthy.
        PPL above threshold → possible PPL collapse / catastrophic forgetting.

    This is teacher-free (no target-model forward pass) so it runs fast.
    We reuse the existing tok_cache when possible; val prompts not yet in the
    cache are tokenised on the fly and cached for future calls.

    Lower PPL = better language model quality.
    """
    sample = prompts_sample[:max_prompts]
    total_nll  = 0.0
    total_toks = 0

    draft_model.eval()
    with torch.no_grad():
        for prompt in sample:
            if prompt not in tok_cache:
                _msgs = [{"role": "user", "content": prompt}]
                _txt  = tokenizer.apply_chat_template(
                    _msgs, tokenize=False, add_generation_prompt=True)
                tok_cache[prompt] = tokenizer(_txt, return_tensors="pt").input_ids
            input_ids = tok_cache[prompt].to(device)
            with torch.inference_mode():
                out = draft_model(input_ids, labels=input_ids)
            # out.loss is mean cross-entropy; recover sum so we can average later.
            n_toks = max(1, input_ids.shape[1] - 1)
            total_nll  += out.loss.item() * n_toks
            total_toks += n_toks

    draft_model.train()
    if total_toks == 0:
        return float("nan")
    return math.exp(total_nll / total_toks)


# ---------------------------------------------------------------------------
# Periodic health report — printed as a bordered box in the log
# ---------------------------------------------------------------------------

def _print_health_report(step: int, total_steps: int,
                         losses: list, val_loss_history: list,
                         nan_count: int,
                         current_ppl, baseline_ppl,
                         best_val_loss: float,
                         ppl_threshold: float = 1.25) -> None:
    """
    Print a clear, bordered health summary to stdout.

    Checks (matching the researcher health-check table):
      1. Loss trend      — last-10 avg vs first-10 avg; should be decreasing.
      2. NaN events      — any NaN/Inf in loss over the run.
      3. Val loss        — latest val vs minimum seen; ⚠ if >10% above minimum.
      4. PPL             — current vs pre-training baseline; ⚠ if >threshold×.
    """
    W = 64  # box width

    # 1. Loss trend
    if len(losses) >= 20:
        first10 = sum(losses[:10]) / 10
        last10  = sum(losses[-10:]) / 10
        trend   = "↓ IMPROVING" if last10 < first10 else "↑ WORSENING"
        trend_str = f"{first10:.4f} → {last10:.4f}  ({trend})"
    elif len(losses) > 0:
        trend_str = f"latest={losses[-1]:.4f}  (< 20 steps — need more data)"
    else:
        trend_str = "no data"

    # 2. NaN events
    nan_str = "✓ none" if nan_count == 0 else f"⚠ {nan_count} NaN/Inf event(s)"

    # 3. Val loss
    valid_vals = [v for v in val_loss_history if math.isfinite(v)]
    if valid_vals:
        last_val  = valid_vals[-1]
        val_min   = min(valid_vals)
        threshold = val_min + 0.10 * abs(val_min)   # works for both +ve and -ve losses
        if last_val > threshold:
            val_str = f"{last_val:.4f}  ⚠ RISING >10%% above min={val_min:.4f}"
        else:
            val_str = f"{last_val:.4f}  ✓ (min={val_min:.4f})"
    else:
        val_str = "not computed yet (--val_every controls frequency)"

    # 4. PPL
    if current_ppl is not None and baseline_ppl is not None and math.isfinite(current_ppl):
        ratio = current_ppl / baseline_ppl if baseline_ppl > 0 else float("inf")
        if ratio > ppl_threshold:
            ppl_str = (f"{current_ppl:.2f}  ⚠ COLLAPSED "
                       f"({ratio:.2f}× baseline={baseline_ppl:.2f})")
        else:
            ppl_str = (f"{current_ppl:.2f}  ✓ "
                       f"({ratio:.2f}× baseline={baseline_ppl:.2f})")
    elif baseline_ppl is not None:
        ppl_str = f"not yet measured  (baseline={baseline_ppl:.2f})"
    else:
        ppl_str = "not computed (--ppl_check_every 0)"

    bar = "─" * W
    print(f"\n┌{bar}┐")
    print(f"│  HEALTH CHECK  step {step}/{total_steps:<{W-26}}│")
    print(f"│{bar}│")
    print(f"│  1. Loss trend : {trend_str:<{W-18}}│")
    print(f"│  2. NaN events : {nan_str:<{W-18}}│")
    print(f"│  3. Val loss   : {val_str:<{W-18}}│")
    print(f"│  4. PPL        : {ppl_str:<{W-18}}│")
    print(f"└{bar}┘\n")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.merge_only:
        merge_and_save(args)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _setup_hw_opts(torch.device(device))
    training_mode = "full SFT" if args.no_lora else f"LoRA r={args.lora_r}"
    print(f"Device  : {device}")
    print(f"Draft   : {args.draft}  ({training_mode})")
    print(f"Target  : {args.target}  (frozen bfloat16)")
    print(f"Loss    : {args.loss}")
    print(f"Attn    : {_ATTN_IMPL}  "
          f"({'install flash-attn for FA2' if _ATTN_IMPL == 'eager' else 'auto-selected'})")
    print(f"Steps   : {args.steps}")
    if args.dataset:
        print(f"Dataset : {args.dataset}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.target)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("\nLoading target (frozen)...")
    if args.load_in_4bit:
        # QLoRA mode: load teacher in 4-bit NF4 so an 8B model fits on a T4 (15 GB).
        # Teacher is frozen so quantisation doesn't hurt training — it only runs
        # forward passes to produce logits.
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise SystemExit("bitsandbytes not found. Run: pip install bitsandbytes")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        target_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.target, quantization_config=bnb_cfg, device_map="auto",
            attn_implementation=_ATTN_IMPL)
        print(f"  Loaded in 4-bit NF4 (QLoRA mode, attn={_ATTN_IMPL})")
    else:
        target_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.target, **_dtype_kwargs(torch.bfloat16),
            attn_implementation=_ATTN_IMPL).to(device)
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad_(False)
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

    print("Loading draft...")
    draft_base = transformers.AutoModelForCausalLM.from_pretrained(
        args.draft, **_dtype_kwargs(torch.bfloat16),
        attn_implementation=_ATTN_IMPL).to(device)
    if args.no_lora:
        draft_model = draft_base
        draft_model.gradient_checkpointing_enable()
        trainable = sum(p.numel() for p in draft_model.parameters())
        print(f"  Full SFT: {trainable/1e6:.1f}M trainable params")
    else:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        draft_model = get_peft_model(draft_base, lora_cfg)
        draft_model.print_trainable_parameters()
    print(f"  VRAM both: {torch.cuda.memory_allocated()/1024**2:.0f} MB\n")

    optimizer = AdamW(
        [p for p in draft_model.parameters() if p.requires_grad], lr=args.lr)

    import random
    torch.manual_seed(args.seed)

    # ── Crash-safe resume ─────────────────────────────────────────────────────
    # If {output}/training_state.json + {output}/ckpt_latest/ both exist,
    # resume automatically — no extra flag needed.  Just re-run the same command.
    #
    # On Colab: point --output to Google Drive so the checkpoint survives
    # session death:
    #   --output /content/drive/MyDrive/OSD/checkpoints/kl-run
    # Then on restart, mount Drive and re-run the same training command.
    ckpt_latest = os.path.join(args.output, "ckpt_latest")
    state_path  = os.path.join(args.output, "training_state.json")
    start_step, losses, accept_weights = 0, [], []

    _ckpt_marker = "adapter_config.json" if not args.no_lora else "model.pt"
    _resuming = (os.path.exists(state_path)
                 and os.path.isdir(ckpt_latest)
                 and os.path.exists(os.path.join(ckpt_latest, _ckpt_marker)))

    if _resuming:
        with open(state_path, encoding="utf-8-sig") as _f:   # utf-8-sig silently strips Windows BOM
            _saved = json.load(_f)
        start_step     = _saved.get("step", 0)
        losses         = _saved.get("loss_history", [])
        accept_weights = _saved.get("accept_weight_history", [])
        # Restore early-stop state so patience counter isn't reset on resume.
        # Use float() cast to guard against PowerShell ConvertTo-Json writing
        # numbers as strings (e.g. "-8.3953" instead of -8.3953).
        _raw_bvl = _saved.get("best_val_loss", None)
        _resumed_best_val   = float(_raw_bvl) if _raw_bvl is not None else None
        _resumed_no_improve = int(_saved.get("val_no_improve_count", 0))
        print(f"\n[RESUME] Checkpoint found — {start_step}/{args.steps} steps done  "
              f"→  continuing from step {start_step + 1}")
        if not args.no_lora:
            # Load saved LoRA adapter into the already-created PEFT model
            _tmp = PeftModel.from_pretrained(draft_base, ckpt_latest, is_trainable=True)
            draft_model.load_state_dict(_tmp.state_dict())
            del _tmp
            print(f"[RESUME] LoRA weights loaded from {ckpt_latest}")
        else:
            draft_model.load_state_dict(
                torch.load(os.path.join(ckpt_latest, "model.pt"), map_location=device))
            print(f"[RESUME] Full SFT weights loaded from {ckpt_latest}")
        _opt_path = os.path.join(ckpt_latest, "optimizer.pt")
        if os.path.exists(_opt_path):
            optimizer.load_state_dict(torch.load(_opt_path, map_location=device))
            print(f"[RESUME] Optimizer state restored.")
        # ── Health summary from the prior run ────────────────────────────────
        if losses:
            _n = min(10, len(losses))
            _prior_first = sum(losses[:_n]) / _n
            _prior_last  = sum(losses[-_n:]) / _n
            _trend = "↓ improving" if _prior_last < _prior_first else "↑ worsening"
            print(f"[RESUME] Prior loss trend (first/last {_n} records): "
                  f"{_prior_first:.4f} → {_prior_last:.4f}  ({_trend})")
            print(f"[RESUME] Loss history length: {len(losses)} steps recorded")
        print(f"[RESUME] ─── continuing from step {start_step + 1} ───────────────\n")
    else:
        os.makedirs(args.output, exist_ok=True)
        _resumed_best_val   = None
        _resumed_no_improve = 0

    if start_step >= args.steps:
        print(f"[RESUME] Training already complete ({start_step}/{args.steps} steps). "
              f"Run: python train_qwen3.py --merge_only --adapter {args.output}")
        return

    # ── W&B (optional — silently skips if not installed / not logged in / --no_wandb)
    # What gets logged:
    #   train/loss, train/lr, train/peak_vram_mb  — every --log_every steps
    #   train/steps_per_sec                        — throughput for GPU efficiency tracking
    #   train/gpu_util_pct                         — GPU utilisation % (requires pynvml)
    #   train/ppl, train/ppl_ratio                 — PPL health check (every --ppl_check_every)
    #   val/loss, val/accept_weight                — every --val_every steps
    #   hardware config (gpu_name, gpu_vram_gb)    — stored in run config at init
    #   gradient histograms                        — via wandb.watch() (every 100 steps)
    # ─────────────────────────────────────────────────────────────────────────
    _wandb = None
    if not getattr(args, "no_wandb", False):
        try:
            import wandb as _wand

            # Collect GPU info for the run config so you can filter runs by hardware
            _gpu_name = "cpu"
            _gpu_vram_gb = 0.0
            if torch.cuda.is_available():
                _gpu_name    = torch.cuda.get_device_name(0)
                _gpu_vram_gb = round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)

            _run_name = (f"{args.loss}_{args.steps}steps_"
                         f"{os.path.basename(args.output)}")
            _wandb = _wand.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=args.wandb_group,
                name=_run_name,
                config={
                    **vars(args),
                    # Hardware metadata — useful for filtering runs by GPU in sweep analysis
                    "gpu_name":     _gpu_name,
                    "gpu_vram_gb":  _gpu_vram_gb,
                    "draft_model":  args.draft,
                    "target_model": args.target,
                },
                resume="allow",
            )
            # Watch gradients and parameter histograms — visible in W&B "Gradients" panel.
            # log_freq=100: capture every 100 gradient steps (more frequent = slower).
            _wand.watch(draft_model, log="gradients", log_freq=100)
            print(f"  [wandb] Run: {_wandb.url}")
        except Exception as _we:
            print(f"  [wandb] Skipped ({type(_we).__name__}: {_we})")

    all_prompts = load_prompts(args.dataset)

    # ── Train / validation split ──────────────────────────────────────────────
    # Validation set is carved out BEFORE training so it is never trained on.
    # --val_dataset: separate file (preferred for real experiments)
    # --val_split:   hold out last X% of the dataset (quick default)
    # If val_every=0, no validation is run and we use all prompts for training.
    if args.val_dataset:
        val_prompts  = load_prompts(args.val_dataset)
        train_prompts = all_prompts
        print(f"Validation: {len(val_prompts)} prompt(s) from {args.val_dataset}")
    elif args.val_every > 0 and args.val_split > 0.0:
        n_val         = max(1, int(len(all_prompts) * args.val_split))
        train_prompts = all_prompts[:-n_val]
        val_prompts   = all_prompts[-n_val:]
        print(f"Validation: held-out last {n_val}/{len(all_prompts)} prompts "
              f"({args.val_split*100:.0f}%%)  — every {args.val_every} steps")
    else:
        train_prompts = all_prompts
        val_prompts   = []
        print("Validation: disabled (--val_every 0 or --val_split 0)")

    prompts = train_prompts
    n = len(prompts)
    if n == 0:
        raise ValueError("No training prompts after train/val split — "
                         "check --val_split (currently too large) or --dataset path.")

    # ── Optional torch.compile ─────────────────────────────────────────────
    # ~60s one-time compile overhead; then 10-30% faster per step.
    # Skip on Windows (triton backend often unavailable).
    if args.compile:
        import platform
        if platform.system() == "Windows":
            print("[WARN] --compile skipped: torch.compile has limited support on Windows. "
                  "Try on Linux/Colab.")
        elif not hasattr(torch, "compile"):
            print("[WARN] --compile skipped: PyTorch < 2.0 detected.")
        else:
            print("Compiling draft model (one-time ~60s)...")
            draft_model = torch.compile(draft_model, mode="reduce-overhead")
            print("  Done.")

    # ── Pre-tokenise all unique prompts once ──────────────────────────────
    # Avoids per-step tokenizer + Jinja2 template overhead (~5-15 ms/step).
    print("Pre-tokenising prompts...")
    _tok_cache: dict = {}
    for _p in prompts:
        _msgs = [{"role": "user", "content": _p}]
        _txt  = tokenizer.apply_chat_template(_msgs, tokenize=False,
                                              add_generation_prompt=True)
        _tok_cache[_p] = tokenizer(_txt, return_tensors="pt").input_ids  # CPU tensor
    print(f"  {len(_tok_cache)} unique prompt(s) cached.")

    def _epoch_prompts(epoch: int) -> list:
        """Deterministic per-epoch shuffle — fully reproducible on resume."""
        if args.no_shuffle:
            return list(prompts)
        rng = random.Random(args.seed + epoch)
        order = list(range(n))
        rng.shuffle(order)
        return [prompts[i] for i in order]

    # Pre-compute the shuffled order for the epoch we're starting in
    shuffled = _epoch_prompts(start_step // n)

    # ── Health-check tracking state ───────────────────────────────────────────
    _nan_count            = 0       # cumulative NaN/Inf loss events this run
    _val_loss_history     = []      # val loss at each val_every checkpoint
    # Restore from checkpoint if available (so patience counter survives crash-resume)
    _best_val_loss        = (_resumed_best_val if _resuming and _resumed_best_val is not None
                             else float("inf"))
    _val_no_improve_count = (_resumed_no_improve if _resuming else 0)
    if _resuming and _best_val_loss != float("inf"):
        print(f"[RESUME] Early-stop state restored: best_val={_best_val_loss:.4f}, "
              f"no-improve streak={_val_no_improve_count}")
    _current_ppl          = None    # latest measured PPL
    _baseline_ppl         = None    # PPL measured before any weight updates
    _stop_training        = False   # set True by NaN-stop or early-stop

    # Baseline PPL: measured pre-training so all future checks have a reference.
    # Uses val_prompts if available, else first 5 train prompts.
    # Skip if ppl_check_every=0 (saves ~2 min on large models).
    if args.ppl_check_every > 0:
        _ppl_sample = val_prompts[:5] if val_prompts else prompts[:5]
        print("Computing pre-training baseline PPL (no weight updates yet)...")
        _baseline_ppl = _compute_ppl(
            draft_model, _ppl_sample, _tok_cache, tokenizer, device, max_prompts=5)
        _current_ppl = _baseline_ppl
        _warn_at = _baseline_ppl * args.ppl_threshold
        print(f"  [HEALTH] Baseline PPL = {_baseline_ppl:.2f}  "
              f"(will warn if PPL > {_warn_at:.2f}  = "
              f"{args.ppl_threshold:.2f}× baseline)\n")

    _step_timer    = time.time()   # tracks elapsed time for steps/sec logging
    _train_crashed = False         # set True if we catch an unhandled exception mid-step

    for step in range(start_step, args.steps):
        idx = step % n
        # Reshuffle deterministically at the start of each new epoch
        if idx == 0 and step > 0:
            shuffled = _epoch_prompts(step // n)
        prompt = shuffled[idx]

        # Look up pre-tokenised prompt (avoids per-step tokenizer overhead)
        prompt_ids = _tok_cache[prompt].to(device)

        # ── Teacher: generate + capture logits in ONE pass ──────────────────
        # output_scores=True stores the per-step logits produced during
        # auto-regressive generation, so we do NOT need a second
        # target_model(full_ids) forward pass to score the sequence.
        # Saves ~40-50% of teacher compute per step (one forward pass removed).
        #
        # NOTE on temperature: output_scores are temperature-scaled logits.
        # We divide by _TEACHER_TEMP to recover the raw unscaled distribution
        # that the loss functions expect (they apply softmax internally).
        _TEACHER_TEMP = args.teacher_temp
        with torch.no_grad():
            gen_out = target_model.generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=_TEACHER_TEMP,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,          # ← key: captures logits during generation
            )
        full_ids = gen_out.sequences.clone()                  # break inference-tensor chain: generate() uses @inference_mode internally
        plen     = prompt_ids.shape[1]

        # Guard: target model hit EOS on token 0 — generated nothing.
        # torch.stack(gen_out.scores, dim=0) would raise RuntimeError on empty tuple.
        # Skip this step; it's rare (<0.1% of GSM8K prompts) but crashes the run.
        if not gen_out.scores:
            losses.append(float("nan"))
            continue

        # Stack per-step scores → (gen_len, vocab_size); undo temperature scaling.
        # HuggingFace output_scores = raw_logits / temperature, so we MULTIPLY to recover
        # raw_logits. (Previous code divided, which applied temperature twice — fixed.)
        t_log = torch.stack(gen_out.scores, dim=0).squeeze(1).float() * _TEACHER_TEMP

        # ── Draft: one forward pass (with grad for backprop) ─────────────────
        draft_model.train()
        student_logits = draft_model(full_ids).logits[:, :-1, :].float()

        # Generated portion only (not prompt)
        s_log   = student_logits[:, plen-1:, :].squeeze(0)  # [gen_len, V]
        gen_ids = full_ids[0, plen:]                          # [gen_len]

        try:
            if args.loss == "forward_kl":
                loss = forward_kl_loss(s_log, t_log)
                aw = None
            elif args.loss == "reverse_kl":
                loss = reverse_kl_loss(s_log, t_log)
                aw = None
            elif args.loss == "jsd":
                loss = jsd_loss(s_log, t_log, alpha=args.jsd_alpha)
                aw = None
            elif args.loss == "l1":
                loss = l1_loss(s_log, t_log)
                aw = None
            elif args.loss == "ebe":
                loss, aw_dict = ebe_loss(s_log, t_log, gen_ids,
                                         kl_weight=args.ebe_kl_weight)
                # aw_dict contains full α distribution; store mean for history
                aw = aw_dict["mean"] if isinstance(aw_dict, dict) else aw_dict
                accept_weights.append(aw)
                # Log α distribution every log_every steps so we can see how
                # much real EBE gradient is flowing (frac_lt_0.95 > 0 means it is).
                if (step + 1) % args.log_every == 0 and isinstance(aw_dict, dict):
                    print(
                        f"  [EBE-α]  mean={aw_dict['mean']:.4f}  "
                        f"std={aw_dict['std']:.4f}  "
                        f"min={aw_dict['min']:.4f}  "
                        f"frac<0.95={aw_dict['frac_lt_0.95']:.3f}  "
                        f"frac<0.80={aw_dict['frac_lt_0.80']:.3f}"
                    )
                    if _wandb:
                        _wandb.log({
                            "ebe/alpha_mean":      aw_dict["mean"],
                            "ebe/alpha_std":       aw_dict["std"],
                            "ebe/alpha_min":       aw_dict["min"],
                            "ebe/frac_lt_0.95":    aw_dict["frac_lt_0.95"],
                            "ebe/frac_lt_0.80":    aw_dict["frac_lt_0.80"],
                        }, step=step + 1)
            elif args.loss == "ebe_single":
                loss, aw_dict = ebe_single_loss(s_log, t_log, gen_ids)
                aw = aw_dict["mean"] if isinstance(aw_dict, dict) else aw_dict
                accept_weights.append(aw)
                if (step + 1) % args.log_every == 0 and isinstance(aw_dict, dict):
                    print(
                        f"  [EBEs-α] mean={aw_dict['mean']:.4f}  "
                        f"std={aw_dict['std']:.4f}  "
                        f"min={aw_dict['min']:.4f}  "
                        f"frac<0.95={aw_dict['frac_lt_0.95']:.3f}  "
                        f"frac<0.80={aw_dict['frac_lt_0.80']:.3f}"
                    )
                    if _wandb:
                        _wandb.log({
                            "ebe_single/alpha_mean":   aw_dict["mean"],
                            "ebe_single/alpha_std":    aw_dict["std"],
                            "ebe_single/alpha_min":    aw_dict["min"],
                            "ebe_single/frac_lt_0.95": aw_dict["frac_lt_0.95"],
                            "ebe_single/frac_lt_0.80": aw_dict["frac_lt_0.80"],
                        }, step=step + 1)
        except Exception as _loss_exc:
            # Catch runtime errors inside loss functions (e.g. EBE cumprod error,
            # KL divergence shape mismatch) — save what we have and stop gracefully.
            _nan_count += 1
            print(f"\n  [HEALTH] ⚠ Loss function exception at step {step+1}: "
                  f"{type(_loss_exc).__name__}: {_loss_exc}")
            import traceback as _tb; _tb.print_exc()
            if args.nan_action == "stop":
                print("  [HEALTH] Saving crash-safe checkpoint then stopping.")
                _save_checkpoint(draft_model, optimizer, step + 1,
                                 args, losses, accept_weights)
                _train_crashed = True
                _stop_training = True
                break
            elif args.nan_action == "skip":
                print("  [HEALTH] Skipping step — weights unchanged.")
                losses.append(float("nan"))
                continue
            # "warn": fall through to NaN loss (loss may be undefined — force stop)
            _save_checkpoint(draft_model, optimizer, step + 1,
                             args, losses, accept_weights)
            _train_crashed = True
            _stop_training = True
            break

        # ── NaN / Inf guard ───────────────────────────────────────────────────
        # Check BEFORE backward so a single bad batch doesn't corrupt weights.
        # Pass condition: "No nan in loss column" (health-check table row 2).
        if not torch.isfinite(loss):
            _nan_count += 1
            print(f"  [HEALTH] ⚠ NaN/Inf loss at step {step+1}  "
                  f"(total events this run: {_nan_count})  "
                  f"--nan_action={args.nan_action}")
            if args.nan_action == "stop":
                print("  [HEALTH] Saving crash-safe checkpoint then aborting.")
                _save_checkpoint(draft_model, optimizer, step + 1,
                                 args, losses, accept_weights)
                _stop_training = True
                break
            elif args.nan_action == "skip":
                print("  [HEALTH] Skipping step — weights unchanged.")
                losses.append(float("nan"))
                continue
            # "warn": fall through (NaN may propagate into weights — use with care)

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in draft_model.parameters() if p.requires_grad],
                args.grad_clip)
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % args.log_every == 0:
            avg_l  = sum(losses[-args.log_every:]) / args.log_every
            vram   = torch.cuda.max_memory_allocated() / 1024**2
            avg_aw = (sum(accept_weights[-args.log_every:]) / args.log_every
                      if accept_weights else None)
            extra  = f"  avg_accept_weight={avg_aw:.3f}" if avg_aw is not None else ""
            print(f"Step {step+1:4d}/{args.steps} | train_loss: {avg_l:.4f} | "
                  f"peak VRAM: {vram:.0f} MB{extra}")
            if _wandb:
                _elapsed = time.time() - _step_timer
                _sps     = args.log_every / max(_elapsed, 1e-6)
                _step_timer = time.time()   # reset for next window
                _gpu_util = _gpu_util_percent()
                _wlog = {
                    "train/loss":         avg_l,
                    "train/peak_vram_mb": vram,
                    "train/lr":           args.lr,
                    "train/steps_per_sec": _sps,
                }
                if avg_aw     is not None: _wlog["train/accept_weight"] = avg_aw
                if _gpu_util  is not None: _wlog["train/gpu_util_pct"]  = _gpu_util
                _wandb.log(_wlog, step=step + 1)
            # Save to results DB (train split) — write_train_step handles path resolution
            # and is silently no-op if the DB is unavailable.
            _write_train_step(
                label=os.path.basename(args.output),
                loss_name=args.loss,
                step=step + 1,
                loss=avg_l,
                learning_rate=args.lr,
                lora_rank=args.lora_r,
                accept_weight=avg_aw,
                split="train",
            )

        # ── Validation loss ───────────────────────────────────────────────────
        if (args.val_every > 0 and val_prompts
                and (step + 1) % args.val_every == 0):
            val_loss, val_aw = _compute_val_loss(
                draft_model, target_model, val_prompts, _tok_cache,
                tokenizer, args, device,
                max_prompts=min(30, len(val_prompts)),  # 10 was too noisy; 30 gives reliable early-stop signal
            )
            val_extra = (f"  val_accept_weight={val_aw:.3f}" if val_aw is not None else "")
            print(f"Step {step+1:4d}/{args.steps} | val_loss:   {val_loss:.4f}{val_extra}")
            if _wandb:
                _vlog = {"val/loss": val_loss}
                if val_aw is not None:
                    _vlog["val/accept_weight"] = val_aw
                _wandb.log(_vlog, step=step + 1)
            _write_train_step(
                label=os.path.basename(args.output),
                loss_name=args.loss,
                step=step + 1,
                loss=val_loss,
                learning_rate=args.lr,
                lora_rank=args.lora_r,
                accept_weight=val_aw,
                split="val",
            )

            # ── Health tracking: record val loss, update best, check patience ─
            if math.isfinite(val_loss):
                _val_loss_history.append(val_loss)
                if val_loss < _best_val_loss:
                    _best_val_loss = val_loss
                    _val_no_improve_count = 0
                    # Always checkpoint the best val weights so we can merge from
                    # ckpt_best/ even if later steps overfit.
                    import shutil as _shutil
                    _save_checkpoint(draft_model, optimizer, step + 1,
                                     args, losses, accept_weights,
                                     best_val_loss=_best_val_loss,
                                     val_no_improve_count=_val_no_improve_count)
                    # Copy ckpt_latest → ckpt_best using per-file copy2 with retry.
                    # Avoids PermissionError on OneDrive/GDrive/NFS mounts that hold
                    # a short sync lock on the directory (WinError 5, errno 13).
                    _best_dir = os.path.join(args.output, "ckpt_best")
                    _src_dir  = os.path.join(args.output, "ckpt_latest")
                    os.makedirs(_best_dir, exist_ok=True)
                    for _fname in os.listdir(_src_dir):
                        _src_f = os.path.join(_src_dir, _fname)
                        _dst_f = os.path.join(_best_dir, _fname)
                        for _attempt in range(4):          # up to 4 attempts
                            try:
                                _shutil.copy2(_src_f, _dst_f)
                                break
                            except PermissionError as _pe:
                                if _attempt == 3:
                                    raise
                                import time as _time_mod
                                print(f"  [best] PermissionError on copy, retry {_attempt+1}/3 "
                                      f"({_pe}) — waiting 2s for sync lock to release")
                                _time_mod.sleep(2)
                    print(f"  [best] New best val {val_loss:.4f} at step {step+1}"
                          f"  ->  {_best_dir}")
                else:
                    _val_no_improve_count += 1
                    print(f"  [best] val {val_loss:.4f}"
                          f" (best={_best_val_loss:.4f},"
                          f" no-improve streak={_val_no_improve_count})")
                    if (args.early_stop_patience > 0
                            and _val_no_improve_count >= args.early_stop_patience):
                        print(f"\n[HEALTH] Early stopping triggered: val loss has not improved "
                              f"for {args.early_stop_patience} consecutive checks "
                              f"(best={_best_val_loss:.4f}, current={val_loss:.4f}).\n"
                              f"[HEALTH] Best weights already saved in ckpt_best/  --  merge from there.")
                        _stop_training = True

        # ── Mid-training checkpoint (crash-safe) ─────────────────────────────
        _is_milestone = (args.milestone_every > 0
                         and (step + 1) % args.milestone_every == 0)
        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            _save_checkpoint(draft_model, optimizer, step + 1,
                             args, losses, accept_weights,
                             milestone=_is_milestone,
                             best_val_loss=_best_val_loss,
                             val_no_improve_count=_val_no_improve_count)
        elif _is_milestone:
            # milestone_every != save_every — save explicitly for milestone
            _save_checkpoint(draft_model, optimizer, step + 1,
                             args, losses, accept_weights,
                             milestone=True,
                             best_val_loss=_best_val_loss,
                             val_no_improve_count=_val_no_improve_count)

        # ── Checkpoint verification ───────────────────────────────────────────
        # Confirm the file was actually written so we catch disk-full / permission
        # errors immediately rather than finding out only on the next resume attempt.
        # Pass condition: "Checkpoint files exist" (health-check table row 4).
        _ckpt_was_saved = (args.save_every > 0
                           and (step + 1) % args.save_every == 0)
        if _ckpt_was_saved:
            _verify_marker = "adapter_config.json" if not args.no_lora else "model.pt"
            _verify_path   = os.path.join(args.output, "ckpt_latest", _verify_marker)
            if os.path.exists(_verify_path):
                print(f"  [HEALTH] ✓ Checkpoint verified  ({_verify_marker} present)")
            else:
                print(f"  [HEALTH] ⚠ Checkpoint MISSING — {_verify_path} not found!  "
                      f"Check disk space / permissions.")

        # ── PPL health check ──────────────────────────────────────────────────
        # Pass condition: "PPL within 20% of baseline" (health-check table row 3).
        if (args.ppl_check_every > 0
                and (step + 1) % args.ppl_check_every == 0):
            _ppl_sample  = val_prompts[:5] if val_prompts else prompts[:5]
            _current_ppl = _compute_ppl(
                draft_model, _ppl_sample, _tok_cache, tokenizer, device, max_prompts=5)
            if _baseline_ppl is not None and _baseline_ppl > 0:
                _ratio = _current_ppl / _baseline_ppl
                _flag  = ("⚠ COLLAPSED — check training!"
                          if _ratio > args.ppl_threshold else "✓ within threshold")
                print(f"  [HEALTH] PPL={_current_ppl:.2f}  "
                      f"ratio={_ratio:.2f}×  (baseline={_baseline_ppl:.2f})  {_flag}")
                if _wandb:
                    _wandb.log({
                        "train/ppl":       _current_ppl,
                        "train/ppl_ratio": _ratio,
                    }, step=step + 1)
            else:
                print(f"  [HEALTH] PPL={_current_ppl:.2f}  (no baseline)")
                if _wandb:
                    _wandb.log({"train/ppl": _current_ppl}, step=step + 1)

        # ── Periodic health report ─────────────────────────────────────────────
        if args.health_every > 0 and (step + 1) % args.health_every == 0:
            _print_health_report(
                step + 1, args.steps, losses, _val_loss_history,
                _nan_count, _current_ppl, _baseline_ppl, _best_val_loss,
                args.ppl_threshold,
            )

        # ── Stop flag (NaN-stop or early-stop) ───────────────────────────────
        if _stop_training:
            break

    # ── Final save ───────────────────────────────────────────────────────────
    # Runs whether training completed, NaN-stopped, early-stopped, OR crashed
    # mid-step (loss function exception).  Any partial model is still usable
    # for eval — better a 300-step EBE model than nothing.
    #
    # If save_best checkpointing ran during training, ckpt_best/ holds the
    # adapter weights from the val-loss minimum.  Promote those to the root
    # output dir (where done_check and --merge_only expect them) rather than
    # saving the potentially-overfit final weights.
    import shutil as _final_shutil
    os.makedirs(args.output, exist_ok=True)
    _best_ckpt_dir = os.path.join(args.output, "ckpt_best")
    if os.path.exists(_best_ckpt_dir):
        print(f"\n  [best] Promoting best-val checkpoint to {args.output}/")
        for _fname in os.listdir(_best_ckpt_dir):
            if _fname == "optimizer.pt":
                continue   # skip large optimizer state — not needed for merge
            _final_shutil.copy2(
                os.path.join(_best_ckpt_dir, _fname),
                os.path.join(args.output, _fname),
            )
        tokenizer.save_pretrained(args.output)
        print(f"  [best] Done — root adapter = best-val weights (not final-step weights)")
    else:
        draft_model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)

    if _train_crashed and losses:
        # Stopped due to an exception but completed at least 1 step.
        # Exit 0 so the pipeline marks this step as DONE (partial), not FAILED.
        n_completed = start_step + len([l for l in losses if not math.isnan(l)])
        print(f"\n  [TRAIN] Partial model saved after {n_completed} steps "
              f"({args.output}). Pipeline will continue to merge + eval.")
        if os.path.exists(state_path):
            os.rename(state_path, state_path.replace(".json", ".partial.json"))
        sys.exit(0)   # ← explicit 0 → pipeline sees rc=0 → marks "done"

    # Archive the training state file so a subsequent re-run doesn't resume
    if os.path.exists(state_path):
        os.rename(state_path, state_path.replace(".json", ".done.json"))
        print(f"  [ckpt] Training complete — state archived.")
    if args.no_lora:
        print(f"\nFull SFT model -> {args.output}  (ready to use directly)")
    else:
        print(f"\nLoRA adapters -> {args.output}")
        print(f"Merge: python train_qwen3.py --merge_only --adapter {args.output}")

    # W&B: log final summary metrics and close the run cleanly
    if _wandb:
        try:
            _final_train_loss = (sum(losses[-10:]) / len(losses[-10:])
                                 if losses else float("nan"))
            _wandb.summary.update({
                "final_train_loss": _final_train_loss,
                "best_val_loss":    _best_val_loss if math.isfinite(_best_val_loss) else None,
                "final_ppl":        _current_ppl,
                "nan_events":       _nan_count,
                "total_steps":      step + 1 if not _stop_training else step,
            })
            _wandb.finish()
            print("  [wandb] Run finished and synced.")
        except Exception as _wfe:
            print(f"  [wandb] finish() failed ({_wfe}) — run may need manual sync.")


if __name__ == "__main__":
    main()
