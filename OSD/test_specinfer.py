"""
Measure speculative decoding acceptance rate (alpha) for a draft/target pair.

Usage:
    python test_specinfer.py --draft Qwen/Qwen2.5-0.5B --target Qwen/Qwen3-0.6B
    python test_specinfer.py --draft ./checkpoints/ebe200_merged --target Qwen/Qwen3-0.6B
    python test_specinfer.py --draft Qwen/Qwen2.5-0.5B --target Qwen/Qwen3-0.6B --n 50
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "distill"))

import torch
import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from specInfer.generator import Generator

# 50 diverse prompts — 10 per category so we can check per-category breakdown
PROMPTS = [
    # Factual / knowledge (10)
    "What is the capital of France?",
    "What is the capital of Japan?",
    "What is Newton's second law of motion?",
    "What is the speed of light?",
    "What is the Pythagorean theorem?",
    "What is Ohm's law?",
    "What is the difference between RAM and ROM?",
    "What is Moore's law?",
    "Summarize the French Revolution in two sentences.",
    "What is quantum entanglement in simple terms?",
    # Coding (10)
    "Write a Python function that returns the factorial of n.",
    "Write a Python function to check if a string is a palindrome.",
    "Write a function to reverse a string.",
    "Write a function to check if a number is prime.",
    "Write a function to find the maximum subarray sum.",
    "Write a function to do binary search on a sorted array.",
    "Write a function to flatten a nested list.",
    "Write a function to find the first non-repeating character.",
    "Write a Python decorator that measures execution time.",
    "Write a function to compute Fibonacci numbers efficiently.",
    # CS concepts (10)
    "Explain what a linked list is in one sentence.",
    "How does a hash table work?",
    "What is the difference between a stack and a queue?",
    "What is the time complexity of merge sort?",
    "What is the difference between BFS and DFS?",
    "What is dynamic programming?",
    "What is the difference between a process and a thread?",
    "How does virtual memory work?",
    "What is a deadlock?",
    "What is the difference between TCP and UDP?",
    # ML / AI (10)
    "How does gradient descent work?",
    "What is overfitting in machine learning and how do you prevent it?",
    "What is backpropagation?",
    "What is the softmax function?",
    "Explain what attention mechanism does in transformers.",
    "What is the difference between supervised and unsupervised learning?",
    "What is cross-entropy loss?",
    "What is batch normalization?",
    "What is transfer learning?",
    "What is the bias-variance tradeoff?",
    # Math / logic (10)
    "What is 2 + 2?",
    "What is the derivative of sin(x)?",
    "Solve the equation x^2 - 5x + 6 = 0.",
    "What is Bayes' theorem?",
    "What is Big-O notation? Give an example.",
    "What is a normal distribution?",
    "What is the central limit theorem?",
    "What is a p-value in statistics?",
    "What is standard deviation?",
    "What is linear regression?",
]

CATEGORIES = ["factual", "coding", "cs_concepts", "ml_ai", "math"]

MAX_PROPOSE = 5
MAX_TOKENS  = 30
TEMPERATURE = 0.6


def load_model(path, dtype, device, lora_adapter=None):
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    if lora_adapter:
        model = PeftModel.from_pretrained(model, lora_adapter)
        model = model.merge_and_unload()
        model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--draft",  default="Qwen/Qwen3-0.6B")
    p.add_argument("--target", default="Qwen/Qwen3-0.6B")
    p.add_argument("--lora_adapter", default=None)
    p.add_argument("--n", type=int, default=50, help="Number of prompts to evaluate (max 50)")
    args = p.parse_args()

    n = min(args.n, len(PROMPTS))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}  |  Evaluating {n} prompts")
    print(f"Draft  : {args.draft}" + (f" + {args.lora_adapter}" if args.lora_adapter else ""))
    print(f"Target : {args.target}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.target, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    draft_model = load_model(args.draft, torch.float16, device, args.lora_adapter)
    same = (args.draft == args.target and args.lora_adapter is None)
    target_model = draft_model if same else load_model(args.target, torch.float16, device)
    print(f"VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB\n")

    generator = Generator(
        small_model=draft_model, large_model=target_model,
        tokenizer=tokenizer, max_propose_num=MAX_PROPOSE,
        is_encoder_decoder=False, use_cache=True,
    )

    alphas = []
    cat_alphas = {c: [] for c in CATEGORIES}
    total_tokens_generated = 0
    total_wall_time = 0.0

    for i in range(n):
        prompt = PROMPTS[i]
        category = CATEGORIES[i // 10]
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            output = generator.generate(
                input_ids=input_ids,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                attention_mask=torch.ones_like(input_ids),
            )

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        # count generated tokens (output includes prompt; subtract it)
        gen_tokens = output.output[0].shape[-1] - input_ids.shape[-1] if hasattr(output.output[0], 'shape') else MAX_TOKENS
        total_tokens_generated += gen_tokens
        total_wall_time += elapsed

        alpha = float(output.alpha_sum) / output.sample_steps if output.sample_steps > 0 else 0.0
        alphas.append(alpha)
        cat_alphas[category].append(alpha)

    alphas = np.array(alphas)
    throughput = total_tokens_generated / total_wall_time if total_wall_time > 0 else 0
    ms_per_tok  = (total_wall_time / total_tokens_generated * 1000) if total_tokens_generated > 0 else 0

    print(f"{'='*50}")
    print(f"Draft  : {args.draft}")
    print(f"Target : {args.target}")
    print(f"N      : {n}")
    print(f"\nOverall alpha   : {alphas.mean():.4f}  (std={alphas.std():.4f}, 95% CI +/-{1.96*alphas.std()/np.sqrt(n):.4f})")
    print(f"Throughput      : {throughput:.2f} tok/sec")
    print(f"Walltime/token  : {ms_per_tok:.1f} ms/tok")
    print(f"\nPer-category breakdown:")
    for cat in CATEGORIES:
        vals = np.array(cat_alphas[cat])
        if len(vals):
            print(f"  {cat:12s}: {vals.mean():.4f}  (std={vals.std():.4f})")
    print(f"\nPeak VRAM: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")


if __name__ == "__main__":
    main()
