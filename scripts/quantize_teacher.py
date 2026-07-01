"""
quantize_teacher.py — one-time NF4 quantization of a teacher model.

Loads a teacher in 4-bit NF4 (same BitsAndBytesConfig as verifiers/util.py's
--load_in_4bit path) and saves the quantized weights to disk. Every
subsequent training/eval run can then point --teacher at the saved directory
and load it directly (see verifiers/util.py::_dir_has_quant_config) — no
per-process re-quantization from raw bf16 weights.

Run this ONCE before launching a batch of parallel runs that share a large
quantized teacher (e.g. 8 concurrent 1.7B/32B training runs).

Usage:
    python scripts/quantize_teacher.py --model Qwen/Qwen3-32B \
        --output /sensei-fs-3/users/rkrishna/Qwen32B-Qwen1.7B/checkpoints/Qwen3-32B-nf4
"""
import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen3-32B")
    ap.add_argument("--output", required=True, help="Directory to save the quantized checkpoint")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"[quantize] loading {args.model} in 4-bit NF4 (one-time — this is the slow step)...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        quantization_config=bnb_cfg,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    print(f"[quantize] saving quantized checkpoint to {args.output} ...")
    model.save_pretrained(args.output)
    tok.save_pretrained(args.output)

    print(f"[quantize] done. Use --teacher {args.output} --load_in_4bit in future runs — "
          f"it will load directly from the packed NF4 weights, no re-quantization.")


if __name__ == "__main__":
    main()
