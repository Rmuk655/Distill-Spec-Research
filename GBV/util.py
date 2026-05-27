import random
import torch
import json
import torch.nn.functional as F
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

"""
Load prompts from a JSONL file, given a path to the directory.
Assumes that each prompt has key "prompt".
"""
def load_prompts_jsonl(path: str) -> List[str]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                prompts.append(obj["prompt"])
            except Exception as e:
                continue
    return prompts


"""
Set a seed for reproducibility.
"""
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


"""
Load target and draft models and tokenizers, given the model names, device, and data type.
Note: for speculative decoding, you are only allowed to load models with the same vocabulary.

compile_draft=True applies torch.compile(mode="reduce-overhead", dynamic=True) to the draft
model only.  The target model uses a custom {"full_attention": mask} that changes shape every
iteration, which is incompatible with static CUDA graphs; compiling it would cause graph
breaks or wrong results.  The draft model uses standard causal attention and is safe to
compile.  Benefit: reduces Python-side kernel-launch overhead (~100-200 ms/iter on CPU) that
accounts for ~90% of wall time on small GPUs with fast forward passes.
"""
def load_models(
    p_name: str,
    q_name: str,
    device: str = "cuda",
    dtype: str = "bf16",
    compile_draft: bool = False,
):
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    tok = AutoTokenizer.from_pretrained(p_name, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if dtype == "auto":
        torch_dtype = "auto"       # let HF read torch_dtype from model config (usually bf16)
    elif dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp32":
        torch_dtype = torch.float32
    else:
        # Fallback: use bf16 on CUDA, fp32 on CPU — avoids silent float32 OOM on small GPUs
        torch_dtype = torch.bfloat16 if "cuda" in str(device) else torch.float32

    p_model = AutoModelForCausalLM.from_pretrained(
        p_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=None,
        use_safetensors=True
    ).to(dev)
    p_model.eval()

    q_model = AutoModelForCausalLM.from_pretrained(
        q_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=None,
        use_safetensors=True
    ).to(dev)
    q_model.eval()

    torch.set_grad_enabled(False)

    # Validate custom attention mask compatibility BEFORE compile so we can still
    # inspect the underlying model's layer attributes (compile wraps the module).
    assert p_model.model.layers[0].attention_type == "full_attention", \
        "Custom attention mask not compatible with this HF Model"
    assert q_model.model.layers[0].attention_type == "full_attention", \
        "Custom attention mask not compatible with this HF Model"

    # Optionally compile the draft model to reduce Python→CUDA dispatch overhead.
    # dynamic=True handles the varying KV-cache length across autoregressive steps.
    # fullgraph=False (default) allows graph breaks — needed for the cache manipulations.
    # NOTE: do NOT compile p_model — its custom tree-attention mask changes shape every
    # iteration in ways that prevent stable CUDA graph capture.
    if compile_draft and "cuda" in str(dev):
        try:
            q_model = torch.compile(q_model, mode="reduce-overhead", dynamic=True)
            print("[INFO] Draft model compiled with torch.compile "
                  "(mode=reduce-overhead, dynamic=True).  "
                  "First few iterations will be slow (warm-up); subsequent ones faster.")
        except Exception as e:
            print(f"[WARN] torch.compile failed ({e}); running draft model in eager mode.")

    return tok, p_model, q_model


"""
Takes in:
    - cache: model KV-cache (DynamicCache), each KV tensor has shape (batch_sz, num_heads, context_len, head_dim)
    - batch_idx: list of indices to slice the batch_sz dimension along
    - token_idx: list of indices to slice the context_len dimension along
Returns:
    - cache: model KV-cache (DynamicCache), each KV tensor has shape (|batch_idx|, num_heads, |token_idx|, head_dim)
"""
def slice_cache(cache, batch_idx, token_idx):
    for layer_cache in cache.layers:
        layer_cache.keys = layer_cache.keys[batch_idx][:, :, token_idx, :].contiguous()
        layer_cache.values = layer_cache.values[batch_idx][:, :, token_idx, :].contiguous()
    return cache


"""
Takes in:
    - cache: model KV-cache (DynamicCache), each KV tensor has shape (1, num_heads, context_len, head_dim)
    - batch_sz: desired expansion along batch dimension
Returns:
    - cache: model KV-cache (DynamicCache), each KV tensor has shape (batch_sz, num_heads, context_len, head_dim)
"""
def expand_cache(cache, batch_sz):
    for layer_cache in cache.layers:
        layer_cache.keys = layer_cache.keys.expand(batch_sz, -1, -1, -1).contiguous()
        layer_cache.values = layer_cache.values.expand(batch_sz, -1, -1, -1).contiguous()
    return cache
