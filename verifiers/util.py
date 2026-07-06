import os
import random
import torch
import json
import torch.nn.functional as F
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache


def _dir_has_quant_config(path: str) -> bool:
    """True if path/config.json already carries a bnb quantization_config —
    i.e. this directory was produced by scripts/quantize_teacher.py and can be
    loaded directly as NF4 without a re-quantization pass."""
    cfg_path = os.path.join(path, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    try:
        with open(cfg_path, encoding="utf-8") as f:
            return "quantization_config" in json.load(f)
    except Exception:
        return False

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
def _best_free_cuda_device() -> str:
    """Return the cuda:N device with the most free memory."""
    n = torch.cuda.device_count()
    if n <= 1:
        return "cuda:0"
    best, best_free = 0, -1
    for i in range(n):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best, best_free = i, free
    return f"cuda:{best}"


def load_models(
    p_name: str,
    q_name: str,
    device: str = "cuda",
    dtype: str = "bf16",
    compile_draft: bool = False,
    load_in_4bit: bool = False,
    force_attn: str | None = None,
):
    """Load target (p) and draft (q) models.

    load_in_4bit=True loads the target model in 4-bit NF4 via bitsandbytes
    (requires: pip install bitsandbytes accelerate).  Use this on free T4
    GPUs (15 GB) where the 8B target in bfloat16 (~16 GB) does not fit.
    The draft model is always loaded in the requested dtype.

    force_attn ("sdpa" | "flash_attention_2" | None, default None): pin the
    DRAFT model's attention implementation explicitly, overriding whichever
    backend Transformers auto-selects (which otherwise depends on whether
    flash-attn happens to be installed on this particular machine — different
    machines silently picking different backends is a real source of
    eval-to-eval divergence: decoding is stochastic and speculative-decoding
    acceptance is a threshold test, so bf16 rounding differences between
    kernels can flip a sampled token or an accept/reject decision, cascading
    through the rest of that prompt's generation. Only the DRAFT is
    controllable here — the TARGET always stays on whatever the custom
    tree-attention mask requires (its per-iteration mask shape is
    incompatible with flash_attention_2 regardless of this flag; forcing it
    would silently break tree verification, not just change speed).
    """
    if device == "cuda" and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = _best_free_cuda_device()
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

    if load_in_4bit and "cuda" in str(device):
        # QLoRA / bitsandbytes path: target (large) model in 4-bit NF4.
        # device_map is required by bitsandbytes — do NOT call .to(dev) afterwards.
        #
        # device_map="auto" lets accelerate spread the model across EVERY visible
        # GPU on the node, ignoring the requested --device entirely. That's fine
        # for a single lone process, but with N concurrent runs each pinned to
        # its own GPU (e.g. 8 parallel training runs on cuda:0..cuda:7), "auto"
        # causes cross-device tensor mismatches ("index is on cuda:1, different
        # from cuda:0") because the quantized model's layers land wherever
        # accelerate's node-wide balancer puts them, not on `dev`. Pin every
        # layer to the single physical GPU explicitly instead.
        _phys_idx = dev.index if (dev.type == "cuda" and dev.index is not None) else 0
        _device_map = {"": _phys_idx}

        # If p_name is already a pre-quantized local directory (produced by
        # scripts/quantize_teacher.py --save_pretrained), its config.json already
        # carries the bnb quantization_config — transformers reconstructs the
        # NF4 layers straight from the packed weights on disk with NO re-quantization
        # pass. Re-quantizing from raw bf16 on every process (e.g. 8 concurrent
        # training runs sharing one 32B teacher) is the slow path this skips.
        _pre_quantized = (os.path.isdir(p_name)
                          and _dir_has_quant_config(p_name))
        if _pre_quantized:
            p_model = AutoModelForCausalLM.from_pretrained(
                p_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=_device_map,
            ).eval()
            print(f"  [load_models] {p_name} loaded from pre-quantized NF4 checkpoint on cuda:{_phys_idx} (no re-quantize)")
        else:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                raise SystemExit("bitsandbytes required for --load_in_4bit. "
                                 "Run: pip install bitsandbytes accelerate")
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            p_model = AutoModelForCausalLM.from_pretrained(
                p_name,
                trust_remote_code=True,
                quantization_config=bnb_cfg,
                low_cpu_mem_usage=True,
                device_map=_device_map,
            ).eval()
            print(f"  [load_models] {p_name} loaded in 4-bit NF4 on cuda:{_phys_idx} (target, fits T4)")
    else:
        p_model = AutoModelForCausalLM.from_pretrained(
            p_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=None,
            use_safetensors=True,
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

    if force_attn is not None:
        if force_attn not in ("sdpa", "flash_attention_2"):
            raise ValueError(f"force_attn must be 'sdpa' or 'flash_attention_2', got {force_attn!r}")
        if hasattr(q_model, "set_attn_implementation"):
            q_model.set_attn_implementation(force_attn)
        else:
            q_model.config._attn_implementation = force_attn   # fallback for older transformers
        print(f"[INFO] Draft model attention implementation pinned to '{force_attn}' (--force_attn).")

    torch.set_grad_enabled(False)

    # Validate custom attention mask compatibility BEFORE compile so we can still
    # inspect the underlying model's layer attributes (compile wraps the module).
    # transformers >= 4.52 moved attention_type off the layer; fall back to layer_type,
    # then config.layer_types, and finally assume full_attention for standard Qwen3.
    def _attn_type(model) -> str:
        layer = model.model.layers[0]
        if hasattr(layer, "attention_type"):
            return layer.attention_type
        if hasattr(layer, "layer_type"):
            return layer.layer_type
        cfg_types = getattr(model.config, "layer_types", None)
        return cfg_types[0] if cfg_types else "full_attention"

    assert _attn_type(p_model) == "full_attention", \
        "Custom attention mask not compatible with this HF Model"
    assert _attn_type(q_model) == "full_attention", \
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
