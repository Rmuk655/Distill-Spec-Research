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
def _dtype_kwargs(torch_dtype) -> dict:
    """Return correct dtype kwarg for from_pretrained().
    transformers >= 4.51 (required for Qwen3) deprecates torch_dtype= in favour of dtype=.
    Always use dtype= — no version check needed.
    """
    return {"dtype": torch_dtype}


def load_models(
    p_name: str,
    q_name: str,
    device: str = "cuda",
    dtype: str = "bf16",
    compile_draft: bool = False,
    load_in_4bit: bool = False,
):
    """Load target (p_model) and draft (q_model) for speculative decoding.

    load_in_4bit=True loads the TARGET in 4-bit NF4 via bitsandbytes.
    Required on free Colab T4 (15 GB) for Qwen3-8B (bfloat16 = ~16 GB → OOM).
    The draft is always loaded in the requested dtype.
    """
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    tok = AutoTokenizer.from_pretrained(p_name, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if dtype == "auto":
        torch_dtype = "auto"
    elif dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16 if "cuda" in str(device) else torch.float32

    if load_in_4bit and device != "cpu":
        # NF4 quantisation — CUDA only; device_map="auto" handles multi-GPU
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise SystemExit(
                "bitsandbytes is required for --load_in_4bit.\n"
                "Install with:  pip install 'bitsandbytes>=0.46.1'"
            )
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
            device_map="auto",      # quantized models must use device_map; "auto"
                                    # uses cuda:0 on a single GPU, or splits across
                                    # both T4s on Kaggle T4 x2 (the only Kaggle GPU)
            low_cpu_mem_usage=True, # load shard-by-shard; avoids RAM spike
            use_safetensors=True,
        )
        print("[INFO] Target model loaded in 4-bit NF4 (QLoRA mode).")
    else:
        # BF16 or CPU fallback (NF4 requires CUDA; CPU retry uses BF16)
        if load_in_4bit and device == "cpu":
            print("[WARN] 4-bit NF4 requires CUDA — loading teacher in BF16 on CPU for OOM retry.")
            torch_dtype = torch.bfloat16
        p_model = AutoModelForCausalLM.from_pretrained(
            p_name,
            trust_remote_code=True,
            **_dtype_kwargs(torch_dtype),
            low_cpu_mem_usage=True,
            device_map=None,
            use_safetensors=True,
        ).to(dev)
    p_model.eval()

    q_model = AutoModelForCausalLM.from_pretrained(
        q_name,
        trust_remote_code=True,
        **_dtype_kwargs(torch_dtype),
        low_cpu_mem_usage=True,
        device_map=None,
        use_safetensors=True,
    ).to(dev)
    q_model.eval()

    torch.set_grad_enabled(False)

    # Validate custom attention mask compatibility BEFORE compile so we can still
    # inspect the underlying model's layer attributes (compile wraps the module).
    # Architecture-agnostic: different model families expose layers differently:
    #   Qwen / LLaMA / Gemma : model.model.layers[0]
    #   GPT-2               : model.transformer.h[0]
    # Use _first_layer() to handle all cases without AttributeError.
    def _first_layer(m):
        """Return the first decoder layer regardless of architecture, or None."""
        inner  = getattr(m, "model", None) or getattr(m, "transformer", None)
        layers = getattr(inner, "layers", None) or getattr(inner, "h", None)
        return layers[0] if layers else None

    for _m in (p_model, q_model):
        _layer0 = _first_layer(_m)
        if _layer0 is not None and hasattr(_layer0, "attention_type"):
            assert _layer0.attention_type == "full_attention", \
                f"Custom attention mask not compatible with {type(_m).__name__}"

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


# ---------------------------------------------------------------------------
# Architecture-agnostic KV-cache adapter
# ---------------------------------------------------------------------------

class _CacheLayer:
    """Single-layer (key, value) pair with named attributes."""
    __slots__ = ("keys", "values")
    def __init__(self, k, v):
        self.keys   = k
        self.values = v


class CompatCache:
    """Wraps any model's past_key_values into a uniform .layers[i].keys/.values
    interface so the verifier works with GPT-2, LLaMA, Gemma, etc. in addition
    to Qwen3's native custom cache.

    Input formats handled
    ─────────────────────
    • Qwen3 native cache  — already has .layers[i].keys/.values  → pass-through
    • Legacy tuple         — ((k0,v0),(k1,v1),…) from GPT-2 etc. → wrapped
    • HF DynamicCache      — .key_cache[i] / .value_cache[i]      → wrapped

    After slice_cache() / expand_cache() modify .layers[i].keys in place,
    call .to_model_format() to reconstruct the format the model expects before
    the next forward() call.
    """
    def __init__(self, raw):
        if isinstance(raw, CompatCache):
            # Already wrapped — share layers (no copy)
            self.layers = raw.layers
            self._raw   = raw._raw
            self._style = raw._style
        elif hasattr(raw, "layers"):
            # Qwen3-style native cache with .layers[i].keys/.values
            self.layers = raw.layers
            self._raw   = raw
            self._style = "native"
        elif isinstance(raw, (tuple, list)):
            # Legacy tuple ((k0,v0),(k1,v1),…) — GPT-2, older HF models
            self.layers = [_CacheLayer(k, v) for k, v in raw]
            self._raw   = raw
            self._style = "tuple"
        elif hasattr(raw, "key_cache"):
            # HuggingFace DynamicCache (.key_cache / .value_cache lists)
            n = len(raw.key_cache)
            self.layers = [_CacheLayer(raw.key_cache[i], raw.value_cache[i])
                           for i in range(n)]
            self._raw   = raw
            self._style = "hf_dc"
        else:
            raise TypeError(f"CompatCache: unrecognised past_key_values type {type(raw)}")

    def to_model_format(self):
        """Reconstruct the format the model's forward() expects.

        For native Qwen3 caches: returns the original object (the verifier
        modifies .layers[i] in-place, so it stays consistent).
        For tuple/HF caches: reconstructs from the (possibly modified) layers.
        """
        if self._style == "native":
            return self._raw
        if self._style == "tuple":
            return tuple((l.keys, l.values) for l in self.layers)
        # hf_dc: rebuild DynamicCache
        try:
            from transformers import DynamicCache as _DC
            dc = _DC()
            for l in self.layers:
                dc.key_cache.append(l.keys)
                dc.value_cache.append(l.values)
            return dc
        except Exception:
            return tuple((l.keys, l.values) for l in self.layers)

    def __iter__(self):
        """Iterate as (keys, values) pairs per layer (for profiling code)."""
        return iter((l.keys, l.values) for l in self.layers)


def _attn_mask_for_model(model, mask_4d: torch.Tensor):
    """Return the attention mask in the format model.forward() expects.

    Qwen3:  attention_mask={"full_attention": tensor}  (custom dict)
    Others: attention_mask=tensor  (standard 4D additive bias)
    """
    inner  = getattr(model, "model", None) or getattr(model, "transformer", None)
    layers = getattr(inner, "layers", None) or getattr(inner, "h", None)
    if layers and hasattr(layers[0], "attention_type"):
        return {"full_attention": mask_4d}
    return mask_4d


"""
Takes in:
    - cache: model KV-cache (CompatCache), each KV tensor has shape (batch_sz, num_heads, context_len, head_dim)
    - batch_idx: list of indices to slice the batch_sz dimension along
    - token_idx: list of indices to slice the context_len dimension along
Returns:
    - cache: model KV-cache (CompatCache), each KV tensor has shape (|batch_idx|, num_heads, |token_idx|, head_dim)
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
