# Adding a New Model Family

This guide walks through adding support for a new LLM architecture (e.g. Gemma, LLaMA, Mistral) to the distillation trainer.

## What "model family" controls

Each family encapsulates four model-specific behaviours:

| Concern | Where it affects training |
|---|---|
| LoRA target modules | Which Linear layers get adapters (architecture-specific names) |
| Temperature pre-scaling | Whether `generate(output_scores=True)` returns `logits/temp` or raw `logits` |
| Forbidden-token masking | Whether the model applies `-inf` logit bias to control tokens |
| Chat template | How raw prompts are formatted before tokenisation |

If any of these are wrong, training will silently produce bad results or NaN.

## Step-by-step

### 1. Verify the four behaviours experimentally

Before writing code, run these checks for your new model:

```python
import torch, transformers

model_id = "google/gemma-2-2b"
model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda()
tok   = transformers.AutoTokenizer.from_pretrained(model_id)

prompt_ids = tok("Hello world", return_tensors="pt").input_ids.cuda()
TEMP = 0.8

# Run generate() with two temperatures, compare output_scores
out1 = model.generate(prompt_ids, max_new_tokens=5, temperature=TEMP,
                       do_sample=True, output_scores=True, return_dict_in_generate=True)
out2 = model.generate(prompt_ids, max_new_tokens=5, temperature=1.0,
                       do_sample=True, output_scores=True, return_dict_in_generate=True)

s1 = torch.stack(out1.scores, 0).float()
s2 = torch.stack(out2.scores, 0).float()

# If s1 ≈ s2 * TEMP: the model pre-divides by temp → recover_raw_logits should return scores * TEMP
# If s1 ≈ s2:        the model does NOT pre-divide  → recover_raw_logits should return scores as-is
print("Max ratio s1/s2:", (s1 / (s2 + 1e-9)).abs().max().item())

# Check for -inf in log_softmax output (forbidden tokens)
import torch.nn.functional as F
log_p = F.log_softmax(s1[0], dim=-1)
n_neginf = (log_p == float('-inf')).sum().item()
print(f"Tokens with -inf log_prob: {n_neginf} / {log_p.shape[-1]}")
# If n_neginf > 0: clamp_log_probs should return log_probs.clamp(min=-100.0)
# If n_neginf == 0: clamp_log_probs can return log_probs unchanged
```

### 2. List the LoRA target modules for your architecture

```python
# Print all Linear layer names in the model
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        # Use the suffix after the last '.' as the target_modules entry
        print(name)
```

Typical results:
- **Gemma 2**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **LLaMA 3**: same as Gemma 2
- **Mistral**: same as LLaMA 3

### 3. Create the family file

Copy `capsules/distillation/model_families/gemma.py` as a starting point:

```bash
cp capsules/distillation/model_families/gemma.py \
   capsules/distillation/model_families/llama.py
```

Edit the copy, filling in:

```python
class LlamaFamily(ModelFamily):
    @property
    def name(self) -> str:
        return "llama"

    def lora_target_modules(self):
        return ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

    def recover_raw_logits(self, output_scores, temperature):
        # From Step 1: LLaMA does NOT pre-divide
        return output_scores

    def clamp_log_probs(self, log_probs):
        # From Step 1: LLaMA has no -inf entries
        return log_probs   # or .clamp(min=-100.0) to be safe

    def format_prompt(self, prompt, tokenizer):
        try:
            messages = [{"role": "user", "content": prompt}]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return prompt

    @property
    def default_draft_model_id(self):
        return "meta-llama/Llama-3.2-1B"

    @property
    def default_target_model_id(self):
        return "meta-llama/Llama-3.1-8B"

FAMILY = LlamaFamily()
```

### 4. Register in `__init__.py`

Edit `capsules/distillation/model_families/__init__.py`:

```python
from .llama import FAMILY as _llama

FAMILY_REGISTRY = {
    "qwen":  _qwen,
    "gemma": _gemma,
    "llama": _llama,   # ← add this line
}
```

### 5. Run a smoke test

```bash
python -m capsules.distillation.trainer \
    --model_family llama \
    --draft meta-llama/Llama-3.2-1B \
    --target meta-llama/Llama-3.1-8B \
    --loss forward_kl --steps 10 \
    --dataset capsules/datasets/raw/gsm8k_30.jsonl \
    --no_wandb
```

Expected output: 10 steps with finite loss values (no NaN/Inf).

### 6. Verify vocabulary compatibility

**Critical**: speculative decoding requires the draft and target to share the same tokenizer vocabulary.

```python
tok_draft  = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tok_target = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
assert tok_draft.vocab_size == tok_target.vocab_size, \
    f"Vocabulary mismatch: {tok_draft.vocab_size} vs {tok_target.vocab_size}"
print(f"Vocabulary size: {tok_draft.vocab_size} ✓")
```

Cross-family pairs (e.g. Qwen draft + LLaMA target) are **not supported**.

## Common mistakes

| Symptom | Likely cause | Fix |
|---|---|---|
| NaN from step 1 | `-inf` tokens + reverse KL | Set `clamp_log_probs` to clamp |
| Loss doesn't decrease | Temperature applied twice | Verify `recover_raw_logits` |
| Poor acceptance rate | Wrong LoRA modules | Re-check module names |
| Tokenizer error | Cross-family vocab mismatch | Use same-family draft+target |
