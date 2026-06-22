"""
validation.py — during-training val metrics: block efficiency + forgetting.

Called every VAL_EVERY steps from the training loop.  Uses 25 prompts at a
low temperature so val overhead stays small.  _update_forgetting is a backward-
transfer (forgetting) diagnostic computed as a free byproduct of the same val
pass — no extra forward passes.

Why _update_forgetting lives here and not in diagnostics.py:
  diagnostics.py is for offline post-hoc analysis (the --diagnose flag in eval.py).
  _update_forgetting is a live training-time metric that reuses the val pass; it
  has no meaning outside of that pass.
"""
from __future__ import annotations

import torch

from main import speculative_decoding_loop
from verifier_safe import VerifierError
from config import block_eff


@torch.no_grad()
def compute_val_metrics(draft, teacher, tokenizer, val_prompts,
                        mode, val_temp, max_new_tokens, val_k, val_l,
                        n_prompts=25):
    """Returns (aggregate_block_eff, per_prompt_block_eff) over n_prompts prompts.

    per_prompt_block_eff is a {prompt_index: block_eff} dict — used by the caller
    to compute a backward-transfer (forgetting) metric without any extra forward
    passes: it is the SAME val pass, just keeping per-prompt numbers instead of
    only their aggregate.

    VerifierError (from verifier_safe.py) is caught per-prompt so a bug in
    verifier.py does not abort training.  Skipped prompts are excluded from the
    block_eff denominator; if all prompts fail, aggregate returns nan.
    """
    draft.eval()
    total_gen, total_calls, skipped = 0, 0, 0
    per_prompt: dict[int, float] = {}
    for i, prompt in enumerate(val_prompts[:n_prompts]):
        teacher._spec_prompt_idx = i
        try:
            speculative_decoding_loop(
                p_model=teacher, q_model=draft, tok=tokenizer,
                prompt=prompt, verification_algo=mode,
                max_new_tokens=max_new_tokens, K=val_k, L=val_l,
                p_temp=val_temp, q_temp=val_temp,
            )
        except VerifierError as ve:
            skipped += 1
            print(f"  [val-skip] prompt {i}: {ve}")
            continue
        s = teacher._spec_run_stats
        total_gen   += s["gen_tokens"]
        total_calls += s["target_calls"]
        per_prompt[i] = block_eff(s["gen_tokens"], s["target_calls"])
    if skipped:
        print(f"  [val] {skipped}/{n_prompts} prompts skipped (verifier errors)")
    draft.train()



def _update_forgetting(best_per_prompt: dict, val_pp: dict) -> float:
    """Backward-transfer (forgetting) metric (Lopez-Paz & Ranzato 2017).

    For each val prompt, track its best-ever block_eff.  Forgetting at this step
    is the mean drop from each prompt's personal best to its current value:
        forgetting = mean_i max(0, best_i - current_i)
    0 = no prompt has regressed below its peak; large = the student learned
    some prompts then lost them (signature of H3: teacher confusing the student).
    Updates best_per_prompt in place.  Uses the SAME val prompts as block_eff —
    no extra forward passes.
    """
    if not val_pp:
        return 0.0
    drops = []
    for idx, be in val_pp.items():
        prev_best = best_per_prompt.get(idx, be)
        drops.append(max(0.0, prev_best - be))
        best_per_prompt[idx] = max(prev_best, be)
    return sum(drops) / len(drops)