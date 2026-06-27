"""
losses/compute.py — model-forward scaffolding for each loss family.

The pure divergence math lives in losses/flat.py and losses/tree.py.
These functions wire up the teacher/draft forward passes around it.

Reading guide:
  core training paths  → compute_flat_loss, compute_tree_loss
  loss variants        → compute_flat_enrich_loss, compute_offpolicy_tree_loss,
                         compute_enrichment_loss
  depth probe          → expected_depth_scalar  (only needed by --aux_mode depth_weight)
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from inference_util import iid_draft, target_tree_pass
from verifier import TreeVerifier


def draft_tree_forward_with_grad(
    draft_model, prompt_ids: torch.Tensor,
    q_paths: List[List[int]], L: int, q_temp: float,
) -> Dict[str, torch.Tensor]:
    """
    Re-run the student over the K sampled paths in ONE tree-attention forward,
    keeping requires_grad=True on the resulting distributions.

    The paths themselves were sampled without grad (iid_draft).  Here we just
    score every node of that fixed tree under the current student weights so
    the loss can back-propagate through q_probs_dict.

    Returns {prefix: q_probs[V]} for every NON-LEAF node (depth < L).
    """
    device = prompt_ids.device
    dtype  = next(draft_model.parameters()).dtype

    # 1. Build unique prefix list (same node ordering as target_tree_pass).
    q_prefixes, q_token_ids = [], []
    for path in q_paths:
        for i, tok in enumerate(path):
            pfx = ",".join(str(x) for x in path[:i + 1])
            if pfx not in q_prefixes:
                q_prefixes.append(pfx)
                q_token_ids.append(tok)
    n_nodes = len(q_prefixes)
    q_tokens = torch.tensor(q_token_ids, device=device, dtype=torch.long).unsqueeze(0)

    # 2. Prompt prefill (no grad — prompt is fixed).
    with torch.no_grad():
        out        = draft_model(prompt_ids, use_cache=True, return_dict=True)
        p_cache    = out.past_key_values
        cached_len = prompt_ids.shape[1]

    # 3. Tree attention mask: node i attends to ancestors j (and itself).
    mask = torch.zeros((n_nodes, cached_len + n_nodes), device=device, dtype=dtype)
    mask[:, cached_len:] = torch.finfo(dtype).min          # default: no cross-node attn
    for i, pi in enumerate(q_prefixes):
        for j, pj in enumerate(q_prefixes):
            if pi.startswith(pj + ",") or pi == pj:
                mask[i, cached_len + j] = 0.0
    mask = mask.unsqueeze(0).unsqueeze(0)                  # [1, 1, n_nodes, total]

    # 4. Tree forward WITH grad.  Qwen3 expects mask in dict form (same as
    #    verifiers/inference_util.py:target_tree_pass).
    draft_model.train()
    out = draft_model(
        q_tokens, past_key_values=p_cache,
        attention_mask={"full_attention": mask},
        use_cache=True, return_dict=True,
    )
    logits = out.logits[0].float()                         # [n_nodes, V]
    probs  = F.softmax(logits / q_temp, dim=-1)            # [n_nodes, V] WITH grad

    # 5. Keep only non-leaf nodes (depth 0 .. L-1) — the verifier losses only
    #    query q_probs_dict at internal nodes.
    return {
        pfx: probs[i]
        for i, pfx in enumerate(q_prefixes)
        if len(pfx.split(",")) - 1 < L
    }


# ---------------------------------------------------------------------------
# Flat-loss path: teacher generates a sequence, student forward, divergence.
# ---------------------------------------------------------------------------

def compute_flat_loss(loss_fn, draft, teacher, prompt_ids,
                       max_new_tokens=128):
    """
    Teacher greedily generates max_new_tokens.  Student is then forwarded on
    [prompt + generated_tokens] WITH grad.  Loss = divergence(student_logits,
    teacher_logits) on the generated portion only.
    """
    with torch.no_grad():
        # Teacher rollout — argmax (do_sample=False) for stable training data.
        # Drop `temperature` because it's silently ignored when do_sample=False
        # (HF warns about it).  Pass an explicit attention_mask of all-ones
        # because we set pad_token=eos_token, and HF cannot infer the mask
        # in that case for a single un-padded prompt.
        attn_mask = torch.ones_like(prompt_ids)
        gen = teacher.generate(
            prompt_ids, attention_mask=attn_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=teacher.config.eos_token_id,
            use_cache=True,
        )
        # Forward teacher once on the full sequence to grab logits for the loss.
        t_out = teacher(gen, return_dict=True)
        t_logits = t_out.logits[0, prompt_ids.shape[1]-1:-1].float()   # [T, V]

    s_out = draft(gen, return_dict=True)
    s_logits = s_out.logits[0, prompt_ids.shape[1]-1:-1].float()        # [T, V]
    return loss_fn(s_logits, t_logits)


# ---------------------------------------------------------------------------
# Flat enrichment loss: K stochastic teacher rollouts, student scored on each.
# Apples-to-apples with compute_flat_loss — same max_new_tokens, same JSD at
# every token.  Only variable vs jsd flat: do_sample=True (stochastic teacher)
# and K paths averaged.  K=1 isolates greedy-vs-stochastic; K>1 adds diversity.
# Cost: K × compute_flat_loss per step.
# ---------------------------------------------------------------------------

def compute_flat_enrich_loss(loss_fn, draft, teacher, prompt_ids,
                             K, max_new_tokens=128, teacher_temp=1.0):
    """Sample K stochastic teacher rollouts; score student on each; average loss.

    Returns (loss, path_diversity) where path_diversity is the fraction of token
    positions where at least one path disagrees with path 0 (0 = all paths identical,
    1 = all positions differ).  Used to diagnose whether K paths add novel contexts.
    """
    attn_mask = torch.ones_like(prompt_ids)
    losses = []
    gen_tokens = []   # collect generated token ids for diversity measurement
    for _ in range(K):
        with torch.no_grad():
            gen = teacher.generate(
                prompt_ids, attention_mask=attn_mask,
                max_new_tokens=max_new_tokens, do_sample=True,
                temperature=teacher_temp,
                pad_token_id=teacher.config.eos_token_id,
                use_cache=True,
            )
            gen_tokens.append(gen[0, prompt_ids.shape[1]:])   # [T_i]
            t_out    = teacher(gen, return_dict=True)
            t_logits = t_out.logits[0, prompt_ids.shape[1]-1:-1].float()   # [T, V]

        s_out    = draft(gen, return_dict=True)
        s_logits = s_out.logits[0, prompt_ids.shape[1]-1:-1].float()        # [T, V]
        losses.append(loss_fn(s_logits, t_logits))

    # Path diversity: fraction of positions where paths disagree (only when K > 1).
    if K > 1:
        min_len = min(t.shape[0] for t in gen_tokens)
        stacked = torch.stack([t[:min_len] for t in gen_tokens], dim=0)  # [K, T]
        path_diversity = (stacked != stacked[0:1]).any(dim=0).float().mean().item()
    else:
        path_diversity = 0.0

    return sum(losses) / K, path_diversity


# ---------------------------------------------------------------------------
# Prefix-overlap loss (Rahul's "Prefix-Overlap Distillation Objective").
# Sample M teacher continuations from the prompt; reward the student's
# probability of every teacher prefix:
#     J = (1/M) Σ_m Σ_{t=1}^L qθ(P^(m)_{1:t} | c),   loss = -J.
# Exact-match acceptance (per-token factor = qθ at the teacher token) and all
# prefix lengths weighted equally (no depth reweighting) — faithful to the spec
# doc §6.  One teacher-forced forward pass per continuation (vs L in the doc's
# reference sketch); Σ_t exp(S_t) = exp(logsumexp(S)) for numerical stability.
# Pair with a CE term via --aux_loss forward_kl --aux_weight λ (doc §7).
# ---------------------------------------------------------------------------

def compute_prefix_overlap_loss(draft, teacher, prompt_ids,
                                M, L, teacher_temp=1.0, ce_weight=0.0):
    """Sample M teacher continuations of length L; loss = -mean_m Σ_t qθ(P_{1:t}|c).

    When ce_weight > 0, add the doc §7 cross-entropy term on the SAME sampled
    continuations: + ce_weight · mean_m mean_t (-log qθ(P_t|·)).  This is the
    faithful L_total = L_prefix + λ·L_CE — no separate rollout, no aux path.
    """
    attn_mask = torch.ones_like(prompt_ids)
    C = prompt_ids.shape[1]
    prefix_terms, ce_terms = [], []
    for _ in range(M):
        with torch.no_grad():
            gen = teacher.generate(
                prompt_ids, attention_mask=attn_mask,
                max_new_tokens=L, do_sample=True, temperature=teacher_temp,
                pad_token_id=teacher.config.eos_token_id, use_cache=True,
            )
        cont = gen[0, C:]                                   # teacher tokens P_1..P_len  [<=L]
        if cont.numel() == 0:
            continue                                        # teacher emitted EOS immediately
        s_out    = draft(gen, return_dict=True)
        s_logits = s_out.logits[0, C - 1:-1].float()        # predicts P_1..P_len  [len, V]
        logp     = F.log_softmax(s_logits, dim=-1)
        tok_lp   = logp.gather(-1, cont.unsqueeze(-1)).squeeze(-1)   # log qθ(P_t|·)  [len]
        S        = torch.cumsum(tok_lp, dim=0)              # log qθ(P_{1:t}|c)        [len]
        prefix_terms.append(torch.logsumexp(S, dim=0).exp())  # Σ_t exp(S_t) = Σ_t qθ(P_{1:t})
        ce_terms.append(-tok_lp.mean())                    # teacher-forcing CE, same tokens
    if not prefix_terms:
        # All M continuations were empty (teacher emitted EOS) — zero loss w/ grad.
        return draft(prompt_ids, return_dict=True).logits.sum() * 0.0
    loss = -torch.stack(prefix_terms).mean()
    if ce_weight > 0.0:
        loss = loss + ce_weight * torch.stack(ce_terms).mean()
    return loss


# ---------------------------------------------------------------------------
# Tree-loss path: sample K student paths, target tree pass, student tree pass.
# ---------------------------------------------------------------------------

def compute_tree_loss(loss_fn, draft, teacher, prompt_ids,
                      K, L, draft_temp, teacher_temp):
    """
    Build a fresh K×L draft tree from the student, score it under both models,
    then call the tree loss on (q_probs_dict_grad, p_probs_dict, q_paths, L, K).
    """
    # 1. Pending token from teacher's last position (matches inference).
    with torch.no_grad():
        t_out = teacher(prompt_ids, use_cache=True, return_dict=True)
        p_cache = t_out.past_key_values
        p_probs_last = F.softmax(t_out.logits[:, -1, :] / teacher_temp, dim=-1)
        context_pending = torch.multinomial(p_probs_last, num_samples=1)

    # 2. Sample K student draft paths (no grad — paths are fixed inputs).
    with torch.no_grad():
        q_out_prefill = draft(prompt_ids, use_cache=True, return_dict=True)
        q_cache = q_out_prefill.past_key_values
        q_paths, _, _ = iid_draft(
            draft, q_cache, context_pending, K=K, L=L, q_temp=draft_temp,
        )

    # 3. Target tree forward (no grad — teacher is frozen).
    with torch.no_grad():
        _, _, _, p_probs_dict = target_tree_pass(
            teacher, p_cache, q_paths, K=K, L=L, p_temp=teacher_temp,
        )

    # 4. Student tree forward WITH grad → q_probs_dict_grad.
    q_probs_dict_grad = draft_tree_forward_with_grad(
        draft, prompt_ids, q_paths, L=L, q_temp=draft_temp,
    )

    # 5. Call the loss.
    return loss_fn(q_probs_dict_grad, p_probs_dict, q_paths, L, K)


# ---------------------------------------------------------------------------
# Off-policy tree loss: teacher's greedy sequence is the training path.
# This avoids on-policy survival collapse — teacher tokens have near-unit
# self-acceptance so survival products stay non-negligible at depth L.
# q_probs are scored with grad; p_probs are frozen teacher logits on same path.
# ---------------------------------------------------------------------------

def compute_offpolicy_tree_loss(loss_fn, draft, teacher, prompt_ids,
                                K, L, draft_temp, teacher_temp):
    """
    Use teacher's greedy rollout (L+1 tokens) as the single training path.
    Teacher scores its own tokens → α stays high → survival products non-negligible.
    Student is scored on the same path with grad → acceptance gradient flows cleanly.
    """
    with torch.no_grad():
        attn_mask = torch.ones_like(prompt_ids)
        gen = teacher.generate(
            prompt_ids, attention_mask=attn_mask,
            max_new_tokens=L + 1, do_sample=False,
            pad_token_id=teacher.config.eos_token_id,
            use_cache=True,
        )
        teacher_tokens = gen[0, prompt_ids.shape[1]:].tolist()  # L+1 tokens
        # Build single-path list: [context_pending, tok1, …, tokL].
        # target_tree_pass expects paths of length L+1.
        q_paths = [teacher_tokens[: L + 1]]

        t_cache = teacher(prompt_ids, use_cache=True, return_dict=True).past_key_values
        _, _, _, p_probs_dict = target_tree_pass(
            teacher, t_cache, q_paths, K=1, L=L, p_temp=teacher_temp,
        )

    q_probs_dict_grad = draft_tree_forward_with_grad(
        draft, prompt_ids, q_paths, L=L, q_temp=draft_temp,
    )

    return loss_fn(q_probs_dict_grad, p_probs_dict, q_paths, L, K)


# ---------------------------------------------------------------------------
# Enrichment tree loss: the K branches are sampled from the TEACHER (its own
# plausible continuations), not the draft.  Identical to compute_tree_loss
# except iid_draft runs on the teacher.  The draft is pulled toward the teacher
# by JSD at every node of that teacher-generated tree — covering the off-greedy
# branch states the verifier visits at inference, where flat JSD never trains.
# K=1 reduces to a single teacher path (the matched control); K>1 is enrichment.
# No acceptance/telescoping term: the loss is whatever loss_fn is (jsd_enrich).
# ---------------------------------------------------------------------------

def compute_enrichment_loss(loss_fn, draft, teacher, prompt_ids,
                            K, L, draft_temp, teacher_temp):
    """Teacher-enriched tree: sample K teacher branches, score under both models,
    then call the loss on (q_probs_dict_grad, p_probs_dict, q_paths, L, K)."""
    with torch.no_grad():
        # 1. Pending token from the teacher's last position.
        t_out = teacher(prompt_ids, use_cache=True, return_dict=True)
        p_cache_score = t_out.past_key_values
        p_probs_last = F.softmax(t_out.logits[:, -1, :] / teacher_temp, dim=-1)
        context_pending = torch.multinomial(p_probs_last, num_samples=1)

        # 2. Sample K TEACHER paths.  iid_draft expands/consumes its cache, so
        #    use a fresh teacher prefill here, separate from the scoring cache.
        p_cache_sample = teacher(prompt_ids, use_cache=True,
                                 return_dict=True).past_key_values
        q_paths, _, _ = iid_draft(
            teacher, p_cache_sample, context_pending, K=K, L=L, q_temp=teacher_temp,
        )

        # 3. Target tree pass — teacher distributions at every node (JSD targets).
        _, _, _, p_probs_dict = target_tree_pass(
            teacher, p_cache_score, q_paths, K=K, L=L, p_temp=teacher_temp,
        )

    # 4. Draft tree forward WITH grad over the teacher-generated tree.
    q_probs_dict_grad = draft_tree_forward_with_grad(
        draft, prompt_ids, q_paths, L=L, q_temp=draft_temp,
    )

    # 5. Per-node loss (jsd_enrich → JSD at every node).
    return loss_fn(q_probs_dict_grad, p_probs_dict, q_paths, L, K)


# ---------------------------------------------------------------------------
# (3) Depth-as-weight: scalar E[τ_V] over the draft tree, used to MULTIPLY a
# flat loss.  The depth has NO gradient (pure-Python DP) — this is per-prompt
# loss reweighting, not an acceptance gradient.  Cost: one extra target tree
# pass per step.  See --aux_mode depth_weight.
# ---------------------------------------------------------------------------

@torch.no_grad()
def expected_depth_scalar(draft, teacher, prompt_ids, K, L, verifier,
                          draft_temp, teacher_temp) -> float:
    """E[accepted depth] for `verifier` on a fresh student draft tree (no grad)."""
    t_out = teacher(prompt_ids, use_cache=True, return_dict=True)
    p_cache = t_out.past_key_values
    p_probs_last = F.softmax(t_out.logits[:, -1, :] / teacher_temp, dim=-1)
    context_pending = torch.multinomial(p_probs_last, num_samples=1)

    q_out = draft(prompt_ids, use_cache=True, return_dict=True)
    q_paths, _, _ = iid_draft(draft, q_out.past_key_values, context_pending,
                              K=K, L=L, q_temp=draft_temp)
    q_prefixes, _, _, p_probs_dict = target_tree_pass(
        teacher, p_cache, q_paths, K=K, L=L, p_temp=teacher_temp)
    q_probs_dict = draft_tree_forward_with_grad(draft, prompt_ids, q_paths,
                                                L=L, q_temp=draft_temp)

    tv = TreeVerifier(q_paths, q_prefixes, q_probs_dict, p_probs_dict)
    depths = getattr(tv, f"expected_{verifier}_depths")(L)   # list over cutoffs
    return float(depths[-1])                                 # full-depth E[τ_V]