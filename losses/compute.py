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
from .flat import jsd


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
    #
    # This dict-style full-attention mask is NOT a standard causal/padding mask
    # — it's an arbitrary pairwise bias matrix encoding tree ancestry. HF's
    # flash_attention_2 integration only understands causal or padding masks
    # (see modeling_flash_attention_utils.py:_upad_input); handed this instead,
    # it corrupts index computation and crashes with a device-side assert
    # ("index out of bounds") deep in flash_attn's unpad_input. This is the
    # exact incompatibility scripts/setup_a100.sh already documented for the
    # TARGET model (hence target stays off FA2 there) — but the small DRAFT
    # model (hidden_size < 3000) gets FA2 auto-selected by the same setup
    # script's .pth patch, and this is the one draft-side call that also needs
    # the custom mask. Force SDPA for just this call, restore FA2 afterward so
    # flat-loss / validation draft calls (plain causal, no custom mask) keep
    # the FA2 speedup.
    def _set_attn_impl(model, impl):
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation(impl)
        else:
            model.config._attn_implementation = impl   # fallback for older transformers

    _orig_attn_impl = getattr(draft_model.config, "_attn_implementation", None)
    if _orig_attn_impl == "flash_attention_2":
        _set_attn_impl(draft_model, "sdpa")
    try:
        draft_model.train()
        out = draft_model(
            q_tokens, past_key_values=p_cache,
            attention_mask={"full_attention": mask},
            use_cache=True, return_dict=True,
        )
    finally:
        if _orig_attn_impl == "flash_attention_2":
            _set_attn_impl(draft_model, _orig_attn_impl)
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
                             K, max_new_tokens=128, teacher_temp=1.0,
                             grad_accum=None):
    """Sample K stochastic teacher rollouts; score student on each; average loss.

    Memory note: with grad_accum=None (default), this returns a single summed
    loss tensor for the CALLER to backward (the original behavior) — all K
    draft-forward computation graphs are held simultaneously until that single
    backward() call, i.e. peak memory scales with K. This OOM'd in practice at
    K=3 against a 32B teacher with a thin memory margin (1.7B/32B capacity
    study, 2026-07-01).

    Pass grad_accum=GRAD_ACCUM to instead backward EACH rollout immediately
    (scaled by 1/(K*grad_accum)) and free its graph before starting the next
    rollout — peak memory then stays ~1x a single rollout regardless of K.
    In this mode the returned loss is DETACHED (for logging only); the caller
    must NOT call .backward() on it again, and must not combine it with
    depth_weight/aux_loss afterward (that combination would silently not
    contribute to the gradient, since backward has already happened here).

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
        loss_i   = loss_fn(s_logits, t_logits)
        if grad_accum is not None:
            (loss_i / (K * grad_accum)).backward()
            losses.append(loss_i.detach())
        else:
            losses.append(loss_i)

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

def _aux_term(aux, s_logits, tok_lp, teacher, gen, C):
    """Secondary term over one continuation, on the SAME sampled tokens.

      aux="ce"  : teacher-forcing cross-entropy  -Σ_t log qθ(P_t|·)  — doc §7,
                  verbatim (a SUM over t, matching the prefix term's per-root
                  scale; the doc never divides by L).  Reuses the gathered token
                  log-probs (no teacher forward).
      aux="jsd" : symmetric JSD(student, teacher) per token — beyond the doc,
                  needs the teacher distribution (one extra teacher forward).
    """
    if aux == "ce":
        return -tok_lp.sum()
    with torch.no_grad():
        t_logits = teacher(gen, return_dict=True).logits[0, C - 1:-1].float()
    return jsd(s_logits, t_logits)


def _prefix_score(S, objective, tok_lp=None, K=None, tok_p=None):
    """Map per-prefix log-probs S_t = log qθ(P_{1:t}) to the per-root reward.

      objective="prob"     : Σ_t qθ(P_{1:t}) = Σ_t exp(S_t)  — the doc's exact
                             E[LCP] objective (§4).  Gradient is survival-weighted
                             and COLLAPSES from a cold start (∏ aᵢ → 0 at depth).
      objective="logprob"  : Σ_t log qθ(P_{1:t}) = Σ_t S_t  — log-space variant.
                             Equals position-weighted CE Σ_i (L-i+1)·log aᵢ;
                             gradient never collapses.  NOT the doc's objective —
                             a trainability variant for cold-start experiments.
      objective="traversal": traversal-BE surrogate (needs tok_lp and K).  With
                             α_d = exp(S_t) (draft's joint prob of the teacher's
                             first d tokens) and K i.i.d. branches, acceptance at
                             depth d is A_d = 1-(1-α_d)^K and E[BE] = 1+Σ_d A_d.
                             We realise its gradient
                                 dE[BE]/d log q_i = Σ_{d≥i} K(1-α_d)^{K-1}α_d
                             via a detached-weight surrogate Σ_i w_i·log q(P_i).
                             Weight is adaptive in α AND K (per-depth term
                             K·α(1-α)^{K-1} peaks at α≈1/K), giving a built-in
                             curriculum.  This is an independent-branches
                             APPROXIMATION to the traversal verifier (shared
                             prefixes / node merging / verifier DP are ignored),
                             NOT the exact traversal objective.
      objective="nss"      : NSS-aligned one-sided CE (needs tok_lp and tok_p).
                             NSS acceptance per token is min(1, p_i/q_i); gradient
                             of the log-space survival sum flows only where q_i<p_i.
                             Same triangular weight L-i+1 as logprob, masked to the
                             catch-up zone: -Σ_i (L-i+1)·𝟙[p_i>q_i]·log q_i.
                             Requires tok_p=log p(P_i|·) from a teacher forward on
                             the same sampled tokens (shared with jsd-aux if active).
    """
    if objective == "traversal":
        alpha = torch.exp(S.detach())                                  # α_d
        wd    = K * (1.0 - alpha).pow(K - 1) * alpha                   # K α (1-α)^{K-1}
        wi    = torch.flip(torch.cumsum(torch.flip(wd, [0]), 0), [0])  # suffix sum: w_i
        return (wi * tok_lp).sum()
    if objective == "nss":
        active = ((tok_p - tok_lp) > 0).float().detach()              # 1 where q < p [L]
        wi = torch.arange(len(tok_lp), 0, -1,
                          dtype=tok_lp.dtype, device=tok_lp.device)   # L, L-1, ..., 1
        return (wi * active * tok_lp).sum()
    if objective == "logprob":
        return S.sum()
    return torch.logsumexp(S, dim=0).exp()


def compute_prefix_overlap_loss(draft, teacher, prompt_ids,
                                M, L, teacher_temp=1.0,
                                aux="ce", aux_weight=0.0, objective="prob", K=3):
    """Single-root prefix overlap (doc §4): M teacher continuations from the prompt.

    loss = -mean_m score(P^(m))  [ + aux_weight · mean_m aux_term ], where
    score = Σ_t qθ(P_{1:t})  (objective="prob", doc) or
            Σ_t log qθ(P_{1:t})  (objective="logprob", trainability variant) or
            the traversal-BE surrogate (objective="traversal"; uses K branches).
    The continuations are sampled from the prompt only (root = prompt).
    """
    attn_mask = torch.ones_like(prompt_ids)
    C = prompt_ids.shape[1]
    prefix_terms, aux_terms = [], []
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
        tok_tp = None
        if objective == "nss":
            with torch.no_grad():
                t_logits_nss = teacher(gen, return_dict=True).logits[0, C - 1:-1].float()
                tok_tp = F.log_softmax(t_logits_nss, dim=-1).gather(
                    -1, cont.unsqueeze(-1)).squeeze(-1)     # log p(P_t|·)  [len]
        prefix_terms.append(_prefix_score(S, objective, tok_lp=tok_lp, K=K, tok_p=tok_tp))
        if aux_weight > 0.0:
            aux_terms.append(_aux_term(aux, s_logits, tok_lp, teacher, gen, C))
    if not prefix_terms:
        # All M continuations were empty (teacher emitted EOS) — zero loss w/ grad.
        return draft(prompt_ids, return_dict=True).logits.sum() * 0.0
    loss = -torch.stack(prefix_terms).mean()
    if aux_weight > 0.0:
        loss = loss + aux_weight * torch.stack(aux_terms).mean()
    return loss


def compute_prefix_overlap_multiroot_loss(draft, teacher, prompt_ids,
                                          L, N, rollout_len, teacher_temp=1.0,
                                          aux="ce", aux_weight=0.0,
                                          objective="prob", random_offset=False, K=3,
                                          min_root=0):
    """Multi-root prefix overlap (doc §5), efficient M=1 estimator.

    Generate ONE teacher rollout; its tail from each root y_{r+1:r+L} is a valid
    sample from p(·|c_r) (teacher is autoregressive), so sliding a length-L window
    every N positions gives an unbiased M=1 estimate of the §5 root-averaged
    objective.  Cost: one teacher generate + one student forward for all roots.
    NOTE: this is the tail-reuse variant — cheaper but correlated and M=1 — not
    the doc's literal "fresh M continuations per root."

    random_offset: roots start at o~Unif{0..N-1} (doc §5 uniform-over-positions)
                   instead of a fixed 0 (every-Nth objective).
    min_root:      skip all roots before this position (deep-bias: warm-start from
                   jsd_flat then only supervise the deep half of each rollout).
    objective:     "prob" (doc §4/§5), "logprob" (trainability variant),
                   "traversal" (K-aware surrogate), "nss" (NSS-aligned one-sided CE).
    loss = -mean_r score(y_{r+1:r+L})  [ + aux_weight · mean_r aux_term ].
    """
    attn_mask = torch.ones_like(prompt_ids)
    C = prompt_ids.shape[1]
    with torch.no_grad():
        gen = teacher.generate(
            prompt_ids, attention_mask=attn_mask,
            max_new_tokens=rollout_len, do_sample=True, temperature=teacher_temp,
            pad_token_id=teacher.config.eos_token_id, use_cache=True,
        )
    cont = gen[0, C:]                                       # rollout tokens y_1..y_T  [T]
    T = cont.numel()
    if T == 0:
        return draft(prompt_ids, return_dict=True).logits.sum() * 0.0
    s_out    = draft(gen, return_dict=True)
    s_logits = s_out.logits[0, C - 1:C - 1 + T].float()     # predicts y_1..y_T  [T, V]
    logp     = F.log_softmax(s_logits, dim=-1)
    tok_lp   = logp.gather(-1, cont.unsqueeze(-1)).squeeze(-1)   # log qθ(y_{i+1}|...)  [T]

    # Teacher logits over the whole rollout: computed once, shared by jsd-aux and nss.
    t_logits_all = None
    _need_teacher_logits = (aux_weight > 0.0 and aux != "ce") or objective == "nss"
    if _need_teacher_logits:
        with torch.no_grad():
            t_logits_all = teacher(gen, return_dict=True).logits[0, C - 1:C - 1 + T].float()

    # Teacher log-probs at teacher tokens — needed for NSS objective only.
    tok_lp_teacher = None
    if objective == "nss":
        tok_lp_teacher = F.log_softmax(t_logits_all, dim=-1).gather(
            -1, cont.unsqueeze(-1)).squeeze(-1)             # log p(y_t|·)  [T]

    start = int(torch.randint(0, N, (1,)).item()) if random_offset else 0
    start = max(start, min_root)                            # deep-bias: skip shallow roots
    prefix_terms, aux_terms = [], []
    for r in range(start, T - 1, N):                       # roots every N positions
        win_lp = tok_lp[r:r + L]                            # student log-probs from root r
        if win_lp.numel() == 0:
            break
        S = torch.cumsum(win_lp, dim=0)
        win_tp = tok_lp_teacher[r:r + L] if tok_lp_teacher is not None else None
        prefix_terms.append(_prefix_score(S, objective, tok_lp=win_lp, K=K, tok_p=win_tp))
        if aux_weight > 0.0:
            if aux == "ce":
                aux_terms.append(-win_lp.sum())            # doc §7: SUM over t (verbatim)
            else:                                           # jsd over the same window
                aux_terms.append(jsd(s_logits[r:r + L], t_logits_all[r:r + L]))
    if not prefix_terms:                                    # offset past end of rollout
        return draft(prompt_ids, return_dict=True).logits.sum() * 0.0
    loss = -torch.stack(prefix_terms).mean()
    if aux_weight > 0.0:
        loss = loss + aux_weight * torch.stack(aux_terms).mean()
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