import argparse
import os
import time
from typing import Dict, List, Tuple
import gc

import copy
import torch
from tqdm import tqdm

from inference_util import target_tree_pass
from util import load_models, set_seed
from verifier import TreeVerifier
from node import Node

from transformers import AutoModelForCausalLM, DynamicCache
from transformers.cache_utils import DynamicLayer


@torch.no_grad()
def build_cache_for_context(model, input_ids: torch.Tensor, chunk: int = 128) -> DynamicCache:
    n_layers = getattr(model.config, "num_hidden_layers", None)
    if n_layers is None:
        n_layers = getattr(getattr(model.config, "text_config", None), "num_hidden_layers", None)
    if n_layers is None:
        raise RuntimeError("Could not infer num_hidden_layers from config")
    cache = DynamicCache()
    cache.layers = [DynamicLayer() for _ in range(int(n_layers))]
    for s in range(0, input_ids.shape[1], chunk):
        x = input_ids[:, s:s+chunk]
        out = model(x, use_cache=True, past_key_values=cache, return_dict=True)
        cache = out.past_key_values
    return cache



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft_trees_pt", type=str, required=True)
    ap.add_argument("--nss", action="store_true")
    ap.add_argument("--naive", action="store_true")
    ap.add_argument("--specinfer", action="store_true")
    ap.add_argument("--spectr", action="store_true")
    ap.add_argument("--khisti", action="store_true")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    return args


"""
Loads all rollout data from path, including meta data.
Returns a dict on CPU consisting of "ex_id" -> (prompt_len, rollout_ids).
Also returns metadata for target sampling: p_model, p_temp, p_nucleus.
"""
def load_rollouts(path: str) -> Tuple[str, float, float, Dict[str, Tuple[int, torch.Tensor]]]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    meta = data["meta"]
    p_model = meta["p_model"]
    p_temp = float(meta["p_temp"])
    p_nucleus = float(meta["p_nucleus"])
    rollouts = {str(ex_id): (int(rec["prompt_len"]), rec["rollout_ids"]) for ex_id, rec in data["data"].items()}
    return p_model, p_temp, p_nucleus, rollouts


"""
Loads all branched draft tree data from path, including meta data.
Returns a dict on CPU consisting of "ex_id:offset" -> (draft_tok, branch_tok).
Also returns metadata for draft sampling: rollouts_pt, q_model, q_temp, q_nucleus, and full meta dict.
"""
def load_branched_draft_trees(
    path: str,
) -> Tuple[str, str, float, float, Dict, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    meta = data["meta"]
    rollouts_pt = meta["rollouts_pt"]
    q_model = meta["q_model"]
    q_temp = float(meta["q_temp"])
    q_nucleus = float(meta["q_nucleus"])
    draft_paths = data["draft_paths"]
    return rollouts_pt, q_model, q_temp, q_nucleus, meta, draft_paths


"""
Takes in:
    - cache: model KV-cache (DynamicCache), each KV tensor has shape (batch_sz, num_heads, context_len, head_dim)
    - max_len: what to limit the the context_len to
Returns a new cache:
    - cache: model KV-cache (DynamicCache), each KV tensor has shape (|batch_idx|, num_heads, max_len, head_dim)
"""
def create_prefix_cache(cache: DynamicCache, max_len: int) -> DynamicCache:
    new_cache = copy.copy(cache)
    new_cache.layers = [copy.copy(l) for l in cache.layers]
    new_cache.crop(max_len)
    return new_cache


"""
Builds a list of dictionaries, each containing the following data about each root:
    - key: "ex_id:offset"
    - ex_id: Dataset question ID
    - offset: What position the root is in the generated rollout, excluding prompt
    - prompt_len: Prompt length
    - ctx_end: prompt_len + offset
    - root_token: rollout_ids[ctx_end - 1]
    - context: Full context for root, including prompt and excluding the root token
    - draft_tok: Shape (draft_paths, draft_depth)
    - branch_tok: Shape (draft_paths, draft_depth + 1, branch_paths, branch_depth)
NOTE: This must iterate through the prompts in lexicographic order of (ex_id, offset) for maximum KV cache reuse.
"""
def build_roots_branched(
    rollouts: Dict[str, Tuple[int, torch.Tensor]],
    branched: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
) -> List[Dict]:
    roots = []
    for key, val in branched.items():
        ex_id, offset = key.rsplit(":", 1)
        offset = int(offset)
        if ex_id not in rollouts:
            continue
        prompt_len, rollout_ids = rollouts[ex_id]
        ctx_end = prompt_len + offset
        if ctx_end < 2 or ctx_end > len(rollout_ids):
            continue
        draft_tok, branch_tok = val
        root_token = int(rollout_ids[ctx_end - 1].item())
        roots.append(
            {
                "key": f"{ex_id}:{offset}",
                "ex_id": ex_id,
                "offset": offset,
                "prompt_len": int(prompt_len),
                "ctx_end": int(ctx_end),
                "root_token": root_token,
                "context": rollout_ids[: ctx_end - 1],
                "draft_tok": draft_tok,
                "branch_tok": branch_tok,
            }
        )
    roots.sort(key=lambda r: (r["ex_id"], r["offset"]))
    return roots


"""
Applies temperature and nucleus truncation (top-p) to produce a probability distribution.
Returns a torch.Tensor shape (V,) in float32 on the same device as logits.
"""
def probs_with_temp_top_p(logits: torch.Tensor, temp: float, top_p: float, vocab_size: int) -> torch.Tensor:
    if temp <= 0:
        raise ValueError("temp must be > 0")
    x = logits[..., :vocab_size].float() / float(temp)
    probs = torch.softmax(x, dim=-1)
    if top_p is None or float(top_p) >= 1.0:
        return probs
    tp = float(top_p)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    mask = cdf > tp
    if mask.numel() > 0:
        mask[0] = False
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    out = torch.zeros_like(probs)
    out.scatter_(0, sorted_idx, sorted_probs)
    z = out.sum()
    if z.item() <= 0:
        return probs
    return out / z


"""
Computes entropy, L1, and both KL directions between two distributions.
Inputs are torch.Tensor shape (V,) on any device, will compute in float32.
Returns (ent_p, ent_q, l1, kl_pq, kl_qp) as float32 tensors on CPU.
"""
def dist_metrics(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    p = p.float()
    q = q.float()
    logp = p.clamp_min(eps).log()
    logq = q.clamp_min(eps).log()
    ent_p = -(p * logp).sum()
    ent_q = -(q * logq).sum()
    l1 = (p - q).abs().sum()
    kl_pq = (p * (logp - logq)).sum()
    kl_qp = (q * (logq - logp)).sum()
    return ent_p.detach().cpu(), ent_q.detach().cpu(), l1.detach().cpu(), kl_pq.detach().cpu(), kl_qp.detach().cpu()


"""
Runs a single-token forward using a cropped cache.
Returns (last_hidden_state, logits) for that token, where logits predict the next token.
"""
@torch.no_grad()
def hidden_and_logits_at_token(
    model: AutoModelForCausalLM,
    full_cache: DynamicCache,
    prefix_len: int,
    token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cache = create_prefix_cache(full_cache, prefix_len)
    x = torch.tensor([[int(token_id)]], dtype=torch.long, device=model.device)
    out = model(x, use_cache=True, past_key_values=cache, return_dict=True, output_hidden_states=True)
    h = out.hidden_states[-1][0, -1, :].detach()
    logits = out.logits[0, -1, :].detach()
    return h, logits


"""
Builds a covering set of leaf paths for a single root.
For each L1 in [0..draft_depth], includes all i in [0..draft_paths-1] and all b in [0..max_k-1]
using full branch_depth tokens, so all prefixes needed by any (K,L1,L2) slice are covered.
Returns a list of q_paths where each path begins with root_token.
"""
def build_cover_paths(
    root_token: int,
    draft_tok: torch.Tensor,
    branch_tok: torch.Tensor,
    max_k: int,
    draft_depth: int,
    branch_depth: int,
) -> List[List[int]]:
    q_paths = []
    draft_lists = draft_tok.tolist()
    for l1 in range(draft_depth + 1):
        for i in range(draft_tok.shape[0]):
            prefix = [int(root_token)] + [int(x) for x in draft_lists[i][:l1]]
            for b in range(max_k):
                suffix = branch_tok[i, l1, b, :branch_depth].tolist()
                q_paths.append(prefix + [int(x) for x in suffix])
    return q_paths


"""
Runs target_tree_pass on the covering leaf paths and returns a merged probs_dict mapping prefix string -> probs tensor.
"""
@torch.no_grad()
def cover_tree_probs(
    model: AutoModelForCausalLM,
    cache: DynamicCache,
    root_token: int,
    draft_tok: torch.Tensor,
    branch_tok: torch.Tensor,
    vocab_size: int,
    temp: float,
    nucleus: float,
    max_k: int,
    draft_depth: int,
    branch_depth: int,
) -> Dict[str, torch.Tensor]:
    q_paths = build_cover_paths(root_token, draft_tok, branch_tok, max_k, draft_depth, branch_depth)
    K = len(q_paths)
    L = len(q_paths[0]) - 1
    _, _, _, probs_dict, _ = target_tree_pass(
        model,
        cache,
        q_paths,
        vocab_size,
        K=K,
        L=L,
        p_temp=temp,
        p_nucleus=nucleus,
    )
    return {k: v.detach() for k, v in probs_dict.items()}


"""
Forms a list of unique prefix strings from q_paths using the provided snippet behavior.
"""
def make_prefixes_from_paths(q_paths: List[List[int]]) -> List[str]:
    q_prefixes: List[str] = []
    for q_path in q_paths:
        for i in range(len(q_path)):
            q_prefix = ",".join(str(x) for x in q_path[: i + 1])
            if q_prefix not in q_prefixes:
                q_prefixes.append(q_prefix)
    return q_prefixes


"""
Builds sliced q_paths for a single (K, L1, L2) instance and a fixed draft path index i.
Returns q_paths (list of K paths) and q_prefixes constructed from them.
"""
def build_sliced_paths_and_prefixes(
    root_token: int,
    draft_tok_i: torch.Tensor,
    branch_tok_i: torch.Tensor,
    K: int,
    L1: int,
    L2: int,
) -> Tuple[List[List[int]], List[str]]:
    prefix = [int(root_token)] + [int(x) for x in draft_tok_i[:L1].tolist()]
    suffix = branch_tok_i[L1, :K, :L2].tolist()
    q_paths = [prefix + [int(x) for x in suffix[k]] for k in range(K)]
    q_prefixes = make_prefixes_from_paths(q_paths)
    return q_paths, q_prefixes


def main():
    args = parse_args()
    set_seed(args.seed)

    rollouts_pt, q_model_name, q_temp, q_nucleus, bt_meta, branched = load_branched_draft_trees(args.draft_trees_pt)
    p_model_name, p_temp, p_nucleus, rollouts = load_rollouts(rollouts_pt)
    roots = build_roots_branched(rollouts, branched)

    tok, p_model, q_model = load_models(p_model_name, q_model_name, device_map="auto", dtype=args.dtype)
    vocab_size = tok.vocab_size

    def get_layers(model):
        return getattr(model, "layers", getattr(getattr(model, "language_model", None), "layers", None))

    for layer in get_layers(p_model.model):
        layer.attention_type = "full_attention"
    for layer in get_layers(q_model.model):
        layer.attention_type = "full_attention"

    draft_depth = int(bt_meta["draft_depth"])
    branch_depth = int(bt_meta["branch_depth"])
    branch_paths = int(bt_meta["branch_paths"])
    draft_paths_count = int(bt_meta["draft_paths"])
    max_k = branch_paths

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    input_dict: Dict[str, Dict[str, torch.Tensor]] = {}
    nss_output_dict: Dict[str, torch.Tensor] = {}
    naive_output_dict: Dict[str, torch.Tensor] = {}
    spectr_output_dict: Dict[str, torch.Tensor] = {}
    specinfer_output_dict: Dict[str, torch.Tensor] = {}
    khisti_output_dict: Dict[str, torch.Tensor] = {}

    last_ex_id = None
    p_rollout_cache = None
    q_rollout_cache = None

    with torch.inference_mode():
        for root in tqdm(roots, desc="Building input/output"):
            cur_ex_id = root["ex_id"]
            prompt_len, rollout_ids = rollouts[cur_ex_id]
            if cur_ex_id != last_ex_id:
                last_ex_id = cur_ex_id
                p_context = rollout_ids.long().unsqueeze(0).to(p_model.device)
                q_context = rollout_ids.long().unsqueeze(0).to(q_model.device)
                p_rollout_cache = build_cache_for_context(p_model, p_context, chunk=128)
                q_rollout_cache = build_cache_for_context(q_model, q_context, chunk=128)

            key = root["key"]
            ctx_end = int(root["ctx_end"])
            root_token = int(root["root_token"])
            draft_tok = root["draft_tok"]
            branch_tok = root["branch_tok"]

            prev_pos = ctx_end - 2
            cur_pos = ctx_end - 1
            prev_token = int(rollout_ids[prev_pos].item())
            cur_token = int(rollout_ids[cur_pos].item())

            p_feat_prev, p_logits_prev = hidden_and_logits_at_token(p_model, p_rollout_cache, prev_pos, prev_token)
            q_feat_prev, q_logits_prev = hidden_and_logits_at_token(q_model, q_rollout_cache, prev_pos, prev_token)
            q_feat_cur, q_logits_cur = hidden_and_logits_at_token(q_model, q_rollout_cache, cur_pos, cur_token)

            p_prev_probs = probs_with_temp_top_p(p_logits_prev, p_temp, p_nucleus, vocab_size)
            q_prev_probs = probs_with_temp_top_p(q_logits_prev, q_temp, q_nucleus, vocab_size)
            q_cur_probs = probs_with_temp_top_p(q_logits_cur, q_temp, q_nucleus, vocab_size)

            p_ent_prev, q_ent_prev, l1_prev, kl_pq_prev, kl_qp_prev = dist_metrics(p_prev_probs, q_prev_probs)
            q_ent_cur = (-(q_cur_probs.float().clamp_min(1e-12) * q_cur_probs.float().clamp_min(1e-12).log()).sum()).detach().cpu()

            input_dict[key] = {
                "p_feat_prev": p_feat_prev.detach().cpu().float(),
                "q_feat_prev": q_feat_prev.detach().cpu().float(),
                "q_feat_cur": q_feat_cur.detach().cpu().float(),
                "p_ent_prev": p_ent_prev.float(),
                "q_ent_prev": q_ent_prev.float(),
                "q_ent_cur": q_ent_cur.float(),
                "kl_pq_prev": kl_pq_prev.float(),
                "kl_qp_prev": kl_qp_prev.float(),
                "l1_prev": l1_prev.float(),
                "context_len": torch.tensor(int(prompt_len + root["offset"]), dtype=torch.int32),
                "temp": torch.tensor(float(p_temp), dtype=torch.float32),
                "top_p": torch.tensor(float(p_nucleus), dtype=torch.float32),
            }

            root_context_len = int(root["context"].shape[0])
            p_root_cache = create_prefix_cache(p_rollout_cache, root_context_len)
            q_root_cache = create_prefix_cache(q_rollout_cache, root_context_len)

            p_probs_dict = cover_tree_probs(
                p_model,
                p_root_cache,
                root_token,
                draft_tok,
                branch_tok,
                vocab_size,
                p_temp,
                p_nucleus,
                max_k,
                draft_depth,
                branch_depth,
            )
            q_probs_dict = cover_tree_probs(
                q_model,
                q_root_cache,
                root_token,
                draft_tok,
                branch_tok,
                vocab_size,
                q_temp,
                q_nucleus,
                max_k,
                draft_depth,
                branch_depth,
            )

            nss_out = torch.empty((max_k, draft_depth + 1, branch_depth + 1, draft_paths_count), dtype=torch.float32) if args.nss else None
            naive_out = torch.empty((max_k, draft_depth + 1, branch_depth + 1, draft_paths_count), dtype=torch.float32) if args.naive else None
            spectr_out = torch.empty((max_k, draft_depth + 1, branch_depth + 1, draft_paths_count), dtype=torch.float32) if args.spectr else None
            specinfer_out = torch.empty((max_k, draft_depth + 1, branch_depth + 1, draft_paths_count), dtype=torch.float32) if args.specinfer else None
            khisti_out = torch.empty((max_k, draft_depth + 1, branch_depth + 1, draft_paths_count), dtype=torch.float32) if args.khisti else None

            for k_idx in range(max_k):
                K = k_idx + 1
                for L1 in range(draft_depth + 1):
                    for i_path in range(draft_paths_count):
                        q_paths_s, q_prefixes_s = build_sliced_paths_and_prefixes(
                            root_token,
                            draft_tok[i_path],
                            branch_tok[i_path],
                            K,
                            L1,
                            branch_depth,   # max L2 only
                        )
                        tree = TreeVerifier(q_paths_s, q_prefixes_s, q_probs_dict, p_probs_dict)
                        
                        if args.nss:
                            vals = tree.expected_nss_depths(branch_depth)
                            nss_out[k_idx, L1, :, i_path] = torch.tensor(vals, dtype=torch.float32)

                        if args.naive:
                            vals = tree.expected_naive_depths(branch_depth)
                            naive_out[k_idx, L1, :, i_path] = torch.tensor(vals, dtype=torch.float32)

                        if args.spectr:
                            vals = tree.expected_spectr_depths(branch_depth)
                            spectr_out[k_idx, L1, :, i_path] = torch.tensor(vals, dtype=torch.float32)
                        
                        if args.specinfer:
                            vals = tree.expected_specinfer_depths(branch_depth)
                            specinfer_out[k_idx, L1, :, i_path] = torch.tensor(vals, dtype=torch.float32)

                        if args.khisti:
                            vals = tree.expected_khisti_depths(branch_depth)
                            khisti_out[k_idx, L1, :, i_path] = torch.tensor(vals, dtype=torch.float32)

            Node.naive_cache.clear()
            Node.spectr_cache.clear()
            Node.specinfer_cache.clear()
            Node.khisti_cache.clear()

            nss_output_dict[key] = nss_out
            naive_output_dict[key] = naive_out
            spectr_output_dict[key] = spectr_out
            specinfer_output_dict[key] = specinfer_out
            khisti_output_dict[key] = khisti_out

    torch.cuda.synchronize()
    _ = time.perf_counter() - t0

    all_methods = ["nss", "naive", "specinfer", "spectr", "khisti"]
    method_desc = "_".join(k for k in all_methods if getattr(args, k))
    out_path = args.out or os.path.join(
        os.path.dirname(args.draft_trees_pt),
        f"{os.path.splitext(os.path.basename(args.draft_trees_pt))[0]}_io_{method_desc}.pt",
    )
    torch.save({
        "input": input_dict, 
        "nss": nss_output_dict, 
        "naive": naive_output_dict, 
        "spectr": spectr_output_dict, 
        "specinfer": specinfer_output_dict,
        "khisti": khisti_output_dict,
    }, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
