import random
import torch
import json
import torch.nn.functional as F
from typing import List, Tuple, Dict, Callable
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from util import *
from khisti import *


"""
Represents a single node in the draft tree. Nodes have:
    - idx: numerical index representing position in tree, e.g. 0 for root node
    - rep: string representation of the context, e.g. "1,2,56,127" for appended context [1, 2, 56, 127]
    - token: last token in the context, e.g. 127 for the above rep example
    - depth: depth in the tree, setting 0 at the root node
    - parent: Node object for parent in tree, not initialized in constructor
    - children: list of Nodes for children in tree, not initialized in constructor
Importantly, children is allowed to have duplicate Nodes in the case that drafted paths overlap.
All methods are OTLP solvers: these take in p (target) and q (draft) distributions at the node as 1D tensors.
Using child token data, which is assumed to be sampled i.i.d. from q, these OTLP solvers then return a token sampled from p.
The current OTLP solvers are:
    - nss: used in NSS from the SpecInfer paper
    - naive: used in the original Speculative Decoding paper
    - spectr: used as the K-SEQ algorithm in the SpecTr paper
    - specinfer: used in SpecInfer from the SpecInfer paper
    - khisti: used for approximate OTLP solver from the Canonical Decomposition paper (Khisti)
    - max: chooses the OTLP solver with highest acceptance rate
To implement the max solver, we also include acceptance rate computation functions for all other solvers.
NOTE: Khisti does not have an efficient exact acceptance rate formula unfortunately, so we use a lower bound.
"""
class Node:
    # Shared across all Nodes, for use in training generation with a high volume of OTLP branch calls.
    naive_cache = {}
    spectr_cache = {}
    specinfer_cache = {}
    khisti_cache = {}

    def __init__(
        self,
        idx: int,
        rep: str,
        token: int,
        depth: int,            
    ):
        self.idx = idx
        self.rep = rep
        self.token = token
        self.depth = depth
        self.parent : Node = None
        self.children : List[Node] = []




    """
    OTLP solver for NSS (Naive speculative sampling). From https://arxiv.org/pdf/2305.09781.
    This samples directly from the target distribution p, and ignores q.
    """
    def nss_otlp_solver(self, p: torch.Tensor, q: torch.Tensor) -> int:
        return torch.multinomial(p, num_samples=1).item()

    def nss_otlp_accept(self, p: torch.Tensor, q: torch.Tensor, k: int) -> float:
        accept = (p * (1 - (1 - q) ** k)).sum()
        return accept.item()

    def nss_otlp_branch(self, p: torch.Tensor, q: torch.Tensor) -> int:
        probs = {}
        for child in self.children:
            t = int(child.token)
            probs[t] = float(p[t]) if t < len(p) else 0.0
        return probs




    """
    OTLP solver for the original Speculative Decoding paper. From https://arxiv.org/pdf/2211.17192.
    See lines after "Determine the number of accepted guesses" in Algorithm 1 for implementation details.
    """
    def naive_otlp_solver(self, p: torch.Tensor, q: torch.Tensor, child_token=None) -> int:
        # Defaulting to NSS (sample directly from p) if there are no children.
        k = len(self.children)
        if k == 0:
            return self.nss_otlp_solver(p, q)

        # Use the same token acceptance + residual process as in speculative decoding, for specified child (default first).
        token = self.children[0].token if child_token is None else child_token
        qt = float(q[token]) if token < len(q) else 0.0
        pt = float(p[token]) if token < len(p) else 0.0
        if random.random() * qt <= pt:
            return token
        p_res = F.relu(p - q)
        if p_res.sum() <= 0:
            p_res += 1e-4
        p_res /= p_res.sum(dim=-1)
        return torch.multinomial(p_res, num_samples=1).item()

    def naive_otlp_accept(self, p: torch.Tensor, q: torch.Tensor, k: int) -> float:
        # Standard single-draft acceptance rate for the first token.
        accept = torch.minimum(p, q).sum()

        # Add correction term for sampling one of the remaining k - 1 draft tokens from the residual.
        if k > 1:  
            p_res = F.relu(p - q)
            accept += (p_res * (1 - (1 - q) ** (k - 1))).sum()
        return accept.item()

    def naive_otlp_branch(self, p: torch.Tensor, q: torch.Tensor, child_token=None) -> Dict[int, float]:
        probs = {}
        if len(self.children) == 0:
            return probs

        # Get conditional acceptance for first token.
        token = self.children[0].token if child_token is None else child_token
        qt = float(q[token]) if token < len(q) else 0.0
        pt = float(p[token]) if token < len(p) else 0.0
        if qt <= 0.0:
            cond_accept = 1.0
        else:
            cond_accept = min(1.0, pt / qt)

        # Use a node-shared cache for residual computation.
        cache_key = (id(p), id(q))
        p_res = Node.naive_cache.get(cache_key)
        if p_res is None:
            p_res = F.relu(p - q)
            if float(p_res.sum()) <= 0.0:
                p_res = p_res + 1e-4
            p_res = p_res / p_res.sum(dim=-1)
            Node.naive_cache[cache_key] = p_res

        # Mix conditional acceptance and residual sampling probabilities.
        out = (1.0 - cond_accept) * p_res
        if token < len(out):
            out[token] += cond_accept
        for child in self.children:
            t = int(child.token)
            probs[t] = float(out[t]) if t < len(out) else 0.0
        return probs




    """
    OTLP solver for SpecTr. From https://arxiv.org/pdf/2310.15141.
    See Algorithm 2 for implementation details, and Theorem 1/Appendix C.1 for divison factor details.
    """
    def spectr_binary_search(self, p: torch.Tensor, q: torch.Tensor, k: int, tol: float) -> float:
        rho_low, rho_high = 1.0, float(k)
        while (rho_high - rho_low) > tol:
            rho = (rho_low + rho_high) / 2
            beta = float(torch.minimum(p / rho, q).sum())
            p_acc = 1 - (1 - beta) ** k
            if p_acc >= rho * beta:
                rho_low = rho
            else:
                rho_high = rho
        return rho_high

    def spectr_binary_search_approx(self, p: torch.Tensor, q: torch.Tensor, k: int, iters: int = 4) -> float:
        rho_low, rho_high = 1.0, float(k)
        for _ in range(iters):
            rho = (rho_low + rho_high) / 2
            beta = float(torch.minimum(p / rho, q).sum())
            p_acc = 1 - (1 - beta) ** k
            if p_acc >= rho * beta:
                rho_low = rho
            else:
                rho_high = rho
        return rho_high

    def spectr_otlp_solver(self, p: torch.Tensor, q: torch.Tensor, tol=1e-4) -> int:
        # Defaulting to NSS if there are no children, or Naive if there is one child.
        k = len(self.children)
        if k == 0:
            return self.nss_otlp_solver(p, q)
        elif k == 1:
            return self.naive_otlp_solver(p, q)

        # Compute divison factor rho needed for K-SEQ (round up), by using binary search up to a certain tolerance.
        rho = self.spectr_binary_search(p, q, k, tol)
        beta = torch.minimum(p / rho, q).sum()
        p_acc = 1 - (1 - beta) ** k

        # Perform K-SEQ with the computed divison factor.
        child_tokens = [child.token for child in self.children]
        for token in child_tokens:
            if rho * random.random() * q[token] <= p[token]:
                return token
        p_acc_beta_ratio = p_acc / torch.clamp(beta, min=1e-6) if beta > 0 else rho
        p_res = F.relu(p - torch.minimum(p / rho, q) * p_acc_beta_ratio)
        if p_res.sum() <= 0:
            p_res += 1e-4
        p_res = p_res / p_res.sum()
        return torch.multinomial(p_res, num_samples=1).item()

    def spectr_otlp_accept(self, p: torch.Tensor, q: torch.Tensor, k: int, tol=1e-4) -> float:
        # Compute division factor and the probability that one of the draft tokens is accepted.
        rho = self.spectr_binary_search(p, q, k, tol)
        beta = torch.minimum(p / rho, q).sum()
        if beta >= 1.0:
            return 1.0
        p_acc = 1 - (1 - beta) ** k

        # Compute the residual distribution.
        p_acc_beta_ratio = p_acc / torch.clamp(beta, min=1e-6) if beta > 0 else rho
        p_res = F.relu(p - torch.minimum(p / rho, q) * p_acc_beta_ratio)
        if p_res.sum() <= 0:
            p_res += 1e-4
        p_res = p_res / p_res.sum(dim = -1)
        
        # Add correction term for sampling a draft token from the residual, conditioned on rejecting all draft tokens.
        r = F.relu(q - p / rho) / torch.clamp(1 - beta, min=1e-6)
        accept = p_acc + (1 - p_acc) * (p_res * (1 - (1 - r) ** k)).sum()
        return accept.item()
    
    def spectr_otlp_branch(self, p: torch.Tensor, q: torch.Tensor) -> dict:
        k = len(self.children)
        if k == 0:
            return {}
        if k == 1:
            return self.naive_otlp_branch(p, q)

        # Use a node-shared cache for full-vocab operations.
        cache_key = (id(p), id(q), k)
        hit = Node.spectr_cache.get(cache_key)
        if hit is None:
            # Compute divison factor rho needed for K-SEQ (round up), by using binary search up to a certain tolerance.
            rho = self.spectr_binary_search_approx(p, q, k)
            beta = float(torch.minimum(p / rho, q).sum())
            p_acc = 1 - (1 - beta) ** k

            # Compute residual distribution.
            p_acc_beta_ratio = (p_acc / max(beta, 1e-6)) if beta > 0.0 else float(rho)
            p_res = F.relu(p - torch.minimum(p / rho, q) * p_acc_beta_ratio)
            if float(p_res.sum()) <= 0.0:
                p_res = p_res + 1e-4
            p_res = p_res / p_res.sum()

            # Store entry in cache.
            hit = (float(rho), beta, float(p_acc), p_res)
            Node.spectr_cache[cache_key] = hit
        rho, beta_f, p_acc, p_res = hit

        # Per-occurrence accept probability using rho * U * q[t] <= p[t].
        a = []
        child_tokens = [int(ch.token) for ch in self.children]
        pN, qN = int(p.numel()), int(q.numel())
        for t in child_tokens:
            if t < 0 or t >= pN or t >= qN:
                a.append(0.0)
                continue
            qt = float(q[t])
            a.append(1.0 if qt <= 0.0 else min(1.0, float(p[t]) / (rho * qt)))

        # Residual distribution used only if all k accept tests fail.
        probs, pref_reject = {}, 1.0
        for t, ai in zip(child_tokens, a):
            probs[t] = probs.get(t, 0.0) + pref_reject * ai
            pref_reject *= (1.0 - ai)

        # If all tests reject, add residual mass on each distinct child token.
        for t in set(child_tokens):
            if 0 <= t < pN and t < qN:
                probs[t] = probs.get(t, 0.0) + pref_reject * float(p_res[t])
            else:
                probs[t] = probs.get(t, 0.0)
        return probs




    """
    OTLP solver for SpecInfer. From https://arxiv.org/pdf/2305.09781.
    See lines 27 to 40 of Algorithm 2 for implementation details.
    """
    def specinfer_otlp_solver(self, p: torch.Tensor, q: torch.Tensor) -> int:
        # Defaulting to NSS (sample directly from p) if there are no children.
        k = len(self.children)
        if k == 0:
            return self.nss_otlp_solver(p, q)
        
        # Iteratively perform uniform child selection, either accepting it or removing it from child nodes.
        child_tokens = [child.token for child in self.children]
        while child_tokens != []:
            token = random.choice(child_tokens)
            if random.random() * q[token] <= p[token]:                  # Acceptance: return child node.
                return token
            else:                                                       # Rejection: update target distribution and remove child node.
                p = F.relu(p - q)
                if p.sum() <= 0:
                    p += 1e-4                               
                p /= p.sum(dim=-1)                              
                child_tokens.remove(token)
        return torch.multinomial(p, num_samples=1).item()

    def specinfer_otlp_accept(self, p: torch.Tensor, q: torch.Tensor, k: int) -> float:
        reject = 1.0
        miss = torch.ones_like(p)

        # Iteratively update the residual distribution p and rejection probability.
        for _ in range(k):
            r = 1 - torch.minimum(p, q).sum()
            reject *= r
            miss *= 1 - F.relu(q - p) / torch.clamp(r, min=1e-6)
            p = F.relu(p - q)
            if p.sum() <= 0:
                p += 1e-4
            p /= p.sum(dim=-1)

        # Add correction term for sampling a draft token from the final residual.
        accept = (1 - reject) + reject * (p * (1 - miss)).sum()
        return accept.item()

    def specinfer_otlp_branch(self, p: torch.Tensor, q: torch.Tensor) -> Dict[int, float]:
        k = len(self.children)
        if k == 0:
            return {}
        if k == 1:
            return self.naive_otlp_branch(p, q)

        # Use the cache if available.
        child_tokens = [int(ch.token) for ch in self.children]
        cache_key = (id(p), id(q), tuple(sorted(child_tokens)))
        cache_hit = Node.specinfer_cache.get(cache_key)
        if cache_hit is not None:
            return cache_hit

        # Compute mapping of distinct child tokens to indices.
        unique_tokens = sorted(set(child_tokens))
        token_to_u = {t: i for i, t in enumerate(unique_tokens)}
        vocab_sz = len(p)

        # Compute rejection and acceptance probability for each child token at each rejection step.
        res = p.clone()
        p_child = []
        accept_child = []
        for reject_step in range(k):
            p_vals = [float(res[t]) if 0 <= t < vocab_sz else 0.0 for t in unique_tokens]
            p_child.append(p_vals)
            a_vals = []
            for t, pt in zip(unique_tokens, p_vals):
                if not (0 <= t < vocab_sz):
                    a_vals.append(0.0)
                    continue
                qt = float(q[t])
                a_vals.append(1.0 if qt <= 0.0 else min(1.0, pt / qt))
            accept_child.append(a_vals)
            res = F.relu(res - q)
            if float(res.sum()) <= 0.0:
                res = res + 1e-4
            res = res / res.sum()

        # After all children rejected, sampling is from residual, and branch only needs mass on child tokens.
        p_child.append([float(res[t]) if 0 <= t < vocab_sz else 0.0 for t in unique_tokens])

        # DP over remaining occurrences with a bitmask, since SpecInfer picks uniformly among remaining occurrences.
        occ_u = [token_to_u[t] for t in child_tokens]
        U = len(unique_tokens)
        dp = [None] * (1 << k)
        dp[0] = p_child[k][:]
        for mask in range(1, 1 << k):
            remaining = mask.bit_count()
            reject_step = k - remaining
            inv_remaining = 1.0 / remaining
            out = [0.0] * U
            mm = mask
            while mm:
                lsb = mm & -mm
                i = lsb.bit_length() - 1
                mm ^= lsb
                ui = occ_u[i]
                a = accept_child[reject_step][ui]
                out[ui] += inv_remaining * a
                if a < 1.0:
                    child = dp[mask ^ (1 << i)]
                    w = inv_remaining * (1.0 - a)
                    for u in range(U):
                        out[u] += w * child[u]
            dp[mask] = out
        root = dp[(1 << k) - 1]
        probs = {t: root[token_to_u[t]] for t in unique_tokens}
        Node.specinfer_cache[cache_key] = probs
        return probs






    """
    Khisti importance sampling-based selector. From https://arxiv.org/pdf/2410.18234.
    See Algorithm 2 for implementation details.
    """
    def khisti_otlp_solver(self, p: torch.Tensor, q: torch.Tensor, s_threshold: int = 5) -> int:
        # Defaulting to NSS if there are no children, or Naive if there is one child.
        k = len(self.children)
        if k == 0:
            return self.nss_otlp_solver(p, q)
        elif k == 1:
            return self.naive_otlp_solver(p, q)

        # Perform tournament selection, using the pairwise LP-based importance sampling method.
        child_tokens = [child.token for child in self.children]
        q_imp = q
        x1, x2 = child_tokens[0], child_tokens[1]
        for i in range(len(child_tokens) - 1):
            q_imp, w, rank = khisti_lp_solver(p, q_imp, q, s=s_threshold)
            x1 = khisti_pair_select(w, rank, x1, x2)
            if i < len(child_tokens) - 2:
                x2 = child_tokens[i + 2]

        # Run naive speculative sampling on the final importance-sampled draft q_imp and draft token x1.
        return self.naive_otlp_solver(p, q_imp, child_token=x1)

    def khisti_otlp_accept_lower_bound(self, p: torch.Tensor, q: torch.Tensor, k: int, s_threshold: int = 5) -> float:
        # Update the importance sampling distribution using the LP solver.
        q_imp = q
        for i in range(k - 1):
            q_imp, w, rank = khisti_lp_solver(p, q_imp, q, s=s_threshold)

        # Compute the single-draft acceptance rate from draft q_imp and target p under naive speculative sampling.
        accept = torch.minimum(p, q_imp).sum()
        return accept.item()

    def khisti_otlp_branch(self, p: torch.Tensor, q: torch.Tensor, s_threshold: int = 5) -> Dict[int, float]:
        k = len(self.children)
        if k == 0:
            return {}
        if k == 1:
            return self.naive_otlp_branch(p, q)

        # Check if there is a cache hit.
        child_tokens = [int(ch.token) for ch in self.children]
        cache_key = (id(p), id(q), tuple(child_tokens), int(s_threshold))
        cache_hit = Node.khisti_cache.get(cache_key)
        if cache_hit is not None:
            return cache_hit

        # Tournament winner distribution over token values (matches khisti_pair_select in expectation).
        v = len(p)
        s = int(s_threshold)
        q_imp = q
        winner_dist: Dict[int, float] = {child_tokens[0]: 1.0}
        for r in range(k - 1):
            challenger = child_tokens[r + 1]
            q_imp, w, rank = khisti_lp_solver(p, q_imp, q, s=s)
            new_dist: Dict[int, float] = {}
            for cur_winner, mass in winner_dist.items():
                if cur_winner == challenger:
                    new_dist[cur_winner] = new_dist.get(cur_winner, 0.0) + mass
                    continue

                # Safe rank handling for OOB tokens.
                if 0 <= cur_winner < v:
                    r1 = int(rank[cur_winner].item())
                else:
                    r1 = v + 1
                if 0 <= challenger < v:
                    r2 = int(rank[challenger].item())
                else:
                    r2 = v + 1

                # Probability that khisti_pair_select(w, rank, cur_winner, challenger) returns cur_winner.
                if r1 < s and r2 < s:
                    prob_keep = float(w[r1, r2]) if r1 < r2 else float(1.0 - w[r1, r2])
                else:
                    prob_keep = 1.0 if r1 < r2 else 0.0
                new_dist[cur_winner] = new_dist.get(cur_winner, 0.0) + mass * prob_keep
                new_dist[challenger] = new_dist.get(challenger, 0.0) + mass * (1.0 - prob_keep)
            winner_dist = new_dist

        # Final step is naive speculative sampling with q_imp and child_token to get tournament winner.
        probs: Dict[int, float] = {}
        for winner, pw in winner_dist.items():
            tmp = self.naive_otlp_branch(p, q_imp, child_token=winner)
            for t, val in tmp.items():
                probs[t] = probs.get(t, 0.0) + pw * float(val)
        Node.khisti_cache[cache_key] = probs
        return probs





    """
    The max-OTLP solver chooses the OTLP solver with the highest acceptance rate with p, q.
    """
    def max_otlp_solver(self, p: torch.Tensor, q: torch.Tensor) -> int:
        # Defaulting to NSS if there are no children, or Naive if there is one child.
        k = len(self.children)
        if k == 0:
            return self.nss_otlp_solver(p, q)
        elif k == 1:
            return self.naive_otlp_solver(p, q)

        # Compute exact acceptance rates for all OTLP solvers, except for a lower bound for Khisti.
        candidates = [
            (self.nss_otlp_solver, self.nss_otlp_accept(p, q, k)),
            (self.naive_otlp_solver, self.naive_otlp_accept(p, q, k)),
            (self.spectr_otlp_solver, self.spectr_otlp_accept(p, q, k)),
            (self.specinfer_otlp_solver, self.specinfer_otlp_accept(p, q, k)),
            (self.khisti_otlp_solver, self.khisti_otlp_accept_lower_bound(p, q, k)),
        ]

        # Run the best OTLP solver.
        best_solver = max(candidates, key=lambda x: x[1])[0]
        return best_solver(p, q)
