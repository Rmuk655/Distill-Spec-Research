import numpy as np
import torch
import torch.nn.functional as F
import random
from scipy.optimize import linprog
from typing import List, Tuple, Dict, Mapping

"""
Solve the linear program defined by Khisti in Appendix I of https://arxiv.org/abs/2410.18234.
This is an approximation to the OTLP (optimal transport linear program) for 2 independent (but not necessarily i.i.d.) drafts.
In particular, the approximation is s-truncation: only maintain O(s^2) variables and set the rest to zero or one.
    - p: Target probability distribution, tensor on GPU
    - q1: First draft token probability distribution, tensor on GPU
    - q2: Second draft token probability distribution, tensor on GPU 
    - s: Truncation threshold, cannot exceed 15 for LP efficiency reasons
Returns three relevant quantities:
    - q_imp: Importance-sampled distribution from the LP solution, tensor on GPU
    - w: LP variables solution of shape (s, s), numpy array on CPU
    - rank: Ranks of vocab tokens: 0,1,.. in descending order of p - q1 * q2
"""
def khisti_lp_solver(
    p: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    s: int = 5,
) -> Tuple[torch.Tensor, np.array, torch.Tensor]:
    v = int(p.numel())
    assert (s >= 1) and (s <= 15), "Max Khisti LP-truncation threshold must lie in {1,...,15}"
    device = p.device

    ### Expensive Vocab Sorting and Computation on GPU

    # Sort the vocabulary by p - q1 * q2 in descending order, and assign ranks 0,1,... to unsorted vocabulary tokens in this order.
    vocab_sort_t = torch.argsort(p - q1 * q2, descending=True)
    omega1_t = vocab_sort_t[:s]
    rank = torch.empty(v, dtype=torch.long, device=device)
    rank[vocab_sort_t] = torch.arange(v, device=device)

    # Compute tail mass of q1 and q2: tail1[i] = sum_{rank[j] > rank[i]} q1[j], likewise for q1.
    q1_sort = q1[vocab_sort_t]
    q2_sort = q2[vocab_sort_t]
    q1_tail = torch.cumsum(q1_sort.flip(0), dim=0).flip(0) - q1_sort
    q2_tail = torch.cumsum(q2_sort.flip(0), dim=0).flip(0) - q2_sort
    q1_tail = q1_tail[rank]
    q2_tail = q2_tail[rank]

    # Full-vocab operations: compute importance-sampled distribution q_I for Omega_2, and mass of q1 and q2 over Omega_2.
    # Note: q_I is only accurate over indices in Omega_2 now, it will be rewritten over Omega_1 after LP solving.
    q_imp = q1 * q2 + q1 * q2_tail + q1_tail * q2
    q1_sum_omega2 = float((q1.sum() - q1[omega1_t].sum()).detach().cpu().item())
    q2_sum_omega2 = float((q2.sum() - q2[omega1_t].sum()).detach().cpu().item())

    ### Solving s-Vocab-Truncated LP on CPU

    # Move top-s entries onto CPU, not full vocab.
    p_top = p[omega1_t].detach().double().cpu().numpy()
    q1_top = q1[omega1_t].detach().double().cpu().numpy()
    q2_top = q2[omega1_t].detach().double().cpu().numpy()

    # LP variables and *minimization* objective weights c (linprog minimizes):
    #     w[i,j] for 0 <= i,j < s     weight 0
    #     z[i] for 0 <= i < s         weight -1
    num_w = s * s
    c = np.r_[np.zeros(num_w, dtype=np.float64), -np.ones(s, dtype=np.float64)]

    # Initialize sparse matrix format for LP.
    A_ub, b_ub = np.zeros((2 * s, num_w + s)), np.zeros((2 * s))

    # Add constraints z[i] <= p[Omega_1[i]] for 0 <= i < s.
    for i in range(s):
        A_ub[i, num_w + i] += 1.0
        b_ub[i] += p_top[i]

    # Add constraints z[i] <= q_I(Omega_1[i]) for 0 <= i < s. Note that q_I contains variables w[i,j].
    for i in range(s):
        qi1 = q1_top[i]
        qi2 = q2_top[i]
        for j in range(s):
            qj1 = q1_top[j]
            qj2 = q2_top[j]
            if j < i:
                A_ub[s + i, s * i + j] += qi1 * qj2
                A_ub[s + i, s * j + i] += qj1 * qi2
                b_ub[s + i] += qi1 * qj2 + qj1 * qi2
            elif j > i:
                A_ub[s + i, s * i + j] -= qi1 * qj2
                A_ub[s + i, s * j + i] -= qj1 * qi2
        A_ub[s + i, num_w + i] += 1.0
        b_ub[s + i] += qi1 * qi2 + qi1 * q2_sum_omega2 + q1_sum_omega2 * qi2

    # Solve linear program with all variables constrained to [0, 1].
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[0, 1], method="highs")
    except Exception:
        return q1, np.full((s, s), 0.5, dtype=np.float64), rank
    if (not res.success) or (not np.isfinite(res.x).all()):
        return q1, np.full((s, s), 0.5, dtype=np.float64), rank

    ### Update results on GPU from LP solution

    # Compute importance-sampled distribution q_I for Omega_1 (Omega_2 already done).
    q_imp_omega1 = res.x[-s:] + b_ub[-s:] - A_ub[-s:] @ res.x
    q_imp[omega1_t] = torch.from_numpy(q_imp_omega1).to(device=device, dtype=q_imp.dtype)
    q_imp = F.relu(q_imp)
    q_imp_sum = q_imp.sum()
    q_imp = q_imp / q_imp_sum if q_imp_sum > 0 else q1
    return q_imp, res.x[:num_w].reshape((s, s)), rank


"""
Select one of two draft tokens based on the methodology by Khisti in Appendix I of https://arxiv.org/abs/2410.18234.
This must be called after khisti_lp_solver, and uses the following quantities:
    - w: LP variables solution of shape (s, s), numpy array on CPU
    - rank: Ranks of vocab tokens: 0,1,.. in descending order of p - q1 * q2
    - x1: Sampled first draft token
    - x2: Sampled second draft token
Returns either x1 or x2 based on the importance-sampled distribution.
"""
def khisti_pair_select(
    w: np.array,
    rank: torch.Tensor,
    x1: int,
    x2: int,
) -> int:
    if x1 == x2:
        return x1
    s, _ = w.shape

    # If both tokens are in the truncated alphabet, use w weights to randomly select one.
    r1, r2 = int(rank[x1].item()), int(rank[x2].item())
    if r1 < s and r2 < s:
        prob_x1 = w[r1, r2] if r1 < r2 else 1.0 - w[r1, r2]
        return x1 if np.random.rand() < prob_x1 else x2

    # Otherwise, simply choose the token with lower rank.
    return x1 if r1 < r2 else x2
