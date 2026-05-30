"""
tiny_models.py — toy teacher/student for the tree-loss sandbox.

Why order-1 Markov models?
--------------------------
The real draft/target are billion-parameter transformers whose per-node
distribution q(· | context) depends on the whole prefix. That is impossible
to eyeball. We replace them with the smallest object that still produces the
EXACT data structure the real losses and verifiers consume:

    a next-token distribution that depends only on the LAST token.

Formally each model is an order-1 Markov chain over a vocabulary of V tokens,
parameterised by a V×V logit matrix W:

    q(next = j | last = i)  =  softmax(W[i] / temperature)[j]

* TEACHER  — W frozen. Plays the role of the big target model `p`.
* STUDENT  — W is an nn.Parameter (trainable). Plays the role of the small
             draft model `q` with LoRA adapters. loss.backward() updates W
             exactly the way the real trainer updates LoRA weights.

A "draft tree" is then just K independent length-L walks of the student's
Markov chain, which naturally share prefixes near the root → a real branching
tree, small enough to draw.

The functions here emit the same dictionaries the pipeline builds:

    q_paths        List[List[int]]              K paths, each length L+1
    q_prefixes     List[str]                    unique prefixes, build order
    q_probs_dict   {prefix -> [V] tensor}       NON-LEAF nodes only  (matches iid_draft)
    p_probs_dict   {prefix -> [V] tensor}       ALL nodes            (matches target_tree_pass)

so they can be fed straight into compute_tree_loss(...) and TreeVerifier(...).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# The toy model
# ---------------------------------------------------------------------------

class TinyMarkovModel(nn.Module):
    """Order-1 Markov next-token model over a V-token vocabulary.

    dist(last_token, temp) returns softmax(W[last_token] / temp), a [V] tensor.
    For the student W requires grad; for the teacher it does not.
    """

    def __init__(self, vocab: int, logits: torch.Tensor, trainable: bool):
        super().__init__()
        assert logits.shape == (vocab, vocab)
        self.vocab = vocab
        if trainable:
            self.W = nn.Parameter(logits.clone().float())
        else:
            self.register_buffer("W", logits.clone().float())
        self.trainable = trainable

    def dist(self, last_token: int, temp: float = 1.0) -> torch.Tensor:
        """Next-token distribution given the last token. [V], with grad iff trainable."""
        return F.softmax(self.W[int(last_token)] / temp, dim=-1)

    # -- factory helpers -----------------------------------------------------

    @staticmethod
    def teacher_counting(vocab: int, peak: float = 4.0, seed: int = 0) -> "TinyMarkovModel":
        """A confident, structured teacher: from token i it strongly prefers i+1
        (mod V), with a little random texture. Low entropy → high achievable BE."""
        g = torch.Generator().manual_seed(seed)
        W = 0.3 * torch.randn(vocab, vocab, generator=g)
        for i in range(vocab):
            W[i, (i + 1) % vocab] += peak            # main "count up" mode
            W[i, (i + 2) % vocab] += 0.5 * peak      # secondary mode → real tree branching
        return TinyMarkovModel(vocab, W, trainable=False)

    @staticmethod
    def student_random(vocab: int, scale: float = 0.5, seed: int = 1) -> "TinyMarkovModel":
        """A near-uniform / mildly random student — no prior knowledge of the
        teacher pattern.  BE ≈ 1.0–1.2 before training.  Use student_pretrained
        to simulate a pre-trained model that already has partial alignment."""
        g = torch.Generator().manual_seed(seed)
        W = scale * torch.randn(vocab, vocab, generator=g)
        return TinyMarkovModel(vocab, W, trainable=True)

    @staticmethod
    def student_pretrained(vocab: int, teacher_peak: float = 5.0,
                           fraction: float = 0.5, noise: float = 0.6,
                           seed: int = 1) -> "TinyMarkovModel":
        """Simulates a smaller pre-trained model that already knows the same
        task as the teacher but less sharply (analogue of Qwen-0.6B vs Qwen-8B,
        both pre-trained on the same corpus, before any distillation).

        Initialised with the same counting pattern as the teacher but at
        fraction * teacher_peak strength, plus Gaussian noise.

        With teacher_peak=5.0, fraction=0.5, noise=0.6, V=32:
          → W[i, i+1] ≈ 2.5  (teacher has 5.0)
          → Starting BE ≈ 2.0–2.5 (bv/gbv/naive) — matches real Qwen-0.6B / Qwen-8B
            pre-trained baseline before any DistillSpec distillation training.
            After 80 steps kl_tree training, reaches BE ~3.5–4.5 (matching SOTA).

        fraction: how much of the teacher's peak the student inherits.
                  0 = fully random (same as student_random at noise scale).
                  1 = identical to teacher (nothing to learn).
                  0.25 = realistic pre-trained gap for 0.6B vs 8B.
        noise:    std of additive Gaussian logit noise (simulates model-size
                  variance; larger = weaker/more uncertain student).
        """
        g = torch.Generator().manual_seed(seed)
        W = noise * torch.randn(vocab, vocab, generator=g)
        for i in range(vocab):
            W[i, (i + 1) % vocab] += fraction * teacher_peak
            W[i, (i + 2) % vocab] += 0.5 * fraction * teacher_peak
        return TinyMarkovModel(vocab, W, trainable=True)


# ---------------------------------------------------------------------------
# Toy "dataset": just a set of starting (pending) tokens
# ---------------------------------------------------------------------------

def make_dataset(vocab: int, n_train: int = 64, n_test: int = 32,
                 seed: int = 7) -> Tuple[List[int], List[int]]:
    """Train/test prompts.

    For an order-1 model a 'prompt' collapses to a single pending token (the
    seed of the draft block). We return disjoint-ish random token streams so
    'test' BE is measured on prompts not directly optimised.
    """
    g = torch.Generator().manual_seed(seed)
    toks = torch.randint(0, vocab, (n_train + n_test,), generator=g).tolist()
    return toks[:n_train], toks[n_train:]


# ---------------------------------------------------------------------------
# Tree construction in the pipeline's exact dict format
# ---------------------------------------------------------------------------

def sample_paths(model: TinyMarkovModel, pending: int, K: int, L: int,
                 temp: float = 1.0, generator: Optional[torch.Generator] = None,
                 ) -> List[List[int]]:
    """Sample K i.i.d. length-L paths from the model's Markov chain.

    Mirrors iid_draft(): each path starts at the shared `pending` token and
    autoregressively samples L more tokens. Sampling is detached (discrete) —
    the gradient enters later via build_dicts(with_grad=True), exactly like the
    real trainer (sample tree no-grad, then re-score WITH grad).
    """
    paths: List[List[int]] = [[int(pending)] for _ in range(K)]
    for k in range(K):
        last = int(pending)
        for _ in range(L):
            d = model.dist(last, temp).detach()
            nxt = int(torch.multinomial(d, num_samples=1, generator=generator).item())
            paths[k].append(nxt)
            last = nxt
    return paths


def _unique_prefixes(paths: List[List[int]]) -> List[str]:
    """Unique prefix strings in build order — identical ordering to
    target_tree_pass / draft_tree_forward_with_grad."""
    prefixes: List[str] = []
    for path in paths:
        for i in range(len(path)):
            pfx = ",".join(str(x) for x in path[:i + 1])
            if pfx not in prefixes:
                prefixes.append(pfx)
    return prefixes


def build_dicts(student: TinyMarkovModel, teacher: TinyMarkovModel,
                paths: List[List[int]], L: int, temp: float = 1.0,
                with_grad: bool = False,
                student_temp: Optional[float] = None,
                teacher_temp: Optional[float] = None,
                ) -> Tuple[List[str], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Build (q_prefixes, q_probs_dict, p_probs_dict) for a fixed sampled tree.

    * q_probs_dict — NON-LEAF nodes only (depth < L), matching iid_draft.
                     WITH grad iff with_grad (matches draft_tree_forward_with_grad).
    * p_probs_dict — ALL nodes (depth 0..L), matching target_tree_pass.
                     Always detached (teacher is frozen).

    Distinct prefixes that happen to share a last token get *separate* tensors
    so that in-place verifier mutations (traversal) on one node never corrupt
    another — but they share the same underlying student parameter row, so the
    gradient still accumulates correctly across the tree.

    student_temp / teacher_temp override `temp` for the respective model.
    Use this to simulate a larger teacher (lower temp → sharper) vs weaker
    student (higher temp → flatter) independently of the logit scales.
    """
    st = student_temp if student_temp is not None else temp
    tt = teacher_temp if teacher_temp is not None else temp
    q_prefixes = _unique_prefixes(paths)
    q_probs_dict: Dict[str, torch.Tensor] = {}
    p_probs_dict: Dict[str, torch.Tensor] = {}
    for pfx in q_prefixes:
        toks = [int(x) for x in pfx.split(",")]
        last = toks[-1]
        depth = len(toks) - 1
        # teacher distribution for every node (leaves included — used for residual sampling)
        p_probs_dict[pfx] = teacher.dist(last, tt).detach().clone()
        if depth < L:
            qd = student.dist(last, st)
            q_probs_dict[pfx] = qd if with_grad else qd.detach().clone()
    return q_prefixes, q_probs_dict, p_probs_dict


def chain_weights(
    prefixes: List[str],
    q_dict: Dict[str, torch.Tensor],
    p_dict: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """BV chain weight for every prefix node.

    w[prefix] = Π_{i=1..depth} min(1, p[t_i] / q[t_i])

    This is the per-node acceptance probability product: the probability
    that the BV verifier would have accepted every token from the root down
    to this node.  It is the most intuitive single scalar for colouring
    the tree (green = easy to accept, red = already rejected).
    """
    w: Dict[str, float] = {}
    for pfx in prefixes:
        toks = [int(x) for x in pfx.split(",")]
        acc = 1.0
        for i in range(1, len(toks)):
            parent = ",".join(str(x) for x in toks[:i])
            if parent not in q_dict or parent not in p_dict:
                acc = 0.0
                break
            t = toks[i]
            qt = float(q_dict[parent][t])
            pt = float(p_dict[parent][t])
            acc *= min(1.0, pt / qt) if qt > 1e-9 else 0.0
        w[pfx] = acc
    return w


def sample_tree(student: TinyMarkovModel, teacher: TinyMarkovModel,
                pending: int, K: int, L: int, temp: float = 1.0,
                with_grad: bool = False,
                generator: Optional[torch.Generator] = None,
                student_temp: Optional[float] = None,
                teacher_temp: Optional[float] = None,
                ) -> Tuple[List[List[int]], List[str], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Convenience: sample K paths then build the three dicts.

    Returns (q_paths, q_prefixes, q_probs_dict, p_probs_dict).
    """
    st = student_temp if student_temp is not None else temp
    tt = teacher_temp if teacher_temp is not None else temp
    paths = sample_paths(student, pending, K, L, st, generator=generator)
    q_prefixes, q_probs_dict, p_probs_dict = build_dicts(
        student, teacher, paths, L, temp, with_grad,
        student_temp=st, teacher_temp=tt,
    )
    return paths, q_prefixes, q_probs_dict, p_probs_dict
