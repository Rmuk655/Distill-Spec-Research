"""Author: Krishnan R, IIT Hyderabad — deterministic first-token-rejection regression test
First-draft-token rejection in main.speculative_decoding_iter (naive, K=1, L=2).

Setup: cached context [10, 11], pending token 7, draft path [7, 3, 4].
At the root the draft is one-hot on 3 and the target is one-hot on 9, so the
first draft token (3) is rejected. The acceptance draw is fixed to 0.5.

Expected: no draft token accepted (tau = 0), context -> [10, 11, 7],
new pending token = 9, both caches have length 3 (3 and 4 discarded,
9 not yet cached).
"""
import os
import sys
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import verifiers  # noqa: F401  (puts verifiers/ on sys.path)
import main

V = 16


def onehot(i):
    v = torch.zeros(V)
    v[i] = 1.0
    return v


def make_cache(tokens):
    # One row per token; the value equals the token id so we can see what survives.
    t = torch.tensor(tokens, dtype=torch.float32).view(1, 1, len(tokens), 1)
    return SimpleNamespace(layers=[SimpleNamespace(keys=t.clone(), values=t.clone())])


def cache_rows(cache):
    return [int(x) for x in cache.layers[0].keys.flatten().tolist()]


def cache_values(cache):
    return [int(x) for x in cache.layers[0].values.flatten().tolist()]

def test_first_draft_token_rejected():
    q_paths = [[7, 3, 4]]
    q_prefixes = ["7", "7,3", "7,3,4"]
    q_tokens = torch.tensor([[7, 3, 4]])
    q_probs = {"7": onehot(3), "7,3": onehot(4), "7,3,4": onehot(5)}
    p_probs = {"7": onehot(9), "7,3": onehot(4), "7,3,4": onehot(5)}
    p_cache = make_cache([10, 11, 7, 3, 4])
    q_cache = make_cache([10, 11, 7, 3, 4])

    context_cached = torch.tensor([[10, 11]])
    context_pending = torch.tensor([[7]])

    with mock.patch.object(main, "iid_draft", return_value=(q_paths, q_cache, q_probs)), \
         mock.patch.object(main, "target_tree_pass",
                           return_value=(q_prefixes, q_tokens, p_cache, p_probs)), \
         mock.patch("random.random", return_value=0.5):
        p_cache, q_cache, ctx, pending = main.speculative_decoding_iter(
            SimpleNamespace(), SimpleNamespace(), p_cache, q_cache,
            context_cached, context_pending, "naive", K=1, L=2,
        )

    assert ctx.tolist() == [[10, 11, 7]]
    assert pending.item() == 9
    assert cache_rows(p_cache) == [10, 11, 7]
    assert cache_rows(q_cache) == [10, 11, 7]
    assert cache_values(p_cache) == [10, 11, 7]
    assert cache_values(q_cache) == [10, 11, 7]


if __name__ == "__main__":
    test_first_draft_token_rejected()
    print("PASS")
