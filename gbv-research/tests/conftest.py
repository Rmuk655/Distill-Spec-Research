"""
conftest.py — shared pytest fixtures and path setup for all unit tests.

All tests run on CPU with tiny tensors — no GPU, no model downloads.
Import paths are injected via sys.path so the GBV/ and OSD/ source trees
can be imported without any package-level restructuring.
"""
import os
import sys
import json
import tempfile
import pytest
import torch

# ---------------------------------------------------------------------------
# Path setup — inject source directories so test files can import directly
# ---------------------------------------------------------------------------

_HERE          = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT     = os.path.dirname(_HERE)                              # gbv-research/
_SUMMER_ROOT   = os.path.dirname(_REPO_ROOT)                        # 2026 summer/
_GBV_SRC       = os.path.join(_SUMMER_ROOT, "GBV")                  # GBV/ algorithms
_OSD_SRC       = os.path.join(_SUMMER_ROOT, "OSD")                  # OSD/ training
_ORCH_SRC      = os.path.join(_REPO_ROOT, "orchestration")          # pipeline.py

for _p in [_GBV_SRC, _OSD_SRC, _ORCH_SRC, _REPO_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Shared vocabulary / distribution helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 8   # small enough for all tests; large enough to expose OOB bugs

def make_uniform(vocab_size=VOCAB_SIZE):
    """Return a uniform probability tensor (CPU)."""
    return torch.full((vocab_size,), 1.0 / vocab_size)

def make_peaked(peak_token: int, peak_prob: float = 0.6, vocab_size=VOCAB_SIZE):
    """Return a distribution with most mass on peak_token."""
    rest = (1.0 - peak_prob) / (vocab_size - 1)
    p = torch.full((vocab_size,), rest)
    p[peak_token] = peak_prob
    return p

def make_random_probs(vocab_size=VOCAB_SIZE, seed=42):
    """Reproducible random probability distribution."""
    g = torch.Generator()
    g.manual_seed(seed)
    raw = torch.rand(vocab_size, generator=g)
    return raw / raw.sum()


# ---------------------------------------------------------------------------
# Mock DynamicCache (mirrors the .layers[i].keys/.values interface used in GBV)
# ---------------------------------------------------------------------------

class _MockCacheLayer:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        self.keys   = keys
        self.values = values

class MockDynamicCache:
    """
    Minimal stand-in for the Qwen DynamicCache used by GBV/util.py.
    Provides both:
      - .layers[i].keys / .layers[i].values   (used by slice_cache / expand_cache)
      - iteration as list-of-(k,v)-tuples      (used by KV profiling in main.py)
    """
    def __init__(self, batch=1, heads=2, seq_len=5, head_dim=4, n_layers=2):
        self.layers = [
            _MockCacheLayer(
                torch.randn(batch, heads, seq_len, head_dim),
                torch.randn(batch, heads, seq_len, head_dim),
            )
            for _ in range(n_layers)
        ]

    def __iter__(self):
        return iter([(l.keys, l.values) for l in self.layers])


# ---------------------------------------------------------------------------
# Tree fixtures — a tiny draft tree used by verifier tests
#
# Vocab: 0-7   K=3 paths  L=2
# Pending token (root): 0
#   Path 0:  [0, 1, 3]
#   Path 1:  [0, 1, 4]
#   Path 2:  [0, 2, 5]
#
# Tree structure:
#     0
#    / \
#   1   2
#  / \   \
# 3   4   5
# ---------------------------------------------------------------------------

def make_tree_fixture(vocab_size=VOCAB_SIZE, seed=0):
    """
    Return (q_paths, q_prefixes, q_probs_dict, p_probs_dict) for a small
    K=3 L=2 draft tree.  All distributions are on CPU.
    """
    q_paths = [
        [0, 1, 3],
        [0, 1, 4],
        [0, 2, 5],
    ]
    # All distinct path prefixes in DFS path order (same as target_tree_pass output)
    q_prefixes = ["0", "0,1", "0,1,3", "0,1,4", "0,2", "0,2,5"]

    torch.manual_seed(seed)
    def _rand():
        r = torch.rand(vocab_size).abs() + 1e-3
        return (r / r.sum()).float()

    q_probs_dict = {pfx: _rand() for pfx in q_prefixes}
    p_probs_dict = {pfx: _rand() for pfx in q_prefixes}

    return q_paths, q_prefixes, q_probs_dict, p_probs_dict


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tree_fixture():
    return make_tree_fixture()

@pytest.fixture
def mock_cache():
    return MockDynamicCache()

@pytest.fixture
def tmp_jsonl(tmp_path):
    """Write a tiny JSONL file and return its path."""
    path = tmp_path / "test.jsonl"
    records = [
        {"prompt": "What is 2+2?"},
        {"prompt": "Explain Newton's laws."},
        {"prompt": ""},        # empty prompt — should be skipped
        {},                    # missing key — should be skipped
        {"question": "What is gravity?"},  # alternate key
    ]
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(path)
