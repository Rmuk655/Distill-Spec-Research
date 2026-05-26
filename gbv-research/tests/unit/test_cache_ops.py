"""
test_cache_ops.py — unit tests for KV-cache manipulation in GBV/util.py.

Functions under test:
  - slice_cache(cache, batch_idx, token_idx) → cache with sliced dims
  - expand_cache(cache, batch_sz)            → cache with batch dim expanded

Design issue documented here:
  main.py profiling iterates the cache as:
      for _layer in p_cache:
          _k = _layer[0]; _v = _layer[1]
  ...using DynamicCache's tuple-iteration API.  But slice_cache/expand_cache
  use the custom .layers[i].keys / .layers[i].values attribute API.
  These are the SAME cache object, so both APIs must coexist on MockDynamicCache.
  The tests verify both APIs work correctly on the same object.

All tests run on CPU.  No GPU, no model loading.
"""

import sys
import pytest
import torch

from util import slice_cache, expand_cache

# Use conftest fixture
from tests.conftest import MockDynamicCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache(batch=1, heads=2, seq_len=6, head_dim=4, n_layers=2):
    return MockDynamicCache(batch, heads, seq_len, head_dim, n_layers)


# ===========================================================================
# expand_cache
# ===========================================================================

class TestExpandCache:

    def test_batch_dim_expands(self):
        cache = _cache(batch=1, seq_len=5)
        expanded = expand_cache(cache, batch_sz=3)
        for layer in expanded.layers:
            assert layer.keys.shape[0]   == 3, f"keys batch dim should be 3"
            assert layer.values.shape[0] == 3, f"values batch dim should be 3"

    def test_other_dims_unchanged(self):
        heads, seq_len, head_dim = 2, 5, 4
        cache = _cache(batch=1, heads=heads, seq_len=seq_len, head_dim=head_dim)
        expanded = expand_cache(cache, batch_sz=4)
        for layer in expanded.layers:
            assert layer.keys.shape[1]   == heads,    "heads dim changed"
            assert layer.keys.shape[2]   == seq_len,  "seq_len dim changed"
            assert layer.keys.shape[3]   == head_dim, "head_dim changed"

    def test_expand_to_1_is_noop(self):
        cache = _cache(batch=1, seq_len=4)
        original_shape = cache.layers[0].keys.shape
        expanded = expand_cache(cache, batch_sz=1)
        assert expanded.layers[0].keys.shape == original_shape, \
            "expand_cache(K=1) should not change shape"

    def test_expand_then_iterate_as_tuples(self):
        """
        After expand_cache, the cache must still be iterable as (k, v) tuples
        (the API used by main.py profiling).
        """
        cache = _cache(batch=1)
        expanded = expand_cache(cache, batch_sz=3)
        for k, v in expanded:   # tuple iteration
            assert k.shape[0] == 3, "Tuple iteration after expand gives wrong batch dim"
            assert v.shape[0] == 3

    def test_values_are_replicated_not_independent(self):
        """
        expand_cache uses .expand(), meaning all batch slices share storage.
        Values should be identical across batch dimension after expansion.
        """
        cache = _cache(batch=1, seq_len=3)
        # Set deterministic values
        for layer in cache.layers:
            layer.keys   = torch.arange(24.0).reshape(1, 2, 3, 4)
            layer.values = torch.arange(24.0, 48.0).reshape(1, 2, 3, 4)

        expanded = expand_cache(cache, batch_sz=3)
        k = expanded.layers[0].keys
        assert torch.equal(k[0], k[1]), "Expanded batch slices should be identical"
        assert torch.equal(k[0], k[2]), "Expanded batch slices should be identical"


# ===========================================================================
# slice_cache
# ===========================================================================

class TestSliceCache:

    def test_batch_slice(self):
        """Slice a single batch element from a batched cache."""
        cache = _cache(batch=4, seq_len=8)
        sliced = slice_cache(cache, batch_idx=[1], token_idx=list(range(8)))
        assert sliced.layers[0].keys.shape[0] == 1, "Batch slice should reduce to 1"

    def test_token_slice(self):
        """Slice a subset of token positions."""
        cache = _cache(batch=1, seq_len=10)
        token_idx = [0, 2, 4, 6]
        sliced = slice_cache(cache, batch_idx=[0], token_idx=token_idx)
        assert sliced.layers[0].keys.shape[2] == 4, \
            f"Token slice should give {len(token_idx)} positions"

    def test_slice_preserves_content(self):
        """Sliced values should match the original at the selected indices."""
        cache = _cache(batch=1, seq_len=5)
        # Set deterministic values for the first layer
        keys = torch.arange(40.0).reshape(1, 2, 5, 4)
        cache.layers[0].keys = keys

        sliced = slice_cache(cache, batch_idx=[0], token_idx=[1, 3])
        expected = keys[0:1, :, [1, 3], :]
        assert torch.equal(sliced.layers[0].keys, expected), \
            "Sliced keys do not match original at selected indices"

    def test_all_layers_sliced(self):
        """All cache layers must be sliced, not just the first."""
        n_layers = 4
        cache = _cache(batch=2, seq_len=6, n_layers=n_layers)
        sliced = slice_cache(cache, batch_idx=[0], token_idx=[0, 1, 2])
        for i, layer in enumerate(sliced.layers):
            assert layer.keys.shape[0] == 1,  f"Layer {i} batch not sliced"
            assert layer.keys.shape[2] == 3,  f"Layer {i} seq_len not sliced"

    def test_round_trip_expand_then_slice(self):
        """
        expand_cache(K) then slice_cache(batch=[0], ...) should give back a
        batch-1 cache with correct token positions.
        """
        batch_in, heads, seq_len, head_dim = 1, 2, 4, 4
        cache = _cache(batch=batch_in, heads=heads, seq_len=seq_len, head_dim=head_dim)
        original_keys = cache.layers[0].keys.clone()

        # Expand to K=3, then slice back to batch element 0
        expanded = expand_cache(cache, batch_sz=3)
        sliced = slice_cache(expanded, batch_idx=[0], token_idx=list(range(seq_len)))

        assert sliced.layers[0].keys.shape == original_keys.shape, \
            "Round-trip shape mismatch"
        assert torch.equal(sliced.layers[0].keys, original_keys), \
            "Round-trip values changed"

    def test_partial_sequence_slice_correct_positions(self):
        """Slicing token positions [0, 2, 4] should give those exact rows."""
        cache = _cache(batch=1, seq_len=5)
        # Build (1, 2, 5, 4) tensor with distinct values per seq position
        seq_vals = torch.arange(5.0).reshape(1, 1, 5, 1).expand(1, 2, 5, 4).clone()
        cache.layers[0].keys   = seq_vals.clone()
        cache.layers[0].values = seq_vals.clone()

        sliced = slice_cache(cache, batch_idx=[0], token_idx=[0, 2, 4])
        # Position 0 → row 0, position 2 → row 2, position 4 → row 4
        assert sliced.layers[0].keys.shape[2] == 3
        # Verify actual values: position 0 has value 0, position 2 has value 2, etc.
        assert sliced.layers[0].keys[0, 0, 0, 0].item() == 0.0
        assert sliced.layers[0].keys[0, 0, 1, 0].item() == 2.0
        assert sliced.layers[0].keys[0, 0, 2, 0].item() == 4.0
