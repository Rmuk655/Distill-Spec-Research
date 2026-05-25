# Adding a New Speculative Decoding Algorithm

This guide walks you through adding a new speculative decoding paper as a
plug-and-play algorithm module.  After following these steps the algorithm
will appear in `list_algorithms()` and can be selected anywhere via
`get_algorithm("my_algo")`.

---

## Quick checklist

- [ ] 1. Create `algorithms/my_algo/__init__.py`
- [ ] 2. Create `algorithms/my_algo/algorithm.py` (implement `SpecDecAlgorithm`)
- [ ] 3. Create any helper modules (`heads.py`, `predictor.py`, `tree_optimizer.py`, …)
- [ ] 4. Uncomment the import in `algorithms/__init__.py`
- [ ] 5. Smoke-test with `python -c "from algorithms import get_algorithm; get_algorithm('my_algo')"`

---

## 1. Understand the base class

Every algorithm inherits from `SpecDecAlgorithm` in `algorithms/base.py`:

```python
class SpecDecAlgorithm(ABC):

    meta: AlgorithmMeta   # filled in as a class variable

    @abstractmethod
    def setup(self, target_model, tokenizer, device,
              draft_model=None, model_family=None, **kwargs) -> None: ...

    @abstractmethod
    def generate(self, prompt_ids, max_new_tokens=128, K=4, L=8,
                 temperature=1.0, **kwargs) -> GenerationResult: ...

    def train(self, dataset_path, output_dir, config=None, **kwargs) -> None:
        # Default: raises NotImplementedError unless TrainingType.NONE
        ...
```

`GenerationResult` is a dataclass with these fields:

| Field | Type | Meaning |
|---|---|---|
| `token_ids` | `List[int]` | Generated tokens (not including prompt) |
| `n_draft_tokens` | `int` | Total draft tokens proposed |
| `n_accepted_tokens` | `int` | Draft tokens accepted by target |
| `n_target_calls` | `int` | Number of target forward passes |
| `time_draft_s` | `float` | Wall-clock seconds in draft model |
| `time_target_s` | `float` | Wall-clock seconds in target model |
| `extra` | `Dict` | Algorithm-specific data (verifier name, tree budget, …) |

Derived properties computed automatically:
- `acceptance_rate = n_accepted / n_draft`
- `block_efficiency = len(token_ids) / n_target_calls`

---

## 2. Choose the right enum values for `AlgorithmMeta`

### `DraftSource`
| Value | When to use |
|---|---|
| `INDEPENDENT` | Separate small autoregressive LM (GBV, Sequoia, standard) |
| `ATTACHED` | Extra heads/layers added to the target itself (Medusa) |
| `FEATURE` | Predictor that consumes target hidden states (EAGLE family) |
| `TREE_OPTIMAL` | Reserved for future hybrid methods |

### `VerificationType`
| Value | When to use |
|---|---|
| `TOKEN_LEVEL` | Linear draft, accept/reject each token (Leviathan et al.) |
| `TREE` | Draft token tree, one target pass, accept longest prefix |

### `TrainingType`
| Value | When to use |
|---|---|
| `NONE` | No training beyond a pre-trained draft model (standard, Sequoia) |
| `DISTRIBUTION` | KL / JSD distillation of draft onto target distribution (GBV) |
| `REGRESSION` | MSE regression of predictor onto target hidden states (EAGLE) |
| `SFT_HEADS` | Supervised fine-tuning of attached heads (Medusa) |

### `needs_draft_model`
- `True`:  a `draft_model=` argument must be passed to `setup()`.
- `False`: the draft mechanism lives inside the target model (Medusa).

---

## 3. Create the directory structure

```
algorithms/
└── my_algo/
    ├── __init__.py          # exports ALGORITHM = MyAlgoAlgorithm()
    ├── algorithm.py         # SpecDecAlgorithm subclass
    └── <helpers>.py         # heads.py, predictor.py, etc. if needed
```

### Minimal `__init__.py`

```python
"""
My Algorithm — AuthorName et al. YEAR (arXiv:XXXX.XXXXX).
STUB — not yet implemented.

Core idea:
  One paragraph explaining what makes this algorithm novel.
  ...

Reference: https://github.com/...
"""

from .algorithm import MyAlgoAlgorithm

ALGORITHM = MyAlgoAlgorithm()
__all__ = ["ALGORITHM", "MyAlgoAlgorithm"]
```

### Minimal `algorithm.py`

```python
from __future__ import annotations
from algorithms.base import (
    SpecDecAlgorithm, AlgorithmMeta, GenerationResult,
    DraftSource, VerificationType, TrainingType,
)

class MyAlgoAlgorithm(SpecDecAlgorithm):

    meta = AlgorithmMeta(
        name              = "my_algo",
        paper_title       = "Full paper title",
        arxiv_id          = "XXXX.XXXXX",
        draft_source      = DraftSource.INDEPENDENT,   # change as appropriate
        verification      = VerificationType.TREE,
        training          = TrainingType.NONE,
        needs_draft_model = True,
        notes             = "One-line notes about the algorithm.",
    )

    def setup(self, target_model, tokenizer, device,
              draft_model=None, model_family=None, **kwargs) -> None:
        self.target = target_model
        self.draft  = draft_model
        self.tok    = tokenizer
        self.device = device

    def generate(self, prompt_ids, max_new_tokens=128, K=4, L=8,
                 temperature=1.0, **kwargs) -> GenerationResult:
        raise NotImplementedError("TODO: implement generate()")

    def train(self, dataset_path, output_dir, config=None, **kwargs) -> None:
        raise NotImplementedError("TODO: implement train()")
```

---

## 4. Register in `algorithms/__init__.py`

```python
# Uncomment these two lines once the algorithm is ready:
from .my_algo import ALGORITHM as _my_algo

ALGORITHM_REGISTRY = {
    ...
    "my_algo": _my_algo,
}
```

Also update the `stubs` list in `list_algorithms()` if you want it to appear
before it is fully implemented.

---

## 5. Reference implementations for each paradigm

| Paradigm | Implemented example | Key file to read |
|---|---|---|
| Independent draft + token-level | `standard_speculative/` | `algorithm.py` |
| Independent draft + OTLP tree | `distillspec_gbv/` | `algorithm.py`, `verifiers/runner.py` |
| Attached heads | `medusa/` | `algorithm.py`, `heads.py` |
| Feature predictor + fixed tree | `eagle/` | `algorithm.py`, `predictor.py` |
| Feature predictor + dynamic tree | `eagle2/`, `eagle3/` | `algorithm.py` |
| Hardware-aware DP tree | `sequoia/` | `algorithm.py`, `tree_optimizer.py` |

---

## 6. Testing your implementation

```python
# Smoke test — no GPU needed, just checks imports and meta
from algorithms import get_algorithm, list_algorithms

algo = get_algorithm("my_algo")
print(algo.meta)
print(list_algorithms())
```

For a full end-to-end test with small models:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok   = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
draft = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tgt   = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

algo = get_algorithm("my_algo")
algo.setup(tgt, tok, "cpu", draft_model=draft, model_family="qwen")

prompt_ids = tok("Hello, world!", return_tensors="pt").input_ids
result = algo.generate(prompt_ids, max_new_tokens=20, K=4)

print(tok.decode(result.token_ids))
print(f"acceptance rate: {result.acceptance_rate:.2%}")
print(f"block efficiency: {result.block_efficiency:.2f} tokens/target-call")
```

---

## 7. Algorithm-specific hyperparameters

Pass algorithm-specific kwargs through `generate()` and `setup()`:

```python
result = algo.generate(prompt_ids, K=4, L=8,
                        temperature=0.8,
                        verifier="traversal",   # GBV-specific
                        tree_budget=60,         # EAGLE-2/3/Sequoia-specific
                        exit_threshold=0.05)    # FastEagle-specific
```

Document your new kwargs in the method docstring.  Never use positional-only
arguments — always use `**kwargs` so the base class signature is preserved.

---

## Common pitfalls

1. **Forgetting to call `.detach()` on target hidden states during training.**
   Gradients must not flow into the frozen target LM.

2. **Tree attention mask shape.**
   The mask must be `[num_tree_nodes, num_tree_nodes]` with `True` (or `-inf`)
   blocking positions a node cannot attend to.  Every node attends to its own
   ancestors only, not its siblings or descendants.

3. **EOS token handling.**
   Check for `self.tok.eos_token_id` in the accepted tokens and break the loop.
   See `standard_speculative/algorithm.py` for the pattern.

4. **Temperature.**
   Apply temperature *before* softmax: `logits / temperature`.  Do not modify
   probabilities after softmax — that changes the distribution.

5. **Model family differences.**
   Qwen3 pre-divides logits by temperature internally.  Use
   `model_family.recover_raw_logits(output_scores, temperature)` from
   `core/model_families/` before applying your own temperature.  See
   `docs/ADDING_A_MODEL_FAMILY.md`.
