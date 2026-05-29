# Known Issues

## 1. EBE Loss Broken

**Symptom**: BE delta -0.202 vs baseline on laptop (bv/K=3: 2.846 vs baseline 3.048).

**Evidence**: Perplexity looks fine (7.96 vs 7.47 baseline) — model isn't diverging, it's simply not learning the right objective. The language modeling signal is intact but the EBE signal isn't propagating usefully.

**Disabled in**: `ebe`, `ebe_single`, `ebe_tree` (all free-tier configs: `kaggle.yaml`, `colab.yaml`, `a100.yaml`).

**Investigation pointer**: Check loss gradient computation in [`algorithms/distillspec_gbv/trainer.py`](../algorithms/distillspec_gbv/trainer.py) — verify that EBE gradients are non-zero and correctly scaled relative to the LM loss.

---

## 2. Online Loss Diverged

**Symptom**: Perplexity 15.46 vs baseline 7.47 (~2×); BE delta -0.679 (worst of all losses tested).

**Evidence**: Both metrics degrade together, indicating true training divergence rather than a metric artifact.

**Hypothesis**: LR too high (online training uses 10× the offline LR), reward-scale miscalibration, or a bug in KL computation at reject positions.

**Disabled in**: `online`, `online_ebe`, `online_ebe_single`, `online_kl_tree`, `online_ebe_tree` (all free-tier configs).

**Investigation pointer**: Check [`orchestration/experiment.py`](../orchestration/experiment.py) `online_adapt` section and [`algorithms/online_serve.py`](../algorithms/online_serve.py) — confirm LR schedule, reward normalization, and KL mask at reject positions.
