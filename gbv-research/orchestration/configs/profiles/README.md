# Experiment Profiles — Phase-1 Tiered Workflow

Profiles are self-contained YAML configs under `orchestration/configs/profiles/` that
control exactly what the pipeline does in one session.  Pass them as
`--config profiles/<name>` (no `.yaml` suffix needed).

---

## The four active profiles

| Profile | `--config` arg | Purpose | Trains? | Evals? |
|---------|---------------|---------|---------|--------|
| `smoke` | `profiles/smoke` | ~5 min crash-check every session; proves every code path runs | 10 steps/loss (throwaway) | n=5, smoke-only |
| `train_one_loss` | `profiles/train_one_loss` | **Phase-1 workhorse** — tiered: train ONE flat loss + in-training val_loss + light BE sanity | Yes (full steps) | Light: 1 matched verifier (`bv`), K=3, T=1.0, n=100, GSM8K only |
| `tree_variant_week` | `profiles/tree_variant_week` | **Phase-2 tree survivor** — tiered: train ONE tree variant + light BE sanity | Yes (full steps) | Light: tree-matched non-OT subset `bv,gbv,traversal`, K=3, T=1.0, n=100 |
| `eval_after_train` | `profiles/eval_after_train` | **Deferred heavier eval** — eval already-trained checkpoints; no training | No (`eval_only`) | Full verifier set, multi-dataset, K=3, T=1.0, n=100 |

---

## Tiered design — what every Phase-1/2 session produces

Each `train_one_loss` or `tree_variant_week` session is a single tiered run:

```
train  →  val_loss (always-on, inside trainer.py every health.val_every steps)
       →  LIGHT BE sanity (1 matched verifier or tree-matched subset, n=100, K=3, T=1.0)
```

`val_loss` is computed **inside `trainer.py`** independent of the BE eval — it is the
train-down / val-up overfit signal.  The **heavy** full eval (all verifiers, K∈{3,5},
n=1319, multi-seed) is **deferred to the A100 confirmation** (`eval_after_train` is the
optional in-between free-tier heavier eval).

`--light_eval` (YAML: `experiment.light_eval: true`) is the flag enabling tiered mode.
`--train_only` still exists as an optional flag for a pure training burn (no eval at all)
but is **not the Phase-1 default**.

---

## Usage

```bash
# ── ALWAYS smoke first (every session, ~5 min) ────────────────────────────────
python orchestration/experiment.py --config profiles/smoke --smoke --yes \
  --losses forward_kl --storage_root /kaggle/working/specdist

# ── Phase-1: TIERED flat baseline (train + val_loss + light BE, 1 verifier) ──
python orchestration/experiment.py --config profiles/train_one_loss --yes \
  --losses forward_kl --skip_existing --storage_root /kaggle/working/specdist \
  --experiment_tag p1_forward_kl

# ── Phase-2: TIERED tree variant (train + light BE, tree-matched verifiers) ───
python orchestration/experiment.py --config profiles/tree_variant_week --yes \
  --losses kl_tree --skip_existing --storage_root /kaggle/working/specdist \
  --experiment_tag wk3_kl_tree

# ── Optional deferred heavier eval (full verifier set, multi-dataset, n=100) ──
python orchestration/experiment.py --config profiles/eval_after_train --yes \
  --losses forward_kl,kl_tree --skip_existing \
  --storage_root /kaggle/working/specdist --experiment_tag wk4_eval

# ── Dry-run (no GPU, no model load — confirm exactly what steps will run) ────
python orchestration/experiment.py --config profiles/train_one_loss \
  --losses forward_kl --dry_run
```

`--losses` on the CLI always overrides the profile's YAML default.
`forward_kl` and `reverse_kl` are accepted as aliases for the orchestration keys
`kl` / `rev_kl`.  Both `--losses kl` and `--losses forward_kl` work.

---

## What each profile runs per pipeline phase

| Phase | `smoke` | `train_one_loss` | `tree_variant_week` | `eval_after_train` |
|-------|---------|-----------------|--------------------|--------------------|
| Phase 2 — train + merge | 10 steps (throwaway) | ✅ full steps | ✅ full steps | ❌ skipped |
| val_loss (inside trainer) | ✅ (always-on) | ✅ (always-on) | ✅ (always-on) | n/a |
| Phase 3 — light BE sanity | n=5 (smoke) | 1 verifier `bv`, n=100 | `bv,gbv,traversal`, n=100 | ❌ |
| Phase 1 — baseline group 0 | ✅ (smoke) | ❌ dropped by `light_eval` | ❌ dropped by `light_eval` | ✅ |
| Phase 4 — multi-dataset | ❌ | ❌ dropped by `light_eval` | ❌ dropped by `light_eval` | ✅ |

---

## Algo priority (locked)

Always run in this order — never skip ahead:

1. **`forward_kl`** first — establish the flat baseline before anything else
2. **Tree-loss training** (`kl_tree`, `bv_tree`, `gbv_tree`, `traversal_tree`, …) — core novel contribution
3. **GBV verifier** — verifier-side contribution (eval-only on survivor checkpoint)
4. **Online variants** — lowest priority, deferred to the very end

See `docs/ENGINEER_PLAYBOOK.md` for the full sequenced run plan with dependency graph.

---

## Adding a new profile

```bash
# Copy the closest existing profile and edit it:
cp orchestration/configs/profiles/train_one_loss.yaml \
   orchestration/configs/profiles/my_ablation.yaml
# Set: experiment.losses, wandb_group, run_label, light_eval / eval_only / train_only
```
