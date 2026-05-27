"""
seed_db.py — import historical experimental results into results.db.

This file seeds the DB with results from the bl56mnhxp proof-of-concept run
(200-step offline distillation, Qwen2.5-0.5B draft → Qwen3-0.6B target).
It is a one-time historical record — not an active data pipeline.

Label naming in this file uses the OLD convention (kl200, ebe200) from the
200-step run.  Current naming convention is <loss>-<dataset> (e.g.
kl-gsm8k, ebe-gsm8k).  Do NOT change the labels here; the historical rows
in results.db carry those labels and changing them would break continuity.

Safe to re-run: insert_run() checks for existing rows before inserting.

For adding NEW results interactively, use add_run() from a REPL:
    from seed_db import add_run
    add_run("kl-gsm8k", "gbv", K=3, dataset="gsm8k", block_eff=2.81)

Usage:
    python seed_db.py            # seeds everything
    python seed_db.py --dry_run  # show what would be inserted without writing
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))
import results_db

TARGET = "Qwen/Qwen3-0.6B"
BASE_PATH = "Qwen/Qwen2.5-0.5B"

# Historical checkpoint paths from the 200-step bl56mnhxp proof-of-concept run.
# These paths are stored as metadata strings in the DB (they don't need to exist
# on disk for the DB insert to succeed).  Current runs save checkpoints under
# gbv-research/orchestration/checkpoints/<label>/ — update paths there for new
# DRAFT_META entries if you extend this seeder with new experiments.
_KL_PATH_HISTORICAL  = "OSD/checkpoints/kl200_merged"
_EBE_PATH_HISTORICAL = "OSD/checkpoints/ebe200_merged"

# Historical label → metadata mapping.
# Labels use the OLD 200-step naming (kl200, ebe200).  Current convention is
# <loss>-<dataset> (e.g. kl-gsm8k, ebe-gsm8k).  Keep these as-is so the DB
# rows remain consistent with what was actually run.
DRAFT_META = {
    "baseline": dict(draft_path=BASE_PATH,             loss_name="baseline", train_steps=0,
                     learning_rate=0.0, lora_rank=0),
    "kl200":    dict(draft_path=_KL_PATH_HISTORICAL,   loss_name="kl",       train_steps=200,
                     learning_rate=3e-5, lora_rank=8),
    "ebe200":   dict(draft_path=_EBE_PATH_HISTORICAL,  loss_name="ebe",      train_steps=200,
                     learning_rate=3e-5, lora_rank=8),
}


def _insert(row, dry_run):
    if dry_run:
        print(f"  [DRY] {row}")
        return -1
    rid = results_db.insert_run(row)
    print(f"  -> run_id={rid}  {row.get('draft_label')} {row.get('mode')} K={row.get('K')} "
          f"dataset={row.get('dataset')}")
    return rid


def seed_alpha_50(dry_run=False):
    """Alpha-50 results from bl56mnhxp steps 1-3."""
    print("\n--- Alpha-50 results ---")

    per_cat_data = {
        # format: (factual, coding, cs_concepts, ml_ai, math)
        "baseline": (0.5291, 0.4962, 0.4509, 0.4669, 0.5158),
        "kl200":    (0.4760, 0.4986, 0.4924, 0.5139, 0.5487),
        "ebe200":   (0.5205, 0.5345, 0.4680, 0.5328, 0.5485),
    }
    categories = ["factual", "coding", "cs_concepts", "ml_ai", "math"]

    alpha_results = [
        # (draft_label, alpha_mean, alpha_std, alpha_ci95, throughput, ms_per_tok, peak_vram)
        ("baseline", 0.4918, 0.1217, 0.0337, 0.92, 1085.1, 2119),
        ("kl200",    0.5059, 0.0970, 0.0269, 0.90, 1107.6, 2119),
        ("ebe200",   0.5209, 0.1109, 0.0307, 0.96, 1044.1, 2119),
    ]

    for label, alpha_mean, alpha_std, alpha_ci95, tp, ms, vram in alpha_results:
        row = dict(
            draft_label=label, target_path=TARGET, dataset="diverse50",
            n_prompts=50, mode="alpha", K=1, L=5, temperature=0.6,
            alpha_mean=alpha_mean, alpha_std=alpha_std, alpha_ci95=alpha_ci95,
            throughput=tp, ms_per_tok=ms, peak_vram_mb=vram,
            notes="bl56mnhxp steps 1-3",
            **DRAFT_META[label],
        )
        run_id = _insert(row, dry_run)

        # Insert per-category synthetic per_prompt rows (10 prompts each, uniform alpha)
        if not dry_run and run_id > 0:
            cat_alphas = per_cat_data[label]
            rows = []
            for cat_idx, (cat, alpha) in enumerate(zip(categories, cat_alphas)):
                for j in range(10):
                    rows.append(dict(
                        run_id=run_id,
                        prompt_idx=cat_idx * 10 + j,
                        category=cat,
                        alpha=alpha,
                        block_eff=None,
                        gen_tokens=30,
                        wall_ms=None,
                    ))
            results_db.insert_per_prompt_batch(rows)


def seed_k_sweep_specinfer(dry_run=False):
    """K-sweep specinfer results from bl56mnhxp steps 4-6."""
    print("\n--- K-sweep specinfer (baseline & ebe200) ---")

    # (draft_label, K, block_eff)
    results = [
        ("baseline", 1, 2.654179),
        ("ebe200",   1, 2.634146),
        ("baseline", 3, 2.373057),
        ("ebe200",   3, 2.578652),
        ("baseline", 5, 2.376943),
        ("ebe200",   5, 2.448000),
    ]

    for label, K, be in results:
        row = dict(
            draft_label=label, target_path=TARGET, dataset="diverse50",
            n_prompts=30, mode="specinfer", K=K, L=5, temperature=1.0,
            block_eff=be, notes="bl56mnhxp steps 4-6",
            **DRAFT_META[label],
        )
        _insert(row, dry_run)


def seed_lr_ablation(dry_run=False):
    """LR ablation training results — EBE at three learning rates."""
    print("\n--- LR ablation training curves ---")

    # (label, lr, step, loss, accept_weight)
    training_points = [
        # LR=1e-5 (bl56mnhxp step 7/9)
        ("ebe_lr1e5", 1e-5, 50,  2.0555, 0.976),
        ("ebe_lr1e5", 1e-5, 100, 2.0828, 0.977),
        ("ebe_lr1e5", 1e-5, 150, 1.8167, 0.978),
        ("ebe_lr1e5", 1e-5, 200, 1.5343, 0.974),
        # LR=3e-5 ablation run (bl56mnhxp step 8/9)
        ("ebe_lr3e5", 3e-5, 50,  1.8774, 0.976),
        ("ebe_lr3e5", 3e-5, 100, 1.6309, 0.972),
        ("ebe_lr3e5", 3e-5, 150, 1.3743, 0.973),
        ("ebe_lr3e5", 3e-5, 200, 1.1253, 0.971),
        # LR=1e-4 (bl56mnhxp step 9/9)
        ("ebe_lr1e4", 1e-4, 50,  1.5707, 0.975),
        ("ebe_lr1e4", 1e-4, 100, 1.2600, 0.964),
        ("ebe_lr1e4", 1e-4, 150, 1.1833, 0.965),
        ("ebe_lr1e4", 1e-4, 200, 0.9893, 0.966),
        # Main ebe200 training run (earlier, same LR=3e-5)
        ("ebe200",    3e-5, 50,  1.87,   0.979),
        ("ebe200",    3e-5, 100, 1.54,   0.975),
        ("ebe200",    3e-5, 150, 1.20,   0.969),
        ("ebe200",    3e-5, 200, 1.08,   0.966),
        # KL=3e-5
        ("kl200",     3e-5, 50,  2.03,   None),
        ("kl200",     3e-5, 100, 1.68,   None),
        ("kl200",     3e-5, 150, 1.43,   None),
        ("kl200",     3e-5, 200, 1.26,   None),
    ]

    if dry_run:
        print(f"  [DRY] would insert {len(training_points)} training curve points")
        return

    for label, lr, step, loss, aw in training_points:
        results_db.insert_train_step(
            label=label, loss_name="ebe" if "ebe" in label else "kl",
            step=step, loss=loss, learning_rate=lr, lora_rank=8,
            accept_weight=aw,
        )
    print(f"  Inserted {len(training_points)} training curve points")


def seed_gbv_same_model(dry_run=False):
    """GBV baseline: same-model (draft=target=Qwen3-0.6B) block efficiency."""
    print("\n--- Same-model GBV baseline ---")

    # From earlier verification mode test
    results = [
        ("naive",     1, 5.17, 17.95, 55.7),
        ("spectr",    1, 5.17, 13.85, 72.2),
        ("bv",        1, 5.17,  9.51, 105.1),
        ("traversal", 1, 5.17,  7.96, 125.6),
        ("gbv",       1, 3.00, 10.29,  97.2),
        ("specinfer", 1, 3.24, 11.18,  89.4),
        ("nss",       1, 2.04,  7.12, 140.4),
    ]
    for mode, K, be, tp, ms in results:
        row = dict(
            draft_label="same_model", draft_path=TARGET, target_path=TARGET,
            loss_name="baseline", train_steps=0, learning_rate=0.0, lora_rank=0,
            dataset="test3", n_prompts=3, mode=mode, K=K, L=5, temperature=1.0,
            block_eff=be, throughput=tp, ms_per_tok=ms,
            notes="same-model upper bound (draft=target=Qwen3-0.6B)",
        )
        _insert(row, dry_run)


def seed_all(dry_run=False):
    print(f"Seeding results.db {'(DRY RUN)' if dry_run else ''}\n")
    results_db._connect().close()  # ensure schema created
    seed_alpha_50(dry_run)
    seed_k_sweep_specinfer(dry_run)
    seed_lr_ablation(dry_run)
    seed_gbv_same_model(dry_run)
    print("\nDone.")


def add_run(draft_label: str, mode: str, K: int, dataset: str,
            block_eff: float = None, alpha_mean: float = None,
            alpha_std: float = None, alpha_ci95: float = None,
            throughput: float = None, ms_per_tok: float = None,
            temperature: float = 1.0, n_prompts: int = 30,
            notes: str = None):
    """
    Helper to add a single result from the REPL after an experiment completes.

    Use current label naming (<loss>-<dataset>), not historical 200-step names.

    Example:
        from seed_db import add_run
        add_run("ebe-gsm8k", "gbv", K=3, dataset="gsm8k", block_eff=2.81)
        add_run("kl-gsm8k",  "specinfer", K=5, dataset="gsm8k", block_eff=2.64)

    If draft_label is not in DRAFT_META (historical entries), draft_path defaults
    to the label string itself and train_steps/lr/rank are zeroed — fill in notes
    with any extra metadata you want to record.
    """
    meta = DRAFT_META.get(draft_label, dict(
        draft_path=draft_label, loss_name="custom",
        train_steps=0, learning_rate=0.0, lora_rank=0,
    ))
    row = dict(
        draft_label=draft_label, target_path=TARGET, dataset=dataset,
        n_prompts=n_prompts, mode=mode, K=K, L=5, temperature=temperature,
        block_eff=block_eff, alpha_mean=alpha_mean, alpha_std=alpha_std,
        alpha_ci95=alpha_ci95, throughput=throughput, ms_per_tok=ms_per_tok,
        notes=notes or "manual entry",
        **meta,
    )
    rid = results_db.insert_run(row)
    print(f"Inserted run_id={rid}")
    return rid


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    seed_all(dry_run=args.dry_run)
