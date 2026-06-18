"""algo_sanity.py — does the tree loss actually optimise acceptance?

A cheap (~minutes, no 32B) correctness check that disambiguates the two
hypotheses our 8B/0.6B negatives confound:
  H1  capacity ceiling   — algo fine, just no headroom on GSM8K
  H2  algorithm wrong    — the tree loss is a no-op / mis-wired

Method: overfit a *handful* of prompts with the loss under test and measure
E[tau] (full-depth expected accepted length) ON THE SAME PROMPTS every few
steps.  Overfitting guarantees headroom, so:

  * naive_tree (tree loss):  E[tau] MUST climb clearly.  If it does not move
    while the train loss drops, the loss optimises something disconnected from
    acceptance -> H2 (broken) -> fix before scaling.
  * jsd (flat control):      run it too; q->p should also raise E[tau].  If
    flat reaches the SAME E[tau] as tree, that is positive evidence the 8B
    result is genuine no-headroom (H1), not a bug.

Usage:
    python algo_sanity.py --loss naive_tree --device cuda:0
    python algo_sanity.py --loss jsd        --device cuda:0   # control
"""
from __future__ import annotations

import argparse
import statistics

import torch

import train  # module-level imports only; importing does NOT run main()


def measure_etau(draft, teacher, ids_list, verifier, dt, tt, reps=4):
    """Stochastic estimate of E[tau] averaged over prompts x reps (fresh trees)."""
    draft.eval()
    vals = []
    for ids in ids_list:
        for _ in range(reps):
            train.Node.naive_cache.clear()
            train.Node.spectr_cache.clear()
            train.Node.specinfer_cache.clear()
            try:
                vals.append(train.expected_depth_scalar(
                    draft, teacher, ids, train.K, train.L, verifier, dt, tt))
            except Exception as e:                       # verifier.py:433 ZeroDiv etc.
                print(f"    [measure-skip] {type(e).__name__}: {e}")
    draft.train()
    return statistics.mean(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Tree-loss correctness / overfit probe.")
    ap.add_argument("--loss", default="naive_tree",
                    help="Loss to probe (naive_tree to test the algo; jsd for control).")
    ap.add_argument("--n_prompts", type=int, default=3,
                    help="How many prompts to overfit (small = guaranteed headroom).")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--measure_every", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--train_dataset", default="gsm8k_train")
    ap.add_argument("--verifier", default="naive",
                    help="Verifier whose E[tau] we track (fixed across losses to compare).")
    args = ap.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(train._gpu_index_from_device(args.device)
                              if hasattr(train, "_gpu_index_from_device")
                              else int(args.device.split(":")[-1]))
    train.set_seed(args.seed)

    dt = getattr(train, "DRAFT_TEMP", train.DEFAULT_TEMP)
    tt = getattr(train, "TEACHER_TEMP", train.DEFAULT_TEMP)

    tok, draft, teacher = train.load_models(train.DRAFT_MODEL, train.TEACHER_MODEL,
                                            device=args.device)

    prompts = train.load_prompts_jsonl(train.dataset_path(args.train_dataset))
    import random
    random.Random(args.seed).shuffle(prompts)
    prompts = prompts[:args.n_prompts]
    ids_list = [torch.tensor(tok.encode(p), device=draft.device,
                             dtype=torch.long).unsqueeze(0) for p in prompts]

    loss_fn = train.get_loss(args.loss)
    is_tree = train.is_tree_loss(args.loss)

    print("=" * 70)
    print(f"  algo_sanity  loss={args.loss} ({'tree' if is_tree else 'flat'})  "
          f"n_prompts={args.n_prompts}  steps={args.steps}  K={train.K} L={train.L}")
    print(f"  tracking E[tau] under verifier='{args.verifier}' (full depth, max={train.L})")
    print("=" * 70)

    trainable = [p for p in draft.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    d0 = measure_etau(draft, teacher, ids_list, args.verifier, dt, tt)
    print(f"  step    0  E[tau]={d0:.3f}  (baseline, before any training)")

    best = d0
    for step in range(1, args.steps + 1):
        ids = ids_list[step % len(ids_list)]
        train.Node.naive_cache.clear()
        train.Node.spectr_cache.clear()
        train.Node.specinfer_cache.clear()
        if is_tree:
            loss = train.compute_tree_loss(loss_fn, draft, teacher, ids,
                                           K=train.K, L=train.L,
                                           draft_temp=dt, teacher_temp=tt)
        else:
            loss = train.compute_flat_loss(loss_fn, draft, teacher, ids,
                                           max_new_tokens=train.MAX_NEW_TOKENS)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if step % args.measure_every == 0:
            d = measure_etau(draft, teacher, ids_list, args.verifier, dt, tt)
            best = max(best, d)
            print(f"  step {step:4d}  E[tau]={d:.3f}  loss={loss.item():+.4f}  "
                  f"(Δ from baseline {d - d0:+.3f})")

    print("-" * 70)
    delta = best - d0
    print(f"  baseline E[tau]={d0:.3f}   best E[tau]={best:.3f}   Δ={delta:+.3f}")
    if is_tree:
        if delta > 0.5:
            print("  VERDICT: tree loss RAISES acceptance on the overfit set → algo works "
                  "(H2 rejected). The 8B null is consistent with no-headroom (H1).")
        elif delta > 0.15:
            print("  VERDICT: weak rise — tree loss moves acceptance but feebly. Inspect "
                  "gradient scale / survival weighting before trusting it at scale.")
        else:
            print("  VERDICT: tree loss does NOT raise acceptance even when overfitting "
                  "→ formulation is broken (H2). Fix BEFORE any 32B run.")
    print("=" * 70)


if __name__ == "__main__":
    main()
