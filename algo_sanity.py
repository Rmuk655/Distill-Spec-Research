"""algo_sanity.py — do the tree losses actually optimise acceptance?

A cheap (~minutes, no 32B) correctness check that disambiguates the two
hypotheses our 8B/0.6B negatives confound:
  H1  capacity ceiling   — algo fine, just no headroom on GSM8K
  H2  algorithm wrong    — the tree loss is a no-op / mis-wired

Method: overfit a *small* set of prompts with each loss and measure E[tau]
(full-depth expected accepted length) ON THE SAME PROMPTS every few steps.
Overfitting a handful guarantees headroom, so a working acceptance loss MUST
push E[tau] up; if it cannot, the formulation is broken.

Sample-size note — this is a CORRECTNESS test, not a generalisation estimate,
so a small set is the point (it guarantees fittability).  But the verdict is
asymmetric: a clear RISE is trustworthy even at small n; a NON-rise is
ambiguous (broken algo vs an unlucky sample with no headroom vs noise).  So we
use ~8 prompts, average E[tau] over several sampled trees, report the noise,
print per-prompt baselines (so you can see whether headroom existed), and judge
by effect size vs noise — never a single number.

Only the tree/telescoping family has an acceptance gradient worth testing.
depth_weight (w*jsd) is jsd's gradient times a DETACHED scalar — nothing to
validate.  additive (jsd + lambda*tree) reuses the tree gradient — covered
transitively by testing the tree loss itself.

Usage:
    # sweep the whole telescoping family in one run (fresh draft per loss):
    python algo_sanity.py --loss naive_tree,kl_tree,gbv_tree,bv_tree,traversal_tree --device cuda:0
    # single loss + flat control:
    python algo_sanity.py --loss naive_tree --device cuda:0
    python algo_sanity.py --loss jsd        --device cuda:0

    # credit-assignment ablation (the gate before a full GSM8K run):
    #   naive_tree      = detached survival (first-order surrogate, current default)
    #   naive_tree_full = un-detached survival → exact ∇E[τ] with path credit
    #   jsd             = flat control / ceiling
    # naive_tree_full MUST still raise E[tau] here (necessary condition); if it
    # stalls/destabilises even on the overfit set, abort before the day-long run.
    python algo_sanity.py --loss naive_tree,naive_tree_full,jsd --device cuda:0

    # off-policy gate (run BEFORE full GSM8K on op_ variants):
    #   op_naive_tree      = off-policy detached survival (teacher greedy path)
    #   op_naive_tree_full = off-policy exact ∇E[τ] (teacher path + un-detach)
    # Teacher's own tokens have near-unit acceptance → survival products stay large
    # → gradient reaches all depths without collapse.  Must WORKS here.
    python algo_sanity.py --loss op_naive_tree,op_naive_tree_full,naive_tree,jsd --device cuda:0
"""
from __future__ import annotations

import argparse
import copy
import random
import statistics

import torch

import train  # module-level imports only; importing does NOT run main()


def measure_etau(draft, teacher, ids_list, verifier, dt, tt, reps):
    """E[tau] over prompts x reps (fresh stochastic trees).

    Returns (overall_mean, overall_std, per_prompt_means)."""
    draft.eval()
    per_prompt, allvals = [], []
    for ids in ids_list:
        vals = []
        for _ in range(reps):
            train.Node.naive_cache.clear()
            train.Node.spectr_cache.clear()
            train.Node.specinfer_cache.clear()
            try:
                vals.append(train.expected_depth_scalar(
                    draft, teacher, ids, train.K, train.L, verifier, dt, tt))
            except Exception as e:                       # verifier.py:433 ZeroDiv etc.
                print(f"    [measure-skip] {type(e).__name__}: {e}")
        if vals:
            per_prompt.append(statistics.mean(vals))
            allvals.extend(vals)
    draft.train()
    mean = statistics.mean(allvals) if allvals else float("nan")
    disp = statistics.pstdev(allvals) if len(allvals) > 1 else 0.0   # prompt-to-prompt spread
    se = disp / (len(allvals) ** 0.5) if allvals else 0.0            # uncertainty of the MEAN
    return mean, disp, se, per_prompt


def verdict(loss, is_tree, baseline, best, noise):
    """Effect-size-vs-noise call.  Returns a one-line string."""
    delta = best - baseline
    ceiling = train.L
    if baseline > ceiling - 0.5:
        return (f"INCONCLUSIVE: baseline E[tau]={baseline:.2f} already near ceiling "
                f"{ceiling} — no headroom in this sample to demonstrate a rise.")
    if not is_tree:
        return f"(flat control) Δ={delta:+.3f}"
    strong = max(0.5, 3 * noise)   # noise = standard error of the mean
    weak = max(0.15, 1.5 * noise)
    if delta > strong:
        return (f"WORKS: tree loss raises acceptance (Δ={delta:+.3f} > {strong:.2f}) → "
                f"H2 rejected; 8B null is consistent with no-headroom (H1).")
    if delta > weak:
        return (f"WEAK: Δ={delta:+.3f} just above SE ({noise:.3f}) — moves acceptance "
                f"feebly; inspect gradient scale / survival weighting before scaling.")
    return (f"BROKEN?: Δ={delta:+.3f} not above SE ({noise:.3f}) even when overfitting. "
            f"Likely H2 — but re-check with more/headroom-ier prompts before concluding.")


def run_one_loss(loss_name, tok, draft, teacher, init_state, ids_list,
                 args, dt, tt):
    """Reset draft to init weights, overfit ids_list with loss_name, track E[tau]."""
    draft.load_state_dict(init_state)            # fresh draft per loss (fair start)
    loss_fn    = train.get_loss(loss_name)
    is_tree    = train.is_tree_loss(loss_name)
    is_offpol  = train.is_offpolicy_tree_loss(loss_name)
    # Measure E[tau] under 'naive' for ALL losses: a universal acceptance proxy
    # that only needs L.  Matched verifiers (traversal, ...) need extra args
    # (vocab_size) that expected_depth_scalar does not pass, so they crash.
    verifier = args.verifier or "naive"

    mode_tag = "off-policy tree" if is_offpol else ("tree" if is_tree else "flat")
    print("\n" + "=" * 74)
    print(f"  LOSS={loss_name} ({mode_tag})  "
          f"E[tau]-verifier='{verifier}'  n_prompts={len(ids_list)}  steps={args.steps}  "
          f"K={train.K} L={train.L}")
    print("=" * 74)

    trainable = [p for p in draft.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    b_mean, b_disp, b_se, b_pp = measure_etau(draft, teacher, ids_list, verifier,
                                              dt, tt, reps=args.baseline_reps)
    pp_str = ", ".join(f"{x:.2f}" for x in b_pp)
    print(f"  baseline E[tau]={b_mean:.3f}  spread={b_disp:.2f}  SE={b_se:.3f}")
    print(f"    per-prompt: [{pp_str}]")
    if b_mean > train.L - 0.5:
        print(f"  ⚠ baseline near ceiling {train.L} — little headroom; treat verdict as weak.")

    best = b_mean
    for step in range(1, args.steps + 1):
        ids = ids_list[step % len(ids_list)]
        train.Node.naive_cache.clear()
        train.Node.spectr_cache.clear()
        train.Node.specinfer_cache.clear()
        if is_offpol:
            loss = train.compute_offpolicy_tree_loss(loss_fn, draft, teacher, ids,
                                                     K=train.K, L=train.L,
                                                     draft_temp=dt, teacher_temp=tt)
        elif is_tree:
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
            m, d, _, _ = measure_etau(draft, teacher, ids_list, verifier,
                                      dt, tt, reps=args.reps)
            best = max(best, m)
            print(f"  step {step:4d}  E[tau]={m:.3f}  spread={d:.2f}  loss={loss.item():+.4f}  "
                  f"(Δ {m - b_mean:+.3f})")

    v = verdict(loss_name, is_tree or is_offpol, b_mean, best, b_se)
    print(f"  → {v}")
    return dict(loss=loss_name, verifier=verifier, baseline=b_mean,
                noise=b_se, best=best, delta=best - b_mean, verdict=v)


def main():
    ap = argparse.ArgumentParser(description="Tree-loss correctness / overfit probe.")
    ap.add_argument("--loss", default="naive_tree",
                    help="Comma-separated loss list. Tree losses test the algo; "
                         "jsd as flat control. e.g. naive_tree,kl_tree,gbv_tree,jsd")
    ap.add_argument("--n_prompts", type=int, default=8,
                    help="Prompts to overfit. Small = guaranteed headroom, but too "
                         "small risks an unlucky no-headroom sample. 8 is a safe handful.")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--measure_every", type=int, default=50)
    ap.add_argument("--reps", type=int, default=4,
                    help="Sampled trees per E[tau] measurement (periodic).")
    ap.add_argument("--baseline_reps", type=int, default=8,
                    help="More reps for the baseline (the verdict hinges on it).")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--train_dataset", default="gsm8k_train")
    ap.add_argument("--verifier", default=None,
                    help="Force one verifier for E[tau] across all losses. "
                         "Default: per-loss matched (LOSS_TO_VERIFIER).")
    args = ap.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device.split(":")[-1]) if ":" in args.device else 0)
    train.set_seed(args.seed)

    dt = getattr(train, "DRAFT_TEMP", train.DEFAULT_TEMP)
    tt = getattr(train, "TEACHER_TEMP", train.DEFAULT_TEMP)

    tok, draft, teacher = train.load_models(train.DRAFT_MODEL, train.TEACHER_MODEL,
                                            device=args.device)
    init_state = copy.deepcopy(draft.state_dict())   # snapshot for fresh-draft resets

    prompts = train.load_prompts_jsonl(train.dataset_path(args.train_dataset))
    random.Random(args.seed).shuffle(prompts)
    prompts = prompts[:args.n_prompts]
    ids_list = [torch.tensor(tok.encode(p), device=draft.device,
                             dtype=torch.long).unsqueeze(0) for p in prompts]

    losses = [s.strip() for s in args.loss.split(",") if s.strip()]
    results = [run_one_loss(name, tok, draft, teacher, init_state, ids_list,
                            args, dt, tt) for name in losses]

    print("\n" + "#" * 74)
    print(f"  SUMMARY  (n_prompts={args.n_prompts}, steps={args.steps}, "
          f"K={train.K} L={train.L})")
    print("#" * 74)
    print(f"  {'loss':16s} {'verifier':10s} {'base':>6s} {'best':>6s} "
          f"{'Δ':>7s} {'SE':>6s}")
    for r in results:
        print(f"  {r['loss']:16s} {r['verifier']:10s} {r['baseline']:6.3f} "
              f"{r['best']:6.3f} {r['delta']:+7.3f} {r['noise']:6.3f}")
    print()
    for r in results:
        print(f"  {r['loss']:16s} → {r['verdict']}")
    print("#" * 74)


if __name__ == "__main__":
    main()
