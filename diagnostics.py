"""
diagnostics.py — post-hoc objective/BE diagnostic analysis (--diagnose in eval.py).

Never called on throughput runs.  Runs as a separate pass after the timed eval
loop.  Skip this file entirely if you are not running --diagnose.

Public API:
  run_objective_be_diagnostic(p_model, q_model, tok, prompts, per_prompt_be,
                               args, mode_name, device)
      -> (jsd_xs, fkl_xs, be_ys, valid_indices)   [also logs to W&B]

  run_decomposition_radar(p_model, tok, prompts, jsd_trained, be_ys,
                           valid_indices, args, mode_name, device)
      [logs stacked-bar + radar W&B images; requires args.baseline_checkpoint]
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import DEFAULT_DTYPE, BE_EASY_FRAC, BE_HARD_FRAC
from transformers import AutoModelForCausalLM


@torch.no_grad()
def _prompt_divergence(p_model, q_model, tok, prompt, max_new_tokens, device):
    """Per-prompt mean JSD and forward-KL(p||q) between teacher (p) and draft (q).

    Mirrors the flat training objective: teacher greedily rolls out, both models
    are forwarded on the same sequence, divergence is averaged over the generated
    positions.  Pure measurement, called outside any timer — cannot affect
    throughput.  Returns (jsd, fwd_kl), or (None, None) if nothing was generated.
    """
    import torch.nn.functional as F
    ids  = torch.tensor(tok.encode(prompt), device=device, dtype=torch.long).unsqueeze(0)
    attn = torch.ones_like(ids)
    gen = p_model.generate(ids, attention_mask=attn, max_new_tokens=max_new_tokens,
                           do_sample=False, pad_token_id=p_model.config.eos_token_id,
                           use_cache=True)
    sl = ids.shape[1]
    p_logits = p_model(gen, return_dict=True).logits[0, sl - 1:-1].float()
    q_logits = q_model(gen, return_dict=True).logits[0, sl - 1:-1].float()
    if p_logits.shape[0] == 0:
        return None, None
    logp = F.log_softmax(p_logits, dim=-1)
    logq = F.log_softmax(q_logits, dim=-1)
    p, q = logp.exp(), logq.exp()
    fkl  = (p * (logp - logq)).sum(-1).mean().item()                 # forward KL(p||q)
    m    = (0.5 * (p + q)).clamp_min(1e-12)
    logm = m.log()
    jsd  = 0.5 * ((p * (logp - logm)).sum(-1)
                  + (q * (logq - logm)).sum(-1)).mean().item()       # JSD
    return jsd, fkl


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / (sxx ** 0.5 * syy ** 0.5)


def _pval_pearson(r: float, n: int) -> float:
    """Two-tailed p-value for Pearson r via t-distribution normal approx (good for n>=30)."""
    if n < 3 or r != r:
        return float("nan")
    t = r * ((n - 2) ** 0.5) / max((1.0 - r * r) ** 0.5, 1e-12)
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / (2.0 ** 0.5))))


def _ranks(xs: list) -> list:
    """Return rank vector (1-based) for xs.  Average ranks for ties."""
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and xs[order[j + 1]] == xs[order[j]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0   # 1-based average rank for tied block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list, ys: list) -> float:
    """Spearman rank correlation.

    Preferred over Pearson here because it is invariant to monotone compression
    of either axis.  As the draft model improves, JSD values compress toward zero
    (range restriction), which shrinks Pearson r even when the rank ordering is
    unchanged.  Spearman rho measures whether lower JSD ranks predict higher BE
    ranks — the right question for H0 testing.
    """
    if len(xs) < 2:
        return float("nan")
    return _pearson(_ranks(xs), _ranks(ys))


def run_objective_be_diagnostic(p_model, q_model, tok, prompts, per_prompt_be,
                                args, mode_name, device, trained_loss: str = "jsd"):
    """Correlate per-prompt training divergence with per-prompt block efficiency.

    trained_loss: 'jsd' or 'fwdkl' — which divergence was minimised during training.
    Primary metrics (σ, mean, Spearman ρ, scatter plot) are reported for the trained
    loss.  The other divergence is computed and reported as secondary for reference.

    Flat (|r| ~ 0) correlation  → objective mismatch (H0): minimising this loss
                                   will not move BE.
    Negative correlation        → lower divergence ⇒ higher BE: the objective IS
                                   the right lever.

    W&B output (all scalars go to run.summary, NOT run.log):
      • diag/bucket_summary       — Table: easy/medium/hard counts, mean BE, mean loss
      • diag/per_prompt           — Table: prompt_idx, preview, jsd, fwd_kl, be, bucket
      • diag/{trained_loss}_vs_be — Scatter of trained loss vs BE (primary)
      • run.summary scalars       — spearman, pearson, sigma, mean for trained loss
    """
    _BE_EASY = BE_EASY_FRAC * args.L   # e.g. 0.75 × 8 = 6.0
    _BE_HARD = BE_HARD_FRAC * args.L   # e.g. 0.375 × 8 = 3.0

    # Resolve primary vs secondary divergence labels
    if trained_loss == "fwdkl":
        primary_label, secondary_label = "fwdKL", "JSD"
    else:
        primary_label, secondary_label = "JSD", "fwdKL"

    print("\n" + "=" * 78)
    print("  [diagnose] objective-vs-BE: per-prompt divergence vs block efficiency")
    print(f"             mode='{mode_name}'  L={args.L}  trained_loss={trained_loss}  "
          f"primary={primary_label}  BE thresholds: easy≥{_BE_EASY} hard<{_BE_HARD}")
    print("=" * 78)

    rows: list[tuple] = []   # (prompt_idx, preview, jsd, fkl, be, bucket)
    jsd_xs, fkl_xs, be_ys, valid_indices = [], [], [], []

    for i, prompt in enumerate(tqdm(prompts, desc="diagnose", ncols=80)):
        if i not in per_prompt_be:
            continue
        jsd, fkl = _prompt_divergence(p_model, q_model, tok, prompt,
                                      args.max_new_tokens, device)
        if jsd is None:
            continue
        be = per_prompt_be[i]
        jsd_xs.append(jsd); fkl_xs.append(fkl); be_ys.append(be)
        valid_indices.append(i)
        bucket = "easy" if be >= _BE_EASY else ("hard" if be < _BE_HARD else "medium")
        preview = prompt[:60].replace("\n", " ")
        rows.append((i, preview, round(jsd, 5), round(fkl, 5), round(be, 4), bucket))

    # Assign primary/secondary arrays based on trained_loss
    pri_xs  = fkl_xs if trained_loss == "fwdkl" else jsd_xs
    sec_xs  = jsd_xs if trained_loss == "fwdkl" else fkl_xs

    r_pri   = _pearson(pri_xs, be_ys)
    r_sec   = _pearson(sec_xs, be_ys)
    rho_pri = _spearman(pri_xs, be_ys)
    rho_sec = _spearman(sec_xs, be_ys)
    n = len(be_ys)
    p_pri     = _pval_pearson(r_pri, n)
    p_rho_pri = _pval_pearson(rho_pri, n)

    def _std(xs):
        if len(xs) < 2:
            return float("nan")
        mu = sum(xs) / len(xs)
        return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    std_pri = _std(pri_xs)
    std_sec = _std(sec_xs)
    std_be  = _std(be_ys)
    mean_pri = sum(pri_xs) / n if n else float("nan")
    mean_be_all = sum(be_ys) / n if n else float("nan")

    # ── Bucket stats — mean of PRIMARY loss per bucket ───────────────────────
    # row layout: (prompt_idx, preview, jsd, fkl, be, bucket)
    pri_col = 3 if trained_loss == "fwdkl" else 2   # index into row tuple
    bucket_stats: dict[str, dict] = {}
    for bkt in ("easy", "medium", "hard"):
        pts = [(r[pri_col], r[4]) for r in rows if r[5] == bkt]   # (pri_loss, be)
        if pts:
            bucket_stats[bkt] = dict(
                count=len(pts),
                mean_be=sum(b for _, b in pts) / len(pts),
                mean_pri=sum(j for j, _ in pts) / len(pts),
                pct=100.0 * len(pts) / n,
            )
        else:
            bucket_stats[bkt] = dict(count=0, mean_be=float("nan"),
                                     mean_pri=float("nan"), pct=0.0)

    # ── Terminal output ─────────────────────────────────────────────────────
    pl = primary_label    # short alias
    sl = secondary_label
    print(f"\n  n={n}  mean_{pl}={mean_pri:.4f}  σ({pl})={std_pri:.4f}  "
          f"mean_BE={mean_be_all:.3f}  σ(BE)={std_be:.3f}")
    print(f"  Spearman ρ({pl},BE)={rho_pri:+.3f}  p≈{p_rho_pri:.1e}  "
          f"ρ({sl},BE)={rho_sec:+.3f}  [primary — rank-stable]")
    print(f"  Pearson  r({pl},BE)={r_pri:+.3f}  p≈{p_pri:.1e}  R²={r_pri**2:.2f}  "
          f"r({sl},BE)={r_sec:+.3f}  [secondary — shrinks if σ({pl}) collapses]")
    print(f"  → if σ({pl}) shrinks + ρ stable = range restriction (model uniformly better)")
    print(f"  → if σ({pl}) stable  + ρ drops  = true signal loss (objective decoupled from BE)")
    print(f"\n  {'Bucket':<8} {'N':>5} {'%':>5}  {'Mean-BE':>8}  {f'Mean-{pl}':>11}")
    print(f"  {'-'*47}")
    for bkt in ("easy", "medium", "hard"):
        s = bucket_stats[bkt]
        if s["count"] > 0:
            print(f"  {bkt:<8} {s['count']:>5} {s['pct']:>4.0f}%  "
                  f"{s['mean_be']:>8.3f}  {s['mean_pri']:>11.4f}")
        else:
            print(f"  {bkt:<8} {0:>5} {0:>4.0f}%  {'--':>8}  {'--':>9}")

    # ── Verdict — based on Spearman rho (primary) + Pearson r (secondary) ───
    # Spearman is the primary signal: it measures rank order agreement and is
    # invariant to JSD range compression (which shrinks Pearson r as the model
    # improves, even when the underlying relationship is unchanged).
    if rho_pri != rho_pri:
        h0 = "INCONCLUSIVE"
        verdict = "insufficient data (need >=2 prompts with finite divergence values)."
    elif abs(rho_pri) < 0.15:
        h0 = "NOT RULED OUT"
        verdict = (f"H0 NOT RULED OUT (rho={rho_pri:+.3f}, p≈{p_rho_pri:.1e}, "
                   f"trained_loss={trained_loss}): {pl} rank does not predict BE rank "
                   f"— minimising {pl} may not move BE.")
    elif rho_pri < 0:
        h0 = "RULED OUT"
        verdict = (
            f"H0 RULED OUT (rho={rho_pri:+.3f}, p≈{p_rho_pri:.1e}, trained_loss={trained_loss}): "
            f"lower {pl} rank predicts higher BE rank. The objective IS connected to BE. "
            f"Note: Pearson r={r_pri:+.3f} may be weaker than rho if σ({pl}) is "
            f"compressed (better model → narrower spread → smaller r, same signal)."
        )
    else:
        h0 = "UNEXPECTED"
        verdict = (f"positive {pl}-BE rank correlation (rho={rho_pri:+.3f}) — unexpected; "
                   f"verify orientation before trusting this result.")
    print(f"\n  [H0] {h0}")
    print(f"  [verdict] {verdict}")

    # ── W&B logging ─────────────────────────────────────────────────────────
    try:
        import wandb
        # Run name: include training-run directory (parent of ckpt_best/ckpt_latest)
        # so runs from different training jobs are distinguishable in W&B.
        # e.g. "diag_jsd_mathhard_s123_ckpt_best_math_eval_nss"
        _ckpt = args.checkpoint.rstrip("/")
        _ckpt_leaf   = os.path.basename(_ckpt)                   # ckpt_best
        _ckpt_parent = os.path.basename(os.path.dirname(_ckpt))  # jsd_mathhard_s123
        _run_name = f"diag_{_ckpt_parent}_{_ckpt_leaf}_{args.dataset}_{mode_name}"

        run = wandb.init(
            project="distillspec-pipeline",
            name=_run_name,
            job_type="diagnose",
            config={"checkpoint": args.checkpoint, "dataset": args.dataset,
                    "mode": mode_name, "K": args.K, "L": args.L, "n": n,
                    "trained_loss": trained_loss},
        )

        # Per-prompt table (full 6 columns — filter by bucket='hard' to find struggling prompts).
        pp_table = wandb.Table(
            columns=["prompt_idx", "prompt_preview", "jsd", "fwd_kl", "block_eff", "bucket"])
        for row in rows:
            pp_table.add_data(*row)

        # Minimal 2-column table for the scatter plot only.
        # Passing pp_table directly to wandb.plot.scatter causes W&B to log it a
        # second time as diag/{key}_table, duplicating the per_prompt table.
        scatter_col = "fwd_kl" if trained_loss == "fwdkl" else "jsd"
        scatter_table = wandb.Table(columns=[scatter_col, "block_eff"])
        for row in rows:
            scatter_table.add_data(row[3] if trained_loss == "fwdkl" else row[2], row[4])

        # Bucket summary: mean of PRIMARY loss per bucket.
        bkt_table = wandb.Table(
            columns=["bucket", "n", "pct", "mean_be", f"mean_{trained_loss}",
                     "be_thresh_lo", "be_thresh_hi"])
        for bkt, lo, hi in [("easy", _BE_EASY, None), ("medium", _BE_HARD, _BE_EASY),
                             ("hard", None, _BE_HARD)]:
            s = bucket_stats[bkt]
            bkt_table.add_data(
                bkt, s["count"], round(s["pct"], 1),
                round(s["mean_be"], 3) if s["count"] else None,
                round(s["mean_pri"], 4) if s["count"] else None,
                lo, hi,
            )

        # All scalars → run.summary so they appear as numbers in Overview, not charts.
        run.summary.update({
            "diag/n":                    n,
            "diag/trained_loss":         trained_loss,
            # ── primary: trained loss ────────────────────────────────────────
            f"diag/spearman_{trained_loss}":   round(rho_pri, 4),
            f"diag/p_spearman_{trained_loss}": round(p_rho_pri, 6),
            f"diag/pearson_{trained_loss}":    round(r_pri, 4),
            f"diag/p_pearson_{trained_loss}":  round(p_pri, 6),
            f"diag/r2_{trained_loss}":         round(r_pri ** 2, 4),
            f"diag/mean_{trained_loss}":       round(mean_pri, 5),
            f"diag/std_{trained_loss}":        round(std_pri, 5),
            # ── secondary: other divergence ──────────────────────────────────
            f"diag/spearman_{secondary_label.lower()}": round(rho_sec, 4),
            f"diag/pearson_{secondary_label.lower()}":  round(r_sec, 4),
            f"diag/std_{secondary_label.lower()}":      round(std_sec, 5),
            # ── BE spread ────────────────────────────────────────────────────
            "diag/mean_be":              round(mean_be_all, 4),
            "diag/std_be":               round(std_be, 4),
            # ── verdict ──────────────────────────────────────────────────────
            "diag/h0":                   h0,
            "diag/verdict":              verdict,
            "diag/n_easy":               bucket_stats["easy"]["count"],
            "diag/n_medium":             bucket_stats["medium"]["count"],
            "diag/n_hard":               bucket_stats["hard"]["count"],
        })

        run.log({
            f"diag/{trained_loss}_vs_be": wandb.plot.scatter(
                scatter_table, scatter_col, "block_eff",
                title=f"{pl} vs BE  ρ={rho_pri:+.3f} p≈{p_rho_pri:.0e}  [H0: {h0}]"),
            "diag/per_prompt":     pp_table,
            "diag/bucket_summary": bkt_table,
        })
        run.finish()
        print(f"  [diagnose] {_run_name} → W&B")
    except Exception as e:
        print(f"  [diagnose] W&B logging skipped ({e}); values printed above.")

    return jsd_xs, fkl_xs, be_ys, valid_indices


# ═══════════════════════════════════════════════════════════════════════════
#  Decomposition stacked-bar + radar  (--diagnose --baseline_checkpoint)
#  Requires: pip install matplotlib
# ═══════════════════════════════════════════════════════════════════════════

def run_decomposition_radar(
    p_model, tok, prompts,
    jsd_trained, be_ys, valid_indices,
    args, mode_name, device,
):
    """Decomposition stacked-bar + radar chart.

    Requires --baseline_checkpoint (typically Qwen/Qwen3-0.6B, the untrained
    draft).  Loads the base draft, computes G₀ (baseline JSD per prompt), then:

    1. Stacked-bar: for each prompt (sorted by G₀), shows learned = G₀ − G
       (green) stacked on remaining = G (red), with the G₀ step-line overlaid.
       A second panel plots per-prompt BE so difficulty ↔ BE is readable at a
       glance.  Answers: did training close the gap uniformly or only on easy
       prompts?

    2. Radar: splits prompts into 4 quartiles by G₀ (baseline difficulty), plots
       mean trained-BE in each quartile, with an overall-mean reference ring.
       Answers: where did training help most — on easy or hard prompts?

    Both figures are logged as W&B images in a 'decompose' run.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [decompose] matplotlib not installed — skipping decomposition plots")
        return

    from transformers import AutoModelForCausalLM

    _dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = _dtype_map.get(DEFAULT_DTYPE, torch.bfloat16)

    # ── 1. Load baseline (untrained) draft ──────────────────────────────────
    print(f"\n  [decompose] loading base draft: {args.baseline_checkpoint}")
    try:
        q_base = AutoModelForCausalLM.from_pretrained(
            args.baseline_checkpoint, torch_dtype=torch_dtype
        ).to(device).eval()
    except Exception as e:
        print(f"  [decompose] could not load baseline ({e}) — skipping")
        return

    # ── 2. Compute G₀ per prompt ────────────────────────────────────────────
    g0_list = []
    for idx in tqdm(valid_indices, desc="decompose-baseline", ncols=80):
        jsd0, _ = _prompt_divergence(p_model, q_base, tok, prompts[idx],
                                     args.max_new_tokens, device)
        g0_list.append(jsd0 if jsd0 is not None else float("nan"))

    del q_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 3. Align and filter ─────────────────────────────────────────────────
    g0s = np.array(g0_list, dtype=float)
    gs  = np.array(jsd_trained, dtype=float)
    bes = np.array(be_ys, dtype=float)

    valid = np.isfinite(g0s) & np.isfinite(gs) & np.isfinite(bes)
    g0s, gs, bes = g0s[valid], gs[valid], bes[valid]
    n = len(g0s)
    if n < 4:
        print(f"  [decompose] only {n} valid prompts after filtering — skipping plots")
        return

    learned   = np.clip(g0s - gs, 0, None)  # positive = closed gap; clipped so regressions don't go negative
    remaining = gs

    frac_learned = float(learned.mean() / g0s.mean()) if g0s.mean() > 0 else 0.0

    # Sort by G₀ (easiest → hardest) for the bar chart
    order = np.argsort(g0s)
    g0s_s  = g0s[order];  gs_s  = gs[order]
    learned_s = learned[order];  remaining_s = remaining[order];  bes_s = bes[order]

    # ── 4. Stacked-bar figure ───────────────────────────────────────────────
    bar_w = max(10, n // 5)
    fig_bar, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(bar_w, 6),
        gridspec_kw={"height_ratios": [2, 1]}, sharex=True,
    )
    x = np.arange(n)
    ax1.bar(x, remaining_s, label="Remaining JSD (trained)",  color="#F44336", width=1.0)
    ax1.bar(x, learned_s,  bottom=remaining_s,
            label="Gap closed (learned)", color="#4CAF50", width=1.0)
    ax1.step(x, g0s_s, where="mid", color="black", linewidth=1.5,
             label="Baseline JSD (G₀)")
    ax1.set_ylabel("JSD")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title(
        f"Decomposition: {n} prompts — "
        f"mean_learned={learned.mean():.3f}  mean_remaining={remaining.mean():.3f}  "
        f"frac_closed={frac_learned:.1%}"
    )

    ax2.bar(x, bes_s, color="steelblue", width=1.0)
    ax2.set_ylabel("Block Eff")
    ax2.set_xlabel("Prompt index (sorted by baseline JSD, easiest → hardest)")
    ax2.axhline(bes.mean(), color="gray", linewidth=1, linestyle="--",
                label=f"mean={bes.mean():.3f}")
    ax2.legend(fontsize=7)
    fig_bar.tight_layout()

    # ── 5. Radar figure ────────────────────────────────────────────────────
    q_cuts = np.percentile(g0s, [25, 50, 75])
    labels = [
        f"Easy\n(G₀≤{q_cuts[0]:.2f})",
        f"Med-easy\n(≤{q_cuts[1]:.2f})",
        f"Med-hard\n(≤{q_cuts[2]:.2f})",
        f"Hard\n(>{q_cuts[2]:.2f})",
    ]
    masks = [
        g0s <= q_cuts[0],
        (g0s > q_cuts[0]) & (g0s <= q_cuts[1]),
        (g0s > q_cuts[1]) & (g0s <= q_cuts[2]),
        g0s > q_cuts[2],
    ]
    mean_bes = [float(bes[m].mean()) if m.any() else 0.0 for m in masks]
    overall_be = float(bes.mean())

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    values = mean_bes + mean_bes[:1]

    fig_radar, ax_r = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
    ax_r.plot(angles, values, "o-", linewidth=2, color="#1976D2", label="Trained")
    ax_r.fill(angles, values, alpha=0.25, color="#1976D2")
    ax_r.plot(angles, [overall_be] * (N + 1), "--", linewidth=1,
              color="gray", label=f"Overall mean ({overall_be:.3f})")
    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(
        [f"{l}\n{v:.3f}" for l, v in zip(labels, mean_bes)], fontsize=7,
    )
    ax_r.set_title("BE by difficulty quartile (baseline JSD)", pad=15)
    ax_r.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)
    fig_radar.tight_layout()

    # Print console summary
    print(f"  [decompose] mean_learned={learned.mean():.4f}  "
          f"mean_remaining={remaining.mean():.4f}  "
          f"frac_closed={frac_learned:.1%}")
    print("  [decompose] BE by quartile: " +
          " | ".join(f"{l.split(chr(10))[0]}: {v:.3f}"
                     for l, v in zip(labels, mean_bes)))

    # ── 6. Log to W&B ──────────────────────────────────────────────────────
    try:
        import wandb
        run = wandb.init(
            project="distillspec-pipeline",
            name=(f"decomp_{os.path.basename(args.checkpoint.rstrip('/'))}"
                  f"_{args.dataset}_{mode_name}"),
            job_type="decompose",
            config={
                "checkpoint":          args.checkpoint,
                "baseline_checkpoint": args.baseline_checkpoint,
                "dataset":             args.dataset,
                "mode":                mode_name,
                "n_prompts":           n,
                "mean_learned_jsd":    float(learned.mean()),
                "mean_remaining_jsd":  float(remaining.mean()),
                "mean_g0_jsd":         float(g0s.mean()),
                "frac_closed":         frac_learned,
            },
        )
        log_dict = {
            "decompose/stacked_bar":    wandb.Image(fig_bar),
            "decompose/radar":          wandb.Image(fig_radar),
            "decompose/mean_learned":   float(learned.mean()),
            "decompose/mean_remaining": float(remaining.mean()),
            "decompose/frac_closed":    frac_learned,
        }
        for lbl, val in zip(["q1_easy", "q2_med_easy", "q3_med_hard", "q4_hard"], mean_bes):
            log_dict[f"decompose/be_{lbl}"] = val
        run.log(log_dict)
        run.finish()
        print("  [decompose] stacked-bar + radar logged to W&B (job_type='decompose')")
    except Exception as e:
        print(f"  [decompose] W&B logging skipped ({e}); figures printed above.")
    finally:
        plt.close(fig_bar)
        plt.close(fig_radar)