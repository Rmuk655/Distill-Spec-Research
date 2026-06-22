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


def run_objective_be_diagnostic(p_model, q_model, tok, prompts, per_prompt_be,
                                args, mode_name, device):
    """Correlate per-prompt training divergence with per-prompt block efficiency.

    Flat (|r| ~ 0) correlation  → objective mismatch (H0): minimising JSD/KL
                                   will not move BE.
    Negative correlation        → lower divergence ⇒ higher BE (expected): the
                                   objective IS the right lever.

    W&B output (all scalars go to run.summary, NOT run.log, so they appear in
    the Overview tab as numbers — not as single-point line charts):
      • diag/bucket_summary  — Table: easy/medium/hard counts, mean BE, mean JSD
      • diag/per_prompt      — Table: one row per prompt with prompt_idx,
                               prompt_preview, jsd, fwd_kl, block_eff, bucket
                               (filter by bucket='hard' to find struggling prompts)
      • diag/jsd_vs_be       — Scatter plot against JSD only (the trained loss)
      • run.summary scalars  — n, corr_jsd_be, p_jsd, r2_jsd, h0, verdict, counts
    """
    _BE_EASY = BE_EASY_FRAC * args.L   # e.g. 0.75 × 8 = 6.0
    _BE_HARD = BE_HARD_FRAC * args.L   # e.g. 0.375 × 8 = 3.0

    print("\n" + "=" * 78)
    print("  [diagnose] objective-vs-BE: per-prompt divergence vs block efficiency")
    print(f"             mode='{mode_name}'  L={args.L}  BE thresholds: easy≥{_BE_EASY} hard<{_BE_HARD}")
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

    r_jsd = _pearson(jsd_xs, be_ys)
    r_fkl = _pearson(fkl_xs, be_ys)
    n = len(be_ys)
    p_jsd = _pval_pearson(r_jsd, n)

    # ── Bucket stats ────────────────────────────────────────────────────────
    bucket_stats: dict[str, dict] = {}
    for bkt in ("easy", "medium", "hard"):
        pts = [(r[2], r[4]) for r in rows if r[5] == bkt]   # (jsd, be)
        if pts:
            bucket_stats[bkt] = dict(
                count=len(pts),
                mean_be=sum(b for _, b in pts) / len(pts),
                mean_jsd=sum(j for j, _ in pts) / len(pts),
                pct=100.0 * len(pts) / n,
            )
        else:
            bucket_stats[bkt] = dict(count=0, mean_be=float("nan"),
                                     mean_jsd=float("nan"), pct=0.0)

    # ── Terminal output ─────────────────────────────────────────────────────
    print(f"\n  n={n}  corr(JSD,BE)={r_jsd:+.3f}  p≈{p_jsd:.1e}  "
          f"R²={r_jsd**2:.2f}  corr(fwdKL,BE)={r_fkl:+.3f}")
    print(f"\n  {'Bucket':<8} {'N':>5} {'%':>5}  {'Mean-BE':>8}  {'Mean-JSD':>9}")
    print(f"  {'-'*44}")
    for bkt in ("easy", "medium", "hard"):
        s = bucket_stats[bkt]
        if s["count"] > 0:
            print(f"  {bkt:<8} {s['count']:>5} {s['pct']:>4.0f}%  "
                  f"{s['mean_be']:>8.3f}  {s['mean_jsd']:>9.4f}")
        else:
            print(f"  {bkt:<8} {0:>5} {0:>4.0f}%  {'--':>8}  {'--':>9}")

    # ── Verdict ─────────────────────────────────────────────────────────────
    if r_jsd != r_jsd:
        h0 = "INCONCLUSIVE"
        verdict = "insufficient data (need >=2 prompts with finite divergence values)."
    elif abs(r_jsd) < 0.15:
        h0 = "NOT RULED OUT"
        verdict = (f"H0 NOT RULED OUT (r={r_jsd:+.3f}, p≈{p_jsd:.1e}): "
                   f"JSD does not predict BE — minimising JSD/KL may not move BE.")
    elif r_jsd < 0:
        h0 = "RULED OUT"
        verdict = (
            f"H0 RULED OUT (r={r_jsd:+.3f}, p≈{p_jsd:.1e}, R²={r_jsd**2:.2f}): "
            f"lower JSD predicts higher BE. The objective IS connected to BE. "
            f"JSD explains {r_jsd**2*100:.0f}% of BE variance; remaining "
            f"{(1-r_jsd**2)*100:.0f}% = teacher-entropy noise (high JSD but both "
            f"models uncertain, tokens still accepted) + capacity ceiling on hard prompts."
        )
    else:
        h0 = "UNEXPECTED"
        verdict = (f"positive JSD-BE correlation (r={r_jsd:+.3f}) — unexpected; "
                   f"verify orientation before trusting this result.")
    print(f"\n  [H0] {h0}")
    print(f"  [verdict] {verdict}")

    # ── W&B logging ─────────────────────────────────────────────────────────
    try:
        import wandb
        run = wandb.init(
            project="distillspec-pipeline",
            name=(f"diag_{os.path.basename(args.checkpoint.rstrip('/'))}"
                  f"_{args.dataset}_{mode_name}"),
            job_type="diagnose",
            config={"checkpoint": args.checkpoint, "dataset": args.dataset,
                    "mode": mode_name, "K": args.K, "L": args.L, "n": n},
        )

        # Per-prompt table: filter by bucket='hard' in W&B UI to see struggling prompts.
        pp_table = wandb.Table(
            columns=["prompt_idx", "prompt_preview", "jsd", "fwd_kl", "block_eff", "bucket"])
        for row in rows:
            pp_table.add_data(*row)

        # Bucket summary table: replaces the confusing scalar line charts for counts.
        bkt_table = wandb.Table(
            columns=["bucket", "n", "pct", "mean_be", "mean_jsd",
                     "be_thresh_lo", "be_thresh_hi"])
        for bkt, lo, hi in [("easy", _BE_EASY, None), ("medium", _BE_HARD, _BE_EASY),
                             ("hard", None, _BE_HARD)]:
            s = bucket_stats[bkt]
            bkt_table.add_data(
                bkt, s["count"], round(s["pct"], 1),
                round(s["mean_be"], 3) if s["count"] else None,
                round(s["mean_jsd"], 4) if s["count"] else None,
                lo, hi,
            )

        # Scalars → run.summary (not run.log) so they show as numbers in the
        # Overview tab, NOT as single-point line charts.
        run.summary.update({
            "diag/n":          n,
            "diag/corr_jsd":   round(r_jsd, 4),
            "diag/p_jsd":      round(p_jsd, 6),
            "diag/r2_jsd":     round(r_jsd ** 2, 4),
            "diag/corr_fkl":   round(r_fkl, 4),
            "diag/h0":         h0,
            "diag/verdict":    verdict,
            "diag/n_easy":     bucket_stats["easy"]["count"],
            "diag/n_medium":   bucket_stats["medium"]["count"],
            "diag/n_hard":     bucket_stats["hard"]["count"],
        })

        # JSD scatter only — fwdKL scatter omitted; fwdKL is not the trained loss
        # and its range (0–0.7+) makes it hard to read next to JSD (0–0.12).
        run.log({
            "diag/jsd_vs_be":      wandb.plot.scatter(
                pp_table, "jsd", "block_eff",
                title=f"JSD vs BE  r={r_jsd:+.3f} p≈{p_jsd:.0e}  [H0: {h0}]"),
            "diag/per_prompt":     pp_table,
            "diag/bucket_summary": bkt_table,
        })
        run.finish()
        print("  [diagnose] per-prompt table + bucket summary + JSD scatter → W&B")
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