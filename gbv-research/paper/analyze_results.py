"""
analyze_results.py — paper-style statistical analysis of results.db.

Produces:
  - Significance-tested comparison tables (t-test, Cohen's d)
  - Best configuration recommendations
  - Convergence analysis
  - Quality regression flags
  - Dataset robustness analysis
  - Ready-to-paste Markdown tables for DELIVERABLES.md

Usage:
    python analyze_results.py                    # full analysis, prints to stdout
    python analyze_results.py --out report.md    # save Markdown report
    python analyze_results.py --section alpha    # only alpha section
    python analyze_results.py --baseline baseline --compare kl-gsm8k,ebe-gsm8k
"""

import sys, os, json, argparse
# Ensure UTF-8 output on Windows (cp1252 terminal can't print special chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from collections import defaultdict
from typing import Optional
import math

# results_db.py lives in gbv-research/db/, not in paper/.
# Resolve the path relative to this file so the script runs from any cwd.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "db"))
import results_db

# ---------------------------------------------------------------------------
# Stats helpers (no scipy required — pure Python)
# ---------------------------------------------------------------------------

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m)**2 for x in xs) / (len(xs) - 1))

def ci95(xs):
    if not xs:
        return float("nan")
    return 1.96 * std(xs) / math.sqrt(len(xs))

def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_std = math.sqrt((std(a)**2 + std(b)**2) / 2)
    return (mean(a) - mean(b)) / pooled_std if pooled_std > 0 else 0.0

def welch_t_pvalue(a, b):
    """Two-sample Welch t-test, returns p-value approximation."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = mean(a), mean(b)
    va, vb = std(a)**2 / len(a), std(b)**2 / len(b)
    se = math.sqrt(va + vb)
    if se == 0:
        return 1.0
    t = (ma - mb) / se
    # Welch-Satterthwaite degrees of freedom
    df = (va + vb)**2 / (va**2 / max(len(a)-1, 1) + vb**2 / max(len(b)-1, 1))
    # Approximate p-value via t-distribution CDF (Abramowitz & Stegun)
    return _t_pvalue(abs(t), df)

def _t_pvalue(t, df):
    """Approximate two-tailed p-value for t-distribution."""
    # Simple approximation using normal for large df
    if df > 30:
        z = t
        p = 2 * (1 - _norm_cdf(z))
    else:
        # Use beta function approximation
        x = df / (df + t*t)
        p = _regularized_incomplete_beta(df/2, 0.5, x)
    return max(0.0, min(1.0, p))

def _norm_cdf(z):
    """Standard normal CDF approximation."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

def _regularized_incomplete_beta(a, b, x):
    """Very rough approximation — good enough for p < 0.05 detection."""
    try:
        import math
        # Use continued fraction approximation
        if x <= 0: return 0.0
        if x >= 1: return 1.0
        lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a+b)
        return math.exp(a * math.log(x) + b * math.log(1-x) - lbeta) / (a * (1 + 0.1))
    except:
        return 0.5

def significance_label(p):
    if math.isnan(p): return "n/a"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def effect_label(d):
    if math.isnan(d): return "n/a"
    d = abs(d)
    if d >= 0.8: return "large"
    if d >= 0.5: return "medium"
    if d >= 0.2: return "small"
    return "trivial"

def pct_change(new, old):
    if old == 0: return float("nan")
    return 100 * (new - old) / old

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_alpha_runs(filters=None):
    runs = results_db.query_runs({"mode": "alpha"})
    if filters:
        for k, v in filters.items():
            runs = [r for r in runs if str(r.get(k)) == str(v)]
    return runs

def load_be_runs(mode=None):
    all_runs = results_db.query_runs()
    be_modes = {"specinfer", "gbv", "traversal", "bv", "nss"}
    runs = [r for r in all_runs if r.get("mode") in be_modes and r.get("block_eff") is not None]
    if mode:
        runs = [r for r in runs if r["mode"] == mode]
    return runs

def group_by(runs, key):
    g = defaultdict(list)
    for r in runs:
        g[r.get(key)].append(r)
    return dict(g)

# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def md_table(headers, rows, alignments=None):
    if not rows:
        return "_No data_\n"
    col_w = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    aligns = alignments or ["l"] + ["r"] * (len(headers) - 1)

    def fmt_sep(w, a):
        if a == "r": return "-" * (w-1) + ":"
        if a == "c": return ":" + "-" * (w-2) + ":"
        return "-" * w

    lines = []
    lines.append("| " + " | ".join(h.ljust(col_w[i]) for i, h in enumerate(headers)) + " |")
    lines.append("| " + " | ".join(fmt_sep(col_w[i], aligns[i]) for i in range(len(headers))) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(v).ljust(col_w[i]) if aligns[i]=="l" else str(v).rjust(col_w[i])
                                         for i, v in enumerate(row)) + " |")
    return "\n".join(lines) + "\n"

def section(title, level=2):
    return f"\n{'#' * level} {title}\n"

# ---------------------------------------------------------------------------
# Analysis sections
# ---------------------------------------------------------------------------

def analyze_alpha(baseline_label="baseline", compare_labels=None, out=None):
    runs = load_alpha_runs()
    if not runs:
        return "_No alpha results in DB yet._\n"

    all_labels = sorted(set(r["draft_label"] for r in runs))
    if compare_labels is None:
        compare_labels = [l for l in all_labels if l != baseline_label]

    buf = []
    buf.append(section("Alpha (Token Acceptance Rate) Analysis"))

    # Overall summary table
    buf.append("### Overall Alpha by Condition\n")
    headers = ["Draft", "Dataset", "n", "Alpha", "Std", "95% CI", "Throughput (tok/s)", "ms/tok"]
    rows = []
    for r in sorted(runs, key=lambda x: (x["draft_label"], x["dataset"])):
        if r.get("alpha_mean") is None:
            continue
        rows.append([
            r["draft_label"], r["dataset"], r["n_prompts"],
            f"{r['alpha_mean']:.4f}", f"{r.get('alpha_std',0):.4f}",
            f"±{r.get('alpha_ci95',0):.4f}",
            f"{r.get('throughput',0):.2f}" if r.get("throughput") else "—",
            f"{r.get('ms_per_tok',0):.0f}" if r.get("ms_per_tok") else "—",
        ])
    buf.append(md_table(headers, rows))

    # Pairwise significance tests vs baseline
    buf.append("### Significance Tests vs Baseline\n")
    base_runs = [r for r in runs if r["draft_label"] == baseline_label]
    headers2 = ["Compare", "Dataset", "Delta Alpha", "%Change", "Cohen's d", "p-value", "Sig", "Effect"]
    sig_rows = []
    for label in compare_labels:
        cmp_runs = [r for r in runs if r["draft_label"] == label]
        datasets = set(r["dataset"] for r in base_runs) & set(r["dataset"] for r in cmp_runs)
        for ds in sorted(datasets):
            b_vals = [r["alpha_mean"] for r in base_runs if r["dataset"] == ds and r.get("alpha_mean")]
            c_vals = [r["alpha_mean"] for r in cmp_runs if r["dataset"] == ds and r.get("alpha_mean")]
            if not b_vals or not c_vals:
                continue
            bm, cm = mean(b_vals), mean(c_vals)
            d = cohens_d(c_vals, b_vals)
            p = welch_t_pvalue(c_vals, b_vals)
            sig_rows.append([
                f"{baseline_label} -> {label}", ds,
                f"{cm-bm:+.4f}", f"{pct_change(cm,bm):+.1f}%",
                f"{d:.3f}", f"{p:.4f}" if not math.isnan(p) else "n/a",
                significance_label(p), effect_label(d),
            ])
    buf.append(md_table(headers2, sig_rows))

    # Per-category breakdown (if per_prompt data available)
    all_pp = []
    for r in runs:
        if r.get("id"):
            pp = results_db.query_per_prompt(r["id"])
            for row in pp:
                row["draft_label"] = r["draft_label"]
                row["dataset"] = r["dataset"]
            all_pp.extend(pp)

    if all_pp:
        buf.append("### Per-Category Alpha Breakdown\n")
        cats = sorted(set(r.get("category") for r in all_pp if r.get("category")))
        headers3 = ["Draft"] + cats + ["Overall"]
        cat_rows = []
        for label in all_labels:
            pp_label = [r for r in all_pp if r["draft_label"] == label]
            row = [label]
            for cat in cats:
                vals = [r["alpha"] for r in pp_label if r.get("category") == cat and r.get("alpha") is not None]
                row.append(f"{mean(vals):.3f}" if vals else "—")
            all_vals = [r["alpha"] for r in pp_label if r.get("alpha") is not None]
            row.append(f"{mean(all_vals):.3f}" if all_vals else "—")
            cat_rows.append(row)
        buf.append(md_table(headers3, cat_rows))

    return "\n".join(buf)


def analyze_block_efficiency(baseline_label="baseline", compare_labels=None, out=None):
    runs = load_be_runs()
    if not runs:
        return "_No block efficiency results in DB yet._\n"

    all_labels = sorted(set(r["draft_label"] for r in runs if r["draft_label"] != "same_model"))
    if compare_labels is None:
        compare_labels = [l for l in all_labels if l != baseline_label]

    buf = []
    buf.append(section("Block Efficiency Analysis"))

    # Full matrix table: mode x K x draft
    buf.append("### Block Efficiency by Mode, K, and Draft\n")
    modes = sorted(set(r["mode"] for r in runs))
    Ks = sorted(set(r["K"] for r in runs))
    headers = ["Mode", "K"] + all_labels + ["Best (EBE vs BASE %)"]
    be_rows = []
    for mode in modes:
        for K in Ks:
            row = [mode, K]
            vals_by_label = {}
            for label in all_labels:
                matching = [r["block_eff"] for r in runs
                            if r["mode"] == mode and r["K"] == K and r["draft_label"] == label]
                v = mean(matching) if matching else None
                vals_by_label[label] = v
                row.append(f"{v:.4f}" if v else "—")
            # improvement col
            base_v = vals_by_label.get(baseline_label)
            best_v = max((v for v in vals_by_label.values() if v is not None), default=None)
            if base_v and best_v:
                row.append(f"{pct_change(best_v, base_v):+.1f}%")
            else:
                row.append("—")
            be_rows.append(row)
    buf.append(md_table(headers, be_rows))

    # K sensitivity: how does BE change with K for each draft x mode?
    buf.append("### K Sensitivity (Block Efficiency vs K)\n")
    for mode in modes:
        mode_runs = [r for r in runs if r["mode"] == mode]
        if not mode_runs:
            continue
        buf.append(f"**Mode: {mode}**\n")
        headers_k = ["Draft"] + [f"K={k}" for k in Ks] + ["Best K", "Peak BE"]
        k_rows = []
        for label in all_labels:
            label_runs = [r for r in mode_runs if r["draft_label"] == label]
            row = [label]
            be_by_k = {}
            for K in Ks:
                vals = [r["block_eff"] for r in label_runs if r["K"] == K]
                v = mean(vals) if vals else None
                be_by_k[K] = v
                row.append(f"{v:.4f}" if v else "—")
            best_k = max(be_by_k, key=lambda k: be_by_k[k] or 0)
            peak = be_by_k[best_k]
            row += [f"K={best_k}", f"{peak:.4f}" if peak else "—"]
            k_rows.append(row)
        buf.append(md_table(headers_k, k_rows))

    # EBE gain analysis — detect KL and EBE labels dynamically so the table
    # works with any naming convention (kl200, kl-gsm8k, etc.).
    buf.append("### EBE vs KL vs Baseline Gain Summary\n")
    kl_label  = next((l for l in compare_labels if "kl"  in l.lower()), None)
    ebe_label = next((l for l in compare_labels if "ebe" in l.lower()), None)
    if not kl_label or not ebe_label:
        buf.append(
            f"_Skipped: need at least one KL label and one EBE label in the compare set. "
            f"Pass --compare <kl-label>,<ebe-label> explicitly._\n"
        )
    else:
        gain_rows = []
        headers_g = ["Mode", "K", "Dataset",
                     f"KL ({kl_label}) vs BASE",
                     f"EBE ({ebe_label}) vs BASE",
                     "EBE vs KL", "Winner"]
        for mode in modes:
            for K in Ks:
                datasets = sorted(set(r["dataset"] for r in runs if r["mode"] == mode and r["K"] == K))
                for ds in datasets:
                    def _be(label):
                        vals = [r["block_eff"] for r in runs
                                if r["mode"]==mode and r["K"]==K and r["draft_label"]==label
                                and r["dataset"]==ds and r["block_eff"] is not None]
                        return mean(vals) if vals else None
                    base_be = _be(baseline_label)
                    kl_be   = _be(kl_label)
                    ebe_be  = _be(ebe_label)
                    if not base_be:
                        continue
                    kl_gain   = f"{pct_change(kl_be,  base_be):+.1f}%" if kl_be  else "—"
                    ebe_gain  = f"{pct_change(ebe_be, base_be):+.1f}%" if ebe_be else "—"
                    ebe_vs_kl = f"{pct_change(ebe_be, kl_be):+.1f}%"  if (ebe_be and kl_be) else "—"
                    vals = {k: v for k, v in
                            [("baseline", base_be), (kl_label, kl_be), (ebe_label, ebe_be)] if v}
                    winner = max(vals, key=vals.get) if vals else "—"
                    gain_rows.append([mode, K, ds, kl_gain, ebe_gain, ebe_vs_kl, winner])
        buf.append(md_table(headers_g, gain_rows))

    return "\n".join(buf)


def analyze_convergence():
    curves = results_db.query_train_curves()
    if not curves:
        return "_No training curve data in DB yet._\n"

    buf = []
    buf.append(section("Training Convergence Analysis"))

    from collections import defaultdict
    by_label = defaultdict(list)
    for row in curves:
        by_label[row["label"]].append(row)

    headers = ["Label", "Loss Name", "LR", "Step 50", "Step 100", "Step 150", "Step 200", "Reduction %", "Final AW"]
    rows = []
    for label, pts in sorted(by_label.items()):
        pts = sorted(pts, key=lambda x: x["step"])
        loss_name = pts[0].get("loss_name", "?")
        lr = pts[0].get("learning_rate", 0)
        step_losses = {p["step"]: p["loss"] for p in pts}
        aw_vals = [p["accept_weight"] for p in pts if p.get("accept_weight")]
        s50  = step_losses.get(50,  "—")
        s100 = step_losses.get(100, "—")
        s150 = step_losses.get(150, "—")
        s200 = step_losses.get(200, "—")
        first_loss = pts[0]["loss"] if pts else None
        last_loss  = pts[-1]["loss"] if pts else None
        reduction = pct_change(first_loss, last_loss) if (first_loss and last_loss) else float("nan")
        # reduction should be negative (loss going down)
        reduction = -reduction  # flip sign to show % reduction
        final_aw = f"{aw_vals[-1]:.4f}" if aw_vals else "—"
        rows.append([
            label, loss_name, f"{lr:.0e}" if lr else "—",
            f"{s50:.4f}" if isinstance(s50, float) else s50,
            f"{s100:.4f}" if isinstance(s100, float) else s100,
            f"{s150:.4f}" if isinstance(s150, float) else s150,
            f"{s200:.4f}" if isinstance(s200, float) else s200,
            f"{reduction:.1f}%" if not math.isnan(reduction) else "—",
            final_aw,
        ])
    buf.append(md_table(headers, rows))

    # LR recommendation
    buf.append("### Learning Rate Recommendation\n")
    lr_summary = []
    for label, pts in sorted(by_label.items()):
        # Include labels that are LR-ablation runs ("lr" in name) OR
        # standard KL / EBE training runs (any label containing "kl" or "ebe").
        if not any(x in label.lower() for x in ("lr", "kl", "ebe")):
            continue
        pts_sorted = sorted(pts, key=lambda x: x["step"])
        if not pts_sorted:
            continue
        final = pts_sorted[-1]
        aw_final = final.get("accept_weight")
        lr_summary.append((label, final["loss"], aw_final or 0, final.get("learning_rate", 0)))

    if lr_summary:
        # Best LR: lowest final loss with accept_weight > 0.965
        quality_ok = [(l, loss, aw, lr) for l, loss, aw, lr in lr_summary if aw >= 0.965]
        if quality_ok:
            best = min(quality_ok, key=lambda x: x[1])
            buf.append(f"**Recommended LR:** `{best[3]:.0e}` ({best[0]}) — "
                       f"final loss {best[1]:.4f}, accept_weight {best[2]:.4f}\n")
            buf.append(f"> accept_weight >= 0.965 threshold ensures the model hasn't collapsed "
                       f"to greedy teacher-copying. Lower loss with aw < 0.965 risks mode collapse.\n")

    return "\n".join(buf)


def analyze_quality_flags():
    buf = []
    buf.append(section("Quality & Sanity Checks"))

    alpha_runs = load_alpha_runs()
    be_runs = load_be_runs()

    flags = []

    # Flag 1: Any draft shows alpha BELOW baseline
    base_alpha = {r["dataset"]: r["alpha_mean"] for r in alpha_runs
                  if r["draft_label"] == "baseline" and r.get("alpha_mean")}
    for r in alpha_runs:
        if r["draft_label"] == "baseline" or not r.get("alpha_mean"):
            continue
        base = base_alpha.get(r["dataset"])
        if base and r["alpha_mean"] < base - 0.02:
            flags.append(f"REGRESSION: {r['draft_label']} alpha {r['alpha_mean']:.4f} < "
                         f"baseline {base:.4f} on {r['dataset']} (delta: {r['alpha_mean']-base:+.4f})")

    # Flag 2: BE at K=1 should be approximately = alpha * max_propose + 1
    for r in be_runs:
        if r["K"] != 1 or r["mode"] != "specinfer":
            continue
        label = r["draft_label"]
        alpha_run = next((a for a in alpha_runs
                          if a["draft_label"] == label and a["dataset"] == r["dataset"]), None)
        if alpha_run and alpha_run.get("alpha_mean"):
            # BE at K=1 ~ alpha_mean * L + 1 (rough), where L = max_propose
            expected_approx = alpha_run["alpha_mean"] * 5 + 1
            actual = r["block_eff"]
            if abs(actual - expected_approx) > 1.5:
                flags.append(f"BE/ALPHA MISMATCH: {label} on {r['dataset']} "
                             f"BE={actual:.3f} but alpha-implied ~{expected_approx:.3f} "
                             f"(diff={actual-expected_approx:+.3f}) — check alpha eval temperature vs BE temperature")

    # Flag 3: Block efficiency should increase from K=1 to K=3 for same mode
    by_label_mode_ds = defaultdict(list)
    for r in be_runs:
        by_label_mode_ds[(r["draft_label"], r["mode"], r["dataset"])].append(r)

    for (label, mode, ds), runs in by_label_mode_ds.items():
        by_K = {r["K"]: r["block_eff"] for r in runs if r.get("block_eff")}
        if 1 in by_K and 3 in by_K and by_K[1] > by_K[3] + 0.1:
            flags.append(f"UNEXPECTED: {label} {mode} on {ds}: BE(K=1)={by_K[1]:.3f} > BE(K=3)={by_K[3]:.3f} "
                         f"— more tree paths should generally help or be neutral")

    # Flag 4: accept_weight should decrease over training (model is improving)
    curves = results_db.query_train_curves()
    by_label = defaultdict(list)
    for c in curves:
        by_label[c["label"]].append(c)
    for label, pts in by_label.items():
        pts_sorted = sorted(pts, key=lambda x: x["step"])
        aw_vals = [p["accept_weight"] for p in pts_sorted if p.get("accept_weight")]
        if len(aw_vals) >= 2:
            if aw_vals[-1] > aw_vals[0] + 0.005:
                flags.append(f"TRAINING ANOMALY: {label} accept_weight INCREASED from "
                             f"{aw_vals[0]:.4f} to {aw_vals[-1]:.4f} — expected to decrease")

    if flags:
        buf.append("### Flags Found\n")
        for f in flags:
            buf.append(f"- {f}\n")
    else:
        buf.append("No quality flags raised. All checks passed.\n")

    buf.append("### Interpretation Guide\n")
    buf.append("""
- **alpha_mean ~ 0.5** for untrained draft is typical for 0.5B vs 0.6B model pair
- **BE(K=1) ~ alpha x L + 1** where L=max_propose_num (approx; temperature matters)
- **BE should increase with K** (more paths = more likely one is accepted)
- **accept_weight decreasing** = model is learning to produce more target-accepted tokens
- **Temperature mismatch**: alpha eval uses T=0.6, GBV eval uses T=1.0 — expect BE/alpha to differ
""")

    return "\n".join(buf)


def analyze_datasets():
    """Cross-dataset robustness analysis."""
    alpha_runs = load_alpha_runs()
    if not alpha_runs:
        return "_No alpha data across datasets yet._\n"

    buf = []
    buf.append(section("Dataset Robustness Analysis"))

    datasets = sorted(set(r["dataset"] for r in alpha_runs))
    labels = sorted(set(r["draft_label"] for r in alpha_runs))

    if len(datasets) < 2:
        buf.append("_Need results on multiple datasets for robustness analysis._\n")
        return "\n".join(buf)

    # Variance across datasets for each draft model
    buf.append("### Alpha Variance Across Datasets (robustness measure)\n")
    buf.append("_Low std = robust to data distribution. High std = sensitive to dataset._\n\n")
    headers = ["Draft"] + datasets + ["Mean", "Std", "CV (Std/Mean)"]
    rows = []
    for label in labels:
        label_runs = [r for r in alpha_runs if r["draft_label"] == label and r.get("alpha_mean")]
        row = [label]
        ds_vals = []
        for ds in datasets:
            ds_run = [r["alpha_mean"] for r in label_runs if r["dataset"] == ds]
            v = mean(ds_run) if ds_run else None
            row.append(f"{v:.3f}" if v else "—")
            if v:
                ds_vals.append(v)
        if ds_vals:
            m, s = mean(ds_vals), std(ds_vals)
            cv = s / m if m > 0 else float("nan")
            row += [f"{m:.3f}", f"{s:.3f}", f"{cv:.3f}"]
        else:
            row += ["—", "—", "—"]
        rows.append(row)
    buf.append(md_table(headers, rows))

    # Best performing dataset for each loss
    buf.append("### Best Dataset per Draft Model\n")
    for label in labels:
        label_runs = [r for r in alpha_runs if r["draft_label"] == label and r.get("alpha_mean")]
        if not label_runs:
            continue
        best = max(label_runs, key=lambda r: r["alpha_mean"])
        worst = min(label_runs, key=lambda r: r["alpha_mean"])
        buf.append(f"- **{label}**: best on `{best['dataset']}` (α={best['alpha_mean']:.4f}), "
                   f"worst on `{worst['dataset']}` (α={worst['alpha_mean']:.4f}), "
                   f"range={best['alpha_mean']-worst['alpha_mean']:.4f}\n")

    return "\n".join(buf)


def analyze_throughput():
    runs = load_alpha_runs()
    runs = [r for r in runs if r.get("throughput") and r["throughput"] > 0]
    if not runs:
        return "_No throughput data yet._\n"

    buf = []
    buf.append(section("Throughput & Latency Analysis"))

    headers = ["Draft", "Dataset", "tok/s", "ms/tok", "vs Baseline (tok/s)"]
    base_tp = {r["dataset"]: r["throughput"] for r in runs if r["draft_label"] == "baseline"}

    rows = []
    for r in sorted(runs, key=lambda x: (x["dataset"], x["draft_label"])):
        base = base_tp.get(r["dataset"])
        delta = f"{pct_change(r['throughput'], base):+.1f}%" if base else "—"
        rows.append([r["draft_label"], r["dataset"],
                     f"{r['throughput']:.2f}", f"{r['ms_per_tok']:.0f}", delta])
    buf.append(md_table(headers, rows))
    return "\n".join(buf)


def generate_full_report(baseline_label="baseline", compare_labels=None):
    buf = ["# SpecDist Analysis Report\n"]
    buf.append(f"*Generated from results.db — {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

    # Run counts
    all_runs = results_db.query_runs()
    buf.append(f"**Total runs in DB:** {len(all_runs)}  |  "
               f"**Alpha runs:** {sum(1 for r in all_runs if r['mode']=='alpha')}  |  "
               f"**BE runs:** {sum(1 for r in all_runs if r.get('block_eff') is not None)}\n")

    buf.append(analyze_alpha(baseline_label, compare_labels))
    buf.append(analyze_block_efficiency(baseline_label, compare_labels))
    buf.append(analyze_convergence())
    buf.append(analyze_quality_flags())
    buf.append(analyze_datasets())
    buf.append(analyze_throughput())

    buf.append(section("Recommendations"))
    buf.append(_generate_recommendations())

    return "\n".join(buf)


def _generate_recommendations():
    buf = []
    runs = results_db.query_runs()
    be_runs = [r for r in runs if r.get("block_eff") is not None]
    alpha_runs = [r for r in runs if r.get("alpha_mean") is not None and r["mode"] == "alpha"]

    if not runs:
        return "_Not enough data yet for recommendations._\n"

    # Best overall BE
    if be_runs:
        best_be = max(be_runs, key=lambda r: r["block_eff"])
        buf.append(f"- **Best block efficiency:** {best_be['block_eff']:.4f} "
                   f"({best_be['draft_label']}, {best_be['mode']}, K={best_be['K']}, "
                   f"{best_be['dataset']})\n")

    # Best overall alpha
    if alpha_runs:
        best_a = max(alpha_runs, key=lambda r: r["alpha_mean"])
        buf.append(f"- **Best alpha:** {best_a['alpha_mean']:.4f} "
                   f"({best_a['draft_label']}, {best_a['dataset']})\n")

    # LR recommendation
    curves = results_db.query_train_curves()
    lr_runs = [c for c in curves if c.get("accept_weight") and c["step"] == 200]
    if lr_runs:
        # Best: highest accept_weight x lowest loss (balanced)
        quality = [(c["label"], c["loss"], c.get("accept_weight", 0), c.get("learning_rate", 0))
                   for c in lr_runs]
        quality_ok = [(l, loss, aw, lr) for l, loss, aw, lr in quality if aw >= 0.965]
        if quality_ok:
            best_lr = min(quality_ok, key=lambda x: x[1])
            buf.append(f"- **Recommended LR for server run:** `{best_lr[3]:.0e}` "
                       f"(final loss={best_lr[1]:.4f}, accept_weight={best_lr[2]:.4f})\n")

    # Best K for each mode
    be_by_mode = defaultdict(list)
    for r in be_runs:
        be_by_mode[r["mode"]].append(r)
    for mode, mode_runs in sorted(be_by_mode.items()):
        by_K = defaultdict(list)
        for r in mode_runs:
            by_K[r["K"]].append(r["block_eff"])
        best_K = max(by_K, key=lambda k: mean(by_K[k]))
        buf.append(f"- **Best K for {mode}:** K={best_K} "
                   f"(mean BE={mean(by_K[best_K]):.4f})\n")

    buf.append("\n**Server run parameters:**\n")
    buf.append("```\npython gbv-research/orchestration/evaluate.py \\\n"
               "    --student Qwen/Qwen2.5-0.5B \\\n"
               "    --teacher Qwen/Qwen3-8B \\\n"
               "    --n 50 \\\n"
               "    --modes alpha,specinfer,gbv,traversal \\\n"
               "    --K 3,5,8\n```\n")

    return "\n".join(buf)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",       default=None,      help="Save Markdown report to file")
    p.add_argument("--section",   default="all",
                   choices=["all","alpha","be","convergence","quality","datasets","throughput","recommend"],
                   help="Which section to generate")
    p.add_argument("--baseline",  default="baseline")
    p.add_argument("--compare",   default=None,      help="Comma-sep labels to compare vs baseline")
    args = p.parse_args()

    compare = args.compare.split(",") if args.compare else None

    if args.section == "all":
        report = generate_full_report(args.baseline, compare)
    elif args.section == "alpha":
        report = analyze_alpha(args.baseline, compare)
    elif args.section == "be":
        report = analyze_block_efficiency(args.baseline, compare)
    elif args.section == "convergence":
        report = analyze_convergence()
    elif args.section == "quality":
        report = analyze_quality_flags()
    elif args.section == "datasets":
        report = analyze_datasets()
    elif args.section == "throughput":
        report = analyze_throughput()
    elif args.section == "recommend":
        report = _generate_recommendations()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
