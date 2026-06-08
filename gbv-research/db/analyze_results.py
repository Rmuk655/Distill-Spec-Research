"""
analyze_results.py — hypothesis-focused analysis of results.db.

Answers the paper questions (see docs/GUIDE.md §1):
  - Which (trained model, verifier) pair maximises block efficiency?
  - Does traversal beat specinfer? (H1)
  - Do mode-seeking / tree-aware objectives beat forward KL? (H2/H5)
  - Do gains generalise across datasets? (H4)
  - Alpha vs BE: is improvement acceptance-structure or rate? (gap analysis)

Does NOT compare raw training losses across objectives (meaningless — different scales).

Usage:
    python db/analyze_results.py
    python db/analyze_results.py --out report.md
    python db/analyze_results.py --section be --dataset gsm8k
    python db/analyze_results.py --experiment_tag KrishnanRIITHServer
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))
import results_db

# All verifier modes that produce block_eff (excludes alpha / perplexity).
BE_MODES = frozenset({
    "naive", "nss", "specinfer", "spectr", "khisti", "bv", "gbv", "traversal",
})

# Flat offline losses for ordering the metrics table (display order only).
FLAT_LOSSES = ("baseline", "l1", "kl", "jsd", "rev_kl", "ebe", "ebe_single")
TREE_SUFFIX = "_tree"

# Verifier taxonomy (theory, NOT a result): expected algorithmic strength.
# tree-search verifiers explore K paths; sequential accept a single prefix;
# token verifiers accept one token at a time. Used to phrase hypotheses, never
# to assert an outcome — the verdict is always computed from the DB.
VERIFIER_CLASS = {
    "traversal": "tree-search",
    "gbv": "tree-search",
    "bv": "sequential",
    "specinfer": "sequential",
    "spectr": "sequential",
    "khisti": "sequential",
    "nss": "token",
    "naive": "token",
}

# Objective taxonomy (distillation theory, NOT a result): how each loss shapes
# the student q relative to teacher p. Reverse-KL is mode-seeking, forward-KL is
# mode-covering, JSD is symmetric, L1 is a norm match, EBE targets block accept.
# This is standard ML theory used only to *frame* comparisons; which family wins
# is always read from the data.
OBJECTIVE_FAMILY = {
    "baseline": "untrained reference",
    "kl": "mode-covering (forward KL)",
    "forward_kl": "mode-covering (forward KL)",
    "rev_kl": "mode-seeking (reverse KL)",
    "reverse_kl": "mode-seeking (reverse KL)",
    "jsd": "symmetric (Jensen-Shannon)",
    "l1": "norm matching (L1)",
    "ebe": "block-acceptance (EBE)",
    "ebe_single": "block-acceptance (EBE, single path)",
    "online": "online adaptation",
}


def _family(label: str) -> str:
    base = label.replace(TREE_SUFFIX, "") if label.endswith(TREE_SUFFIX) else label
    fam = OBJECTIVE_FAMILY.get(base, "other")
    if label.endswith(TREE_SUFFIX):
        fam += " + tree-aware"
    return fam


def _is_tree(label: str) -> bool:
    return label.endswith(TREE_SUFFIX)


# ---------------------------------------------------------------------------
# Stats helpers (no scipy)
# ---------------------------------------------------------------------------

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def pct_change(new, old):
    if old is None or old == 0 or new is None:
        return float("nan")
    return 100 * (new - old) / old


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_std = math.sqrt((std(a) ** 2 + std(b) ** 2) / 2)
    return (mean(a) - mean(b)) / pooled_std if pooled_std > 0 else 0.0


def welch_t_pvalue(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = mean(a), mean(b)
    va, vb = std(a) ** 2 / len(a), std(b) ** 2 / len(b)
    se = math.sqrt(va + vb)
    if se == 0:
        return 1.0
    t = (ma - mb) / se
    df = (va + vb) ** 2 / (va ** 2 / max(len(a) - 1, 1) + vb ** 2 / max(len(b) - 1, 1))
    if df > 30:
        return max(0.0, min(1.0, 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))))
    return 0.5


def significance_label(p):
    if math.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def md_table(headers, rows, alignments=None):
    if not rows:
        return "_No data_\n"
    col_w = [
        max(len(h), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    aligns = alignments or ["l"] + ["r"] * (len(headers) - 1)

    def fmt_sep(w, a):
        if a == "r":
            return "-" * (w - 1) + ":"
        if a == "c":
            return ":" + "-" * (w - 2) + ":"
        return "-" * w

    lines = [
        "| " + " | ".join(h.ljust(col_w[i]) for i, h in enumerate(headers)) + " |",
        "| " + " | ".join(fmt_sep(col_w[i], aligns[i]) for i in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(
                str(v).ljust(col_w[i]) if aligns[i] == "l" else str(v).rjust(col_w[i])
                for i, v in enumerate(row)
            ) + " |"
        )
    return "\n".join(lines) + "\n"


def section(title, level=2):
    return f"\n{'#' * level} {title}\n"


# ---------------------------------------------------------------------------
# Filter context
# ---------------------------------------------------------------------------

@dataclass
class ReportContext:
    baseline: str = "baseline"
    compare_labels: Optional[list] = None
    experiment_tag: Optional[str] = None
    hw_tier: Optional[str] = None
    dataset: Optional[str] = None
    K: Optional[int] = None
    temperature: Optional[float] = None
    _all_runs: list = field(default_factory=list, repr=False)

    def load(self):
        self._all_runs = results_db.query_runs()
        return self

    def runs(self, *, mode: str | None = None, be_only: bool = False,
             alpha_only: bool = False) -> list:
        rows = self._all_runs
        if self.experiment_tag:
            rows = [r for r in rows if r.get("experiment_tag") == self.experiment_tag]
        if self.hw_tier:
            rows = [r for r in rows if r.get("hw_tier") == self.hw_tier]
        if self.dataset:
            rows = [r for r in rows if r.get("dataset") == self.dataset]
        if self.K is not None:
            rows = [r for r in rows if r.get("K") == self.K]
        if self.temperature is not None:
            rows = [r for r in rows if abs(r.get("temperature", 0) - self.temperature) < 0.01]
        if mode:
            rows = [r for r in rows if r.get("mode") == mode]
        if be_only:
            rows = [r for r in rows
                    if r.get("mode") in BE_MODES and r.get("block_eff") is not None]
        if alpha_only:
            rows = [r for r in rows if r.get("mode") == "alpha"]
        return rows

    def primary_gsm8k_dataset(self) -> Optional[str]:
        gsm = [r.get("dataset") for r in self.runs(be_only=True) + self.runs(alpha_only=True)
               if r.get("dataset") and "gsm8k" in r.get("dataset", "")]
        if not gsm:
            return self.dataset
        return max(set(gsm), key=gsm.count)

    def draft_labels(self, be_only: bool = False) -> list:
        src = self.runs(be_only=be_only) if be_only else self.runs()
        return sorted({r["draft_label"] for r in src if r.get("draft_label")})


def _be_for(ctx: ReportContext, label: str, mode: str, dataset: str, K: int) -> Optional[float]:
    vals = [
        r["block_eff"] for r in ctx.runs(be_only=True)
        if r["draft_label"] == label and r["mode"] == mode
        and r["dataset"] == dataset and r["K"] == K
    ]
    return mean(vals) if vals else None


def _alpha_for(ctx: ReportContext, label: str, dataset: str) -> Optional[float]:
    vals = [
        r["alpha_mean"] for r in ctx.runs(alpha_only=True)
        if r["draft_label"] == label and r["dataset"] == dataset
        and r.get("alpha_mean") is not None
    ]
    return mean(vals) if vals else None


def _task_score_for(ctx: ReportContext, label: str, dataset: str) -> Optional[float]:
    vals = [
        r["task_score"] for r in ctx.runs(alpha_only=True)
        if r["draft_label"] == label and r["dataset"] == dataset
        and r.get("task_score") is not None
    ]
    return mean(vals) if vals else None


def _best_be_for_label(ctx: ReportContext, label: str, dataset: str, K: int
                       ) -> tuple[Optional[float], Optional[str]]:
    best_v, best_m = None, None
    for mode in BE_MODES:
        v = _be_for(ctx, label, mode, dataset, K)
        if v is not None and (best_v is None or v > best_v):
            best_v, best_m = v, mode
    return best_v, best_m


def _trained_labels(ctx: ReportContext, be_only: bool = True) -> list:
    """Draft labels excluding the baseline reference."""
    return [l for l in ctx.draft_labels(be_only=be_only) if l != ctx.baseline]


def _be_leader(ctx: ReportContext, dataset: str, K: int
               ) -> tuple[Optional[str], Optional[float], Optional[str]]:
    """Return (label, BE, verifier) of the best trained objective on this slice.

    Fully data-driven — no objective is privileged. Returns (None, None, None)
    when no trained BE rows exist for the slice.
    """
    best = (None, None, None)
    for lab in _trained_labels(ctx):
        be, mode = _best_be_for_label(ctx, lab, dataset, K)
        if be is not None and (best[1] is None or be > best[1]):
            best = (lab, be, mode)
    return best


def _strip_ckpt(label: str) -> str:
    """'rev_kl-gsm8k-q0.6b-q8b' → 'rev_kl' (best-effort loss name from a ckpt label)."""
    return label.split("-")[0]


def _best_val_loss_label(ctx: ReportContext) -> Optional[tuple]:
    """Return (label, last_val_loss) for the checkpoint with the lowest final val loss.

    Reads train_curves rows with split='val'. Returns None when no val rows exist.
    Note: this is within-objective convergence only — never used to rank objectives
    against each other on loss scale.
    """
    curves = results_db.query_train_curves()
    val_pts = [c for c in curves if c.get("split") == "val"]
    if not val_pts:
        return None
    by_label: dict[str, list] = defaultdict(list)
    for c in val_pts:
        by_label[c["label"]].append(c)
    finals = []
    for label, pts in by_label.items():
        pts = sorted(pts, key=lambda x: x["step"])
        finals.append((label, pts[-1]["loss"]))
    return min(finals, key=lambda x: x[1]) if finals else None


def _objective_gain_rows(ctx: ReportContext, dataset: str, K: int) -> list:
    """Per-objective summary: family, best BE + verifier, BE gain, alpha gain.

    Used to derive structural-vs-rate insights generically for whatever
    objectives are present in the DB.
    """
    base_be, _ = _best_be_for_label(ctx, ctx.baseline, dataset, K)
    base_alpha = _alpha_for(ctx, ctx.baseline, dataset)
    rows = []
    for lab in _trained_labels(ctx):
        be, mode = _best_be_for_label(ctx, lab, dataset, K)
        alpha = _alpha_for(ctx, lab, dataset)
        be_gain = pct_change(be, base_be) if (be is not None and base_be) else float("nan")
        a_gain = (pct_change(alpha, base_alpha)
                  if (alpha is not None and base_alpha) else float("nan"))
        rows.append({
            "label": lab,
            "family": _family(lab),
            "be": be,
            "verifier": mode,
            "be_gain": be_gain,
            "alpha": alpha,
            "alpha_gain": a_gain,
        })
    return rows


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def analyze_eval_conditions(ctx: ReportContext) -> str:
    buf = [section("Eval Conditions (under which results were measured)")]
    runs = ctx.runs()
    if not runs:
        return "_No runs in DB (check results_db.DB_PATH and filters)._\n"

    buf.append(f"**DB path:** `{results_db.DB_PATH}`\n")
    filters = []
    if ctx.experiment_tag:
        filters.append(f"experiment_tag=`{ctx.experiment_tag}`")
    if ctx.hw_tier:
        filters.append(f"hw_tier=`{ctx.hw_tier}`")
    if ctx.dataset:
        filters.append(f"dataset=`{ctx.dataset}`")
    if ctx.K is not None:
        filters.append(f"K={ctx.K}")
    if ctx.temperature is not None:
        filters.append(f"T={ctx.temperature}")
    buf.append("**Active filters:** " + (", ".join(filters) if filters else "none (all rows)\n"))

    # One row per unique eval configuration
    groups: dict[tuple, list] = defaultdict(list)
    for r in runs:
        key = (
            r.get("draft_label"), r.get("loss_name"), r.get("mode"),
            r.get("dataset"), r.get("K"), r.get("L"),
            round(float(r.get("temperature") or 0), 4),
            r.get("n_prompts"), r.get("train_steps"),
            r.get("learning_rate"), r.get("lora_rank"),
            r.get("hw_tier"), r.get("experiment_tag"),
        )
        groups[key].append(r)

    headers = [
        "draft", "loss", "mode", "dataset", "K", "L", "T", "n",
        "train_steps", "lr", "lora_r", "hw_tier", "tag",
    ]
    rows = []
    for key in sorted(groups.keys()):
        rows.append([
            key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7],
            key[8],
            f"{key[9]:.0e}" if key[9] else "—",
            key[10] or "—",
            key[11] or "—",
            (key[12] or "—")[:24],
        ])
    buf.append(md_table(headers, rows))
    buf.append(
        "_Each row is one eval cell. Compare models only within the same "
        "(dataset, K, L, T, n_prompts, hw_tier) slice._\n"
    )
    return "\n".join(buf)


def analyze_paper_metrics_table(ctx: ReportContext) -> str:
    """Primary table: alpha, best BE (+ verifier), GSM8K EM — flags missing alpha."""
    buf = [section("Paper Metrics — Alpha × Best BE × Task Score")]
    ds = ctx.dataset or ctx.primary_gsm8k_dataset()
    K = ctx.K if ctx.K is not None else 3
    if not ds:
        return "_No GSM8K eval data._\n"

    buf.append(f"**Slice:** dataset=`{ds}`, K={K} "
               f"(override with `--dataset` / `--K`)\n\n")

    labels = ctx.draft_labels(be_only=True) or ctx.draft_labels()
    if ctx.compare_labels:
        labels = [ctx.baseline] + [l for l in ctx.compare_labels if l in labels]
    else:
        # Prefer flat losses in canonical order, then tree losses
        order = {l: i for i, l in enumerate(FLAT_LOSSES)}
        labels = sorted(labels, key=lambda l: (order.get(l, 99), l))

    headers = [
        "Objective", "Alpha", "BE (best)", "Best verifier",
        "GSM8K EM", "BE vs base", "Alpha gap",
    ]
    rows = []
    base_alpha = _alpha_for(ctx, ctx.baseline, ds)
    base_be, base_be_mode = _best_be_for_label(ctx, ctx.baseline, ds, K)

    for lab in labels:
        alpha = _alpha_for(ctx, lab, ds)
        be, be_mode = _best_be_for_label(ctx, lab, ds, K)
        em = _task_score_for(ctx, lab, ds)
        be_gain = (
            f"{pct_change(be, base_be):+.1f}%"
            if be is not None and base_be else "—"
        )
        alpha_gap = (
            f"{alpha - base_alpha:+.4f}"
            if alpha is not None and base_alpha is not None else "—"
        )
        rows.append([
            lab,
            f"{alpha:.4f}" if alpha is not None else "**missing**",
            f"{be:.4f}" if be is not None else "—",
            be_mode or "—",
            f"{em:.2f}" if em is not None else "—",
            be_gain,
            alpha_gap,
        ])
    buf.append(md_table(headers, rows))

    missing_alpha = [lab for lab in labels if _alpha_for(ctx, lab, ds) is None]
    if missing_alpha:
        buf.append(
            f"\n**Data gap:** no alpha row for {', '.join(missing_alpha)} on `{ds}`. "
            "Run `eval_*_gsm8k` with `--task_score` for each loss before ranking objectives.\n"
        )

    buf.append(
        "\n**How to read:** A large BE gain with a small alpha gain means the "
        "improvement is in *where* acceptance happens (chunk/path structure), not the "
        "per-token rate. The best verifier varies by objective — do not rank losses "
        "using gbv alone. See H3/H4 for the per-objective verdicts.\n"
    )
    return "\n".join(buf)


def analyze_block_efficiency(ctx: ReportContext) -> str:
    runs = ctx.runs(be_only=True)
    if not runs:
        return "_No block efficiency results in DB yet._\n"

    all_labels = sorted({r["draft_label"] for r in runs if r["draft_label"] != "same_model"})
    if ctx.compare_labels:
        all_labels = [l for l in all_labels
                      if l == ctx.baseline or l in ctx.compare_labels]

    buf = [section("Block Efficiency — Loss × Verifier Matrix")]
    modes = sorted({r["mode"] for r in runs})
    Ks = sorted({r["K"] for r in runs})

    buf.append("### Block Efficiency by Mode, K, and Draft\n")
    headers = ["Mode", "K"] + all_labels + ["Best gain vs baseline"]
    be_rows = []
    for mode in modes:
        for K in Ks:
            row = [mode, K]
            vals_by_label = {}
            for label in all_labels:
                matching = [
                    r["block_eff"] for r in runs
                    if r["mode"] == mode and r["K"] == K and r["draft_label"] == label
                ]
                v = mean(matching) if matching else None
                vals_by_label[label] = v
                row.append(f"{v:.4f}" if v else "—")
            base_v = vals_by_label.get(ctx.baseline)
            trained = {l: v for l, v in vals_by_label.items()
                       if l != ctx.baseline and v is not None}
            if base_v and trained:
                best_l = max(trained, key=trained.get)
                row.append(f"{best_l} {pct_change(trained[best_l], base_v):+.1f}%")
            else:
                row.append("—")
            be_rows.append(row)
    buf.append(md_table(headers, be_rows))

    # Per-mode ranking for key verifiers (bv, gbv, traversal)
    buf.append("### Flat-Loss Ranking by Verifier (vs baseline)\n")
    ds = ctx.dataset or ctx.primary_gsm8k_dataset()
    K = ctx.K if ctx.K is not None else 3
    if ds:
        for mode in ("bv", "gbv", "traversal"):
            base = _be_for(ctx, ctx.baseline, mode, ds, K)
            if base is None:
                continue
            buf.append(f"**{mode}** on `{ds}` K={K} (baseline={base:.4f}):\n\n")
            rank = []
            for lab in all_labels:
                if lab == ctx.baseline:
                    continue
                v = _be_for(ctx, lab, mode, ds, K)
                if v is not None:
                    rank.append((lab, v, pct_change(v, base)))
            rank.sort(key=lambda x: -x[1])
            for lab, v, gain in rank:
                buf.append(f"- {lab}: {v:.4f} ({gain:+.1f}% vs baseline)\n")
            buf.append("\n")

    return "\n".join(buf)


def analyze_alpha(ctx: ReportContext) -> str:
    runs = ctx.runs(alpha_only=True)
    if not runs:
        return "_No alpha results in DB yet._\n"

    compare = ctx.compare_labels
    if compare is None:
        compare = [l for l in ctx.draft_labels() if l != ctx.baseline]

    buf = [section("Alpha (Token Acceptance Rate)")]
    headers = [
        "Draft", "Dataset", "n", "Alpha", "Std", "95% CI",
        "Throughput", "ms/tok", "GSM8K EM", "train_steps", "tag",
    ]
    rows = []
    for r in sorted(runs, key=lambda x: (x["draft_label"], x["dataset"])):
        if r.get("alpha_mean") is None:
            continue
        rows.append([
            r["draft_label"], r["dataset"], r["n_prompts"],
            f"{r['alpha_mean']:.4f}", f"{r.get('alpha_std', 0):.4f}",
            f"±{r.get('alpha_ci95', 0):.4f}",
            f"{r.get('throughput', 0):.2f}" if r.get("throughput") else "—",
            f"{r.get('ms_per_tok', 0):.0f}" if r.get("ms_per_tok") else "—",
            f"{r.get('task_score', 0):.2f}" if r.get("task_score") is not None else "—",
            r.get("train_steps", "—"),
            (r.get("experiment_tag") or "—")[:20],
        ])
    buf.append(md_table(headers, rows))

    buf.append("### vs Baseline (same dataset)\n")
    sig_headers = ["Compare", "Dataset", "Δ Alpha", "% change", "p-value", "Sig"]
    sig_rows = []
    base_runs = [r for r in runs if r["draft_label"] == ctx.baseline]
    for label in compare:
        for ds in sorted({r["dataset"] for r in runs}):
            b = [r["alpha_mean"] for r in base_runs
                 if r["dataset"] == ds and r.get("alpha_mean") is not None]
            c = [r["alpha_mean"] for r in runs
                 if r["draft_label"] == label and r["dataset"] == ds
                 and r.get("alpha_mean") is not None]
            if not b or not c:
                continue
            bm, cm = mean(b), mean(c)
            p = welch_t_pvalue(c, b) if len(c) > 1 and len(b) > 1 else float("nan")
            sig_rows.append([
                label, ds, f"{cm - bm:+.4f}", f"{pct_change(cm, bm):+.1f}%",
                "n/a (1 cell)" if math.isnan(p) else f"{p:.4f}",
                significance_label(p) if not math.isnan(p) else "n/a",
            ])
    buf.append(md_table(sig_headers, sig_rows))
    buf.append(
        "_Significance needs multiple runs or per-prompt data per cell; "
        "with one eval row per (loss, dataset) treat p-values as indicative only._\n"
    )
    return "\n".join(buf)


def _verdict(supported: Optional[bool], n_obs: int = 0) -> str:
    if supported is None or n_obs == 0:
        return "INSUFFICIENT DATA"
    return "SUPPORTED" if supported else "NOT SUPPORTED"


def analyze_hypotheses(ctx: ReportContext) -> str:
    """Evaluate the project's research hypotheses purely from DB rows.

    Hypotheses are fixed (they define what the project is trying to prove);
    every verdict is computed from data — no objective or verifier is assumed
    to win. INSUFFICIENT DATA is printed whenever the required rows are absent.
    """
    buf = [section("Hypothesis Checks")]
    buf.append(
        "_Hypotheses are the questions the project tests; verdicts below are "
        "computed from the current DB only. `INSUFFICIENT DATA` = the rows "
        "needed to decide are not present yet._\n"
    )
    ds = ctx.dataset or ctx.primary_gsm8k_dataset()
    K = ctx.K if ctx.K is not None else 3
    datasets_all = sorted({r["dataset"] for r in ctx.runs(be_only=True)})
    Ks_all = sorted({r["K"] for r in ctx.runs(be_only=True)}) or [K]

    # ── H1: verifier hierarchy (tree-search > sequential > token) ────────────
    buf.append("\n### H1 — Verifier hierarchy: tree-search ≥ sequential ≥ token\n")
    buf.append(
        "_Theory: tree-search verifiers (traversal, gbv) explore K paths and "
        "should accept ≥ as many tokens as single-prefix (specinfer, bv) or "
        "token-level (naive, nss) verifiers, holding the draft fixed._\n\n"
    )
    h1_rows = []
    h1_support, h1_total = 0, 0
    for dataset in (datasets_all or ([ds] if ds else [])):
        for Kv in Ks_all:
            for label in ctx.draft_labels(be_only=True):
                tree_bes = [
                    _be_for(ctx, label, m, dataset, Kv)
                    for m, c in VERIFIER_CLASS.items() if c == "tree-search"
                ]
                seq_bes = [
                    _be_for(ctx, label, m, dataset, Kv)
                    for m, c in VERIFIER_CLASS.items() if c == "sequential"
                ]
                tree_bes = [v for v in tree_bes if v is not None]
                seq_bes = [v for v in seq_bes if v is not None]
                if not tree_bes or not seq_bes:
                    continue
                best_tree = max(tree_bes)
                best_seq = max(seq_bes)
                h1_total += 1
                ok = best_tree >= best_seq
                h1_support += int(ok)
                h1_rows.append([
                    label, dataset, Kv,
                    f"{best_tree:.4f}", f"{best_seq:.4f}",
                    "✓" if ok else "✗",
                ])
    if h1_rows:
        buf.append(md_table(
            ["Draft", "Dataset", "K", "best tree-search BE", "best sequential BE", "tree≥seq"],
            h1_rows,
        ))
        buf.append(f"\n**Verdict:** {_verdict(h1_support == h1_total, h1_total)} "
                   f"({h1_support}/{h1_total} cells with tree-search ≥ sequential)\n")
    else:
        buf.append("_Need both a tree-search and a sequential verifier on the same cell._\n")

    # ── H2: distillation improves BE over the untrained baseline ─────────────
    buf.append("\n### H2 — Distillation improves block efficiency over baseline\n")
    if ds:
        gains = _objective_gain_rows(ctx, ds, K)
        scored = [g for g in gains if g["be"] is not None and not math.isnan(g["be_gain"])]
        h2_rows = [
            [g["label"], g["family"], g["verifier"], f"{g['be']:.4f}", f"{g['be_gain']:+.1f}%",
             "✓" if g["be_gain"] > 0 else "✗"]
            for g in sorted(scored, key=lambda x: -(x["be"] or 0))
        ]
        if h2_rows:
            buf.append(md_table(
                ["Objective", "Family", "Best verifier", "Best BE", "vs baseline", "improves"],
                h2_rows,
            ))
            n_pos = sum(1 for g in scored if g["be_gain"] > 0)
            buf.append(f"\n**Verdict:** {_verdict(n_pos > 0, len(scored))} "
                       f"({n_pos}/{len(scored)} objectives beat baseline on `{ds}` K={K})\n")
        else:
            buf.append("_No trained objective has BE on this slice yet._\n")
    else:
        buf.append("_No GSM8K BE rows yet._\n")

    # ── H3: best objective family — mode-seeking vs mode-covering vs symmetric ─
    buf.append("\n### H3 — Which objective family maximises BE?\n")
    buf.append(
        "_Theory predicts mode-seeking (reverse KL) suits acceptance-maximisation "
        "better than mode-covering (forward KL); this checks whether the DB agrees, "
        "without assuming it does._\n\n"
    )
    if ds:
        by_family: dict[str, list] = defaultdict(list)
        for g in _objective_gain_rows(ctx, ds, K):
            if g["be"] is not None:
                by_family[g["family"]].append(g["be"])
        if by_family:
            fam_rows = sorted(
                ([fam, f"{max(v):.4f}", f"{mean(v):.4f}", len(v)]
                 for fam, v in by_family.items()),
                key=lambda r: -float(r[1]),
            )
            buf.append(md_table(["Objective family", "max BE", "mean BE", "n objectives"], fam_rows))
            lead_family = fam_rows[0][0]
            buf.append(f"\n**Observed BE leader (this slice):** {lead_family} family. "
                       "This is a data observation, not a pre-registered outcome.\n")
        else:
            buf.append("_No BE rows to group by family._\n")
    else:
        buf.append("_No GSM8K BE rows yet._\n")

    # ── H4: BE improvement decouples from alpha (structural gains) ────────────
    buf.append("\n### H4 — BE gains exceed alpha gains (structural acceptance)\n")
    buf.append(
        "_If BE rises more than the token-acceptance rate α, the improvement is "
        "structural (longer accepted runs / better tree paths) rather than a "
        "uniform per-token gain. Flagged per objective._\n\n"
    )
    if ds:
        h4_rows = []
        n_struct, n_eval = 0, 0
        for g in _objective_gain_rows(ctx, ds, K):
            if g["be"] is None or g["alpha"] is None:
                continue
            if math.isnan(g["be_gain"]) or math.isnan(g["alpha_gain"]):
                continue
            n_eval += 1
            structural = g["be_gain"] > 2 * max(g["alpha_gain"], 0.0) and g["be_gain"] > 0
            n_struct += int(structural)
            h4_rows.append([
                g["label"], f"{g['alpha_gain']:+.1f}%", f"{g['be_gain']:+.1f}%",
                "structural" if structural else "rate-driven/none",
            ])
        if h4_rows:
            buf.append(md_table(
                ["Objective", "α gain", "BE gain", "interpretation"], h4_rows,
            ))
            buf.append(f"\n**Verdict:** {_verdict(n_struct > 0, n_eval)} "
                       f"({n_struct}/{n_eval} objectives show BE gain ≫ α gain). "
                       "Requires alpha + BE for the same objective.\n")
        else:
            buf.append(
                "_Need both alpha and BE for the same trained objective "
                "(see Data Completeness Gaps)._\n"
            )
    else:
        buf.append("_No GSM8K rows yet._\n")

    # ── H5: verifier-aligned tree losses beat flat under their paired verifier ─
    buf.append("\n### H5 — Tree-aware losses beat flat under matched verifier\n")
    buf.append(
        "_On-policy tree training should help most under the verifier it is "
        "aligned with (bv_tree→bv, gbv_tree→gbv, traversal_tree→traversal; and "
        "X_tree vs X under the best verifier)._\n\n"
    )
    labels_be = set(ctx.draft_labels(be_only=True))
    h5_rows = []
    h5_support, h5_total = 0, 0
    if ds:
        # paired verifier comparisons
        for tree_label, mode in (("bv_tree", "bv"), ("gbv_tree", "gbv"),
                                 ("traversal_tree", "traversal")):
            tree_be = _be_for(ctx, tree_label, mode, ds, K) if tree_label in labels_be else None
            flat_cands = [
                _be_for(ctx, flat, mode, ds, K)
                for flat in _trained_labels(ctx) if not _is_tree(flat)
            ]
            flat_be = max((v for v in flat_cands if v is not None), default=None)
            if tree_be is None and flat_be is None:
                continue
            if tree_be is not None and flat_be is not None:
                h5_total += 1
                h5_support += int(tree_be > flat_be)
            h5_rows.append([
                f"best flat vs {tree_label}", mode,
                f"{flat_be:.4f}" if flat_be else "—",
                f"{tree_be:.4f}" if tree_be else "—",
                ("tree" if (tree_be or 0) > (flat_be or 0) else "flat")
                if (tree_be is not None and flat_be is not None) else "—",
            ])
        # X vs X_tree (same family) under best verifier
        for flat in [l for l in _trained_labels(ctx) if not _is_tree(l)]:
            tree = flat + TREE_SUFFIX
            if tree not in labels_be:
                continue
            flat_be = _best_be_for_label(ctx, flat, ds, K)[0]
            tree_be = _best_be_for_label(ctx, tree, ds, K)[0]
            if flat_be is None and tree_be is None:
                continue
            if flat_be is not None and tree_be is not None:
                h5_total += 1
                h5_support += int(tree_be > flat_be)
            h5_rows.append([
                f"{flat} vs {tree}", "best",
                f"{flat_be:.4f}" if flat_be else "—",
                f"{tree_be:.4f}" if tree_be else "—",
                ("tree" if (tree_be or 0) > (flat_be or 0) else "flat")
                if (tree_be is not None and flat_be is not None) else "—",
            ])
    if h5_rows:
        buf.append(md_table(
            ["Comparison", "Verifier", "Flat BE", "Tree BE", "Winner"], h5_rows,
        ))
        buf.append(f"\n**Verdict:** {_verdict(h5_support > h5_total / 2 if h5_total else None, h5_total)} "
                   f"({h5_support}/{h5_total} comparisons where tree > flat)\n")
    else:
        buf.append("_No tree-loss eval rows yet — train any `*_tree` loss and eval._\n")

    # ── H6: gains generalise across datasets ─────────────────────────────────
    buf.append("\n### H6 — Improvements generalise across datasets\n")
    if len(datasets_all) < 2:
        buf.append(
            "_Need Phase 4 eval on ≥2 datasets (e.g. gsm8k + humaneval/math500). "
            "Finalization default is gsm8k-only._\n"
        )
    else:
        h6_headers = ["Objective"] + datasets_all + ["sign consistent?"]
        h6_rows = []
        consistent_n, total_n = 0, 0
        for lab in _trained_labels(ctx):
            row = [lab]
            signs = []
            for dataset in datasets_all:
                base_be = _best_be_for_label(ctx, ctx.baseline, dataset, K)[0]
                be = _best_be_for_label(ctx, lab, dataset, K)[0]
                if be is not None and base_be:
                    g = pct_change(be, base_be)
                    row.append(f"{g:+.1f}%")
                    signs.append(g > 0)
                else:
                    row.append("—")
            if len(signs) >= 2:
                total_n += 1
                ok = all(signs) or not any(signs)
                consistent_n += int(ok)
                row.append("✓" if ok else "✗ (sign flips)")
            else:
                row.append("—")
            h6_rows.append(row)
        buf.append(md_table(h6_headers, h6_rows))
        buf.append(f"\n**Verdict:** {_verdict(consistent_n == total_n if total_n else None, total_n)} "
                   f"({consistent_n}/{total_n} objectives keep gain sign across datasets)\n")

    # ── H7: quality preserved (task accuracy not regressed) ──────────────────
    buf.append("\n### H7 — Distillation preserves task accuracy (GSM8K EM)\n")
    if ds:
        base_em = _task_score_for(ctx, ctx.baseline, ds)
        h7_rows = []
        n_ok, n_eval = 0, 0
        for lab in _trained_labels(ctx):
            em = _task_score_for(ctx, lab, ds)
            if em is None:
                continue
            n_eval += 1
            if base_em is not None:
                drop = em - base_em
                ok = drop >= -0.05
                n_ok += int(ok)
                h7_rows.append([lab, f"{em:.2f}", f"{base_em:.2f}", f"{drop:+.2f}",
                                "✓" if ok else "✗ regressed"])
            else:
                h7_rows.append([lab, f"{em:.2f}", "—", "—", "—"])
        if h7_rows:
            buf.append(md_table(
                ["Objective", "EM", "baseline EM", "Δ", "preserved (≥ −0.05)"], h7_rows,
            ))
            buf.append(f"\n**Verdict:** {_verdict(n_ok == n_eval if base_em is not None else None, n_eval)} "
                       f"({n_ok}/{n_eval} objectives within 5pp of baseline EM)\n")
        else:
            buf.append("_No task_score rows — run eval with `--task_score`._\n")
    else:
        buf.append("_No GSM8K rows yet._\n")

    # ── H8: best optimiser ≠ best decoder (loss-metric decoupling) ───────────
    buf.append("\n### H8 — Best training loss ≠ best decoding metric\n")
    buf.append(
        "_Checks whether the objective with the lowest validation loss is the "
        "same one with the highest BE. A mismatch is itself a finding "
        "(optimisation quality ≠ speculative-decoding quality)._\n\n"
    )
    val_best = _best_val_loss_label(ctx)
    be_lead, be_val, be_mode = (_be_leader(ctx, ds, K) if ds else (None, None, None))
    if val_best and be_lead:
        same = _strip_ckpt(val_best[0]) == be_lead or val_best[0].startswith(be_lead)
        buf.append(f"- Lowest val loss: `{val_best[0]}` (val={val_best[1]:.4f})\n")
        buf.append(f"- Highest BE: `{be_lead}` + `{be_mode}` (BE={be_val:.4f})\n")
        buf.append(f"\n**Verdict:** {'NOT SUPPORTED (same objective leads both)' if same else 'SUPPORTED (decoupled)'}\n")
    else:
        buf.append("_Need val-loss rows (split='val') and BE rows to compare._\n")

    return "\n".join(buf)


def analyze_training_health(ctx: ReportContext) -> str:
    """Val loss only — never compare raw train loss across objectives."""
    buf = [section("Training Health (validation loss only)")]
    buf.append(
        "> **Do not compare train-loss 'reduction %' across objectives.** "
        "KL, JSD, rev_kl, and L1 live on different scales; early-grad-step "
        "snapshots are noisy. Use **val loss** (split=`val` in train_curves) "
        "or W&B `train/val_loss` for within-loss convergence.\n\n"
    )

    curves = results_db.query_train_curves()
    val_pts = [c for c in curves if c.get("split") == "val"]
    if not val_pts:
        buf.append(
            "_No val rows in train_curves yet. Val loss is logged by trainer.py "
            "to W&B and optionally to train_curves with split='val'._\n"
        )
    else:
        by_label: dict[str, list] = defaultdict(list)
        for c in val_pts:
            by_label[c["label"]].append(c)
        headers = ["Checkpoint label", "loss_name", "val steps", "first val", "last val", "Δ%"]
        rows = []
        for label, pts in sorted(by_label.items()):
            pts = sorted(pts, key=lambda x: x["step"])
            first, last = pts[0]["loss"], pts[-1]["loss"]
            rows.append([
                label, pts[0].get("loss_name", "?"),
                len(pts), f"{first:.4f}", f"{last:.4f}",
                f"{-pct_change(first, last):+.1f}%",
            ])
        buf.append(md_table(headers, rows))

    ckpt = results_db.query_checkpoint_evals()
    if ckpt:
        buf.append("### Checkpoint sweep (alpha / BE vs train_step)\n")
        headers = ["label", "step", "mode", "alpha", "BE", "task_score"]
        rows = []
        for r in ckpt[:40]:
            rows.append([
                r["label"], r["train_step"], r["mode"],
                f"{r['alpha_mean']:.4f}" if r.get("alpha_mean") else "—",
                f"{r['block_eff']:.4f}" if r.get("block_eff") else "—",
                f"{r['task_score']:.2f}" if r.get("task_score") is not None else "—",
            ])
        buf.append(md_table(headers, rows))
        if len(ckpt) > 40:
            buf.append(f"_… {len(ckpt) - 40} more rows in checkpoint_evals._\n")

    return "\n".join(buf)


def analyze_data_gaps(ctx: ReportContext) -> str:
    buf = [section("Data Completeness Gaps")]
    ds = ctx.dataset or ctx.primary_gsm8k_dataset()
    K = ctx.K if ctx.K is not None else 3
    be_labels = set(ctx.draft_labels(be_only=True))
    alpha_labels = {r["draft_label"] for r in ctx.runs(alpha_only=True)}

    missing_alpha = sorted(be_labels - alpha_labels - {ctx.baseline})
    if missing_alpha:
        buf.append("**Missing alpha eval** (have BE but no acceptance rate):\n")
        for lab in missing_alpha:
            modes = sorted({r["mode"] for r in ctx.runs(be_only=True) if r["draft_label"] == lab})
            buf.append(f"- `{lab}` — BE modes present: {', '.join(modes)}\n")
        buf.append(
            "\nFix: `python deploy/aip_run.py --config a100_qwen --loss <L> "
            "--eval --force --no_smoke`\n"
        )
    else:
        buf.append("All models with BE rows also have alpha rows.\n")

    if ds:
        buf.append(f"\n**Verifier coverage** on `{ds}` K={K}:\n\n")
        headers = ["Draft", "Verifiers measured", "Missing verifiers"]
        rows = []
        for lab in sorted(be_labels):
            present = {r["mode"] for r in ctx.runs(be_only=True)
                       if r["draft_label"] == lab and r["dataset"] == ds and r["K"] == K}
            missing = sorted(BE_MODES - present)
            rows.append([lab, ", ".join(sorted(present)) or "—",
                         ", ".join(missing) if missing else "—"])
        buf.append(md_table(headers, rows))

    return "\n".join(buf)


def analyze_quality_flags(ctx: ReportContext) -> str:
    buf = [section("Quality & Sanity Checks")]
    alpha_runs = ctx.runs(alpha_only=True)
    be_runs = ctx.runs(be_only=True)
    flags = []

    base_alpha = {
        r["dataset"]: r["alpha_mean"] for r in alpha_runs
        if r["draft_label"] == ctx.baseline and r.get("alpha_mean")
    }
    for r in alpha_runs:
        if r["draft_label"] == ctx.baseline or not r.get("alpha_mean"):
            continue
        base = base_alpha.get(r["dataset"])
        if base and r["alpha_mean"] < base - 0.02:
            flags.append(
                f"REGRESSION: {r['draft_label']} α={r['alpha_mean']:.4f} < "
                f"baseline {base:.4f} on {r['dataset']}"
            )

    for (label, mode, ds), runs in group_be_by(be_runs).items():
        by_K = {r["K"]: r["block_eff"] for r in runs if r.get("block_eff")}
        if 1 in by_K and 3 in by_K and by_K[1] > by_K[3] + 0.1:
            flags.append(
                f"UNEXPECTED: {label} {mode} on {ds}: BE(K=1)={by_K[1]:.3f} > "
                f"BE(K=3)={by_K[3]:.3f}"
            )

    if flags:
        buf.append("### Flags\n")
        for f in flags:
            buf.append(f"- {f}\n")
    else:
        buf.append("No quality flags raised.\n")

    buf.append("""
### Interpretation (general — not tied to any run)

- **α** is the per-token acceptance rate; **BE** is effective tokens per teacher call.
- **BE gain can exceed α gain** — a verifier may accept longer coherent chunks even
  when the per-token rate barely moves (see H4 for the per-objective check).
- **Mode-seeking vs mode-covering** (reverse vs forward KL) is a theory framing only;
  which family wins is reported from the data in H3, never assumed here.
- **Compare BE only within the same** (dataset, K, L, T, n_prompts, hw_tier) slice.
""")
    return "\n".join(buf)


def group_be_by(runs):
    g = defaultdict(list)
    for r in runs:
        g[(r["draft_label"], r["mode"], r["dataset"])].append(r)
    return g


def analyze_datasets(ctx: ReportContext) -> str:
    alpha_runs = ctx.runs(alpha_only=True)
    datasets = sorted({r["dataset"] for r in alpha_runs})
    if len(datasets) < 2:
        return (
            section("Dataset Robustness")
            + "_Need alpha eval on ≥2 datasets (Phase 4) for cross-dataset robustness._\n"
        )

    buf = [section("Dataset Robustness (alpha)")]
    labels = sorted({r["draft_label"] for r in alpha_runs})
    headers = ["Draft"] + datasets + ["Mean", "Std"]
    rows = []
    for label in labels:
        row = [label]
        vals = []
        for ds in datasets:
            v = _alpha_for(ctx, label, ds)
            row.append(f"{v:.3f}" if v is not None else "—")
            if v is not None:
                vals.append(v)
        if vals:
            row += [f"{mean(vals):.3f}", f"{std(vals):.3f}"]
        else:
            row += ["—", "—"]
        rows.append(row)
    buf.append(md_table(headers, rows))
    return "\n".join(buf)


def analyze_throughput(ctx: ReportContext) -> str:
    runs = [r for r in ctx.runs(alpha_only=True) if r.get("throughput")]
    if not runs:
        return "_No throughput data yet._\n"

    buf = [section("Throughput & Latency (alpha eval)")]
    base_tp = {r["dataset"]: r["throughput"] for r in runs if r["draft_label"] == ctx.baseline}
    headers = ["Draft", "Dataset", "tok/s", "ms/tok", "vs baseline"]
    rows = []
    for r in sorted(runs, key=lambda x: (x["dataset"], x["draft_label"])):
        base = base_tp.get(r["dataset"])
        rows.append([
            r["draft_label"], r["dataset"],
            f"{r['throughput']:.2f}", f"{r.get('ms_per_tok', 0):.0f}",
            f"{pct_change(r['throughput'], base):+.1f}%" if base else "—",
        ])
    buf.append(md_table(headers, rows))
    return "\n".join(buf)


def analyze_narrative(ctx: ReportContext) -> str:
    """Auto-generated, data-driven observations. No objective is hardcoded.

    Everything here is derived from the current DB: the leading objective,
    structural-vs-rate classification, and follow-ups are computed each run.
    """
    buf = [section("Auto-Derived Observations")]
    buf.append(
        "_Generated from the current DB only. Statements name whichever objective "
        "leads in the data now — they are not fixed conclusions about the method._\n"
    )
    ds = ctx.dataset or ctx.primary_gsm8k_dataset()
    K = ctx.K if ctx.K is not None else 3
    if not ds:
        return "\n".join(buf) + "\n_Insufficient data (no GSM8K eval rows)._\n"

    lead_label, lead_be, lead_mode = _be_leader(ctx, ds, K)
    base_best_be, base_best_mode = _best_be_for_label(ctx, ctx.baseline, ds, K)

    buf.append("\n### Current leader (this slice)\n")
    if lead_label is None:
        buf.append("_No trained objective has BE on this slice yet._\n")
    else:
        gain = pct_change(lead_be, base_best_be) if base_best_be else float("nan")
        buf.append(
            f"- **BE leader:** `{lead_label}` ({_family(lead_label)}) under `{lead_mode}` "
            f"= {lead_be:.4f}"
            + (f" ({gain:+.1f}% vs baseline {base_best_be:.4f} under {base_best_mode})\n"
               if not math.isnan(gain) else "\n")
        )

    # Structural vs rate-driven classification (per objective, computed)
    struct_objs, rate_objs = [], []
    for g in _objective_gain_rows(ctx, ds, K):
        if g["alpha"] is None or g["be"] is None:
            continue
        if math.isnan(g["be_gain"]) or math.isnan(g["alpha_gain"]):
            continue
        if g["be_gain"] > 2 * max(g["alpha_gain"], 0.0) and g["be_gain"] > 0:
            struct_objs.append(g)
        else:
            rate_objs.append(g)
    if struct_objs:
        buf.append("\n### Structural acceptance gains (BE gain ≫ α gain)\n")
        for g in sorted(struct_objs, key=lambda x: -x["be_gain"]):
            buf.append(
                f"- `{g['label']}`: α {g['alpha_gain']:+.1f}% but BE {g['be_gain']:+.1f}% "
                "→ improvement in *where* acceptance happens (chunk/path structure), "
                "not just token rate.\n"
            )

    # Loss-vs-metric decoupling (computed, not assumed)
    val_best = _best_val_loss_label(ctx)
    if val_best and lead_label:
        same = _strip_ckpt(val_best[0]) == lead_label or val_best[0].startswith(lead_label)
        buf.append("\n### Optimisation vs decoding\n")
        buf.append(
            f"- Lowest val loss: `{val_best[0]}`; highest BE: `{lead_label}`. "
            + ("These coincide.\n" if same else
               "These **differ** — best optimiser ≠ best speculative decoder (H8).\n")
        )

    # Data-driven follow-ups derived strictly from gaps, not research opinions.
    buf.append("\n### Follow-ups implied by data gaps\n")
    follow = _derive_followups(ctx, ds, K, lead_label)
    if follow:
        for f in follow:
            buf.append(f"- {f}\n")
    else:
        buf.append("- No obvious completeness gaps on this slice.\n")

    buf.append(
        "\n_Interpretation of novelty/significance is left to the researcher; this "
        "section only reports what the numbers show and what is missing to decide a "
        "hypothesis._\n"
    )
    return "\n".join(buf)


def _derive_followups(ctx: ReportContext, ds: str, K: int,
                      lead_label: Optional[str]) -> list:
    """Mechanical, data-driven suggestions — purely from missing rows/coverage."""
    out = []
    be_labels = set(ctx.draft_labels(be_only=True))
    alpha_labels = {r["draft_label"] for r in ctx.runs(alpha_only=True)}

    # 1. Missing alpha for any objective that has BE
    missing_alpha = sorted(be_labels - alpha_labels - {ctx.baseline})
    if missing_alpha:
        out.append(
            f"Missing alpha/EM for {', '.join(missing_alpha)} — cannot evaluate "
            "H4 (structural) or H7 (quality) for them. Run "
            "`--eval --force` for each."
        )

    # 2. Tree variant of the current leader not evaluated
    if lead_label and not _is_tree(lead_label):
        tree_variant = lead_label + TREE_SUFFIX
        if tree_variant not in be_labels:
            out.append(
                f"Current BE leader `{lead_label}` has no `{tree_variant}` rows — "
                "its tree-aware variant is untested (relevant to H5)."
            )

    # 3. Only one dataset → H6 undecidable
    datasets_all = sorted({r["dataset"] for r in ctx.runs(be_only=True)})
    if len(datasets_all) < 2:
        out.append(
            "Only one eval dataset present — H6 (cross-dataset generalisation) "
            "cannot be decided. Run Phase 4 (humaneval/math500/…)."
        )

    # 4. Single seed → significance undecidable
    seeds = {r.get("seed") for r in ctx.runs(be_only=True) if r.get("seed") is not None}
    if len(seeds) <= 1:
        out.append(
            "≤1 seed in BE rows — significance tests return n/a. Repeat key cells "
            "with ≥3 seeds for p-values."
        )

    # 5. No checkpoint sweep → convergence-of-SD-metrics unknown
    if not results_db.query_checkpoint_evals():
        out.append(
            "No `checkpoint_evals` rows — BE/α vs training-step trajectory unknown. "
            "Run the checkpoint sweep to see when SD metrics converge."
        )

    # 6. Verifier coverage holes for the leader
    if lead_label:
        present = {r["mode"] for r in ctx.runs(be_only=True)
                   if r["draft_label"] == lead_label and r["dataset"] == ds and r["K"] == K}
        missing_modes = sorted(BE_MODES - present)
        if missing_modes:
            out.append(
                f"Leader `{lead_label}` missing verifiers {', '.join(missing_modes)} "
                f"on `{ds}` K={K} — full verifier-alignment picture incomplete."
            )
    return out


def _generate_recommendations(ctx: ReportContext) -> str:
    buf = []
    be_runs = ctx.runs(be_only=True)
    alpha_runs = ctx.runs(alpha_only=True)

    if not be_runs and not alpha_runs:
        return "_Not enough eval data yet._\n"

    if be_runs:
        best = max(be_runs, key=lambda r: r["block_eff"])
        buf.append(
            f"- **Best (model, verifier) BE:** {best['block_eff']:.4f} — "
            f"`{best['draft_label']}` + `{best['mode']}` "
            f"(K={best['K']}, `{best['dataset']}`, "
            f"n={best['n_prompts']}, T={best.get('temperature')}, "
            f"hw={best.get('hw_tier')}, tag={best.get('experiment_tag') or '—'})\n"
        )

    ds = ctx.dataset or ctx.primary_gsm8k_dataset()
    K = ctx.K if ctx.K is not None else 3
    if ds:
        for mode in ("bv", "gbv", "traversal"):
            ranked = []
            for lab in ctx.draft_labels(be_only=True):
                v = _be_for(ctx, lab, mode, ds, K)
                if v is not None:
                    ranked.append((lab, v))
            ranked.sort(key=lambda x: -x[1])
            if ranked:
                buf.append(f"- **Best on {mode}** (`{ds}` K={K}): "
                           f"{ranked[0][0]} ({ranked[0][1]:.4f})\n")

    missing = sorted(set(ctx.draft_labels(be_only=True))
                     - {r["draft_label"] for r in alpha_runs})
    if missing:
        buf.append(f"- **Run alpha eval for:** {', '.join(missing)}\n")

    return "\n".join(buf)


def generate_full_report(ctx: ReportContext) -> str:
    ctx.load()
    all_runs = ctx.runs()
    buf = ["# SpecDist Analysis Report\n"]
    buf.append(f"*Generated from `{results_db.DB_PATH}` — "
               f"{datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    buf.append(
        f"**Rows in scope:** {len(all_runs)}  |  "
        f"**Alpha:** {len(ctx.runs(alpha_only=True))}  |  "
        f"**BE:** {len(ctx.runs(be_only=True))}\n"
    )

    buf.append(analyze_eval_conditions(ctx))
    buf.append(analyze_paper_metrics_table(ctx))
    buf.append(analyze_block_efficiency(ctx))
    buf.append(analyze_hypotheses(ctx))
    buf.append(analyze_alpha(ctx))
    buf.append(analyze_data_gaps(ctx))
    buf.append(analyze_training_health(ctx))
    buf.append(analyze_quality_flags(ctx))
    buf.append(analyze_datasets(ctx))
    buf.append(analyze_throughput(ctx))
    buf.append(analyze_narrative(ctx))
    buf.append(section("Recommendations"))
    buf.append(_generate_recommendations(ctx))
    return "\n".join(buf)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Analyze results.db for paper metrics and hypotheses")
    p.add_argument("--out", default=None, help="Save Markdown report to file")
    p.add_argument(
        "--section", default="all",
        choices=[
            "all", "conditions", "metrics", "be", "alpha", "hypotheses",
            "gaps", "training", "convergence", "quality", "datasets", "throughput",
            "narrative", "recommend",
        ],
        help="'convergence' is deprecated alias for 'training' (val loss only)",
    )
    p.add_argument("--baseline", default="baseline")
    p.add_argument("--compare", default=None, help="Comma-separated draft_labels")
    p.add_argument("--experiment_tag", default=None)
    p.add_argument("--hw_tier", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    args = p.parse_args()

    ctx = ReportContext(
        baseline=args.baseline,
        compare_labels=args.compare.split(",") if args.compare else None,
        experiment_tag=args.experiment_tag,
        hw_tier=args.hw_tier,
        dataset=args.dataset,
        K=args.K,
        temperature=args.temperature,
    )
    ctx.load()

    _training = lambda: analyze_training_health(ctx)
    sections = {
        "all": lambda: generate_full_report(ctx),
        "conditions": lambda: analyze_eval_conditions(ctx),
        "metrics": lambda: analyze_paper_metrics_table(ctx),
        "be": lambda: analyze_block_efficiency(ctx),
        "alpha": lambda: analyze_alpha(ctx),
        "hypotheses": lambda: analyze_hypotheses(ctx),
        "gaps": lambda: analyze_data_gaps(ctx),
        "training": _training,
        "convergence": _training,
        "quality": lambda: analyze_quality_flags(ctx),
        "datasets": lambda: analyze_datasets(ctx),
        "throughput": lambda: analyze_throughput(ctx),
        "narrative": lambda: analyze_narrative(ctx),
        "recommend": lambda: section("Recommendations") + _generate_recommendations(ctx),
    }
    report = sections[args.section]()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
