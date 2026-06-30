"""
plot_diag_scatter.py — parse diag_logs/*.log and produce the four-quadrant scatter.

Usage (run on cluster after the diagnostic loop finishes):
  python plot_diag_scatter.py --log_dir diag_logs --out scatter.png

Each point = one checkpoint. Axes:
  1. forward-KL(p||q) vs BE     — coverage-collapse test
  2. reverse-KL(q||p) vs BE     — fidelity-drift test
  3. entropy gap H(p)-H(q) vs BE — student-peaked test
  4. forward-KL vs reverse-KL   — mode-seeking vs mass-covering quadrant map
"""
import argparse
import glob
import os
import re
import sys


def parse_logs(log_dir: str) -> list[dict]:
    records = []
    pattern = re.compile(
        r"\[DIAG_SUMMARY\]\s+"
        r"ckpt=(\S+)\s+"
        r"fwdKL=([0-9eE+\-.]+)\s+"
        r"rkl=([0-9eE+\-.]+)\s+"
        r"Hp=([0-9eE+\-.]+)\s+"
        r"Hq=([0-9eE+\-.]+)\s+"
        r"entropy_gap=([0-9eE+\-.]+)\s+"
        r"jsd=([0-9eE+\-.]+)\s+"
        r"mean_be=([0-9eE+\-.]+)"
    )
    for path in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
        fname = os.path.basename(path)
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        m = pattern.search(text)
        if m is None:
            print(f"  WARN: no [DIAG_SUMMARY] in {fname}", file=sys.stderr)
            continue
        records.append({
            "label":       m.group(1),
            "fwdKL":       float(m.group(2)),
            "rkl":         float(m.group(3)),
            "Hp":          float(m.group(4)),
            "Hq":          float(m.group(5)),
            "entropy_gap": float(m.group(6)),
            "jsd":         float(m.group(7)),
            "mean_be":     float(m.group(8)),
            "log_file":    fname,
        })
    return records


def scatter(ax, xs, ys, labels, xlabel, ylabel, title, colors):
    ax.scatter(xs, ys, c=colors, s=80, zorder=3)
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), fontsize=7, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="diag_logs")
    ap.add_argument("--out", default="scatter.png")
    args = ap.parse_args()

    records = parse_logs(args.log_dir)
    if not records:
        sys.exit("No [DIAG_SUMMARY] lines found — run the diagnostic loop first.")

    print(f"Parsed {len(records)} checkpoints:")
    print(f"  {'label':<35} {'fwdKL':>7} {'rkl':>7} {'H(p)':>7} {'H(q)':>7} "
          f"{'gap':>7} {'BE':>7}")
    for r in sorted(records, key=lambda x: -x["mean_be"]):
        print(f"  {r['label']:<35} {r['fwdKL']:>7.4f} {r['rkl']:>7.4f} "
              f"{r['Hp']:>7.4f} {r['Hq']:>7.4f} {r['entropy_gap']:>+7.4f} "
              f"{r['mean_be']:>7.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        sys.exit("pip install matplotlib numpy")

    labels = [r["label"] for r in records]
    fwdKL  = [r["fwdKL"]      for r in records]
    rkl    = [r["rkl"]        for r in records]
    gap    = [r["entropy_gap"] for r in records]
    be     = [r["mean_be"]    for r in records]

    # Color by BE (blue=high, red=low)
    be_arr = np.array(be)
    norm   = (be_arr - be_arr.min()) / max(be_arr.max() - be_arr.min(), 1e-9)
    colors = plt.cm.RdYlBu(norm)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Diagnostic scatter: KL / entropy quadrant vs Block Efficiency\n"
                 "(color = BE, blue=high, red=low)", fontsize=11)

    scatter(axes[0, 0], fwdKL, be, labels,
            "forward KL(p||q)  ↑ = student misses teacher mass",
            "mean BE",
            "Panel A: coverage-collapse test\n(H0: fwdKL↑ → BE↓)",
            colors)

    scatter(axes[0, 1], rkl, be, labels,
            "reverse KL(q||p)  ↑ = student too peaked",
            "mean BE",
            "Panel B: fidelity-drift test\n(H0: rkl↑ → BE↓ for mode-seeking losses)",
            colors)

    scatter(axes[1, 0], gap, be, labels,
            "entropy gap H(p)−H(q)  positive = student more peaked than teacher",
            "mean BE",
            "Panel C: student-peaked test\n(H0: large gap → coverage collapse → BE↓)",
            colors)

    scatter(axes[1, 1], fwdKL, rkl, labels,
            "forward KL(p||q)",
            "reverse KL(q||p)",
            "Panel D: objective-type map\n"
            "(top-left=mode-seeking, bottom-right=mass-covering, diag=JSD-like)",
            colors)
    # Reference diagonal on panel D
    lo = min(min(fwdKL), min(rkl))
    hi = max(max(fwdKL), max(rkl))
    axes[1, 1].plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1, label="fwdKL=rkl")
    axes[1, 1].legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
