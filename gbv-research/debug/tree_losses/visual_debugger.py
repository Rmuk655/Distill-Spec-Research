"""
visual_debugger.py — a creative visual stepper for the tree losses.

It does three things, all on the toy order-1 Markov models so everything fits
on screen:

  1. DRAW THE TREE        a draft tree coloured by chain weight w (green = the
                          target loves this token as much as the draft → likely
                          accepted; red = the draft over-proposes it).

  2. STEP THROUGH A LOSS  for one tree, print + plot the node-by-node terms the
                          chosen loss computes (α_i, w_i, h_i, the telescoping
                          E[τ]), and verify the trace matches the REAL loss.

  3. TRAIN AND WATCH      optimise the student's W to minimise the loss and watch
                          block efficiency (measured with the REAL verifier)
                          climb — for the matching verifier and for all others.

CLI
---
    python visual_debugger.py --loss bv_tree --K 3 --L 4 --steps 150
    python visual_debugger.py --loss gbv_tree --trace-only
    python visual_debugger.py --loss specinfer_tree --vocab 6 --K 4 --L 5

All figures are written to debug/tree_losses/out/.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tiny_models import TinyMarkovModel, make_dataset, sample_tree, sample_paths, build_dicts, chain_weights
from tree_harness import (
    estimate_block_efficiency, block_efficiency_table, tree_loss_value,
    LOSS_TO_VERIFIER, ALL_VERIFIERS,
    FLAT_LOSS_NAMES as _FLAT_LOSS_NAMES, flat_loss as _flat_loss,
)
import loss_tracer

OUT_DIR = os.path.join(_HERE, "out")   # only used when --save is passed

# Controlled by --save CLI flag; set before any figure is created.
_SAVE = False

def _show_or_save(fig, name: str):
    """Show figure interactively, and optionally save to out/ if --save was passed."""
    fig.tight_layout()
    if _SAVE:
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, name + ".png")
        fig.savefig(path, dpi=120)
        print(f"  saved → {path}")
    plt.show()
    plt.close(fig)


# ===========================================================================
# 1. Tree drawing
# ===========================================================================

_chain_weight_per_node = chain_weights  # canonical impl in tiny_models


def draw_tree(q_paths, q_prefixes, q_dict, p_dict, title,
              selected_path=None):
    """Draw the draft tree; node colour = chain weight w ∈ [0,1]."""
    G = nx.DiGraph()
    for pfx in q_prefixes:
        G.add_node(pfx)
    for pfx in q_prefixes:
        toks = pfx.split(",")
        if len(toks) > 1:
            parent = ",".join(toks[:-1])
            G.add_edge(parent, pfx, tok=toks[-1])

    # layered layout: y = -depth, x spread within each depth
    by_depth = {}
    for pfx in q_prefixes:
        d = len(pfx.split(",")) - 1
        by_depth.setdefault(d, []).append(pfx)
    pos = {}
    for d, nodes in by_depth.items():
        n = len(nodes)
        for i, pfx in enumerate(nodes):
            pos[pfx] = ((i - (n - 1) / 2.0), -d)

    w = _chain_weight_per_node(q_prefixes, q_dict, p_dict)
    colors = [w[pfx] for pfx in G.nodes()]
    labels = {pfx: pfx.split(",")[-1] for pfx in G.nodes()}

    sel_edges = set()
    if selected_path:
        for i in range(1, len(selected_path)):
            sel_edges.add((",".join(str(x) for x in selected_path[:i]),
                           ",".join(str(x) for x in selected_path[:i + 1])))

    fig, ax = plt.subplots(figsize=(8, 5))
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap="RdYlGn",
                           vmin=0.0, vmax=1.0, node_size=900, ax=ax)
    nx.draw_networkx_labels(G, pos, labels, font_size=11, font_weight="bold", ax=ax)
    normal = [e for e in G.edges() if e not in sel_edges]
    nx.draw_networkx_edges(G, pos, edgelist=normal, edge_color="#888",
                           arrows=True, ax=ax)
    if sel_edges:
        nx.draw_networkx_edges(G, pos, edgelist=list(sel_edges), edge_color="blue",
                               width=2.5, arrows=True, ax=ax,
                               label="GBV-selected path")
    edge_labels = {(u, v): G[u][v]["tok"] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="chain weight w = Π min(1, p/q)  (green ≈ accepted)")
    ax.set_title(title)
    ax.axis("off")
    _show_or_save(fig, f"tree_{title[:40].replace(' ','_')}")


# ===========================================================================
# 2. Step through one loss
# ===========================================================================

def print_and_plot_trace(tr):
    """Pretty-print a trace dict and render a matching figure."""
    name = tr["loss_name"]
    print(f"\n=== LOSS TRACE: {name} ===")
    print(f"  real compute_tree_loss = {tr['real_value']:+.5f}")
    print(f"  traced (mirror)        = {tr['traced_value']:+.5f}")
    agree = abs(tr["real_value"] - tr["traced_value"])
    print(f"  |Δ| = {agree:.2e}  {'OK (tracer matches implementation)' if agree < 5e-2 else 'WARN'}")

    fig, ax = plt.subplots(figsize=(9, 5))

    if tr["kind"] == "node_divergence":
        rows = tr["rows"]
        xs = [r["prefix"] for r in rows]
        ys = [r["term"] for r in rows]
        ax.bar(range(len(xs)), ys, color="#4477aa")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("per-node divergence")
        ax.set_title(f"{name}: divergence at each tree node (mean = loss)")

    elif tr["kind"] in ("bv", "alpha_product", "ebe", "traversal"):
        # plot per-depth alpha along the first traced path
        per_path = tr.get("per_path", [])
        for pp in per_path[:4]:
            rows = pp["rows"]
            depths = [r["depth"] for r in rows]
            alphas = [r["alpha"] for r in rows]
            ax.plot(depths, alphas, marker="o",
                    label=f"path {pp['path']}")
        ax.set_xlabel("depth i")
        ax.set_ylabel(r"per-node acceptance $\alpha_i$")
        etau = tr.get("E_tau", tr.get("mean_w_leaf"))
        ax.set_title(f"{name}: α per depth   (E[τ] ≈ {etau:.3f}, loss = {tr['real_value']:+.3f})")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    elif tr["kind"] == "gbv":
        rows = tr["rows"]
        depths = [r["depth"] for r in rows]
        ax.plot(depths, [r["alpha"] for r in rows], marker="o", label=r"$\alpha_i$ (on q_skew)")
        ax.plot(depths, [r["w"] for r in rows], marker="s", label=r"chain weight $w_i$")
        ax.plot(depths, [r["h"] for r in rows], marker="^", label=r"block accept $h_i$")
        ax.set_xlabel("depth i")
        ax.set_ylabel("value")
        ax.set_title(f"{name} on selected path {tr['selected_path']}\n"
                     f"E[τ] ≈ {tr['E_tau']:.3f}, loss = {tr['real_value']:+.3f}")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    _show_or_save(fig, f"trace_{name}")

    # also dump the numeric rows to stdout
    if tr["kind"] == "gbv":
        for r in tr["rows"]:
            print(f"    depth {r['depth']}: token={r['token']} "
                  f"α={r['alpha']:.3f}  w={r['w']:.3f}  h={r['h']:.3f}")
    elif "per_path" in tr:
        for pp in tr["per_path"]:
            print(f"    path {pp['path']}:")
            for r in pp["rows"]:
                extra = {k: round(v, 3) for k, v in r.items()
                         if k not in ("depth", "token")}
                print(f"      depth {r['depth']}: token={r['token']}  {extra}")


# ===========================================================================
# 3. Train and watch
# ===========================================================================

def train_and_watch(loss_name, vocab=6, K=3, L=4, temp=1.0, steps=150,
                     lr=0.05, batch=8, eval_every=15, n_trials=160, seed=0,
                     teacher_peak=6.0, student_scale=0.3,
                     teacher_temp=None, student_temp=None):
    torch.manual_seed(seed)
    tt = teacher_temp if teacher_temp is not None else temp
    st = student_temp if student_temp is not None else temp
    teacher = TinyMarkovModel.teacher_counting(vocab, peak=teacher_peak, seed=seed)
    student = TinyMarkovModel.student_random(vocab, scale=student_scale, seed=seed + 1)
    train_toks, test_toks = make_dataset(vocab, seed=seed + 2)

    is_flat = loss_name in _FLAT_LOSS_NAMES
    verifier = "gbv" if is_flat else LOSS_TO_VERIFIER.get(loss_name, "gbv")
    mode_tag = "(flat sequence-level BASELINE)" if is_flat else f"(tree loss, matched verifier={verifier})"
    print(f"\nTraining loss = {loss_name}  {mode_tag}")
    print(f"vocab={vocab} K={K} L={L} teacher_temp={tt} student_temp={st} steps={steps} lr={lr}")
    if is_flat:
        print("  NOTE: flat losses train on token-pair sequences. "
              "BE is still evaluated with tree verifiers for fair comparison.")

    # ── snapshot BE across all verifiers BEFORE training ──────────────────────
    be_before = block_efficiency_table(student, teacher, test_toks, K, L, temp,
                                        verifiers=ALL_VERIFIERS, n_trials=n_trials, seed=seed,
                                        student_temp=st, teacher_temp=tt)

    # ── a fixed sample tree to visualise before/after (same pending + seed) ──
    snap_gen = torch.Generator().manual_seed(seed + 99)
    snap_pending = int(test_toks[0])
    qp0, qpf0, qd0, pd0 = sample_tree(student, teacher, snap_pending, K, L, temp,
                                      with_grad=False, generator=snap_gen,
                                      student_temp=st, teacher_temp=tt)
    draw_tree(qp0, qpf0, qd0, pd0, f"draft tree BEFORE training ({loss_name})")

    # ── loss trace (tree losses only; flat losses have no tree tracer) ─────────
    if not is_flat:
        qp_g, qpf_g, qd_g, pd_g = sample_tree(student, teacher, snap_pending, K, L, temp,
                                              with_grad=True,
                                              generator=torch.Generator().manual_seed(seed + 99),
                                              student_temp=st, teacher_temp=tt)
        tr = loss_tracer.trace(loss_name, qd_g, pd_g, qp_g, L, K)
        print_and_plot_trace(tr)
    else:
        print("  (skipping per-node trace — flat losses operate on sequences, not trees)")

    # ── training loop ─────────────────────────────────────────────────────────
    opt = torch.optim.Adam([student.W], lr=lr)
    gen = torch.Generator().manual_seed(seed + 1000)
    hist = {"step": [], "loss": [], "be_train": [], "be_test": []}

    for step in range(steps + 1):
        opt.zero_grad()
        loss_accum = torch.zeros(1)
        nb = 0
        if is_flat:
            # Flat training: student.W[pending] are the logits for token `pending`
            for _ in range(batch):
                pending = int(train_toks[torch.randint(len(train_toks), (1,), generator=gen).item()])
                s_logit = student.W[pending].unsqueeze(0)           # [1, V] with grad
                t_logit = teacher.W[pending].unsqueeze(0).detach()  # [1, V] frozen
                out = _flat_loss(loss_name, s_logit, t_logit, generator=gen)
                if out.loss.requires_grad:
                    loss_accum = loss_accum + out.loss
                    nb += 1
        else:
            for _ in range(batch):
                pending = int(train_toks[torch.randint(len(train_toks), (1,), generator=gen).item()])
                l = tree_loss_value(student, teacher, pending, loss_name, K, L, temp,
                                    generator=gen, student_temp=st, teacher_temp=tt)
                if l.requires_grad:
                    loss_accum = loss_accum + l
                    nb += 1
        if nb > 0:
            (loss_accum / nb).backward()
            opt.step()

        if step % eval_every == 0:
            be_tr = estimate_block_efficiency(student, teacher, verifier, train_toks,
                                              K, L, temp, n_trials=n_trials, seed=seed,
                                              student_temp=st, teacher_temp=tt)
            be_te = estimate_block_efficiency(student, teacher, verifier, test_toks,
                                              K, L, temp, n_trials=n_trials, seed=seed + 1,
                                              student_temp=st, teacher_temp=tt)
            hist["step"].append(step)
            hist["loss"].append(float((loss_accum / max(nb, 1)).item()))
            hist["be_train"].append(be_tr)
            hist["be_test"].append(be_te)
            print(f"  step {step:4d}  loss={hist['loss'][-1]:+.4f}  "
                  f"BE[{verifier}] train={be_tr:.3f} test={be_te:.3f}")

    # ── snapshot BE across all verifiers AFTER training ───────────────────────
    be_after = block_efficiency_table(student, teacher, test_toks, K, L, temp,
                                       verifiers=ALL_VERIFIERS, n_trials=n_trials, seed=seed,
                                       student_temp=st, teacher_temp=tt)

    # after-training tree (same pending + seed so the drawing is comparable)
    snap_gen2 = torch.Generator().manual_seed(seed + 99)
    qp1, qpf1, qd1, pd1 = sample_tree(student, teacher, snap_pending, K, L, temp,
                                      with_grad=False, generator=snap_gen2,
                                      student_temp=st, teacher_temp=tt)
    sel = None
    if loss_name == "gbv_tree":
        from distillspec_gbv.losses.tree_losses import _gbv_select_path
        sel = _gbv_select_path(qd1, pd1, qp1, L)
    draw_tree(qp1, qpf1, qd1, pd1, f"draft tree AFTER training ({loss_name})",
              selected_path=sel)

    _plot_curves(hist, loss_name, verifier)
    _plot_be_before_after(be_before, be_after, loss_name, verifier)

    print("\nBlock efficiency (test) before → after:")
    for v in ALL_VERIFIERS:
        star = "  <-- matched" if v == verifier else ""
        print(f"  {v:10s}: {be_before[v]:.3f} → {be_after[v]:.3f}{star}")
    return hist, be_before, be_after


def _plot_curves(hist, loss_name, verifier):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(hist["step"], hist["loss"], marker="o", color="#cc3311")
    a1.set_xlabel("training step"); a1.set_ylabel("loss")
    a1.set_title(f"{loss_name}: loss over training")
    a2.plot(hist["step"], hist["be_train"], marker="o", label="train BE")
    a2.plot(hist["step"], hist["be_test"], marker="s", label="test BE")
    a2.set_xlabel("training step"); a2.set_ylabel(f"block efficiency  (verifier = {verifier})")
    a2.set_title(f"{loss_name}: block efficiency — higher is better")
    a2.legend()
    _show_or_save(fig, f"curves_{loss_name}")


def _plot_be_before_after(be_before, be_after, loss_name, verifier):
    verifiers = list(be_before.keys())
    x = np.arange(len(verifiers))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, [be_before[v] for v in verifiers], width=0.4, label="before training", color="#bbbbbb")
    ax.bar(x + 0.2, [be_after[v] for v in verifiers], width=0.4, label="after training", color="#3388cc")
    ax.set_xticks(x); ax.set_xticklabels(verifiers, rotation=20)
    ax.set_ylabel("block efficiency (test)  — higher is better")
    ax.set_title(f"BE before vs after  |  loss={loss_name}  |  matched verifier={verifier} (bold red)")
    for i, v in enumerate(verifiers):
        if v == verifier:
            ax.get_xticklabels()[i].set_color("red")
            ax.get_xticklabels()[i].set_fontweight("bold")
    ax.legend()
    _show_or_save(fig, f"be_before_after_{loss_name}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--loss", default="bv_tree",
                   help="Loss to study. Tree losses: kl_tree, bv_tree, gbv_tree, ebe_tree, "
                        "specinfer_tree, traversal_tree, naive_tree, nss_tree, spectr_tree. "
                        "Flat baseline losses (train on sequences, BE still measured with tree "
                        "verifiers): forward_kl, reverse_kl, jsd, l1, ebe, ebe_single.")
    p.add_argument("--vocab", type=int, default=6)
    p.add_argument("--K", type=int, default=3, help="number of i.i.d. draft paths")
    p.add_argument("--L", type=int, default=4, help="draft block length")
    p.add_argument("--temp", type=float, default=1.0,
                   help="shared sampling temperature (overridden by --teacher-temp / --student-temp)")
    p.add_argument("--teacher-temp", type=float, default=None,
                   help="teacher inference temperature. Lower = sharper teacher = bigger capability gap. "
                        "Default: same as --temp.")
    p.add_argument("--student-temp", type=float, default=None,
                   help="student draft temperature. Higher = more diffuse student = smaller/weaker model. "
                        "Default: same as --temp.")
    p.add_argument("--teacher-peak", type=float, default=4.0,
                   help="logit scale of the teacher's structured distribution. Higher = more peaked. "
                        "Default 4.0.")
    p.add_argument("--student-scale", type=float, default=0.5,
                   help="init std of the student's random logit matrix. Smaller = starts closer to uniform. "
                        "Default 0.5.")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trace-only", action="store_true",
                   help="just draw one tree + print a single loss trace, no training")
    p.add_argument("--save", action="store_true",
                   help="also save figures as PNGs to out/ (gitignored). "
                        "Default: show interactively only.")
    args = p.parse_args()

    # --trace-only is tree-only (needs a tree tracer); catch flat + trace-only early
    global _SAVE
    _SAVE = args.save

    if args.trace_only and args.loss in _FLAT_LOSS_NAMES:
        tree_equiv = args.loss + "_tree" if args.loss not in ("forward_kl", "reverse_kl", "jsd", "l1") else None
        hint = f"  Try --loss {tree_equiv}" if tree_equiv else ""
        p.error(
            f"--trace-only requires a tree loss (flat losses have no tree tracer).{hint}"
        )

    if args.trace_only:
        torch.manual_seed(args.seed)
        teacher = TinyMarkovModel.teacher_counting(args.vocab, peak=args.teacher_peak,
                                                    seed=args.seed)
        student = TinyMarkovModel.student_random(args.vocab, scale=args.student_scale,
                                                  seed=args.seed + 1)
        _, test_toks = make_dataset(args.vocab, seed=args.seed + 2)
        gen = torch.Generator().manual_seed(args.seed + 99)
        pending = int(test_toks[0])
        tt = args.teacher_temp if args.teacher_temp is not None else args.temp
        st = args.student_temp if args.student_temp is not None else args.temp
        qp, qpf, qd, pd = sample_tree(student, teacher, pending, args.K, args.L,
                                      args.temp, with_grad=True, generator=gen,
                                      student_temp=st, teacher_temp=tt)
        draw_tree(qp, qpf, {k: v.detach() for k, v in qd.items()}, pd,
                  f"sample draft tree ({args.loss})")
        tr = loss_tracer.trace(args.loss, qd, pd, qp, args.L, args.K)
        print_and_plot_trace(tr)
    else:
        train_and_watch(args.loss, vocab=args.vocab, K=args.K, L=args.L,
                        temp=args.temp, steps=args.steps, lr=args.lr, seed=args.seed,
                        teacher_peak=args.teacher_peak, student_scale=args.student_scale,
                        teacher_temp=args.teacher_temp, student_temp=args.student_temp)


if __name__ == "__main__":
    main()
