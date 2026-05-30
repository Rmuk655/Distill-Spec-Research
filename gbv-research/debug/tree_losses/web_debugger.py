"""
web_debugger.py — a web-based step-through debugger for the tree losses.

Train the toy student on a chosen tree loss, then SCRUB / PLAY / JUMP through
the training steps in your browser and watch, all on one page:

  * the draft tree's node colours morph slowly (chain weight w = Π min(1,p/q);
    red = the draft over-proposes this token → rejected, green = the target
    accepts it),
  * the loss curve,
  * block efficiency under **every verifier** (not just the matched one) — so
    you can see whether a loss helps all verifiers or only its own,
  * the **flat (K=1)** block efficiency of every verifier before vs after
    training (K=1 is where the OT methods collapse to vanilla speculative
    sampling and gbv/traversal collapse to block verification),
  * the per-node loss breakdown (α_i, w_i / cumulative product) for the
    currently displayed step.

The training is fast (toy order-1 models), so the server pre-computes one
snapshot per step up front; the browser then plays them back at a speed you
control, with smooth CSS colour transitions for the "slow animation" feel.

This file is a UNIT-TEST + VISUAL DEBUGGER: it drives the REAL loss code
(distillspec_gbv.losses.tree_losses) and the REAL verifier code
(distillspec_gbv.verifiers.otlp_registry) directly. If anyone changes a loss
function or the verifier / training chain, re-run this and the curves / tree /
breakdown will immediately show whether block efficiency still improves.

Run:
    pip install flask          # one-time
    python web_debugger.py     # -> http://127.0.0.1:5050
    python web_debugger.py --port 8090
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

from tiny_models import (  # noqa: E402
    TinyMarkovModel, make_dataset, sample_paths, build_dicts, sample_tree, chain_weights,
)
from tree_harness import (  # noqa: E402
    estimate_block_efficiency, tree_loss_value, compute_tree_loss,
    LOSS_TO_VERIFIER, ALL_VERIFIERS, TREE_LOSS_NAMES,
    FLAT_LOSS_NAMES as _FLAT_LOSS_NAMES, flat_loss as _flat_loss,
)
import loss_tracer  # noqa: E402
import loss_variants as LV  # noqa: E402

try:
    from flask import Flask, jsonify, request, render_template_string
except ImportError:
    print("Flask not installed. Run: pip install flask")
    sys.exit(1)

app = Flask(__name__)

# Verifiers scored at EVERY step (all fast; khisti's scipy LP is too slow for
# per-step, so it is offered only as an opt-in endpoint measurement).
PERSTEP_VERIFIERS = ["naive", "nss", "bv", "specinfer", "spectr",
                     "traversal", "gbv", "max"]

# Background training jobs: job_id -> {phase, cur, total, done, error, data}
JOBS: dict = {}
_JOBS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Per-node helpers
# ---------------------------------------------------------------------------

def _chain_weights(ref_prefixes, q_dict, p_dict):
    """Per-prefix BV chain weight as an ordered list matching ref_prefixes."""
    w = chain_weights(ref_prefixes, q_dict, p_dict)
    return [round(w[pfx], 4) for pfx in ref_prefixes]


def _layout(ref_prefixes):
    """Stable layered layout: x within depth, y = depth. Normalised coords."""
    by_depth = {}
    for pfx in ref_prefixes:
        d = len(pfx.split(",")) - 1
        by_depth.setdefault(d, []).append(pfx)
    max_d = max(by_depth)
    nodes = []
    idx = {pfx: i for i, pfx in enumerate(ref_prefixes)}
    for d, group in by_depth.items():
        n = len(group)
        for j, pfx in enumerate(group):
            x = (j + 0.5) / n
            y = d / max(max_d, 1)
            nodes.append({"id": idx[pfx], "token": pfx.split(",")[-1],
                          "depth": d, "x": round(x, 4), "y": round(y, 4)})
    edges = []
    for pfx in ref_prefixes:
        toks = pfx.split(",")
        if len(toks) > 1:
            parent = ",".join(toks[:-1])
            edges.append({"u": idx[parent], "v": idx[pfx], "token": toks[-1]})
    return nodes, edges


def _step_info(tr):
    if tr["kind"] == "node_divergence":
        return f"mean node divergence = {tr['traced_value']:.4f}"
    if tr["kind"] == "traversal":
        return f"mean leaf weight = {tr['mean_w_leaf']:.4f}  (loss {tr['real_value']:+.4f})"
    if "E_tau" in tr:
        return f"E[τ] = {tr['E_tau']:.4f}   (loss {tr['real_value']:+.4f})"
    return f"loss {tr['real_value']:+.4f}"


def _breakdown(tr):
    """Compact per-node loss decomposition for the current step.

    Returns {"k": kind, "blabel": str, "paths": [[{d,t,a,b,h?}], ...]}  or
    {"k":"div","rows":[{l,v}]} for the divergence losses.
    """
    kind = tr["kind"]
    if kind == "node_divergence":
        return {"k": "div", "blabel": "divergence",
                "rows": [{"l": r["prefix"], "v": round(r["term"], 4)} for r in tr["rows"]]}

    paths = []
    if kind == "gbv":
        rows = [{"d": r["depth"], "t": r["token"], "a": round(r["alpha"], 3),
                 "b": round(r["w"], 3), "h": round(r["h"], 3)} for r in tr["rows"]]
        return {"k": "bv", "blabel": "chain weight w   (line: block-accept h)",
                "paths": [rows]}
    for pp in tr.get("per_path", []):
        rows = []
        for r in pp["rows"]:
            row = {"d": r["depth"], "t": r["token"], "a": round(r["alpha"], 3)}
            if "w" in r:               # bv
                row["b"] = round(r["w"], 3)
                row["h"] = round(r.get("h", 0.0), 3)
            elif "cum_product" in r:   # alpha_product
                row["b"] = round(r["cum_product"], 3)
            elif "prefix_prod" in r:   # ebe
                row["b"] = round(r["prefix_prod"], 3)
            elif "w_leaf" in r:        # traversal
                row["b"] = round(r["w_leaf"], 3)
            else:
                row["b"] = round(r["alpha"], 3)
            rows.append(row)
        paths.append(rows)
    blabel = {"bv": "chain weight w   (line: block-accept h)",
              "alpha_product": "cumulative Π α   (= E[τ] contribution)",
              "ebe": "prefix product Π α",
              "traversal": "leaf weight w_leaf"}.get(kind, "secondary")
    return {"k": kind, "blabel": blabel, "paths": paths}


# ---------------------------------------------------------------------------
# Training + snapshot recording
# ---------------------------------------------------------------------------

def build_session(loss_name, vocab=32, K=3, L=4, steps=80, lr=0.1,
                  seed=0, be_trials=15, batch=8, include_khisti=False, lam=0.1,
                  be_every=5, teacher_peak=2.5, student_scale=0.8,
                  teacher_temp=1.0, student_temp=1.0, progress=None):
    torch.manual_seed(seed)
    teacher = TinyMarkovModel.teacher_counting(vocab, peak=teacher_peak, seed=seed)
    student = TinyMarkovModel.student_random(vocab, scale=student_scale, seed=seed + 1)
    train_toks, test_toks = make_dataset(vocab, seed=seed + 2)
    is_flat = loss_name in _FLAT_LOSS_NAMES
    matched = "gbv" if is_flat else LV.matched_verifier(loss_name)
    trace_base = None if is_flat else LV.base_loss(loss_name)

    # Fixed REFERENCE tree (topology frozen so only colours animate).
    # Use student_temp for draft sampling; teacher_temp for teacher scoring.
    ref_pending = int(test_toks[0])
    ref_paths = sample_paths(student, ref_pending, K, L, student_temp,
                             generator=torch.Generator().manual_seed(seed + 99))
    ref_prefixes, _, _ = build_dicts(student, teacher, ref_paths, L, student_temp, with_grad=False)
    nodes, edges = _layout(ref_prefixes)

    opt = torch.optim.Adam([student.W], lr=lr)
    gen = torch.Generator().manual_seed(seed + 1000)

    perstep_v = list(PERSTEP_VERIFIERS)
    snaps = {"step": [], "loss": [], "info": [], "w": [], "sel": [],
             "be": {v: [] for v in perstep_v}, "bd": []}

    _be_pool = ThreadPoolExecutor(max_workers=len(perstep_v))

    def _be_one(v, K_eval, n):
        return v, round(estimate_block_efficiency(
            student, teacher, v, test_toks, K_eval, L,
            n_trials=n, seed=seed,
            student_temp=student_temp, teacher_temp=teacher_temp), 4)

    def be_all(K_eval):
        futs = {_be_pool.submit(_be_one, v, K_eval, be_trials): v for v in perstep_v}
        return {v: r for f in as_completed(futs) for v, r in [f.result()]}

    def be_flat():
        # K=1 collapses the tree to a single chain → the FLAT verifiers.
        vlist = perstep_v + (["khisti"] if include_khisti else [])
        futs = {_be_pool.submit(_be_one, v, 1,
                                min(be_trials, 15) if v == "khisti" else be_trials): v
                for v in vlist}
        return {v: r for f in as_completed(futs) for v, r in [f.result()]}

    _last_be = {v: None for v in perstep_v}

    def _flat_record_loss(qd, pd):
        """For flat losses: compute loss on all training-token rows and return the mean."""
        rows = []
        for t in range(vocab):
            s_logit = student.W[t].unsqueeze(0)                     # [1, V] with grad
            t_logit = teacher.W[t].unsqueeze(0).detach()            # [1, V] frozen
            out = _flat_loss(loss_name, s_logit, t_logit)
            rows.append(float(out.loss.detach()))
        return sum(rows) / len(rows)

    def _flat_breakdown(qd, pd):
        """Per-node KL(p‖q) for the breakdown panel — delegates to the real tracer."""
        tr = loss_tracer.trace("kl_tree", qd, pd, ref_paths, L, K)
        tr["kind"] = "flat_nodes"
        tr["loss_name"] = loss_name
        tr["real_value"] = tr["traced_value"]
        return tr

    def record(step):
        nonlocal _last_be
        # no-grad build for display/weights; with-grad build shared by tracer + loss
        # (tracer never calls backward, so the graph stays alive for compute_any)
        _, qd, pd   = build_dicts(student, teacher, ref_paths, L, with_grad=False,
                                  student_temp=student_temp, teacher_temp=teacher_temp)
        _, qdg, pdg = build_dicts(student, teacher, ref_paths, L, with_grad=True,
                                  student_temp=student_temp, teacher_temp=teacher_temp)
        if is_flat:
            tr   = _flat_breakdown(qd, pd)
            lval = _flat_record_loss(qd, pd)
        else:
            tr   = loss_tracer.trace(trace_base, qdg, pdg, ref_paths, L, K)
            lval = float(LV.compute_any(loss_name, qdg, pdg, ref_paths, L, K, lam))
        # BE is expensive; only recompute every be_every steps, forward-fill otherwise
        if step % be_every == 0 or step == steps:
            _last_be = be_all(K)
        be = _last_be
        sel = []
        if trace_base == "gbv_tree":
            from distillspec_gbv.losses.tree_losses import _gbv_select_path
            path = _gbv_select_path(qd, pd, ref_paths, L)
            for i in range(1, len(path)):
                sel.append(f"{','.join(str(x) for x in path[:i])}|"
                           f"{','.join(str(x) for x in path[:i+1])}")
        snaps["step"].append(step)
        snaps["loss"].append(round(lval, 5))
        snaps["info"].append(_step_info(tr))
        snaps["w"].append(_chain_weights(ref_prefixes, qd, pd))
        snaps["sel"].append(sel)
        snaps["bd"].append(_breakdown(tr))
        for v in perstep_v:
            snaps["be"][v].append(be[v])

    def _p(phase, cur):
        if progress:
            progress(phase, cur, steps)

    # endpoint flat (K=1) BE for the before→after comparison
    _p("flat (K=1) baseline", 0)
    flat_before = be_flat()
    tree_before = None

    for step in range(steps + 1):
        record(step)
        if step == 0:
            tree_before = {v: snaps["be"][v][0] for v in perstep_v}
        _p("training + scoring all verifiers", step)
        opt.zero_grad()
        acc = torch.zeros(1)
        nb = 0
        if is_flat:
            # Flat training: sample pending tokens, apply published sequence-level loss
            # directly on student.W[pending] vs teacher.W[pending].
            for _ in range(batch):
                pending = int(train_toks[torch.randint(len(train_toks), (1,), generator=gen).item()])
                s_logit = student.W[pending].unsqueeze(0)           # [1, V] with grad
                t_logit = teacher.W[pending].unsqueeze(0).detach()  # [1, V] frozen
                out = _flat_loss(loss_name, s_logit, t_logit, generator=gen)
                if out.loss.requires_grad:
                    acc = acc + out.loss
                    nb += 1
        else:
            # Tree training: sample on-policy trees, apply tree/variant loss
            for _ in range(batch):
                pending = int(train_toks[torch.randint(len(train_toks), (1,), generator=gen).item()])
                qp_t, _, qd_t, pd_t = sample_tree(
                    student, teacher, pending, K, L, with_grad=True, generator=gen,
                    student_temp=student_temp, teacher_temp=teacher_temp)
                l = LV.compute_any(loss_name, qd_t, pd_t, qp_t, L, K, lam)
                if l.requires_grad:
                    acc = acc + l
                    nb += 1
        if nb > 0:
            (acc / nb).backward()
            opt.step()

    _p("flat (K=1) final", steps)
    flat_after = be_flat()
    tree_after = {v: snaps["be"][v][-1] for v in perstep_v}
    khisti_endpoint = None
    if include_khisti:
        khisti_endpoint = {
            "before": round(estimate_block_efficiency(
                student, teacher, "khisti", test_toks, K, L,
                n_trials=min(be_trials, 25), seed=seed,
                student_temp=student_temp, teacher_temp=teacher_temp), 4),
        }

    return {
        "loss_name": loss_name, "matched": matched,
        "is_flat": is_flat,
        "is_variant": (not is_flat) and LV.is_variant(loss_name), "lam": lam,
        "K": K, "L": L, "vocab": vocab,
        "teacher_peak": teacher_peak, "student_scale": student_scale,
        "teacher_temp": teacher_temp, "student_temp": student_temp,
        "verifiers": perstep_v,
        "nodes": nodes, "edges": edges,
        "n": len(snaps["step"]),
        "step": snaps["step"], "loss": snaps["loss"], "info": snaps["info"],
        "w": snaps["w"], "sel": snaps["sel"], "be": snaps["be"], "bd": snaps["bd"],
        "flat_before": flat_before, "flat_after": flat_after,
        "tree_before": tree_before, "tree_after": tree_after,
        "khisti": khisti_endpoint,
    }


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

_FLAT_DESCRIPTIONS = {
    # Standard published baselines (sequence-level KL / JSD / L1 — not novel to this work)
    "forward_kl":  "PUBLISHED BASELINE · KL(p_teacher ‖ p_student) — mode-covering; penalises student for missing tokens the teacher favours. DistillSpec Section 3.1 baseline.",
    "reverse_kl":  "PUBLISHED BASELINE · KL(p_student ‖ p_teacher) — mode-seeking; student concentrates on teacher's top tokens, ignores tail. Sharper drafts, worse coverage.",
    "jsd":         "PUBLISHED BASELINE · Jensen–Shannon divergence = 0.5·KL(p‖m)+0.5·KL(q‖m), symmetric, bounded [0,log2]. Softer than forward-KL, more stable.",
    "l1":          "PUBLISHED BASELINE · L1 / Total-Variation: 0.5·Σ|p-q|. Simple, symmetric, bounded [0,1]. Weak training signal; use as ablation only.",
    # Novel flat losses from this work — experimental, not yet published
    "ebe":         "EXPERIMENTAL FLAT · Expected Block Efficiency (flat): –E[τ+1] over a block of L tokens. Novel contribution — direct optimisation of the spec-dec metric on offline data. Not yet published.",
    "ebe_single":  "EXPERIMENTAL FLAT · Per-token acceptance rate –mean α_t, α_t=min(1,p/q). Novel ablation of block-level EBE; no cumprod, no KL. Not yet published.",
}
_TREE_DESCRIPTIONS = {
    "kl_tree":        "EXPERIMENTAL · KL(p‖q) summed over tree nodes with chain weights. Generic; works with any verifier. Strongest baseline in this codebase.",
    "rev_kl_tree":    "EXPERIMENTAL · Reverse KL(q‖p) over tree nodes. Mode-seeking tree variant; tends to over-sharpen.",
    "jsd_tree":       "EXPERIMENTAL · JSD over tree nodes. Symmetric; more stable than kl_tree / rev_kl_tree.",
    "bv_tree":        "EXPERIMENTAL · BV-verifier-aligned surrogate. Surrogate path weight detached; weak gradient at nodes where student over-proposes.",
    "gbv_tree":       "EXPERIMENTAL · GBV-verifier-aligned (this work). Selects the greedy-best path; similar detachment weakness as bv_tree.",
    "traversal_tree": "EXPERIMENTAL · Traversal-verifier-aligned. Follows the verifier's left-to-right path walk; gradient only on accepted prefix.",
    "naive_tree":     "EXPERIMENTAL · Naive-verifier-aligned (independent per-path). Ignores path interactions; simple but ignores correlation.",
    "nss_tree":       "EXPERIMENTAL · NSS-verifier-aligned. No-shortcut-sampling style; penalises student for any path that exceeds teacher probability.",
    "specinfer_tree": "EXPERIMENTAL · SpecInfer-verifier-aligned. Multi-candidate best-path acceptance.",
    "spectr_tree":    "EXPERIMENTAL · SpecTr-verifier-aligned. Optimal transport over tree; ρ gradient detached (LP not differentiable).",
    "khisti_tree":    "EXPERIMENTAL · Khisti-verifier-aligned (LP-based). Tightest acceptance bound; scipy LP solver — very slow.",
    "ebe_tree":       "EXPERIMENTAL · ON-POLICY EBE on the student's own draft tree. α_i=min(1,p[t_i]/q[t_i]) along each sampled path; exact match to naive verifier for K=1, approximate for K>1. Ablation: contrast with flat ebe (off-policy, uses training tokens) and bv_tree (full-vocab integral, theoretically stronger).",
}

@app.route("/api/losses")
def api_losses():
    var_desc = LV.variant_descriptions()
    # merge all descriptions into one dict
    all_desc = {**_FLAT_DESCRIPTIONS, **_TREE_DESCRIPTIONS, **{k: f"EXPERIMENTAL VARIANT · {v}" for k, v in var_desc.items()}}
    uses_lam = [k for k, v in var_desc.items() if 'λ' in v]
    return jsonify({
        "flat":         sorted(_FLAT_DESCRIPTIONS.keys()),
        "production":   sorted(TREE_LOSS_NAMES),
        "variants":     list(LV.VARIANTS.keys()),
        "descriptions": all_desc,
        "uses_lam":     uses_lam,
        "lam_defaults": LV.VARIANT_LAM_DEFAULTS,
        "map":          LOSS_TO_VERIFIER,
    })


def _run_job(job_id, kwargs):
    def prog(phase, cur, total):
        with _JOBS_LOCK:
            j = JOBS.get(job_id)
            if j is not None:
                j.update(phase=phase, cur=cur, total=total)
    try:
        data = build_session(progress=prog, **kwargs)
        with _JOBS_LOCK:
            JOBS[job_id].update(phase="done", cur=kwargs["steps"], done=True, data=data)
    except Exception as e:  # noqa: BLE001
        with _JOBS_LOCK:
            JOBS[job_id].update(phase="error", done=True,
                                error=f"{type(e).__name__}: {e}",
                                trace=traceback.format_exc())


@app.route("/api/train_start")
def api_train_start():
    a = request.args
    steps = min(int(a.get("steps", 80)), 300)
    kwargs = dict(
        loss_name=a.get("loss", "kl_tree"),
        vocab=max(6, int(a.get("vocab", 32))),
        K=int(a.get("K", 3)),
        L=int(a.get("L", 4)),
        steps=steps,
        lr=float(a.get("lr", 0.1)),
        be_trials=min(int(a.get("be_trials", 15)), 200),
        be_every=max(1, int(a.get("be_every", 5))),
        include_khisti=a.get("khisti", "0") in ("1", "true", "on"),
        lam=max(0.0, float(a.get("lam", 0.5))),
        teacher_peak=max(0.5, float(a.get("teacher_peak", 2.5))),
        student_scale=max(0.05, float(a.get("student_scale", 0.8))),
        teacher_temp=max(0.1, float(a.get("teacher_temp", 1.0))),
        student_temp=max(0.1, float(a.get("student_temp", 1.0))),
    )
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        # keep memory bounded: drop everything but the most recent few jobs
        for old in list(JOBS.keys())[:-3]:
            JOBS.pop(old, None)
        JOBS[job_id] = {"phase": "starting", "cur": 0, "total": steps,
                        "done": False, "error": None, "data": None}
    threading.Thread(target=_run_job, args=(job_id, kwargs), daemon=True).start()
    return jsonify({"job": job_id, "total": steps})


@app.route("/api/progress")
def api_progress():
    job = request.args.get("job", "")
    with _JOBS_LOCK:
        j = JOBS.get(job)
        if j is None:
            return jsonify({"error": "unknown job", "done": True}), 404
        return jsonify({"phase": j["phase"], "cur": j["cur"], "total": j["total"],
                        "done": j["done"], "error": j["error"]})


@app.route("/api/result")
def api_result():
    job = request.args.get("job", "")
    with _JOBS_LOCK:
        j = JOBS.get(job)
        if j is None:
            return jsonify({"ok": False, "error": "unknown job"}), 404
        if not j["done"]:
            return jsonify({"ok": False, "error": "not finished"}), 409
        if j["error"]:
            return jsonify({"ok": False, "error": j["error"],
                            "trace": j.get("trace", "")}), 500
        return jsonify({"ok": True, "data": j["data"]})


# ---------------------------------------------------------------------------
# Compare-all: train every core loss with the same hyperparams, store results
# ---------------------------------------------------------------------------

# Core losses shown in the compare view.  Flat baselines first so the chart
# reads "baseline → tree" left-to-right.
COMPARE_LOSSES = [
    "forward_kl",                                          # published flat baseline
    "kl_tree", "bv_tree", "gbv_tree",                     # core tree losses
    "traversal_tree", "ebe_tree",                          # more tree losses
]

_COMPARE_STORE: dict = {}          # loss_name → session data
_COMPARE_LOCK  = threading.Lock()


def _run_compare_all(job_id: str, base_kwargs: dict):
    """Train each loss in COMPARE_LOSSES sequentially and store results."""
    n_losses = len(COMPARE_LOSSES)
    steps     = base_kwargs["steps"]
    with _COMPARE_LOCK:
        _COMPARE_STORE.clear()

    for li, loss_name in enumerate(COMPARE_LOSSES):
        def _prog(phase, cur, total, _li=li, _ln=loss_name):
            with _JOBS_LOCK:
                j = JOBS.get(job_id)
                if j:
                    j["phase"] = f"[{_li+1}/{n_losses}] {_ln} — {phase}"
                    j["cur"]   = _li * steps + cur
        try:
            data = build_session(progress=_prog, **{**base_kwargs, "loss_name": loss_name})
        except Exception as exc:  # noqa: BLE001
            data = {"error": f"{type(exc).__name__}: {exc}",
                    "loss_name": loss_name, "tree_after": {}, "flat_after": {}, "loss": []}
        with _COMPARE_LOCK:
            _COMPARE_STORE[loss_name] = data
        # mark step done even on error so progress keeps moving
        with _JOBS_LOCK:
            j = JOBS.get(job_id)
            if j:
                j["cur"] = (li + 1) * steps

    with _JOBS_LOCK:
        j = JOBS.get(job_id)
        if j:
            j.update(phase="done", cur=n_losses * steps, done=True)


@app.route("/api/train_all_start")
def api_train_all_start():
    a = request.args
    steps = min(int(a.get("steps", 80)), 200)
    base_kwargs = dict(
        vocab         = max(6,   int(a.get("vocab",          32))),
        K             = int(a.get("K",             3)),
        L             = int(a.get("L",             4)),
        steps         = steps,
        lr            = float(a.get("lr",          0.1)),
        be_trials     = min(int(a.get("be_trials", 15)), 200),
        be_every      = max(1,   int(a.get("be_every",       5))),
        include_khisti= False,   # always off for batch compare (too slow)
        lam           = max(0.0, float(a.get("lam",          0.5))),
        teacher_peak  = max(0.5, float(a.get("teacher_peak", 2.5))),
        student_scale = max(0.05,float(a.get("student_scale",0.8))),
        teacher_temp  = max(0.1, float(a.get("teacher_temp", 1.0))),
        student_temp  = max(0.1, float(a.get("student_temp", 1.0))),
    )
    job_id = uuid.uuid4().hex[:12]
    total  = len(COMPARE_LOSSES) * steps
    with _JOBS_LOCK:
        for old in list(JOBS.keys())[:-3]:
            JOBS.pop(old, None)
        JOBS[job_id] = {"phase": "starting", "cur": 0, "total": total,
                        "done": False, "error": None, "data": None}
    threading.Thread(target=_run_compare_all,
                     args=(job_id, base_kwargs), daemon=True).start()
    return jsonify({"job": job_id, "total": total, "losses": COMPARE_LOSSES})


@app.route("/api/compare")
def api_compare():
    with _COMPARE_LOCK:
        return jsonify({"losses": COMPARE_LOSSES, "store": dict(_COMPARE_STORE)})


# ---------------------------------------------------------------------------
# Single-page UI
# ---------------------------------------------------------------------------

PAGE = r"""
<!doctype html><html><head><meta charset="utf-8">
<title>Tree-loss step debugger</title>
<style>
  :root { --bg:#0f1421; --panel:#19223a; --ink:#e7ecf5; --mut:#90a0c0; --acc:#3fa7ff; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif;
         background:var(--bg); color:var(--ink); }
  header { padding:10px 16px; background:var(--panel); display:flex; gap:12px;
           align-items:center; flex-wrap:wrap; border-bottom:1px solid #26314f; }
  h1 { font-size:15px; margin:0 10px 0 0; font-weight:700; }
  label { font-size:12px; color:var(--mut); margin-right:4px; }
  select,input { background:#0d1422; color:var(--ink); border:1px solid #2b3960;
                 border-radius:6px; padding:4px 6px; font-size:13px; }
  input[type=number]{ width:60px; }
  button { background:var(--acc); color:#04111f; border:0; border-radius:6px;
           padding:6px 12px; font-weight:700; cursor:pointer; font-size:13px; }
  button.ghost { background:#23304f; color:var(--ink); }
  button:disabled { opacity:.4; cursor:not-allowed; }
  .wrap { display:grid; grid-template-columns: 1.4fr 1fr; gap:12px; padding:12px; }
  .col { display:flex; flex-direction:column; gap:12px; }
  .card { background:var(--panel); border:1px solid #26314f; border-radius:10px; padding:12px; }
  .card h2 { font-size:12px; margin:0 0 8px; color:var(--mut); font-weight:600;
             text-transform:uppercase; letter-spacing:.04em; }
  svg { width:100%; display:block; }
  circle.node { stroke:#0a0f1c; stroke-width:1.5; transition: fill 0.6s ease; }
  circle.node.sel { stroke:#3fa7ff; stroke-width:3.5; }
  line.edge { stroke:#42507a; stroke-width:1.6; }
  line.edge.sel { stroke:#3fa7ff; stroke-width:3; }
  text.tok { fill:#dfe7f7; font-size:13px; font-weight:700; text-anchor:middle;
             dominant-baseline:central; pointer-events:none; }
  text.elab { fill:#7a89b3; font-size:10px; text-anchor:middle; }
  .controls { display:flex; gap:8px; align-items:center; flex-wrap:wrap; padding:0 12px 4px; }
  .controls .big { font-variant-numeric:tabular-nums; font-size:13px; color:var(--mut); }
  input[type=range]{ width:100%; }
  .stat { display:flex; gap:16px; margin:4px 0; flex-wrap:wrap; }
  .stat div { font-size:13px; } .stat b{ color:#fff; font-variant-numeric:tabular-nums; }
  .legend { font-size:11px; color:var(--mut); margin-top:6px; }
  .swatch { display:inline-block; width:34px; height:10px; border-radius:3px;
            background:linear-gradient(90deg,#dc322f,#e8c020,#28a046); vertical-align:middle; }
  #info { font-size:12px; color:#bcd; min-height:16px; }
  #err { color:#ff6b6b; font-size:12px; padding:0 12px; white-space:pre-wrap; }
  .muted{ color:var(--mut); font-size:12px; }
  .vlegend { display:flex; flex-wrap:wrap; gap:8px 12px; margin-top:6px; font-size:11px; }
  .vlegend span { display:flex; align-items:center; gap:4px; }
  .vlegend i { width:14px; height:3px; display:inline-block; border-radius:2px; }
  .bd { font-size:12px; font-variant-numeric:tabular-nums; }
  .bd table { border-collapse:collapse; width:100%; margin-bottom:8px; }
  .bd th,.bd td { padding:2px 6px; text-align:right; border-bottom:1px solid #243152; }
  .bd th { color:var(--mut); font-weight:600; }
  .bd .pl { color:#8fa3cc; text-align:left; }
</style></head><body>
<header>
  <h1>🌳 Tree-loss step debugger</h1>
  <span><label title="Loss / variant to train">loss</label><select id="loss"></select></span>

  <span style="border-left:1px solid #2b3960;padding-left:10px">
    <label title="Vocabulary size V ≥ 6.  All tokens are integer IDs 0…V-1.  K, L, and model size all scale with V.">V</label>
    <input type="number" id="vocab" value="32" min="6" max="64" style="width:52px"
           oninput="clampToVocab()">
  </span>
  <span>
    <label title="K = number of independent draft paths sampled per step. Must be &lt; V (max distinct next-tokens). Higher K → richer tree but slower verify.">K</label>
    <input type="number" id="K" value="3" min="1" max="5" style="width:48px"
           oninput="clampKL(this,'K')">
  </span>
  <span>
    <label title="L = draft depth — tokens drafted per path. Must be &lt; V. Higher L → longer sequences, bigger tree, more compute.">L</label>
    <input type="number" id="L" value="4" min="1" max="5" style="width:48px"
           oninput="clampKL(this,'L')">
  </span>

  <span style="border-left:1px solid #2b3960;padding-left:10px">
    <label title="Teacher 'peak' p — controls how peaked (deterministic) the frozen teacher model is.  Higher → more concentrated next-token distribution → harder for student to match.  Analogue of teacher model size: bigger model → higher peak.  Default 4.0.">T-peak</label>
    <input type="number" id="teacher_peak" value="2.5" min="0.5" max="20" step="0.5" style="width:52px">
  </span>
  <span>
    <label title="Teacher inference temperature τ_T — sharpens (< 1) or flattens (> 1) the teacher's distribution at inference time, independently of its logit scale.  Lower τ_T = more deterministic teacher = stronger capability gap.  Default 1.0.">T-temp</label>
    <input type="number" id="teacher_temp" value="1.0" min="0.1" max="5" step="0.1" style="width:48px">
  </span>
  <span style="border-left:1px solid #2b3960;padding-left:10px">
    <label title="Student init scale s — standard deviation of the student's initial random logit matrix.  Smaller → student starts closer to uniform (easier).  Larger → more random start (harder).  Analogue of student model size: weaker student → lower scale.  Default 0.5.">S-scale</label>
    <input type="number" id="student_scale" value="0.8" min="0.05" max="5" step="0.05" style="width:52px">
  </span>
  <span>
    <label title="Student inference temperature τ_S — sharpens (< 1) or flattens (> 1) the student's distribution at draft time.  Higher τ_S = more diffuse student = harder to achieve high BE.  Default 1.0.">S-temp</label>
    <input type="number" id="student_temp" value="1.0" min="0.1" max="5" step="0.1" style="width:48px">
  </span>

  <span style="border-left:1px solid #2b3960;padding-left:10px">
    <label title="Training steps (gradient updates).  Each step samples a fresh on-policy tree.">steps</label>
    <input type="number" id="steps" value="80" min="10" max="300">
  </span>
  <span>
    <label title="Adam learning rate for the student's logit matrix W.">lr</label>
    <input type="number" id="lr" value="0.1" step="0.01" min="0.001">
  </span>
  <span id="lam_wrap">
    <label id="lam_label" title="KL(p‖q) anchor weight λ — only active for *_kl and *_faithful variants. 0 = no anchor (raw variant).  Ignored by production losses and full-grad / IFT variants.">KL λ</label>
    <input type="number" id="lam" value="0.1" step="0.05" min="0" style="width:56px">
  </span>
  <span>
    <label title="Include khisti verifier in charts — uses a scipy LP solver (~0.5 s/call), much slower.">+khisti</label>
    <input type="checkbox" id="khisti">
  </span>
  <button id="train">Train</button>
  <button id="trainall" class="ghost" title="Train ALL core losses with these hyperparams, then show a side-by-side comparison chart. Takes ~N × single-run time.">▶ Compare All</button>
  <span id="status" class="muted"></span>
</header>
<div id="treewarn" style="display:none;padding:4px 16px;background:#2a1a10;border-bottom:1px solid #8b4513;color:#f4a261;font-size:12px">
  ⚗ <b>Experimental loss</b> — not yet published. Published standard baselines are:
  <code>forward_kl</code>, <code>reverse_kl</code>, <code>jsd</code>, <code>l1</code>.
  (<code>ebe</code> and <code>ebe_single</code> are novel flat contributions from this work, also not yet published.)
</div>
<div id="vardesc" class="muted" style="padding:0 16px 6px"></div>
<div id="progwrap" style="display:none; padding:0 16px 8px">
  <div style="background:#0d1422; border:1px solid #2b3960; border-radius:6px; height:16px; overflow:hidden">
    <div id="progbar" style="height:100%; width:0%; background:linear-gradient(90deg,#3fa7ff,#06d6a0); transition:width .25s"></div>
  </div>
  <div id="progtext" class="muted" style="margin-top:4px"></div>
</div>
<div id="err"></div>

<div class="controls">
  <button class="ghost" id="first">⏮</button>
  <button class="ghost" id="back">◀ Step</button>
  <button id="play">► Play</button>
  <button class="ghost" id="fwd">Step ▶</button>
  <button class="ghost" id="last">⏭</button>
  <span><label>jump to step</label><input type="number" id="jump" value="100" min="0">
       <button class="ghost" id="go">Go</button></span>
  <span><label>speed</label><input type="range" id="speed" min="60" max="1200" value="450" style="width:110px"></span>
  <span class="big" id="stepread">step — / —</span>
</div>
<input type="range" id="scrub" min="0" max="0" value="0" style="width:calc(100% - 24px); margin:0 12px;">

<div class="card" id="whatami" style="margin:6px 12px">
  <h2>What you're looking at</h2>
  <div class="muted" style="font-size:12px; line-height:1.5">
    A toy <b>speculative-decoding</b> simulation on a <b>V=<span id="wa_V">6</span></b>-token vocabulary —
    token labels are <b>integer IDs 0 … V−1, not text</b>. Two order-1 Markov
    "models" (each just a V×V logit matrix):
    <span style="color:#28a046">●</span> <b>Teacher</b> = frozen <b>target</b> model <i>p</i>
      (peak=<span id="wa_peak">4.0</span> → logit concentration, τ_T=<span id="wa_ttemp">1.0</span> → inference sharpness;
      together these simulate teacher model <i>size</i>: high peak + low τ_T = large confident teacher) ·
    <span style="color:#3fa7ff">●</span> <b>Student</b> = trainable <b>draft</b> model <i>q</i>
      (init scale=<span id="wa_scale">0.5</span> → how random its start is, τ_S=<span id="wa_stemp">1.0</span> → draft sharpness;
      low scale + high τ_S = small weak student; matrix W updated every training step).
    <br>
    The tree = <b id="wa_K">K</b> draft paths, depth <b id="wa_L">L</b>, from
    <b>pending token <span id="wa_pending">·</span></b>.
    Node label = last token of path; colour = chain weight w=∏min(1,p/q):
    <span style="color:#dc322f">red = student over-proposes</span>,
    <span style="color:#28a046">green = teacher accepts</span>.
    Training pushes q→p → tree greens → block efficiency rises.
  </div>
</div>

<div class="wrap">
  <div class="col">
    <div class="card">
      <h2>Draft tree — node colour = chain weight w = Π min(1, p/q)</h2>
      <svg id="tree" viewBox="0 0 600 380" preserveAspectRatio="xMidYMid meet"></svg>
      <div class="legend"><span class="swatch"></span>
        &nbsp;red = draft over-proposes (rejected) &nbsp;→&nbsp; green = target accepts</div>
    </div>
    <div class="card">
      <h2>Block efficiency under EVERY verifier (tree, current K)</h2>
      <svg id="bechart" viewBox="0 0 540 200"></svg>
      <div class="vlegend" id="velegend"></div>
    </div>
  </div>

  <div class="col">
    <div class="card">
      <h2>Metrics</h2>
      <div class="stat">
        <div title="Raw training loss fed to the optimizer (gradient signal). More negative = student better aligned to teacher.">train loss <b id="m_loss">—</b></div>
        <div>matched verifier <b id="m_ver">—</b></div>
        <div title="Block efficiency under the matched verifier at the current step.">BE(matched) <b id="m_be">—</b></div>
      </div>
      <div id="info" title="Tracer decomposition: pure tree-acceptance term (excludes KL-anchor λ). Differs from train loss when λ > 0."></div>
      <h2 style="margin-top:12px">Loss</h2>
      <svg id="losschart" viewBox="0 0 360 110"></svg>
    </div>
    <div class="card">
      <h2>Per-node loss breakdown — current step</h2>
      <div class="bd" id="bd"></div>
    </div>
    <div class="card">
      <h2>Flat (K=1) BE — before vs after  (does the loss help flat verifiers too?)</h2>
      <svg id="flatchart" viewBox="0 0 540 200"></svg>
      <div class="legend">flat = single chain (K=1): <b>naive</b>=vanilla speculative sampling, <b>bv</b>=block verification.</div>
    </div>
  </div>
</div>

<!-- ── Compare All panel (hidden until train_all completes) ── -->
<div id="cmppanel" style="display:none; padding:12px; border-top:2px solid #3fa7ff22">
  <div style="display:flex; align-items:center; gap:16px; margin-bottom:10px">
    <h2 style="margin:0; font-size:13px; color:#90a0c0; text-transform:uppercase; letter-spacing:.05em">
      ▤ Compare All — final block efficiency after training</h2>
    <span id="cmpstatus" class="muted"></span>
  </div>
  <div style="display:grid; grid-template-columns:1.6fr 1fr; gap:12px">
    <div class="card">
      <h2>Final BE per verifier × loss  <span class="muted" style="font-weight:400;font-size:10px">(ghost = before training)</span></h2>
      <svg id="cmp_be_chart" viewBox="0 0 700 220"></svg>
      <div class="vlegend" id="cmp_legend"></div>
    </div>
    <div class="card">
      <h2>BE improvement (Δ after − before)  <span class="muted" style="font-weight:400;font-size:10px">higher = better</span></h2>
      <svg id="cmp_delta_chart" viewBox="0 0 360 200"></svg>
      <div class="vlegend" id="cmp_delta_legend"></div>
    </div>
  </div>
  <div id="cmp_load_row" style="display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; font-size:12px"></div>
</div>

<script>
const $ = id => document.getElementById(id);
let S = null, cur = 0, playing = false, timer = null;
const VCOL = {naive:'#7fd1ff', nss:'#b48bff', bv:'#ff9f6b', specinfer:'#ffd166',
              spectr:'#06d6a0', traversal:'#ef476f', gbv:'#3fa7ff', max:'#9aa7c7', khisti:'#f78fb3'};

function color(w){
  w = Math.max(0, Math.min(1, w)); let r,g,b;
  if(w < 0.5){ const t=w/0.5; r=220+(232-220)*t; g=50+(192-50)*t; b=47+(32-47)*t; }
  else { const t=(w-0.5)/0.5; r=232+(40-232)*t; g=192+(160-192)*t; b=32+(70-32)*t; }
  return `rgb(${r|0},${g|0},${b|0})`;
}
const NS='http://www.w3.org/2000/svg';
function el(tag,attrs){ const e=document.createElementNS(NS,tag);
  for(const k in attrs) e.setAttribute(k,attrs[k]); return e; }

let ALLDESC = {}, USES_LAM = new Set(), LAM_DEFAULTS = {};

function clampToVocab(){
  const V = Math.max(6, parseInt($('vocab').value)||6);
  const Kmax = V - 1, Lmax = V - 1;
  const Ki = $('K'), Li = $('L');
  Ki.max = Kmax; if(parseInt(Ki.value) > Kmax) Ki.value = Kmax;
  Li.max = Lmax; if(parseInt(Li.value) > Lmax) Li.value = Lmax;
}
function clampKL(inp, which){
  const V = Math.max(6, parseInt($('vocab').value)||6);
  const max = V - 1;
  inp.max = max;
  const v = parseInt(inp.value);
  if(v > max){ inp.value = max; inp.style.outline='2px solid #e55a4e'; }
  else { inp.style.outline=''; }
}

function updateLamState(){
  const loss = $('loss').value;
  const relevant = USES_LAM.has(loss);
  const wrap = $('lam_wrap'), lbl = $('lam_label'), inp = $('lam');
  wrap.style.opacity = relevant ? '1' : '0.35';
  inp.disabled = !relevant;
  lbl.title = relevant
    ? 'KL(p‖q) anchor weight λ — active for this variant.  0 = no anchor.'
    : 'KL λ not used by this loss — parameter is ignored.';
}

function makeOpt(name, desc){
  const o = document.createElement('option');
  o.value = name; o.textContent = name;
  if(desc) o.title = desc;
  return o;
}

async function loadLosses(){
  const r = await (await fetch('/api/losses')).json();
  ALLDESC   = r.descriptions || {};
  USES_LAM  = new Set(r.uses_lam || []);
  const sel = $('loss');

  const gf = document.createElement('optgroup');
  gf.label = '📗 flat losses — PUBLISHED (sequence-level, no tree)';
  (r.flat||[]).forEach(l=>sel.appendChild(makeOpt(l, ALLDESC[l])));  // flat at top level for easy access

  const gt = document.createElement('optgroup');
  gt.label = '⚗ tree losses — EXPERIMENTAL (under active research, not published)';
  r.production.forEach(l=>gt.appendChild(makeOpt(l, ALLDESC[l])));
  sel.appendChild(gt);

  const gv = document.createElement('optgroup');
  gv.label = '🔬 tree loss variants — EXPERIMENTAL VARIANTS (debug/research use only)';
  (r.variants||[]).forEach(l=>gv.appendChild(makeOpt(l, ALLDESC[l])));
  sel.appendChild(gv);

  sel.value = 'kl_tree';
  sel.onchange = ()=>{ showDesc(); updateLamState(); };
  clampToVocab(); showDesc(); updateLamState();
}
let _FLAT_LOSSES = new Set();

function showDesc(){
  const loss = $('loss').value;
  const d = ALLDESC[loss] || '';
  const isPublished  = d.startsWith('PUBLISHED BASELINE');
  const isExpFlat    = d.startsWith('EXPERIMENTAL FLAT');
  const isFlatLoss   = isPublished || isExpFlat;      // uses flat training loop
  const isExpVar     = d.startsWith('EXPERIMENTAL VARIANT');
  const isExpTree    = d.startsWith('EXPERIMENTAL') && !isExpVar && !isExpFlat;
  const badge = isPublished ? '<span style="color:#28a046;font-weight:bold">PUBLISHED BASELINE</span>'
              : isExpFlat   ? '<span style="color:#e55a4e;font-weight:bold">EXPERIMENTAL FLAT</span>'
              : isExpVar    ? '<span style="color:#f4a261;font-weight:bold">EXPERIMENTAL VARIANT</span>'
              : isExpTree   ? '<span style="color:#e55a4e;font-weight:bold">EXPERIMENTAL</span>'
              : '';
  const rest = d.replace(/^(PUBLISHED BASELINE|EXPERIMENTAL FLAT|EXPERIMENTAL VARIANT|EXPERIMENTAL)\s·\s/,'');

  if(isFlatLoss){
    const note = isPublished
      ? '<span style="color:#5bc4a0">✔ Standard published baseline. Trains on token-pair sequences. BE evaluated under all tree verifiers.</span>'
      : '<span style="color:#f4a261">⚗ Novel flat loss from this work — not yet published. Trains on token-pair sequences. BE evaluated under all tree verifiers.</span>';
    $('vardesc').innerHTML = badge + ' · ' + rest + '<br>' + note;
    $('train').disabled = false;
    $('train').title = 'Train as flat sequence-level loss. BE is evaluated under tree verifiers for fair comparison.';
    $('train').textContent = '▶ Train (flat)';
    $('treewarn').style.display = isExpFlat ? 'block' : 'none';
  } else {
    $('vardesc').innerHTML = d ? (badge + (badge?' · ':'')+rest) : '';
    $('train').disabled = false;
    $('train').title = '';
    $('train').textContent = 'Train';
    $('treewarn').style.display = 'block';
  }
}

function buildTree(){
  const svg = $('tree'); svg.innerHTML='';
  const W=600,H=380,padX=70,padTop=40,padBot=30;
  const sx = x => padX + x*(W-2*padX);
  const sy = y => padTop + y*(H-padTop-padBot);
  const ns = S.nodes, es = S.edges;
  const maxD = Math.max(...ns.map(n=>n.depth), 1);

  // depth ruler on the left: d0 = pending token, d1..dL = drafted tokens
  for(let d=0; d<=maxD; d++){
    const y = sy(d/maxD);
    const lab = d===0 ? 'd0 · pending' : ('d'+d);
    svg.appendChild(Object.assign(el('text',{x:4,y:y+4,fill:'#6f7ea8','font-size':10}),{textContent:lab}));
    svg.appendChild(el('line',{x1:padX-8,y1:y,x2:W-10,y2:y,stroke:'#1c2740','stroke-dasharray':'2 4'}));
  }
  svg.appendChild(Object.assign(el('text',{x:4,y:14,fill:'#6f7ea8','font-size':10}),
                  {textContent:'↓ each level = +1 drafted token'}));

  es.forEach((e,i)=>{
    const u=ns[e.u], v=ns[e.v];
    const ln=el('line',{x1:sx(u.x),y1:sy(u.y),x2:sx(v.x),y2:sy(v.y),class:'edge',id:'e'+i});
    svg.appendChild(ln);
    svg.appendChild(Object.assign(el('text',{x:(sx(u.x)+sx(v.x))/2+6,y:(sy(u.y)+sy(v.y))/2,class:'elab'}),{textContent:e.token}));
  });
  ns.forEach((n,i)=>{
    const c=el('circle',{cx:sx(n.x),cy:sy(n.y),r:17,class:'node',id:'n'+i});
    const tip=el('title',{}); tip.id='nt'+i; c.appendChild(tip);  // hover tooltip, filled in render()
    svg.appendChild(c);
    svg.appendChild(Object.assign(el('text',{x:sx(n.x),y:sy(n.y),class:'tok'}),{textContent:n.token}));
    if(n.depth===0){
      svg.appendChild(Object.assign(el('text',{x:sx(n.x),y:sy(n.y)-24,fill:'#9aa7c7','font-size':10,'text-anchor':'middle'}),
                      {textContent:'pending token (the prompt)'}));
    }
  });
}

function lineChart(svgId, ys, idx, col, ylab){
  const svg=$(svgId); svg.innerHTML=''; const W=360,H=110,pad=24;
  if(!ys.length) return;
  let lo=Math.min(...ys), hi=Math.max(...ys); if(hi-lo<1e-9){hi=lo+1;}
  const sx=i=> pad + i*(W-2*pad)/Math.max(ys.length-1,1);
  const sy=v=> H-pad - (v-lo)*(H-2*pad)/(hi-lo);
  svg.appendChild(el('line',{x1:pad,y1:H-pad,x2:W-pad,y2:H-pad,stroke:'#33406a'}));
  let d=''; ys.forEach((v,i)=>{ d += (i?'L':'M')+sx(i)+' '+sy(v)+' '; });
  svg.appendChild(el('path',{d,fill:'none',stroke:col,'stroke-width':2}));
  svg.appendChild(el('circle',{cx:sx(idx),cy:sy(ys[idx]),r:4,fill:'#fff'}));
  svg.appendChild(el('line',{x1:sx(idx),y1:pad-6,x2:sx(idx),y2:H-pad,stroke:'#ffffff44'}));
  svg.appendChild(Object.assign(el('text',{x:2,y:pad-4,fill:'#7a89b3','font-size':9}),{textContent:hi.toFixed(2)}));
  svg.appendChild(Object.assign(el('text',{x:2,y:H-pad,fill:'#7a89b3','font-size':9}),{textContent:lo.toFixed(2)}));
  // axis labels
  svg.appendChild(Object.assign(el('text',{x:W/2,y:H-2,fill:'#6f7ea8','font-size':9,'text-anchor':'middle'}),{textContent:'training step →'}));
  if(ylab) svg.appendChild(Object.assign(el('text',{x:8,y:12,fill:'#6f7ea8','font-size':9}),{textContent:ylab}));
}

function multiLine(svgId, series, idx, matched){
  const svg=$(svgId); svg.innerHTML=''; const W=540,H=200,pad=26;
  const names=Object.keys(series); if(!names.length) return;
  let all=[]; names.forEach(n=>all=all.concat(series[n]));
  let lo=Math.min(...all), hi=Math.max(...all); if(hi-lo<1e-9){hi=lo+1;}
  const L=series[names[0]].length;
  const sx=i=> pad + i*(W-2*pad)/Math.max(L-1,1);
  const sy=v=> H-pad - (v-lo)*(H-2*pad)/(hi-lo);
  svg.appendChild(el('line',{x1:pad,y1:H-pad,x2:W-pad,y2:H-pad,stroke:'#33406a'}));
  // y gridlines
  [lo,(lo+hi)/2,hi].forEach(v=>{ svg.appendChild(Object.assign(el('text',{x:2,y:sy(v)+3,fill:'#7a89b3','font-size':9}),{textContent:v.toFixed(2)})); });
  names.forEach(n=>{
    let d=''; series[n].forEach((v,i)=>{ d += (i?'L':'M')+sx(i)+' '+sy(v)+' '; });
    svg.appendChild(el('path',{d,fill:'none',stroke:VCOL[n]||'#888','stroke-width': n===matched?3:1.5,
                               opacity: n===matched?1:0.8}));
    const v=series[n][idx];
    svg.appendChild(el('circle',{cx:sx(idx),cy:sy(v),r: n===matched?4:2.5,fill:VCOL[n]||'#888'}));
  });
  svg.appendChild(el('line',{x1:sx(idx),y1:pad-6,x2:sx(idx),y2:H-pad,stroke:'#ffffff33'}));
  // axis labels
  svg.appendChild(Object.assign(el('text',{x:W/2,y:H-4,fill:'#6f7ea8','font-size':10,'text-anchor':'middle'}),{textContent:'training step →  (vertical line = current step)'}));
  svg.appendChild(Object.assign(el('text',{x:10,y:14,fill:'#6f7ea8','font-size':10}),{textContent:'↑ block efficiency (accepted tokens / call)'}));
}

function groupedBars(svgId, before, after){
  const svg=$(svgId); svg.innerHTML=''; const W=540,H=200,pad=30;
  const names=Object.keys(after); if(!names.length) return;
  let all=names.map(n=>after[n]).concat(names.map(n=>before[n]??0));
  let hi=Math.max(...all,1);
  const bw=(W-2*pad)/names.length, sy=v=> H-pad-(v/hi)*(H-2*pad);
  svg.appendChild(el('line',{x1:pad,y1:H-pad,x2:W-pad,y2:H-pad,stroke:'#33406a'}));
  names.forEach((n,i)=>{
    const x=pad+i*bw;
    const b=before[n]??0, a=after[n]??0;
    svg.appendChild(el('rect',{x:x+bw*0.12,y:sy(b),width:bw*0.34,height:H-pad-sy(b),fill:'#5a6890'}));
    svg.appendChild(el('rect',{x:x+bw*0.50,y:sy(a),width:bw*0.34,height:H-pad-sy(a),fill:VCOL[n]||'#3fa7ff'}));
    svg.appendChild(Object.assign(el('text',{x:x+bw*0.5,y:H-pad+12,fill:'#9aa7c7','font-size':9,'text-anchor':'middle'}),{textContent:n}));
    svg.appendChild(Object.assign(el('text',{x:x+bw*0.5,y:sy(a)-3,fill:'#cdd8f0','font-size':9,'text-anchor':'middle'}),{textContent:a.toFixed(1)}));
  });
  svg.appendChild(Object.assign(el('text',{x:W-90,y:14,fill:'#5a6890','font-size':10}),{textContent:'■ before'}));
  svg.appendChild(Object.assign(el('text',{x:W-44,y:14,fill:'#3fa7ff','font-size':10}),{textContent:'■ after'}));
  svg.appendChild(Object.assign(el('text',{x:10,y:14,fill:'#6f7ea8','font-size':10}),{textContent:'↑ block efficiency (K=1)'}));
}

// ── Compare-all chart helpers ──────────────────────────────────────────────

// Palette for loss functions in the compare view
const LOSS_COL = {
  forward_kl:      '#94c2f5',
  kl_tree:         '#fd7e14',
  bv_tree:         '#28a046',
  gbv_tree:        '#3fa7ff',
  traversal_tree:  '#ef476f',
  ebe_tree:        '#f5c542',
};
function lossCol(n){ return LOSS_COL[n] || '#aaa'; }

/** Grouped bar chart: x=verifier, one bar per loss, showing final tree BE. */
function compareBEChart(svgId, store){
  const svg=$(svgId); svg.innerHTML='';
  const lossNames = Object.keys(store).filter(k=>!store[k].error);
  if(!lossNames.length){ svg.innerHTML='<text x="10" y="20" fill="#888" font-size="11">no data yet</text>'; return; }

  // Collect verifier names from first session
  const first = store[lossNames[0]];
  const verifiers = first.verifiers || [];
  if(!verifiers.length) return;

  const W=700, H=220, padL=30, padB=40, padT=16, padR=10;
  const cw = (W - padL - padR) / verifiers.length;   // width per verifier group
  const bw = cw * 0.7 / lossNames.length;             // width per bar

  // find max BE across all data
  let maxBE = 1;
  lossNames.forEach(ln=>{ verifiers.forEach(v=>{ const a=(store[ln].tree_after||{})[v]; if(a>maxBE) maxBE=a; }); });
  const sy = v => H - padB - (v / maxBE) * (H - padB - padT);

  // x-axis
  svg.appendChild(el('line',{x1:padL,y1:H-padB,x2:W-padR,y2:H-padB,stroke:'#33406a'}));
  // y gridlines
  [0, maxBE/2, maxBE].forEach(v=>{
    const y=sy(v);
    svg.appendChild(el('line',{x1:padL,y1:y,x2:W-padR,y2:y,stroke:'#1e2d4a'}));
    svg.appendChild(Object.assign(el('text',{x:padL-2,y:y+3,fill:'#7a89b3','font-size':8,'text-anchor':'end'}),{textContent:v.toFixed(1)}));
  });

  verifiers.forEach((vname, vi)=>{
    const gx = padL + vi * cw + cw * 0.15;
    lossNames.forEach((ln, li)=>{
      const be = (store[ln].tree_after||{})[vname] || 0;
      const beBefore = (store[ln].tree_before||{})[vname] || 0;
      const x = gx + li * bw;
      // ghost bar: BE before
      svg.appendChild(el('rect',{x, y:sy(beBefore), width:bw*0.85, height:H-padB-sy(beBefore),
                                  fill:'#2a3a5a', stroke:'#3a4a6a', 'stroke-width':0.5}));
      // coloured bar: BE after
      svg.appendChild(el('rect',{x, y:sy(be), width:bw*0.85, height:H-padB-sy(be),
                                  fill:lossCol(ln), opacity:0.85}));
      // value label on top bar
      if(be > maxBE*0.05)
        svg.appendChild(Object.assign(el('text',{x:x+bw*0.4,y:sy(be)-2,fill:'#ddd','font-size':7,'text-anchor':'middle'}),{textContent:be.toFixed(1)}));
    });
    // verifier label
    svg.appendChild(Object.assign(el('text',{x:gx+cw*0.35,y:H-padB+12,fill:'#9aa7c7','font-size':9,'text-anchor':'middle'}),{textContent:vname}));
  });

  // axis label
  svg.appendChild(Object.assign(el('text',{x:W/2,y:H-2,fill:'#6f7ea8','font-size':9,'text-anchor':'middle'}),{textContent:'verifier →'}));
  svg.appendChild(Object.assign(el('text',{x:12,y:padT,fill:'#6f7ea8','font-size':9}),{textContent:'↑ BE after training'}));

  // Loss legend
  const legBox = $('cmp_legend'); legBox.innerHTML='';
  lossNames.forEach(ln=>{
    const sp=document.createElement('span');
    sp.innerHTML=`<i style="background:${lossCol(ln)}"></i>${ln}`;
    legBox.appendChild(sp);
  });
  // ghost legend
  const ghost=document.createElement('span');
  ghost.innerHTML='<i style="background:#2a3a5a;border:1px solid #3a4a6a"></i>before';
  legBox.appendChild(ghost);
}

/** Bar chart: x=loss function, y=mean ΔBE across verifiers, with per-verifier breakdown lines. */
function compareDeltaChart(svgId, store){
  const svg=$(svgId); svg.innerHTML='';
  const lossNames = Object.keys(store).filter(k=>!store[k].error);
  if(!lossNames.length) return;

  const first = store[lossNames[0]];
  const verifiers = (first.verifiers || []).slice(0, 6); // cap for readability
  const W=360, H=200, padL=28, padB=40, padT=16;

  // compute mean ΔBE per loss and per-verifier ΔBE
  const deltas = {};   // loss → mean delta
  const byVer  = {};   // loss → {verifier → delta}
  lossNames.forEach(ln=>{
    const d = store[ln];
    const vDeltas = {};
    verifiers.forEach(v=>{
      const before = (d.tree_before||{})[v] || 0;
      const after  = (d.tree_after ||{})[v] || 0;
      vDeltas[v] = after - before;
    });
    byVer[ln] = vDeltas;
    const vals = Object.values(vDeltas);
    deltas[ln] = vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : 0;
  });

  const maxD = Math.max(...Object.values(deltas), 0.01);
  const minD = Math.min(...Object.values(deltas), 0);
  const range = Math.max(maxD - minD, 0.01);
  const bw = (W - padL - 10) / lossNames.length;
  const sy = v => H - padB - ((v - minD) / range) * (H - padB - padT);

  // zero line
  const zy = sy(0);
  svg.appendChild(el('line',{x1:padL,y1:zy,x2:W-10,y2:zy,stroke:'#5a6890','stroke-dasharray':'3 3'}));
  svg.appendChild(el('line',{x1:padL,y1:H-padB,x2:W-10,y2:H-padB,stroke:'#33406a'}));

  lossNames.forEach((ln, i)=>{
    const x = padL + i * bw;
    const d = deltas[ln];
    const barTop = sy(d), barBot = zy;
    const barH = Math.abs(barTop - barBot);
    // main bar
    svg.appendChild(el('rect',{x:x+bw*0.1, y:Math.min(barTop,barBot),
                                width:bw*0.8, height:Math.max(barH,1),
                                fill:lossCol(ln), opacity:0.85}));
    // per-verifier tick marks (small horizontal lines showing spread)
    verifiers.forEach(v=>{
      const vd = (byVer[ln]||{})[v] || 0;
      const vy = sy(vd);
      svg.appendChild(el('line',{x1:x+bw*0.2,y1:vy,x2:x+bw*0.8,y2:vy,
                                  stroke:VCOL[v]||'#888','stroke-width':1.5}));
    });
    // value label
    svg.appendChild(Object.assign(el('text',{x:x+bw*0.5,
      y:d>=0?barTop-3:barBot+10,
      fill:'#ddd','font-size':8,'text-anchor':'middle'}),
      {textContent:(d>=0?'+':'')+d.toFixed(2)}));
    // loss name label (rotated)
    const t=el('text',{x:x+bw*0.5,y:H-padB+14,fill:'#9aa7c7','font-size':8,
                        'text-anchor':'middle','transform':`rotate(-30,${x+bw*0.5},${H-padB+14})`});
    t.textContent=ln.replace('_tree','★').replace('_',' ');
    svg.appendChild(t);
  });

  // y-axis labels
  [minD, 0, maxD].forEach(v=>{
    svg.appendChild(Object.assign(el('text',{x:padL-2,y:sy(v)+3,fill:'#7a89b3','font-size':8,'text-anchor':'end'}),{textContent:v.toFixed(2)}));
  });
  svg.appendChild(Object.assign(el('text',{x:W/2,y:H-2,fill:'#6f7ea8','font-size':8,'text-anchor':'middle'}),{textContent:'loss function (★ = tree variant)'}));
  svg.appendChild(Object.assign(el('text',{x:padL,y:padT,fill:'#6f7ea8','font-size':8}),{textContent:'↑ Δ BE (mean across verifiers)'}));

  // verifier colour legend
  const legBox=$('cmp_delta_legend'); legBox.innerHTML='<span class="muted" style="font-size:10px">tick colour = verifier: </span>';
  verifiers.forEach(v=>{
    const sp=document.createElement('span');
    sp.innerHTML=`<i style="background:${VCOL[v]||'#888'}"></i>${v}`;
    legBox.appendChild(sp);
  });
}

/** "Load into debugger" row of buttons. */
function renderLoadRow(store){
  const row=$('cmp_load_row'); row.innerHTML='<span class="muted" style="margin-right:4px">Load into debugger:</span>';
  Object.keys(store).forEach(ln=>{
    if(store[ln].error){ row.innerHTML+=`<span class="muted" title="${store[ln].error}">${ln} ✗</span>`; return; }
    const btn=document.createElement('button');
    btn.className='ghost'; btn.style.fontSize='11px'; btn.style.padding='3px 8px';
    btn.textContent=ln;
    btn.onclick=()=>{ S=store[ln]; cur=S.n-1; vlegend(); render(); $('status').textContent=`Loaded: ${ln} (from Compare All)`; };
    row.appendChild(btn);
  });
}

// ── Compare-all flow ───────────────────────────────────────────────────────
let _cmpJob=null;

function startCompareAll(){
  $('trainall').disabled=true;
  $('cmppanel').style.display='block';
  $('cmpstatus').textContent='Training…';
  const p=new URLSearchParams({
    vocab:$('vocab').value, K:$('K').value, L:$('L').value,
    steps:$('steps').value, lr:$('lr').value,
    be_trials:'15', be_every:$('steps').value>80?'10':'5',
    lam:$('lam').value,
    teacher_peak:$('teacher_peak').value, student_scale:$('student_scale').value,
    teacher_temp:$('teacher_temp').value, student_temp:$('student_temp').value,
  });
  fetch('/api/train_all_start?'+p).then(r=>r.json()).then(j=>{
    _cmpJob=j.job;
    $('cmpstatus').textContent=`Job started — ${j.losses.length} losses × ${$('steps').value} steps`;
    pollCompare();
  });
}

function pollCompare(){
  if(!_cmpJob) return;
  fetch('/api/progress?job='+_cmpJob).then(r=>r.json()).then(j=>{
    if(j.error && j.done){ $('cmpstatus').textContent='Error: '+j.error; $('trainall').disabled=false; return; }
    const pct=j.total>0?Math.round(100*j.cur/j.total):0;
    $('cmpstatus').textContent=`${j.phase} (${pct}%)`;
    if(!j.done){ setTimeout(pollCompare, 1200); return; }
    // done — fetch compare data
    fetch('/api/compare').then(r=>r.json()).then(cmp=>{
      $('cmpstatus').textContent=`Done — ${Object.keys(cmp.store).length} losses trained`;
      compareBEChart('cmp_be_chart', cmp.store);
      compareDeltaChart('cmp_delta_chart', cmp.store);
      renderLoadRow(cmp.store);
      $('trainall').disabled=false;
    });
  });
}

document.getElementById('trainall').addEventListener('click', startCompareAll);

function renderBreakdown(bd){
  const box=$('bd'); box.innerHTML='';
  if(!bd){ return; }
  if(bd.k==='div'){
    let h='<table><tr><th class="pl">node</th><th>'+bd.blabel+'</th></tr>';
    bd.rows.forEach(r=>{ h+=`<tr><td class="pl">${r.l}</td><td>${r.v.toFixed(4)}</td></tr>`; });
    h+='</table>'; box.innerHTML=h; return;
  }
  let html='';
  bd.paths.forEach((rows,pi)=>{
    const hasH = rows.length && ('h' in rows[0]);
    html+=`<table><tr><th class="pl">path ${pi}</th><th>α</th><th>${bd.blabel.split(' ')[0]+' '+(bd.blabel.split(' ')[1]||'')}</th>`+(hasH?'<th>h</th>':'')+'</tr>';
    rows.forEach(r=>{ html+=`<tr><td class="pl">d${r.d} → ${r.t}</td><td>${r.a.toFixed(3)}</td><td>${(r.b??0).toFixed(3)}</td>`+(hasH?`<td>${(r.h??0).toFixed(3)}</td>`:'')+'</tr>'; });
    html+='</table>';
  });
  box.innerHTML=html || '<span class="muted">no breakdown</span>';
}

function vlegend(){
  const box=$('velegend'); box.innerHTML='';
  S.verifiers.forEach(n=>{
    const sp=document.createElement('span');
    sp.innerHTML=`<i style="background:${VCOL[n]||'#888'}"></i>${n}${n===S.matched?' (matched)':''}`;
    if(n===S.matched) sp.style.fontWeight='700';
    box.appendChild(sp);
  });
}

function render(){
  if(!S) return;
  const w=S.w[cur];
  S.nodes.forEach((n,i)=>{
    const c=$('n'+i); if(c) c.setAttribute('fill', color(w[i]));
    const t=$('nt'+i);
    if(t){ const path=(S._keys[i]||'').split(',').join('→');
      t.textContent = `path ${path}  ·  depth ${n.depth}  ·  w=${w[i].toFixed(3)} `+
        `(~${Math.round(w[i]*100)}% chance accepted to here)`; }
  });
  const sel=new Set(S.sel[cur]||[]);
  S.edges.forEach((e,i)=>{ const ln=$('e'+i); if(ln) ln.classList.toggle('sel', sel.has(e.u+'|'+e.v)); });
  S.nodes.forEach((n,i)=>{ const c=$('n'+i); if(!c) return;
     const onSel=[...sel].some(k=>k.endsWith('|'+S._keys[i])||k.startsWith(S._keys[i]+'|'));
     c.classList.toggle('sel', onSel); });
  $('m_loss').textContent = S.loss[cur].toFixed(4);
  $('m_ver').textContent = S.matched;
  $('m_be').textContent = (S.be[S.matched]||[])[cur]?.toFixed(3) ?? '—';
  $('info').textContent = S.info[cur];
  $('stepread').textContent = `step ${S.step[cur]} / ${S.step[S.n-1]}`;
  $('scrub').value = cur;
  lineChart('losschart', S.loss, cur, '#ff7b54', '↑ loss (lower = better)');
  multiLine('bechart', S.be, cur, S.matched);
  renderBreakdown(S.bd[cur]);
}

function goto(i){ cur=Math.max(0,Math.min(S.n-1,i)); render(); }
function step(d){ goto(cur+d); }
function play(){ if(!S) return; playing=!playing; $('play').textContent=playing?'❚❚ Pause':'► Play';
  if(playing) tick(); else clearTimeout(timer); }
function tick(){ if(!playing) return; if(cur>=S.n-1){ playing=false; $('play').textContent='► Play'; return; }
  step(1); timer=setTimeout(tick, 1260 - $('speed').value); }

function computeKeys(S){
  const keys=new Array(S.nodes.length).fill(null);
  const parentOf={}; S.edges.forEach(e=>{ parentOf[e.v]=e.u; });
  S.nodes.forEach((n,i)=>{ let chain=[n.token], v=i;
    while(parentOf[v]!==undefined){ v=parentOf[v]; chain.unshift(S.nodes[v].token); }
    keys[i]=chain.join(','); });
  return keys;
}

function setProg(cur,total,phase){
  const pct = total>0 ? Math.round(100*cur/total) : 0;
  $('progbar').style.width = pct+'%';
  $('progtext').textContent = `${phase} — step ${cur}/${total} (${pct}%)`;
}
function showProg(on){ $('progwrap').style.display = on?'block':'none'; }

function finishTrain(data){
  S=data; cur=0; S._keys=computeKeys(S);
  $('scrub').max=S.n-1; $('jump').max=S.step[S.n-1];
  // fill the "what you're looking at" panel with the concrete run values
  $('wa_K').textContent='K='+S.K; $('wa_L').textContent='L='+S.L;
  $('wa_V').textContent=S.vocab;
  $('wa_peak').textContent=S.teacher_peak; $('wa_scale').textContent=S.student_scale;
  $('wa_ttemp').textContent=S.teacher_temp??1.0; $('wa_stemp').textContent=S.student_temp??1.0;
  const root=(S.nodes.find(n=>n.depth===0)||S.nodes[0]);
  $('wa_pending').textContent = root? root.token : '·';
  buildTree(); vlegend(); render();
  groupedBars('flatchart', S.flat_before, S.flat_after);
  const tb=S.tree_before, ta=S.tree_after;
  const modeTag = S.is_flat ? '(flat baseline)' : S.is_variant ? `(variant λ=${S.lam})` : '(tree)';
  $('status').textContent = `${S.loss_name} ${modeTag} • ${S.n} steps • `+
    `matched verifier=${S.matched} • BE ${tb[S.matched].toFixed(2)}→${ta[S.matched].toFixed(2)}`;
}

async function train(){
  $('err').textContent=''; $('status').textContent='training in background…';
  $('train').disabled=true; showProg(true); setProg(0,+$('steps').value,'starting');
  const beEvery=Math.max(1, Math.round(+$('steps').value/16));  // ~16 BE evaluations per run
  const q=new URLSearchParams({loss:$('loss').value,vocab:$('vocab').value,
                               K:$('K').value,L:$('L').value,
                               steps:$('steps').value,lr:$('lr').value,
                               lam:USES_LAM.has($('loss').value)?$('lam').value:'0',
                               teacher_peak:$('teacher_peak').value,
                               teacher_temp:$('teacher_temp').value,
                               student_scale:$('student_scale').value,
                               student_temp:$('student_temp').value,
                               be_every:beEvery,
                               khisti:$('khisti').checked?'1':'0'}).toString();
  let job;
  try{
    const r=await (await fetch('/api/train_start?'+q)).json();
    job=r.job;
  }catch(e){ $('err').textContent=String(e); $('train').disabled=false; showProg(false); return; }

  async function poll(){
    let p;
    try{ p=await (await fetch('/api/progress?job='+job)).json(); }
    catch(e){ $('err').textContent=String(e); $('train').disabled=false; showProg(false); return; }
    setProg(p.cur,p.total,p.phase||'');
    if(!p.done){ setTimeout(poll, 350); return; }
    if(p.error){ $('err').textContent=p.error; $('status').textContent=''; $('train').disabled=false; showProg(false); return; }
    // fetch the full result once
    try{
      const res=await (await fetch('/api/result?job='+job)).json();
      if(!res.ok){ $('err').textContent=res.error+'\n'+(res.trace||''); }
      else { finishTrain(res.data); }
    }catch(e){ $('err').textContent=String(e); }
    $('train').disabled=false; showProg(false);
  }
  poll();
}

$('train').onclick=train;
$('first').onclick=()=>goto(0); $('last').onclick=()=>goto(S.n-1);
$('back').onclick=()=>step(-1); $('fwd').onclick=()=>step(1);
$('play').onclick=play;
$('go').onclick=()=>{ if(!S) return; const t=+$('jump').value;
  let best=0,bd=1e9; S.step.forEach((s,i)=>{ if(Math.abs(s-t)<bd){bd=Math.abs(s-t);best=i;} }); goto(best); };
$('scrub').oninput=e=>goto(+e.target.value);
clampToVocab();   // enforce K/L <= V-1 on first load
loadLosses();
</script>
</body></html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5050)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    print(f"\nTree-loss step debugger -> http://{args.host}:{args.port}/\n")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
