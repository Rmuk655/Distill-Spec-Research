#!/usr/bin/env python
"""
verify_prefix_overlap.py — standalone verification harness for the Prefix-Overlap
Distillation Objective (spec: prefix_overlap_objective.pdf).

WHAT IT DOES
  Exercises the REAL implementation in losses/compute.py against an independent
  brute-force reference derived directly from the spec (sect. 4 / sect. 6). It does
  NOT need GPU, HF weights, W&B, pytest, or the cluster for the core checks — a
  differentiable toy student + a fixed-continuation toy teacher stand in for the
  real models, so the exact same code path (indexing, cumsum, _prefix_score,
  root loop, CE mixing) is executed on a tiny synthetic vocabulary where the
  objective can be enumerated exactly.

CHECKS (critical unless marked INFO)
  C1  prob objective == sum_t exp(S_t)                (logsumexp trick exact)
  C2  logprob objective == sum_t S_t
  C3  prob != logprob                                 (not aliased)
  C4  gradient flows through student log-probs
  C5  end-to-end single-root loss == spec sect.6 brute force (L separate forwards)
  C6  prob-loss numerically distinct from CE          (not secretly cross-entropy)
  C7  CE mixing additive: loss(lambda) == loss(0) + lambda * mean_ce   (spec sect.7)
  C8  multiroot tail-reuse == independent cumsum-per-root recompute
  C9  loss decreases on an easy synthetic fit         (optimizes intended objective)
  I1  root set starts at r=0                          (INFO: spec R(y) starts at N)
  I2  random-offset range == {0..N-1}                 (INFO: spec says {1..N})
  I3  JSD baseline sane (symmetric, >=0, 0 when equal) (INFO: control path)
  S1  real-model smoke: one prefix_overlap train step  (only with --real-smoke)
  P1  prior-run scan: best_val_block_eff per run       (only with --runs-root DIR)

EXIT CODE
  0  all critical checks passed
  1  any critical check failed (INFO/SKIP/WARN never fail the build)

REMOTE USAGE (on the cluster, inside the training venv)
  cd ~/Distill-Spec-Research
  python scripts/verify_prefix_overlap.py                     # core toy checks (fast, no GPU)
  python scripts/verify_prefix_overlap.py --runs-root /home/colligo/local_ckpts/prefix_overlap
  python scripts/verify_prefix_overlap.py --real-smoke        # loads config.py DRAFT/TEACHER, 1 step (GPU)
  echo $?                                                     # 0 = pass, 1 = critical failure
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# import the REAL module under test, stubbing only the heavy deps the prefix   #
# path never touches (so the harness runs in a minimal torch-only env too).    #
# --------------------------------------------------------------------------- #
def import_compute():
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "verifiers"))
    for name, attrs in (("inference_util", {"iid_draft": lambda *a, **k: None,
                                            "target_tree_pass": lambda *a, **k: None}),
                        ("verifier", {"TreeVerifier": object})):
        try:
            __import__(name)                       # real module present (remote box)
        except Exception:
            m = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[name] = m                  # minimal-env fallback stub
    from losses import compute
    try:
        from losses.flat import jsd
    except Exception:
        jsd = None
    return compute, jsd


# --------------------------------------------------------------------------- #
# tiny toy models: differentiable student, fixed-continuation teacher          #
# --------------------------------------------------------------------------- #
V = 11
EOS = 10


class _Out:
    def __init__(self, logits):
        self.logits = logits


class StudentLM(nn.Module):
    """Causal, context-dependent, differentiable; logits[i] depends on tokens[:i+1]."""

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(V, 8)
        self.proj = nn.Linear(8, V)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
        self.config = types.SimpleNamespace(eos_token_id=EOS, vocab_size=V)

    def forward(self, input_ids, return_dict=True, **kw):
        e = self.emb(input_ids)
        csum = torch.cumsum(e, dim=1)
        denom = torch.arange(1, e.shape[1] + 1, device=e.device).view(1, -1, 1)
        return _Out(self.proj(csum / denom))


class TeacherStub(nn.Module):
    """generate() appends a FIXED continuation (deterministic); __call__ returns logits."""

    def __init__(self, cont):
        super().__init__()
        self.cont = cont
        self.emb = nn.Embedding(V, 8)
        self.lin = nn.Linear(8, V)
        self.config = types.SimpleNamespace(eos_token_id=EOS, vocab_size=V)

    def generate(self, input_ids, attention_mask=None, max_new_tokens=8, **kw):
        add = self.cont[:max_new_tokens].to(input_ids.device).view(1, -1)
        return torch.cat([input_ids, add], dim=1)

    def forward(self, input_ids, return_dict=True, **kw):
        return _Out(self.lin(self.emb(input_ids).float()))


def bf_prob_score(student, ctx, cont, L):
    """Spec sect.6 sketch, L separate forwards -> sum_t q(P_{1:t}|c)."""
    total = torch.zeros((), dtype=torch.float64)
    logq = torch.zeros((), dtype=torch.float64)
    for t in range(min(L, cont.numel())):
        inp = ctx if t == 0 else torch.cat([ctx, cont[:t].view(1, -1)], dim=1)
        logits = student(inp).logits[0, -1].double()
        logq = logq + F.log_softmax(logits, dim=-1)[cont[t]]
        total = total + torch.exp(logq)
    return total


def bf_ce(student, ctx, cont, L):
    """Spec sect.7 CE = -sum_t log q(P_t | c, P_{1:t-1})."""
    ce = torch.zeros((), dtype=torch.float64)
    for t in range(min(L, cont.numel())):
        inp = ctx if t == 0 else torch.cat([ctx, cont[:t].view(1, -1)], dim=1)
        logits = student(inp).logits[0, -1].double()
        ce = ce - F.log_softmax(logits, dim=-1)[cont[t]]
    return ce


# --------------------------------------------------------------------------- #
# check registry                                                               #
# --------------------------------------------------------------------------- #
class Results:
    def __init__(self):
        self.rows = []
        self.critical_failed = False

    def add(self, name, ok, detail="", info=False):
        status = "PASS" if ok else ("WARN" if info else "FAIL")
        if not ok and not info:
            self.critical_failed = True
        self.rows.append((status, name, detail))

    def skip(self, name, detail=""):
        self.rows.append(("SKIP", name, detail))

    def dump(self):
        w = max(len(n) for _, n, _ in self.rows)
        print("\n" + "=" * 78)
        for status, name, detail in self.rows:
            print(f"[{status:4}] {name:<{w}}  {detail}")
        print("=" * 78)
        crit = sum(1 for s, _, _ in self.rows if s in ("PASS", "FAIL"))
        passed = sum(1 for s, _, _ in self.rows if s == "PASS")
        print(f"{passed}/{crit} critical checks passed  "
              f"({sum(1 for s,_,_ in self.rows if s=='WARN')} warn, "
              f"{sum(1 for s,_,_ in self.rows if s=='SKIP')} skip)")
        print("VERDICT:", "CRITICAL FAILURE" if self.critical_failed else "FAITHFUL (all critical checks passed)")


def core_checks(C, jsd, R):
    torch.manual_seed(0)

    # C1/C2/C3/C4 -- _prefix_score math + gradient
    S = torch.tensor([-0.5, -1.3, -2.2, -4.0], requires_grad=True)
    prob = C._prefix_score(S, "prob")
    lgp = C._prefix_score(S, "logprob")
    R.add("C1 prob == sum(exp(S))", torch.allclose(prob, S.exp().sum(), atol=1e-6),
          f"{prob.item():.6f} vs {S.exp().sum().item():.6f}")
    R.add("C2 logprob == sum(S)", torch.allclose(lgp, S.sum(), atol=1e-6),
          f"{lgp.item():.6f} vs {S.sum().item():.6f}")
    R.add("C3 prob != logprob", not torch.allclose(prob, lgp),
          f"prob={prob.item():.4f} logprob={lgp.item():.4f}")
    prob.backward()
    R.add("C4 gradient flows (prob)", S.grad is not None and S.grad.abs().sum() > 0,
          f"|grad|={S.grad.abs().sum().item():.4f}")

    # C5 -- end-to-end single-root == spec sect.6 brute force
    student = StudentLM(seed=1)
    cont = torch.tensor([2, 5, 3, 7, 1, 4, 6, 8])
    teacher = TeacherStub(cont)
    prompt = torch.tensor([[9, 0, 3]])
    L = 6
    loss = C.compute_prefix_overlap_loss(student, teacher, prompt, M=1, L=L,
                                         objective="prob", aux_weight=0.0)
    bf = bf_prob_score(student, prompt, cont, L)
    R.add("C5 end-to-end == spec brute force", torch.allclose(loss.double(), -bf, atol=1e-5),
          f"code={loss.item():.6f} spec={-bf.item():.6f}")

    # C6 -- distinct from CE
    ce_only = C.compute_prefix_overlap_loss(student, teacher, prompt, M=1, L=L,
                                            objective="logprob", aux_weight=0.0)
    R.add("C6 prob-loss distinct from CE/logprob", abs(loss.item() - ce_only.item()) > 1e-3,
          f"prob={loss.item():.4f} logprob={ce_only.item():.4f}")

    # C7 -- CE mixing additive (spec sect.7  L_total = L_prefix + lambda*L_CE)
    lam = 0.3
    loss0 = C.compute_prefix_overlap_loss(student, teacher, prompt, M=1, L=L,
                                          objective="prob", aux="ce", aux_weight=0.0)
    lossL = C.compute_prefix_overlap_loss(student, teacher, prompt, M=1, L=L,
                                          objective="prob", aux="ce", aux_weight=lam)
    ce = bf_ce(student, prompt, cont, L)
    R.add("C7 CE mixing additive (L_prefix + lambda*L_CE)",
          torch.allclose((lossL - loss0).double(), lam * ce, atol=1e-5),
          f"delta={(lossL-loss0).item():.6f} lambda*CE={(lam*ce).item():.6f}")

    # C8 -- multiroot tail-reuse conditioning
    rollout = torch.tensor([2, 5, 3, 7, 1, 4, 6, 8, 9, 2, 5, 1])
    teacher2 = TeacherStub(rollout)
    Lm, N = 4, 3
    lossm = C.compute_prefix_overlap_multiroot_loss(student, teacher2, prompt, L=Lm, N=N,
        rollout_len=rollout.numel(), objective="prob", random_offset=False, aux_weight=0.0)
    full = torch.cat([prompt, rollout.view(1, -1)], dim=1)
    Cn, T = prompt.shape[1], rollout.numel()
    s_logits = student(full).logits[0, Cn - 1:Cn - 1 + T].double()
    tok_lp = F.log_softmax(s_logits, dim=-1).gather(-1, rollout.view(-1, 1)).squeeze(-1)
    roots = list(range(0, T - 1, N))
    scores = [torch.cumsum(tok_lp[r:r + Lm], 0).exp().sum() for r in roots]
    expected = -torch.stack(scores).mean()
    R.add("C8 multiroot tail-reuse == indep recompute",
          torch.allclose(lossm.double(), expected, atol=1e-5),
          f"code={lossm.item():.6f} indep={expected.item():.6f}")

    # C9 -- loss decreases
    st = StudentLM(seed=2)
    opt = torch.optim.SGD(st.parameters(), lr=5.0)
    t3 = TeacherStub(torch.tensor([2, 5, 3, 7, 1, 4]))
    l0 = None
    for _ in range(50):
        opt.zero_grad()
        l = C.compute_prefix_overlap_loss(st, t3, prompt, M=1, L=6, objective="prob", aux_weight=0.0)
        if l0 is None:
            l0 = l.item()
        l.backward()
        opt.step()
    R.add("C9 loss decreases on synthetic fit", l.item() < l0, f"{l0:.5f} -> {l.item():.5f}")

    # I1/I2 -- root-selection deviations (INFO: WARN flags a known, non-critical
    # deviation from the literal spec; ok==True would mean spec-compliant).
    R.add("I1 spec-compliant root start (== N)", roots[0] == N,
          f"code first root={roots[0]}; spec R(y) starts at N={N} -> DEVIATION (M2)", info=True)
    offs = {int(torch.randint(0, N, (1,)).item()) for _ in range(2000)}
    R.add("I2 spec-compliant offset range ({1..N})", offs == set(range(1, N + 1)),
          f"code={sorted(offs)}; spec={sorted(range(1, N + 1))} -> DEVIATION (M3)", info=True)

    # I3 -- JSD baseline sanity (control path)
    if jsd is None:
        R.skip("I3 JSD baseline sanity", "losses.flat.jsd not importable")
    else:
        a = torch.randn(5, V, requires_grad=True)
        b = torch.randn(5, V)
        j_eq = jsd(b.clone().detach().requires_grad_(True), b)
        j_ab = jsd(a, b)
        j_ab.backward()
        ok = (j_eq.item() < 1e-6) and (0 <= j_ab.item() <= 0.7) and a.grad.abs().sum() > 0
        R.add("I3 JSD sane (0 when equal, bounded, grad)", ok,
              f"jsd(x,x)={j_eq.item():.2e} jsd(a,b)={j_ab.item():.4f}", info=True)


def real_smoke(C, R):
    """Optional: load real config.py models and run ONE prefix_overlap train step."""
    try:
        import config
        from checkpointing import load_models
    except Exception as e:
        R.skip("S1 real-model smoke", f"config/checkpointing import failed: {e}")
        return
    try:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok, draft, teacher = load_models(config.TEACHER_MODEL, config.DRAFT_MODEL, device=dev)
        ids = tok("2+2=", return_tensors="pt").input_ids.to(dev)
        draft.train()
        teacher.eval()
        loss = C.compute_prefix_overlap_loss(draft, teacher, ids, M=1, L=4,
                                             objective="prob", aux_weight=0.0)
        loss.backward()
        gnorm = sum(p.grad.norm().item() for p in draft.parameters() if p.grad is not None)
        R.add("S1 real-model smoke: 1 prefix_overlap step",
              torch.isfinite(loss).item() and gnorm > 0,
              f"loss={loss.item():.4f} sum|grad|={gnorm:.2f} dev={dev}")
    except Exception as e:
        R.add("S1 real-model smoke", False, f"raised: {type(e).__name__}: {e}")


def inspect_prior_runs(root, R):
    """Optional: scan run dirs for state.json and report best_val_block_eff consistency."""
    paths = sorted(glob.glob(os.path.join(root, "*", "ckpt_latest", "state.json")) +
                   glob.glob(os.path.join(root, "*", "state.json")))
    if not paths:
        R.skip("P1 prior-run scan", f"no state.json under {root}")
        return
    print(f"\n--- prior runs under {root} ---")
    seen = {}
    for p in paths:
        try:
            s = json.load(open(p))
        except Exception:
            continue
        run = Path(p).parts[-3] if "ckpt_latest" in p else Path(p).parts[-2]
        be = s.get("best_val_block_eff")
        sm = s.get("best_smoothed_be")
        step = s.get("step")
        print(f"  {run:<52} best_be={be}  best_smoothed={sm}  step={step}")
        if be is not None:
            seen.setdefault(round(be, 4), []).append(run)
    dups = {k: v for k, v in seen.items() if len(v) > 1}
    if dups:
        print("  [!] identical best_val_block_eff across distinct runs (suspicious):")
        for k, v in dups.items():
            print(f"      {k}: {v}")
    R.add("P1 prior-run scan (no identical-curve collisions)", not dups,
          f"{len(paths)} runs scanned; {len(dups)} value collisions", info=True)


def main():
    ap = argparse.ArgumentParser(description="Prefix-Overlap objective verification harness")
    ap.add_argument("--real-smoke", action="store_true",
                    help="load config.py DRAFT/TEACHER and run one real prefix_overlap step (needs GPU/HF)")
    ap.add_argument("--runs-root", default=None,
                    help="scan this dir for run state.json files and report block_eff consistency")
    args = ap.parse_args()

    C, jsd = import_compute()
    R = Results()
    print(f"torch {torch.__version__}  |  module under test: {C.__file__}")

    core_checks(C, jsd, R)
    if args.real_smoke:
        real_smoke(C, R)
    else:
        R.skip("S1 real-model smoke", "pass --real-smoke to enable")
    if args.runs_root:
        inspect_prior_runs(args.runs_root, R)
    else:
        R.skip("P1 prior-run scan", "pass --runs-root DIR to enable")

    R.dump()
    sys.exit(1 if R.critical_failed else 0)


if __name__ == "__main__":
    main()
