"""
variant_test.py — sanity + efficacy check for loss_variants.

For each variant (and its production baseline) it:
  * confirms the loss runs and backprops a FINITE gradient to student.W,
  * trains the toy student a few steps,
  * reports matched-verifier block efficiency before → after.

A variant "mathematically sticks" if BE-after > BE-before (and grad is finite).

    python variant_test.py
"""

import torch

from tiny_models import TinyMarkovModel, make_dataset, sample_tree
from tree_harness import estimate_block_efficiency
import loss_variants as LV

V, K, L, T = 6, 3, 4, 1.0
STEPS, BATCH, LR, TRIALS, SEED = 40, 8, 0.1, 120, 0


def run(name, fn, matched):
    torch.manual_seed(SEED)
    teacher = TinyMarkovModel.teacher_counting(V, seed=SEED)
    student = TinyMarkovModel.student_random(V, seed=SEED + 1)
    train, test = make_dataset(V, seed=SEED + 2)
    opt = torch.optim.Adam([student.W], lr=LR)
    gen = torch.Generator().manual_seed(SEED + 1000)

    be0 = estimate_block_efficiency(student, teacher, matched, test, K, L, T,
                                    n_trials=TRIALS, seed=SEED)
    grad_ok = True
    for step in range(STEPS):
        opt.zero_grad()
        acc = torch.zeros(1)
        nb = 0
        for _ in range(BATCH):
            pend = int(train[torch.randint(len(train), (1,), generator=gen).item()])
            _, _, qd, pd = sample_tree(student, teacher, pend, K, L, T,
                                       with_grad=True, generator=gen)
            paths, _, _, _ = (None, None, None, None)
            qp, _, qdg, pdg = sample_tree(student, teacher, pend, K, L, T,
                                          with_grad=True, generator=gen)
            l = fn(qdg, pdg, qp, L, K)
            if l.requires_grad:
                acc = acc + l
                nb += 1
        if nb:
            (acc / nb).backward()
            if student.W.grad is None or not torch.isfinite(student.W.grad).all():
                grad_ok = False
            opt.step()
    be1 = estimate_block_efficiency(student, teacher, matched, test, K, L, T,
                                    n_trials=TRIALS, seed=SEED)
    return be0, be1, grad_ok


def main():
    rows = []
    # production baselines for comparison
    base = {"bv_tree": "bv", "gbv_tree": "gbv", "spectr_tree": "spectr",
            "khisti_tree": "khisti", "traversal_tree": "traversal", "kl_tree": "gbv"}
    print(f"{'loss':20s} {'matched':9s} {'BE before':>9s} {'BE after':>9s} {'Δ':>7s}  grad")
    print("-" * 64)
    for name, ver in base.items():
        be0, be1, ok = run(name, lambda q, p, pa, L, K, n=name: LV.compute_any(n, q, p, pa, L, K), ver)
        rows.append((name, ver, be0, be1, ok))
    print("-- variants --")
    for name, (fn, b, ver, desc) in LV.VARIANTS.items():
        be0, be1, ok = run(name, fn, ver)
        rows.append((name, ver, be0, be1, ok))

    print(f"{'loss':20s} {'matched':9s} {'BE before':>9s} {'BE after':>9s} {'Δ':>7s}  grad")
    print("-" * 64)
    for name, ver, be0, be1, ok in rows:
        print(f"{name:20s} {ver:9s} {be0:9.3f} {be1:9.3f} {be1-be0:+7.3f}  {'ok' if ok else 'BAD'}")


if __name__ == "__main__":
    main()
