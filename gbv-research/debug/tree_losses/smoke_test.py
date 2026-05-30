"""
smoke_test.py — fast sanity check for the sandbox.

Verifies, on one toy tree, that:
  * every one of the 12 tree losses runs, and
  * the transparent tracer matches the REAL compute_tree_loss.

Also runs each verifier once to confirm the tree wiring is valid.

    python smoke_test.py
"""

import torch

from tiny_models import TinyMarkovModel, make_dataset, sample_tree
from tree_harness import TREE_LOSS_NAMES, ALL_VERIFIERS, block_length_once
import loss_tracer

V, K, L, T = 6, 3, 4, 1.0


def main():
    teacher = TinyMarkovModel.teacher_counting(V)
    student = TinyMarkovModel.student_random(V)
    _, test = make_dataset(V)
    g = torch.Generator().manual_seed(99)

    # WITH grad — for losses
    qp, qpf, qd, pd = sample_tree(student, teacher, int(test[0]), K, L, T,
                                  with_grad=True, generator=g)

    print("== loss tracer vs real compute_tree_loss ==")
    ok = True
    for name in sorted(TREE_LOSS_NAMES):
        try:
            tr = loss_tracer.trace(name, qd, pd, qp, L, K)
            d = abs(tr["real_value"] - tr["traced_value"])
            flag = "OK" if d < 5e-2 else "MISMATCH"
            if d >= 5e-2:
                ok = False
            print(f"  {name:16s} real={tr['real_value']:+.5f} "
                  f"traced={tr['traced_value']:+.5f} |d|={d:.1e} {flag}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  {name:16s} ERROR {type(e).__name__}: {e}")

    # backprop reaches the student parameter?
    loss = loss_tracer.compute_tree_loss("kl_tree", qd, pd, qp, L=L, K=K)
    loss.backward()
    grad_ok = student.W.grad is not None and torch.isfinite(student.W.grad).all()
    print(f"\n== backprop to student.W: {'OK' if grad_ok else 'FAILED'} ==")

    print("\n== each verifier runs on the tree ==")
    qp2, qpf2, qd2, pd2 = sample_tree(student, teacher, int(test[0]), K, L, T,
                                      with_grad=False,
                                      generator=torch.Generator().manual_seed(1))
    ver_ok = True
    for v in ALL_VERIFIERS:
        try:
            # fresh detached dicts per verifier (traversal mutates in place)
            _, _, qd_v, pd_v = sample_tree(student, teacher, int(test[0]), K, L, T,
                                           with_grad=False,
                                           generator=torch.Generator().manual_seed(1))
            blen = block_length_once(qp2, qpf2, qd_v, pd_v, v)
            print(f"  {v:10s} block_len={blen}")
        except Exception as e:  # noqa: BLE001
            ver_ok = False
            print(f"  {v:10s} ERROR {type(e).__name__}: {e}")

    print("\nRESULT:", "ALL OK" if (ok and grad_ok and ver_ok) else "SOME FAILED")
    return 0 if (ok and grad_ok and ver_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
