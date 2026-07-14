"""OrgElMorph course - Differentiable D1 driver (the file the student runs).

    python run.py                 # three-way gradient check, 8x8, 3 steps
    python run.py --steps 4 --order 2   # BDF2 (variable-coefficient) march

Computes dJ/d{M, kappa, chi, A} through a short Cahn-Hilliard march three
ways - hand adjoint, autograd twin, finite difference - prints the table,
and SELF-CHECKS that they agree (this agreement is the gate: if it fails,
the gradient is not to be trusted).  Compare with EXPECTED.md.  Figures
for the document come from gen_figures.py using this same core."""
import argparse

import numpy as np

from three_way import three_way


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    r = three_way(level=args.level, n_steps=args.steps, order=args.order,
                  device=args.device)
    print(f"\n=== Three-way gradient check "
          f"({r['side']}x{r['side']}, {r['n_steps']} steps, "
          f"BDF{r['order']}) ===")
    print(f"  objective J = 1/2 |c_N - 1/2|^2 = {r['J']:.6e}\n")
    print(f"  {'param':6s} {'adjoint':>14s} {'autograd twin':>15s} "
          f"{'finite diff':>14s} {'adj/twin':>10s} {'adj/fd':>10s}")
    for p in r["names"]:
        a, t, f = r["grads"][p]
        print(f"  {p:6s} {a:+14.6e} {t:+15.6e} {f:+14.6e} "
              f"{r['rel_tw'][p]:10.2e} {r['rel_fd'][p]:10.2e}")

    worst_tw = max(r["rel_tw"].values())
    worst_fd = max(r["rel_fd"].values())
    print(f"\n  worst adj-vs-twin agreement: {worst_tw:.2e}  "
          f"(same math, two engines -> machine precision)")
    print(f"  worst adj-vs-FD   agreement: {worst_fd:.2e}  "
          f"(FD truncation/roundoff floor)")

    # THE GATE - the FD-vs-adjoint agreement is the FD-verified self-check.
    # Locked with headroom (measured adj/twin ~1e-16, adj/fd ~1e-9).
    assert worst_tw < 1e-10, ("adjoint disagrees with autograd twin", worst_tw)
    assert worst_fd < 1e-6, ("adjoint disagrees with finite diff", worst_fd)
    print("\n  SELF-CHECK PASSED: all three methods agree. The gradient is "
          "trustworthy.")
    print("\nCompare the printed table with EXPECTED.md; render figures with "
          "gen_figures.py.")


if __name__ == "__main__":
    main()
