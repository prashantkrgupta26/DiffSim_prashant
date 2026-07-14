"""OrgElMorph course - Differentiable D4 driver (the file the student runs).

    python run.py                 # dJ/dT_n time series of the quench schedule

Computes the gradient of a crystallisation objective with respect to the
temperature schedule T(t) - a whole TIME SERIES dJ/dT_n from one adjoint
reverse sweep - and SELF-CHECKS it step by step against central finite
differences.  Compare with EXPECTED.md."""
import argparse

from schedule import schedule_gradient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    r = schedule_gradient(level=args.level, device=args.device)
    print(f"\n=== D4: process gradient of the quench schedule T(t) "
          f"({r['side']}x{r['side']} nodes, {r['NS']} steps) ===")
    print(f"  objective J (final phi,psi vs target) = {r['J']:.6e}")
    print(f"  crystallinity <psi>: "
          + " -> ".join(f"{v:.4f}" for v in r["cryst"]))
    print(f"\n  step   T_n      dJ/dT_n (adjoint)   dJ/dT_n (finite diff)"
          f"     adj/fd")
    for n in range(r["NS"]):
        print(f"   {n:2d}   {r['Tsched'][n]:5.3f}    "
              f"{r['gT_adj'][n]:+.6e}       {r['gT_fd'][n]:+.6e}   "
              f"{r['rel'][n]:.2e}")

    worst = r["rel"].max()
    print(f"\n  worst adjoint-vs-FD agreement over the schedule: {worst:.2e}")
    # THE GATE: the whole time-series gradient matches finite differences.
    assert worst < 1e-6, ("schedule gradient disagrees with FD", worst)
    print("  SELF-CHECK PASSED: the whole dJ/dT_n time series matches FD "
          "(one sweep, N gradients).")
    print("\nCompare with EXPECTED.md; render figures with gen_figures.py.")


if __name__ == "__main__":
    main()
