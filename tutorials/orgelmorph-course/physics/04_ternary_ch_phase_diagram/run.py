"""OrgElMorph course - Physics P4 driver (the file the student runs).

    python run.py                 # ternary quench, 64x64
    python run.py --level 6

Marches a ternary spinodal quench and prints how the composition cloud
spreads across the Gibbs triangle (from a tight blob to the two
coexisting phases) while the mean composition stays conserved.  The two
coexisting phases are read off by CLUSTERING the local-composition cloud
(a 2-component Gaussian mixture), not a crude median split.  Compare with
EXPECTED.md; render the phase-diagram figures with gen_figures.py, or run
the full workflow (admissibility, quadrature conservation, lever rule,
spinodal, N-shift) with run_harness.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                os.pardir, os.pardir,
                                                "materials")))
from loader import load_blend                               # noqa: E402
from ternary import (build_mesh_dm, run, cluster_phases,    # noqa: E402
                     spinodal_hessian)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--t_end", type=float, default=0.6)
    ap.add_argument("--blend", default="synthetic_demix")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    b = load_blend(args.blend)
    phi0 = (0.35, 0.35)
    _, det, ev = spinodal_hessian(phi0[0], phi0[1], b.chi, b.N)

    dm, mesh, cons = build_mesh_dm(args.level, device=args.device)
    r = run(dm, mesh, cons, chi=b.chi, N=b.N, phi0=phi0, t_end=args.t_end,
            device=args.device)
    clus = cluster_phases(r["p1f"], r["p2f"], r["side"])
    A, B = clus["phaseA_mean"], clus["phaseB_mean"]

    print(f"\n=== Ternary Cahn--Hilliard ({b.name}: chi={b.chi}, N={b.N}, "
          f"{r['side']}x{r['side']}, quench to t={args.t_end}) ===")
    print(f"  initial blend (phi1, phi2, phis) = "
          f"({phi0[0]:.2f}, {phi0[1]:.2f}, {1 - sum(phi0):.2f})")
    print(f"  spinodal @IC: det H = {det:.3g}  "
          f"({'UNSTABLE (demixes)' if det < 0 else 'stable'})")
    print(f"  composition spread (std phi1): {r['spread0']:.4f} "
          f"(start) -> {r['spreadf']:.4f} (end)")
    print(f"  coexisting phases (GMM cluster means, phi1, phi2):")
    print(f"      phase A = ({A[0]:.3f}, {A[1]:.3f}), pop "
          f"{clus['population'][0]:.2f}")
    print(f"      phase B = ({B[0]:.3f}, {B[1]:.3f}), pop "
          f"{clus['population'][1]:.2f}")
    print(f"      method sensitivity (GMM vs interface-excluded): "
          f"{clus['endpoint_sensitivity']:.3g}")
    print(f"  mean drift |dphi1| = {r['mass1_drift']:.2e}, "
          f"|dphi2| = {r['mass2_drift']:.2e}  (conserved)")
    print("\nThe cloud opens along a tie-line while the mean is pinned "
          "(conservation).  Compare with EXPECTED.md; run run_harness.py "
          "for the full workflow.")


if __name__ == "__main__":
    main()
