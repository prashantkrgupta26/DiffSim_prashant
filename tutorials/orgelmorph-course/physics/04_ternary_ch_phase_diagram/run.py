"""OrgElMorph course - Physics P4 driver (the file the student runs).

    python run.py                 # ternary quench, 64x64
    python run.py --level 6

Marches a ternary spinodal quench and prints how the composition cloud
spreads across the Gibbs triangle (from a tight blob to the two
coexisting phases) while the mean composition stays conserved.  Compare
with EXPECTED.md; render the phase-diagram figures with gen_figures.py.
"""
import argparse

import numpy as np

from ternary import build_mesh_dm, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--t_end", type=float, default=0.6)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dm, mesh, cons = build_mesh_dm(args.level, device=args.device)
    r = run(dm, mesh, cons, t_end=args.t_end, device=args.device)

    print(f"\n=== Ternary Cahn--Hilliard (chi={r['chi']}, "
          f"{r['side']}x{r['side']}, quench to t={args.t_end}) ===")
    print(f"  initial blend (phi1, phi2, phis) = "
          f"({r['phi0'][0]:.2f}, {r['phi0'][1]:.2f}, "
          f"{1 - sum(r['phi0']):.2f})")
    print(f"  composition spread (std phi1): {r['spread0']:.4f} "
          f"(start) -> {r['spreadf']:.4f} (end)")
    print(f"  coexisting phases (phi1, phi2):")
    print(f"      phase A = ({r['phaseA'][0]:.3f}, {r['phaseA'][1]:.3f})"
          f"      phase B = ({r['phaseB'][0]:.3f}, {r['phaseB'][1]:.3f})")
    print(f"  mean drift |dphi1| = {r['mass1_drift']:.2e}, "
          f"|dphi2| = {r['mass2_drift']:.2e}  (conserved)")
    print("\nThe cloud opens along a tie-line while the mean is pinned "
          "(conservation).  Compare with EXPECTED.md.")


if __name__ == "__main__":
    main()
