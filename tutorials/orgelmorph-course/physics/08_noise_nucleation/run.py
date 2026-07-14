"""OrgElMorph course - Physics P8 driver (the file the student runs).

    python run.py                 # noise-amplitude sweep -> nucleation
    python run.py --level 6

Runs an undercooled melt (psi starts at 0, NO seed) at several FDT noise
amplitudes and prints, for each, the final crystalline fraction, the
number of grains nucleated, and the induction time.  Zero noise stays
amorphous; larger noise nucleates sooner and more.  Compare with
EXPECTED.md; figures via gen_figures.py."""
import argparse

import numpy as np

from nucleation import build_mesh_dm, sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dm, mesh, cons = build_mesh_dm(args.level, device=args.device)
    s = sweep(dm, mesh, cons, device=args.device)

    print("\n=== FDT noise and nucleation (undercooled melt, no seed) ===")
    print(f"  {'noise_psi':>10}{'X_end':>9}{'grains':>8}"
          f"{'induction t':>13}{'psi_max':>10}")
    for r in s["runs"]:
        ind = "-" if np.isnan(r["induction"]) else f"{r['induction']:.3f}"
        print(f"  {r['noise_psi']:>10.3f}{r['X_end']:>9.3f}"
              f"{r['n_grains']:>8d}{ind:>13}{r['psi_max']:>10.3f}")
    print("\nZero noise stays amorphous (no nucleation); increasing the "
          "FDT amplitude nucleates crystal sooner and in more grains -- "
          "the noise IS the physical seed.  Compare with EXPECTED.md.")


if __name__ == "__main__":
    main()
