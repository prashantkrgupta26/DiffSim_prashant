"""OrgElMorph course - Physics P6 driver (the file the student runs).

    python run.py                 # grow/melt + Avrami kinetics
    python run.py --level 6

Runs a seeded crystal below and above the melting point (growth vs
melting), then a multi-nucleus run whose crystalline fraction is fit to
the Avrami/JMAK law.  Compare with EXPECTED.md; render the figures with
gen_figures.py."""
import argparse

import numpy as np

from crystallization import (build_mesh_dm, run_grow_melt, run_avrami, TM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dm, mesh, cons = build_mesh_dm(args.level, device=args.device)

    print(f"\n=== Allen--Cahn crystallization (PCBM-class, Tm={TM:g} K) ===")
    grow = run_grow_melt(dm, mesh, cons, 333.0, device=args.device)
    melt = run_grow_melt(dm, mesh, cons, 700.0, device=args.device)
    print(f"  seeded disc, T = 333 K (< Tm): area "
          f"{grow['a0']:.4f} -> {grow['a1']:.4f}  (GROWS)")
    print(f"  seeded disc, T = 700 K (> Tm): area "
          f"{melt['a0']:.4f} -> {melt['a1']:.4f}  (MELTS)")

    av = run_avrami(dm, mesh, cons, device=args.device)
    print(f"  Avrami run: {av['n_seeds']} nuclei -> X_end = "
          f"{av['X_end']:.3f}, {av['n_grains']} grains resolved")
    print(f"  fitted Avrami exponent n = {av['n_avrami']:.2f} "
          f"(2-D pre-placed nuclei, interface-limited: expect ~2)")
    print("\nBelow Tm the crystal grows, above Tm it melts; the Avrami "
          "exponent reports the growth dimensionality.  Compare with "
          "EXPECTED.md.")


if __name__ == "__main__":
    main()
