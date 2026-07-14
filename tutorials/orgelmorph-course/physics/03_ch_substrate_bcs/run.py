"""OrgElMorph course - Physics P3 driver (the file the student runs).

    python run.py                 # no-flux vs attracting vs repelling wall
    python run.py --level 6

Marches the same binary quench with three substrate boundary conditions
-- neutral no-flux, an attracting wall (g<0), and a repelling wall
(g>0) -- and prints the substrate-enrichment self-check.  Compare with
EXPECTED.md; render the document's figures with gen_figures.py."""
import argparse

import numpy as np

from substrate import build_mesh_dm, run


# quadratic wall well f_w = g*phi + h*phi^2, preferred surface
# composition phi* = -g/(2h): attract -> phi*=0.75, repel -> phi*=0.25.
CASES = [("no-flux (neutral)", 0.0, 0.0),
         ("attracting wall (phi*=0.75)", -1.5, 1.0),
         ("repelling wall (phi*=0.25)", -0.5, 1.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--chi", type=float, default=2.2)
    ap.add_argument("--t_end", type=float, default=0.4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dm, mesh, cons = build_mesh_dm(args.level, device=args.device)

    print(f"\n=== Substrate wall energy (binary CH, chi={args.chi}, "
          f"quench to t={args.t_end}) ===")
    print(f"  {'case':<26}{'substrate phi':>14}{'film-mean phi':>15}"
          f"{'enrichment':>12}{'mass drift':>13}")
    for name, g, h in CASES:
        r = run(dm, mesh, cons, wall_g=g, wall_h=h, chi=args.chi,
                t_end=args.t_end, device=args.device)
        enr = r["wall_phi"][-1] - r["bulk_phi"][-1]
        drift = abs(r["mass_final"] - r["mass0"])
        print(f"  {name:<26}{r['wall_phi'][-1]:>14.3f}"
              f"{r['bulk_phi'][-1]:>15.3f}{enr:>+12.3f}{drift:>13.1e}")
    print("\nThe attracting wall enriches the substrate (positive "
          "enrichment); the repelling wall depletes it; mass is "
          "conserved in every case.  Compare with EXPECTED.md.")


if __name__ == "__main__":
    main()
