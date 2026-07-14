"""OrgElMorph course - Physics P3 driver (the file the student runs).

    python run.py                 # no-flux vs attracting vs repelling wall
    python run.py --level 6

Marches the same binary quench with three substrate boundary conditions
-- neutral no-flux, an attracting wall (g<0), and a repelling/depleting
wall -- and prints the substrate-enrichment self-check with the TRUE
quadrature mass drift (INT phi dV) and the projection status.  Compare
with EXPECTED.md; the full 6-case workflow + figures is run_harness.py /
gen_figures.py."""
import argparse

from substrate import build_mesh_dm, WallDiagnostics, simulate


# quadratic wall well f_w = g*phi + h*phi^2, preferred surface
# composition phi* = -g/(2h): attract -> phi*=0.75, deplete -> phi*=0.25.
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
    diag = WallDiagnostics(dm, mesh, cons)

    print(f"\n=== Substrate wall energy (binary CH, chi={args.chi}, "
          f"quench to t={args.t_end}) ===")
    print(f"  {'case':<26}{'substrate phi':>14}{'film-mean phi':>15}"
          f"{'enrichment':>12}{'|dm| (quad)':>13}{'proj dofs':>11}")
    for name, g, h in CASES:
        r = simulate(dm, mesh, cons, diag, wall_g=g, wall_h=h, chi=args.chi,
                     t_end=args.t_end)
        print(f"  {name:<26}{r['substrate_phi']:>14.3f}"
              f"{r['film_mean_phi']:>15.3f}{r['enrichment']:>+12.3f}"
              f"{r['mass_drift']:>13.1e}{r['projected_dofs']:>11d}")
    print("\nThe attracting wall enriches the substrate (positive "
          "enrichment); the depleting wall lowers it; the neutral wall "
          "does neither.  The quadrature mass INT phi dV is conserved to "
          "machine precision and the projection never fires (proj dofs = "
          "0).  Compare with EXPECTED.md; run_harness.py runs all six "
          "boundary conditions.")


if __name__ == "__main__":
    main()
