"""OrgElMorph course - Computational C2 driver (the file you run).

    python run.py                # both boundary treatments, compared

Runs the same binary spinodal blend under natural (no-flux) and
Dirichlet boundaries, then prints a self-check table contrasting mass
conservation and the boundary layer.  Compare with EXPECTED.md."""
import argparse

from bc import compare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--wall", type=float, default=0.9,
                    help="Dirichlet boundary composition")
    args = ap.parse_args()

    r = compare(device=args.device, wall=args.wall)
    nf, di = r["noflux"], r["dirichlet"]

    print(f"=== Boundary conditions on a binary spinodal blend "
          f"({nf['side']}x{nf['side']}, {nf['steps']} steps) ===\n")
    print("NATURAL (no-flux) boundary:")
    print(f"  mass drift |dm|   = {nf['mass_drift']:.2e}   "
          f"(conserved -- sealed box)")
    print(f"  edge composition  = {nf['edge_mean']:+.3f}   "
          f"(free; two phases meet the wall)")
    print(f"  field range       = [{nf['c_min']:+.3f}, {nf['c_max']:+.3f}]")
    print("\nDIRICHLET boundary (c pinned to "
          f"{args.wall:+.2f} on every edge):")
    print(f"  mass drift |dm|   = {di['mass_drift']:.3f}    "
          f"(NOT conserved -- reservoir wall)")
    print(f"  edge composition  = {di['edge_mean']:+.3f}   "
          f"(pinned to the wall value)")
    print(f"  field range       = [{di['c_min']:+.3f}, {di['c_max']:+.3f}]")
    print(f"\nfield difference max|c_noflux - c_dirichlet| = "
          f"{r['field_diff']:.3f}   (the boundary's measured effect)\n")

    # gate-style checks
    ok = (nf["mass_drift"] < 1e-10 and di["mass_drift"] > 1e-2
          and abs(di["edge_mean"] - args.wall) < 1e-2
          and r["field_diff"] > 0.5)
    print("checks: no-flux conserves (< 1e-10), Dirichlet does not "
          "(> 1e-2),")
    print("        Dirichlet edge pinned to wall, fields differ (> 0.5)")
    print(f"  ALL CHECKS: {'PASS' if ok else 'FAIL'}")
    print("\nCompare with EXPECTED.md; render figures with gen_figures.py.")


if __name__ == "__main__":
    main()
