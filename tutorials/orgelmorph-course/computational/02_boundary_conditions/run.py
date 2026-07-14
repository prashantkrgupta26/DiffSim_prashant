"""OrgElMorph course - Computational C2 driver (the file you run).

    python run.py                # the full boundary-condition study

Runs, in order:
  1. The same binary spinodal blend under natural (no-flux) and Dirichlet
     boundaries, contrasting mass conservation and the boundary layer.
  2. The FLUX BALANCE d/dt Int c = -Int J.n: measured dm/dt is zero for
     no-flux and a decaying reservoir influx for Dirichlet.
  3. A BC TEST MATRIX over the boundary taxonomy.
  4. WEAK vs STRONG Dirichlet on a tiny system: row replacement (strong,
     asymmetric), symmetric elimination, and a penalty (weakly imposed,
     O(1/beta)).
Then prints PASS/FAIL checks.  Compare with EXPECTED.md."""
import argparse

from bc import (compare, flux_balance, bc_test_matrix,
                strong_vs_weak_dirichlet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--wall", type=float, default=0.9,
                    help="Dirichlet boundary composition")
    args = ap.parse_args()

    r = compare(device=args.device, wall=args.wall)
    nf, di = r["noflux"], r["dirichlet"]
    print(f"=== 1. Natural vs Dirichlet ({nf['side']}x{nf['side']}, "
          f"{nf['steps']} steps) ===")
    print(f"  no-flux   : mass drift {nf['mass_drift']:.2e} (conserved), "
          f"edge {nf['edge_mean']:+.3f} (free)")
    print(f"  Dirichlet : mass drift {di['mass_drift']:.3f} (reservoir), "
          f"edge {di['edge_mean']:+.3f} (pinned to {args.wall:+.2f})")
    print(f"  field difference max|c_nf - c_dir| = {r['field_diff']:.3f}")

    print("\n=== 2. Flux balance  d/dt Int c = -Int J.n ===")
    fb = {m: flux_balance(m, wall=args.wall, device=args.device)
          for m in ("noflux", "dirichlet")}
    print(f"  no-flux   : |net flux|_max = {fb['noflux']['flux_abs_max']:.2e}"
          f"  (zero: mass exactly conserved)")
    print(f"  Dirichlet : influx {fb['dirichlet']['flux_early']:+.2f} (early) "
          f"-> {fb['dirichlet']['flux_late']:+.2f} (late; reservoir shuts off)")

    print("\n=== 3. BC test matrix ===")
    mat = bc_test_matrix(device=args.device)
    print(f"  {'configuration':40s} {'#pinned':>7} {'mass drift':>11} "
          f"{'edge':>7}")
    for m in mat:
        print(f"  {m['name']:40s} {m['npin']:7d} {m['mass_drift']:11.3e} "
              f"{m['edge_mean']:+7.3f}")

    print("\n=== 4. Weak vs strong Dirichlet (tiny -u''=0 system) ===")
    w = strong_vs_weak_dirichlet()
    print(f"  row-replacement matrix symmetric? {w['rr_symmetric']}  "
          f"(strong, but breaks SPD)")
    print(f"  symmetric-elimination symmetric?  {w['sym_symmetric']}  "
          f"(strong, keeps SPD)")
    print(f"  strong solution error vs exact: {w['strong_err']:.1e}")
    print("  weak (penalty) boundary error vs beta:")
    for p in w["penalty"]:
        print(f"     beta {p['beta']:.0e}: u0={p['u0']:.4f}  "
              f"bc_err {p['bc_err']:.1e}")

    print("\n--- self-check summary ---")
    checks = {
        "no-flux conserves (< 1e-10)": nf["mass_drift"] < 1e-10,
        "Dirichlet reservoir (> 1e-2)": di["mass_drift"] > 1e-2,
        "Dirichlet edge pinned": abs(di["edge_mean"] - args.wall) < 1e-2,
        "fields differ (> 0.5)": r["field_diff"] > 0.5,
        "no-flux net flux ~ 0": fb["noflux"]["flux_abs_max"] < 1e-8,
        "Dirichlet flux decays": (abs(fb["dirichlet"]["flux_late"])
                                  < abs(fb["dirichlet"]["flux_early"])),
        "row-replace breaks symmetry": not w["rr_symmetric"],
        "sym-elim keeps symmetry": w["sym_symmetric"],
        "penalty converges O(1/beta)": (w["penalty"][-1]["bc_err"]
                                        < 0.1 * w["penalty"][0]["bc_err"]),
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  ALL CHECKS: {'PASS' if all(checks.values()) else 'FAIL'}")
    print("\nCompare with EXPECTED.md; render figures with gen_figures.py.")


if __name__ == "__main__":
    main()
