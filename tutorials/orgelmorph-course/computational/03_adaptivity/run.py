"""OrgElMorph course - Computational C3 driver (the file you run).

    python run.py                # octree + transfer + real cost + BDF2 order

Runs:
  1. OCTREE refinement + hanging-node constraints (static geometric
     criterion -- honest: this is not solution-adaptive AMR).
  2. The CONSERVATIVE-TRANSFER error (one piece of what true dynamic AMR
     needs): naive injection loses sub-cell mass; averaging is exact.
  3. TEMPORAL adaptivity with REAL cost accounting (accepted/rejected
     steps, full+half solves, Newton iterations, wall) + a matched-accuracy
     fixed-dt sweep.
  4. Variable- vs constant-coefficient BDF2 order under a varying step,
     both MEASURED here.
Then prints PASS/FAIL checks.  Compare with EXPECTED.md."""
import argparse

from adaptivity import (octree_refinement, transfer_error, adaptive_cost,
                        bdf2_variable_order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print("=== 1. Octree refinement + hanging-node constraints ===")
    o = octree_refinement(device=args.device)
    print(f"  adaptive (refined along a STATIC circle): {o['adaptive_nodes']} "
          f"nodes, {o['adaptive_elems']} elements")
    print(f"  uniform L{o['max_level']} (same finest h): {o['uniform_nodes']} "
          f"nodes, {o['uniform_elems']} elements")
    print(f"  node savings {o['node_savings']:.1f}x; CH brick steps on the "
          f"hanging-node mesh: {o['step_ok']}")
    print("  NOTE: this is octree refinement to a fixed geometric criterion,"
          " NOT solution-adaptive AMR (Phase 3).")

    print("\n=== 2. Conservative-transfer error (a piece of true AMR) ===")
    te = transfer_error()
    print(f"  restrict {te['nf']}x{te['nf']} -> coarse, {te['n_drops']} "
          f"sub-cell droplets:")
    print(f"  injection (naive):   mass error {te['inj_err']:.2e} "
          f"({100 * te['inj_err']:.1f}% -- droplets missed)")
    print(f"  cell averaging:      mass error {te['avg_err']:.2e} "
          f"(conservative, exact)")

    print("\n=== 3. Temporal adaptivity -- REAL cost accounting ===")
    ac = adaptive_cost(device=args.device)
    ad = ac["adaptive"]
    print(f"  adaptive ladder to t={ac['t_end']} (tol {ac['tol']:g}):")
    print(f"    accepted {ad['accepted']}, rejected {ad['rejected']}, "
          f"full solves {ad['full_solves']}, half solves {ad['half_solves']}")
    print(f"    Newton iters {ad['newton_iters']}, wall {ad['wall']:.1f}s, "
          f"error {ad['err']:.2e}")
    print("  matched-accuracy fixed-dt sweep:")
    for f in ac["fixed"]:
        print(f"    dt={f['dt']:.0e}: {f['steps']} steps, "
              f"{f['newton_iters']} Newton, {f['wall']:.1f}s, "
              f"err {f['err']:.2e}")
    m = ac["matched"]
    print(f"  matched fixed dt={m['dt']:.0e}: {m['newton_iters']} Newton, "
          f"{m['wall']:.1f}s (adaptive saves "
          f"{ac['newton_savings']:.1f}x Newton, {ac['wall_savings']:.1f}x wall)")

    print("\n=== 4. Variable- vs constant-coefficient BDF2 order ===")
    b = bdf2_variable_order(device=args.device)
    print(f"  variable-coefficient (brick):  order {b['order']:.2f} "
          f"(~2, preserved)  orders {[f'{o:.2f}' for o in b['orders']]}")
    print(f"  constant-coefficient (forced r=1, MEASURED here): "
          f"order {b['const_order']:.2f}  orders "
          f"{[f'{o:.2f}' for o in b['const_orders']]}  (collapses toward 1)")

    print("\n--- self-check summary ---")
    checks = {
        "octree saves > 2x dofs": o["node_savings"] > 2.0,
        "CH steps on hanging-node mesh": o["step_ok"],
        "injection loses mass (> 1%)": te["inj_err"] > 1e-2,
        "averaging conserves (< 1e-10)": te["avg_err"] < 1e-10,
        "adaptive accounting recorded (solves/newton/wall)":
            ad["full_solves"] > 0 and ad["newton_iters"] > 0
            and ad["wall"] > 0,
        "adaptivity saves Newton work vs matched fixed":
            ac["newton_savings"] > 1.0,
        "variable-coeff BDF2 order > 1.7": b["order"] > 1.7,
        "constant-coeff BDF2 order < 1.4": b["const_order"] < 1.4,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  ALL CHECKS: {'PASS' if all(checks.values()) else 'FAIL'}")
    print("\nCompare with EXPECTED.md; render figures with gen_figures.py.")


if __name__ == "__main__":
    main()
