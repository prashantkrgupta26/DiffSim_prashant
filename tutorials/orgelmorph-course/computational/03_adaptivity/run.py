"""OrgElMorph course - Computational C3 driver (the file you run).

    python run.py                # octree + time-ladder + BDF2 order

Runs the three adaptivity studies and prints a self-check table.
Compare with EXPECTED.md."""
import argparse

from adaptivity import (octree_refinement, adaptive_time_stepping,
                        bdf2_variable_order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print("=== 1. SPATIAL adaptivity: octree refinement near an "
          "interface ===")
    o = octree_refinement(device=args.device)
    print(f"  adaptive mesh: {o['adaptive_nodes']} nodes, "
          f"{o['adaptive_elems']} elements (refined along a circle)")
    print(f"  uniform L{o['max_level']}: {o['uniform_nodes']} nodes, "
          f"{o['uniform_elems']} elements (same finest resolution)")
    print(f"  node savings {o['node_savings']:.1f}x, "
          f"element savings {o['elem_savings']:.1f}x")
    print(f"  CH brick steps on the (hanging-node) adaptive mesh: "
          f"{o['step_ok']}")

    print("\n=== 2. TEMPORAL adaptivity: LTE-controlled step ladder over "
          "a quench ===")
    t = adaptive_time_stepping(device=args.device)
    print(f"  {t['n_adaptive']} adaptive steps to t={t['t_end']} "
          f"(tol={t['tol']:g})")
    print(f"  dt spans {t['dt_min']:.1e} (onset floor) -> "
          f"{t['dt_max']:.1e} (coarsening): {t['span']:.0f}x range")
    print(f"  a fixed-dt march safe throughout would need ~{t['n_fixed']} "
          f"steps -> {t['savings']:.0f}x step savings")
    print(f"  march stays physical: c in "
          f"[{t['c_min']:+.2f}, {t['c_max']:+.2f}]")

    print("\n=== 3. Why variable-coefficient BDF2 (order under varying "
          "dt) ===")
    b = bdf2_variable_order(device=args.device)
    os_ = ", ".join(f"{o:.2f}" for o in b["orders"])
    print(f"  variable-coefficient BDF2 on alternating dt: orders "
          f"[{os_}]  mean {b['order']:.2f} (~2, order preserved)")
    print(f"  constant-coefficient baseline (cited, G3 audit): "
          f"{b['const_coeff_order'][0]:.2f}/{b['const_coeff_order'][1]:.2f}"
          f"  (order COLLAPSES toward 1)")

    ok = (o["node_savings"] > 2.0 and o["step_ok"]
          and t["growth"] > 4.0 and b["order"] > 1.7)
    print(f"\n  ALL CHECKS: {'PASS' if ok else 'FAIL'}")
    print("  (octree saves > 2x dofs; dt grows > 4x; var-coeff BDF2 "
          "order > 1.7)")
    print("\nCompare with EXPECTED.md; render figures with gen_figures.py.")


if __name__ == "__main__":
    main()
