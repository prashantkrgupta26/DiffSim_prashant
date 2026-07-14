"""OrgElMorph course - Computational C3 driver (the file you run).

    python run.py                # static octree + dynamic AMR + cost + BDF2

Runs:
  1. STATIC octree refinement + hanging-node constraints (fixed geometric
     criterion -- honest: this is not solution-adaptive AMR).
  2. The CONSERVATIVE-TRANSFER core: the numpy injection-vs-averaging demo,
     then the REAL finite-element L2 projection that conserves mass to
     machine precision under refine AND coarsen (the primitive AMR needs).
  3. DYNAMIC (solution-adaptive) AMR: the full estimate -> mark ->
     refine/coarsen -> 2:1 balance -> rebuild -> transfer (field + BDF
     history) -> continue cycle, with mass/energy continuity across
     remeshes, interface tracking, and an error-vs-dofs payoff study.
  4. SPACE-TIME interaction: a remesh during a variable-step BDF2 march ->
     both history levels must transfer for order 2 to survive.
  5. TEMPORAL adaptivity with REAL cost accounting + matched-accuracy sweep.
  6. Variable- vs constant-coefficient BDF2 order, both MEASURED.
Then prints PASS/FAIL checks.  Compare with EXPECTED.md."""
import argparse

from adaptivity import (octree_refinement, transfer_error,
                        fe_conservative_transfer, dynamic_amr_cycle,
                        amr_error_vs_dofs, spacetime_order, adaptive_cost,
                        bdf2_variable_order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print("=== 1. Static octree refinement + hanging-node constraints ===")
    o = octree_refinement(device=args.device)
    print(f"  adaptive (refined along a STATIC circle): {o['adaptive_nodes']} "
          f"nodes, {o['adaptive_elems']} elements")
    print(f"  uniform L{o['max_level']} (same finest h): {o['uniform_nodes']} "
          f"nodes; node savings {o['node_savings']:.1f}x; CH steps: "
          f"{o['step_ok']}")
    print("  NOTE: fixed geometric criterion, NOT solution-adaptive AMR "
          "(that is section 3).")

    print("\n=== 2. Conservative transfer (the primitive AMR needs) ===")
    te = transfer_error()
    print(f"  numpy demo: injection loses {100 * te['inj_err']:.0f}% mass, "
          f"averaging {te['avg_err']:.1e} (exact)")
    ft = fe_conservative_transfer()
    print(f"  FE L2 projection: mass change refine {ft['mass_refine']:.1e}, "
          f"coarsen {ft['mass_coarsen']:.1e} (machine precision)")
    print(f"  FE transfer convergence order {ft['order']:.2f} (2nd-order)")

    print("\n=== 3. DYNAMIC (solution-adaptive) AMR cycle ===")
    cy = dynamic_amr_cycle(device=args.device)
    print(f"  {cy['remeshes']} remeshes; mass jump {cy['mass_jump']:.1e}, "
          f"energy jump {cy['energy_jump']:.1e} across remeshes")
    print(f"  refined region / interface overlap {cy['overlap_min']:.2f}; "
          f"peak {cy['dofs_peak']} dofs; field finite {cy['finite']}")
    ed = amr_error_vs_dofs(device=args.device)
    print(f"  error vs dofs (ref uniform-L{ed['ref_level']}): AMR err "
          f"{ed['amr_err']:.2e} at {ed['amr_dofs']} dofs")
    print(f"    -> {ed['dof_ratio']:.1f}x fewer dofs and "
          f"{ed['wall_ratio']:.1f}x less wall than uniform at equal error")

    print("\n=== 4. Space-time: remesh during a BDF2 march ===")
    so = spacetime_order(device=args.device)
    print(f"  order across remesh: both-history {so['both_order']:.2f}, "
          f"drop-2nd-level {so['drop_order']:.2f} "
          f"(drop error {so['drop_err_penalty']:.1f}x larger)")

    print("\n=== 5. Temporal adaptivity -- REAL cost accounting ===")
    ac = adaptive_cost(device=args.device)
    ad = ac["adaptive"]
    print(f"  adaptive to t={ac['t_end']} (tol {ac['tol']:g}): "
          f"acc {ad['accepted']}, rej {ad['rejected']}, "
          f"Newton {ad['newton_iters']}, err {ad['err']:.2e}")
    m = ac["matched"]
    print(f"  matched fixed dt={m['dt']:.0e}: {m['newton_iters']} Newton "
          f"(adaptive saves {ac['newton_savings']:.1f}x Newton, "
          f"{ac['wall_savings']:.1f}x wall)")

    print("\n=== 6. Variable- vs constant-coefficient BDF2 order ===")
    b = bdf2_variable_order(device=args.device)
    print(f"  variable-coefficient (brick): order {b['order']:.2f} (~2)")
    print(f"  constant-coefficient (forced r=1): order {b['const_order']:.2f} "
          f"(collapses toward 1)")

    print("\n--- self-check summary ---")
    checks = {
        "octree saves > 2x dofs": o["node_savings"] > 2.0,
        "CH steps on hanging-node mesh": o["step_ok"],
        "injection loses mass (> 1%)": te["inj_err"] > 1e-2,
        "FE transfer conserves refine (< 1e-12)": ft["mass_refine"] < 1e-12,
        "FE transfer conserves coarsen (< 1e-12)": ft["mass_coarsen"] < 1e-12,
        "FE transfer 2nd order (> 1.7)": ft["order"] > 1.7,
        "AMR mass conserved across remesh (< 1e-11)": cy["mass_jump"] < 1e-11,
        "AMR energy continuous across remesh (< 1e-3)":
            cy["energy_jump"] < 1e-3,
        "AMR tracks interface (overlap > 0.8)": cy["overlap_min"] > 0.8,
        "AMR stable (finite, bounded)":
            cy["finite"] and cy["c_min"] > -1.2 and cy["c_max"] < 1.2,
        "AMR reaches equal error at >2x fewer dofs": ed["dof_ratio"] > 2.0,
        "BDF2 order recovered across remesh (> 1.7)": so["both_order"] > 1.7,
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
