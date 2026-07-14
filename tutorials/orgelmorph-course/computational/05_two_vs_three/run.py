"""OrgElMorph course - Computational C5 driver (the file you run).

    python run.py                # scaling + sparsity + memory + physics

Runs:
  1. LIVE scaling of real CH systems: dofs and nnz/dof growth, 2-D vs 3-D.
  2. SPARSITY decomposed to its FE origins (coupled nodes x fields), not a
     3^d-1 finite-difference stencil.
  3. COMPLETE memory accounting via estimate_capacity (all buffers), + the
     int32 CSR ceiling (an impl choice; int64/distributed lifts it).
  4. PHYSICS at equal resolution: 2-D is not cheap 3-D (interfacial-area
     density, phase fractions, S(q) wavelength differ).
  5. Cited device-scale ladder (dev notes).
Then prints PASS/FAIL checks.  Compare with EXPECTED.md."""
import argparse

from scaling import (measure_scaling, sparsity_breakdown, memory_accounting,
                     physics_comparison, CITED_LADDER, INT32_MAX,
                     int32_headroom)


def _print_rows(recs, tag):
    print(f"  {tag}:")
    print(f"    {'level':>5} {'dofs':>8} {'nnz':>10} {'nnz/dof':>8} "
          f"{'mem MB':>8} {'step s':>7}")
    for r in recs:
        print(f"    {r['level']:>5} {r['dofs']:>8} {r['nnz']:>10} "
              f"{r['nnz_per_dof']:>8.1f} {r['mem_mb']:>8.1f} "
              f"{r['step_s']:>7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print("=== 1. LIVE scaling: real Cahn-Hilliard systems ===")
    s = measure_scaling(device=args.device)
    _print_rows(s["two_d"], "2-D (uniform box)")
    _print_rows(s["three_d"], "3-D (uniform box)")
    print(f"  dofs per refinement level: 2-D x{s['dof_growth_2d']:.1f}, "
          f"3-D x{s['dof_growth_3d']:.1f}  (theory: x4 vs x8)")

    print("\n=== 2. SPARSITY, decomposed to its FE origins ===")
    for dim in (2, 3):
        sb = sparsity_breakdown(dim)
        print(f"  {dim}-D (p=1, fields=2): {sb['coupled_nodes']} coupled "
              f"nodes x {sb['fields']} fields = {sb['nnz_per_dof']} nnz/dof "
              f"(measured ~{s[f'nnz_per_dof_{dim}d']:.0f}); a 3^d-1 "
              f"stencil would wrongly say {sb['fd_stencil_would_say']}")

    print("\n=== 3. COMPLETE memory accounting (estimate_capacity) ===")
    for dim, n, solver in ((2, 256, "splu"), (3, 128, "blockch"),
                           (3, 256, "blockch")):
        est = memory_accounting(dim, n, solver=solver)
        biggest = max(est["components"].items(), key=lambda kv: kv[1])
        print(f"  {dim}-D n={n} ({solver}): {est['dofs']:,} dofs, "
              f"{est['total_gb']:.2f} GB total; biggest = {biggest[0]} "
              f"({100 * biggest[1] / est['total_bytes']:.0f}%); "
              f"fits 48 GB: {est['fits_card']}; "
              f"int32 headroom {est['int32_headroom']:.2f}")
    print("  (int32 CSR indexing is an IMPLEMENTATION choice; an int64 build "
          "or a distributed/block-masked pattern lifts the 2^31 wall.)")

    print("\n=== 4. PHYSICS at equal resolution (2-D is not cheap 3-D) ===")
    pc = physics_comparison(device=args.device)
    for dim in (2, 3):
        p = pc[dim]
        print(f"  {dim}-D (h={p['dx']:.3f}): interfacial-area density "
              f"{p['interfacial_area_density']:.2f} (length^{dim - 1}/vol), "
              f"S(q) wavelength {p['peak_wavelength']:.3f}, "
              f"phase split {p['phase_frac_low']:.2f}/{p['phase_frac_high']:.2f}")
    print("  -> interfaces are curves in 2-D, surfaces in 3-D; the topology "
          "and coarsening differ, so 2-D answers a DIFFERENT question.")

    print("\n=== 5. CITED device-scale ladder (dev notes; not re-run) ===")
    print(f"  {'case':>28} {'dofs':>11} {'nnz':>14} {'int32?':>7}")
    for label, dofs, nnz, note in CITED_LADDER:
        frac = int32_headroom(nnz)
        flag = "OVER" if frac > 1.0 else f"{frac * 100:.0f}%"
        print(f"  {label:>28} {dofs:>11,} {nnz:>14,} {flag:>7}   {note}")

    print("\n--- self-check summary ---")
    checks = {
        "3-D dofs grow faster (x8 vs x4)":
            s["dof_growth_3d"] > s["dof_growth_2d"],
        "3-D denser (nnz/dof)":
            s["nnz_per_dof_3d"] > s["nnz_per_dof_2d"],
        "sparsity matches FE model (not 3^d-1)":
            abs(sparsity_breakdown(3)["nnz_per_dof"] - 54) < 1,
        "estimate flags 256^3 over 48 GB":
            not memory_accounting(3, 256)["fits_card"],
        "physics differs 2-D vs 3-D (area density)":
            (pc[3]["interfacial_area_density"]
             != pc[2]["interfacial_area_density"]),
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  ALL CHECKS: {'PASS' if all(checks.values()) else 'FAIL'}")
    print("\nCompare with EXPECTED.md; render figures with gen_figures.py.")


if __name__ == "__main__":
    main()
