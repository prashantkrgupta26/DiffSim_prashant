"""OrgElMorph course - Computational C4 driver (the file you run).

    python run.py                # live 2-D solver benchmark + cited 3-D

Times direct CPU (splu) vs direct GPU (cuDSS) on the real Cahn-Hilliard
Jacobian at a few 2-D sizes, prints the crossover, and echoes the
measured 3-D scaling story from the dev notes.  Compare with
EXPECTED.md."""
import argparse

from solvers import (benchmark_solvers_2d, CITED_3D, CITED_FACTS,
                     choose_solver)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print("=== LIVE 2-D linear-solve benchmark (real CH Jacobian) ===")
    print("  factorize + solve; splu = CPU direct, cuDSS = GPU direct\n")
    print(f"  {'level':>5} {'dofs':>8} {'nnz':>9} {'splu ms':>10} "
          f"{'cuDSS ms':>10} {'speedup':>9}")
    recs = benchmark_solvers_2d(device=args.device)
    for r in recs:
        print(f"  {r['level']:>5} {r['dofs']:>8} {r['nnz']:>9} "
              f"{r['splu_ms']:>10.1f} {r['cudss_ms']:>10.1f} "
              f"{r['speedup']:>8.2f}x")
    # locate the crossover (splu -> cuDSS)
    cross = next((r for r in recs if r["speedup"] > 1.0), None)
    print()
    if cross:
        print(f"  crossover: cuDSS overtakes splu by "
              f"~{cross['dofs']} dofs (2-D)")
    print("  -> in 2-D use splu when tiny, cuDSS as it grows; "
          "blockch is 26-500x SLOWER here (cited) -- never in 2-D.")

    print("\n=== CITED 3-D scaling (dev notes; too slow to re-run) ===")
    print(f"  {'case':>22} {'dofs':>9} {'cuDSS s/call':>13} "
          f"{'blockch s/call':>15}")
    for case, dofs, cu, bk, note in CITED_3D:
        cus = "CEILING" if cu is None else f"{cu:.2f}"
        print(f"  {case:>22} {dofs:>9} {cus:>13} {bk:>13.2f}   {note}")
    print(f"\n  cuDSS factorization ceiling on a 48 GB card: "
          f"~{CITED_FACTS['cudss_ceiling_dofs']} dofs")
    print(f"  blockch_dev beats cuDSS {CITED_FACTS['blockch_slab64_speedup']}"
          f"x on the slab64 solve, and marches WHERE cuDSS ceilings")
    print(f"  block-masked cuDSS cuts the 3-D fill (17.7 -> "
          f"{CITED_FACTS['masked_cudss_slab64_scall']} s/call) -- strong "
          f"under ~5e5 dofs")
    print(f"  AMGX verdict: {CITED_FACTS['amgx_verdict'].upper()} "
          f"(mass-dominated inners; AMG adds nothing at production dt)")
    print(f"  matrix-free / blockch: 128x128x64 = "
          f"{CITED_FACTS['matrixfree_dofs']:,} dofs runs on ONE 48 GB "
          f"card (cuDSS cannot)")

    print("\n=== the choice, as a rule ===")
    for dim, dofs in ((2, 3_000), (2, 100_000), (3, 200_000),
                      (3, 800_000), (3, 6_000_000)):
        print(f"  dim={dim}  dofs={dofs:>9}  ->  {choose_solver(dim, dofs)}")

    ok = recs[0]["speedup"] < 1.0 and recs[-1]["speedup"] > 1.0
    print(f"\n  CHECK (splu wins tiny, cuDSS wins large): "
          f"{'PASS' if ok else 'FAIL'}")
    print("\nCompare with EXPECTED.md; render figures with gen_figures.py.")


if __name__ == "__main__":
    main()
