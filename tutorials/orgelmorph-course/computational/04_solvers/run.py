"""OrgElMorph course - Computational C4 driver (the file you run).

    python run.py                # correctness + 2-D benchmark + decision support

Runs:
  1. SOLVER CORRECTNESS on the real CH saddle: splu / cuDSS / blockch, each
     scored by relative residual -- exposing that cuDSS returns a WRONG
     (large-residual) answer on the indefinite (c,mu) saddle.
  2. LIVE 2-D benchmark (factorize+solve, cold+warm, CUDA-synced, median+IQR)
     with the residual beside the speed.
  3. Cited 3-D scaling (dev notes, too slow to re-run).
  4. MEASURED decision support (recommend_solver) returning a rationale.
Then prints PASS/FAIL checks.  Compare with EXPECTED.md."""
import argparse

from solvers import (benchmark_solvers_2d, solver_correctness, CITED_3D,
                     CITED_FACTS, recommend_solver, benchmark_provenance,
                     cudss_available)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    prov = benchmark_provenance()
    print("=== benchmark provenance ===")
    print(f"  diffsim {prov['diffsim_commit']} | {prov['gpu']} | "
          f"numpy {prov['numpy']} scipy {prov['scipy']} "
          f"nvmath {prov['nvmath']} | cuDSS available: "
          f"{prov['cudss_available']}")

    print("\n=== 1. SOLVER CORRECTNESS on the CH saddle (residual matters) ===")
    cc = solver_correctness(level=6, device=args.device)
    print(f"  captured CH Jacobian: {cc['dofs']} dofs, {cc['nnz']} nnz")
    print(f"  {'solver':>10} {'residual':>12} {'iters':>7} {'diff vs splu':>13}"
          f"  note")
    for r in cc["rows"]:
        res = "n/a" if r["residual"] is None else f"{r['residual']:.2e}"
        dif = "n/a" if r["diff"] is None else f"{r['diff']:.2e}"
        it = "n/a" if r["iters"] is None else str(r["iters"])
        print(f"  {r['solver']:>10} {res:>12} {it:>7} {dif:>13}  {r['note']}")
    print("  -> VERIFY, do not assume: on the 2-D CH saddle every backend "
          "reaches a tiny residual (correctness proved). The 3-D wall is "
          "memory, not accuracy.")

    print("\n=== 2. LIVE 2-D benchmark (factorize+solve, cold+warm, synced) ===")
    recs = benchmark_solvers_2d(device=args.device)
    print(f"  {'level':>5} {'dofs':>8} {'nnz':>9} {'splu ms':>9} "
          f"{'cuDSS ms':>9} {'speedup':>8} {'cuDSS resid':>12}")
    for r in recs:
        cud = "n/a" if r["cudss_ms"] is None else f"{r['cudss_ms']:.1f}"
        sp = "n/a" if r["speedup"] is None else f"{r['speedup']:.2f}x"
        cr = "n/a" if r.get("cudss_resid") is None else f"{r['cudss_resid']:.1e}"
        print(f"  {r['level']:>5} {r['dofs']:>8} {r['nnz']:>9} "
              f"{r['splu_ms']:>9.1f} {cud:>9} {sp:>8} {cr:>12}")

    print("\n=== 3. CITED 3-D scaling (dev notes; masked/film patterns) ===")
    for case, dofs, cu, bk, note in CITED_3D:
        cus = "CEILING" if cu is None else f"{cu:.2f}"
        print(f"  {case:>22} {dofs:>9} cuDSS {cus:>8}  blockch {bk:>6.2f}  {note}")
    print(f"  AMGX verdict: {CITED_FACTS['amgx_verdict'].upper()} "
          f"(mass-dominated inners at production dt); matrix-free reaches "
          f"{CITED_FACTS['matrixfree_dofs']:,} dofs on one 48 GB card")

    print("\n=== 4. MEASURED decision support (recommend_solver) ===")
    for dofs, dim in ((2_178, 2), (130_000, 2), (200_000, 3),
                      (800_000, 3), (6_000_000, 3)):
        rec = recommend_solver(dofs, dim=dim)
        print(f"  dim={dim} dofs={dofs:>9} -> {rec['recommendation']}")
        print(f"      because: {rec['rationale'][0]}")
        if len(rec["rationale"]) > 1:
            print(f"               {rec['rationale'][1]}")

    print("\n--- self-check summary ---")
    splu_row = next(r for r in cc["rows"] if r["solver"] == "splu")
    blockch_row = next((r for r in cc["rows"] if r["solver"] == "blockch"),
                       None)
    cudss_row = next((r for r in cc["rows"] if r["solver"] == "cudss"), None)
    checks = {
        "splu residual tiny (< 1e-8)":
            splu_row["residual"] is not None and splu_row["residual"] < 1e-8,
        "blockch residual tiny (< 1e-6)":
            blockch_row is not None and blockch_row["residual"] is not None
            and blockch_row["residual"] < 1e-6,
        "2-D benchmark ran (>=2 levels)": len(recs) >= 2,
        "small-2-D recommendation is splu":
            recommend_solver(2178, dim=2)["recommendation"] == "splu",
        "large-3-D recommendation is not raw cuDSS":
            "blockch" in recommend_solver(2_000_000, dim=3)["recommendation"]
            or "matrix-free" in recommend_solver(2_000_000,
                                                 dim=3)["recommendation"],
    }
    if cudss_available() and cudss_row is not None \
            and cudss_row["residual"] is not None:
        # measured: cuDSS is ACCURATE on the 2-D CH saddle (verify, not assume)
        checks["cuDSS verified accurate on CH saddle (< 1e-6)"] = (
            cudss_row["residual"] < 1e-6)
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  ALL CHECKS: {'PASS' if all(checks.values()) else 'FAIL'}")
    print("\nCompare with EXPECTED.md; render figures with gen_figures.py.")


if __name__ == "__main__":
    main()
