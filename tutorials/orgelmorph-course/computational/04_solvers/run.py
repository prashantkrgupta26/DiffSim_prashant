"""OrgElMorph course - Computational C4 driver (the file you run).

    python run.py                # divergence + benchmark + decision support

Runs:
  1. MARCH DIVERGENCE (the headline): march the real 20-step spinodal with
     each (energy, solver) and report the final field range -- cuDSS DIVERGES
     on the polynomial CH saddle (indefinite, no pivoting), survives FH; splu
     is safe on both.
  2. THE TRAP: a residual captured at ONE Newton iterate is deceptively small
     for cuDSS even though the march diverges -- measure the SOLUTION, not a
     one-shot residual.
  3. LIVE 2-D benchmark (factorize+solve, cold+warm, CUDA-synced, median).
  4. Cited 3-D scaling (dev notes).
  5. MEASURED decision support (recommend_solver) returning a rationale.
Then prints PASS/FAIL checks.  Compare with EXPECTED.md."""
import argparse

from solvers import (benchmark_solvers_2d, march_divergence, solver_correctness,
                     CITED_3D, CITED_FACTS, recommend_solver,
                     benchmark_provenance, cudss_available)


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

    print("\n=== 1. MARCH DIVERGENCE (the real correctness test) ===")
    md = march_divergence(device=args.device)
    print(f"  20-step spinodal, level {md['level']}, dt {md['dt']}:")
    print(f"  {'energy':>6} {'solver':>7} {'final c range':>22}  verdict")
    for r in md["rows"]:
        if r["cmin"] is None:
            print(f"  {r['energy']:>6} {r['solver']:>7} {'n/a':>22}  {r['note']}")
            continue
        rng = f"[{r['cmin']:.3g}, {r['cmax']:.3g}]"
        print(f"  {r['energy']:>6} {r['solver']:>7} {rng:>22}  "
              f"{'DIVERGED' if r['diverged'] else 'ok'}")
    print("  -> cuDSS DIVERGES on the poly saddle (f''=3c^2-1 < 0 in the "
          "spinodal band, no pivoting); splu (pivoted) is safe.")

    print("\n=== 2. THE TRAP: one-iterate residual lies ===")
    cc = solver_correctness(level=6, device=args.device)
    for r in cc["rows"]:
        res = "n/a" if r["residual"] is None else f"{r['residual']:.2e}"
        print(f"  {r['solver']:>8} one-iterate residual {res:>10}  {r['note']}")
    print("  -> cuDSS's captured residual is tiny, yet its march diverges: a "
          "small residual != a small error on an indefinite, unpivoted system.")

    print("\n=== 3. LIVE 2-D benchmark (factorize+solve, cold+warm, synced) ===")
    recs = benchmark_solvers_2d(device=args.device)
    print(f"  {'level':>5} {'dofs':>8} {'splu ms':>9} {'cuDSS ms':>9} "
          f"{'speedup':>8}  (timing only; cuDSS is DISQUALIFIED for CH by 1)")
    for r in recs:
        cud = "n/a" if r["cudss_ms"] is None else f"{r['cudss_ms']:.1f}"
        sp = "n/a" if r["speedup"] is None else f"{r['speedup']:.2f}x"
        print(f"  {r['level']:>5} {r['dofs']:>8} {r['splu_ms']:>9.1f} "
              f"{cud:>9} {sp:>8}")

    print("\n=== 4. CITED 3-D scaling (dev notes; masked/film patterns) ===")
    for case, dofs, cu, bk, note in CITED_3D:
        cus = "CEILING" if cu is None else f"{cu:.2f}"
        print(f"  {case:>22} {dofs:>9} cuDSS(masked) {cus:>8}  blockch {bk:>6.2f}")
    print(f"  AMGX verdict: {CITED_FACTS['amgx_verdict'].upper()}; "
          f"matrix-free reaches {CITED_FACTS['matrixfree_dofs']:,} dofs on one "
          f"48 GB card")

    print("\n=== 5. MEASURED decision support (recommend_solver) ===")
    for dofs, dim in ((2_178, 2), (130_000, 2), (200_000, 3),
                      (800_000, 3), (6_000_000, 3)):
        rec = recommend_solver(dofs, dim=dim)
        print(f"  dim={dim} dofs={dofs:>9} -> {rec['recommendation']}")
        print(f"      because: {rec['rationale'][0]}")

    print("\n--- self-check summary ---")
    def row(energy, solver):
        return next(r for r in md["rows"]
                    if r["energy"] == energy and r["solver"] == solver)
    splu_poly = row("poly", "splu")
    checks = {
        "splu safe on poly march (no divergence)": not splu_poly["diverged"],
        "splu safe on FH march": not row("fh", "splu")["diverged"],
        "2-D benchmark ran (>=2 levels)": len(recs) >= 2,
        "small-2-D recommendation is splu":
            recommend_solver(2178, dim=2)["recommendation"] == "splu",
        "recommendation never picks raw-CH cuDSS":
            all("cudss" not in recommend_solver(d, dim=dm)["recommendation"]
                .lower().replace("masked cudss", "")
                for d, dm in ((2178, 2), (130000, 2), (800000, 3))),
    }
    cudss_poly = row("poly", "cudss")
    if cudss_available() and cudss_poly["cmin"] is not None:
        # THE measured finding: cuDSS diverges on the polynomial CH saddle
        checks["cuDSS DIVERGES on poly march (the finding)"] = \
            cudss_poly["diverged"]
        checks["cuDSS survives FH march"] = not row("fh", "cudss")["diverged"]
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  ALL CHECKS: {'PASS' if all(checks.values()) else 'FAIL'}")
    print("\nCompare with EXPECTED.md; render figures with gen_figures.py.")


if __name__ == "__main__":
    main()
