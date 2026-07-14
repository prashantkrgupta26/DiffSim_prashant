"""OrgElMorph course - Computational C1 driver (the file you run).

    python run.py                # full verification study

Runs, in order:
  1. SPATIAL steady-MMS convergence, p=1 (4 levels) and p=2 (4 levels),
     reporting L2 AND H1 orders for BOTH fields c and mu, plus mass.
  2. ALGEBRAIC-error control: the discretization error is invariant to the
     Newton tolerance (tightening it does not move the plot).
  3. TEMPORAL self-convergence, BDF1 and BDF2 (4 dt each), with the
     reference VERIFIED (halve its dt, order stable, Richardson bound).
  4. Four DELIBERATE FAILURES for you to diagnose.

Then prints a PASS/FAIL gate table.  Compare with EXPECTED.md; the
document's figures and numbers come from this same core via gen_figures.py.
"""
import argparse

from convergence import (spatial_mms, temporal_convergence, algebraic_control,
                         verify_reference, DELIBERATE_FAILURES)

SPATIAL = ((1, (3, 4, 5, 6)), (2, (2, 3, 4, 5)))
DTS = (1.6e-2, 8e-3, 4e-3, 2e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print("=== 1. SPATIAL convergence (steady manufactured solution) ===")
    print("  L2 error ~ h^(p+1); H1 seminorm ~ h^p; BOTH fields reported")
    sp = {}
    for p, levels in SPATIAL:
        r = spatial_mms(p, levels, device=args.device)
        sp[p] = r
        print(f"  p={p}  levels {r['levels']}")
        print(f"     L2(c)  {[f'{e:.2e}' for e in r['l2_c']]}  "
              f"order {r['order']:.2f} (expect {p + 1})")
        print(f"     H1(c)  {[f'{e:.2e}' for e in r['h1_c']]}  "
              f"order {r['h1_order']:.2f} (expect {p})")
        print(f"     L2(mu) {[f'{e:.2e}' for e in r['l2_mu']]}  "
              f"order {r['l2_mu_order']:.2f} (expect {p + 1})")
        print(f"     mass   {[f'{e:.1e}' for e in r['mass_err']]}  "
              f"Newton |dx| <= {max(r['newton_dx_inf']):.1e}")

    print("\n=== 2. ALGEBRAIC-error control (p=1, level 5) ===")
    alg = algebraic_control(1, 5, device=args.device)
    for row in alg["rows"]:
        print(f"  Newton tol {row['newton_tol']:.0e}: "
              f"L2(c)={row['l2_c']:.6e}  (|dx|={row['dx_inf']:.1e}, "
              f"{row['iters']} its)")
    print(f"  discretization error moves by {alg['spread_frac']:.1e} of "
          f"itself as tol tightens 1e-4 -> 1e-12  (must be << 1)")

    print("\n=== 3. TEMPORAL convergence (fixed over-resolved mesh) ===")
    tp = {}
    for order in (1, 2):
        r = temporal_convergence(order, DTS, ref_dt=2e-4, device=args.device)
        tp[order] = r
        name = "BDF1" if order == 1 else "BDF2"
        print(f"  {name}  errs {[f'{e:.2e}' for e in r['errs']]}  "
              f"order {r['order_est']:.2f} (expect {order})")
    vref = verify_reference(device=args.device)
    print(f"  reference verified: BDF2 order stable across ref dt "
          f"{vref['ref_dts']} (delta {vref['order_stable']:.3f}); "
          f"Richardson ref error {vref['richardson_ref_err']:.1e}")

    print("\n=== 4. DELIBERATE FAILURES (diagnose these) ===")
    for fn in DELIBERATE_FAILURES:
        d = fn(device=args.device)
        print(f"  {d['name']:42s} measured order {d['order']:+.2f}  "
              f"-> {d['diagnosis']}")

    print("\n--- self-check summary ---")
    checks = {
        "p1 spatial L2(c) order > 1.8": sp[1]["order"] > 1.8,
        "p1 spatial H1(c) order > 0.8": sp[1]["h1_order"] > 0.8,
        "p2 spatial L2(c) order > 2.6": sp[2]["order"] > 2.6,
        "p2 spatial H1(c) order > 1.8": sp[2]["h1_order"] > 1.8,
        "algebraic spread < 1e-6":       alg["spread_frac"] < 1e-6,
        "BDF1 order in [0.8,1.3]":  0.8 < tp[1]["order_est"] < 1.3,
        "BDF2 order > 1.7":              tp[2]["order_est"] > 1.7,
        "reference order stable < 0.1":  vref["order_stable"] < 0.1,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  ALL GATES: {'PASS' if all(checks.values()) else 'FAIL'}")
    print("\nCompare with EXPECTED.md; render figures with gen_figures.py.")


if __name__ == "__main__":
    main()
