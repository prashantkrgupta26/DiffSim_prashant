"""OrgElMorph course - Computational C1 driver (the file you run).

    python run.py                # both studies, default sizes

Runs the spatial MMS study (p=1 and p=2) and the temporal
self-convergence study (BDF1 and BDF2), then prints a self-check table.
Compare it with EXPECTED.md.  The document's figures and numbers come
from this same core via gen_figures.py."""
import argparse

import numpy as np

from convergence import spatial_mms, temporal_convergence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print("=== SPATIAL convergence (method of manufactured solutions) ===")
    print("  error ~ h^(p+1); observed order printed per level pair")
    sp = {}
    for p, levels in ((1, (4, 5, 6)), (2, (3, 4))):
        r = spatial_mms(p, levels, device=args.device)
        sp[p] = r
        es = ", ".join(f"{e:.2e}" for e in r["errs"])
        os_ = ", ".join(f"{o:.2f}" for o in r["orders"])
        print(f"  p={p}  levels {r['levels']}  errs [{es}]  "
              f"orders [{os_}]  (expect ~{p + 1})")

    print("\n=== TEMPORAL convergence (fixed mesh, shrinking dt) ===")
    print("  BDF1 ~ dt^1, BDF2 ~ dt^2; error vs a fine-dt reference march")
    dts = (8e-3, 4e-3, 2e-3)
    tp = {}
    for order in (1, 2):
        r = temporal_convergence(order, dts, device=args.device)
        tp[order] = r
        es = ", ".join(f"{e:.2e}" for e in r["errs"])
        os_ = ", ".join(f"{o:.2f}" for o in r["orders"])
        name = "BDF1" if order == 1 else "BDF2"
        print(f"  {name}  dt {list(dts)}  errs [{es}]  orders [{os_}]  "
              f"(expect ~{order})")

    print("\n--- self-check summary ---")
    print(f"  p1 spatial order  {sp[1]['order']:.2f}  (> 1.8 required)")
    print(f"  p2 spatial order  {sp[2]['order']:.2f}  (> 2.6 required)")
    print(f"  BDF1 temporal order {tp[1]['order_est']:.2f}  (0.8-1.3)")
    print(f"  BDF2 temporal order {tp[2]['order_est']:.2f}  (> 1.7)")
    ok = (sp[1]["order"] > 1.8 and sp[2]["order"] > 2.6
          and 0.8 < tp[1]["order_est"] < 1.3 and tp[2]["order_est"] > 1.7)
    print(f"  ALL GATES: {'PASS' if ok else 'FAIL'}")
    print("\nCompare with EXPECTED.md; render figures with gen_figures.py.")


if __name__ == "__main__":
    main()
