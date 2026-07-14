"""OrgElMorph course - Physics P5 driver (the file the student runs).

    python run.py                 # sweep evaporation rate (Biot number)
    python run.py --level 6

Dries the same dilute ternary film at several evaporation rates and
prints, for each, the drying time, the final film height, and the
morphology domain scale.  Compare with EXPECTED.md; render the drying
figures with gen_figures.py."""
import argparse

import numpy as np

from evaporation import build_mesh_dm, run

RATES = [0.15, 0.3, 0.6]     # k_e (Biot number): slow -> fast drying


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--t_end", type=float, default=6.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dm, mesh, cons = build_mesh_dm(args.level, device=args.device)

    print(f"\n=== Drying ternary film: evaporation-rate sweep "
          f"({dm.n_nodes} nodes) ===")
    print(f"  {'k_e (Biot)':>11}{'t_dry':>9}{'phi_s final':>13}"
          f"{'h final':>10}{'domain scale':>14}{'contrast':>11}")
    for k_e in RATES:
        r = run(dm, mesh, cons, k_e=k_e, t_end=args.t_end,
                device=args.device)
        print(f"  {k_e:>11.3f}{r['t_dry']:>9.2f}{r['phis_final']:>13.3f}"
              f"{r['h_final']:>10.3f}{r['domain_scale']:>14.2f}"
              f"{r['contrast']:>11.3f}")
    print("\nFaster evaporation (larger k_e) dries sooner and freezes a "
          "FINER morphology (smaller domain scale) -- less time to "
          "coarsen before vitrification.  Compare with EXPECTED.md.")


if __name__ == "__main__":
    main()
