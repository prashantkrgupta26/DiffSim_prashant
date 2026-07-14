"""OrgElMorph course - Physics P2 driver (the file the student runs).

    python run.py                 # fixed vs adaptive, 64x64
    python run.py --tol 3e-4      # tighter error tolerance
    python run.py --level 6

Marches the SAME binary quench with a fixed time step and with the
LTE-controlled adaptive stepper, then prints a self-check table
comparing step count (cost) and final energy/morphology (accuracy).
Compare the printed table with EXPECTED.md; render the document's
figures with gen_figures.py."""
import argparse

import numpy as np

from adaptive import build_mesh_dm, compare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--t_end", type=float, default=0.8)
    ap.add_argument("--dt_fixed", type=float, default=4e-3)
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--order", type=int, default=2)   # BDF2
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dm, mesh, cons = build_mesh_dm(args.level, device=args.device)
    r = compare(dm, mesh, t_end=args.t_end, dt_fixed=args.dt_fixed,
                tol=args.tol, order=args.order, device=args.device)
    fx, ad = r["fixed"], r["adaptive"]

    print(f"\n=== Naive fixed vs adaptive BDF{args.order} "
          f"({fx['side']}x{fx['side']}, quench to t={args.t_end}) ===")
    print(f"  fixed    dt = {args.dt_fixed:g}:   "
          f"{fx['nsteps']:5d} steps   F_end = {fx['F'][-1]:.5g}")
    print(f"  adaptive tol = {args.tol:g}: "
          f"{ad['nsteps']:5d} steps   F_end = {ad['F'][-1]:.5g}")
    print(f"  adaptive dt range: {r['dt_min']:.2e} "
          f"-> {ad['dt_hist'].max():.3f}")
    print(f"  step count: adaptive takes {r['speedup']:.1f}x fewer accepted "
          f"steps than this fixed dt")
    print(f"  NAIVE WORST CASE (not the speed-up): a fixed dt pinned at the "
          f"quench's smallest step (dt<={r['dt_min']:.1e}) would need "
          f"~{r['implied_fixed']:,} steps")
    print(f"  final energy: fixed {fx['F'][-1]:.4g} vs adaptive "
          f"{ad['F'][-1]:.4g} (|dF| = {r['dF_end']:.3g}); note: energy does "
          f"NOT rank accuracy -- see run_harness.py for the reference study")
    print(f"  robust statistics agree: domain scale "
          f"{r['dscale_fixed']:.1f} vs {r['dscale_adapt']:.1f} cells; "
          f"phases c in [{r['crange_adapt'][0]:.2f}, "
          f"{r['crange_adapt'][1]:.2f}]")
    both_decrease = bool(np.all(np.diff(fx["F"][1:]) <= 1e-9)
                         and np.all(np.diff(ad["F"][1:]) <= 1e-9))
    print(f"  energy monotone decreasing (both): {both_decrease}")

    np.savez("out_p2.npz", t_fixed=fx["t"], F_fixed=fx["F"],
             t_adapt=ad["t"], F_adapt=ad["F"], dt_hist=ad["dt_hist"],
             field_fixed=fx["field"], field_adapt=ad["field"])
    print("\nArrays written to out_p2.npz.  Compare with EXPECTED.md; "
          "render figures with gen_figures.py.")


if __name__ == "__main__":
    main()
