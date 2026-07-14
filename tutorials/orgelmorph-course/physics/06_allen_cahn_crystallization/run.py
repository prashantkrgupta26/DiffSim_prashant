"""OrgElMorph course - Physics P6 driver (the file the student runs).

    python run.py                 # full crystallization battery + checks
    python run.py --level 6       # finer mesh (slower)

Runs, in order:
  1. a seeded crystal below and above the melting point (growth vs
     melting), reported by the QUADRATURE crystallinity <psi> (primary)
     and a thresholded area (secondary);
  2. the interface velocity v as a function of undercooling dT = Tm - T
     (a sweep below Tm);
  3. the critical radius r* that separates growing from redissolving
     seeds, at two undercoolings (deeper undercooling -> smaller r*);
  4. a multi-nucleus run whose crystalline fraction is fit to the
     Avrami/JMAK law (baseline-corrected exponent with R^2 and a 95% CI).

Results are written to outputs/results.json and checked against
baseline.yaml (tolerance-based, not bit-identical).  Compare with
EXPECTED.md; render the figures with gen_figures.py.

Note: T = 250 K and L_psi = 11 in the Avrami run are ACCELERATED
pedagogical values (see crystallization.py) chosen so X sweeps the full
range in a short run; they are not the physical PCBM rate.
"""
import argparse
import json
import os
import sys

import numpy as np

from crystallization import (
    build_mesh_dm, run_grow_melt, run_avrami, sweep_interface_velocity,
    run_critical_radius, drive_of, TM)

# make the shared course harness importable (common/check_results.py)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from common import check_results as _check  # noqa: E402

# undercooling sweep for the interface velocity: all deep enough that the
# r0 = 0.15 seed is supercritical, so v is a genuine growth-front velocity
# (at milder undercooling the same finite seed is subcritical and retreats
# -- that is experiment [3], the critical radius).
IV_TEMPS = [420.0, 390.0, 360.0, 333.0, 300.0]
# critical-radius brackets at two undercoolings
CRIT_DEEP_T, CRIT_DEEP_R = 333.0, (0.02, 0.03, 0.04, 0.05, 0.06)
CRIT_MILD_T, CRIT_MILD_R = 400.0, (0.08, 0.11, 0.14, 0.17, 0.20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=os.path.join(_HERE, "outputs"))
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()
    dm, mesh, cons = build_mesh_dm(args.level, device=args.device)

    print(f"\n=== Allen--Cahn crystallization (PCBM-class, Tm={TM:g} K, "
          f"level {args.level}) ===")

    # 1. grow / melt (quadrature primary, threshold secondary)
    grow = run_grow_melt(dm, mesh, cons, 333.0, device=args.device)
    melt = run_grow_melt(dm, mesh, cons, 700.0, device=args.device)
    print("\n[1] grow / melt  (<psi> quadrature; area = psi>0.5)")
    print(f"  T = 333 K (< Tm, drive {grow['drive']:+.3f}): "
          f"<psi> {grow['qpsi0']:.4f} -> {grow['qpsi1']:.4f}  "
          f"(area {grow['a0']:.4f} -> {grow['a1']:.4f})  GROWS")
    print(f"  T = 700 K (> Tm, drive {melt['drive']:+.3f}): "
          f"<psi> {melt['qpsi0']:.4f} -> {melt['qpsi1']:.4f}  "
          f"(area {melt['a0']:.4f} -> {melt['a1']:.4f})  MELTS")
    print(f"  threshold sensitivity of grown end area {grow['thr_end']}")

    # 2. interface velocity vs undercooling
    sw = sweep_interface_velocity(dm, mesh, cons, IV_TEMPS,
                                  device=args.device)
    print("\n[2] interface velocity v vs undercooling dT = Tm - T")
    for r in sw["runs"]:
        print(f"  T = {r['T']:.0f} K  dT = {r['dT']:5.0f}  "
              f"drive {r['drive']:+.3f}  v = {r['v']:.4f}  (R2 {r['r2']:.3f})")
    v_monotone = bool(np.all(np.diff(sw["v"]) > 0))
    print(f"  v increases monotonically with undercooling: {v_monotone} "
          f"(constant mobility -> no thermal-transport maximum)")

    # 3. critical radius at two undercoolings
    cd = run_critical_radius(dm, mesh, cons, CRIT_DEEP_T, CRIT_DEEP_R,
                             device=args.device)
    cm = run_critical_radius(dm, mesh, cons, CRIT_MILD_T, CRIT_MILD_R,
                             device=args.device)
    ratio_meas = cm["r_star"] / cd["r_star"]
    ratio_pred = abs(cd["drive"]) / abs(cm["drive"])   # r* ~ 1/|drive|
    print("\n[3] critical radius r* (sub/supercritical transition)")
    print(f"  T = {cd['T']:.0f} K (drive {cd['drive']:+.3f}): "
          f"r* in [{cd['r_lo']:.3f}, {cd['r_hi']:.3f}] ~ {cd['r_star']:.3f}")
    print(f"  T = {cm['T']:.0f} K (drive {cm['drive']:+.3f}): "
          f"r* in [{cm['r_lo']:.3f}, {cm['r_hi']:.3f}] ~ {cm['r_star']:.3f}")
    print(f"  measured r*(mild)/r*(deep) = {ratio_meas:.2f}; "
          f"Gibbs-Thomson r* ~ 1/|drive| predicts {ratio_pred:.2f} "
          f"(same sense: weaker drive -> larger r*)")

    # 4. Avrami / JMAK kinetics
    av = run_avrami(dm, mesh, cons, device=args.device)
    f = av["fit"]
    print("\n[4] Avrami / JMAK kinetics (ACCELERATED T = 250 K, L_psi = 11)")
    print(f"  {av['n_seeds']} nuclei -> X_end(psi>0.5) = {av['X_end']:.3f}, "
          f"<psi>_end = {av['Xq_end']:.3f}, {av['n_grains']} grains")
    print(f"  baseline-corrected Avrami exponent n = {f['n']:.2f} "
          f"+/- {f['ci95']:.2f} (95% CI), R2 = {f['r2']:.3f}, "
          f"X0 = {f['X0']:.3f}, window X* in [{f['lo']}, {f['hi']}], "
          f"{f['npts']} pts")
    print("  (finite pre-placed supercritical seeds -> n below the ideal "
          "2-D value 2)")

    results = {
        "grow": {"qpsi0": grow["qpsi0"], "qpsi1": grow["qpsi1"],
                 "qpsi_ratio": grow["qpsi1"] / max(grow["qpsi0"], 1e-9),
                 "area0": grow["a0"], "area1": grow["a1"],
                 "drive": grow["drive"]},
        "melt": {"qpsi0": melt["qpsi0"], "qpsi1": melt["qpsi1"],
                 "area1": melt["a1"], "drive": melt["drive"]},
        "interface_velocity": {
            "dT": sw["dT"].tolist(), "v": sw["v"].tolist(),
            "monotone_increasing": v_monotone},
        "critical_radius": {
            "deep": {"T": cd["T"], "drive": cd["drive"],
                     "r_star": cd["r_star"], "r_lo": cd["r_lo"],
                     "r_hi": cd["r_hi"]},
            "mild": {"T": cm["T"], "drive": cm["drive"],
                     "r_star": cm["r_star"], "r_lo": cm["r_lo"],
                     "r_hi": cm["r_hi"]},
            "ratio_measured": ratio_meas, "ratio_predicted": ratio_pred},
        "avrami": {"n": f["n"], "ci95": f["ci95"], "r2": f["r2"],
                   "X0": f["X0"], "X_end": av["X_end"],
                   "Xq_end": av["Xq_end"], "n_seeds": av["n_seeds"],
                   "n_grains": av["n_grains"], "npts": f["npts"]},
    }
    os.makedirs(args.output, exist_ok=True)
    rpath = os.path.join(args.output, "results.json")
    with open(rpath, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print(f"\nresults -> {rpath}")

    if not args.no_check:
        baseline = os.path.join(_HERE, "baseline.yaml")
        ok, rows = _check.check_files(rpath, baseline)
        print("\nbaseline check:\n" + _check.format_table(rows))
        if not ok:
            print("\nBASELINE CHECK FAILED")
            raise SystemExit(2)
        print("\nbaseline check PASSED")

    print("\nBelow Tm the crystal grows, above Tm it melts; v rises with "
          "undercooling; r* shrinks with undercooling; the Avrami exponent "
          "reports the growth dimensionality.  Compare with EXPECTED.md.")


if __name__ == "__main__":
    main()
