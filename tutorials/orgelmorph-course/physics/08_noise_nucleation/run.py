"""OrgElMorph course - Physics P8 driver (the file the student runs).

    python run.py                 # FDT verification + nucleation ensemble
    python run.py --level 6       # finer mesh (slower)

Runs, in order:
  1. the CENTRAL FDT verification -- fix the physical noise, put psi in a
     stable quadratic well, and show the equilibrium variance converges to
     a dt-INDEPENDENT plateau (the discrete FDT normalization is correct)
     and scales as 1/V_cell across meshes (equipartition);
  2. a noise-amplitude (kB*T calibration) ENSEMBLE sweep of an undercooled
     melt -- several noise realizations per amplitude, reported as the
     nucleation probability, the crystalline-fraction distribution, the
     nuclei density and the induction time (mean + 95% interval), plus the
     clipping fraction.

This is a NOISE-AMPLITUDE sweep at FIXED undercooling, NOT a temperature
sweep (a real T change would also move the driving force and the barrier).
Results -> outputs/results.json, checked against baseline.yaml.
"""
import argparse
import json
import os
import sys

import numpy as np

import nucleation as Nuc
from nucleation import build_mesh_dm, sweep, verify_fdt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from common import check_results as _check  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=os.path.join(_HERE, "outputs"))
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    print("\n=== FDT thermal noise and nucleation ===")

    # 1. central FDT verification (coarse CPU/splu well; fast)
    vf = verify_fdt()
    print("\n[1] FDT verification: equilibrium variance in a stable "
          "quadratic well")
    print("  dt-independence (fixed physical noise):")
    for r in vf["dt_runs"]:
        print(f"    dt = {r['dt']:.1e}   Var(psi) = {r['var']:.4e}")
    print(f"    -> dt-CoV of Var = {vf['dt_cov']:.3f}  (small => variance "
          f"converged => FDT normalization verified)")
    print("  mesh scaling (equipartition Var*V_cell ~ const):")
    for r in vf["mesh_runs"]:
        print(f"    level {r['level']}  V_cell = {r['cell_volume']:.3e}  "
              f"Var = {r['var']:.4e}  Var*V_cell = {r['var_times_vcell']:.4e}")
    print(f"    -> mesh-CoV of Var*V_cell = {vf['mesh_cov']:.3f}")

    # 2. nucleation ensemble sweep
    dm, mesh, cons = build_mesh_dm(args.level, device=args.device)
    s = sweep(dm, mesh, cons, device=args.device)
    print("\n[2] noise-amplitude ENSEMBLE sweep (undercooled melt, no "
          f"seed; {s['ensembles'][-1]['n']} seeds/amplitude)")
    hdr = (f"  {'noise':>7}{'P(nucl)':>9}{'X mean':>9}{'X sd':>7}"
           f"{'nuclei':>8}{'induct.':>9}{'clip':>7}")
    print(hdr)
    for e in s["ensembles"]:
        p = e["nucleation_prob"]["p"]
        ind = e["induction"]["estimate"]
        inds = "-" if (ind is None or (isinstance(ind, float)
                                       and np.isnan(ind))) else f"{ind:.3f}"
        print(f"  {e['noise_psi']:>7.3f}{p:>9.2f}{e['X_mean']:>9.3f}"
              f"{e['X_sd']:>7.3f}{e['nuclei_density_mean']:>8.1f}"
              f"{inds:>9}{e['sat_frac_mean']:>7.3f}")
    print("\nZero noise stays amorphous (metastable, no barrier crossing); "
          "the FDT amplitude IS the physical seed. The ensemble gives the "
          "nucleation probability and the induction/X spread with CIs.")

    results = {
        "fdt": {"dt_cov": vf["dt_cov"], "mesh_cov": vf["mesh_cov"],
                "var_mean": vf["var_mean"], "var": vf["var"],
                "var_times_vcell": vf["var_times_vcell"]},
        "sweep": {
            "levels": s["levels"],
            "prob": s["prob"],
            "X_mean": s["X_mean"],
            "X_sd": s["X_sd"],
            "density": s["density"],
            "zero_noise_X": s["ensembles"][0]["X_mean"],
            "zero_noise_prob": s["ensembles"][0]["nucleation_prob"]["p"],
            "hi_density": s["ensembles"][-1]["nuclei_density_mean"],
            "hi_prob": s["ensembles"][-1]["nucleation_prob"]["p"],
            "hi_sat_frac": s["ensembles"][-1]["sat_frac_mean"],
            "density_increasing": bool(
                s["density"][-1] > s["density"][1])},
    }
    os.makedirs(args.output, exist_ok=True)
    rpath = os.path.join(args.output, "results.json")
    with open(rpath, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print(f"\nresults -> {rpath}")

    if not args.no_check:
        ok, rows = _check.check_files(rpath, os.path.join(_HERE,
                                                          "baseline.yaml"))
        print("\nbaseline check:\n" + _check.format_table(rows))
        if not ok:
            print("\nBASELINE CHECK FAILED")
            raise SystemExit(2)
        print("\nbaseline check PASSED")


if __name__ == "__main__":
    main()
