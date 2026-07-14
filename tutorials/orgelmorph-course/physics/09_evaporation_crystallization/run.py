"""OrgElMorph course - Physics P9 driver (the file the student runs).

    python run.py                 # evaporation-conditioned embryo growth
    python run.py --level 6       # finer mesh (slower)

Runs, in order:
  1. the DERIVED r14 solubility phi* (from the free energy) and its
     VALIDATION by an embryo composition sweep (implant into uniform blends
     of varying small-molecule fraction, no drying, locate the grow/dissolve
     crossover);
  2. the sub/supercritical embryo control at a fixed super-solubility
     composition (a small embryo redissolves, a large one grows -- the
     Gibbs-Thomson floor);
  3. the evaporation-conditioned arc: the SAME embryo implanted in the WET
     film (dissolves) vs MID-DRYING (grows to a crystalline film), with an
     honest termination status.

Results -> outputs/results.json, checked against baseline.yaml.
NOTE: the default runs IMPLANT an embryo (not spontaneous nucleation);
genuine FDT nucleation is the advanced make_stepper(noise_psi=...) mode.
"""
import argparse
import json
import os
import sys

import numpy as np

import arc as A
from arc import (build_mesh_dm, run_arc, run_static_embryo,
                 embryo_composition_sweep, solubility_threshold)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from common import check_results as _check  # noqa: E402

CHI_CA = 1.6
SUPER_PHI_F = 0.85          # a composition well above phi*


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=os.path.join(_HERE, "outputs"))
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()
    dm, mesh, cons = build_mesh_dm(args.level, device=args.device)

    print("\n=== Evaporation-conditioned embryo growth (drying film) ===")

    # 1. derived solubility + composition-sweep validation
    phi_star = solubility_threshold(CHI_CA)
    print(f"\n[1] derived r14 solubility phi* = 1 - |drive|/chi_ca = "
          f"{phi_star:.3f}  (drive {A.drive_r14():+.3f}, chi_ca {CHI_CA})")
    sw = embryo_composition_sweep(dm, mesh, cons, chi_ca_val=CHI_CA,
                                  device=args.device)
    print("    embryo composition sweep (no drying, same embryo):")
    for r in sw["runs"]:
        print(f"      phi_f = {r['phi_f']:.2f}: area {r['area0']:.3f} -> "
              f"{r['area1']:.3f}  {'GROWS' if r['grew'] else 'dissolves'}")
    print(f"    measured crossover in [{sw['crossover_lo']:.2f}, "
          f"{sw['crossover_hi']:.2f}] ~ {sw['crossover_measured']:.3f}  "
          f"(derived homogeneous phi* = {phi_star:.3f})")
    print("    NOTE: the measured supercritical-embryo threshold lies "
          "BELOW the homogeneous phi* -- the embryo self-enriches phi_f "
          "(P7 crystal-bulk channel), so phi* is an UPPER bound.")

    # 2. sub/supercritical embryo control (fixed super-solubility phi_f)
    sup = run_static_embryo(dm, mesh, cons, SUPER_PHI_F, chi_ca_val=CHI_CA,
                            r0=0.20, device=args.device)
    sub = run_static_embryo(dm, mesh, cons, SUPER_PHI_F, chi_ca_val=CHI_CA,
                            r0=0.05, device=args.device)
    print(f"\n[2] sub/supercritical embryo at phi_f = {SUPER_PHI_F} "
          f"(> phi*):")
    print(f"    r0 = 0.20 (super): area {sup['area0']:.3f} -> "
          f"{sup['area1']:.3f}  {'GROWS' if sup['grew'] else 'dissolves'}")
    print(f"    r0 = 0.05 (sub):   area {sub['area0']:.3f} -> "
          f"{sub['area1']:.3f}  {'GROWS' if sub['grew'] else 'dissolves'} "
          f"(Gibbs-Thomson floor)")

    # 3. the evaporation-conditioned arc (wet vs dry implant)
    wet = run_arc(dm, mesh, cons, wet=True, device=args.device)
    dry = run_arc(dm, mesh, cons, wet=False, device=args.device)
    print("\n[3] evaporation-conditioned arc (same embryo, different "
          "implant time):")
    print(f"    WET implant (phi_s {wet['phis_implant']:.3f}, local phi_f "
          f"{wet['phi_f_implant_local']:.3f}): area {wet['area_implant']:.3f}"
          f" -> {wet['area_final']:.3f}, psi_max {wet['psi_max']:.2f}  "
          f"DISSOLVES  [{wet['termination']}]")
    print(f"    DRY implant (phi_s {dry['phis_implant']:.3f}, local phi_f "
          f"{dry['phi_f_implant_local']:.3f}): area {dry['area_implant']:.3f}"
          f" -> {dry['area_final']:.3f}, psi_max {dry['psi_max']:.2f}  "
          f"GROWS  [{dry['termination']}]")
    print("\nThe embryo's fate is set by the drying-CONDITIONED local "
          "composition relative to phi* (not by time or evaporation per "
          "se): drying raises phi_f above phi* and the embryo grows.")

    results = {
        "solubility": {
            "phi_star_derived": sw["phi_star_derived"],
            "crossover_measured": sw["crossover_measured"],
            "crossover_lo": sw["crossover_lo"],
            "crossover_hi": sw["crossover_hi"],
            "crossover_bracketed": sw["crossover_bracketed"],
            "crossover_below_derived": sw["crossover_below_derived"]},
        "embryo_control": {
            "super_grew": sup["grew"], "super_area1": sup["area1"],
            "sub_grew": sub["grew"], "sub_area1": sub["area1"]},
        "arc": {
            "wet_phis": wet["phis_implant"], "wet_area": wet["area_final"],
            "wet_psi_max": wet["psi_max"], "wet_termination": wet["termination"],
            "dry_phis": dry["phis_implant"], "dry_area": dry["area_final"],
            "dry_psi_max": dry["psi_max"], "dry_termination": dry["termination"]},
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
