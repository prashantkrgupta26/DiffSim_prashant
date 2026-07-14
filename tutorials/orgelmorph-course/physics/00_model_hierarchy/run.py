"""OrgElMorph course - Physics P0 driver (harness-based).

The model-hierarchy chapter is written, but the ONE worked nondimensionalization
it walks through is computed here (with provenance + a tolerance baseline), not
hand-copied.  The Cahn number and the interface-resolution figures reproduce the
P1 tutorial's measured interface-cell counts exactly (level 6, kappa=5e-4 ->
2.02 poly / 1.43 FH cells) -- a cross-tutorial consistency check.

    PYTHONPATH=<repo>/src python run.py --config configs/p0.yaml \\
        --device cpu --mode reference --output outputs/p0 --overwrite
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                os.pardir, os.pardir)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import config as cfgmod                          # noqa: E402
from common.run_base import build_parser, run_tutorial        # noqa: E402

import nondim                                                 # noqa: E402

SCHEMA = cfgmod.ConfigSchema(name="p0", fields={
    # dimensional inputs (nm; the CH nondimensionalization)
    "interface_length_nm": cfgmod.Field(float, default=11.18, min=0.0),
    "domain_size_nm": cfgmod.Field(float, default=500.0, min=0.0),
    "level": cfgmod.Field(int, default=6, min=1, max=10),
    "target_interface_cells": cfgmod.Field(float, default=3.0, min=1.0),
    # Flory-Huggins (nondimensional) at the symmetric point
    "fh_A": cfgmod.Field(float, default=1.0, min=0.0),
    "fh_B": cfgmod.Field(float, default=2.5, min=0.0),
    "fh_c_bar": cfgmod.Field(float, default=0.5, min=0.0, max=1.0),
    # crystallization (P6 PCBM class, materials.yaml): accelerated vs physical
    "cryst_Tm_K": cfgmod.Field(float, default=558.0, min=0.0),
    "cryst_dh": cfgmod.Field(float, default=1.3072, min=0.0),
    "cryst_T_accel_K": cfgmod.Field(float, default=250.0, min=0.0),
    "cryst_T_phys_K": cfgmod.Field(float, default=500.0, min=0.0),
    "precision": cfgmod.Field(str, default="fp64"),
})


def p0_run(cfg, ctx):
    ctx.provenance.update(
        computation="nondimensionalization + resolution calculator (analytic)",
        model="binary Cahn-Hilliard nondimensionalization; P6 crystallization drive")
    r = nondim.compute(cfg)
    ctx.log(f"Cahn number kappa~ = (lam/L)^2 = ({r['lam_over_L']:.4f})^2 "
            f"= {r['cahn_number']:.3e}")
    ctx.log(f"  poly interface ell~ = sqrt(2 k~) = {r['poly_interface_ell_tilde']:.4f}"
            f"  -> {r['poly_interface_cells']:.2f} cells at level {r['level']}")
    ctx.log(f"  FH   f''({cfg['fh_c_bar']}) = {r['fh_curvature_fpp']:.3f} "
            f"(spinodal-unstable={r['spinodal_unstable']}); "
            f"ell~ = {r['fh_spinodal_ell_tilde']:.4f} -> {r['fh_interface_cells']:.2f} cells")
    ctx.log(f"  min level for >= {r['target_interface_cells']:.0f} cells across "
            f"interface: level {r['min_level_for_target']}")
    ctx.log(f"crystallization drive: accelerated (T={cfg['cryst_T_accel_K']}K) "
            f"{r['cryst_drive_accel']:.3f} vs physical (T={cfg['cryst_T_phys_K']}K) "
            f"{r['cryst_drive_phys']:.3f}  (ratio {r['cryst_drive_ratio']:.2f}x)")

    # figure data: interface cells vs mesh level (the resolution curve)
    levels = np.arange(3, 10)
    poly_cells = np.array([nondim.poly_interface_cells(r["cahn_number"], int(l))
                           for l in levels])
    ctx.history["levels"] = levels
    ctx.history["poly_cells_vs_level"] = poly_cells
    ctx.history["cahn_number"] = np.array([r["cahn_number"]])
    return r


def main():
    args = build_parser("OrgElMorph P0 (model hierarchy + nondimensionalization)").parse_args()
    here = os.path.dirname(__file__)
    baseline = (os.path.join(here, "baseline.yaml")
                if args.mode == "reference" else None)
    run_tutorial(p0_run, schema=SCHEMA, args=args,
                 default_output=os.path.join(here, "outputs", "p0"),
                 baseline=baseline, default_solver="splu")


if __name__ == "__main__":
    main()
