"""NS-SBM course — Chapter 02 driver: the pressure-projection engine.

The scalable engine: predictor -> SPD pressure-Poisson (PPE) -> L2 velocity
correction. This chapter re-runs the SAME lid-driven cavity as Chapter 01
through the projection engine, and checks that it reproduces the same-mesh
monolithic oracle (faithfulness) and Ghia. The narrative payoff is that the
pressure step is a symmetric-positive-definite Poisson problem — the operator
that admits AMG/CG at 100M DOF (Chapter 06), unlike the indefinite saddle.

    python run.py                                    # level 4, Re=100, both engines
    python run.py --config configs/ldc.yaml --mode reference --output outputs/ldc

Prints the centerline self-check table (compare against EXPECTED.md) and runs
the tolerance gate against baseline.yaml. The core (`cavity.py`) is shared with
Chapter 01, so the two engines are compared on identical fixtures.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                os.pardir)))
from common import config as cfgmod          # noqa: E402
from common.run_base import build_parser, run_tutorial   # noqa: E402

from cavity import run_cavity, GHIA_Y         # noqa: E402

SCHEMA = cfgmod.ConfigSchema(name="ldc", fields={
    "level": cfgmod.Field(int, default=4, min=2, max=7),
    "Re": cfgmod.Field(int, default=100, choices=(100, 400)),
    "dt": cfgmod.Field(float, default=0.05, min=0.0),
    "nsteps": cfgmod.Field(int, default=200, min=1),
    "device": cfgmod.Field(str, default="cpu"),
    "precision": cfgmod.Field(str, default="fp64"),
})


def ldc_run(cfg, ctx):
    device = ctx.device or cfg.get("device")
    level, Re, dt, nsteps = cfg["level"], cfg["Re"], cfg["dt"], cfg["nsteps"]
    ctx.log(f"lid-driven cavity (projection focus): {2**level}x{2**level}, "
            f"Re={Re}, {nsteps} steps, on {device}")
    ctx.provenance.update(
        mesh={"level": level, "side": 2 ** level, "dim": 2, "p": 1,
              "ndof": 3}, Re=Re, time_integrator="BDF1",
        ppe="SPD Poisson (AMG/CG-friendly)")
    r = run_cavity(level=level, Re=Re, dt=dt, nsteps=nsteps, device=device,
                   engines=("monolithic", "projection"))

    ctx.log(f"  projection ||div u|| = {r['div_proj']:.3e}  (weak-div VMS; "
            f"the PPE-space divergence is the right gate)")
    ctx.log(f"  monolithic ||div u|| = {r['div_mono']:.3e}")
    ctx.log("  centerline u(y) @ x=0.5:  proj / mono / Ghia")
    for i, y in enumerate(GHIA_Y):
        ctx.log(f"    y={y:.4f}  {r['u_proj'][i]:+.4f} / "
                f"{r['u_mono'][i]:+.4f} / {r['ghia'][i]:+.4f}")
    ctx.log(f"  max|proj - mono| = {r['d_proj_mono']:.4f}  (same-mesh faithfulness)")
    ctx.log(f"  max|proj - Ghia| = {r['d_proj_ghia']:.4f}  (coarse mesh vs 129^2)")
    ctx.log(f"  max|mono - Ghia| = {r['d_mono_ghia']:.4f}")
    ctx.history = {"ghia_y": GHIA_Y, "u_mono": r["u_mono"],
                   "u_proj": r["u_proj"], "ghia": r["ghia"]}
    r["u_mono_lid"] = r["u_mono"][1]
    r["u_proj_lid"] = r["u_proj"][1]
    r["ghia_lid"] = r["ghia"][1]
    return r


def main():
    args = build_parser("NS-SBM 02 pressure projection (lid-driven cavity)").parse_args()
    here = os.path.dirname(__file__)
    run_tutorial(ldc_run, schema=SCHEMA, args=args,
                 default_output=os.path.join(here, "outputs", "ldc"),
                 baseline=os.path.join(here, "baseline.yaml"),
                 default_solver="splu")


if __name__ == "__main__":
    main()
