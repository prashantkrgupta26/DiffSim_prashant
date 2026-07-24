"""NS-SBM course — Chapter 05 driver: three dimensions (the SBM sphere).

An immersed 3-D sphere carved from a unit-box octree channel; the SBM shift is
genuinely active in 3-D. Marches the MONOLITHIC drag path (the working, stable,
physical 3-D drag: Cd = +0.381) and optionally checks the projection+SBM
composition invariants over a short window (finite, BDF2 engaged, axisymmetric).

    python run.py                                    # monolithic drag, level 4, Re=100
    python run.py --config configs/sphere.yaml --mode reference --output outputs/sp
    python run.py --pipeline                          # + projection composition invariants

Prints the self-check table (compare against EXPECTED.md) and runs the gate.
Solver path: host splu here (small); to scale, build on CUDA and route the
monolithic solve through solver="cudss" or "blockamgx".
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                os.pardir)))
from common import config as cfgmod          # noqa: E402
from common.run_base import build_parser, run_tutorial   # noqa: E402

from sphere import run_sphere                 # noqa: E402

SCHEMA = cfgmod.ConfigSchema(name="sphere", fields={
    "level": cfgmod.Field(int, default=4, min=3, max=5),
    "Re": cfgmod.Field(int, default=100, min=1),
    "alpha": cfgmod.Field(float, default=10.0, min=0.0),
    "dt": cfgmod.Field(float, default=0.05, min=0.0),
    "max_steps": cfgmod.Field(int, default=60, min=1),
    "rate_tol": cfgmod.Field(float, default=5e-3),
    "pipeline": cfgmod.Field(bool, default=False),
    "device": cfgmod.Field(str, default="cpu"),
    "precision": cfgmod.Field(str, default="fp64"),
})


def sphere_run(cfg, ctx):
    device = ctx.device or cfg.get("device")
    ctx.log(f"3-D immersed sphere (SBM-NS): Re={cfg['Re']} level={cfg['level']} "
            f"alpha={cfg['alpha']} dt={cfg['dt']} on {device}")
    r = run_sphere(level=cfg["level"], Re=cfg["Re"], alpha=cfg["alpha"],
                   device=device, dt=cfg["dt"], max_steps=cfg["max_steps"],
                   rate_tol=cfg["rate_tol"], pipeline=cfg["pipeline"])
    ctx.provenance.update(
        mesh={"level": cfg["level"], "dim": 3, "p": 1, "ndof": 4,
              "n_free": r["n_free"]}, Re=cfg["Re"], alpha=cfg["alpha"],
        n_surrogate_faces=r["n_faces"], dmax=r["dmax"],
        D_over_h=r["D_over_h"], time_integrator="BDF (bootstrap->BDF2)")
    ctx.log(f"  n_free={r['n_free']}  sphere surrogate faces={r['n_faces']}  "
            f"D/h={r['D_over_h']:.2f}  SBM shift dmax={r['dmax']:.5f} "
            f"(d != 0 => genuine 3-D shift)")
    ctx.log(f"  MONOLITHIC steady Cd = {r['cd_mono']:+.4f}  steps={r['steps']}  "
            f"(positive, physical — the working 3-D drag)")
    if "pipeline" in r:
        p = r["pipeline"]
        ctx.log(f"  [pipeline] projection composition: finite={p['finite']}  "
                f"BDF2 engaged={p['bdf2']}  axisymmetric={p['axisym']}  "
                f"|C_lat|/|Cd|={p['clat_ratio']:.2e}")
        # surface the pipeline invariants for the baseline gate
        r["pipe_finite"] = p["finite"]
        r["pipe_bdf2"] = p["bdf2"]
        r["pipe_axisym"] = p["axisym"]
        r["pipe_clat_ratio"] = p["clat_ratio"]
    return r


def main():
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--pipeline", action="store_true",
                        help="additionally check the projection+SBM composition "
                             "invariants over a short window")
    args = build_parser("NS-SBM 05 three dimensions (SBM sphere)",
                        parents=[parent]).parse_args()
    here = os.path.dirname(__file__)
    run_tutorial(sphere_run, schema=SCHEMA, args=args,
                 default_output=os.path.join(here, "outputs", "sp"),
                 baseline=os.path.join(here, "baseline.yaml"),
                 cli_overrides={"pipeline": args.pipeline},
                 default_solver="splu")


if __name__ == "__main__":
    main()
