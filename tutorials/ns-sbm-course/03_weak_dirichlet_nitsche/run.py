"""NS-SBM course — Chapter 03 driver: weak Dirichlet (Nitsche), flow past a square.

A cell-aligned square (d=0) in a channel: the SBM shift is OFF, so this is
standard Nitsche. Marches the projection engine (consistent_projection) and the
same-mesh monolithic oracle with the chosen no-slip mode (weak/strong), and
compares drag Cd and mean|u| — the faithfulness check with an immersed body.

    python run.py                                    # weak Nitsche, level 4, Re=40
    python run.py --config configs/square.yaml --mode reference --output outputs/sq
    python run.py --mode-noslip strong               # row-replacement no-slip

Prints the self-check table (compare against EXPECTED.md) and runs the gate.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                os.pardir)))
from common import config as cfgmod          # noqa: E402
from common.run_base import build_parser, run_tutorial   # noqa: E402

from square import run_square, ALPHA          # noqa: E402

SCHEMA = cfgmod.ConfigSchema(name="square", fields={
    "level": cfgmod.Field(int, default=4, min=3, max=6),
    "Re": cfgmod.Field(int, default=40, min=1),
    "noslip": cfgmod.Field(str, default="weak", choices=("weak", "strong")),
    "alpha": cfgmod.Field(float, default=float(ALPHA), min=0.0),
    "dt": cfgmod.Field(float, default=0.02, min=0.0),
    "nsteps": cfgmod.Field(int, default=600, min=1),
    "device": cfgmod.Field(str, default="cpu"),
    "precision": cfgmod.Field(str, default="fp64"),
})


def square_run(cfg, ctx):
    device = ctx.device or cfg.get("device")
    mode = cfg["noslip"]
    ctx.log(f"flow past a square (Nitsche, d=0): mode={mode}  Re={cfg['Re']}  "
            f"level={cfg['level']}  alpha={cfg['alpha']}  on {device}")
    r = run_square(mode=mode, level=cfg["level"], Re=cfg["Re"],
                   alpha=cfg["alpha"], device=device, dt=cfg["dt"],
                   nsteps=cfg["nsteps"])
    ctx.provenance.update(
        mesh={"level": cfg["level"], "dim": 2, "p": 1, "ndof": 3},
        Re=cfg["Re"], D=r["D"], dmax=r["dmax"], noslip=mode,
        alpha=cfg["alpha"], time_integrator="BDF (bootstrap->BDF2)")
    ctx.log(f"  fluid cells={r['n_fluid_cells']}  obstacle nodes={r['obstacle_nodes']}"
            f"  (dmax={r['dmax']} => standard Nitsche)")
    ctx.log(f"  projection : Cd={r['cd_proj']:+.4f}  mean|u|={r['mean_u_proj']:.4f}"
            f"  ||div||={r['div_proj']:.3e}  steps={r['steps_proj']}"
            + ("  BLEW UP" if r["blew_up"] else ""))
    ctx.log(f"  monolithic : Cd={r['cd_mono']:+.4f}  mean|u|={r['mean_u_mono']:.4f}"
            f"  ||div||={r['div_mono']:.3e}  steps={r['steps_mono']}")
    ctx.log(f"  Cd rel-diff      = {r['cd_rel']:.3%}   (bar: match same-mesh mono)")
    ctx.log(f"  mean|u| rel-diff = {r['mu_rel']:.3%}")
    return r


def main():
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--mode-noslip", dest="noslip",
                        choices=["weak", "strong"], default=None,
                        help="no-slip imposition (weak Nitsche / strong rows)")
    parent.add_argument("--alpha", type=float, default=None,
                        help="Nitsche penalty scale (0 removes the penalty)")
    args = build_parser("NS-SBM 03 weak Dirichlet / Nitsche (flow past a square)",
                        parents=[parent]).parse_args()
    overrides = {}
    if args.noslip is not None:
        overrides["noslip"] = args.noslip
    if args.alpha is not None:
        overrides["alpha"] = args.alpha
    here = os.path.dirname(__file__)
    run_tutorial(square_run, schema=SCHEMA, args=args,
                 default_output=os.path.join(here, "outputs", "sq"),
                 baseline=os.path.join(here, "baseline.yaml"),
                 cli_overrides=overrides, default_solver="splu")


if __name__ == "__main__":
    main()
