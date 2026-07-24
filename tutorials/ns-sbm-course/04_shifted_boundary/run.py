"""NS-SBM course — Chapter 04 driver: the Shifted Boundary Method (d != 0).

A square whose center is offset by a sub-cell fraction (offset=0.05), so the
grid-aligned surrogate no longer coincides with the true Box face: 0 < dmax < h.
The Taylor (grad N).d term and the area correction do REAL work. Marches the
consistent-projection split vs the same-mesh-with-shift monolithic oracle and
compares Cd — projection + SBM end-to-end in 2-D.

    python run.py                                    # level 4, Re=40, offset=0.05
    python run.py --config configs/shift.yaml --mode reference --output outputs/sh
    python run.py --zero-shift                        # anti-vacuity: break the match

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

from shifted import run_shifted, ALPHA        # noqa: E402

SCHEMA = cfgmod.ConfigSchema(name="shift", fields={
    "level": cfgmod.Field(int, default=4, min=3, max=6),
    "Re": cfgmod.Field(int, default=40, min=1),
    "offset": cfgmod.Field(float, default=0.05, min=0.0, max=0.06),
    "alpha": cfgmod.Field(float, default=float(ALPHA), min=0.0),
    "dt": cfgmod.Field(float, default=0.02, min=0.0),
    "nsteps": cfgmod.Field(int, default=600, min=1),
    "device": cfgmod.Field(str, default="cpu"),
    "precision": cfgmod.Field(str, default="fp64"),
    # anti-vacuity toggle (set by --zero-shift); default off
    "_zero_shift": cfgmod.Field(bool, default=False),
})


def shift_run(cfg, ctx):
    device = ctx.device or cfg.get("device")
    do_zero = bool(cfg.get("_zero_shift", False))
    ctx.log(f"flow past a shifted body (SBM d!=0): Re={cfg['Re']} "
            f"level={cfg['level']} offset={cfg['offset']} "
            f"{'(SHIFT ZEROED)' if do_zero else ''} on {device}")
    r = run_shifted(level=cfg["level"], Re=cfg["Re"], offset=cfg["offset"],
                    do_zero_shift=do_zero, device=device, dt=cfg["dt"],
                    nsteps=cfg["nsteps"], alpha=cfg["alpha"])
    ctx.provenance.update(
        mesh={"level": cfg["level"], "dim": 2, "p": 1, "ndof": 3},
        Re=cfg["Re"], offset=cfg["offset"], dmax=r["dmax"],
        dmax_over_h=r["dmax_over_h"], zero_shift=do_zero,
        time_integrator="BDF (bootstrap->BDF2)")
    ctx.log(f"  dmax={r['dmax']:.5f}  (0<dmax<h={r['h']:.5f}, "
            f"dmax/h={r['dmax_over_h']:.3f})  area-corrected GPs="
            f"{r['area_corrected_gps']}")
    tag = " (SHIFT ZEROED — expect a broken match)" if do_zero else ""
    ctx.log(f"  projection{tag}: Cd={r['cd_proj']:+.4f}  "
            f"mean|u|={r['mean_u_proj']:.4f}  steps={r['steps_proj']}")
    ctx.log(f"  monolithic (true shift): Cd={r['cd_mono']:+.4f}  "
            f"mean|u|={r['mean_u_mono']:.4f}  steps={r['steps_mono']}")
    ctx.log(f"  Cd rel-diff vs TRUE shifted oracle = {r['cd_rel']:.3%}")
    ctx.log(f"  mean|u| rel-diff                    = {r['mu_rel']:.3%}")
    return r


def main():
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--zero-shift", action="store_true",
                        help="anti-vacuity: zero geo.d/geo.corr in the "
                             "projection and break the match")
    args = build_parser("NS-SBM 04 shifted boundary (SBM d != 0)",
                        parents=[parent]).parse_args()
    here = os.path.dirname(__file__)
    run_tutorial(shift_run, schema=SCHEMA, args=args,
                 default_output=os.path.join(here, "outputs", "sh"),
                 baseline=os.path.join(here, "baseline.yaml"),
                 cli_overrides={"_zero_shift": args.zero_shift},
                 default_solver="splu")


if __name__ == "__main__":
    main()
