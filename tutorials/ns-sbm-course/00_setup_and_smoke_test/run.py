"""Chapter 00 smoke test — the first run that exercises the whole workflow.

Marches a tiny lid-driven cavity through BOTH NS engines via the standard
course harness (config -> provenance -> results.json -> tolerance check), so a
new student confirms in one command that (a) the device toolchain works and
(b) the course's config/provenance/checker plumbing works:

    python run.py --config configs/smoke_cpu.yaml --output outputs/smoke --device cpu

It writes the standard outputs/<run>/ layout and checks the results against
baseline.yaml. Pair it with ``doctor.py`` (env probe). See EXPECTED.md.
"""
from __future__ import annotations

import os
import sys

# make the course common/ importable (course root is one level up)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                os.pardir)))
from common import config as cfgmod          # noqa: E402
from common.run_base import build_parser, run_tutorial   # noqa: E402

from ns_smoke import smoke                    # noqa: E402

SCHEMA = cfgmod.ConfigSchema(name="smoke", fields={
    "level": cfgmod.Field(int, default=3, min=2, max=6),
    "Re": cfgmod.Field(int, default=100, min=1),
    "dt": cfgmod.Field(float, default=0.05, min=0.0),
    "nsteps": cfgmod.Field(int, default=20, min=1),
    "device": cfgmod.Field(str, default="cpu"),
    "precision": cfgmod.Field(str, default="fp64"),
})


def smoke_run(cfg, ctx):
    device = ctx.device or cfg.get("device")
    level, Re, dt, nsteps = cfg["level"], cfg["Re"], cfg["dt"], cfg["nsteps"]
    ctx.log(f"smoke: {2**level}x{2**level} lid-driven cavity, Re={Re}, "
            f"{nsteps} steps, both engines, on {device}")
    ctx.provenance.update(
        mesh={"level": level, "side": 2 ** level, "dim": 2, "p": 1,
              "ndof": 3}, Re=Re, time_integrator="BDF1")
    r = smoke(level=level, Re=Re, dt=dt, nsteps=nsteps, device=device)
    ctx.log(f"  monolithic: |u|max={r['umax_mono']:.3f}  "
            f"||div||={r['div_mono']:.2e}")
    ctx.log(f"  projection: |u|max={r['umax_proj']:.3f}  "
            f"||div||={r['div_proj']:.2e}")
    ctx.log(f"  max|proj-mono| = {r['max_proj_minus_mono']:.3f}  finite="
            f"{r['all_finite']}")
    return r


def main():
    args = build_parser("NS-SBM 00 smoke test").parse_args()
    here = os.path.dirname(__file__)
    # The monolithic saddle is indefinite; on the small tutorial mesh scipy
    # SuperLU (splu, host direct, exact pivoting) is the documented auto choice.
    run_tutorial(smoke_run, schema=SCHEMA, args=args,
                 default_output=os.path.join(here, "outputs", "smoke"),
                 baseline=os.path.join(here, "baseline.yaml"),
                 default_solver="splu")


if __name__ == "__main__":
    main()
