"""P1 — the harness-based entry point (the Phase-0 reference port).

Same physics as ``run.py`` (the student-facing driver), but driven through the
course's standard harness so it is a *repeatable workflow*: a YAML config
(``configs/p1.yaml``, the canonical record), a provenance ``metadata.json``, a
``results.json`` checked against ``baseline.yaml``, and the standard output
layout. It reproduces EXPECTED.md's numbers — the regression that guards the
Phase-0 foundation. ``run.py`` is intentionally left untouched (that per-chapter
port is a later phase); this file shows tutorials how to adopt the harness.

    python run_harness.py --config configs/p1.yaml --mode reference \\
        --output outputs/p1 --overwrite

Reuses the diagnostics library for the energy budget so the numbers come from
the shared, unit-tested metrics, not a bespoke re-computation.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                os.pardir, os.pardir)))
from common import config as cfgmod                      # noqa: E402
from common.run_base import build_parser, run_tutorial    # noqa: E402
from diffsim.diagnostics import energy as dgen            # noqa: E402
from diffsim.diagnostics import conservation as dcons     # noqa: E402

from spinodal import run_spinodal                          # noqa: E402

SCHEMA = cfgmod.ConfigSchema(name="p1", fields={
    "energy": cfgmod.Field(str, default="both",
                           choices=("poly", "fh", "both")),
    "level": cfgmod.Field(int, default=6, min=2, max=9),
    "steps": cfgmod.Field(int, default=250, min=1),
    "dt": cfgmod.Field(float, default=0.02, min=0.0),
    "M": cfgmod.Field(float, default=1.0, min=0.0),
    "kappa": cfgmod.Field(float, default=5e-4, min=0.0),
    "fh_A": cfgmod.Field(float, default=1.0, min=0.0),
    "fh_B": cfgmod.Field(float, default=2.5, min=0.0),
    "amp": cfgmod.Field(float, default=0.05, min=0.0),
    "seed": cfgmod.Field(int, default=3),
    "precision": cfgmod.Field(str, default="fp64"),
})


def _summarize(rec):
    """Scalar summary from a run_spinodal record, using the shared diagnostics
    library for the energy-budget and mass-conservation numbers."""
    F = rec["F_total"]
    Fint = rec["F_interface"]
    snap = rec["snaps"][max(rec["snaps"])]
    return {
        "F0": float(F[0]),
        "F1": float(F[1]),
        "Fend": float(F[-1]),
        "Fbulk_end": float(rec["F_bulk"][-1]),
        "Fint_end": float(Fint[-1]),
        "Fint_peak": float(np.max(Fint)),
        "c_min": float(snap.min()),
        "c_max": float(snap.max()),
        # mass drift via the shared conservation diagnostic
        "mass_drift": dcons.mass_drift(rec["mass"]),
        # discrete monotonicity via the shared energy diagnostic
        "monotone_after_step1": dgen.is_monotone_decreasing(F, skip=1),
        "largest_positive_increment": dgen.largest_positive_increment(F[1:]),
    }


def p1_run(cfg, ctx):
    which = ["poly", "fh"] if cfg["energy"] == "both" else [cfg["energy"]]
    ctx.provenance.update(
        mesh={"level": cfg["level"], "side": 2 ** cfg["level"] + 1,
              "dim": 2, "p": 1},
        time_integrator="BDF1",
        linear_tol=1e-10,
        nonlinear_tol=1e-8)
    results = {}
    for e in which:
        ctx.log(f"P1 {e}: level {cfg['level']} ({2**cfg['level']}^2), "
                f"{cfg['steps']} steps, solver={ctx.solver}")
        rec = run_spinodal(energy=e, level=cfg["level"], steps=cfg["steps"],
                           dt=cfg["dt"], M=cfg["M"], kappa=cfg["kappa"],
                           fh_A=cfg["fh_A"], fh_B=cfg["fh_B"], amp=cfg["amp"],
                           seed=cfg["seed"], device=ctx.device,
                           linsolver=ctx.solver)
        s = _summarize(rec)
        results[e] = s
        ctx.log(f"  F: {s['F0']:.5g} -> {s['Fend']:.5g}; "
                f"interface peak {s['Fint_peak']:.5g} -> {s['Fint_end']:.5g}; "
                f"c in [{s['c_min']:.3f}, {s['c_max']:.3f}]; "
                f"|dm|={s['mass_drift']:.2e}; monotone="
                f"{s['monotone_after_step1']}")
        # store time series for figures
        for k in ("t", "F_total", "F_bulk", "F_interface", "mass"):
            ctx.history[f"{e}_{k}"] = np.asarray(rec[k])
    return results


def main():
    args = build_parser("OrgElMorph P1 (harness port)").parse_args()
    here = os.path.dirname(__file__)
    # baseline.yaml encodes the reference-mode EXPECTED numbers; the
    # quick/research tiers use different meshes/horizons, so only gate on it
    # in reference mode (the regression that guards the Phase-0 port).
    baseline = (os.path.join(here, "baseline.yaml")
                if args.mode == "reference" else None)
    # This chapter's (c, mu) system is small and INDEFINITE: scipy SuperLU's
    # partial pivoting is exact here, whereas cuDSS (no pivoting) diverges on
    # this tiny saddle-point block. splu is therefore P1's documented `auto`
    # solver and the one the EXPECTED numbers were generated with. cuDSS is the
    # measured default at scale (C4 / research runs), reachable via
    # --solver cudss.
    run_tutorial(p1_run, schema=SCHEMA, args=args,
                 default_output=os.path.join(here, "outputs", "p1"),
                 baseline=baseline, default_solver="splu")


if __name__ == "__main__":
    main()
