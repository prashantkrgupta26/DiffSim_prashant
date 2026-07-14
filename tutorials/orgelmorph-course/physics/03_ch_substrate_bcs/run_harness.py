"""P3 — the harness-based entry point (substrate / surface energy).

Same physics as ``run.py`` (the student driver), driven through the
course's standard harness so it is a *repeatable workflow*: a YAML config
(``configs/p3.yaml``, the canonical record), a provenance ``metadata.json``,
a ``results.json`` checked against ``baseline.yaml``, and the standard
output layout.

    PYTHONPATH=<repo>/src python run_harness.py --config configs/p3.yaml \\
        --mode reference --output outputs/p3 --overwrite

What it measures (spec P3, all by QUADRATURE):
  * 6 boundary-condition cases (neutral, attracting, repelling, opposing
    walls, confined-lateral, demixing+wetting): substrate phi, film-mean
    phi, enrichment, boundary-layer thickness, TRUE quadrature mass drift
    (INT phi dV), projected_dofs/max_correction, and the full energy
    budget F = F_bulk + F_grad + F_wall;
  * a boundary-layer thickness study delta ~ sqrt(kappa) across >=3 kappa;
  * an honest CLIPPING demonstration: a preference pushed to the wall so
    the box projection fires and injects mass (contrast with the
    machine-eps drift of the interior-phi* cases).

Reuses the shared diffsim.diagnostics library (conservation / energy /
admissibility) for every number.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                os.pardir, os.pardir)))
from common import config as cfgmod                       # noqa: E402
from common.run_base import build_parser, run_tutorial     # noqa: E402

from substrate import (build_mesh_dm, WallDiagnostics,      # noqa: E402
                       simulate)

SCHEMA = cfgmod.ConfigSchema(name="p3", fields={
    "level": cfgmod.Field(int, default=6, min=2, max=8),
    "chi": cfgmod.Field(float, default=2.2, min=0.0),
    "chi_demix": cfgmod.Field(float, default=2.7, min=0.0),
    "kappa": cfgmod.Field(float, default=1e-3, min=0.0),
    "onsager": cfgmod.Field(float, default=0.2, min=0.0),
    "phi0": cfgmod.Field(float, default=0.5, min=0.0, max=1.0),
    "amp": cfgmod.Field(float, default=0.02, min=0.0),
    "dt": cfgmod.Field(float, default=2e-4, min=0.0),
    "dt_max": cfgmod.Field(float, default=2e-2, min=0.0),
    "t_end": cfgmod.Field(float, default=0.4, min=0.0),
    "seed": cfgmod.Field(int, default=5),
    "chi_bl": cfgmod.Field(float, default=1.0, min=0.0),
    "kappa_sweep": cfgmod.Field(list, default=[2e-3, 4e-3, 8e-3, 1.6e-2]),
    "clip_amp": cfgmod.Field(float, default=0.30, min=0.0),
    "clip_t_end": cfgmod.Field(float, default=0.1, min=0.0),
    "precision": cfgmod.Field(str, default="fp64"),
})

# The 6 boundary-condition cases (spec P3).  g, h are the wall-energy
# coefficients f_w = g phi + h phi^2 (phi* = -g/(2h)); g2, h2 an OPPOSING
# air-side wall; lateral is the x boundary (periodic film vs confined box);
# chi_key selects the base or demixing interaction.
CASES = [
    dict(key="neutral", label="neutral (no-flux)",
         g=0.0, h=0.0, lateral="periodic"),
    dict(key="attract", label="attracting substrate (phi*=0.75)",
         g=-1.5, h=1.0, lateral="periodic"),
    dict(key="repel", label="repelling substrate (phi*=0.25)",
         g=-0.5, h=1.0, lateral="periodic"),
    dict(key="opposing", label="opposing walls (attract sub / repel air)",
         g=-1.5, h=1.0, g2=-0.5, h2=1.0, lateral="periodic"),
    dict(key="confined", label="attracting, confined lateral (no-flux x)",
         g=-1.5, h=1.0, lateral="confined"),
    dict(key="demix", label="demixing + wetting (chi>spinodal)",
         g=-1.5, h=1.0, lateral="periodic", chi_key="demix"),
]

_KEEP = ("substrate_phi", "air_phi", "film_mean_phi", "enrichment",
         "phi_star", "F_bulk", "F_grad", "F_wall", "F_total", "F_wall0",
         "mass0", "mass_final", "mass_drift", "rel_mass_drift",
         "projected_dofs", "max_correction", "phi_min", "phi_max",
         "delta_1e", "delta_fit", "wall_g", "wall_h", "wall2_g", "wall2_h",
         "chi", "kappa")


def _scalars(rec):
    return {k: rec[k] for k in _KEEP}


def p3_run(cfg, ctx):
    lvl = cfg["level"]
    ctx.provenance.update(
        mesh={"level": lvl, "side": 2 ** lvl, "dim": 2, "p": 1,
              "lateral": "periodic-x / non-periodic-y (substrate)"},
        time_integrator="BDF1", nonlinear_tol=1e-8,
        wall_energy="f_w = g*phi + h*phi^2 (natural BC on mu eqn)")

    # meshes (periodic default; confined built on demand) + diagnostics
    dm_p, mesh_p, cons_p = build_mesh_dm(lvl, device=ctx.device,
                                         lateral_periodic=True)
    diag_p = WallDiagnostics(dm_p, mesh_p, cons_p)
    meshes = {"periodic": (dm_p, mesh_p, cons_p, diag_p)}

    common = dict(phi0=cfg["phi0"], amp=cfg["amp"], dt=cfg["dt"],
                  t_end=cfg["t_end"], dt_max=cfg["dt_max"], seed=cfg["seed"],
                  onsager=cfg["onsager"], kappa=cfg["kappa"],
                  linsolver=ctx.solver)

    results = {"chi": cfg["chi"], "chi_demix": cfg["chi_demix"],
               "level": lvl, "kappa": cfg["kappa"], "cases": {}}

    # ---- the 6 BC cases -------------------------------------------------
    for c in CASES:
        lat = c["lateral"]
        if lat not in meshes:
            dm, mesh, cons = build_mesh_dm(lvl, device=ctx.device,
                                           lateral_periodic=(lat == "periodic"))
            meshes[lat] = (dm, mesh, cons, WallDiagnostics(dm, mesh, cons))
        dm, mesh, cons, diag = meshes[lat]
        chi = cfg["chi_demix"] if c.get("chi_key") == "demix" else cfg["chi"]
        ctx.log(f"P3 case '{c['key']}': {c['label']}; chi={chi}, "
                f"g={c['g']}, h={c['h']}, lateral={lat}, solver={ctx.solver}")
        rec = simulate(dm, mesh, cons, diag, wall_g=c["g"], wall_h=c["h"],
                       chi=chi, wall2_g=c.get("g2"), wall2_h=c.get("h2"),
                       **common)
        s = _scalars(rec)
        s["label"] = c["label"]
        s["lateral"] = lat
        results["cases"][c["key"]] = s
        ctx.log(f"  sub={s['substrate_phi']:.4f} mean={s['film_mean_phi']:.4f} "
                f"enr={s['enrichment']:+.4f} |dm|={s['mass_drift']:.2e} "
                f"projdof={s['projected_dofs']} F_wall={s['F_wall']:.4f} "
                f"delta_1e={s['delta_1e']:.4f}")
        # store fields / trajectories for figures
        h = ctx.history
        fld = rec["snaps"][max(rec["snaps"])]
        h[f"{c['key']}_field"] = fld
        h[f"{c['key']}_profile"] = rec["profile"]
        h[f"{c['key']}_excess"] = rec["excess"]
        h[f"{c['key']}_ys"] = rec["ys"]
        h[f"{c['key']}_t"] = rec["t"]
        h[f"{c['key']}_mass"] = rec["mass_series"]
        h[f"{c['key']}_Fwall"] = rec["F_wall_series"]
        h[f"{c['key']}_Fbulk"] = rec["F_bulk_series"]
        h[f"{c['key']}_Fgrad"] = rec["F_grad_series"]
        h[f"{c['key']}_Ftot"] = rec["F_total_series"]

    # ---- boundary-layer thickness vs kappa (attracting wall) ------------
    # Run on a STABLE bulk (chi_bl < 2) so the only structure in phi(y)
    # is the wall boundary layer; a demixing bulk fills the interior with
    # domains and corrupts the far-field phi_bulk (spec P3: a REAL
    # measured scaling, not an assertion).
    dm, mesh, cons, diag = meshes["periodic"]
    chi_bl = cfg["chi_bl"]
    kswp = [float(k) for k in cfg["kappa_sweep"]]
    d1e, dfit, subk, excesses = [], [], [], {}
    for kap in kswp:
        ctx.log(f"P3 bdlayer: attract wall, stable bulk chi={chi_bl}, "
                f"kappa={kap:.2e}")
        rec = simulate(dm, mesh, cons, diag, wall_g=-1.5, wall_h=1.0,
                       chi=chi_bl, phi0=cfg["phi0"], amp=cfg["amp"],
                       dt=cfg["dt"], t_end=cfg["t_end"], dt_max=cfg["dt_max"],
                       seed=cfg["seed"], onsager=cfg["onsager"], kappa=kap,
                       linsolver=ctx.solver)
        d1e.append(rec["delta_1e"]); dfit.append(rec["delta_fit"])
        subk.append(rec["substrate_phi"])
        excesses[kap] = (rec["excess"], rec["ys"])
        ctx.log(f"  delta_1e={rec['delta_1e']:.4f} delta_fit={rec['delta_fit']:.4f}")
    # store the smallest- and largest-kappa excess profiles for the figure
    ctx.history["bd_excess_lo"] = excesses[kswp[0]][0]
    ctx.history["bd_excess_hi"] = excesses[kswp[-1]][0]
    ctx.history["bd_excess_ys"] = excesses[kswp[0]][1]
    kswp_a = np.asarray(kswp)
    slope_1e = float(np.polyfit(np.log(kswp_a), np.log(np.asarray(d1e)), 1)[0])
    slope_fit = float(np.polyfit(np.log(kswp_a),
                                 np.log(np.asarray(dfit)), 1)[0])
    results["bdlayer"] = {
        "chi": chi_bl,
        "kappa": kswp, "sqrt_kappa": [float(np.sqrt(k)) for k in kswp],
        "delta_1e": [float(x) for x in d1e],
        "delta_fit": [float(x) for x in dfit],
        "substrate_phi": [float(x) for x in subk],
        "slope_1e": slope_1e, "slope_fit": slope_fit}
    ctx.log(f"P3 bdlayer: delta ~ kappa^{slope_1e:.3f} (1/e), "
            f"kappa^{slope_fit:.3f} (fit); expect ~0.5")
    h = ctx.history
    h["bd_kappa"] = kswp_a
    h["bd_delta_1e"] = np.asarray(d1e)
    h["bd_delta_fit"] = np.asarray(dfit)

    # ---- honest projection / clipping demonstration ---------------------
    # An interior-phi* bounded well never leaves (0,1), so the box
    # projection stays inactive.  To make it FIRE, quench from an
    # admissibility-violating IC (large clip_amp -> phi(0) overshoots
    # (0,1)) on a stable bulk with the attracting wall.  The projection
    # clips the early iterates (projected_dofs > 0), but BDF1 + no-flux
    # conserves the CONVERGED-state content, so the quadrature mass still
    # holds to machine precision -- the honest result.
    camp = cfg["clip_amp"]
    ctx.log(f"P3 clip demo: attracting wall, VIOLATING IC amp={camp} "
            f"(phi(0) overshoots (0,1)) on stable bulk chi={chi_bl}")
    recc = simulate(dm, mesh, cons, diag, wall_g=-1.5, wall_h=1.0,
                    chi=chi_bl, phi0=cfg["phi0"], amp=camp,
                    dt=cfg["dt"], t_end=cfg["clip_t_end"], dt_max=cfg["dt_max"],
                    seed=cfg["seed"], onsager=cfg["onsager"],
                    kappa=cfg["kappa"], linsolver=ctx.solver)
    results["clip"] = {
        "clip_amp": camp, "wall_g": -1.5, "wall_h": 1.0,
        "projected_dofs": recc["projected_dofs"],
        "max_correction": recc["max_correction"],
        "mass_drift": recc["mass_drift"],
        "rel_mass_drift": recc["rel_mass_drift"],
        "phi_min": recc["phi_min"], "phi_max": recc["phi_max"]}
    ctx.log(f"  projdof={recc['projected_dofs']} maxcorr={recc['max_correction']:.2e} "
            f"|dm|={recc['mass_drift']:.2e} phi in "
            f"[{recc['phi_min']:.3f}, {recc['phi_max']:.3f}]")
    return results


def main():
    args = build_parser("OrgElMorph P3 (substrate BCs)").parse_args()
    here = os.path.dirname(__file__)
    baseline = (os.path.join(here, "baseline.yaml")
                if args.mode == "reference" else None)
    # The binary CH (phi, mu) block is a small INDEFINITE saddle-point
    # system where scipy SuperLU's partial pivoting is exact while cuDSS
    # (no pivoting) diverges — so splu is P3's documented `auto` solver
    # and the one EXPECTED was generated with (see P1 for the same note).
    run_tutorial(p3_run, schema=SCHEMA, args=args,
                 default_output=os.path.join(here, "outputs", "p3"),
                 baseline=baseline, default_solver="splu")


if __name__ == "__main__":
    main()
