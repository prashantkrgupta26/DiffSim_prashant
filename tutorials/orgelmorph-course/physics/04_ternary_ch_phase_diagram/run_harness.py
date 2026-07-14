"""P4 -- ternary Cahn-Hilliard, the harness-driven scientific workflow.

Same physics as ``run.py`` but run through the course harness: a YAML config
(the canonical record), a provenance ``metadata.json``, a ``results.json``
checked against ``baseline.yaml``, and ``history.npz`` holding every array the
figures need (so ``gen_figures.py`` renders purely from the saved run dir).

What it measures (spec Phase 1, P4):
  * clustered coexisting-phase MEANS / COVARIANCES / POPULATIONS via a
    2-component GMM, plus an interface-excluded sensitivity (ternary.py);
  * the free-energy landscape + spinodal (det H) + eigenvalue sign maps;
  * admissibility -- Gibbs-simplex residual, field bounds, clip fraction
    (diffsim.diagnostics.admissibility);
  * per-solute QUADRATURE content drift (diffsim.diagnostics.conservation)
    and a LEVER-RULE residual (conserved mean == population-weighted phase
    average);
  * the spinodal determinant at the initial blend;
  * the N_i (degree-of-polymerization) shift: how unequal N moves the
    critical chi_12 / det, with an optional asymmetric-N run.

chi and N are READ from materials/ternary_p4.yaml through the validated
loader -- never hand-copied.

    PYTHONPATH=<repo>/src python run_harness.py --config configs/p4.yaml \\
        --mode reference --output outputs/p4 --overwrite
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                os.pardir, os.pardir)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                os.pardir, os.pardir,
                                                "materials")))
from common import config as cfgmod                        # noqa: E402
from common.run_base import build_parser, run_tutorial      # noqa: E402
from diffsim.diagnostics import admissibility as dadm       # noqa: E402
from diffsim.diagnostics import conservation as dcons       # noqa: E402

from loader import load_blend                               # noqa: E402
from ternary import (build_mesh_dm, run, cluster_phases,    # noqa: E402
                     spinodal_hessian, spinodal_chi12_crit,
                     spinodal_maps, predict_binodal)

SCHEMA = cfgmod.ConfigSchema(name="p4", fields={
    "blend": cfgmod.Field(str, default="synthetic_demix"),
    "asym_blend": cfgmod.Field(str, default="synthetic_demix_asymN"),
    "materials": cfgmod.Field(str, default="ternary_p4.yaml"),
    "level": cfgmod.Field(int, default=6, min=2, max=8),
    "t_end": cfgmod.Field(float, default=0.6, min=0.0),
    "dt": cfgmod.Field(float, default=5e-4, min=0.0),
    "dt_max": cfgmod.Field(float, default=0.02, min=0.0),
    "mobility": cfgmod.Field(float, default=0.2, min=0.0),
    "kappa": cfgmod.Field(float, default=6e-4, min=0.0),
    "phi1_0": cfgmod.Field(float, default=0.35, min=0.0, max=1.0),
    "phi2_0": cfgmod.Field(float, default=0.35, min=0.0, max=1.0),
    "amp": cfgmod.Field(float, default=0.02, min=0.0),
    "keep": cfgmod.Field(float, default=0.5, min=0.05, max=1.0),
    "run_asym_sim": cfgmod.Field(bool, default=True),
    # the asymmetric-N blend is stiffer (weaker 1/N1 log barrier); it is
    # marched on its OWN small mesh / short horizon just to demonstrate the
    # shift -- the quantitative shift is the analytic spinodal result.
    "asym_level": cfgmod.Field(int, default=4, min=2, max=7),
    "asym_t_end": cfgmod.Field(float, default=0.1, min=0.0),
    "seed": cfgmod.Field(int, default=6),
    "precision": cfgmod.Field(str, default="fp64"),
})


def _admissibility(p1, p2):
    """Gibbs-simplex residual, per-species bounds, and clip report for the
    final (phi1, phi2, phis) cloud, via the shared diagnostics."""
    phis = 1.0 - p1 - p2
    simp = dadm.simplex_residual([p1, p2, phis])
    rep1 = dadm.projection_report(p1)
    rep2 = dadm.projection_report(p2)
    reps = dadm.projection_report(phis)
    projected = rep1["projected_dofs"] + rep2["projected_dofs"] \
        + reps["projected_dofs"]
    return {
        "simplex_max_abs": simp["max_abs"],
        "simplex_rms": simp["rms"],
        "phi1_min": rep1["phi_min"], "phi1_max": rep1["phi_max"],
        "phi2_min": rep2["phi_min"], "phi2_max": rep2["phi_max"],
        "phis_min": reps["phi_min"], "phis_max": reps["phi_max"],
        "projected_dofs": int(projected),
        "clipped_fraction": float((projected) / (3 * len(p1))),
        "max_correction": max(rep1["max_correction"], rep2["max_correction"],
                              reps["max_correction"]),
    }


def p4_run(cfg, ctx):
    blend = load_blend(cfg["blend"], path=os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, "materials",
        cfg["materials"]))
    asym = load_blend(cfg["asym_blend"], path=os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, "materials",
        cfg["materials"]))
    chi, N = blend.chi, blend.N
    phi0 = (cfg["phi1_0"], cfg["phi2_0"])
    ctx.provenance.update(
        mesh={"level": cfg["level"], "side": 2 ** cfg["level"] + 1,
              "dim": 2, "p": 1},
        time_integrator="BDF1",
        nonlinear_tol=1e-8,
        material={"blend": blend.name, "chi": list(chi), "N": list(N),
                  "source": blend.provenance})

    # ---- spinodal at the initial blend (det H < 0 => unstable) --------
    H, det_ic, ev = spinodal_hessian(phi0[0], phi0[1], chi, N)
    chi12_crit = spinodal_chi12_crit(phi0[0], phi0[1], chi, N)
    ctx.log(f"blend {blend.name}: chi={chi}, N={N}")
    ctx.log(f"spinodal @IC ({phi0[0]},{phi0[1]}): det={det_ic:.4g}, "
            f"eig=[{ev[0]:.3g},{ev[1]:.3g}], chi12_crit={chi12_crit:.4g}, "
            f"unstable={det_ic < 0}")

    # ---- march the primary ternary quench -----------------------------
    ctx.log(f"march: level {cfg['level']} ({2**cfg['level']}^2), "
            f"t_end={cfg['t_end']}, solver={ctx.solver}")
    dm, mesh, cons = build_mesh_dm(cfg["level"], device=ctx.device)
    rec = run(dm, mesh, cons, chi=chi, mobility=cfg["mobility"],
              kappa=(cfg["kappa"], cfg["kappa"]), phi0=phi0,
              amp=cfg["amp"], dt=cfg["dt"], t_end=cfg["t_end"],
              dt_max=cfg["dt_max"], seed=cfg["seed"], N=N,
              device=ctx.device, linsolver=ctx.solver)
    p1f, p2f = rec["p1f"], rec["p2f"]

    # ---- cluster the cloud into the two coexisting phases -------------
    clus = cluster_phases(p1f, p2f, rec["side"], keep=cfg["keep"],
                          seed=cfg["seed"])
    A, B = clus["phaseA_mean"], clus["phaseB_mean"]
    ctx.log(f"phases (GMM): A={np.round(A,3)} B={np.round(B,3)} "
            f"pop={np.round(clus['population'],3)} "
            f"sens={clus['endpoint_sensitivity']:.3g}")

    # ---- admissibility ------------------------------------------------
    adm = _admissibility(p1f, p2f)
    ctx.log(f"admissibility: simplex |sum-1| max={adm['simplex_max_abs']:.2e}, "
            f"phi1 in [{adm['phi1_min']:.3f},{adm['phi1_max']:.3f}], "
            f"clipped_frac={adm['clipped_fraction']:.2e}")

    # ---- quadrature conservation + lever rule -------------------------
    d1 = dcons.mass_drift(rec["content1"])
    d2 = dcons.mass_drift(rec["content2"])
    vol = rec["volume"]
    mean_q = [float(rec["content1"][-1] / vol), float(rec["content2"][-1] / vol)]
    w = clus["population"]
    weighted = [w[0] * A[0] + w[1] * B[0], w[0] * A[1] + w[1] * B[1]]
    lever_res = float(np.hypot(mean_q[0] - weighted[0],
                               mean_q[1] - weighted[1]))
    ctx.log(f"quadrature content drift: |dC1|={d1:.2e}, |dC2|={d2:.2e}; "
            f"lever residual={lever_res:.3g}")

    # ---- predicted binodal (common tangent through the mean) ----------
    bino = predict_binodal(chi, N, mean_q, A, B, lever0=w[0])
    if bino:
        ctx.log(f"predicted binodal: A={np.round(bino['phaseA'],3)} "
                f"B={np.round(bino['phaseB'],3)} lever={bino['lever']:.3f}")
    else:
        ctx.log("predicted binodal: common-tangent solve did not converge")

    # ---- N_i shift: symmetric vs asymmetric-N spinodal ----------------
    Ha, det_a, eva = spinodal_hessian(phi0[0], phi0[1], asym.chi, asym.N)
    crit_a = spinodal_chi12_crit(phi0[0], phi0[1], asym.chi, asym.N)
    nshift = {
        "asym_blend": asym.name, "asym_N": list(asym.N),
        "sym_det": det_ic, "asym_det": det_a,
        "sym_chi12_crit": chi12_crit, "asym_chi12_crit": crit_a,
        "delta_chi12_crit": float(crit_a - chi12_crit),
    }
    ctx.log(f"N-shift ({N}->{asym.N}): chi12_crit {chi12_crit:.3g}->"
            f"{crit_a:.3g} (delta {crit_a - chi12_crit:+.3g}), "
            f"det {det_ic:.3g}->{det_a:.3g}")

    if cfg["run_asym_sim"]:
        ctx.log(f"asymmetric-N run (level {cfg['asym_level']}, "
                f"t_end {cfg['asym_t_end']}) ...")
        dm2, mesh2, cons2 = build_mesh_dm(cfg["asym_level"], device=ctx.device)
        reca = run(dm2, mesh2, cons2, chi=asym.chi, mobility=cfg["mobility"],
                   kappa=(cfg["kappa"], cfg["kappa"]), phi0=phi0,
                   amp=cfg["amp"], dt=cfg["dt"], t_end=cfg["asym_t_end"],
                   dt_max=cfg["dt_max"], seed=cfg["seed"], N=asym.N,
                   device=ctx.device, linsolver=ctx.solver)
        clus_a = cluster_phases(reca["p1f"], reca["p2f"], reca["side"],
                                keep=cfg["keep"], seed=cfg["seed"])
        nshift["asym_phaseA_mean"] = clus_a["phaseA_mean"]
        nshift["asym_phaseB_mean"] = clus_a["phaseB_mean"]
        nshift["asym_spread_end"] = reca["spreadf"]
        ctx.log(f"asym phases: A={np.round(clus_a['phaseA_mean'],3)} "
                f"B={np.round(clus_a['phaseB_mean'],3)} "
                f"spread_end={reca['spreadf']:.3f}")
        # store asym cloud for the figure
        ctx.history["asym_p1f"] = reca["p1f"]
        ctx.history["asym_p2f"] = reca["p2f"]

    # ---- landscape arrays for the figures -----------------------------
    P1g, P2g, fmap, detmap, eigmap = spinodal_maps(chi, N, n=220)
    ctx.history.update(
        p1f=p1f, p2f=p2f,
        mean1=rec["mean1"], mean2=rec["mean2"], t=rec["t"],
        content1=rec["content1"], content2=rec["content2"],
        grid_p1=P1g, grid_p2=P2g, grid_f=fmap,
        grid_det=detmap, grid_min_eig=eigmap,
    )
    for k, (pp1, pp2) in rec["snaps"].items():
        ctx.history[f"snap_p1_{k:.4f}"] = pp1
        ctx.history[f"snap_p2_{k:.4f}"] = pp2

    results = {
        "blend": blend.name, "chi": list(chi), "N": list(N),
        "phi0": list(phi0), "side": rec["side"],
        "spread_start": rec["spread0"], "spread_end": rec["spreadf"],
        "clustering": clus,
        "spinodal_ic": {
            "det": det_ic, "eig_min": float(ev[0]), "eig_max": float(ev[1]),
            "chi12_crit": chi12_crit, "unstable": bool(det_ic < 0)},
        "admissibility": adm,
        "conservation": {
            "content1_start": float(rec["content1"][0]),
            "content1_end": float(rec["content1"][-1]),
            "content1_drift": d1,
            "content2_start": float(rec["content2"][0]),
            "content2_end": float(rec["content2"][-1]),
            "content2_drift": d2,
            "volume": vol, "mean1_quad": mean_q[0], "mean2_quad": mean_q[1]},
        "lever": {"mean_quad": mean_q, "weighted_avg": weighted,
                  "residual": lever_res},
        "binodal": bino,
        "nshift": nshift,
    }
    return results


def main():
    args = build_parser("OrgElMorph P4 (harness)").parse_args()
    here = os.path.dirname(__file__)
    baseline = (os.path.join(here, "baseline.yaml")
                if args.mode == "reference" else None)
    # The coupled multiphase block is well-conditioned for cuDSS (unlike the
    # binary (c, mu) saddle block of P1), so we use the standard `auto`
    # policy: cuDSS when available, documented scipy-SuperLU (splu) fallback
    # for small runs without cuDSS. An explicit --solver still overrides.
    run_tutorial(p4_run, schema=SCHEMA, args=args,
                 default_output=os.path.join(here, "outputs", "p4"),
                 baseline=baseline)


if __name__ == "__main__":
    main()
