"""P10 - material-system reproduction capstone, driven through the course
harness so it is a maintained, tolerance-checked, repeatable REFERENCE CASE.

Pipeline (one real system, PDPP5T_PCBM, every parameter from the loader):

  1. PROVENANCE AUDIT.  Classify each loaded parameter (measured / fitted /
     assumed, its uncertainty, whether its value is even known) -- the honest
     "which numbers can I trust" table.

  2. REPRODUCE the Negi spin-coating rate -> morphology trend.  Map the recorded
     spin-speed Biot ladder onto a runnable drying-rate ladder k_e (ratios
     preserved, one documented scale), march each rung, and read morphology at
     MATCHED dryness phi_s (de-confounded).  Report the "faster = finer"
     ordering AND the resolution-robust invariants (Bi collapse; contrast set by
     dryness).

  3. MESH CONVERGENCE.  The same rung on refining meshes; report the
     matched-dryness wavelength (as a box-width FRACTION, mesh-independent) and
     phase contrast, and the level-to-level relative change.

  4. TIME CONVERGENCE.  The same rung at shrinking dt_max; report the same
     matched-dryness metrics and their change.

  5. PARAMETER-UNCERTAINTY SENSITIVITY.  chi_polymer_fullerene carries an
     order-of-magnitude uncertainty; sweep it across that band and report which
     conclusions survive (demixing degree vs the noisy wavelength ordering).

Results -> results.json (flat `checks` block), gated against baseline.yaml in
--mode reference.  The resolved material provenance is archived alongside.

    PYTHONPATH=<repo>/src python run_harness.py --config configs/p10.yaml \\
        --mode reference --output outputs/p10 --overwrite
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir)))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir,
                                                "materials")))
from common import config as cfgmod                       # noqa: E402
from common.run_base import build_parser, run_tutorial     # noqa: E402
from diffsim.diagnostics import conservation as dcons      # noqa: E402
from loader import load_system, save_resolved_materials     # noqa: E402

from material_case import (build_mesh_dm, run_film,          # noqa: E402
                           ke_ladder_from_biot,
                           classify_provenance, chi_triple,
                           interp_at_phis, film_metrics_vs_phis)

# --- the case-study material: loaded ONCE, never hand-copied ---------------
SYSTEM_NAME = "PDPP5T_PCBM"
SYS = load_system(SYSTEM_NAME)
_CHI = chi_triple(SYS)
_N = [float(x) for x in SYS.value("N")]

SCHEMA = cfgmod.ConfigSchema(name="p10", fields={
    "system": cfgmod.Field(str, default=SYSTEM_NAME),
    "level": cfgmod.Field(int, default=5, min=3, max=8),
    "ds_base": cfgmod.Field(float, default=0.2, min=1e-4),
    "k_e_slow": cfgmod.Field(float, default=0.30, min=1e-3),
    "k_e_fast": cfgmod.Field(float, default=0.60, min=1e-3),
    "max_rungs": cfgmod.Field(int, default=0, min=0),   # 0 = use every rung
    "matched_phis": cfgmod.Field(list, default=[0.30, 0.20, 0.12]),
    "chi": cfgmod.Field(list, default=list(_CHI)),
    "N": cfgmod.Field(list, default=list(_N)),
    "kappa": cfgmod.Field(list, default=[3.0e-4, 3.0e-4]),
    "phi0": cfgmod.Field(list, default=[0.20, 0.20]),
    "amp": cfgmod.Field(float, default=0.01, min=0.0),
    "h0": cfgmod.Field(float, default=1.0, min=0.0),
    "dt": cfgmod.Field(float, default=1e-3, min=0.0),
    "dt_max": cfgmod.Field(float, default=0.01, min=0.0),
    "t_end": cfgmod.Field(float, default=8.0, min=0.0),
    "phis_stop": cfgmod.Field(float, default=0.10, min=0.0),
    "h_min": cfgmod.Field(float, default=0.12, min=0.0),
    "nchecks": cfgmod.Field(int, default=32, min=4),
    "seed": cfgmod.Field(int, default=7),
    # convergence + sensitivity are done at ONE ladder rung (conv_rung: index
    # into the sorted ladder) to keep the cost bounded.
    "conv_rung": cfgmod.Field(int, default=2, min=0),
    "conv_levels": cfgmod.Field(list, default=[6]),      # extra meshes (base = level)
    "conv_dts": cfgmod.Field(list, default=[0.02, 0.005]),  # extra dt_max (base = dt_max)
    "chi_pf_sens": cfgmod.Field(list, default=[0.5, 2.0]),  # order-of-mag band (base excl.)
    "precision": cfgmod.Field(str, default="fp64"),
})


# ---------------------------------------------------------------------------
def _march_metrics(cfg, k_e, D_s, level, dt_max, chi, ctx):
    """March one drying film and reduce it to matched-dryness morphology
    metrics + the per-solute conservation residual + the Biot number."""
    dm, mesh, cons = build_mesh_dm(level, device=ctx.device)
    rec = run_film(
        dm, mesh, cons, k_e=k_e, chi=tuple(chi), N=list(cfg["N"]),
        onsager=(D_s, 0.0, D_s), kappa=list(cfg["kappa"]),
        phi0=tuple(cfg["phi0"]), amp=cfg["amp"], h0=cfg["h0"],
        dt=cfg["dt"], dt_max=dt_max, t_end=cfg["t_end"],
        phis_stop=cfg["phis_stop"], h_min=cfg["h_min"],
        nchecks=cfg["nchecks"], seed=cfg["seed"],
        device=ctx.device, linsolver=ctx.solver)
    wl_frac, wl_cells, iface, contrast = film_metrics_vs_phis(rec)
    phis = rec["phis"]
    tgt = cfg["matched_phis"]
    matched = {}
    mf = interp_at_phis(phis, wl_frac, tgt)
    mc = interp_at_phis(phis, wl_cells, tgt)
    mi = interp_at_phis(phis, iface, tgt)
    mk = interp_at_phis(phis, contrast, tgt)
    for key in mf:
        matched[key] = {"wl_frac": mf[key], "wl_cells": mc[key],
                        "iface": mi[key], "contrast": mk[key]}
    # per-solute moving-frame conservation (the hard gate): h*INT phi_i const.
    h, cp, cf = rec["h"], rec["content_p"], rec["content_f"]
    rp = dcons.moving_domain_balance(cp, h)
    rf = dcons.moving_domain_balance(cf, h)
    Cp0, Cf0 = float(cp[0] * h[0]), float(cf[0] * h[0])
    bal_p = float(np.abs(rp).max() / max(Cp0, 1e-300))
    bal_f = float(np.abs(rf).max() / max(Cf0, 1e-300))
    monotone = bool(np.all(np.diff(phis) <= 1e-9))
    out = {
        "k_e": float(k_e), "D_s": float(D_s), "Bi": float(k_e * cfg["h0"] / D_s),
        "level": int(level), "nx": int(2 ** level), "dt_max": float(dt_max),
        "reason": rec["reason"], "t_dry": float(rec["t_dry"]),
        "h_final": float(rec["h_final"]), "phis_final": float(rec["phis_final"]),
        "monotone_drying": monotone,
        "balance_polymer_rel_max": bal_p, "balance_fullerene_rel_max": bal_f,
        "matched": matched,
        "final_wl_frac": float(wl_frac[-1]), "final_contrast": float(contrast[-1]),
    }
    return out, rec, (wl_frac, wl_cells, iface, contrast)


def _subsample_rungs(rungs, max_rungs):
    """Keep every rung (``max_rungs<=0``) or an evenly-spaced subset that always
    includes the slowest and fastest rungs (for the cheap quick/smoke tier)."""
    if max_rungs <= 0 or max_rungs >= len(rungs):
        return rungs
    if max_rungs == 1:
        return [rungs[-1]]
    idx = np.unique(np.linspace(0, len(rungs) - 1, max_rungs).round().astype(int))
    return [rungs[i] for i in idx]


def _mval(run, ph, key):
    return run["matched"].get(f"{ph:.2f}", {}).get(key, float("nan"))


def p10_run(cfg, ctx):
    D_s = cfg["ds_base"]
    prov = classify_provenance(SYS)
    rungs, window = ke_ladder_from_biot(
        SYS.value("biot"), D_s, h0=cfg["h0"],
        k_e_slow=cfg["k_e_slow"], k_e_fast=cfg["k_e_fast"])
    rungs = _subsample_rungs(rungs, cfg["max_rungs"])

    ctx.provenance.update(
        mesh={"level": cfg["level"], "nx": 2 ** cfg["level"],
              "ny": 2 ** cfg["level"] + 1, "dim": 2, "p": 1},
        time_integrator="BDF1", newton_tol=1e-8,
        film={"h0": cfg["h0"], "mode": "landau-mapped moving frame"},
        material={"system": SYS.name, "chi": list(cfg["chi"]),
                  "N": list(cfg["N"]), "ke_window": list(window),
                  "source_file": "materials.yaml"})
    save_resolved_materials(
        SYS, os.path.join(ctx.paths["root"], "materials.resolved.json"))

    ctx.log(f"P10 system={SYS.name}  chi={tuple(cfg['chi'])}  N={cfg['N']}")
    ctx.log(f"spin-speed -> drying-rate ladder (order-preserving map onto "
            f"k_e window {window}):")
    for r in rungs:
        ctx.log(f"  {r['label']:>9s}  Bi_recorded={r['biot_recorded']:.4f}"
                f"  -> k_e={r['k_e']:.3f}  Bi={r['Bi']:.2f}")

    tgt = cfg["matched_phis"]
    hi_ph, lo_ph = max(tgt), min(tgt)
    mid_ph = sorted(tgt)[len(tgt) // 2]

    # ------------------------------------------------------------------ 1+2
    # provenance audit + the reproduced rate ladder
    results = {"system": SYS.name, "provenance": prov,
               "ke_window": list(window), "ladder": rungs,
               "config": {"level": cfg["level"], "nx": 2 ** cfg["level"],
                          "ds_base": D_s, "matched_phis": tgt,
                          "chi": list(cfg["chi"]), "N": list(cfg["N"])},
               "runs": {}}

    for i, r in enumerate(rungs):
        ctx.log(f"[ladder] {r['label']} k_e={r['k_e']:.3f} Bi={r['Bi']:.2f}")
        run, rec, series = _march_metrics(
            cfg, r["k_e"], D_s, cfg["level"], cfg["dt_max"], cfg["chi"], ctx)
        run["label"] = r["label"]; run["rpm"] = r["rpm"]
        results["runs"][r["label"]] = run
        # history for figures (drying curve + metric-vs-phis + matched fields)
        pre = f"{r['label']}_"
        wl_frac, wl_cells, iface, contrast = series
        ctx.history[pre + "t"] = np.asarray(rec["t"])
        ctx.history[pre + "phis"] = np.asarray(rec["phis"])
        ctx.history[pre + "h"] = np.asarray(rec["h"])
        ctx.history[pre + "wl_cells"] = wl_cells
        ctx.history[pre + "wl_frac"] = wl_frac
        ctx.history[pre + "iface"] = iface
        ctx.history[pre + "contrast"] = contrast
        for ph in tgt:
            j = int(np.argmin(np.abs(rec["phis"] - ph)))
            ctx.history[pre + f"gp_{ph:.2f}"] = np.asarray(rec["traj_gp"][j])
            ctx.history[pre + f"h_{ph:.2f}"] = float(rec["h"][j])
        ctx.log(f"    reason={run['reason']} phis_f={run['phis_final']:.3f} "
                f"bal(P,F)_rel=({run['balance_polymer_rel_max']:.1e},"
                f"{run['balance_fullerene_rel_max']:.1e}) "
                f"contrast@{mid_ph:.2f}={_mval(run, mid_ph, 'contrast'):.3f} "
                f"wl@{mid_ph:.2f}={_mval(run, mid_ph, 'wl_cells'):.1f}c")

    labels = [r["label"] for r in rungs]              # ascending rpm / Bi
    slow_l, fast_l = labels[0], labels[-1]
    slow, fast = results["runs"][slow_l], results["runs"][fast_l]

    # "faster = finer" ordering at matched dryness (the noisy, honest claim)
    finer_ordering = {}
    for ph in tgt:
        sv, fv = _mval(slow, ph, "wl_cells"), _mval(fast, ph, "wl_cells")
        finer_ordering[f"{ph:.2f}"] = bool(np.isfinite(sv) and np.isfinite(fv)
                                           and fv < sv)

    # Bi collapse across the whole ladder: matched wavelength spread at mid phi_s
    wl_mid = [_mval(results["runs"][l], mid_ph, "wl_cells") for l in labels]
    wl_mid = [v for v in wl_mid if np.isfinite(v)]
    wl_spread = ((max(wl_mid) - min(wl_mid)) / np.mean(wl_mid)
                 if len(wl_mid) >= 2 else float("nan"))

    # contrast set by dryness: rises wet->dry, and rate-independent at matched
    def contrast_rises(run):
        a, b = _mval(run, hi_ph, "contrast"), _mval(run, lo_ph, "contrast")
        return bool(np.isfinite(a) and np.isfinite(b) and b > a)
    contrast_rises_all = {l: contrast_rises(results["runs"][l]) for l in labels}
    c_mid = [_mval(results["runs"][l], mid_ph, "contrast") for l in labels]
    c_mid = [v for v in c_mid if np.isfinite(v)]
    contrast_rate_spread = ((max(c_mid) - min(c_mid)) / np.mean(c_mid)
                            if len(c_mid) >= 2 else float("nan"))

    # ------------------------------------------------------------------ 3
    # MESH convergence at the chosen rung (compare box-FRACTION wavelength +
    # contrast, both mesh-independent quantities).
    conv_rung = rungs[min(cfg["conv_rung"], len(rungs) - 1)]
    ck_e = conv_rung["k_e"]
    mesh_conv = {"base_level": cfg["level"], "k_e": ck_e, "levels": {},
                 "phi_s": mid_ph}
    base_run = results["runs"][conv_rung["label"]]
    mesh_conv["levels"][str(cfg["level"])] = {
        "wl_frac": _mval(base_run, mid_ph, "wl_frac"),
        "contrast": _mval(base_run, mid_ph, "contrast")}
    for lv in cfg["conv_levels"]:
        ctx.log(f"[mesh-conv] level {lv} at k_e={ck_e:.3f}")
        run, _, _ = _march_metrics(cfg, ck_e, D_s, int(lv), cfg["dt_max"],
                                   cfg["chi"], ctx)
        mesh_conv["levels"][str(int(lv))] = {
            "wl_frac": _mval(run, mid_ph, "wl_frac"),
            "contrast": _mval(run, mid_ph, "contrast")}
    lv_sorted = sorted(int(k) for k in mesh_conv["levels"])
    c_coarse = mesh_conv["levels"][str(lv_sorted[0])]["contrast"]
    c_fine = mesh_conv["levels"][str(lv_sorted[-1])]["contrast"]
    mesh_conv["contrast_rel_change"] = float(abs(c_fine - c_coarse)
                                             / max(abs(c_fine), 1e-30))
    wf_coarse = mesh_conv["levels"][str(lv_sorted[0])]["wl_frac"]
    wf_fine = mesh_conv["levels"][str(lv_sorted[-1])]["wl_frac"]
    mesh_conv["wl_frac_rel_change"] = float(abs(wf_fine - wf_coarse)
                                            / max(abs(wf_fine), 1e-30))

    # ------------------------------------------------------------------ 4
    # TIME convergence at the chosen rung.
    time_conv = {"level": cfg["level"], "k_e": ck_e, "dts": {}, "phi_s": mid_ph}
    time_conv["dts"][f"{cfg['dt_max']:.4f}"] = {
        "wl_frac": _mval(base_run, mid_ph, "wl_frac"),
        "contrast": _mval(base_run, mid_ph, "contrast")}
    for dtm in cfg["conv_dts"]:
        ctx.log(f"[time-conv] dt_max {dtm:g} at k_e={ck_e:.3f}")
        run, _, _ = _march_metrics(cfg, ck_e, D_s, cfg["level"], float(dtm),
                                   cfg["chi"], ctx)
        time_conv["dts"][f"{float(dtm):.4f}"] = {
            "wl_frac": _mval(run, mid_ph, "wl_frac"),
            "contrast": _mval(run, mid_ph, "contrast")}
    dt_keys = sorted(time_conv["dts"], key=lambda s: float(s))
    c_lo = time_conv["dts"][dt_keys[0]]["contrast"]      # smallest dt_max
    c_hi = time_conv["dts"][dt_keys[-1]]["contrast"]     # largest dt_max
    time_conv["contrast_rel_change"] = float(abs(c_hi - c_lo)
                                             / max(abs(c_lo), 1e-30))

    # ------------------------------------------------------------------ 5
    # PARAMETER-UNCERTAINTY sensitivity: chi_polymer_fullerene over its
    # order-of-magnitude band.  Report matched-dryness contrast + whether the
    # film still demixes; the ROBUST conclusion is that demixing degree tracks
    # chi_pf while the wavelength ordering stays noisy.
    chi_pf0 = cfg["chi"][0]
    sens = {"param": "chi_polymer_fullerene", "base": chi_pf0, "k_e": ck_e,
            "phi_s": mid_ph, "runs": {}}
    sens["runs"][f"{chi_pf0:g}"] = {
        "contrast": _mval(base_run, mid_ph, "contrast"),
        "wl_cells": _mval(base_run, mid_ph, "wl_cells")}
    for chi_pf in cfg["chi_pf_sens"]:
        chi_v = [float(chi_pf), cfg["chi"][1], cfg["chi"][2]]
        ctx.log(f"[chi-sens] chi_pf={chi_pf:g} at k_e={ck_e:.3f}")
        run, _, _ = _march_metrics(cfg, ck_e, D_s, cfg["level"], cfg["dt_max"],
                                   chi_v, ctx)
        sens["runs"][f"{float(chi_pf):g}"] = {
            "contrast": _mval(run, mid_ph, "contrast"),
            "wl_cells": _mval(run, mid_ph, "wl_cells")}
    chi_keys = sorted(sens["runs"], key=lambda s: float(s))
    c_lochi = sens["runs"][chi_keys[0]]["contrast"]
    c_hichi = sens["runs"][chi_keys[-1]]["contrast"]
    sens["contrast_monotone_in_chi"] = bool(
        np.isfinite(c_lochi) and np.isfinite(c_hichi) and c_hichi > c_lochi)
    sens["contrast_rel_span"] = float(abs(c_hichi - c_lochi)
                                      / max(abs(c_hichi), 1e-30))

    # ------------------------------------------------------------------ gates
    bal_p = max(r["balance_polymer_rel_max"] for r in results["runs"].values())
    bal_f = max(r["balance_fullerene_rel_max"] for r in results["runs"].values())
    Bis = [r["Bi"] for r in results["runs"].values()]
    results["mesh_conv"] = mesh_conv
    results["time_conv"] = time_conv
    results["sensitivity"] = sens
    results["finer_ordering"] = finer_ordering
    results["contrast_rises"] = contrast_rises_all
    results["checks"] = {
        "n_rungs": len(rungs),
        "Bi_min": float(min(Bis)), "Bi_max": float(max(Bis)),
        "all_monotone_drying": bool(all(r["monotone_drying"]
                                        for r in results["runs"].values())),
        "all_dried": bool(all(r["reason"] == "phis_stop"
                              for r in results["runs"].values())),
        # hard, resolution-robust conservation gate
        "balance_polymer_rel_max": bal_p,
        "balance_fullerene_rel_max": bal_f,
        # reproduced-trend readouts
        "slow_label": slow_l, "fast_label": fast_l,
        "contrast_slow_hi": _mval(slow, hi_ph, "contrast"),
        "contrast_slow_mid": _mval(slow, mid_ph, "contrast"),
        "contrast_slow_lo": _mval(slow, lo_ph, "contrast"),
        "contrast_fast_mid": _mval(fast, mid_ph, "contrast"),
        "wl_cells_slow_mid": _mval(slow, mid_ph, "wl_cells"),
        "wl_cells_fast_mid": _mval(fast, mid_ph, "wl_cells"),
        "finer_ordering_mid": finer_ordering.get(f"{mid_ph:.2f}", False),
        "contrast_rises_slow": contrast_rises_all.get(slow_l, False),
        "contrast_rises_fast": contrast_rises_all.get(fast_l, False),
        "contrast_rate_spread_mid": float(contrast_rate_spread),
        "bi_collapse_wl_spread_mid": float(wl_spread),
        # convergence
        "mesh_contrast_rel_change": mesh_conv["contrast_rel_change"],
        "mesh_wl_frac_rel_change": mesh_conv["wl_frac_rel_change"],
        "time_contrast_rel_change": time_conv["contrast_rel_change"],
        # sensitivity
        "sens_contrast_monotone_in_chi": sens["contrast_monotone_in_chi"],
        "sens_contrast_rel_span": sens["contrast_rel_span"],
    }
    ctx.log(f"reproduced-trend: finer_ordering={finer_ordering}  "
            f"contrast_rises={contrast_rises_all}")
    ctx.log(f"robust invariants: Bi-collapse wl spread@{mid_ph:.2f}="
            f"{wl_spread:.2%}  contrast rate-spread={contrast_rate_spread:.2%}")
    ctx.log(f"convergence: mesh dContrast={mesh_conv['contrast_rel_change']:.2%}"
            f"  time dContrast={time_conv['contrast_rel_change']:.2%}")
    ctx.log(f"sensitivity: chi_pf band -> contrast monotone="
            f"{sens['contrast_monotone_in_chi']}  span="
            f"{sens['contrast_rel_span']:.2%}")
    ctx.log(f"conservation: bal(P,F)_rel max=({bal_p:.1e},{bal_f:.1e})")
    return results


def main():
    args = build_parser("OrgElMorph P10 (material-system reproduction "
                        "capstone: PDPP5T:PCBM drying)").parse_args()
    baseline = (os.path.join(_HERE, "baseline.yaml")
                if args.mode == "reference" else None)
    run_tutorial(p10_run, schema=SCHEMA, args=args,
                 default_output=os.path.join(_HERE, "outputs", "p10"),
                 baseline=baseline, default_solver="cudss")


if __name__ == "__main__":
    main()
