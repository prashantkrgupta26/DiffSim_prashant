"""P5 — the harness-based entry point (drying film, matched terminal state).

Same film physics as ``run.py`` but driven through the course's standard
harness so it is a *repeatable workflow*: a YAML config (``configs/p5.yaml``,
the canonical record), a provenance ``metadata.json``, a ``results.json``
checked against ``baseline.yaml``, and the standard output layout.

The scientific content that this harness gets RIGHT (the Phase-1 corrections):

1. MATCHED TERMINAL STATE.  ``run.py`` confounded drying RATE with final STATE
   (a slow rate ends at phi_s=0.30 by hitting t_end; a fast one ends dried at
   0.10).  Here every rate is driven PAST the smallest matched dryness, the
   morphology metrics are recorded along the WHOLE phi_s(t) trajectory, and we
   report each metric at MATCHED mean solvent fraction phi_s in {0.30,0.20,0.10}
   (linearly interpolated in phi_s).  Only then is "rate reshapes morphology at
   fixed dryness" an honest claim.

2. UNITS.  The morphology length is the first spectral moment of the LATERAL
   (periodic-x) structure factor, reported as a wavelength both as a fraction
   of the box width AND in CELLS (= fraction * n_x), a real resolved length
   >1 cell — never a bare "cells" number below one.  We report >=2 metrics
   (lateral wavelength, interfacial length, phase contrast).

3. PER-SOLUTE MOVING-FRAME BALANCE.  h(t) * INT phi_i (quadrature on the fixed
   reference domain) is conserved for each solute (only solvent leaves); we
   report the per-solute residual + the height/solvent budget.

5. BIOT REGIME MAP.  Bi = k_e h0 / D_s (drying vs solute diffusion).  A small
   2-D (k_e, D_s) sweep places each run on the Bi axis; morphology collapses
   onto Bi (drying-limited Bi>1 vs diffusion-limited Bi<1).

6. EXIT REASON honesty: the march exit reason is recorded per run; a t_end
   truncated run is never presented as "dried".

    PYTHONPATH=<repo>/src python run_harness.py --config configs/p5.yaml \\
        --mode reference --output outputs/p5 --overwrite
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
from common import config as cfgmod                       # noqa: E402
from common.run_base import build_parser, run_tutorial     # noqa: E402
from diffsim.diagnostics import conservation as dcons      # noqa: E402
from loader import load_system, save_resolved_materials     # noqa: E402

from evaporation import (build_mesh_dm, run_film,            # noqa: E402
                         morphology_metrics)

# The drying blend's chi and N are READ from the materials database
# (materials.yaml drying_demo_ternary), never hand-copied.  These become the
# P5 config DEFAULTS; a YAML/CLI config can still override them.
DRY_BLEND = load_system("drying_demo_ternary")

SCHEMA = cfgmod.ConfigSchema(name="p5", fields={
    "level": cfgmod.Field(int, default=6, min=3, max=9),
    "ke_grid": cfgmod.Field(list, default=[0.3, 0.45, 0.6]),
    "ds_base": cfgmod.Field(float, default=0.2, min=1e-4),
    "ke_regime": cfgmod.Field(list, default=[0.3, 0.6]),
    "ds_grid": cfgmod.Field(list, default=[0.1, 0.2, 0.4]),
    "matched_phis": cfgmod.Field(list, default=[0.30, 0.20, 0.10]),
    "chi": cfgmod.Field(list, default=list(DRY_BLEND.value("chi"))),
    "N": cfgmod.Field(list, default=list(DRY_BLEND.value("N"))),
    "kappa": cfgmod.Field(list, default=[3.0e-4, 3.0e-4]),
    "phi0": cfgmod.Field(list, default=[0.22, 0.22]),
    "amp": cfgmod.Field(float, default=0.01, min=0.0),
    "h0": cfgmod.Field(float, default=1.0, min=0.0),
    "dt": cfgmod.Field(float, default=1e-3, min=0.0),
    "dt_max": cfgmod.Field(float, default=0.01, min=0.0),
    "t_end": cfgmod.Field(float, default=8.0, min=0.0),
    "phis_stop": cfgmod.Field(float, default=0.08, min=0.0),
    "h_min": cfgmod.Field(float, default=0.12, min=0.0),
    "nchecks": cfgmod.Field(int, default=40, min=4),
    "seed": cfgmod.Field(int, default=7),
    "precision": cfgmod.Field(str, default="fp64"),
})


def _interp_at_phis(phis, series, targets):
    """Linear-interpolate ``series`` (a metric sampled along the trajectory) to
    the MATCHED mean-solvent-fraction ``targets``.  ``phis`` decreases along the
    run, so we interpolate on the phi_s-increasing reordering; a target outside
    the sampled range yields NaN (reported honestly, never extrapolated)."""
    phis = np.asarray(phis, float)
    series = np.asarray(series, float)
    order = np.argsort(phis)
    xs, ys = phis[order], series[order]
    out = {}
    lo, hi = xs.min(), xs.max()
    for t in targets:
        out[f"{t:.2f}"] = (float(np.interp(t, xs, ys))
                           if lo <= t <= hi else float("nan"))
    return out


def _one_run(dm, mesh, cons, cfg, k_e, D_s, ctx, store_history=False,
             hist_prefix=""):
    """March one drying film at (k_e, D_s), compute the metric-vs-phi_s
    trajectory, the matched-state metrics, the per-solute balance, the budget,
    and the Biot number."""
    rec = run_film(
        dm, mesh, cons, k_e=k_e, chi=tuple(cfg["chi"]), N=list(cfg["N"]),
        onsager=(D_s, 0.0, D_s), kappa=list(cfg["kappa"]),
        phi0=tuple(cfg["phi0"]), amp=cfg["amp"], h0=cfg["h0"],
        dt=cfg["dt"], dt_max=cfg["dt_max"], t_end=cfg["t_end"],
        phis_stop=cfg["phis_stop"], h_min=cfg["h_min"],
        nchecks=cfg["nchecks"], seed=cfg["seed"],
        device=ctx.device, linsolver=ctx.solver)

    nx = rec["nx"]
    box_w = 1.0                       # lat_scale = 1 -> unit lateral box width
    # morphology metrics along the whole phi_s(t) trajectory (polymer field)
    wl_frac, wl_cells, iface, contrast = [], [], [], []
    for gp in rec["traj_gp"]:
        m = morphology_metrics(gp, box_width=box_w)
        wl_frac.append(m["wl_frac"]); wl_cells.append(m["wl_cells"])
        iface.append(m["iface"]); contrast.append(m["contrast"])
    wl_frac = np.array(wl_frac); wl_cells = np.array(wl_cells)
    iface = np.array(iface); contrast = np.array(contrast)
    phis = rec["phis"]
    tgt = cfg["matched_phis"]

    matched = {}
    mf = _interp_at_phis(phis, wl_frac, tgt)
    mc = _interp_at_phis(phis, wl_cells, tgt)
    mi = _interp_at_phis(phis, iface, tgt)
    mk = _interp_at_phis(phis, contrast, tgt)
    for key in mf:
        matched[key] = {"wl_frac": mf[key], "wl_cells": mc[key],
                        "iface": mi[key], "contrast": mk[key]}

    # per-solute moving-frame balance: h(t) * INT phi_i must be constant.
    h = rec["h"]
    cp, cf = rec["content_p"], rec["content_f"]
    rp = dcons.moving_domain_balance(cp, h)      # h*Cp(t) - h0*Cp(0)
    rf = dcons.moving_domain_balance(cf, h)
    Cp0, Cf0 = float(cp[0] * h[0]), float(cf[0] * h[0])
    bal = {
        "polymer_abs_max": float(np.abs(rp).max()),
        "fullerene_abs_max": float(np.abs(rf).max()),
        "polymer_rel_max": float(np.abs(rp).max() / max(Cp0, 1e-300)),
        "fullerene_rel_max": float(np.abs(rf).max() / max(Cf0, 1e-300)),
        "Cp0": Cp0, "Cf0": Cf0,
    }
    # height / solvent budget.  Solvent content = h - (Cp+Cf) (per-unit box);
    # since solutes conserve, solvent lost == h0 - h(t) exactly.
    solute0 = Cp0 + Cf0
    solv0 = h[0] - solute0
    solvf = h[-1] - float((cp[-1] + cf[-1]) * h[-1])
    budget = {
        "dh": float(h[0] - h[-1]),
        "solvent0": float(solv0),
        "solvent_final": float(solvf),
        "solvent_removed_frac": float((solv0 - solvf) / max(solv0, 1e-300)),
        # budget-closure residual: solvent lost should equal the height drop
        "closure_resid": float(abs((solv0 - solvf) - (h[0] - h[-1]))),
    }
    Bi = float(k_e * cfg["h0"] / D_s)          # drying Peclet / Biot number
    monotone = bool(np.all(np.diff(phis) <= 1e-9))

    out = {
        "k_e": float(k_e), "D_s": float(D_s), "Bi": Bi,
        "regime": "drying-limited" if Bi >= 1.0 else "diffusion-limited",
        "reason": rec["reason"], "t_dry": float(rec["t_dry"]),
        "h_final": float(rec["h_final"]), "phis_final": float(rec["phis_final"]),
        "monotone_drying": monotone,
        "matched": matched, "balance": bal, "budget": budget,
        "final_wl_frac": float(wl_frac[-1]), "final_wl_cells": float(wl_cells[-1]),
        "final_contrast": float(contrast[-1]),
    }
    if store_history:
        ctx.history[f"{hist_prefix}t"] = np.asarray(rec["t"])
        ctx.history[f"{hist_prefix}phis"] = np.asarray(phis)
        ctx.history[f"{hist_prefix}h"] = np.asarray(h)
        ctx.history[f"{hist_prefix}wl_frac"] = wl_frac
        ctx.history[f"{hist_prefix}wl_cells"] = wl_cells
        ctx.history[f"{hist_prefix}iface"] = iface
        ctx.history[f"{hist_prefix}contrast"] = contrast
        # matched-phi_s field snapshots (both frames handled in gen_figures):
        # store the polymer/fullerene grids nearest each matched phi_s + h.
        for t in tgt:
            j = int(np.argmin(np.abs(phis - t)))
            ctx.history[f"{hist_prefix}gp_{t:.2f}"] = np.asarray(rec["traj_gp"][j])
            ctx.history[f"{hist_prefix}gf_{t:.2f}"] = np.asarray(rec["traj_gf"][j])
            ctx.history[f"{hist_prefix}h_{t:.2f}"] = float(h[j])
    return out


def p5_run(cfg, ctx):
    ctx.provenance.update(
        mesh={"level": cfg["level"], "nx": 2 ** cfg["level"],
              "ny": 2 ** cfg["level"] + 1, "dim": 2, "p": 1},
        time_integrator="BDF1", newton_tol=1e-8, linear_tol=1e-10,
        film={"h0": cfg["h0"], "mode": "landau-mapped moving frame"},
        material={"system": DRY_BLEND.name, "chi": list(cfg["chi"]),
                  "N": list(cfg["N"]), "source_file": "materials.yaml"})
    # archive the resolved blend (values + provenance) next to the results
    save_resolved_materials(
        DRY_BLEND, os.path.join(ctx.paths["root"], "materials.resolved.json"))
    dm, mesh, cons = build_mesh_dm(cfg["level"], device=ctx.device)
    ctx.log(f"P5 mesh level {cfg['level']} -> {dm.n_nodes} nodes; "
            f"solver={ctx.solver}")

    # the union of the matched (ds_base) rows and the (ke_regime x ds_grid) map
    ds_base = cfg["ds_base"]
    runset = []
    for k in cfg["ke_grid"]:
        runset.append((float(k), float(ds_base)))
    for k in cfg["ke_regime"]:
        for d in cfg["ds_grid"]:
            runset.append((float(k), float(d)))
    seen, runs = set(), []
    for k, d in runset:
        key = (round(k, 6), round(d, 6))
        if key not in seen:
            seen.add(key)
            runs.append((k, d))

    results = {"runs": {}, "config": {
        "ke_grid": cfg["ke_grid"], "ds_base": ds_base,
        "ke_regime": cfg["ke_regime"], "ds_grid": cfg["ds_grid"],
        "matched_phis": cfg["matched_phis"], "level": cfg["level"],
        "nx": 2 ** cfg["level"]}}

    for k, d in runs:
        is_base = abs(d - ds_base) < 1e-12 and (k in
                  [float(x) for x in cfg["ke_grid"]])
        prefix = f"ke{k:.3f}_" if is_base else ""
        ctx.log(f"  run k_e={k:.3f} D_s={d:.3f}  Bi={k*cfg['h0']/d:.3f}"
                f"{'  [matched row]' if is_base else '  [regime]'}")
        r = _one_run(dm, mesh, cons, cfg, k, d, ctx,
                     store_history=is_base, hist_prefix=prefix)
        rkey = f"ke{k:.3f}_ds{d:.3f}"
        results["runs"][rkey] = r
        ctx.log(f"    reason={r['reason']} phis_final={r['phis_final']:.3f} "
                f"h_final={r['h_final']:.3f} "
                f"bal(P,F)_rel=({r['balance']['polymer_rel_max']:.1e},"
                f"{r['balance']['fullerene_rel_max']:.1e}) "
                f"wl@0.20={r['matched'].get('0.20', {}).get('wl_cells', float('nan')):.2f}cells")

    # ---- convenience roll-ups for figures + the baseline gate -------------
    base_rows = {k: results["runs"][f"ke{k:.3f}_ds{ds_base:.3f}"]
                 for k in [float(x) for x in cfg["ke_grid"]]}
    ke_sorted = sorted(base_rows)
    slow_k, fast_k = ke_sorted[0], ke_sorted[-1]
    slow, fast = base_rows[slow_k], base_rows[fast_k]

    def wl_cells(run, phis):
        return run["matched"].get(f"{phis:.2f}", {}).get("wl_cells", float("nan"))

    # matched-state ordering, reported HONESTLY (this is the claim the old
    # chapter got wrong): does the FASTER (higher-Bi) film freeze a FINER
    # (smaller wavelength) morphology at matched dryness?  Measured: NO — over
    # this Bi range the rate effect is small and, if anything, higher Bi is
    # slightly COARSER (less elapsed time to develop lateral structure).  We
    # report it, but the SCIENTIFIC invariants we gate on are the Bi collapse
    # and the dryness-driven refinement below.
    ordering = {}
    for t in cfg["matched_phis"]:
        sv, fv = wl_cells(slow, t), wl_cells(fast, t)
        ordering[f"{t:.2f}"] = bool(np.isfinite(sv) and np.isfinite(fv)
                                    and fv < sv)

    # DRYNESS REFINEMENT: as the film dries (phi_s falls), the domains refine
    # (wavelength shrinks).  Compare the largest and smallest matched phi_s.
    tgt_sorted = sorted(cfg["matched_phis"])           # e.g. [0.10, 0.20, 0.30]
    lo_ph, hi_ph = tgt_sorted[0], tgt_sorted[-1]

    def refines(run):
        a, b = wl_cells(run, hi_ph), wl_cells(run, lo_ph)   # wet vs dry
        return bool(np.isfinite(a) and np.isfinite(b) and b < a)

    dryness_refines = {k: refines(base_rows[k]) for k in base_rows}

    # Bi COLLAPSE: group runs by Bi; where >=2 runs share a Bi (from different
    # (k_e, D_s)), the matched wavelength must agree — morphology is set by the
    # single group Bi, not by k_e or D_s separately.  Report the worst relative
    # spread at phi_s = the middle matched dryness.
    mid_ph = tgt_sorted[len(tgt_sorted) // 2]
    by_bi = {}
    for r in results["runs"].values():
        by_bi.setdefault(round(r["Bi"], 4), []).append(
            r["matched"].get(f"{mid_ph:.2f}", {}).get("wl_cells", float("nan")))
    collapse_spread = 0.0
    for bi, vals in by_bi.items():
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) >= 2:
            collapse_spread = max(collapse_spread,
                                  (max(vals) - min(vals)) / np.mean(vals))

    # CONTRAST (degree of demixing): the clean, resolution-robust signal.
    def contrast_at(run, ph):
        return run["matched"].get(f"{ph:.2f}", {}).get("contrast", float("nan"))

    # (a) contrast is set by DRYNESS -> rises as phi_s falls (wet hi_ph -> dry
    # lo_ph); (b) at matched dryness it is nearly RATE-INDEPENDENT (small spread
    # across all runs).  These are the de-confounded matched-state invariants.
    def contrast_rises(run):
        a, b = contrast_at(run, hi_ph), contrast_at(run, lo_ph)
        return bool(np.isfinite(a) and np.isfinite(b) and b > a)

    contrast_rises_all = {k: contrast_rises(base_rows[k]) for k in base_rows}
    cvals = [contrast_at(r, mid_ph) for r in results["runs"].values()]
    cvals = [v for v in cvals if np.isfinite(v)]
    contrast_rate_spread = ((max(cvals) - min(cvals)) / np.mean(cvals)
                            if len(cvals) >= 2 else 0.0)

    bal_p = max(r["balance"]["polymer_rel_max"] for r in results["runs"].values())
    bal_f = max(r["balance"]["fullerene_rel_max"] for r in results["runs"].values())
    clos = max(r["budget"]["closure_resid"] for r in results["runs"].values())
    Bis = [r["Bi"] for r in results["runs"].values()]

    results["checks"] = {
        "slow_ke": slow_k, "fast_ke": fast_k,
        "matched_wl_cells_slow_020": wl_cells(slow, 0.20),
        "matched_wl_cells_fast_020": wl_cells(fast, 0.20),
        "matched_wl_cells_slow_030": wl_cells(slow, 0.30),
        "matched_wl_cells_slow_010": wl_cells(slow, 0.10),
        "matched_wl_cells_fast_010": wl_cells(fast, 0.10),
        "contrast_slow_030": contrast_at(slow, 0.30),
        "contrast_slow_020": contrast_at(slow, 0.20),
        "contrast_slow_010": contrast_at(slow, 0.10),
        "contrast_rate_spread_020": float(contrast_rate_spread),
        "contrast_rises_slow": contrast_rises_all.get(slow_k, False),
        "contrast_rises_fast": contrast_rises_all.get(fast_k, False),
        "ordering_higherBi_finer_020": ordering.get("0.20", False),
        "dryness_refines_slow": dryness_refines.get(slow_k, False),
        "dryness_refines_fast": dryness_refines.get(fast_k, False),
        "bi_collapse_spread": float(collapse_spread),
        "balance_polymer_rel_max": bal_p,
        "balance_fullerene_rel_max": bal_f,
        "budget_closure_max": float(clos),
        "all_monotone_drying": bool(all(r["monotone_drying"]
                                        for r in results["runs"].values())),
        "fast_exit_reason": fast["reason"],
        "slow_exit_reason": slow["reason"],
        "Bi_min": float(min(Bis)), "Bi_max": float(max(Bis)),
        "n_runs": len(results["runs"]),
    }
    results["ordering"] = ordering
    results["dryness_refines"] = dryness_refines
    results["contrast_rises"] = contrast_rises_all
    ctx.log(f"matched-state higher-Bi-finer ordering (measured): {ordering}")
    ctx.log(f"contrast rises with dryness: {contrast_rises_all}; "
            f"contrast rate-spread @phi_s={mid_ph:.2f}: {contrast_rate_spread:.2e}")
    ctx.log(f"dryness refines domains: {dryness_refines}; "
            f"Bi-collapse spread @phi_s={mid_ph:.2f}: {collapse_spread:.2e}")
    ctx.log(f"balance rel max: polymer {bal_p:.2e}, fullerene {bal_f:.2e}; "
            f"budget closure {clos:.1e}; Bi in [{min(Bis):.2f}, {max(Bis):.2f}]")
    return results


def main():
    args = build_parser("OrgElMorph P5 (drying film, matched terminal state)"
                        ).parse_args()
    here = os.path.dirname(__file__)
    baseline = (os.path.join(here, "baseline.yaml")
                if args.mode == "reference" else None)
    # evaporation.py's film stepper defaults to cuDSS; the harness passes the
    # resolved solver through so an --solver splu fallback works on a box
    # without cuDSS (documented, like P1).
    run_tutorial(p5_run, schema=SCHEMA, args=args,
                 default_output=os.path.join(here, "outputs", "p5"),
                 baseline=baseline, default_solver="cudss")


if __name__ == "__main__":
    main()
