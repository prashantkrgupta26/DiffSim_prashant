"""OrgElMorph course - Computational C10: validation & uncertainty budget.

"Is the answer right?" is not one question but six, and conflating them is the
most common way a computational study misleads (spec Phase 3 / C10):

  1. CODE verification    -- am I solving the equations right?  (order/MMS,
                             conservation, energy monotonicity -- a bug check,
                             answered against MATH, not nature)
  2. SOLUTION verification -- is the DISCRETIZATION error of THIS run small
                             enough?  (mesh refinement of the observable ->
                             a numerical uncertainty)
  3. MODEL validation     -- is it the right model of REALITY?  (compare the
                             observable against an independent benchmark;
                             here the linear-stability wavelength -- true
                             experimental validation needs lab S(q))
  4. PARAMETER uncertainty -- how much does the observable move because a
                             material input is only known to some spread?
                             (propagate chi's uncertainty from materials.yaml)
  5. STOCHASTIC variability -- how much does it move seed to seed, at FIXED
                             parameters?  (the C9/P8 seed ensemble)
  6. MODEL-EXPERIMENT discrepancy -- the residual gap once 2-5 are accounted
                             (documented honestly: no lab data here).

The deliverable is an UNCERTAINTY BUDGET: one material case (P3HT:PCBM from
materials.yaml), one morphology observable (the interfacial-area domain length
L_area of the binary Cahn--Hilliard spinodal, physics/01), and a table that
propagates the material's chi uncertainty AND the seed-ensemble stochastic
variability AND the discretization error into a combined uncertainty -- and
says which source DOMINATES.

    PYTHONPATH=<repo>/src python uncertainty.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir)))
sys.path.insert(0, os.path.abspath(os.path.join(
    _HERE, os.pardir, os.pardir, "physics", "01_ch_binary_energies")))
sys.path.insert(0, os.path.abspath(os.path.join(
    _HERE, os.pardir, os.pardir, "materials")))

from diffsim.diagnostics import stochastic as dstoch          # noqa: E402
from diffsim.diagnostics import energy as denergy             # noqa: E402
from spinodal import (run_spinodal, _lengthscales,             # noqa: E402
                      linear_dispersion, linear_probe)
from loader import load_system                                 # noqa: E402

# --- the ONE material case (materials.yaml, schema v2) ---------------------
MATERIAL = "P3HT_PCBM"
CHI_PARAM = "chi_polymer_fullerene"

# frozen numerical model (the C9 engine, nominal operating point)
BASE = dict(energy="fh", M=1.0, kappa=5.0e-4, fh_A=0.15, amp=0.05,
            dt=0.02, steps=40, measure_every=5, solver="splu")
NOMINAL_LEVEL = 5


# ---------------------------------------------------------------------------
# one real run (cached by (chi, seed, level) so no GPU work is repeated)
# ---------------------------------------------------------------------------
_CACHE = {}


def _run(chi, seed, level, device):
    key = (round(chi, 6), seed, level)
    if key in _CACHE:
        return _CACHE[key]
    rec = run_spinodal(
        energy=BASE["energy"], level=level, steps=BASE["steps"],
        dt=BASE["dt"], M=BASE["M"], kappa=BASE["kappa"], fh_A=BASE["fh_A"],
        fh_B=chi, amp=BASE["amp"], seed=seed, device=device,
        linsolver=BASE["solver"], measure_every=BASE["measure_every"])
    end = rec["snaps"][max(rec["snaps"])]
    lam_peak, lam_fm, L_area = _lengthscales(end, rec["dx"])
    out = {
        "L_area": float(L_area), "lambda_fm": float(lam_fm),
        "F_total": np.asarray(rec["F_total"]),
        "mass": np.asarray(rec["mass"]),
        "mass_drift": float(abs(rec["mass"][-1] - rec["mass"][0])),
        "c_avg": float(rec["c_avg"]), "kappa": float(rec["kappa"]),
        "dx": float(rec["dx"]),
    }
    _CACHE[key] = out
    return out


# ---------------------------------------------------------------------------
# the six facets
# ---------------------------------------------------------------------------
def code_verification(nominal_run):
    """Facet 1: am I solving the equations right?  Two invariants that a real
    bug would break -- mass conservation (Model-B is conservative) and the
    Lyapunov energy decrease.  (The full MMS order study is Chapter C1.)"""
    F = nominal_run["F_total"]
    return {
        "mass_drift": nominal_run["mass_drift"],
        "energy_monotone": bool(denergy.is_monotone_decreasing(F, skip=1)),
        "largest_positive_increment":
            float(denergy.largest_positive_increment(F[1:])),
        "note": "full spatial/temporal MMS order verification -> Chapter C1",
    }


def solution_verification(chi, seed, device, levels=(4, 5, 6)):
    """Facet 2: is the discretization error of L_area small enough?  Refine the
    mesh and watch the observable; the coarse->fine change is a numerical
    (discretization) uncertainty on the observable."""
    Ls = {lv: _run(chi, seed, lv, device)["L_area"] for lv in levels}
    lv = sorted(levels)
    # conservative discretization uncertainty: the change over the last
    # refinement (finest minus next-finest).
    disc = abs(Ls[lv[-1]] - Ls[lv[-2]])
    return {"levels": list(lv), "L_by_level": {int(k): v for k, v in Ls.items()},
            "discretization_uncertainty": float(disc),
            "L_finest": float(Ls[lv[-1]])}


def model_validation(chi, seed, device):
    """Facet 3: is it the right model of reality?  Absent lab data we validate
    against an INDEPENDENT physical benchmark: Cahn--Hilliard linear stability
    predicts the fastest-growing ONSET wavelength lambda* = 2 pi / k*.  A tiny-dt
    growth-spectrum probe (physics/01 linear_probe) MEASURES the early-time
    fastest mode; its peak must land on lambda* while the perturbation is small
    (the same check as P1).  We report the relative discrepancy -- a
    model/theory validation.  NB the observable L_area is the COARSENED end
    state, many times lambda*, so the onset probe (not the end field) is the
    correct benchmark.  Experimental validation would need measured S(q)."""
    pr = linear_probe(energy=BASE["energy"], level=7, dt=2e-4, n_steps=10,
                      M=BASE["M"], kappa=BASE["kappa"], fh_A=BASE["fh_A"],
                      fh_B=chi, amp=0.02, seed=seed, device=device,
                      linsolver=BASE["solver"])
    lam_star = pr["lambda_star"]
    lam_meas = pr["lambda_meas"]
    rel = abs(lam_meas - lam_star) / lam_star if lam_star > 0 else float("nan")
    return {"lambda_star_predicted": float(lam_star),
            "lambda_measured": float(lam_meas),
            "relative_discrepancy": float(rel),
            "note": "onset-wavelength validation vs linear-stability theory "
                    "(tiny-dt probe); experimental validation requires "
                    "measured S(q)"}


def parameter_uncertainty(chi0, sigma_chi, seeds, device, level,
                          delta=0.2):
    """Facet 4: propagate the MATERIAL parameter uncertainty.  A central
    finite-difference sensitivity dL/dchi (each side averaged over seeds to
    strip the stochastic noise) times the material's chi spread gives the
    parametric standard deviation of the observable."""
    lo = np.mean([_run(chi0 - delta, s, level, device)["L_area"]
                  for s in seeds])
    hi = np.mean([_run(chi0 + delta, s, level, device)["L_area"]
                  for s in seeds])
    dLdchi = (hi - lo) / (2.0 * delta)
    sigma_param = abs(dLdchi) * sigma_chi
    return {"chi0": float(chi0), "sigma_chi": float(sigma_chi),
            "delta": float(delta), "L_lo": float(lo), "L_hi": float(hi),
            "dL_dchi": float(dLdchi), "sigma_param": float(sigma_param)}


def stochastic_variability(chi, seeds, device, level):
    """Facet 5: the seed-ensemble spread at FIXED parameters (C9/P8)."""
    Ls = np.array([_run(chi, s, level, device)["L_area"] for s in seeds])
    agg = dstoch.ensemble_aggregate(Ls)
    ci = dstoch.bootstrap_ci(Ls, seed=0)
    return {"n_seeds": int(agg["n"]), "L_mean": float(agg["mean"]),
            "sigma_stoch": float(agg["sd"]), "sem": float(agg["sem"]),
            "ci_low": ci["low"], "ci_high": ci["high"]}


# ---------------------------------------------------------------------------
# the uncertainty BUDGET (the deliverable)
# ---------------------------------------------------------------------------
def build_budget(device="cuda:0",
                 ens_seeds=(1, 2, 3, 4, 5), sens_seeds=(1, 2, 3)):
    """Run the real budget campaign and assemble the uncertainty budget.

    Combines the parametric (material chi), stochastic (seed), and numerical
    (discretization) standard deviations in quadrature and reports which
    dominates.  All spreads are MEASURED from real runs, not assumed.
    """
    mat = load_system(MATERIAL)
    chi_p = mat.param(CHI_PARAM)
    chi0 = float(chi_p.value)
    # the material's reported spread is our 1-sigma parametric uncertainty
    sigma_chi = float(chi_p.uncertainty.get("value"))
    unc_type = chi_p.uncertainty.get("type")

    nominal = _run(chi0, ens_seeds[0], NOMINAL_LEVEL, device)

    code = code_verification(nominal)
    stoch = stochastic_variability(chi0, ens_seeds, device, NOMINAL_LEVEL)
    param = parameter_uncertainty(chi0, sigma_chi, sens_seeds, device,
                                  NOMINAL_LEVEL)
    soln = solution_verification(chi0, ens_seeds[0], device)
    valid = model_validation(chi0, ens_seeds[0], device)

    sigma_param = param["sigma_param"]
    sigma_stoch = stoch["sigma_stoch"]
    sigma_num = soln["discretization_uncertainty"]
    var = {"parameter (chi)": sigma_param ** 2,
           "stochastic (seed)": sigma_stoch ** 2,
           "numerical (mesh)": sigma_num ** 2}
    total_var = sum(var.values())
    sigma_total = float(np.sqrt(total_var))
    contributions = [
        {"source": k, "sigma": float(np.sqrt(v)),
         "variance": float(v),
         "pct_variance": float(100.0 * v / total_var) if total_var > 0 else 0.0}
        for k, v in var.items()]
    contributions.sort(key=lambda d: -d["variance"])
    dominant = contributions[0]["source"]

    return {
        "material": MATERIAL, "chi_param": CHI_PARAM,
        "chi_nominal": chi0, "chi_uncertainty_type": unc_type,
        "chi_sigma": sigma_chi,
        "observable": "L_area", "L_nominal": stoch["L_mean"],
        "code_verification": code,
        "solution_verification": soln,
        "model_validation": valid,
        "parameter_uncertainty": param,
        "stochastic_variability": stoch,
        "budget": {
            "contributions": contributions,
            "sigma_total": sigma_total,
            "relative_total": (sigma_total / stoch["L_mean"]
                               if stoch["L_mean"] else float("nan")),
            "dominant_source": dominant,
            "dominance_ratio": (contributions[0]["sigma"]
                                / max(contributions[1]["sigma"], 1e-30)),
        },
    }


def _print_budget(b, logger=print):
    logger(f"\n=== C10 uncertainty budget: {b['material']} / {b['observable']} "
           f"===")
    logger(f"material case: {b['material']}, chi = {b['chi_nominal']:.3f} "
           f"+/- {b['chi_sigma']:.3f} ({b['chi_uncertainty_type']})")
    logger(f"observable L_area (nominal) = {b['L_nominal']:.4f}")
    cv = b["code_verification"]
    logger(f"\n[1] code verification: mass_drift={cv['mass_drift']:.2e}, "
           f"energy_monotone={cv['energy_monotone']} "
           f"(largest +inc {cv['largest_positive_increment']:.1e})")
    sv = b["solution_verification"]
    logger(f"[2] solution verification: L by level "
           f"{ {k: round(v,4) for k,v in sv['L_by_level'].items()} }, "
           f"discretization unc = {sv['discretization_uncertainty']:.4f}")
    mv = b["model_validation"]
    logger(f"[3] model validation: lambda* pred {mv['lambda_star_predicted']:.4f}"
           f" vs measured {mv['lambda_measured']:.4f} "
           f"(rel discrepancy {mv['relative_discrepancy']:.1%})")
    pu = b["parameter_uncertainty"]
    logger(f"[4] parameter uncertainty: dL/dchi={pu['dL_dchi']:.3f}, "
           f"sigma_param={pu['sigma_param']:.4f}")
    st = b["stochastic_variability"]
    logger(f"[5] stochastic variability: n={st['n_seeds']}, "
           f"sigma_stoch={st['sigma_stoch']:.4f}")
    logger("[6] model-experiment discrepancy: not quantified (no lab data); "
           "structure in place -- see model_validation for the theory gap.")
    logger("\n--- uncertainty budget (variances in quadrature) ---")
    for c in b["budget"]["contributions"]:
        logger(f"  {c['source']:<20} sigma={c['sigma']:.4f}  "
               f"({c['pct_variance']:5.1f}% of variance)")
    logger(f"  {'TOTAL':<20} sigma={b['budget']['sigma_total']:.4f}  "
           f"({b['budget']['relative_total']:.1%} of the observable)")
    logger(f"  DOMINANT SOURCE: {b['budget']['dominant_source']} "
           f"({b['budget']['dominance_ratio']:.1f}x the next largest)")


def main():
    b = build_budget()
    _print_budget(b)


if __name__ == "__main__":
    main()
