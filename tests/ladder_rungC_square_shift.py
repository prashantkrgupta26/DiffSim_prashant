"""Projection Validation Ladder — Rung C driver: OFFSET square, GENUINE SBM
SHIFT (the THIRD and final scaffolded concept, in 2-D).

Rung A (strong Dirichlet, body-fitted square, dmax==0) PASSED and Rung B (weak
Nitsche, body-fitted square, dmax==0) PASSES with the consistent-projection
scheme + the FN4 rotational wall pin: the projection is faithful to the
same-mesh monolithic in velocity AND drag, at Re=40 and Re=100. Both rungs A/B
sit at `dmax==0`, where the SBM shifted-Nitsche form `S N_a = N_a + (grad N_a).d`
DEGENERATES to standard Nitsche (`d=0`, `geo.corr==1`) — the shift is inert.

Rung C OFFSETS the square center by a SUB-CELL fraction (`offset=delta`) so the
grid-aligned surrogate boundary no longer coincides with the true `Box` face:
`0 < dmax < h`. This ACTIVATES the genuine SBM Taylor shift — the `(grad N_a).d`
term in the Nitsche block AND the area-correction `geo.corr` in the surrogate
traction now do real work. Rung C tests whether ADDING the shift (on top of the
now-working projection+Nitsche) preserves faithfulness:

  * PROJECTION — the SAME consistent-projection stepper config that made rung B
    pass (`consistent_projection`, `rot_pin_wall=True`, grad-div gamma=50, the
    weak-Nitsche SBM face block + backflow + correction re-pin), but now on the
    OFFSET fixture so the SBM block carries `d != 0` (the Taylor shift) and the
    surrogate traction carries `geo.corr != 1` (the area correction). The
    drivers/steppers are UNCHANGED from rung B — rung C reuses `march_projection`
    / `march_monolithic` verbatim (the shift is data, not code: it lives in the
    fixture's `geo`).
  * MONOLITHIC — the SAME same-mesh saddle oracle, ALSO on the offset mesh with
    the SAME shifted `geo` (`sbm_vector_dirichlet` reads the same `d`/`corr`), so
    it is the same-mesh-WITH-SHIFT oracle. The primary, box-free bar.

Drag (Chenghau's guidance): use `surrogate_traction` (approach 2 —
surrogate-boundary + area-correction `oint shifted . (n~.n) dGamma~`) for BOTH
projection and monolithic, so the proj-vs-mono comparison is apples-to-apples on
the SAME shifted mesh with the SAME drag formula (the oracle is the monolithic
on the same offset mesh). At `d!=0` this GENUINELY applies `geo.corr` (unlike
rung A/B where corr==1). `Cd = F_x/qref`, `qref = 0.5 U_IN^2 D`, `D = 2*half`.

Anti-vacuity — the shift terms must be LOAD-BEARING at d!=0:
  1. SHIFT-ACTIVE guard: `dmax > 0` (the anti-vacuity OPPOSITE of rung A/B's
     `dmax==0`) — the shift is genuinely on.
  2. Zeroing the shift (`geo.d=0` AND `geo.corr=1`, i.e. dropping the Taylor
     `(grad N_a).d` term AND the area correction) BREAKS the proj-vs-mono match
     against the TRUE shifted oracle — the shift terms are actually doing work.
  3. Rung C ~= Rung B within the shift's consistency error: a SUB-CELL offset is
     nearly the same body, so the shift should be a small correction (not a
     regime change). Checked in the fast test.

If C passes: projection+SBM works end-to-end in 2-D. If it breaks: the defect
isolates to the shift (Taylor `(grad N_a).d`, surrogate normal `n~`, area
correction `geo.corr`, or their interaction with the wall-pressure pin).

Run (Mac / gpubox, CPU-splu — 2-D is small):
    PYTHONPATH=src:tests .venv/bin/python tests/ladder_rungC_square_shift.py
"""
import dataclasses

import numpy as np

from ladder_fixtures import build_square_channel_2d, U_IN
# Rung C REUSES the rung-B marchers verbatim — the shift is fixture DATA (geo.d,
# geo.corr), not solver code, so the SAME consistent-projection + weak-Nitsche +
# FN4-wall-pin path exercises it. Reuse guarantees the proj-vs-mono comparison
# is on the identical solver as rung B (only the mesh/geo differs: offset != 0).
from ladder_rungB_square_nitsche import (
    march_projection, march_monolithic, ALPHA, FN1_GRADDIV_GAMMA,
    TOL_CD_REL, TOL_MU_REL, WEAK_PLATEAU_FRAC)
# rung A shares the qref / mean_speed / strouhal definitions — reuse verbatim so
# the C-vs-B comparison is on identical observables.
from ladder_rungA_square_strong import qref, mean_speed, strouhal

# Sub-cell offset: 0.05 shifts the center 0.05 to the right of the aligned 0.5.
# At level 4 (h=0.0625) this yields dmax = 0.05 (dmax/h = 0.80) — a genuine
# sub-cell Taylor shift with 0 < dmax < h, plus a NON-trivial area correction
# (geo.corr drops the falsely-intersected surrogate faces). Sub-cell so the
# physics is nearly the same body (rung C ~= rung B), yet large enough that the
# shift terms are UNMISTAKABLY load-bearing: at 0.05 the true proj-vs-mono match
# is ~0.3% while ZEROING the shift throws the projection ~36% off the true
# shifted oracle (measured L4 Re=40) — a decisive anti-vacuity margin (the Taylor
# (grad N).d term alone ~4%, the area correction the rest).
DEFAULT_OFFSET = 0.05


# --------------------------------------------------------------------------
# shift-zeroing (anti-vacuity: drop the Taylor + area-correction shift terms)
# --------------------------------------------------------------------------
def zero_shift(fx):
    """Return a COPY of the fixture with the SBM shift terms ZEROED: `geo.d=0`
    (drops the Taylor `(grad N_a).d` in the Nitsche block AND the field
    extrapolation) and `geo.corr=1` (drops the surrogate-face area correction).
    Everything else (the surrogate faces, the surrogate normal `geo.n`, the
    mesh, the BC masks) is IDENTICAL — so the ONLY difference vs the true
    fixture is the shift. Used for the anti-vacuity load-bearing check: running
    the projection/monolithic with this `geo` and comparing against the TRUE
    shifted oracle must FAIL (the shift is doing real work at d!=0)."""
    geo = fx["geo"]
    geo0 = dataclasses.replace(
        geo, d=np.zeros_like(geo.d), corr=np.ones_like(geo.corr))
    fx0 = dict(fx)
    fx0["geo"] = geo0
    fx0["dmax"] = 0.0
    return fx0


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------
def run_rungC(level=5, half=0.125, offset=DEFAULT_OFFSET, res=(40, 100),
              device="cpu", dt40=0.02, nsteps40=600, dt100=0.01,
              nsteps100=2000, log_every=25, alpha=ALPHA, solver="splu"):
    """Full rung-C driver. Re=40 steady (PRIMARY): projection (weak Nitsche +
    SHIFT) vs monolithic (weak Nitsche + SHIFT, same offset mesh) steady Cd +
    mean|u|. Re=100: mean Cd + Strouhal. Prints PASS/FAIL vs the same-mesh
    (same-shift) oracle.

    THE decisive read: does the consistent-projection split with the GENUINE
    SBM shift active (`dmax>0`) still match the same-mesh-WITH-SHIFT weak-Nitsche
    monolithic Cd AND develop mean|u| — i.e. does adding the Taylor shift + area
    correction preserve the faithfulness that rung B established?"""
    results = {}
    all_pass = True
    D = 2.0 * half
    for Re in res:
        steady = (Re < 50)
        dt = dt40 if steady else dt100
        nsteps = nsteps40 if steady else nsteps100
        rate_tol_p = 5e-4 if steady else None
        rate_tol_m = 2e-4 if steady else None

        print(f"\n{'='*70}\n Rung C — OFFSET square SBM SHIFT  "
              f"Re={Re}  level={level}  half={half} (D={D})  offset={offset}  "
              f"alpha={alpha}\n{'='*70}", flush=True)

        fx = build_square_channel_2d(level, Re, half=half, offset=offset,
                                     device=device)
        dmax = fx["dmax"]
        h = 1.0 / 2 ** level
        # SHIFT-ACTIVE GUARD (anti-vacuity, the OPPOSITE of rung A/B): a genuine
        # sub-cell SBM shift is on. 0 < dmax < h => the Taylor (grad N).d term
        # and the area correction geo.corr are actually load-bearing.
        assert dmax > 0, (f"rung C requires a GENUINE SBM shift (dmax>0); got "
                          f"dmax={dmax} (offset={offset} did not break the "
                          f"body-fitted alignment).")
        assert dmax < h, (f"rung C wants a SUB-CELL shift (dmax<h={h}); got "
                          f"dmax={dmax} — offset too large.")
        corr = np.asarray(fx["geo"].corr)
        n_corr = int((np.abs(corr - 1.0) > 1e-9).sum())
        print(f" shift-active guard: dmax={dmax:.5f}  (0<dmax<h={h:.5f}, "
              f"dmax/h={dmax/h:.3f})  n_area_corrected_gp={n_corr}  "
              f"n_fluid_cells={fx['n_fluid_cells']}  "
              f"n_obstacle_nodes={int(fx['obstacle_node_mask'].sum())}",
              flush=True)

        mono_beta = 0.5
        mono_bvs = True
        # SAME config as rung B's verdict (FN4 rotational wall pin), UNCHANGED —
        # the shift enters only via the fixture geo (d!=0, corr!=1).
        pr = march_projection(fx, dt=dt, nsteps=nsteps, rate_tol=rate_tol_p,
                              log_every=log_every, alpha=alpha, solver=solver,
                              rot_pin_wall=True)
        mo = march_monolithic(fx, dt=dt, nsteps=nsteps, rate_tol=rate_tol_m,
                              log_every=log_every, backflow_beta=mono_beta,
                              boundary_vorticity=mono_bvs, alpha=alpha)

        cd_rel = (abs(pr["cd"] - mo["cd"]) / abs(mo["cd"])
                  if mo["cd"] != 0 else float("inf"))
        mu_rel = (abs(pr["mean_u"] - mo["mean_u"]) / abs(mo["mean_u"])
                  if mo["mean_u"] != 0 else float("inf"))
        mu_frac = pr["mean_u"] / mo["mean_u"] if mo["mean_u"] != 0 else 0.0
        weak_pin = mu_frac < WEAK_PLATEAU_FRAC
        blew = pr.get("blew_up", False)

        if steady:
            cd_ok = (not blew) and cd_rel < TOL_CD_REL
            mu_ok = (not blew) and (mu_rel < TOL_MU_REL) and (not weak_pin)
            verdict = cd_ok and mu_ok
            print(f"\n --- Re={Re} STEADY verdict ---")
            print(f"  projection : Cd={pr['cd']:+.4f}  mean|u|={pr['mean_u']:.4f}"
                  f"  ‖div‖={pr['div']:.3e}  ‖p‖={pr['pnorm']:.3e}  "
                  f"steps={pr['steps']}")
            print(f"  monolithic : Cd={mo['cd']:+.4f}  mean|u|={mo['mean_u']:.4f}"
                  f"  ‖div‖={mo['div']:.3e}  steps={mo['steps']}")
            print(f"  Cd rel-diff = {cd_rel:.3%} (tol {TOL_CD_REL:.0%})  "
                  f"-> {'OK' if cd_ok else 'FAIL'}")
            print(f"  mean|u| proj/mono = {mu_frac:.3f}  rel-diff={mu_rel:.3%} "
                  f"(tol {TOL_MU_REL:.0%})  weak_pin={weak_pin}  "
                  f"-> {'OK' if mu_ok else 'FAIL'}")
        else:
            def tail_mean(hh):
                a = np.asarray(hh)
                return float(a[int(0.4 * len(a)):].mean())
            pr_cdm = tail_mean(pr["cd_hist"])
            mo_cdm = tail_mean(mo["cd_hist"])
            cd_rel = (abs(pr_cdm - mo_cdm) / abs(mo_cdm)
                      if mo_cdm != 0 else float("inf"))
            st_p, _ = strouhal(pr["cl_hist"], dt, D)
            st_m, _ = strouhal(mo["cl_hist"], dt, D)
            cd_ok = (not blew) and cd_rel < TOL_CD_REL
            mu_ok = (not blew) and (mu_rel < TOL_MU_REL) and (not weak_pin)
            verdict = cd_ok and mu_ok
            pr["cd_mean_period"], mo["cd_mean_period"] = pr_cdm, mo_cdm
            pr["St"], mo["St"] = st_p, st_m
            print(f"\n --- Re={Re} SHEDDING verdict ---")
            print(f"  projection : mean Cd={pr_cdm:+.4f}  St={st_p:.4f}  "
                  f"mean|u|={pr['mean_u']:.4f}  ‖div‖={pr['div']:.3e}  "
                  f"steps={pr['steps']}")
            print(f"  monolithic : mean Cd={mo_cdm:+.4f}  St={st_m:.4f}  "
                  f"mean|u|={mo['mean_u']:.4f}  ‖div‖={mo['div']:.3e}  "
                  f"steps={mo['steps']}")
            print(f"  mean-Cd rel-diff = {cd_rel:.3%} (tol {TOL_CD_REL:.0%})  "
                  f"-> {'OK' if cd_ok else 'FAIL'}")
            print(f"  mean|u| proj/mono = {mu_frac:.3f}  weak_pin={weak_pin}  "
                  f"-> {'OK' if mu_ok else 'FAIL'}")

        all_pass = all_pass and verdict
        print(f" ===> Re={Re}  {'PASS' if verdict else 'FAIL'}", flush=True)
        results[Re] = dict(
            cd_proj=pr["cd"], cd_mono=mo["cd"], cd_rel=cd_rel,
            mean_u_proj=pr["mean_u"], mean_u_mono=mo["mean_u"],
            mu_frac=mu_frac, weak_pin=weak_pin,
            div_proj=pr["div"], div_mono=mo["div"],
            steps_proj=pr["steps"], steps_mono=mo["steps"],
            St_proj=pr.get("St"), St_mono=mo.get("St"),
            cd_mean_period_proj=pr.get("cd_mean_period"),
            cd_mean_period_mono=mo.get("cd_mean_period"),
            pnorm_proj=pr["pnorm"], blew_up=blew, dmax=dmax, verdict=verdict)

    print(f"\n{'#'*70}\n RUNG C VERDICT: "
          f"{'PASS — the consistent-projection split WITH the genuine SBM shift (dmax>0) matches the same-mesh-with-shift weak-Nitsche monolithic: projection+SBM works end-to-end in 2-D.' if all_pass else 'FAIL — the projection with the SBM shift does NOT match the same-shifted-mesh monolithic; the defect isolates to the shift.'}"
          f"\n{'#'*70}", flush=True)
    results["all_pass"] = all_pass
    return results


if __name__ == "__main__":
    import json
    import os
    import sys
    lvl = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    res = (40,) if (len(sys.argv) > 2 and sys.argv[2] == "re40") else (40, 100)
    solver = os.environ.get("PROJ_SOLVER", "splu")
    device = os.environ.get("DEVICE", "cpu" if solver == "splu" else "cuda:0")
    out = run_rungC(level=lvl, res=res, solver=solver, device=device)
    print("\n[json]", json.dumps(out, default=lambda o: None))
