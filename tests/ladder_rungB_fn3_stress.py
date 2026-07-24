"""Projection Validation Ladder — FN3 rung-B STRESS TEST (validation gate).

Rung B (weak Nitsche no-slip, body-fitted square) is stable and VELOCITY-faithful
after FN1. The ONLY known gap is the wall PRESSURE (drag Cd). Before adding the
consistent Neumann wall-pressure BC, FN3 confirms the REST of the formulation is
robust — everything EXCEPT the near-wall pressure/drag — across:

  AXIS 1  Reynolds sweep {1,10,20,40,100,200}: projection vs same-mesh monolithic
          mean|u|, ‖div‖, and the INTERIOR pressure field (a near-wall band around
          the obstacle EXCLUDED — that band is the known-bad part). Plus a γ probe:
          is the FN1 constant grad-div (γ=50, "scales with Re") robust across Re,
          or does a τ_C-based (VMS-continuity) grad-div, Re-independent by
          construction, hold fidelity without per-Re tuning?

  AXIS 2  Uniform P2 (equal-order P2 velocity+pressure + PSPG): does the full
          stepper (SBM Nitsche + PPE + PSPG + correction) RUN at p=2, and does
          projection track the monolithic at P2?

  AXIS 3  Variable-p P2 BAND around the obstacle + P1 far field (EQUAL ORDER
          within each element — PSPG stays on). Constructible via build_mesh's
          per-element p_elem + mixed-degree DeviceMesh + p-hanging constraints.
          Does projection track the monolithic, and does near-body field fidelity
          improve vs uniform P1?

This driver does NOT change the stepper numerics; it only adds probes and a
τ_C-grad-div comparison block. Defaults bit-for-bit.

Run (CPU-splu — 2-D is small):
    PYTHONPATH=src:tests python tests/ladder_rungB_fn3_stress.py [axis]
    axis in {re, gamma, p2, band, all}  (default: all)
"""
import numpy as np
import scipy.sparse as sp

import ladder_fixtures as LF
from ladder_fixtures import build_square_channel_2d
from ladder_rungB_square_nitsche import (march_projection, march_monolithic,
                                          mean_speed, qref, ALPHA,
                                          FN1_GRADDIV_GAMMA)
from diffsim.steppers.leray import LerayProjectionStepper


# ------------------------------------------------------------------------
# interior-pressure comparison (exclude a near-wall band around the obstacle)
# ------------------------------------------------------------------------
def _node_pressure(fx, x_full):
    """Full node-major (u,p) -> per-node pressure at the FREE nodes, and the
    free-node coords. x_full is dm.n_nodes*ndof (the drivers' x_full)."""
    dm, ndof, dim = fx["dm"], fx["ndof"], fx["dim"]
    xf = np.asarray(x_full).reshape(dm.n_nodes, ndof)
    p_full = xf[:, dim]
    free = dm.constraints.free_nodes
    return p_full[free], fx["coords"]


def interior_pressure_rel(fx, x_proj, x_mono, band=None):
    """Relative L2 difference of the INTERIOR pressure field (projection vs
    monolithic), EXCLUDING a near-wall band around the obstacle and gauge-shifted
    (pressure is defined up to a constant; both fields are de-meaned over the
    interior set before comparison). `band` defaults to ~2 cell widths.

    Returns (rel_l2_interior, rel_l2_all, n_interior, n_wallband)."""
    half, center = fx["half"], np.asarray(fx["center"])
    coords = fx["coords"]
    h = 1.0 / (2 ** _mesh_level(fx))          # far-field cell width
    if band is None:
        band = 2.5 * h
    p_p, _ = _node_pressure(fx, x_proj)
    p_m, _ = _node_pressure(fx, x_mono)
    # Chebyshev distance to the square surface (0 on the wall, >0 outside).
    d_surf = np.maximum(np.abs(coords - center[None, :]).max(1) - half, 0.0)
    interior = d_surf > band
    def rel(mask):
        if mask.sum() < 4:
            return float("nan")
        a = p_p[mask] - p_p[mask].mean()
        b = p_m[mask] - p_m[mask].mean()
        den = np.linalg.norm(b)
        return float(np.linalg.norm(a - b) / den) if den > 0 else float("nan")
    return (rel(interior), rel(np.ones_like(interior)),
            int(interior.sum()), int((~interior).sum()))


def interior_pressure_abs(fx, x_proj, x_mono, band=None):
    """ABSOLUTE interior-pressure diagnostics (the interpretable companion to
    the relative metric, which inflates at low Re where the pressure SIGNAL is
    tiny). Returns (abs_l2_diff, mono_signal_rms, proj_signal_rms, n_interior):
    both fields de-meaned over the interior set; RMS = L2/sqrt(n)."""
    half, center = fx["half"], np.asarray(fx["center"])
    coords = fx["coords"]
    h = 1.0 / (2 ** _mesh_level(fx))
    if band is None:
        band = 2.5 * h
    p_p, _ = _node_pressure(fx, x_proj)
    p_m, _ = _node_pressure(fx, x_mono)
    d_surf = np.maximum(np.abs(coords - center[None, :]).max(1) - half, 0.0)
    m = d_surf > band
    n = int(m.sum())
    if n < 4:
        return float("nan"), float("nan"), float("nan"), n
    a = p_p[m] - p_p[m].mean()
    b = p_m[m] - p_m[m].mean()
    rn = np.sqrt(n)
    return (float(np.linalg.norm(a - b)), float(np.linalg.norm(b) / rn),
            float(np.linalg.norm(a) / rn), n)


def _mesh_level(fx):
    # infer level from the uniform far-field cell size
    h = fx["dm"].mesh.tree.h().max()
    return int(round(np.log2(1.0 / h)))


# ------------------------------------------------------------------------
# τ_C (VMS continuity fine-scale) grad-div block — the Re-independent
# alternative to the FN1 constant-γ grad-div. Built with velocity_update=
# "graddiv" so the stepper assembles its cached τ_C block, then extracted so
# BOTH the projection predictor and the monolithic saddle carry the SAME block
# (term-for-term fair, exactly as march_monolithic does for the constant-γ one).
# ------------------------------------------------------------------------
def build_tauC_block(fx, dt, graddiv_scale=1.0):
    dm, nu, dim = fx["dm"], fx["nu"], fx["dim"]
    st = LerayProjectionStepper(
        dm, nu, dt, f_fn=lambda x, t: np.zeros((len(x), dim)),
        g_fn=lambda c, t: None, order=1, velocity_update="graddiv",
        graddiv_scale=graddiv_scale)
    return st._graddiv_block


# ------------------------------------------------------------------------
# AXIS 1 — Reynolds sweep
# ------------------------------------------------------------------------
RE_SWEEP = (1, 10, 20, 40, 100, 200)


def _dt_nsteps(Re):
    """Per-Re marching params: low/moderate Re -> steady, high Re -> longer."""
    if Re <= 20:
        return 0.02, 500, 5e-4
    if Re < 50:
        return 0.02, 600, 5e-4
    if Re < 150:
        return 0.01, 1500, None
    return 0.008, 2000, None


def run_re_sweep(level=5, half=0.125, res=RE_SWEEP, alpha=ALPHA,
                 gamma=FN1_GRADDIV_GAMMA, device="cpu", band=None):
    print(f"\n{'#'*72}\n# FN3 AXIS 1 — Reynolds sweep (γ={gamma})  "
          f"level={level} half={half}\n{'#'*72}", flush=True)
    rows = []
    for Re in res:
        dt, nsteps, rtol = _dt_nsteps(Re)
        fx = build_square_channel_2d(level, Re, half=half, offset=0,
                                     device=device)
        assert fx["dmax"] == 0
        pr = march_projection(fx, dt=dt, nsteps=nsteps, rate_tol=rtol,
                              alpha=alpha, graddiv_gamma=gamma)
        mo = march_monolithic(fx, dt=dt, nsteps=nsteps,
                              rate_tol=(2e-4 if rtol else None),
                              backflow_beta=0.5, boundary_vorticity=True,
                              alpha=alpha, graddiv_gamma=gamma)
        mu_p, mu_m = pr["mean_u"], mo["mean_u"]
        mu_rel = abs(mu_p - mu_m) / abs(mu_m) if mu_m else float("nan")
        p_int, p_all, n_int, n_wb = (float("nan"),) * 2 + (0, 0)
        if pr.get("x_full") is not None and mo.get("x_full") is not None:
            p_int, p_all, n_int, n_wb = interior_pressure_rel(
                fx, pr["x_full"], mo["x_full"], band=band)
        row = dict(Re=Re, mu_p=mu_p, mu_m=mu_m, mu_rel=mu_rel,
                   div_p=pr["div"], div_m=mo["div"],
                   p_int_rel=p_int, p_all_rel=p_all, n_int=n_int, n_wb=n_wb,
                   blew=pr.get("blew_up", False),
                   steps_p=pr["steps"], steps_m=mo["steps"])
        rows.append(row)
        print(f" Re={Re:4d} | mean|u| p={mu_p:.4f} m={mu_m:.4f} rel={mu_rel:6.2%}"
              f" | ‖div‖ p={pr['div']:.2e} m={mo['div']:.2e}"
              f" | p_int_rel={p_int:6.2%} (excl {n_wb} wall,{n_int} int)"
              f" | p_all_rel={p_all:6.2%}"
              f" | blew={row['blew']}", flush=True)
    return rows


# ------------------------------------------------------------------------
# AXIS 1b — γ robustness probe: fixed Re, sweep γ; and τ_C (Re-independent).
# ------------------------------------------------------------------------
def run_gamma_probe(level=5, half=0.125, device="cpu",
                    gammas=(0.0, 20.0, 50.0, 100.0), re_list=(40, 100)):
    print(f"\n{'#'*72}\n# FN3 AXIS 1b — γ robustness probe (const-γ sweep + τ_C)"
          f"\n{'#'*72}", flush=True)
    rows = []
    for Re in re_list:
        dt, nsteps, rtol = _dt_nsteps(Re)
        fx = build_square_channel_2d(level, Re, half=half, offset=0,
                                     device=device)
        mo = march_monolithic(fx, dt=dt, nsteps=nsteps,
                              rate_tol=(2e-4 if rtol else None),
                              backflow_beta=0.5, boundary_vorticity=True,
                              alpha=ALPHA, graddiv_gamma=FN1_GRADDIV_GAMMA)
        print(f"\n -- Re={Re} (mono mean|u|={mo['mean_u']:.4f}) --", flush=True)
        for g in gammas:
            fx2 = build_square_channel_2d(level, Re, half=half, offset=0,
                                          device=device)
            pr = march_projection(fx2, dt=dt, nsteps=nsteps, rate_tol=rtol,
                                  alpha=ALPHA, graddiv_gamma=(g if g else None))
            mu_rel = (abs(pr["mean_u"] - mo["mean_u"]) / abs(mo["mean_u"])
                      if mo["mean_u"] else float("nan"))
            p_int = float("nan")
            if pr.get("x_full") is not None:
                p_int = interior_pressure_rel(fx2, pr["x_full"],
                                              mo["x_full"])[0]
            rows.append(dict(Re=Re, kind=f"const-γ={g}", mu_rel=mu_rel,
                             div=pr["div"], p_int=p_int,
                             blew=pr.get("blew_up", False)))
            print(f"   const γ={g:6.1f} | mean|u|={pr['mean_u']:.4f} "
                  f"rel={mu_rel:6.2%} | ‖div‖={pr['div']:.2e} "
                  f"| p_int_rel={p_int:6.2%} | blew={pr.get('blew_up',False)}",
                  flush=True)
    return rows


# ------------------------------------------------------------------------
# AXIS 2 — uniform P2 (equal order) ; AXIS 3 — P2 band + P1 far field
# ------------------------------------------------------------------------
def _fx(level, Re, half, device, **kw):
    return LF._build_channel(level, Re, half, 0, device, dim=2,
                             center=(0.5, 0.5), **kw)


def run_order_axis(level=5, half=0.125, device="cpu", re_list=(40, 100),
                   p2_band=2):
    print(f"\n{'#'*72}\n# FN3 AXIS 2/3 — uniform P2 + variable-p P2-band"
          f"\n{'#'*72}", flush=True)
    rows = []
    for Re in re_list:
        dt, nsteps, rtol = _dt_nsteps(Re)
        configs = [("P1-uniform", dict(p=1)),
                   ("P2-uniform", dict(p=2)),
                   (f"P2-band({p2_band})", dict(p2_band=p2_band))]
        for name, kw in configs:
            fx = _fx(level, Re, half, device, **kw)
            assert fx["dmax"] == 0
            bins = list(fx["dm"].bins.keys())
            npe2 = int((np.asarray(fx["mesh"].p_elem) == 2).sum())
            pr = march_projection(fx, dt=dt, nsteps=nsteps, rate_tol=rtol,
                                  alpha=ALPHA)
            mo = march_monolithic(fx, dt=dt, nsteps=nsteps,
                                  rate_tol=(2e-4 if rtol else None),
                                  backflow_beta=0.5, boundary_vorticity=True,
                                  alpha=ALPHA)
            mu_rel = (abs(pr["mean_u"] - mo["mean_u"]) / abs(mo["mean_u"])
                      if mo["mean_u"] else float("nan"))
            p_int = p_wall = float("nan")
            if pr.get("x_full") is not None and mo.get("x_full") is not None:
                p_int, p_all, n_int, n_wb = interior_pressure_rel(
                    fx, pr["x_full"], mo["x_full"])
            rows.append(dict(Re=Re, config=name, bins=bins, n_p2=npe2,
                             nfree=len(fx["coords"]),
                             mu_p=pr["mean_u"], mu_m=mo["mean_u"],
                             mu_rel=mu_rel, div_p=pr["div"], div_m=mo["div"],
                             p_int_rel=p_int, blew=pr.get("blew_up", False)))
            print(f" Re={Re:4d} {name:14s} bins={bins} n_p2={npe2:4d} "
                  f"nfree={len(fx['coords']):5d} | mean|u| p={pr['mean_u']:.4f} "
                  f"m={mo['mean_u']:.4f} rel={mu_rel:6.2%} | "
                  f"‖div‖ p={pr['div']:.2e} m={mo['div']:.2e} | "
                  f"p_int_rel={p_int:6.2%} | blew={pr.get('blew_up',False)}",
                  flush=True)
    return rows


if __name__ == "__main__":
    import json
    import sys
    axis = sys.argv[1] if len(sys.argv) > 1 else "all"
    lvl = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    out = {}
    if axis in ("re", "all"):
        out["re_sweep"] = run_re_sweep(level=lvl)
    if axis in ("gamma", "all"):
        out["gamma_probe"] = run_gamma_probe(level=lvl)
    if axis in ("p2", "band", "order", "all"):
        out["order_axis"] = run_order_axis(level=lvl)
    print("\n[json]", json.dumps(out, default=lambda o: (
        list(o) if isinstance(o, (np.ndarray,)) else None)))
