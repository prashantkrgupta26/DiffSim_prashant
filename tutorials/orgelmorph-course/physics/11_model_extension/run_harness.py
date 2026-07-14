"""P11 — vitrification / composition-dependent mobility arrest: the harness driver.

Drives the self-contained 1-D Cahn-Hilliard vitrification brick
(``vitrification.py``) through the OrgElMorph course harness, EXACTLY like
``physics/05_evaporation/run_harness.py``: a YAML config
(``configs/p11.yaml``), a provenance ``metadata.json``, a ``results.json``
checked against ``baseline.yaml``, and the standard output layout.

The run performs FIVE verifications plus the scientific result, and records all
gated scalars under a flat, DOT-FREE ``results["checks"]`` block:

  1. analytic-derivative check   f', f'' vs central differences of f;
  2. Jacobian FD-check           hand-derived analytic dR/dphi vs a column-wise
                                 central-difference Jacobian (THE contributor
                                 workflow gate) + an autograd cross-check;
  3. limiting-case dispersion    arrest off (phi_g -> inf, constant mobility):
                                 measured single-mode CH growth rate vs the
                                 analytic sigma(k) = -M0 k^2 (f''(phi_bar)+kappa k^2);
  4. MMS spatial convergence     manufactured phi*(x,t) with the exact analytic
                                 source -> ~2nd-order (h^2) L2 convergence;
  5. temporal order              backward Euler is 1st order: halving dt halves
                                 the time error vs a fine-dt reference (~rate 1);
  6. coarsening / vitrification  random IC coarsens WITHOUT arrest (L grows) but
                                 HALTS with the polymer-rich matrix vitrified
                                 (phi_g inside the demixing gap) -> L plateaus,
                                 L_final(free) > L_final(arrest) by a clear margin.

    PYTHONPATH=<repo>/src python run_harness.py --config configs/p11.yaml \\
        --mode reference --output outputs/p11 --overwrite
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                os.pardir, os.pardir)))
from common import config as cfgmod                       # noqa: E402
from common.run_base import build_parser, run_tutorial     # noqa: E402

import vitrification as V                                  # noqa: E402


SCHEMA = cfgmod.ConfigSchema(name="p11", fields={
    "N": cfgmod.Field(int, default=128, min=8),         # coarsening grid size
    "L": cfgmod.Field(float, default=1.0, min=0.0),     # periodic box length
    "kappa": cfgmod.Field(float, default=3.0e-4, min=0.0),   # gradient energy
    "M0": cfgmod.Field(float, default=1.0, min=0.0),    # mobility prefactor
    "phi_g": cfgmod.Field(float, default=0.7, min=0.0), # glass/arrest composition
    "w": cfgmod.Field(float, default=0.05, min=1e-6),   # arrest transition width
    "dt": cfgmod.Field(float, default=1.0e-3, min=0.0), # coarsening time step
    "nsteps": cfgmod.Field(int, default=300, min=1),    # coarsening steps
    "seed": cfgmod.Field(int, default=11),
    "precision": cfgmod.Field(str, default="fp64"),
})


# --------------------------------------------------------------------------
# verification sub-routines (each returns scalars + stashes figure arrays)
# --------------------------------------------------------------------------
def _np(t):
    """Torch tensor (any device) or array -> host float64 numpy array."""
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _grid(N, L, device):
    h = L / N
    x = torch.arange(N, dtype=V.DTYPE, device=device) * h
    return x, h


def check_derivatives(device):
    """(a) f', f'' vs central differences of f (tol ~1e-6)."""
    phi = torch.linspace(0.02, 0.98, 40, dtype=V.DTYPE, device=device)
    eps = 1e-6
    fp_cd = (V.f(phi + eps) - V.f(phi - eps)) / (2 * eps)
    fpp_cd = (V.fprime(phi + eps) - V.fprime(phi - eps)) / (2 * eps)
    return {
        "fprime_cd_err": float((V.fprime(phi) - fp_cd).abs().max()),
        "fpp_cd_err": float((V.fpp(phi) - fpp_cd).abs().max()),
    }


def check_jacobian(cfg, device):
    """(b) Hand-derived analytic Jacobian vs a column-wise central-difference
    Jacobian on a random smooth state (+ an autograd cross-check)."""
    N = 16
    L = cfg["L"]
    x, h = _grid(N, L, device)
    kappa, M0, phi_g, w, dt = (cfg["kappa"], cfg["M0"], cfg["phi_g"],
                               cfg["w"], cfg["dt"])
    g = torch.Generator(device=device).manual_seed(cfg["seed"])
    phi = (0.5 + 0.18 * torch.sin(2 * math.pi * x / L)
           + 0.05 * torch.cos(6 * math.pi * x / L)
           + 0.02 * torch.randn(N, generator=g, dtype=V.DTYPE, device=device))
    phi_old = phi.clone()
    Jana = V.jacobian(phi, dt, kappa, M0, phi_g, w, h)
    Jfd = torch.zeros_like(Jana)
    eps = 1e-6
    for j in range(N):
        dp = torch.zeros(N, dtype=V.DTYPE, device=device)
        dp[j] = eps
        Rp = V.residual(phi + dp, phi_old, dt, kappa, M0, phi_g, w, h)
        Rm = V.residual(phi - dp, phi_old, dt, kappa, M0, phi_g, w, h)
        Jfd[:, j] = (Rp - Rm) / (2 * eps)
    fd_err = float((Jana - Jfd).abs().max())
    Jag = torch.autograd.functional.jacobian(
        lambda p: V.residual(p, phi_old, dt, kappa, M0, phi_g, w, h),
        phi.clone())
    ag_err = float((Jana - Jag).abs().max())
    return {"jac_fd_max_err": fd_err, "jac_autograd_max_err": ag_err,
            "jac_max_entry": float(Jana.abs().max())}


def check_dispersion(cfg, ctx, device):
    """(c) LIMITING CASE: arrest off (phi_g huge -> constant mobility). Measure
    the single-mode CH growth rate about phi_bar=0.5 and compare to the analytic
    dispersion sigma(k) = -M0 k^2 (f''(phi_bar) + kappa k^2).  Also sweep several
    modes to store a discrete-vs-continuous dispersion CURVE for the figure."""
    N = 64
    L = cfg["L"]
    x, h = _grid(N, L, device)
    kappa, M0, w = cfg["kappa"], cfg["M0"], cfg["w"]
    phi_g = 1.0e6                       # arrest OFF -> M -> M0 constant
    phi_bar = 0.5
    A = 1.0e-3
    dt = 1.0e-4
    fpp_bar = float(V.fpp(torch.tensor(phi_bar, dtype=V.DTYPE)))

    ks, sig_ana, sig_meas = [], [], []
    for n in range(1, 7):
        q = 2.0 * math.pi * n / L
        phi0 = phi_bar + A * torch.sin(q * x)
        phi1, _ = V.step(phi0, dt, kappa, M0, phi_g, w, h)
        proj0 = float((phi0 * torch.sin(q * x)).sum() * 2 / N)
        proj1 = float((phi1 * torch.sin(q * x)).sum() * 2 / N)
        s_ana = -M0 * q * q * (fpp_bar + kappa * q * q)
        s_meas = math.log(proj1 / proj0) / dt   # growth rate from amplitude
        ks.append(q); sig_ana.append(s_ana); sig_meas.append(s_meas)

    ctx.history["disp_k"] = np.array(ks)
    ctx.history["disp_sigma_analytic"] = np.array(sig_ana)
    ctx.history["disp_sigma_measured"] = np.array(sig_meas)
    rel0 = abs(sig_meas[0] - sig_ana[0]) / abs(sig_ana[0])   # fundamental mode
    return {"dispersion_rel_err": float(rel0),
            "dispersion_sigma_analytic": float(sig_ana[0]),
            "dispersion_sigma_measured": float(sig_meas[0])}


def check_mms(cfg, ctx, device):
    """(d) MMS spatial convergence: manufactured phi*(x,t) = phi_bar + A sin(qx)
    e^{-lam t} with the EXACT analytic continuous source added to the residual;
    the discrete L2 error vs phi* must fall ~4x per mesh doubling (2nd order)."""
    L = cfg["L"]
    kappa, M0, phi_g, w = cfg["kappa"], cfg["M0"], cfg["phi_g"], cfg["w"]
    phi_bar, A, lam = 0.5, 0.1, 1.0
    q = 2.0 * math.pi / L
    dt, nsteps = 1.0e-5, 4                  # tiny dt -> spatial error dominates
    meshes = [64, 128, 256] if cfg["N"] >= 256 else [32, 64, 128]

    hs, errs = [], []
    for N in meshes:
        x, h = _grid(N, L, device)
        phi = V.mms_phi(x, 0.0, phi_bar, A, q, lam)
        for n in range(nsteps):
            tn = (n + 1) * dt
            src = V.mms_source(x, tn, phi_bar, A, q, lam, kappa, M0, phi_g, w)
            phi, _ = V.step(phi, dt, kappa, M0, phi_g, w, h, source=src)
        T = nsteps * dt
        exact = V.mms_phi(x, T, phi_bar, A, q, lam)
        errs.append(float(torch.sqrt(((phi - exact) ** 2).mean())))
        hs.append(h)
    rates = [math.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    ctx.history["mms_N"] = np.array(meshes)
    ctx.history["mms_h"] = np.array(hs)
    ctx.history["mms_err"] = np.array(errs)
    ctx.history["mms_rates"] = np.array(rates)
    return {"mms_order_min": float(min(rates)),
            "mms_order_mean": float(np.mean(rates)),
            "mms_err_coarse": float(errs[0]), "mms_err_fine": float(errs[-1])}


def check_temporal(cfg, ctx, device):
    """(e) Temporal order: backward Euler is 1st order in dt. On a STABLE
    (phi_bar=0.15) constant-mobility decaying state, halving dt ~halves the time
    error vs a fine-dt reference; report the observed rate (~1)."""
    N = 48
    L = cfg["L"]
    x, h = _grid(N, L, device)
    kappa, M0, w = cfg["kappa"], cfg["M0"], cfg["w"]
    phi_g = 1.0e6
    g = torch.Generator(device=device).manual_seed(cfg["seed"] + 1)
    phi0 = (0.15 + 0.05 * torch.sin(2 * math.pi * x / L)
            + 0.02 * torch.cos(4 * math.pi * x / L)
            + 0.01 * torch.randn(N, generator=g, dtype=V.DTYPE, device=device))
    T = 0.05

    def run_dt(nst):
        dt = T / nst
        phi = phi0.clone()
        for _ in range(nst):
            phi, _ = V.step(phi, dt, kappa, M0, phi_g, w, h)
        return phi

    ref = run_dt(2048)
    nsts = [16, 32, 64, 128]
    errs = [float(torch.sqrt(((run_dt(n) - ref) ** 2).mean())) for n in nsts]
    rates = [math.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    ctx.history["temporal_dt"] = np.array([T / n for n in nsts])
    ctx.history["temporal_err"] = np.array(errs)
    ctx.history["temporal_rates"] = np.array(rates)
    return {"temporal_order": float(np.mean(rates)),
            "temporal_order_min": float(min(rates))}


def check_coarsening(cfg, ctx, device):
    """(f) SCIENTIFIC RESULT. Random IC at the polymer-rich glass composition
    (mean = phi_g).  WITHOUT arrest (phi_g -> inf) the domains coarsen and L(t)
    grows; WITH the arrested matrix (phi_g inside the demixing gap) coarsening
    HALTS -> L(t) plateaus and L_final(arrest) << L_final(free)."""
    N, L = cfg["N"], cfg["L"]
    x, h = _grid(N, L, device)
    kappa, M0, phi_g, w = cfg["kappa"], cfg["M0"], cfg["phi_g"], cfg["w"]
    dt, nsteps = cfg["dt"], cfg["nsteps"]
    mean = phi_g                        # polymer-rich matrix vitrifies
    g = torch.Generator(device=device).manual_seed(cfg["seed"])
    phi0 = mean + 0.05 * (2 * torch.rand(N, generator=g, dtype=V.DTYPE,
                                         device=device) - 1)
    rec_every = max(1, nsteps // 30)

    recs = {}
    for tag, pg in [("free", 1.0e6), ("arrest", phi_g)]:
        recs[tag] = V.march(phi0, dt, nsteps, kappa, M0, pg, w, h, L,
                            record_every=rec_every)

    Lfree = float(recs["free"]["Lt"][-1])
    Larr = float(recs["arrest"]["Lt"][-1])
    Lt_arr = recs["arrest"]["Lt"]
    half = len(Lt_arr) // 2
    # plateau: relative growth of the arrested domain scale over the last half
    plateau_growth = (float(Lt_arr[-1]) - float(Lt_arr[half])) / float(Lt_arr[-1])
    arrest_plateaus = bool(plateau_growth < 0.10)
    ratio = Lfree / Larr if Larr > 0 else float("inf")

    ctx.history["coarsen_t_free"] = _np(recs["free"]["t"])
    ctx.history["coarsen_Lt_free"] = _np(recs["free"]["Lt"])
    ctx.history["coarsen_t_arrest"] = _np(recs["arrest"]["t"])
    ctx.history["coarsen_Lt_arrest"] = _np(recs["arrest"]["Lt"])
    ctx.history["coarsen_x"] = _np(x)
    ctx.history["coarsen_phi0"] = _np(phi0)
    ctx.history["coarsen_phi_final_free"] = _np(recs["free"]["phi_final"])
    ctx.history["coarsen_phi_final_arrest"] = _np(recs["arrest"]["phi_final"])
    ctx.history["coarsen_mass_free"] = _np(recs["free"]["mass"])
    ctx.history["coarsen_mass_arrest"] = _np(recs["arrest"]["mass"])

    max_iters = max(recs["free"]["max_iters"], recs["arrest"]["max_iters"])
    mass_free = _np(recs["free"]["mass"])
    mass_arr = _np(recs["arrest"]["mass"])
    mass_drift = max(float(np.abs(mass_free - mass_free[0]).max()),
                     float(np.abs(mass_arr - mass_arr[0]).max()))
    return {"L_final_free": Lfree, "L_final_arrest": Larr,
            "arrest_ratio": float(ratio),
            "arrest_plateau_growth": float(plateau_growth),
            "arrest_plateaus": arrest_plateaus,
            "coarsen_max_newton_iters": int(max_iters),
            "coarsen_mass_drift_max": float(mass_drift)}


# --------------------------------------------------------------------------
# the harness run_fn
# --------------------------------------------------------------------------
def p11_run(cfg, ctx):
    device = torch.device(ctx.device if (torch.cuda.is_available()
                          and str(ctx.device).startswith("cuda")) else "cpu")
    ctx.provenance.update(
        model="1-D Cahn-Hilliard + vitrification mobility arrest",
        time_integrator="BDF1 (backward Euler)",
        newton="damped Newton, torch.linalg.solve dense Jacobian",
        jacobian="hand-derived analytic dR/dphi (autograd cross-checked)",
        grid={"N": cfg["N"], "L": cfg["L"], "periodic": True, "dim": 1},
        physics={"kappa": cfg["kappa"], "M0": cfg["M0"],
                 "phi_g": cfg["phi_g"], "w": cfg["w"]},
        device=str(device))
    ctx.log(f"P11 vitrification brick on {device}; coarsening N={cfg['N']}, "
            f"nsteps={cfg['nsteps']}, dt={cfg['dt']}, phi_g={cfg['phi_g']}")

    results = {"config": dict(cfg), "device": str(device)}
    checks = {}

    ctx.log("[a] analytic-derivative check (f', f'' vs central differences)")
    d = check_derivatives(device); checks.update(d)
    ctx.log(f"    f'_err={d['fprime_cd_err']:.2e} f''_err={d['fpp_cd_err']:.2e}")

    ctx.log("[b] Jacobian FD-check (hand-derived analytic vs central diff)")
    d = check_jacobian(cfg, device); checks.update(d)
    ctx.log(f"    jac_fd_max_err={d['jac_fd_max_err']:.3e} "
            f"jac_autograd_max_err={d['jac_autograd_max_err']:.3e} "
            f"(|J|max={d['jac_max_entry']:.1f})")

    ctx.log("[c] limiting-case CH dispersion (arrest off, constant mobility)")
    d = check_dispersion(cfg, ctx, device); checks.update(d)
    ctx.log(f"    sigma_analytic={d['dispersion_sigma_analytic']:.4f} "
            f"sigma_measured={d['dispersion_sigma_measured']:.4f} "
            f"rel_err={d['dispersion_rel_err']:.3e}")

    ctx.log("[d] MMS spatial convergence (manufactured source, ~2nd order)")
    d = check_mms(cfg, ctx, device); checks.update(d)
    ctx.log(f"    rates={list(np.round(ctx.history['mms_rates'], 3))} "
            f"mms_order_min={d['mms_order_min']:.3f}")

    ctx.log("[e] temporal order (backward Euler ~1st order)")
    d = check_temporal(cfg, ctx, device); checks.update(d)
    ctx.log(f"    rates={list(np.round(ctx.history['temporal_rates'], 3))} "
            f"temporal_order={d['temporal_order']:.3f}")

    ctx.log("[f] SCIENTIFIC RESULT: coarsening with/without vitrification arrest")
    d = check_coarsening(cfg, ctx, device); checks.update(d)
    ctx.log(f"    L_final free={d['L_final_free']:.4f} "
            f"arrest={d['L_final_arrest']:.4f} ratio={d['arrest_ratio']:.3f} "
            f"plateaus={d['arrest_plateaus']} "
            f"(last-half growth {d['arrest_plateau_growth']:+.3f}); "
            f"mass_drift={d['coarsen_mass_drift_max']:.2e}")

    checks["n_checks"] = 6
    results["checks"] = checks
    return results


def main():
    args = build_parser("OrgElMorph P11 (vitrification / mobility arrest, 1-D CH)"
                        ).parse_args()
    here = os.path.dirname(__file__)
    baseline = (os.path.join(here, "baseline.yaml")
                if args.mode == "reference" else None)
    # This brick uses its OWN torch dense solver (not the diffsim linsolver);
    # do NOT pass default_solver.
    run_tutorial(p11_run, schema=SCHEMA, args=args,
                 default_output=os.path.join(here, "outputs", "p11"),
                 baseline=baseline)


if __name__ == "__main__":
    main()
