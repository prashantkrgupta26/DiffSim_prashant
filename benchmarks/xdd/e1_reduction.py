"""SP-1 R0 Block E — E1 reduction gates (physics-anchor, runs locally).

Two reduction anchors, each locking a baseline in tests/baselines/xdd_e1_e2.json:

E1(i)  Collapsed / transport-only ordering on 2-D stripe morphologies:
       more interface generation sites  →  more free-carrier generation
       →  more current.  The CMAME-2012 transport-only model current is
       proportional to the free-carrier source it receives at the D/A
       interface; in the excitonic system that source is  ∫ k̂_i X̂_i dV  (the
       dissociation-generated carrier rate).  We build tiny stripe morphologies
       with the A2 utilities (`signed_distance`), march each to a lit steady
       state, and assert the ORDERING (not absolute numbers, per the plan):
         6-stripe  >  2-stripe   (2× the interface sites → more dissociation).

E1(ii) 1-D-in-2-D homogeneous PPV:PCBM-like device vs the Kodali validation:
       Jsc must land in the −30 A/m²-class band.  The XDD forward solver hits
       the documented CPU-mesh drive wall on the PHYSICAL config (Ê_g≈52,
       Debye≈0.44 nm ≪ any feasible CPU element — see test_xdd_run.py's
       `_resolvable_bilayer` docstring and the E-report), so the Kodali
       anchor is locked from the SimSS 1-D reference of the SAME physical
       device (e2_simsalabim.py), and the honest mesh-wall delta is recorded.

Reduced-drive regime (E1(i)): lam2=1e-1, Ê_g=4, symmetric μ̂ — the marchable
CPU-warp config documented in test_xdd_run._resolvable_bilayer.  Absolute
currents are NOT physical here; only the ORDERING is asserted (the plan's
instruction).  The device PARAMETERS (mobilities, DOS, ε, E_g, G) are the
physical PPV:PCBM-class values; only the Poisson drive strength (λ²) and the
built-in (Ê_g) are relaxed to keep the boundary layer mesh-resolvable.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bench_bootstrap  # noqa

import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.poisson import gauss_points
from diffsim.physics.exciton_system import XDDSystem
from diffsim.physics.exciton_closures import (
    LangevinRecombination, OnsagerBraunDissociation, Generation)
from diffsim.xdd.params import XDDParams
from diffsim.xdd.morphology import signed_distance, interface_mask
from diffsim.xdd.run import XDDRun, STEADY_STATE_JV, _post_process


# ── Kodali PPV:PCBM Table-2 device parameters (physical; the anchor) ──────────
def kodali_params() -> XDDParams:
    """PPV:PCBM-class device: Kodali-2012 Table-2 values (spec/plan corpus).

    N_C=N_V=2.5e25, mu_n0=2.5e-7, mu_p0=3.0e-8, E_g=1.34 eV, eps_r≈3.4,
    a=2.5 nm, height=120 nm.  Uniform generation G=2.7e27 is applied by the
    caller (nondim G_hat = G/U0).
    """
    return XDDParams(
        height=120e-9, eps_A=3.4, eps_D=3.4, N_C=2.5e25, N_V=2.5e25,
        mu_n=2.5e-7, mu_p=3.0e-8, E_g=1.34, T=300.0, a=2.5e-9,
    )


# ── Stripe morphology builder (A2 utilities) ─────────────────────────────────
def make_stripes(n_grid: int, n_stripes: int, orient: str) -> np.ndarray:
    """Binary alternating-stripe morphology on an n_grid×n_grid grid.

    orient='vertical'   : bands vary along x (axis 0) — stripes run ⟂ to contacts
    orient='horizontal' : bands vary along y (axis 1) — stripes ∥ to contacts

    n_stripes = number of alternating bands across the device.  Returns a float
    array in {0.,1.} (0=donor, 1=acceptor).
    """
    idx = np.arange(n_grid)
    band = (idx * n_stripes // n_grid) % 2
    m = np.zeros((n_grid, n_grid))
    if orient == "vertical":
        m[:, :] = band[:, None]
    elif orient == "horizontal":
        m[:, :] = band[None, :]
    else:
        raise ValueError(f"orient must be 'vertical'|'horizontal', got {orient!r}")
    return m.astype(float)


def _stripe_dist_gp(xq, n_stripes, orient, dist_scale, n_grid=33):
    """Signed-distance field of a stripe morphology, nearest-sampled to GPs."""
    morph = make_stripes(n_grid, n_stripes, orient)
    spacing = (1.0 / (n_grid - 1), 1.0 / (n_grid - 1))
    dist = signed_distance(morph, spacing)          # [device-length units]
    dist_gp = {}
    for pv in xq:
        gx = np.clip((xq[pv][:, 0] * (n_grid - 1)).astype(int), 0, n_grid - 1)
        gy = np.clip((xq[pv][:, 1] * (n_grid - 1)).astype(int), 0, n_grid - 1)
        dist_gp[pv] = dist[gx, gy] * dist_scale
    return dist_gp


def run_stripe_case(n_stripes: int, orient: str, *, level: int = 3,
                    zeta: float = 1e-6, dist_scale: float = 8e-9) -> dict:
    """March one stripe morphology to a lit steady state; return the reduction
    observables.

    Returns a dict with:
      J             : |Ĵ| = max(|Ĵny|,|Ĵpy|) at V=0 (nondim, reduced-drive)
      carrier_src   : ∫ k̂_D X̂_D + ∫ k̂_A X̂_A  (free-carrier generation content —
                      the transport-only carrier source; the ordering observable)
      n_sites       : Σ interface_mask over GPs (interface-site count proxy)
      fired         : steady criterion reached
    """
    p = kodali_params()
    s = p.scales()
    Eg_hat = 4.0                 # relaxed built-in (marchable regime)
    lam2 = 1e-1                  # resolvable Debye length (marchable regime)
    pdeg = 1
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=pdeg)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(pdeg, dim=2), "cpu")
    xq = gauss_points(mesh, dm.tables_by_p)

    dist_gp = _stripe_dist_gp(xq, n_stripes, orient, dist_scale)
    mu = 0.5                     # symmetric transport (marchable regime)
    mu_n = {}; mu_p = {}; mu_xd = {}; mu_xa = {}; eps = {}
    for pv in xq:
        n = len(dist_gp[pv])
        mu_n[pv] = np.full(n, mu); mu_p[pv] = np.full(n, mu)
        mu_xd[pv] = np.full(n, mu); mu_xa[pv] = np.full(n, mu)
        eps[pv] = np.ones(n)

    lang = LangevinRecombination(p, strategy="sum", zeta=zeta, spatial="uniform")
    ons = OnsagerBraunDissociation(p, width=p.interface_thk)
    tau_inv = s.t0 / p.tau_x_donor
    sysm = XDDSystem(
        dm, lam2=lam2, eps_gp=eps, mu_n_gp=mu_n, mu_p_gp=mu_p,
        mu_xd_gp=mu_xd, mu_xa_gp=mu_xa, dist_gp=dist_gp,
        langevin=lang, onsager=ons, tau_inv_d=tau_inv, tau_inv_a=tau_inv,
        supg=1.0, carrier_vars="log")

    n_sites = 0.0
    for pv in dm.bins:
        n_sites += float(np.sum(interface_mask(dist_gp[pv], p.interface_thk / 2.0)))

    gen = Generation(params=p, profile="constant", waveform="cw")
    runner = XDDRun(
        sysm, mesh, cons, params=p, strategy=STEADY_STATE_JV,
        Eg_hat=Eg_hat, V_sweep=[0.0], generation=gen, G_max_hat=1e-2, G_levels=2,
        dt0_hat=1e-6, dt_max_hat=1e-1, time_stepping_tol=1e-4, flux_floor=1e-3,
        max_steps_per_stage=300, bdf2=False, minority_ln=-Eg_hat, h_axis=1,
        newton_kw={"max_iter": 20})
    res = runner.run()
    V, info = res["sweep_history"][0]
    J = max(abs(info["jny"]), abs(info["jpy"]))
    pp = _post_process(sysm, res["final_state"])
    carrier_src = pp["int_kd_xd"] + pp["int_ka_xa"]
    return dict(J=float(J), carrier_src=float(carrier_src),
                n_sites=float(n_sites), fired=bool(info["criterion_fired"]))


def e1_stripe_ordering(level: int = 3) -> dict:
    """Run the four stripe cases {vertical,horizontal} × {2,6} and return a
    summary dict with the ordering observables."""
    out = {}
    for orient in ("vertical", "horizontal"):
        for ns in (2, 6):
            out[f"{orient}_{ns}"] = run_stripe_case(ns, orient, level=level)
    return out


if __name__ == "__main__":
    import json
    summary = e1_stripe_ordering(level=3)
    for k, v in summary.items():
        print(f"{k:14s} J={v['J']:.4e} csrc={v['carrier_src']:.4e} "
              f"sites={v['n_sites']:.0f} fired={v['fired']}")
    print(json.dumps(summary, indent=2))
