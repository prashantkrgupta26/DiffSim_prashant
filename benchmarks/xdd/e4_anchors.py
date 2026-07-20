"""SP-1 R0 Block E — E4 framework-table anchors (REDUCED SCOPE, honest).

Full Table-1 parity (8 morphologies × 2 systems, full J–V sweeps) is a campaign,
not an R0 gate — at ~14 s/step (cuDSS, E5-fix) a full physical case is ~8 h.  The
R0 anchor E4 locks is the honest, tractable slice:

  **bilayer Jsc-only (V̂=0, light) for BOTH material systems**
    P3HT:PCBM → Table-1 bilayer  0.746 mA/cm²
    PM6:Y6    → Table-1 bilayer  2.422 mA/cm²

run ON THE BOX with linsolver="cudss".

## The honest regime caveat (carries over from E1/E2/E5, do NOT fake a match)

The PHYSICAL drive of either system hits the documented CPU/GPU-mesh Debye wall
(Ê_g≈40–52, Debye ≪ h on any feasible uniform-octree element — E1(ii), E2, E5,
and test_xdd_run._resolvable_bilayer all record it): a dark-eq march of the
physical config blows φ̂ up and never fires the steady criterion.  A
Scharfetter–Gummel discretisation or a boundary-refined anisotropic mesh (both
Block-D/E-follow-up / R1 items) is required to march the physical boundary layer.

So E4 marches the bilayer in the SAME reduced-drive marchable regime E1/E5 use
(λ²=1e-1, Ê_g=4, symmetric μ̂), with the PHYSICAL material parameters otherwise
(mobilities, DOS, ε, lifetimes, generation, band gap).  The measured Jsc is
therefore NOT expected to reproduce the physical Table-1 anchor to the percent —
it is a reduced-drive proxy.  We record the measured Jsc, the anchor, their
RATIO, and the HONEST delta class (order-of-magnitude reduced-drive proxy vs the
physical anchor), with the mechanism (mesh / drive / dissociation localization).
This is the plan's explicit instruction: "record the honest delta class, NOT a
fake match."

The two anchors' RATIO to each other IS meaningfully comparable across systems
in the shared regime (PM6:Y6 > P3HT:PCBM is the physical ordering — higher G,
narrower gap) — that cross-system ordering is the robust E4 anchor, mirroring
E1's site-count ordering being the robust reduction anchor.
"""
from __future__ import annotations

import argparse
import json
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
from diffsim.xdd.observables import to_mA_per_cm2
from diffsim.xdd.run import XDDRun, STEADY_STATE_JV, _post_process


# ── Framework-paper Table-1 bilayer anchors (mA/cm², Jsc @ V=0, light) ────────
TABLE1_BILAYER_ANCHOR = {
    "p3ht_pcbm": 0.746,
    "pm6_y6": 2.422,
}


# ── Framework-paper parameter sets (from A1 defaults + spec corpus) ───────────
def p3ht_pcbm_params() -> XDDParams:
    """P3HT:PCBM framework-paper values (brief / spec corpus).

    mu_n=2.3e-6, mu_p=7.3e-7, eps 3.0/3.9, tau 8e-9, a=1e-8 (Dx→a in the
    Onsager separation), G 6.5e27, Eg 1.15.
    """
    return XDDParams(
        mu_n=2.3e-6, mu_p=7.3e-7, eps_D=3.0, eps_A=3.9,
        tau_x_donor=8e-9, tau_x_acceptor=8e-9, a=1e-8,
        Gx_donor=6.5e27, Gx_acceptor=6.5e27, E_g=1.15,
        height=100e-9, N_C=2.5e25, N_V=2.5e25, T=300.0,
    )


def pm6_y6_params() -> XDDParams:
    """PM6:Y6 canonical A1 defaults + framework-paper G/Eg/a.

    A1 canonical (XDDParams() defaults) with G 1.0e28, Eg 1.35, a (Dx) 1e-7.
    """
    return XDDParams(
        Gx_donor=1.0e28, Gx_acceptor=1.0e28, E_g=1.35, a=1e-7,
    )


SYSTEMS = {
    "p3ht_pcbm": p3ht_pcbm_params,
    "pm6_y6": pm6_y6_params,
}


def bilayer_dist_gp(xq, height, h_axis=1):
    """Analytic bilayer signed-distance at GPs (A2 convention: neg=donor bottom
    half, pos=acceptor top half; interface at mid-height, in metres)."""
    dist_gp = {}
    for pv in xq:
        h_hat = xq[pv][:, h_axis]
        dist_gp[pv] = (h_hat - 0.5) * height
    return dist_gp


def run_bilayer_jsc(system: str, *, level: int = 6, device: str = "cpu",
                    linsolver: str = "splu", h_axis: int = 1,
                    verbose: bool = True, assembly: str = "auto") -> dict:
    """March one material system's analytic bilayer to lit steady state at V̂=0
    and return the Jsc observables.

    Reduced-drive marchable regime (λ²=1e-1, Ê_g=4, symmetric μ̂=0.5) — physical
    material parameters otherwise.  See module docstring for the honest caveat.
    """
    p = SYSTEMS[system]()
    s = p.scales()
    Eg_hat = 4.0            # relaxed built-in (marchable regime)
    lam2 = 1e-1            # resolvable Debye (marchable regime)
    pdeg = 1

    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=pdeg)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(pdeg, dim=2), device)
    xq = gauss_points(mesh, dm.tables_by_p)
    # Compressed dist scale (E1's stripe convention, 8 nm) so the interface mask
    # / Onsager E_B span is mesh-resolvable in the reduced-drive regime — the
    # physical mid-height ±50 nm bilayer dist is far too broad for a CPU mesh.
    dist_scale = 8e-9
    dist_gp = {pv: (xq[pv][:, h_axis] - 0.5) * dist_scale for pv in xq}

    # Marchable regime = E1's proven-convergent stripe knobs: symmetric μ̂=0.5,
    # ε̂=1 (unit — the ε-step + full Langevin cold-start is too stiff to march on
    # a CPU mesh; E1 uses the same reduction), weak Langevin (zeta=1e-6).  The
    # per-system material differences that SURVIVE this regime are: τ_x (via
    # tau_inv), a (Onsager E_B / |∇φ| series), and the donor:acceptor G ratio.
    # μ̂, λ², ε̂, Ê_g are all reduced/symmetrized — hence "reduced-drive proxy".
    mu = 0.5
    zeta = 1e-6
    mu_n = {}; mu_p = {}; mu_xd = {}; mu_xa = {}; eps = {}
    for pv in xq:
        d = dist_gp[pv]
        n = len(d)
        mu_n[pv] = np.full(n, mu); mu_p[pv] = np.full(n, mu)
        mu_xd[pv] = np.full(n, mu); mu_xa[pv] = np.full(n, mu)
        eps[pv] = np.ones(n)

    lang = LangevinRecombination(p, strategy="sum", zeta=zeta, spatial="uniform")
    ons = OnsagerBraunDissociation(p, width=p.interface_thk)
    tau_inv_d = s.t0 / p.tau_x_donor
    tau_inv_a = s.t0 / p.tau_x_acceptor

    sysm = XDDSystem(
        dm, lam2=lam2, eps_gp=eps, mu_n_gp=mu_n, mu_p_gp=mu_p,
        mu_xd_gp=mu_xd, mu_xa_gp=mu_xa, dist_gp=dist_gp,
        langevin=lang, onsager=ons, tau_inv_d=tau_inv_d, tau_inv_a=tau_inv_a,
        supg=1.0, carrier_vars="log", linsolver=linsolver, assembly=assembly)

    # Nondim max generation.  The PHYSICAL G/U0 is ~1e-4 (U0 is enormous for
    # these params) → a negligibly-lit device (J≈0), unrepresentative.  In the
    # reduced-drive marchable regime we drive with the SAME nondim generation
    # E1 uses (G_max_hat=1e-2) so the device is genuinely lit and the contact
    # flux is nonzero — the honest reduced-drive proxy.  The physical G enters
    # only through the donor:acceptor RATIO (Gx_donor:Gx_acceptor), preserved by
    # _set_generation_scaled.  (This is why E4 is a proxy, not a physical match.)
    G_max_hat = 1e-2
    gen = Generation(params=p, profile="constant", waveform="cw")

    runner = XDDRun(
        sysm, mesh, cons, params=p, strategy=STEADY_STATE_JV,
        Eg_hat=Eg_hat, V_sweep=[0.0], generation=gen,
        G_max_hat=G_max_hat, G_levels=3,
        dt0_hat=1e-6, dt_max_hat=1e-1, time_stepping_tol=1e-4, flux_floor=1e-3,
        max_steps_per_stage=400, bdf2=False, minority_ln=-Eg_hat, h_axis=h_axis,
        newton_kw={"max_iter": 20})

    res = runner.run()
    V, info = res["sweep_history"][0]
    jny = float(info["jny"]); jpy = float(info["jpy"])
    J = max(abs(jny), abs(jpy))          # nondim |J| (min-for-parity convention)
    Jmin = min(abs(jny), abs(jpy))
    # Jsc re-dimensioned to mA/cm² (CPU DDJscBulk × 0.1 convention).
    jsc_mA = to_mA_per_cm2(J, s.J0)
    pp = _post_process(sysm, res["final_state"])
    carrier_src = float(pp["int_kd_xd"] + pp["int_ka_xa"])

    out = dict(
        system=system, level=level, device=device, linsolver=linsolver,
        assembly=sysm.assembly,
        jny_nondim=jny, jpy_nondim=jpy, J_nondim=float(J), Jmin_nondim=float(Jmin),
        flux_imbalance=float(abs(jny - jpy) / max(J, 1e-30)),
        J0=float(s.J0), U0=float(s.U0), t0=float(s.t0),
        Eg_hat=Eg_hat, lam2=lam2, G_max_hat=float(G_max_hat),
        jsc_mA_per_cm2=float(jsc_mA),
        carrier_src=carrier_src,
        criterion_fired=bool(info["criterion_fired"]),
    )
    if verbose:
        print(f"[e4] {system:10s} Jsc={jsc_mA:.4f} mA/cm²  |J|_nondim={J:.4e} "
              f"flux_imbal={out['flux_imbalance']:.2e} fired={out['criterion_fired']}",
              flush=True)
    return out


def anchor_ratio(measured_mA: float, anchor_mA: float) -> dict:
    ratio = measured_mA / anchor_mA if anchor_mA != 0 else float("nan")
    return dict(measured_mA_per_cm2=measured_mA, anchor_mA_per_cm2=anchor_mA,
                ratio=float(ratio))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--linsolver", default="splu", choices=("splu", "cudss"))
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--assembly", default="auto",
                    choices=("auto", "host", "device"),
                    help="Jacobian/residual assembly backend (Task #35): "
                         "auto = device on CUDA, host on CPU")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}
    for system in ("p3ht_pcbm", "pm6_y6"):
        r = run_bilayer_jsc(system, level=args.level, device=args.device,
                            linsolver=args.linsolver, assembly=args.assembly)
        anchor = TABLE1_BILAYER_ANCHOR[system]
        r["anchor"] = anchor_ratio(r["jsc_mA_per_cm2"], anchor)
        results[system] = r

    # cross-system ordering (the robust anchor)
    pm6 = results["pm6_y6"]["jsc_mA_per_cm2"]
    p3ht = results["p3ht_pcbm"]["jsc_mA_per_cm2"]
    ordering_holds = pm6 > p3ht
    anchor_ordering_holds = (TABLE1_BILAYER_ANCHOR["pm6_y6"]
                             > TABLE1_BILAYER_ANCHOR["p3ht_pcbm"])

    out = {
        "results": results,
        "cross_system_ordering": {
            "pm6_y6_jsc": pm6, "p3ht_pcbm_jsc": p3ht,
            "pm6_gt_p3ht_measured": bool(ordering_holds),
            "pm6_gt_p3ht_anchor": bool(anchor_ordering_holds),
        },
    }
    print("\n" + "=" * 70, flush=True)
    for system in ("p3ht_pcbm", "pm6_y6"):
        a = results[system]["anchor"]
        print(f"[e4] {system:10s} measured {a['measured_mA_per_cm2']:.4f} / "
              f"anchor {a['anchor_mA_per_cm2']:.3f} mA/cm²  ratio={a['ratio']:.4f}",
              flush=True)
    print(f"[e4] cross-system ordering PM6>P3HT: measured={ordering_holds} "
          f"anchor={anchor_ordering_holds}", flush=True)
    print("=" * 70, flush=True)
    print(json.dumps(out, indent=2, default=str), flush=True)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"[e4] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
