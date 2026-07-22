"""P2-R2a projection-vs-monolithic FIELD-COMPARISON diagnostic (Task 3c).

On the level-4 Re=1 Stokes sphere the outflow-BC projection
(``pressure_update="standard"``, ``ppe_fine_scale=True``,
``pressure_outflow_nodes=<x=1 face>``) converges to Cd approx 0 while the
MONOLITHIC saddle solver on the SAME mesh gives a converged Cd approx +40 — a
~40x faithfulness gap. The outflow probe (tests/p2r2a_outflow_probe.py) measured
the gap; this script LOCALIZES it.

Because BOTH marches feed the IDENTICAL ``surrogate_traction`` observable, an
"observable bug" (fields match, traction differs) is IMPOSSIBLE. The real
question is WHERE the field diverges (velocity vs pressure) and which drag term
(pressure/form vs viscous/friction) carries the gap.

What it does, on the level-4 Re=1 sphere:

  1. Marches BOTH to (near-)steady state and captures the FULL node-major
     ``(u, p)`` field of each:
       - Monolithic: the saddle solve vector ``x`` IS the node-major free
         ``(u, p)`` (velocity comps 0..dim-1 then pressure at index dim per
         free node); ``T_vec @ x`` is the full node-major field — exactly the
         vector fed to ``surrogate_traction`` at line 191 of
         p2r0_task10_sphere_derisk._march / task10.monolithic_cd. We keep the
         FREE vector ``x_mono_free`` (= ``x``) for the field comparison and
         ``x_mono = T_vec @ x`` for the traction.
       - Projection: assembled the SAME way LeraySBMStepper.surrogate_traction
         (leray_sbm.py ~L318) builds its ``x_full``: after ``st.step()``,
         velocity ``u = st.base._uvec(st.base.hist.pre1)`` (the corrected
         ``u_new``) and pressure ``st.base.p_star`` (the corrected ``p_hat``,
         set at leray.py L656) are scattered into a free node-major vector
         ``xfree[n_free, ndof]`` (vel comps 0..dim-1, pressure at index dim),
         then ``x_full = _T_vec @ xfree``. We keep ``x_proj_free`` (= ``xfree``)
         and ``x_proj = _T_vec @ xfree``. IDENTICAL assembly => apples-to-apples.

  2. Field-error report over the FREE FLUID nodes (strong-Dirichlet nodes
     excluded via ``strong_mask``): relative L2 errors of velocity (per comp and
     combined) and pressure, plus the fore-aft pressure asymmetry
     ``Δp = p_front - p_rear`` (front = upstream stagnation x approx CTR_x-R,
     rear = x approx CTR_x+R; nearest free nodes) for projection vs monolithic.

  3. Drag decomposition via a local ``traction_decomposed`` that MIRRORS
     ``surrogate_traction`` exactly but returns ``(F_pressure, F_viscous)``
     separately (the ``pq*n`` form term and the ``-nu*(gradu.T@n)`` friction
     term). Cd_pressure / Cd_viscous / Cd_total (= F_x / qref()) for both.

  4. Verdict lines.

Fast: this is a STANDALONE diagnostic, NOT a pytest gate. Marches the 3-D
sphere (splu on gpubox CPU) — do NOT run locally.

Usage:
    cd /path/to/DiffSim
    STEPS=60 .venv/bin/python tests/p2r2a_field_compare.py 2>&1 | tee /tmp/field_compare.log

Reuses tests/p2r0_task10_sphere_derisk (build_sphere_3d, outflow_free_nodes,
qref, R, CTR, U_IN) and mirrors the monolithic march / outflow-BC projection
setup of tests/p2r2a_outflow_probe.py (Task 3b) with attribution.
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

# ---------------------------------------------------------------------------
# Path bootstrap (mirrors the outflow probe)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from p2r0_task10_sphere_derisk import (build_sphere_3d, outflow_free_nodes,   # noqa: E402
                                       qref, R, CTR, U_IN)
from diffsim.steppers.leray_sbm import LeraySBMStepper                        # noqa: E402
from diffsim.sbm.vector import sbm_vector_dirichlet                           # noqa: E402
from diffsim.api.ns_bricks import assemble_linear_ns                          # noqa: E402
from diffsim.physics.poisson import gauss_points                              # noqa: E402
from diffsim.mesh.faces import face_tables                                    # noqa: E402

# ---------------------------------------------------------------------------
# Constants (match the outflow probe / Task-1 diagnostic)
# ---------------------------------------------------------------------------
DT = 0.05
LEVEL = 4
RE = 1.0        # Re=1: Stokes, isolates pressure coupling
ALPHA = 100.0
STEPS = int(os.environ.get("STEPS", "60"))


# ---------------------------------------------------------------------------
# Local drag decomposition — MIRRORS diffsim.sbm.vector.surrogate_traction
# EXACTLY (same face loop, same n_hat = -geo.n, same area correction), but
# splits the integrand F += w*(pq*n - nu*(gradu.T@n)) into its two terms:
#   F_pressure += w * pq * n            (PRESSURE / form contribution)
#   F_viscous  += w * (-nu*(gradu.T@n)) (VISCOUS / friction contribution)
# so F_pressure + F_viscous == surrogate_traction(...) to round-off.
# ---------------------------------------------------------------------------

def traction_decomposed(dm, sf, geo, x_all, nu, ndof):
    """Return (F_pressure, F_viscous), each a length-dim force vector, such that
    F_pressure + F_viscous == surrogate_traction(dm, sf, geo, x_all, nu, ndof).
    x_all: FULL node-major (u, p) vector."""
    dim = dm.dim
    mesh = dm.mesh
    p_face = np.unique(np.asarray(mesh.p_elem)[sf.elem])
    if len(p_face) != 1:
        from diffsim.errors import ConfigError
        raise ConfigError(
            f"SBM face helper assumes uniform p on the face, got "
            f"orders {p_face.tolist()}")
    pv = int(p_face[0])
    ftab = face_tables(pv, dim)
    nqf, nbf = ftab.nqf, ftab.nbf
    conn = mesh.conn_of[pv][np.searchsorted(mesh.bins[pv], sf.elem)]  # [Nf, nbf]
    xv = x_all.reshape(dm.n_nodes, ndof)
    h = mesh.tree.h()[sf.elem]
    jacS = (h / 2.0) ** (dim - 1)
    dscale = 2.0 / h
    F_pressure = np.zeros(dim)
    F_viscous = np.zeros(dim)
    for fi in range(len(sf.elem)):
        f = int(sf.face[fi])
        un = xv[conn[fi], :dim]                          # [nbf, dim]
        pn = xv[conn[fi], dim]                           # [nbf]
        N = ftab.N[f]                                    # [nqf, nbf]
        dN = ftab.dN[f] * dscale[fi]                     # [nqf, nbf, dim]
        for q in range(nqf):
            w = ftab.w[q] * jacS[fi] * geo.corr[fi * nqf + q]
            n = geo.n[fi * nqf + q]
            gradu = dN[q].T @ un
            pq = N[q] @ pn
            F_pressure += w * (pq * n)                   # n_hat = -geo.n
            F_viscous += w * (-nu * (gradu.T @ n))
    return F_pressure, F_viscous


# ---------------------------------------------------------------------------
# Monolithic march (mirrors p2r0_task10_sphere_derisk.monolithic_cd /
# p2r2a_outflow_probe._march_monolithic EXACTLY) — but RETURNS the full and
# free node-major (u,p) fields at the end so we can compare the field itself.
# The saddle solve vector `x` IS the node-major FREE (u,p) vector; `T_vec @ x`
# is the FULL node-major field (the exact vector fed to surrogate_traction).
# ---------------------------------------------------------------------------

def _march_monolithic(fx, alpha, nsteps):
    dm, mesh, cons = fx["dm"], fx["mesh"], fx["cons"]
    sf, geo = fx["sf"], fx["geo"]
    nu, ndof, dim = fx["nu"], fx["ndof"], fx["dim"]
    coords = fx["coords"]
    T = cons.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    nfree = T.shape[1]
    xq = gauss_points(mesh, dm.tables_by_p)
    strong = np.where(fx["strong_mask"])[0]
    g_strong = fx["u_inf"][strong]
    Af, bf = sbm_vector_dirichlet(
        dm, sf, geo, lambda y: np.zeros((len(y), dim)), nu, ndof, alpha=alpha)
    Af_c = (T_vec.T @ Af @ T_vec).tocsr()
    bf_c = np.asarray(T_vec.T @ bf)

    def gp_field(node_vec):
        full = np.asarray(T @ node_vec)
        aq, dq = {}, {}
        for pv in dm.bins:
            tb = dm.tables_by_p[pv]
            vals = full[mesh.conn_of[pv]]
            aq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
            h = mesh.tree.h()[mesh.bins[pv]]
            dq[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                      * (2.0 / h)[:, None]).reshape(-1)
        return aq, dq

    q = qref()
    x = np.zeros(nfree * ndof)
    sigma = 1.0 / DT
    cds = []
    for step in range(nsteps):
        u_node = x.reshape(nfree, ndof)[:, :dim]
        aq, dq = gp_field(u_node)
        fq = {pv: aq[pv] / DT for pv in xq}
        A, b = assemble_linear_ns(dm, aq, dq, fq, nu, sigma=sigma)
        A = (A + Af_c).tolil()
        b = b + bf_c
        for k, i in enumerate(strong):
            for c in range(dim):
                r = i * ndof + c
                A.rows[r] = [int(r)]
                A.data[r] = [1.0]
                b[r] = g_strong[k, c]
        pin = int(np.argmax(coords.sum(1))) * ndof + dim
        A.rows[pin] = [pin]
        A.data[pin] = [1.0]
        b[pin] = 0.0
        x = splu(A.tocsr().tocsc()).solve(b)
        x_full = np.asarray(T_vec @ x)
        from diffsim.sbm.vector import surrogate_traction
        F = surrogate_traction(dm, sf, geo, x_full, nu, ndof)
        cds.append(float(F[0] / q))
        print(f"[mono] step{step+1:3d}: Cd={cds[-1]:+.4f}", flush=True)
    # x is the node-major FREE (u,p) vector; T_vec @ x is the FULL field.
    return dict(cds=cds, x_free=x.copy(), x_full=np.asarray(T_vec @ x))


# ---------------------------------------------------------------------------
# Outflow-BC projection march (mirrors p2r2a_outflow_probe._march_projection,
# outflow-BC branch) — returns per-step Cd AND the FULL / FREE node-major (u,p)
# field assembled the SAME way LeraySBMStepper.surrogate_traction does.
# ---------------------------------------------------------------------------

def _march_projection(fx, pressure_outflow_nodes, nsteps):
    dim = fx["dim"]

    def f_fn(x, t):
        return np.zeros((len(x), dim))

    st = LeraySBMStepper(
        fx["oracle"], fx["dm"], fx["nu"], DT, f_fn,
        u_inf=fx["u_inf"], strong_mask=fx["strong_mask"],
        lam=0.5, domain="outside", order=2, picard_iters=2,
        solver="splu", ppe_finescale=False, alpha=ALPHA,
        beta_backflow=1.0, velocity_update="consistent",
        pressure_update="standard", ppe_fine_scale=True,
        pressure_outflow_nodes=pressure_outflow_nodes,
        sbm_pressure_coupling=bool(os.environ.get("SBM_PCOUPLE")),
    )
    st.set_initial(lambda c: np.zeros((len(c), dim)))

    q = qref()
    cds = []
    for k in range(nsteps):
        st.step()
        F = st.surrogate_traction()
        cds.append(float(F[0] / q))
        print(f"[proj] step{k+1:3d}: Cd={cds[-1]:+.4f}", flush=True)

    # Assemble the FULL/FREE node-major (u,p) EXACTLY as
    # LeraySBMStepper.surrogate_traction (leray_sbm.py ~L318) does:
    #   velocity = corrected u_new (base._uvec(base.hist.pre1)),
    #   pressure = corrected p_hat (base.p_star, set at leray.py L656),
    #   scattered into xfree[n_free, ndof] (vel 0..dim-1, pressure at dim),
    #   x_full = _T_vec @ xfree.
    u = st.base._uvec(st.base.hist.pre1)
    xfree = np.zeros(st.n_free * st.ndof)
    xv = xfree.reshape(st.n_free, st.ndof)
    xv[:, :st.dim] = u
    xv[:, st.dim] = st.base.p_star
    x_full = np.asarray(st._T_vec @ xfree)
    return dict(cds=cds, x_free=xfree, x_full=x_full, st=st)


# ---------------------------------------------------------------------------
# Field-comparison helpers (operate on the FREE node-major (u,p) vectors,
# which are aligned to fx["coords"] = mesh.node_coords[cons.free_nodes]).
# ---------------------------------------------------------------------------

def _split_free(x_free, n_free, ndof, dim):
    xv = x_free.reshape(n_free, ndof)
    return xv[:, :dim], xv[:, dim]      # (u[n_free, dim], p[n_free])


def _rel_l2(a, b):
    denom = np.linalg.norm(b)
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float(np.linalg.norm(a - b) / denom)


def _foreaft_dp(coords, p, fluid_mask):
    """Δp = p_front - p_rear, sampled at the nearest FREE FLUID nodes to the
    sphere's upstream stagnation (x approx CTR_x - R) and downstream (CTR_x + R)
    points on the axis (y=CTR_y, z=CTR_z)."""
    cx, cy, cz = CTR
    front_pt = np.array([cx - R, cy, cz])
    rear_pt = np.array([cx + R, cy, cz])
    idx = np.where(fluid_mask)[0]
    c = coords[idx]
    i_front = idx[int(np.argmin(np.linalg.norm(c - front_pt, axis=1)))]
    i_rear = idx[int(np.argmin(np.linalg.norm(c - rear_pt, axis=1)))]
    return (float(p[i_front] - p[i_rear]), i_front, i_rear,
            coords[i_front], coords[i_rear])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print(f"[fc] STEPS={STEPS}  Building level-4 sphere fixture at Re=1 "
          f"(Stokes)...", flush=True)
    device = "cuda:0" if os.environ.get("DIFFSIM_CUDA") else "cpu"
    fx = build_sphere_3d(device, level=LEVEL, Re=RE)
    outflow = outflow_free_nodes(fx)
    coords = fx["coords"]
    n_free, ndof, dim = len(coords), fx["ndof"], fx["dim"]
    nu = fx["nu"]
    dm, sf, geo = fx["dm"], fx["sf"], fx["geo"]
    strong_mask = fx["strong_mask"]
    fluid_mask = ~strong_mask                 # FREE FLUID nodes (exclude strong)
    print(f"[fc] fixture ready  n_free={n_free}  n_strong={int(strong_mask.sum())}"
          f"  n_fluid={int(fluid_mask.sum())}  n_outflow_free={len(outflow)}"
          f"  ({time.time()-t0:.1f}s)", flush=True)

    # --- (1) march both, capture full fields ---
    print(f"\n=== (1a) MONOLITHIC march, {STEPS} steps ===", flush=True)
    mono = _march_monolithic(fx, ALPHA, STEPS)
    print(f"\n=== (1b) outflow-BC PROJECTION march, {STEPS} steps ===",
          flush=True)
    proj = _march_projection(fx, outflow, STEPS)

    x_mono, x_proj = mono["x_full"], proj["x_full"]        # FULL node-major
    u_mono, p_mono = _split_free(mono["x_free"], n_free, ndof, dim)
    u_proj, p_proj = _split_free(proj["x_free"], n_free, ndof, dim)

    cd_mono = float(mono["cds"][-1])
    cd_proj = float(proj["cds"][-1])

    # --- (2) field-error report over FREE FLUID nodes ---
    print(f"\n=== (2) FIELD ERRORS over {int(fluid_mask.sum())} free FLUID "
          f"nodes (strong-Dirichlet excluded) ===", flush=True)
    um, up = u_mono[fluid_mask], u_proj[fluid_mask]
    pm, pp = p_mono[fluid_mask], p_proj[fluid_mask]
    comp_names = ["ux", "uy", "uz"][:dim]
    for c in range(dim):
        print(f"[fc]   rel-L2 {comp_names[c]}: {_rel_l2(up[:, c], um[:, c]):.4e}",
              flush=True)
    print(f"[fc]   rel-L2 |u| (combined): {_rel_l2(up.ravel(), um.ravel()):.4e}",
          flush=True)
    print(f"[fc]   rel-L2 p           : {_rel_l2(pp, pm):.4e}", flush=True)
    print(f"[fc]   ||u_mono||={np.linalg.norm(um):.4e}  "
          f"||u_proj||={np.linalg.norm(up):.4e}  "
          f"||p_mono||={np.linalg.norm(pm):.4e}  "
          f"||p_proj||={np.linalg.norm(pp):.4e}", flush=True)

    # fore-aft pressure asymmetry (form-drag signature)
    dp_m, ifm, irm, cfm, crm = _foreaft_dp(coords, p_mono, fluid_mask)
    dp_p, ifp, irp, cfp, crp = _foreaft_dp(coords, p_proj, fluid_mask)
    print(f"[fc]   fore-aft Δp = p_front - p_rear   (front approx x={CTR[0]-R:.3f},"
          f" rear approx x={CTR[0]+R:.3f}):", flush=True)
    print(f"[fc]     monolithic:  Δp={dp_m:+.4e}  "
          f"(front node {ifm} @ {np.round(cfm,3)}  p={p_mono[ifm]:+.4e} | "
          f"rear node {irm} @ {np.round(crm,3)}  p={p_mono[irm]:+.4e})",
          flush=True)
    print(f"[fc]     projection:  Δp={dp_p:+.4e}  "
          f"(front node {ifp} @ {np.round(cfp,3)}  p={p_proj[ifp]:+.4e} | "
          f"rear node {irp} @ {np.round(crp,3)}  p={p_proj[irp]:+.4e})",
          flush=True)

    # --- (3) drag decomposition (pressure/form vs viscous/friction) ---
    print(f"\n=== (3) DRAG DECOMPOSITION (Cd = F_x / qref) ===", flush=True)
    q = qref()
    Fp_m, Fv_m = traction_decomposed(dm, sf, geo, x_mono, nu, ndof)
    Fp_p, Fv_p = traction_decomposed(dm, sf, geo, x_proj, nu, ndof)
    cdp_m, cdv_m = float(Fp_m[0] / q), float(Fv_m[0] / q)
    cdp_p, cdv_p = float(Fp_p[0] / q), float(Fv_p[0] / q)
    print(f"[fc]   monolithic:  Cd_pressure={cdp_m:+.4f}  "
          f"Cd_viscous={cdv_m:+.4f}  Cd_total={cdp_m+cdv_m:+.4f}  "
          f"(reported Cd_final={cd_mono:+.4f})", flush=True)
    print(f"[fc]   projection:  Cd_pressure={cdp_p:+.4f}  "
          f"Cd_viscous={cdv_p:+.4f}  Cd_total={cdp_p+cdv_p:+.4f}  "
          f"(reported Cd_final={cd_proj:+.4f})", flush=True)
    print(f"[fc]   gap: Cd_pressure {cdp_m:+.4f}->{cdp_p:+.4f} "
          f"(Δ={cdp_p-cdp_m:+.4f})  |  Cd_viscous {cdv_m:+.4f}->{cdv_p:+.4f} "
          f"(Δ={cdv_p-cdv_m:+.4f})", flush=True)

    # --- (4) verdict ---
    print(f"\n=== (4) VERDICT ===", flush=True)
    u_rel = _rel_l2(up.ravel(), um.ravel())
    p_rel = _rel_l2(pp, pm)
    u_match = np.isfinite(u_rel) and u_rel < 0.10
    p_match = np.isfinite(p_rel) and p_rel < 0.10
    # which drag component carries the gap (larger |Δ|)?
    dcdp, dcdv = abs(cdp_p - cdp_m), abs(cdv_p - cdv_m)
    carrier = ("PRESSURE (form)" if dcdp >= dcdv else "VISCOUS (friction)")
    proj_dp_zero = abs(dp_p) < 0.10 * (abs(dp_m) + 1e-30)
    print(f"[fc]   (a) velocity fields match (rel-L2<10%)? {u_match}  "
          f"(rel-L2 |u|={u_rel:.4e})", flush=True)
    print(f"[fc]   (b) pressure fields match (rel-L2<10%)? {p_match}  "
          f"(rel-L2 p={p_rel:.4e})", flush=True)
    print(f"[fc]   (c) drag gap carried by: {carrier}  "
          f"(|ΔCd_p|={dcdp:.4f} vs |ΔCd_v|={dcdv:.4f})", flush=True)
    print(f"[fc]   (d) projection fore-aft Δp approx 0 vs monolithic? "
          f"{proj_dp_zero}  (proj Δp={dp_p:+.4e}, mono Δp={dp_m:+.4e})",
          flush=True)
    print(f"[fc]   NOTE: both marches feed the IDENTICAL surrogate_traction, so "
          f"an 'observable bug' (fields match, traction differs) is IMPOSSIBLE. "
          f"The field divergence localized above IS the cause of the ~40x Cd "
          f"gap; the verdict pinpoints velocity-vs-pressure and which drag term.",
          flush=True)
    print(f"\n[fc] done ({time.time()-t0:.1f}s total)", flush=True)


if __name__ == "__main__":
    main()
