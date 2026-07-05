"""Adjoint gradients for SBM Poisson (spec S4.3, S5.2 Tier-2 VJP #1).

Within an epoch, R(u; theta, kappa) = A u - b and geometry enters only
through the frozen face cache (d, n, corr) and mapped data (gbar, qbar).
For a QoI J(u):

    A^T lam = (dJ/du)^T          (adjoint solve — assembled path: splu on A.T)
    dJ/dtheta = -lam^T dR/dtheta (Warp-tape sweep over the pure face-RESIDUAL
                                  kernels below, seeded with lam_full = T lam;
                                  cotangents flow into the torch geometry
                                  chain: {d, n} = distance_torch, gbar =
                                  g(x + d), corr = n_tilde . n, qbar = q(x+d))
    dJ/dkappa = -lam^T (A1 u - bg1)   (scalar-kappa linearity, meta pieces)

THIS MODULE'S KERNELS ARE TAPED: enable_backward=True (the only such module,
per the Task-1 rule). They compute the face part of (A u - b) directly —
pure, all Gauss-point data via arrays — with dvec/gbar (and nvec/corr/qbar
for Neumann) as differentiable inputs; u is a frozen constant.

Strong outer-Dirichlet rows are theta-independent (identity rows): lam is
zeroed there before seeding. The Krylov/BLAS host boundaries are ROUTED
AROUND (adjoint = one transposed solve), never taped through.
"""
import numpy as np
import scipy.sparse as sp
import torch
import warp as wp

from ..assembly.operators import _kernel_cache
from ..geometry.project import distance_torch
from .poisson import _shift_fn_for, _flux_shift_fn_for


def make_sbm_dirichlet_residual(nbf: int, nqf: int, dim: int):
    """Face residual r[conn[a]] += kappa [ -N_a (grad_u.n~) - gna (Su - gbar)
    + (alpha/h) Sa (Su - gbar) ] dS, differentiable in (dvec, gbar)."""
    key = ("sbm_dir_res", nbf, nqf, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    shift_fn = _shift_fn_for(nbf, dim)

    @wp.kernel(module="unique", enable_backward=True)
    def sbm_dir_res(felem: wp.array(dtype=wp.int32),
                    fface: wp.array(dtype=wp.int32),
                    conn: wp.array2d(dtype=wp.int32),
                    h: wp.array(dtype=wp.float64),
                    Nf: wp.array3d(dtype=wp.float64),
                    dNf: wp.array4d(dtype=wp.float64),
                    d2Nf: wp.array4d(dtype=wp.float64),
                    wf: wp.array(dtype=wp.float64),
                    dvec: wp.array2d(dtype=wp.float64),   # DIFF
                    gbar: wp.array(dtype=wp.float64),     # DIFF
                    u: wp.array(dtype=wp.float64),        # frozen state
                    alpha: wp.float64, kappa: wp.float64,
                    r: wp.array(dtype=wp.float64)):
        fi = wp.tid()
        e = felem[fi]
        f = fface[fi]
        he = h[e]
        # NOTE: no loop-reassignment for jacS here — the taped adjoint of the
        # `jacS = jacS * half` unrolled pattern produced gradients scaled by
        # exactly 1/jacS (measured 32x at h=1/16, dim=2); wp.pow is clean.
        jacS = wp.pow(he * wp.float64(0.5), wp.float64(dim - 1))
        dscale = wp.float64(2.0) / he
        ax = f / 2
        sgn = wp.float64(1.0)
        if f % 2 == 0:
            sgn = wp.float64(-1.0)
        for q in range(nqf):
            gp = fi * nqf + q
            dS = wf[q] * jacS
            # field quantities at the GP from the frozen state
            gradun = wp.float64(0.0)
            Su = wp.float64(0.0)
            for b in range(nbf):
                ub = u[conn[e, b]]
                gradun += sgn * dNf[f, q, b, ax] * dscale * ub
                Su += shift_fn(Nf, dNf, d2Nf, f, q, b, gp, dvec, dscale, dim) * ub
            mis = Su - gbar[gp]
            for a in range(nbf):
                Na = Nf[f, q, a]
                gna = sgn * dNf[f, q, a, ax] * dscale
                Sa = shift_fn(Nf, dNf, d2Nf, f, q, a, gp, dvec, dscale, dim)
                wp.atomic_add(r, conn[e, a],
                              kappa * (-Na * gradun - gna * mis
                                       + alpha / he * Sa * mis) * dS)

    _kernel_cache[key] = sbm_dir_res
    return sbm_dir_res


def make_sbm_neumann_residual(nbf: int, nqf: int, dim: int):
    """Neumann face residual (Eq. 21 form), differentiable in
    (dvec, nvec, corr, qbar)."""
    key = ("sbm_neu_res", nbf, nqf, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    fluxshift_fn = _flux_shift_fn_for(nbf, dim)

    @wp.kernel(module="unique", enable_backward=True)
    def sbm_neu_res(felem: wp.array(dtype=wp.int32),
                    fface: wp.array(dtype=wp.int32),
                    conn: wp.array2d(dtype=wp.int32),
                    h: wp.array(dtype=wp.float64),
                    Nf: wp.array3d(dtype=wp.float64),
                    dNf: wp.array4d(dtype=wp.float64),
                    d2Nf: wp.array4d(dtype=wp.float64),
                    wf: wp.array(dtype=wp.float64),
                    dvec: wp.array2d(dtype=wp.float64),   # DIFF
                    nvec: wp.array2d(dtype=wp.float64),   # DIFF
                    corr: wp.array(dtype=wp.float64),     # DIFF
                    qbar: wp.array(dtype=wp.float64),     # DIFF
                    u: wp.array(dtype=wp.float64),        # frozen state
                    kappa: wp.float64,
                    r: wp.array(dtype=wp.float64)):
        fi = wp.tid()
        e = felem[fi]
        f = fface[fi]
        he = h[e]
        # NOTE: no loop-reassignment for jacS here — the taped adjoint of the
        # `jacS = jacS * half` unrolled pattern produced gradients scaled by
        # exactly 1/jacS (measured 32x at h=1/16, dim=2); wp.pow is clean.
        jacS = wp.pow(he * wp.float64(0.5), wp.float64(dim - 1))
        dscale = wp.float64(2.0) / he
        ax = f / 2
        sgn = wp.float64(1.0)
        if f % 2 == 0:
            sgn = wp.float64(-1.0)
        for q in range(nqf):
            gp = fi * nqf + q
            dS = wf[q] * jacS
            sflux_u = wp.float64(0.0)
            gradun_surr = wp.float64(0.0)
            for b in range(nbf):
                ub = u[conn[e, b]]
                sflux_u += fluxshift_fn(dNf, d2Nf, f, q, b, gp, dvec, nvec,
                                        dscale, dim) * ub
                gradun_surr += sgn * dNf[f, q, b, ax] * dscale * ub
            val = corr[gp] * sflux_u - gradun_surr - corr[gp] * qbar[gp]
            for a in range(nbf):
                wp.atomic_add(r, conn[e, a],
                              kappa * Nf[f, q, a] * val * dS)

    _kernel_cache[key] = sbm_neu_res
    return sbm_neu_res


# ---------------------------------------------------------------------------
def solve_adjoint(A: sp.csr_matrix, dJdu_free: np.ndarray) -> np.ndarray:
    """Tier-2 VJP #1, assembled path: lam = A^-T (dJ/du)."""
    from scipy.sparse.linalg import splu
    return splu(A.T.tocsc()).solve(np.asarray(dJdu_free, np.float64))


def _numerical_grad_fn(fn, pts, eps=1e-7):
    """Central-difference gradient of a numpy scalar field fn at pts [M,dim]."""
    g = np.zeros_like(pts)
    for d in range(pts.shape[1]):
        ep = pts.copy(); ep[:, d] += eps
        em = pts.copy(); em[:, d] -= eps
        g[:, d] = (fn(ep) - fn(em)) / (2 * eps)
    return g


def shape_gradient(problem, u_all, lam_free, oracle, meta,
                   g_fn_torch=None, q_fn_torch=None):
    """dJ/dtheta accumulated into oracle.params[i].grad (torch convention).

    problem: the SBMPoisson instance (face sets + data fns); u_all: converged
    full-node solution; lam_free: adjoint solution; meta: from assemble()
    (dir_rows for zeroing). g_fn_torch/q_fn_torch: optional torch versions of
    the boundary data (falls back to numerical gradients of the numpy fns).
    """
    dm = problem.dm
    d = dm.device
    lam = np.asarray(lam_free, np.float64).copy()
    if meta.get("dir_rows") is not None:
        lam[meta["dir_rows"]] = 0.0       # identity rows are theta-independent
    lam_full = np.asarray(dm.constraints.T @ lam)
    lam_full_d = wp.array(lam_full, dtype=wp.float64, device=d)
    u_d = wp.array(np.ascontiguousarray(u_all, np.float64),
                   dtype=wp.float64, device=d)

    for p in oracle.params:
        p.requires_grad_(True)
        if p.grad is not None:
            p.grad = None

    # ---- Dirichlet face set -------------------------------------------
    if problem.dir is not None:
        fs = problem.dir
        b = dm.bins[fs.pv]
        nqf = fs.ftab.nqf
        gbar_np = np.ascontiguousarray(
            problem.g_fn(fs.geo.xq + fs.geo.d), np.float64)
        tape = wp.Tape()
        dvec = wp.array(np.ascontiguousarray(fs.geo.d), dtype=wp.float64,
                        device=d, requires_grad=True)
        gbar = wp.array(gbar_np, dtype=wp.float64, device=d,
                        requires_grad=True)
        r = wp.zeros(dm.n_nodes, dtype=wp.float64, device=d,
                     requires_grad=True)
        k = make_sbm_dirichlet_residual(fs.ftab.nbf, nqf, dm.dim)
        with tape:
            wp.launch(k, dim=len(fs.sf.elem),
                      inputs=[fs.felem_d, fs.fface_d, b["conn"], b["h"],
                              fs.Nf_d, fs.dNf_d, fs.d2Nf_d, fs.wf_d,
                              dvec, gbar, u_d, wp.float64(problem.alpha),
                              wp.float64(problem.kappa), r],
                      device=d)
        tape.backward(grads={r: lam_full_d})
        dbar = tape.gradients[dvec].numpy()          # lam^T dR/dd
        gbarbar = tape.gradients[gbar].numpy()       # lam^T dR/dgbar
        # torch chain: theta -> d -> (d itself, gbar = g(x + d)). Oracles
        # with their own differentiable closest-point chain (TriMesh) are
        # dispatched to it — Newton+IFT on a faceted psi is the wrong tool.
        own = getattr(type(oracle), "distance_torch", None)
        d_t, n_t, ok = (oracle.distance_torch(fs.geo.xq) if own
                        else distance_torch(oracle, fs.geo.xq))
        xq_t = torch.tensor(fs.geo.xq, dtype=torch.float64)
        y_t = xq_t + d_t
        if g_fn_torch is not None:
            gbar_t = g_fn_torch(y_t)
        else:
            grad_g = _numerical_grad_fn(problem.g_fn, fs.geo.xq + fs.geo.d)
            gbar_t = (torch.tensor(grad_g, dtype=torch.float64) * d_t).sum(1)
        loss = -((d_t * torch.tensor(dbar, dtype=torch.float64)).sum()
                 + (gbar_t * torch.tensor(gbarbar, dtype=torch.float64)).sum())
        loss.backward()

    # ---- Neumann face set ---------------------------------------------
    if problem.neu is not None:
        fs = problem.neu
        b = dm.bins[fs.pv]
        nqf = fs.ftab.nqf
        qbar_np = np.ascontiguousarray(
            problem.q_fn(fs.geo.xq + fs.geo.d), np.float64)
        tape = wp.Tape()
        dvec = wp.array(np.ascontiguousarray(fs.geo.d), dtype=wp.float64,
                        device=d, requires_grad=True)
        nvec = wp.array(np.ascontiguousarray(fs.geo.n), dtype=wp.float64,
                        device=d, requires_grad=True)
        corr = wp.array(np.ascontiguousarray(fs.geo.corr), dtype=wp.float64,
                        device=d, requires_grad=True)
        qbar = wp.array(qbar_np, dtype=wp.float64, device=d,
                        requires_grad=True)
        r = wp.zeros(dm.n_nodes, dtype=wp.float64, device=d,
                     requires_grad=True)
        k = make_sbm_neumann_residual(fs.ftab.nbf, nqf, dm.dim)
        with tape:
            wp.launch(k, dim=len(fs.sf.elem),
                      inputs=[fs.felem_d, fs.fface_d, b["conn"], b["h"],
                              fs.Nf_d, fs.dNf_d, fs.d2Nf_d, fs.wf_d,
                              dvec, nvec, corr, qbar, u_d,
                              wp.float64(problem.kappa), r],
                      device=d)
        tape.backward(grads={r: lam_full_d})
        dbar = tape.gradients[dvec].numpy()
        nbar = tape.gradients[nvec].numpy()
        corrbar = tape.gradients[corr].numpy()
        qbarbar = tape.gradients[qbar].numpy()
        own = getattr(type(oracle), "distance_torch", None)
        d_t, n_grad_t, ok = (oracle.distance_torch(fs.geo.xq) if own
                             else distance_torch(oracle, fs.geo.xq))
        sgn = 1.0 if fs.geo.domain == "inside" else -1.0
        n_t = sgn * n_grad_t                       # out of Omega
        from ..octree.lookup import face_offsets
        ntilde = face_offsets(dm.dim).astype(np.float64)[fs.sf.face]
        ntilde_t = torch.tensor(np.repeat(ntilde, nqf, axis=0),
                                dtype=torch.float64)
        corr_t = (ntilde_t * n_t).sum(1)
        xq_t = torch.tensor(fs.geo.xq, dtype=torch.float64)
        y_t = xq_t + d_t
        if q_fn_torch is not None:
            qbar_t = q_fn_torch(y_t)
        else:
            grad_q = _numerical_grad_fn(problem.q_fn, fs.geo.xq + fs.geo.d)
            qbar_t = (torch.tensor(grad_q, dtype=torch.float64) * d_t).sum(1)
        loss = -((d_t * torch.tensor(dbar, dtype=torch.float64)).sum()
                 + (n_t * torch.tensor(nbar, dtype=torch.float64)).sum()
                 + (corr_t * torch.tensor(corrbar, dtype=torch.float64)).sum()
                 + (qbar_t * torch.tensor(qbarbar, dtype=torch.float64)).sum())
        loss.backward()

    return {id(p): (p.grad.clone() if p.grad is not None else None)
            for p in oracle.params}


def kappa_gradient(meta, u_free, lam_free) -> float:
    """dJ/dkappa = -lam^T (A1 u - bg1) — scalar-kappa linearity (meta from
    assemble; field kappa raises)."""
    if meta.get("field_kappa"):
        raise NotImplementedError(
            "field-kappa gradient arrives with the M2 closure interface")
    lam = np.asarray(lam_free, np.float64).copy()
    if meta.get("dir_rows") is not None:
        lam[meta["dir_rows"]] = 0.0
    dRdk = meta["A1"] @ np.asarray(u_free) - meta["bg1"]
    return float(-(lam @ dRdk))


# ---------------------------------------------------------------------------
def volume_qoi(dm):
    """J = int_Omega~ u dV. Returns (eval_fn(u_all) -> J, dJdu_free)."""
    from ..physics.poisson import make_load_kernel, gauss_points
    d = dm.device
    xq_by_bin = gauss_points(dm.mesh, dm.tables_by_p)
    m_full = wp.zeros(dm.n_nodes, dtype=wp.float64, device=d)
    for pv, b in dm.bins.items():
        ones = wp.array(np.ones(len(xq_by_bin[pv])), dtype=wp.float64,
                        device=d)
        lk = make_load_kernel(b["nbf"], b["nqp"], dm.dim)
        wp.launch(lk, dim=len(b["eids"]),
                  inputs=[b["conn"], b["h"], b["N"], b["w"], ones, m_full],
                  device=d)
    m = m_full.numpy()                             # mass row-sums (dJ/du_all)
    dJdu_free = np.asarray(dm.constraints.T.T @ m)

    def eval_fn(u_all):
        return float(m @ np.asarray(u_all))
    return eval_fn, dJdu_free


def probe_qoi(dm, pts, targets):
    """J = sum_i (u(x_i) - u*_i)^2. Returns (eval_fn(u_all) -> J,
    dJdu_free_fn(u_all) -> dJ/du_free)."""
    from ..mesh.pointeval import point_eval_weights
    W = point_eval_weights(dm.mesh, pts)          # [Npts, Nn]
    T = dm.constraints.T.tocsr()
    targets = np.asarray(targets, np.float64)

    def eval_fn(u_all):
        rvec = W @ np.asarray(u_all) - targets
        return float(rvec @ rvec)

    def dJdu_free_fn(u_all):
        rvec = W @ np.asarray(u_all) - targets
        return np.asarray(T.T @ (W.T @ (2.0 * rvec)))
    return eval_fn, dJdu_free_fn
