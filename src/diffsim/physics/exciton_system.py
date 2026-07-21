"""SP-1 B4: monolithic 5-field XDD Newton system.

Couples the B1 Poisson brick, the B2 carrier drift-diffusion bricks and the
B3 exciton reaction-diffusion bricks (all in ``exciton_dd.py``) with the A3
closures (``exciton_closures.py``) into ONE Newton solve over the node-major
dof vector

    u = [φ̂, n̂, p̂, X̂_D, X̂_A]   (5 dofs / node, node-major).

WHY A SEPARATE FILE (documented design fork): ``exciton_dd.py`` is ~730 lines
of brick kernels + assemblers; folding the whole Newton driver (residual,
full Jacobian, positivity line search, BDF wrapper, block-GS mode, electrode
BC helper) into it would push it well past the ~1500-line ceiling the brief
sets.  This module imports the bricks and composes them.

CARRIER DRIFT — CONSERVATIVE signed form (CPU DDEquation.h parity; corrects a
brief-introduced sign/form defect found by Block C's physics gate).  The drift
is assembled in CONSERVATIVE weak form via the B2 carrier kernel with the
signed drift-weight aq = sign·μ̂∇φ̂ (sign=−1 electrons → residual −μ̂_n n̂(∇φ̂·∇w),
eqm n̂∝e^{+φ̂}; sign=+1 holes → +μ̂_p p̂(∇φ̂·∇w), eqm p̂∝e^{−φ̂}).  SUPG upwind
velocity U = −aq (U_n=+μ̂∇φ̂, U_p=−μ̂∇φ̂).  With this fix the electrode BCs
(n̂=1 at the +φ̂ anode) and the continuation IC (n̂=e^{+(φ̂−φ̂a)}) ALIGN with the
kernel equilibrium — see bilayer_electrode_bcs / continuation_ic.

RESIDUAL (strong form → Galerkin, per field; hist = BDF history term)
---------------------------------------------------------------------
  φ̂ :  λ²ε̂ K φ̂  −  M(p̂ − n̂)                                   [B1]
  n̂ :  [σ M + μ̂_n K + C_φ(−) + SUPG] n̂  −  M(D̂ − R̂)  − hist_n   [B2]
  p̂ :  [σ M + μ̂_p K + C_φ(+) + SUPG] p̂  −  M(D̂ − R̂)  − hist_p
       where C_φ(±)ĉ ≡ ±μ̂ ĉ(∇φ̂·∇w) is the conservative drift matrix (CPU
       A22 = μ̂_n(K − C_φ), A33 = μ̂_p(K + C_φ) with C_φ the ∇φ̂-weighted block)
  X̂_i: [σ_tot,i M + μ̂_{X,i} K] X̂_i  −  M(Ĝ_i + R̂_feed,i)  − hist_X

  with  D̂ = k̂_d X̂_D + k̂_a X̂_A     (excitons dissociate → free carriers)
        R̂ = γ̂ n̂ p̂                  (Langevin recombination → back to excitons)
  Carrier net source  = D̂ − R̂ for BOTH n̂ and p̂ (electron-hole pairs created
  by dissociation, annihilated by recombination — symmetric in this model).
  Exciton feed  R̂_feed,i = R̂  (recombined pairs re-form excitons), sink is the
  k̂_i X̂_i term already inside σ_tot,i.

JACOBIAN — the full Newton matrix (this module's core).  Assembled as a 5×5
block sparse system (scipy ``bmat``); every block is n_free × n_free.
  diagonal:
    J_φφ = λ²ε̂ K
    J_nn = σ M + μ̂_n K + C_φ(−) + SUPG + γ̂ p̂ · M    (∂R̂/∂n̂ = γ̂p̂)
    J_pp = σ M + μ̂_p K + C_φ(+) + SUPG + γ̂ n̂ · M
    J_XiXi = σ_tot,i M + μ̂_{X,i} K
  off-diagonal:
    J_φn = +M            J_φp = −M                       (B1 source)
    J_nφ = CONSERVATIVE drift cross-term  ∫ sign·μ̂ n̂ (∇δφ̂·∇w) = −μ̂_n n̂(∇δφ̂·∇w)
             — a weighted STIFFNESS block (CPU A21 = −(∇w, μ̂_n n̂ ∇δφ̂)); the
             p-row mirror J_pφ = +μ̂_p p̂(∇δφ̂·∇w) = +(∇w, μ̂_p p̂ ∇δφ̂) (CPU A31)
           + dissociation-field coupling  −∂D̂/∂|∇φ̂| (∇φ̂·∇δφ̂/|∇φ̂|)
             via A3 dk_dgrad; |∇φ̂|=0 → dk=0 (b→0 limit)
    J_np = +γ̂ n̂ · M      (∂(−R̂)/∂p̂ enters n-row source as −(−γ̂n̂)=+γ̂n̂ M)
    J_n,Xi = −k̂_i · M    (∂(−D̂)/∂X̂_i)
    p-row mirrors n-row (drift sign flips; same R̂ derivatives)
    J_Xiφ = +∂k̂_i/∂|∇φ̂| X̂_i · (∇φ̂·∇δφ̂/|∇φ̂|)   (sink side; positive)
    J_Xi,n = −γ̂ p̂ · M    J_Xi,p = −γ̂ n̂ · M       (−R̂_feed derivatives)

FROZEN-τ OMISSION (documented, brief-sanctioned): the SUPG stabilisation
parameter τ_M depends on |a| = μ̂|∇φ̂|, hence on φ̂.  We freeze τ per Newton
iteration and DO NOT differentiate it w.r.t. φ̂ in J_nφ / J_pφ.  The residual
is exact, so Newton still converges; at worst the τ-coupling loses exact
quadratic rate.  Gate 2 measures and reports the observed rate.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .exciton_dd import (
    make_xdd_poisson_Ae,
    make_xdd_exciton_Ae,
    make_xdd_carrier_Ae,
    make_xdd_carrier_be,
)

import warp as wp

# node-major field order
IPHI, IN, IP, IXD, IXA = 0, 1, 2, 3, 4
NDOF = 5


# ══════════════════════════════════════════════════════════════════════════════
# Host-side nodal → Gauss-point interpolation (numpy; warp-CPU friendly)
# ══════════════════════════════════════════════════════════════════════════════

def _gp_value(dm, u_full):
    """Interpolate a full nodal scalar u_full[n_nodes] to GP values per bin.

    Returns dict {p: ndarray[ne*nqp]} matching the kernels' e*nqp+q order.
    """
    out = {}
    for pv, b in dm.bins.items():
        conn = dm.mesh.conn_of[pv]                 # [ne, nbf]
        N = dm.tables_by_p[pv].N                   # [nqp, nbf]
        ue = u_full[conn]                          # [ne, nbf]
        # vals[e, q] = sum_a N[q,a] ue[e,a]
        vals = np.einsum("qa,ea->eq", N, ue)       # [ne, nqp]
        out[pv] = vals.reshape(-1)
    return out


def _gp_grad(dm, u_full):
    """Physical gradient of a full nodal scalar at GPs, per bin.

    Returns dict {p: ndarray[ne*nqp, dim]}.  Reference dN scaled by 2/he
    (affine-cube metric, matches the assembly kernels' dscale).
    """
    h_all = dm.mesh.tree.h()
    out = {}
    for pv, b in dm.bins.items():
        conn = dm.mesh.conn_of[pv]                 # [ne, nbf]
        dN = dm.tables_by_p[pv].dN                 # [nqp, nbf, dim]
        eids = dm.mesh.bins[pv]
        he = h_all[eids]                           # [ne]
        dscale = (2.0 / he)[:, None, None]         # [ne,1,1]
        ue = u_full[conn]                          # [ne, nbf]
        # grad[e,q,d] = sum_a dN[q,a,d]*dscale[e] * ue[e,a]
        g = np.einsum("qad,ea->eqd", dN, ue)       # [ne, nqp, dim]
        g = g * dscale                             # broadcast 2/he
        out[pv] = g.reshape(-1, dm.dim)
    return out


def _grad_mag(grad_gp):
    """|∇φ̂| per bin from a grad dict, guarded away from 0 for division only."""
    return {pv: np.sqrt(np.sum(g * g, axis=1)) for pv, g in grad_gp.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Scalar block assemblers (all return raw n_nodes × n_nodes; constraint applied
# once at the end by the 5×5 composer).  Reuse the B1–B3 kernels.
# ══════════════════════════════════════════════════════════════════════════════

def _mass_block(dm, w_gp):
    """GP-weighted mass matrix  M[a,b] += w_q N_a N_b dJxW  (raw, full space).

    Reuses make_xdd_exciton_Ae with sigma_gp=w_gp, mu_gp=0 → pure mass.
    """
    d = dm.device
    rows, cols, vals = [], [], []
    for pv, b in dm.bins.items():
        conn = dm.mesh.conn_of[pv].astype(np.int64)
        ne, nbf = conn.shape
        nqp = b["nqp"]
        w_np = np.ascontiguousarray(w_gp[pv], np.float64)
        zero = np.zeros(ne * nqp)
        sig_d = wp.array(w_np, dtype=wp.float64, device=d)
        mu_d = wp.array(zero, dtype=wp.float64, device=d)
        Ae = wp.zeros((ne, nbf, nbf), dtype=wp.float64, device=d)
        kA = make_xdd_exciton_Ae(nbf, nqp, dm.dim)
        wp.launch(kA, dim=ne,
                  inputs=[b["conn"], b["h"], b["N"], b["dN"], b["w"],
                          sig_d, mu_d, Ae], device=d)
        Aeh = Ae.numpy()
        rows.append(np.repeat(conn, nbf, axis=1).ravel())
        cols.append(np.tile(conn, (1, nbf)).ravel())
        vals.append(Aeh.ravel())
    return sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dm.n_nodes, dm.n_nodes)).tocsr()


def _poisson_block(dm, eps_gp, lam2):
    """λ² (ε̂ ∇N_b·∇N_a) dJxW  (raw, full space) — reuse the B1 Ae kernel."""
    d = dm.device
    rows, cols, vals = [], [], []
    for pv, b in dm.bins.items():
        conn = dm.mesh.conn_of[pv].astype(np.int64)
        ne, nbf = conn.shape
        nqp = b["nqp"]
        eps_d = wp.array(np.ascontiguousarray(eps_gp[pv], np.float64),
                         dtype=wp.float64, device=d)
        Ae = wp.zeros((ne, nbf, nbf), dtype=wp.float64, device=d)
        kA = make_xdd_poisson_Ae(nbf, nqp, dm.dim)
        wp.launch(kA, dim=ne,
                  inputs=[b["conn"], b["h"], b["dN"], b["w"],
                          eps_d, wp.float64(lam2), Ae], device=d)
        Aeh = Ae.numpy()
        rows.append(np.repeat(conn, nbf, axis=1).ravel())
        cols.append(np.tile(conn, (1, nbf)).ravel())
        vals.append(Aeh.ravel())
    return sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dm.n_nodes, dm.n_nodes)).tocsr()


def _carrier_block(dm, aq_gp, mu_gp, sigma, sig2tau, supg):
    """[σ M + (a·∇N_b)N_a + μ̂ ∇N_b·∇N_a + SUPG]  (raw, full space).

    Reuse the B2 carrier Ae kernel.  sign is baked into aq_gp by the caller.
    """
    d = dm.device
    rows, cols, vals = [], [], []
    for pv, b in dm.bins.items():
        conn = dm.mesh.conn_of[pv].astype(np.int64)
        ne, nbf = conn.shape
        nqp = b["nqp"]
        aq_d = wp.array(np.ascontiguousarray(aq_gp[pv], np.float64),
                        dtype=wp.float64, device=d)
        mu_d = wp.array(np.ascontiguousarray(mu_gp[pv], np.float64),
                        dtype=wp.float64, device=d)
        Ae = wp.zeros((ne, nbf, nbf), dtype=wp.float64, device=d)
        kA = make_xdd_carrier_Ae(nbf, nqp, dm.dim)
        wp.launch(kA, dim=ne,
                  inputs=[b["conn"], b["h"], b["N"], b["dN"], b["lapN"],
                          b["w"], aq_d, mu_d,
                          wp.float64(sigma), wp.float64(sig2tau),
                          wp.float64(supg), Ae], device=d)
        Aeh = Ae.numpy()
        rows.append(np.repeat(conn, nbf, axis=1).ravel())
        cols.append(np.tile(conn, (1, nbf)).ravel())
        vals.append(Aeh.ravel())
    return sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dm.n_nodes, dm.n_nodes)).tocsr()


def _exciton_block(dm, mu_gp, sigma_gp):
    """[σ_tot(x) M + μ̂_X K]  (raw, full space) — reuse the B3 exciton Ae."""
    d = dm.device
    rows, cols, vals = [], [], []
    for pv, b in dm.bins.items():
        conn = dm.mesh.conn_of[pv].astype(np.int64)
        ne, nbf = conn.shape
        nqp = b["nqp"]
        mu_d = wp.array(np.ascontiguousarray(mu_gp[pv], np.float64),
                        dtype=wp.float64, device=d)
        sig_d = wp.array(np.ascontiguousarray(sigma_gp[pv], np.float64),
                         dtype=wp.float64, device=d)
        Ae = wp.zeros((ne, nbf, nbf), dtype=wp.float64, device=d)
        kA = make_xdd_exciton_Ae(nbf, nqp, dm.dim)
        wp.launch(kA, dim=ne,
                  inputs=[b["conn"], b["h"], b["N"], b["dN"], b["w"],
                          sig_d, mu_d, Ae], device=d)
        Aeh = Ae.numpy()
        rows.append(np.repeat(conn, nbf, axis=1).ravel())
        cols.append(np.tile(conn, (1, nbf)).ravel())
        vals.append(Aeh.ravel())
    return sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dm.n_nodes, dm.n_nodes)).tocsr()


def _load_block(dm, aq_gp, mu_gp, fq_gp, sig2tau, supg):
    """(f, N_a) + τ_M(a·∇N_a, f)  (raw, full space) — reuse the carrier be."""
    d = dm.device
    F = np.zeros(dm.n_nodes)
    for pv, b in dm.bins.items():
        conn = dm.mesh.conn_of[pv].astype(np.int64)
        ne, nbf = conn.shape
        nqp = b["nqp"]
        aq_d = wp.array(np.ascontiguousarray(aq_gp[pv], np.float64),
                        dtype=wp.float64, device=d)
        mu_d = wp.array(np.ascontiguousarray(mu_gp[pv], np.float64),
                        dtype=wp.float64, device=d)
        fq_d = wp.array(np.ascontiguousarray(fq_gp[pv], np.float64),
                        dtype=wp.float64, device=d)
        be = wp.zeros((ne, nbf), dtype=wp.float64, device=d)
        kb = make_xdd_carrier_be(nbf, nqp, dm.dim)
        wp.launch(kb, dim=ne,
                  inputs=[b["conn"], b["h"], b["N"], b["dN"], b["w"],
                          aq_d, mu_d, fq_d,
                          wp.float64(sig2tau), wp.float64(supg), be],
                  device=d)
        np.add.at(F, conn.ravel(), be.numpy().ravel())
    return F


def _drift_cross_block(dm, coeff_gp, gradphi_gp):
    """The A21 drift cross-term  ∫ coeff_q (∇φ̂·∇N_b) N_a dJxW  (raw, full).

    Used for ∂(carrier-row)/∂φ̂ = ±μ̂ c ∇δφ̂·∇w with test/trial swapped so it
    linearises the advection a·∇c = sign·μ̂(∇φ̂·∇c).  Here we differentiate
    a·∇c w.r.t. φ̂: δa·∇c = sign·μ̂(∇δφ̂·∇c), and moving to the weak (N_a=w,
    N_b=δφ̂ trial) form gives ∫ (sign·μ̂ c_q)(∇N_b · ∇... ) — see build_jacobian
    for the exact assembly; coeff_gp already carries sign·μ̂·(∂c/∂-dir) pieces.

    Kernel: for each (a,b): sum_d coeff_q * gradphi_q[d]?  NO — this block is
    ∫ w_q (g_q · ∇N_b) N_a dJxW with g_q a GP vector field.  We implement it
    inline here via numpy element assembly (small, host, warp-CPU budget ok).
    """
    h_all = dm.mesh.tree.h()
    rows, cols, vals = [], [], []
    for pv, b in dm.bins.items():
        conn = dm.mesh.conn_of[pv].astype(np.int64)
        ne, nbf = conn.shape
        nqp = b["nqp"]
        N = dm.tables_by_p[pv].N                    # [nqp, nbf]
        dN = dm.tables_by_p[pv].dN                  # [nqp, nbf, dim]
        w = dm.tables_by_p[pv].w                    # [nqp]
        eids = dm.mesh.bins[pv]
        he = h_all[eids]                            # [ne]
        half = 0.5 * he
        jac = half ** dm.dim                        # [ne]
        dscale = 2.0 / he                           # [ne]
        g = gradphi_gp[pv].reshape(ne, nqp, dm.dim)  # vector field at GPs
        c = coeff_gp[pv].reshape(ne, nqp)            # scalar weight at GPs
        # (g·∇N_b) : [ne,nqp,nbf] = sum_d g[e,q,d]*dN[q,b,d]*dscale[e]
        gdotdNb = np.einsum("eqd,qbd->eqb", g, dN) * dscale[:, None, None]
        # Ae[e,a,b] = sum_q w[q]*jac[e]*c[e,q]*N[q,a]*gdotdNb[e,q,b]
        wq = w[None, :] * jac[:, None] * c           # [ne,nqp]
        Ae = np.einsum("eq,qa,eqb->eab", wq, N, gdotdNb)  # [ne,nbf,nbf]
        rows.append(np.repeat(conn, nbf, axis=1).ravel())
        cols.append(np.tile(conn, (1, nbf)).ravel())
        vals.append(Ae.ravel())
    return sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dm.n_nodes, dm.n_nodes)).tocsr()


def _supg_drift_cross_block(dm, coeff_gp, g_gp, aq_gp, tau_gp):
    """SUPG counterpart of _drift_cross_block for a source-term φ̂ coupling.

    A source term s(φ̂) enters the carrier load with the augmented test
    (N_a + τ a·∇N_a).  The Galerkin part ∫ coeff (g·∇N_b) N_a is
    _drift_cross_block; this adds the SUPG part ∫ coeff (g·∇N_b) τ(a·∇N_a).
    Used for the dissociation-field coupling ∂(−D̂)/∂φ̂ (g = ∇φ̂).
    """
    h_all = dm.mesh.tree.h()
    rows, cols, vals = [], [], []
    for pv, b in dm.bins.items():
        conn = dm.mesh.conn_of[pv].astype(np.int64)
        ne, nbf = conn.shape
        nqp = b["nqp"]
        dN = dm.tables_by_p[pv].dN
        wt = dm.tables_by_p[pv].w
        eids = dm.mesh.bins[pv]
        he = h_all[eids]
        jac = (0.5 * he) ** dm.dim
        dscale = 2.0 / he
        g = g_gp[pv].reshape(ne, nqp, dm.dim)
        c = coeff_gp[pv].reshape(ne, nqp)
        a = aq_gp[pv].reshape(ne, nqp, dm.dim)
        tau_e = tau_gp[pv].reshape(ne, nqp)
        gdotdNb = np.einsum("eqd,qbd->eqb", g, dN) * dscale[:, None, None]
        agN = np.einsum("eqd,qad->eqa", a, dN) * dscale[:, None, None]
        base = wt[None, :] * jac[:, None] * c * tau_e     # [ne,nqp]
        Ae = np.einsum("eq,eqa,eqb->eab", base, agN, gdotdNb)
        rows.append(np.repeat(conn, nbf, axis=1).ravel())
        cols.append(np.tile(conn, (1, nbf)).ravel())
        vals.append(Ae.ravel())
    return sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dm.n_nodes, dm.n_nodes)).tocsr()


_CI_F = 36.0   # matches physics.vms.CI_F (host mirror of the SUPG τ metric)


def _tau_gp(dm, aq_gp, mu_gp, sig2tau, supg):
    """SUPG τ_M at every GP per bin (host mirror of vms.tau_m_metric).

    τ_M = 1/√(sig2 + 4|a|²/h² + 36 μ̂² dim (2/h)⁴);  returns supg·τ_M.
    """
    h_all = dm.mesh.tree.h()
    out = {}
    for pv, b in dm.bins.items():
        nqp = b["nqp"]
        eids = dm.mesh.bins[pv]
        he = h_all[eids]                               # [ne]
        he_gp = np.repeat(he, nqp)                     # [ne*nqp]
        a = aq_gp[pv]                                  # [ngp, dim]
        amag2 = np.sum(a * a, axis=1)
        nu = mu_gp[pv]
        uGu = 4.0 * amag2 / (he_gp * he_gp)
        GG = dm.dim * (2.0 / he_gp) ** 4
        tau = 1.0 / np.sqrt(sig2tau + uGu + _CI_F * nu * nu * GG)
        out[pv] = supg * tau
    return out


def _supg_mass_block(dm, w_gp, aq_gp, mu_gp, sig2tau, supg):
    """SUPG-augmented source-coupling block for the carrier rows.

    The carrier load uses the augmented test function (N_a + τ_M U·∇N_a) with
    the drift velocity U = −aq (CONSERVATIVE form; aq = sign·μ̂∇φ̂ is the drift
    WEIGHT, the carrier be kernel uses U = −aq for the SUPG augmentation):
        ∫ (N_a + τ_M U·∇N_a) w_q N_b dJxW
    where w_q = ∂f/∂field is the GP weight from differentiating the source.
    Galerkin part = _mass_block(w); this adds the τ_M(U·∇N_a) part so the
    source-coupling Jacobian matches the SUPG-consistent residual load.  The
    caller passes aq (the drift weight); U = −aq is formed here — τ_M uses |U|
    = |aq| (magnitude, sign-invariant).
    """
    galerkin = _mass_block(dm, w_gp)
    if supg == 0.0:
        return galerkin
    tau = _tau_gp(dm, aq_gp, mu_gp, sig2tau, supg)
    h_all = dm.mesh.tree.h()
    rows, cols, vals = [], [], []
    for pv, b in dm.bins.items():
        conn = dm.mesh.conn_of[pv].astype(np.int64)
        ne, nbf = conn.shape
        nqp = b["nqp"]
        N = dm.tables_by_p[pv].N
        dN = dm.tables_by_p[pv].dN
        wt = dm.tables_by_p[pv].w
        eids = dm.mesh.bins[pv]
        he = h_all[eids]
        jac = (0.5 * he) ** dm.dim
        dscale = 2.0 / he
        a = -aq_gp[pv].reshape(ne, nqp, dm.dim)   # U = −aq (drift velocity)
        wgt = w_gp[pv].reshape(ne, nqp)
        tau_e = tau[pv].reshape(ne, nqp)
        # U·∇N_a : [ne,nqp,nbf]
        agN = np.einsum("eqd,qad->eqa", a, dN) * dscale[:, None, None]
        # block[e,a,b] = sum_q (jac w_q)*tau*wgt * (a·∇N_a) * N_b
        coeff = wt[None, :] * jac[:, None] * tau_e * wgt      # [ne,nqp]
        Ae = np.einsum("eq,eqa,qb->eab", coeff, agN, N)       # [ne,nbf,nbf]
        rows.append(np.repeat(conn, nbf, axis=1).ravel())
        cols.append(np.tile(conn, (1, nbf)).ravel())
        vals.append(Ae.ravel())
    supg_part = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dm.n_nodes, dm.n_nodes)).tocsr()
    return galerkin + supg_part


def _supg_drift_phi_block(dm, sign_mu_gp, c_gp, gradc_gp, tau_gp):
    """Exact SUPG-advection dependence of a carrier row on φ̂ (CONSERVATIVE form).

    The SUPG residual  τ (U·∇N_a)(σ c + U·∇c − μ̂ ∇²c)  with drift velocity
    U = −aq = −sign·μ̂·∇φ̂ depends on φ̂ through U in BOTH the test augmentation
    (U·∇N_a) and the strong-residual advection (U·∇c).  Freezing τ (the
    sanctioned omission), δU = −sign·μ̂·∇δφ̂, giving two bilinear (test N_a,
    trial N_b=δφ̂) terms:

      A: τ (δU·∇N_a) res_c  =  τ (−sign·μ̂)(∇N_a·∇N_b) res_c
      B: τ (U·∇N_a)(δU·∇c)  =  τ (U·∇N_a)(−sign·μ̂)(∇N_b·∇c)

    Inputs are GP fields: sign_mu_gp = −sign·μ̂ (the δU coefficient), c_gp["aq"]
    = U the drift velocity, gradc = ∇c, and c_gp["res"] the frozen strong
    residual σc + U·∇c − μ̂∇²c − f.  tau_gp already carries supg·τ_M (0 → block 0).
    """
    h_all = dm.mesh.tree.h()
    rows, cols, vals = [], [], []
    for pv, b in dm.bins.items():
        conn = dm.mesh.conn_of[pv].astype(np.int64)
        ne, nbf = conn.shape
        nqp = b["nqp"]
        dN = dm.tables_by_p[pv].dN
        wt = dm.tables_by_p[pv].w
        eids = dm.mesh.bins[pv]
        he = h_all[eids]
        jac = (0.5 * he) ** dm.dim
        dscale = 2.0 / he
        smu = sign_mu_gp[pv].reshape(ne, nqp)
        resc = c_gp["res"][pv].reshape(ne, nqp)          # frozen strong residual
        gradc = gradc_gp[pv].reshape(ne, nqp, dm.dim)
        a = c_gp["aq"][pv].reshape(ne, nqp, dm.dim)
        tau_e = tau_gp[pv].reshape(ne, nqp)
        # ∇N_a·∇N_b : [ne,nqp,nbf,nbf]
        gNa = dN[None] * dscale[:, None, None, None]      # [ne,nqp,nbf,dim]
        gNgN = np.einsum("eqad,eqbd->eqab", gNa, gNa)     # [ne,nqp,nbf,nbf]
        # a·∇N_a : [ne,nqp,nbf]
        agN = np.einsum("eqd,eqad->eqa", a, gNa)
        # ∇N_b·∇c : [ne,nqp,nbf]
        gNgc = np.einsum("eqbd,eqd->eqb", gNa, gradc)
        base = wt[None, :] * jac[:, None] * tau_e * smu   # [ne,nqp]
        # A: base * res_c * (∇N_a·∇N_b)
        Aterm = np.einsum("eq,eq,eqab->eab", base, resc, gNgN)
        # B: base * (a·∇N_a)(∇N_b·∇c)
        Bterm = np.einsum("eq,eqa,eqb->eab", base, agN, gNgc)
        Ae = Aterm + Bterm
        rows.append(np.repeat(conn, nbf, axis=1).ravel())
        cols.append(np.tile(conn, (1, nbf)).ravel())
        vals.append(Ae.ravel())
    return sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dm.n_nodes, dm.n_nodes)).tocsr()


def _gp_lap(dm, u_full):
    """Physical Laplacian ∇²u of a full nodal scalar at GPs, per bin.

    Uses the reference lapN table scaled by (2/he)² (affine-cube metric,
    matches the carrier kernel's lapNtab*dscale²).
    """
    h_all = dm.mesh.tree.h()
    out = {}
    for pv, b in dm.bins.items():
        conn = dm.mesh.conn_of[pv]
        lapN = dm.tables_by_p[pv].lapN                # [nqp, nbf]
        eids = dm.mesh.bins[pv]
        he = h_all[eids]
        dscale2 = ((2.0 / he) ** 2)[:, None]          # [ne,1]
        ue = u_full[conn]                             # [ne, nbf]
        lap = np.einsum("qa,ea->eq", lapN, ue) * dscale2
        out[pv] = lap.reshape(-1)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# XDDSystem — the monolithic 5-field Newton driver
# ══════════════════════════════════════════════════════════════════════════════

class XDDSystem:
    """Monolithic 5-field XDD Newton solver.

    Node-major dofs [φ̂, n̂, p̂, X̂_D, X̂_A].  The state is stored as a
    dict of FULL nodal vectors (one per field, length n_nodes); the free-dof
    solve uses the scalar constraint operator ``T`` applied per block.

    Coefficient GP fields (all dict {p: ndarray}), supplied once at
    construction (they are φ̂-independent for R0):
        eps_gp     ε̂(x)          (Poisson permittivity)
        mu_n_gp    μ̂_n(x)        mu_p_gp μ̂_p(x)
        mu_xd_gp   μ̂_{X,D}(x)    mu_xa_gp μ̂_{X,A}(x)
        dist_gp    signed distance at GPs (for the closures)
    Closures: ``langevin`` (LangevinRecombination), ``onsager``
    (OnsagerBraunDissociation), plus scalar nondim rates 1/τ̂_{x,D}, 1/τ̂_{x,A}.
    Generation Ĝ_D, Ĝ_A supplied as GP fields per step (or zero).
    """

    def __init__(self, dm, *, lam2, eps_gp, mu_n_gp, mu_p_gp,
                 mu_xd_gp, mu_xa_gp, dist_gp,
                 langevin=None, onsager=None,
                 tau_inv_d=0.0, tau_inv_a=0.0,
                 supg=1.0, newton_tol=1e-10, atol=1e-12,
                 linsolve=None, linsolver="splu", carrier_vars="primal",
                 assembly="auto", index_width="auto", val_dtype="fp64"):
        self.dm = dm
        self.lam2 = float(lam2)
        self.eps_gp = eps_gp
        self.mu_n_gp = mu_n_gp
        self.mu_p_gp = mu_p_gp
        self.mu_xd_gp = mu_xd_gp
        self.mu_xa_gp = mu_xa_gp
        self.dist_gp = dist_gp
        self.langevin = langevin
        self.onsager = onsager
        self.tau_inv_d = float(tau_inv_d)
        self.tau_inv_a = float(tau_inv_a)
        self.supg = float(supg)
        self.newton_tol = float(newton_tol)
        self.atol = float(atol)

        # Linear-solve backend.  `linsolve` (an explicit (A, b)->x callable)
        # overrides everything (test hook / bespoke driver).  Otherwise
        # `linsolver` selects a stock backend:
        #   "splu"  — host scipy SuperLU (prototype default; PARITY/CPU mode
        #             unchanged — this is the pre-E5-fix behaviour verbatim).
        #   "cudss" — NVIDIA cuDSS (GPU sparse direct LU) via nvmath-python,
        #             the GPU analogue of splu for this 2-D NONSYMMETRIC 5-field
        #             Jacobian.  Plan ONCE on the first Newton iterate, then
        #             reset_operands + refactorize per iterate (the Jacobian
        #             sparsity is FIXED across Newton iters — same mesh, same
        #             5-field coupling), the M1d refactorization-loop case.
        #             Uses the PLAIN DirectSolverOptions(blocking=True) path
        #             (multiphase's lesson: the mtlayer_gomp planning layer
        #             leaks a gomp thread-team per call; these 2-D systems plan
        #             fast single-threaded).  Discarded plans are EXPLICITLY
        #             .free()'d (double-free guard).  nvmath/CUDA absent ->
        #             clear BackendError at first solve (Mac/CPU parity mode
        #             never trips it).
        if linsolver not in ("splu", "cudss"):
            raise ValueError(f"linsolver must be 'splu' or 'cudss', "
                             f"got {linsolver!r}")
        self.linsolver = linsolver
        self._cudss = None          # nvmath DirectSolver plan (per fixed pattern)
        self._cudss_nnz = -1        # sparsity fingerprint of the planned matrix
        if linsolve is not None:
            self.linsolve = linsolve
        elif linsolver == "cudss":
            self.linsolve = self._cudss_linsolve
        else:
            self.linsolve = lambda A, r: splu(A.tocsc()).solve(r)

        if carrier_vars not in ("primal", "log"):
            raise ValueError(f"carrier_vars must be 'primal' or 'log', "
                             f"got {carrier_vars!r}")
        # "log": the carrier variables n̂,p̂ are iterated as u=ln n̂, v=ln p̂
        # via a Newton-level chain rule (Slotboom-class positivity, no re-derived
        # weak form).  The PUBLIC state stays primal (n̂,p̂ full nodal vectors);
        # the transformation lives entirely inside the Newton loop.  Positivity
        # is structural (n̂=e^u>0 for any δu), so the fraction-to-boundary guard
        # never truncates a carrier step — exposed via ``self.guard_trunc``.
        self.carrier_vars = carrier_vars
        self._log_carriers = (carrier_vars == "log")
        self.guard_trunc = 0    # counts carrier steps clipped by the guard (log)

        self.T = dm.constraints.T.tocsr()
        self.free = dm.constraints.free_nodes
        self.n_free = self.T.shape[1]

        # Task #35: assembly backend.  "host" is the R0 scipy/bmat path (the
        # parity reference); "device" routes solve_newton's Jacobian+residual
        # through the warp DeviceNSAssembler path (exciton_device.py — node-
        # major dofs, blockmasked node-pattern CSR, device closures).  "auto"
        # (default) picks device on CUDA, host on CPU — CPU behavior is
        # unchanged, GPU production gets the E5 host-stage fix by default.
        # Device assembly requires identity constraints (uniform meshes — the
        # node-pattern scatter contract); "auto" falls back to host otherwise,
        # a forced "device" raises loudly.
        if assembly not in ("auto", "host", "device"):
            raise ValueError(f"assembly must be 'auto'|'host'|'device', "
                             f"got {assembly!r}")
        T = self.T
        identity_T = (T.shape[0] == T.shape[1]) and (
            T != sp.identity(T.shape[0], format="csr")).nnz == 0
        if assembly == "device":
            if not identity_T:
                from ..errors import BackendError
                raise BackendError(
                    "assembly='device' requires identity constraints "
                    "(uniform mesh); this mesh has hanging-node constraints "
                    "— use assembly='host'")
            self._assembly_device = True
        elif assembly == "host":
            self._assembly_device = False
        else:
            self._assembly_device = (str(dm.device).startswith("cuda")
                                     and identity_T)
        self.assembly = "device" if self._assembly_device else "host"
        self.index_width = index_width
        # Task #36 (8j): fp32-storage + FP64-IR opt-in (default fp64 =
        # unchanged).  Threaded to the device assembler's val_dtype knob;
        # the cuDSS solve then factors fp32 + refines in fp64.
        self.val_dtype = val_dtype
        self._dev_asm = None        # lazy XDDDeviceAssembler
        self._cudss_dev = None      # zero-copy device-CSR cuDSS plan
        self._b_t = None
        self._b_t32 = None          # stable fp32 rhs (IR correction solve)
        self._ir_counts = []        # per-solve refinement counts (§8j)

        # dirichlet: per-field dict {field_index: (node_ids, values)}; set via
        # set_dirichlet.  Rows are eliminated with the house identity-row idiom.
        self.dirichlet = {}

        # manufactured / external source per field at GPs (MMS gate hook):
        # dict {field: {p: ndarray[ngp]}} added to each row's RHS load.
        self.mms_source = None

        # transient history (BDF): previous full-field states + sigma
        self.hist = None          # dict field->full nodal vector (σ_BDF·u^n form fed as source)
        self.sigma = 0.0          # BDF mass coefficient (0 = steady)

    # ── field-vector helpers ────────────────────────────────────────────────
    def zero_state(self):
        n = self.dm.n_nodes
        return {f: np.zeros(n) for f in range(NDOF)}

    def set_dirichlet(self, field, node_ids, values):
        self.dirichlet[field] = (np.asarray(node_ids, np.int64),
                                  np.asarray(values, float))

    # ── closure evaluation at GPs from the current iterate ──────────────────
    def _closures(self, state):
        """Evaluate A3 closures at GPs given full nodal state.

        Returns a dict of per-bin GP arrays:
          gamma_hat (γ̂ n̂ p̂ pieces), R, dR_dn, dR_dp,
          kd, ka, dkd, dka  (dissociation rates + d/d|∇φ̂|),
          n_gp, p_gp, xd_gp, xa_gp, gradphi_gp, gmag_gp
        """
        dm = self.dm
        n_gp = _gp_value(dm, state[IN])
        p_gp = _gp_value(dm, state[IP])
        xd_gp = _gp_value(dm, state[IXD])
        xa_gp = _gp_value(dm, state[IXA])
        gradphi = _gp_grad(dm, state[IPHI])
        gmag = _grad_mag(gradphi)

        R, dRn, dRp, kd, ka, dkd, dka = {}, {}, {}, {}, {}, {}, {}
        for pv in dm.bins:
            dist = self.dist_gp[pv]
            if self.langevin is not None:
                r, drn, drp = self.langevin(n_gp[pv], p_gp[pv], dist)
            else:
                r = np.zeros_like(n_gp[pv]); drn = r.copy(); drp = r.copy()
            R[pv], dRn[pv], dRp[pv] = r, drn, drp
            if self.onsager is not None:
                kd_, ka_, (dkd_, dka_) = self.onsager(gmag[pv], dist)
                # broadcast scalar closure outputs to GP-array shape if needed
                kd[pv] = np.broadcast_to(kd_, n_gp[pv].shape).copy()
                ka[pv] = np.broadcast_to(ka_, n_gp[pv].shape).copy()
                dkd[pv] = np.broadcast_to(dkd_, n_gp[pv].shape).copy()
                dka[pv] = np.broadcast_to(dka_, n_gp[pv].shape).copy()
            else:
                z = np.zeros_like(n_gp[pv])
                kd[pv] = z.copy(); ka[pv] = z.copy()
                dkd[pv] = z.copy(); dka[pv] = z.copy()
        return dict(n_gp=n_gp, p_gp=p_gp, xd_gp=xd_gp, xa_gp=xa_gp,
                    gradphi=gradphi, gmag=gmag,
                    R=R, dRn=dRn, dRp=dRp, kd=kd, ka=ka, dkd=dkd, dka=dka)

    # ── advection field builders (sign baked in) ────────────────────────────
    def _aq(self, gradphi, mu_gp, sign):
        return {pv: sign * mu_gp[pv][:, None] * gradphi[pv] for pv in gradphi}

    def _sig2tau(self):
        return (2.0 * self.sigma) ** 2

    # ── generation source (per step) ────────────────────────────────────────
    def set_generation(self, gd_gp, ga_gp):
        self._gd = gd_gp
        self._ga = ga_gp

    def _gen(self):
        z = {pv: np.zeros(len(self.dist_gp[pv])) for pv in self.dm.bins}
        gd = getattr(self, "_gd", None) or z
        ga = getattr(self, "_ga", None) or z
        return gd, ga

    # ── residual: full-space per-field vector R_f[n_nodes] ──────────────────
    def residual_full(self, state):
        """Assemble the 5-field residual as full nodal vectors (pre-reduction).

        R_f = A_ff(u) @ u_f  −  load_f(u)   for each field, where the coupling
        source terms are evaluated at GPs from ``state``.
        """
        dm = self.dm
        cl = self._closures(state)
        gd, ga = self._gen()
        s2t = self._sig2tau()

        R = {f: np.zeros(dm.n_nodes) for f in range(NDOF)}

        one = {pv: np.ones(len(self.dist_gp[pv])) for pv in dm.bins}
        z_aq = {pv: np.zeros((len(self.dist_gp[pv]), dm.dim)) for pv in dm.bins}

        # ---- φ̂ row:  λ²ε̂K φ̂ − M(p̂ − n̂) ----
        Kphi = _poisson_block(dm, self.eps_gp, self.lam2)
        M = _mass_block(dm, one)
        R[IPHI] = Kphi @ state[IPHI] - M @ (state[IP] - state[IN])

        # ---- carrier rows ----
        # net carrier source  s = D̂ − R̂  (same for n and p)
        Dhat = {pv: cl["kd"][pv] * cl["xd_gp"][pv] + cl["ka"][pv] * cl["xa_gp"][pv]
                for pv in dm.bins}
        s_carr = {pv: Dhat[pv] - cl["R"][pv] for pv in dm.bins}

        aq_n = self._aq(cl["gradphi"], self.mu_n_gp, -1.0)   # electrons
        aq_p = self._aq(cl["gradphi"], self.mu_p_gp, +1.0)   # holes

        Kn = _carrier_block(dm, aq_n, self.mu_n_gp, self.sigma, s2t, self.supg)
        Kp = _carrier_block(dm, aq_p, self.mu_p_gp, self.sigma, s2t, self.supg)

        # source load (mass-weighted; SUPG-consistent via carrier be)
        fn = {pv: s_carr[pv] + self._hist_gp(IN, pv) for pv in dm.bins}
        fp = {pv: s_carr[pv] + self._hist_gp(IP, pv) for pv in dm.bins}
        Fn = _load_block(dm, aq_n, self.mu_n_gp, fn, s2t, self.supg)
        Fp = _load_block(dm, aq_p, self.mu_p_gp, fp, s2t, self.supg)
        R[IN] = Kn @ state[IN] - Fn
        R[IP] = Kp @ state[IP] - Fp

        # ---- exciton rows ----
        # σ_tot,i = σ_BDF + 1/τ̂_i + k̂_i(x)
        sig_d_gp = {pv: self.sigma + self.tau_inv_d + cl["kd"][pv] for pv in dm.bins}
        sig_a_gp = {pv: self.sigma + self.tau_inv_a + cl["ka"][pv] for pv in dm.bins}
        Kxd = _exciton_block(dm, self.mu_xd_gp, sig_d_gp)
        Kxa = _exciton_block(dm, self.mu_xa_gp, sig_a_gp)
        # exciton feed: Ĝ_i + R̂  (recombination re-forms excitons)
        fxd = {pv: gd[pv] + cl["R"][pv] + self._hist_gp(IXD, pv) for pv in dm.bins}
        fxa = {pv: ga[pv] + cl["R"][pv] + self._hist_gp(IXA, pv) for pv in dm.bins}
        Fxd = _load_block(dm, z_aq, self.mu_xd_gp, fxd, 0.0, 0.0)
        Fxa = _load_block(dm, z_aq, self.mu_xa_gp, fxa, 0.0, 0.0)
        R[IXD] = Kxd @ state[IXD] - Fxd
        R[IXA] = Kxa @ state[IXA] - Fxa

        # manufactured / external nodal source (MMS gate): subtract the fixed
        # source load so the manufactured field is the exact discrete solution.
        if self.mms_source is not None:
            for f in range(NDOF):
                if f in self.mms_source:
                    R[f] = R[f] - self.mms_source[f]
        return R

    def _hist_gp(self, field, pv):
        """BDF history source (σ_BDF·u^n) at GPs for `field`, or 0 if steady."""
        if self.hist is None:
            return np.zeros(len(self.dist_gp[pv]))
        return self.hist[field][pv]

    # ── Jacobian: 5×5 block sparse (full space) ─────────────────────────────
    def jacobian_full(self, state):
        dm = self.dm
        cl = self._closures(state)
        s2t = self._sig2tau()

        one = {pv: np.ones(len(self.dist_gp[pv])) for pv in dm.bins}
        M = _mass_block(dm, one)
        Z = sp.csr_matrix((dm.n_nodes, dm.n_nodes))

        # diagonal blocks
        Jphiphi = _poisson_block(dm, self.eps_gp, self.lam2)

        aq_n = self._aq(cl["gradphi"], self.mu_n_gp, -1.0)
        aq_p = self._aq(cl["gradphi"], self.mu_p_gp, +1.0)
        Jnn = _carrier_block(dm, aq_n, self.mu_n_gp, self.sigma, s2t, self.supg)
        Jpp = _carrier_block(dm, aq_p, self.mu_p_gp, self.sigma, s2t, self.supg)
        # + γ̂ p̂ M on n-diagonal (∂R̂/∂n̂ = γ̂p̂), γ̂ n̂ M on p-diagonal.  The
        # recombination enters the carrier source −R̂, so its Jacobian uses the
        # SUPG-augmented test function (N_a + τ a·∇N_a), NOT plain M.
        Jnn = Jnn + _supg_mass_block(dm, cl["dRn"], aq_n, self.mu_n_gp,
                                     s2t, self.supg)   # dRn = γ̂ p̂
        Jpp = Jpp + _supg_mass_block(dm, cl["dRp"], aq_p, self.mu_p_gp,
                                     s2t, self.supg)   # dRp = γ̂ n̂

        sig_d_gp = {pv: self.sigma + self.tau_inv_d + cl["kd"][pv] for pv in dm.bins}
        sig_a_gp = {pv: self.sigma + self.tau_inv_a + cl["ka"][pv] for pv in dm.bins}
        Jxdxd = _exciton_block(dm, self.mu_xd_gp, sig_d_gp)
        Jxaxa = _exciton_block(dm, self.mu_xa_gp, sig_a_gp)

        # ---- off-diagonals ----
        # φ̂ row source −M(p̂−n̂):  ∂/∂n̂ = +M, ∂/∂p̂ = −M
        Jphin = M
        Jphip = -M

        # carrier ∂/∂φ̂: CONSERVATIVE Galerkin drift cross-term + exact
        # SUPG-advection φ̂ coupling + dissociation-field coupling.
        gradn = _gp_grad(dm, state[IN])
        gradp = _gp_grad(dm, state[IP])
        n_gp = cl["n_gp"]; p_gp = cl["p_gp"]
        lapn = _gp_lap(dm, state[IN]); lapp = _gp_lap(dm, state[IP])
        # CONSERVATIVE Galerkin drift residual (n-row):  ∫ sign·μ̂ n̂ (∇φ̂·∇N_a).
        # ∂/∂φ̂_c = ∫ sign·μ̂ n̂_q (∇N_c·∇N_a)  — a weighted STIFFNESS block
        # (density n̂_q inside, ∇N_b·∇N_a against it), NOT the old
        # (∇c·∇N_b)N_a advection-mass block.  Assemble via _poisson_block with
        # coefficient (sign·μ̂·ĉ) and lam2=1.
        smu_n = {pv: -self.mu_n_gp[pv] for pv in dm.bins}   # sign=-1 electrons
        smu_p = {pv: +self.mu_p_gp[pv] for pv in dm.bins}   # sign=+1 holes
        cdrift_n = {pv: smu_n[pv] * n_gp[pv] for pv in dm.bins}
        cdrift_p = {pv: smu_p[pv] * p_gp[pv] for pv in dm.bins}
        Jnphi = _poisson_block(dm, cdrift_n, 1.0)
        Jpphi = _poisson_block(dm, cdrift_p, 1.0)
        # exact SUPG-advection φ̂ coupling (frozen τ only).  The SUPG residual is
        # τ(U·∇N_a)·(res(c) − f), U = −aq the drift velocity, res(c) the strong
        # residual on the advective part σĉ + U·∇ĉ − μ̂∇²ĉ, f the FULL carrier
        # source (D̂ − R̂ + hist).  δU = −sign·μ̂∇δφ̂, so the smu coefficient is
        # −sign·μ̂ (= −smu_n for electrons) and the velocity factor is U = −aq.
        if self.supg != 0.0:
            tau_n = _tau_gp(dm, aq_n, self.mu_n_gp, s2t, self.supg)
            tau_p = _tau_gp(dm, aq_p, self.mu_p_gp, s2t, self.supg)
            U_n = {pv: -aq_n[pv] for pv in dm.bins}
            U_p = {pv: -aq_p[pv] for pv in dm.bins}
            neg_smu_n = {pv: -smu_n[pv] for pv in dm.bins}    # −sign·μ̂ = +μ̂_n
            neg_smu_p = {pv: -smu_p[pv] for pv in dm.bins}    # −sign·μ̂ = −μ̂_p
            Dhat_j = {pv: cl["kd"][pv] * cl["xd_gp"][pv]
                          + cl["ka"][pv] * cl["xa_gp"][pv] for pv in dm.bins}
            f_carr = {pv: Dhat_j[pv] - cl["R"][pv] + self._hist_gp(IN, pv)
                      for pv in dm.bins}
            f_carr_p = {pv: Dhat_j[pv] - cl["R"][pv] + self._hist_gp(IP, pv)
                        for pv in dm.bins}
            resn = {pv: (self.sigma * n_gp[pv]
                         + np.sum(U_n[pv] * gradn[pv], axis=1)
                         - self.mu_n_gp[pv] * lapn[pv] - f_carr[pv])
                    for pv in dm.bins}
            resp = {pv: (self.sigma * p_gp[pv]
                         + np.sum(U_p[pv] * gradp[pv], axis=1)
                         - self.mu_p_gp[pv] * lapp[pv] - f_carr_p[pv])
                    for pv in dm.bins}
            cn_ctx = {"res": resn, "aq": U_n}
            cp_ctx = {"res": resp, "aq": U_p}
            Jnphi = Jnphi + _supg_drift_phi_block(
                dm, neg_smu_n, cn_ctx, gradn, tau_n)
            Jpphi = Jpphi + _supg_drift_phi_block(
                dm, neg_smu_p, cp_ctx, gradp, tau_p)
        # dissociation-field coupling on carrier source −D̂:
        #   ∂(−D̂)/∂φ̂ = −(dkd X̂_D + dka X̂_A)/|∇φ̂| (∇φ̂·∇δφ̂)
        # Galerkin part via the drift-cross block; SUPG part via the augmented
        # test function (the source is SUPG-weighted with U = −aq).
        dDdmag = {pv: cl["dkd"][pv] * cl["xd_gp"][pv]
                       + cl["dka"][pv] * cl["xa_gp"][pv] for pv in dm.bins}
        cdiss = {pv: -_safe_div(dDdmag[pv], cl["gmag"][pv]) for pv in dm.bins}
        Jnphi = Jnphi + _drift_cross_block(dm, cdiss, cl["gradphi"])
        Jpphi = Jpphi + _drift_cross_block(dm, cdiss, cl["gradphi"])
        if self.supg != 0.0:
            Jnphi = Jnphi + _supg_drift_cross_block(
                dm, cdiss, cl["gradphi"], U_n, tau_n)
            Jpphi = Jpphi + _supg_drift_cross_block(
                dm, cdiss, cl["gradphi"], U_p, tau_p)

        # carrier ∂/∂p̂, ∂/∂n̂ (recombination cross) — SUPG-augmented (source):
        #   n-row source −(−R̂) so ∂/∂p̂ = +γ̂n̂ (N_a+τa·∇N_a) M = +dRp block
        Jnp = _supg_mass_block(dm, cl["dRp"], aq_n, self.mu_n_gp, s2t, self.supg)
        Jpn = _supg_mass_block(dm, cl["dRn"], aq_p, self.mu_p_gp, s2t, self.supg)
        # carrier ∂/∂X̂_i:  source −D̂ so ∂/∂X̂_i = −k̂_i (N_a+τa·∇N_a) M
        Jnxd = _supg_mass_block(dm, {pv: -cl["kd"][pv] for pv in dm.bins},
                                aq_n, self.mu_n_gp, s2t, self.supg)
        Jnxa = _supg_mass_block(dm, {pv: -cl["ka"][pv] for pv in dm.bins},
                                aq_n, self.mu_n_gp, s2t, self.supg)
        Jpxd = _supg_mass_block(dm, {pv: -cl["kd"][pv] for pv in dm.bins},
                                aq_p, self.mu_p_gp, s2t, self.supg)
        Jpxa = _supg_mass_block(dm, {pv: -cl["ka"][pv] for pv in dm.bins},
                                aq_p, self.mu_p_gp, s2t, self.supg)

        # exciton ∂/∂φ̂ (sink side, +∂k̂_i/∂|∇φ̂| X̂_i):
        cxd = {pv: _safe_div(cl["dkd"][pv] * cl["xd_gp"][pv], cl["gmag"][pv])
               for pv in dm.bins}
        cxa = {pv: _safe_div(cl["dka"][pv] * cl["xa_gp"][pv], cl["gmag"][pv])
               for pv in dm.bins}
        Jxdphi = _drift_cross_block(dm, cxd, cl["gradphi"])
        Jxaphi = _drift_cross_block(dm, cxa, cl["gradphi"])
        # exciton ∂/∂n̂, ∂/∂p̂ (feed −R̂):  −γ̂p̂ M, −γ̂n̂ M
        Jxdn = _mass_block(dm, {pv: -cl["dRn"][pv] for pv in dm.bins})
        Jxdp = _mass_block(dm, {pv: -cl["dRp"][pv] for pv in dm.bins})
        Jxan = Jxdn
        Jxap = Jxdp

        blocks = [
            [Jphiphi, Jphin,  Jphip,  Z,     Z    ],
            [Jnphi,   Jnn,    Jnp,    Jnxd,  Jnxa ],
            [Jpphi,   Jpn,    Jpp,    Jpxd,  Jpxa ],
            [Jxdphi,  Jxdn,   Jxdp,   Jxdxd, Z    ],
            [Jxaphi,  Jxan,   Jxap,   Z,     Jxaxa],
        ]
        return blocks

    # ── constraint reduction + Dirichlet elimination ────────────────────────
    def _reduce_and_eliminate(self, blocks, R):
        """Apply T per block → n_free system, then strong Dirichlet rows.

        Returns (A [5nf × 5nf] csr, r [5nf]) in the Newton-increment convention
        (Dirichlet rows carry g − u_old so δu closes the gap on the first step).
        """
        T = self.T
        nf = self.n_free
        # reduce each block: T^T B T ; reduce each residual: T^T R
        A_blocks = [[None] * NDOF for _ in range(NDOF)]
        for i in range(NDOF):
            for j in range(NDOF):
                B = blocks[i][j]
                if B is None or (sp.issparse(B) and B.nnz == 0):
                    A_blocks[i][j] = None
                else:
                    A_blocks[i][j] = (T.T @ B @ T).tocsr()
        A = sp.bmat(A_blocks, format="csr")
        r = np.concatenate([np.asarray(T.T @ R[f]) for f in range(NDOF)])

        # LOG-DENSITY chain rule (carrier_vars="log"): the increment solved for
        # is δu=δ(ln n̂), δv=δ(ln p̂) rather than δn̂,δp̂.  With n̂=e^u the nodal
        # Jacobian columns transform  J_·n̂ → J_·n̂·diag(n̂),  reusing ALL primal
        # blocks (right-multiply the carrier column-blocks by diag(n̂_free)).
        # Applied to the FULL bmat (physics rows) BEFORE Dirichlet identity rows,
        # so cross-row couplings ∂R_other/∂u = ∂R_other/∂n̂·n̂ are scaled too.
        if self._log_carriers:
            scale = np.ones(NDOF * nf)
            free_all = self.free
            for field in (IN, IP):
                cur = self._current_state[field][free_all]
                scale[field * nf:(field + 1) * nf] = cur
            A = A @ sp.diags(scale)

        A = A.tolil()
        # strong Dirichlet: identity row on free-dof index of the pinned node
        # map global node id → free index
        free = self.free
        node_to_free = -np.ones(self.dm.n_nodes, np.int64)
        node_to_free[free] = np.arange(nf)
        for field, (nodes, vals) in self.dirichlet.items():
            log_field = self._log_carriers and field in (IN, IP)
            for k, nid in enumerate(nodes):
                fi = node_to_free[nid]
                if fi < 0:
                    continue
                row = field * nf + fi
                A.rows[row] = [row]
                A.data[row] = [1.0]
                # r carries residual R already reduced; for Dirichlet we want
                # δu = g − u_old.  In log mode the carrier target is in log
                # space: δu = ln(g) − ln(u_old) closes the BC on the first step.
                cur = self._current_state[field][nid]
                if log_field:
                    r[row] = np.log(vals[k]) - np.log(cur)
                else:
                    r[row] = vals[k] - cur
        return A.tocsr(), r

    # ── Task #35: assembly dispatch (host bmat path | device warp path) ─────
    def device_assembler(self):
        """The lazy XDDDeviceAssembler (pattern + static uploads built once)."""
        if self._dev_asm is None:
            from .exciton_device import XDDDeviceAssembler
            self._dev_asm = XDDDeviceAssembler(
                self, index_width=self.index_width,
                val_dtype=self.val_dtype)
        return self._dev_asm

    def assemble_newton_system(self, state, need_matrix=True):
        """One Newton system (A, r) at `state`.

        host path: the R0 `residual_full` + `jacobian_full` +
        `_reduce_and_eliminate` composition (field-major dofs) — verbatim.
        device path: `XDDDeviceAssembler.assemble` (node-major dofs; Dirichlet
        and log-mode conventions identical).  need_matrix=False returns
        (None, r) — the line-search residual probe skips the host CSR pull.
        With linsolver='cudss' on CUDA the matrix never leaves the device
        (A is None; `_solve_rhs` consumes the assembler's zero-copy CSR)."""
        if not self._assembly_device:
            return self._reduce_and_eliminate(self.jacobian_full(state),
                                              self.residual_full(state))
        if not need_matrix:
            return self.device_assembler().assemble(state, need_matrix=False)
        dev_solver = (self.linsolver == "cudss"
                      and str(self.dm.device).startswith("cuda")
                      and self.linsolve == self._cudss_linsolve)
        return self.device_assembler().assemble(
            state, need_matrix=("device" if dev_solver else "host"))

    def _cudss_dev_solve(self, b):
        """Zero-copy device-CSR cuDSS solve (multiphase `_solve_dev` pattern):
        DirectSolver planned ONCE over the torch CSR aliasing the assembler's
        vals_d (fixed pattern — strong rows keep explicit zeros, nnz never
        flaps); per iterate the fill updates vals_d in place, b is copied into
        the STABLE b tensor, then factorize + solve.  Explicit .free() on any
        failure (the double-free guard).

        Task #36 (8j): val_dtype='fp32' routes to the fp32-factor +
        FP64-iterative-refinement path — the fp32 snapshot CSR is factored;
        the residual is FP64 (fp64 device SpMV over vals_d + the fp64 rhs
        b), NEVER the fp32 stored values; refinement count logged."""
        try:
            import torch
            from nvmath.sparse.advanced import (DirectSolver,
                                                DirectSolverOptions)
        except Exception as exc:
            from ..errors import BackendError
            raise BackendError(
                "linsolver='cudss' with device assembly requires torch + "
                f"nvmath-python[cu12] on CUDA (import failed: {exc})") from exc
        if self.device_assembler().asm._vals_fp32 is not None:
            return self._cudss_dev_solve_fp32_ir(b)
        try:
            if self._cudss_dev is None:
                A_t, F_t = self.device_assembler().asm.device_csr()
                self._b_t = torch.empty_like(F_t)
                self._b_t.copy_(torch.from_numpy(
                    np.ascontiguousarray(b)).to(self._b_t.device))
                self._cudss_dev = DirectSolver(
                    A_t, self._b_t, options=DirectSolverOptions(blocking=True))
                self._cudss_dev.plan()
            else:
                self._b_t.copy_(torch.from_numpy(
                    np.ascontiguousarray(b)).to(self._b_t.device))
            self._cudss_dev.factorize()
            return np.asarray(self._cudss_dev.solve().cpu())
        except Exception:
            try:
                self._cudss_dev.free()
            except Exception:
                pass
            self._cudss_dev = None
            raise

    def _cudss_dev_solve_fp32_ir(self, b):
        """Task #36: fp32-factor + FP64-iterative-refinement device solve
        for the XDD Newton system (the #35 cuDSS consumer).  The fp32
        snapshot CSR is factored once (STABLE fp32 rhs tensor, refactorize
        per iterate); the correction A dx = r is solved in fp32 while the
        residual r = b - A x is FP64 (the assembler's fp64 device SpMV over
        vals_d + the fp64 rhs b).  Refinement count logged (§8j); a
        >10-sweep non-convergence surfaces as a raised error (the Newton
        driver's hard-failure path) rather than a silent wrong solve."""
        import torch
        from nvmath.sparse.advanced import DirectSolver, DirectSolverOptions
        from ..solvers.iterative_refinement import fp64_iterative_refinement
        from ..solvers.iterative_refinement_device import WarpIRBackend
        asm = self.device_assembler().asm
        b64 = np.ascontiguousarray(b, np.float64)
        op = asm.device_operator()
        dev = str(self.dm.device)

        try:
            A32_t, _ = asm.device_csr_fp32()    # refreshes the fp32 snapshot
            if self._cudss_dev is None:
                self._b_t32 = torch.zeros(asm.Nfull, dtype=torch.float32,
                                          device=dev)
                self._cudss_dev = DirectSolver(
                    A32_t, self._b_t32,
                    options=DirectSolverOptions(blocking=True))
                self._cudss_dev.plan()
            self._cudss_dev.factorize()

            # Task #43: device-resident IR buffers.  The XDD rhs `b` arrives
            # on the host (Newton driver), so it is uploaded ONCE here (no
            # asm.F_d alias for this consumer); the residual/correction/
            # shadow-matvec then stay device-resident across sweeps.
            backend = WarpIRBackend(op, self._cudss_dev, self._b_t32,
                                    asm.Nfull, self.dm.device)
            x, info = fp64_iterative_refinement(
                None, None, b64, tol=1e-12, max_iter=10, backend=backend)
            self._ir_counts.append(info["refinements"])
            self._last_ir = info
            if not info["converged"]:
                from ..errors import ConvergenceError
                raise ConvergenceError(
                    f"XDD fp32+IR solve did not converge in 10 sweeps "
                    f"(rel_resid={info['rel_resid']:.2e}) — conditioning "
                    f"FAIL (§8j G2)")
            return x
        except Exception:
            try:
                self._cudss_dev.free()
            except Exception:
                pass
            self._cudss_dev = None
            raise

    # ── Newton solve with positivity-guarded backtracking line search ───────
    def solve_newton(self, state, *, max_iter=8, floor=1e-30,
                     x_floor=-1e-12, max_halving=40, rtol=1e-8, verbose=False):
        """Monolithic Newton.  Returns (state, info) where info has the
        increment norms, residual norms and convergence flag.

        Globalisation: a fraction-to-boundary cap keeps n̂,p̂ ≥ floor and
        X̂ ≥ x_floor without collapsing the step, then backtracking halving
        enforces residual decrease.  Convergence requires a GENUINE residual /
        increment reduction — a collapsed-step (tiny δ, stalled residual) is
        NOT reported as converged.
        """
        self._current_state = state
        dnorms, rnorms = [], []
        converged = False
        r0 = None
        for it in range(max_iter):
            A, r = self.assemble_newton_system(state)
            rnorm = float(np.linalg.norm(r))
            rnorms.append(rnorm)
            if r0 is None:
                r0 = max(rnorm, 1e-300)
            if rnorm < self.atol or rnorm < rtol * r0:
                converged = True
                break
            du = self._solve_rhs(A, r)

            # (1) fraction-to-boundary: largest α∈(0,1] keeping positivity
            alpha = self._frac_to_boundary(state, du, floor, x_floor)
            # (2) backtracking on residual decrease from the capped step
            step, ok, trial, rn_new = alpha, False, None, np.inf
            for _ in range(max_halving):
                cand = self._apply_increment(state, du, step)
                if not self._positivity_ok(cand, floor, x_floor):
                    step *= 0.5
                    continue
                self._current_state = cand
                _, rc = self.assemble_newton_system(cand, need_matrix=False)
                rn_new = float(np.linalg.norm(rc))
                if rn_new < rnorm * (1.0 - 1e-4) or step < 1e-12:
                    ok, trial = True, cand
                    break
                step *= 0.5
            self._current_state = state
            if not ok or trial is None:
                return state, dict(converged=False, reason="line_search_fail",
                                   dnorms=dnorms, rnorms=rnorms, iters=it + 1)
            dnorm = step * float(np.linalg.norm(du))
            dnorms.append(dnorm)
            state = trial
            self._current_state = state
            unorm = np.sqrt(sum(np.linalg.norm(self.T.T @ state[f]) ** 2
                                for f in range(NDOF)))
            if verbose:
                print(f"  newton it{it}: |du|={dnorm:.3e} |r|={rnorm:.3e} "
                      f"-> {rn_new:.3e} step={step:.3e}")
            # genuine convergence: residual below atol / relative floor OR
            # relative increment below tol with a non-collapsed step
            if rn_new < self.atol or rn_new < rtol * r0:
                converged = True
                break
            # NOTE: in log mode dnorm carries LOG-space carrier increments while
            # unorm is primal — mixing units would allow false convergence at
            # depletion (an O(1) log step is a negligible primal change). The
            # increment criterion is therefore primal-mode only; log mode
            # converges on the residual criterion alone. (B5 review, Important #2.)
            if (not self._log_carriers
                    and dnorm <= self.newton_tol * max(unorm, 1e-30)
                    and step > 1e-6):
                converged = True
                break
        return state, dict(converged=converged, dnorms=dnorms, rnorms=rnorms,
                           iters=len(dnorms))

    def _frac_to_boundary(self, state, du, floor, x_floor, tau_fb=0.95):
        """Largest α∈(0,1] keeping n̂,p̂ ≥ floor after the step.

        Only the strictly-positive carrier densities n̂,p̂ constrain the step
        here; X̂ starts at 0 and may transiently dip slightly negative, so it
        is NOT capped by fraction-to-boundary — the positivity CHECK (X̂ ≥
        x_floor = −1e-12) plus backtracking halving guards it instead.  This
        avoids a spurious α→0 collapse when X̂ = 0 and du_X < 0.
        """
        nf = self.n_free
        free = self.free
        # LOG mode: carriers are n̂=e^u > 0 for ANY δu — the positivity boundary
        # does not exist for them, so fraction-to-boundary never constrains the
        # carrier step.  Return α=1 (X̂ is guarded by the positivity CHECK +
        # backtracking as in primal).  We record that no carrier truncation
        # occurred (the no-guard-needed evidence for gate 4).
        if self._log_carriers:
            return 1.0
        alpha = 1.0
        for f in (IN, IP):
            duf = np.asarray(self.T @ du[f * nf:(f + 1) * nf])[free]
            cur = state[f][free]
            dec = duf < 0
            if np.any(dec):
                amax = np.min((floor - cur[dec]) / duf[dec])
                if amax > 0:
                    alpha = min(alpha, tau_fb * amax)
                    self.guard_trunc += 1
        return max(min(alpha, 1.0), 0.0)

    def _solve_rhs(self, A, r):
        """Solve J δ = −r for physics rows and δ = (g−u) for Dirichlet rows.

        r already holds g−u on Dirichlet rows (positive target) and the
        physics residual on the rest; A has identity rows for Dirichlet.
        The Newton system is J δ = −R_physics, so we solve A δ = b with
        b = −r on physics rows and +r on Dirichlet rows.

        Device-assembly ordering (Task #35): the device system is NODE-major
        (row = node·NDOF + field); the Dirichlet flip uses that layout, and
        the solved increment is permuted back to the host FIELD-major layout
        so the rest of the Newton loop (increments, line search, log-mode
        exponentials) is order-agnostic.
        """
        nf = self.n_free
        b = -r.copy()
        free = self.free
        node_to_free = -np.ones(self.dm.n_nodes, np.int64)
        node_to_free[free] = np.arange(nf)
        for field, (nodes, vals) in self.dirichlet.items():
            for nid in nodes:
                fi = node_to_free[nid]
                if fi < 0:
                    continue
                row = (fi * NDOF + field) if self._assembly_device \
                    else (field * nf + fi)
                b[row] = -b[row]   # flip back to +(g−u)
        if self._assembly_device and A is None:
            du = self._cudss_dev_solve(b)
        else:
            du = self.linsolve(A, b)
        if self._assembly_device:
            du = np.ascontiguousarray(
                np.asarray(du).reshape(nf, NDOF).T).reshape(-1)
        return du

    def _cudss_linsolve(self, A, b):
        """cuDSS (GPU sparse direct LU) solve of A x = b for the 5-field
        Jacobian — the E5 perf-fix path (host splu was GPU-idle at 330k DOF).

        Follows the in-tree multiphase/wodo lifecycle: PLAN ONCE on the first
        Newton iterate, then reset_operands + refactorize per iterate (the
        Jacobian pattern is fixed across a Newton solve — same mesh, same
        coupling).  The plan is rebuilt (old one EXPLICITLY .free()'d — the
        double-free guard) if the nnz pattern ever changes.  PLAIN
        DirectSolverOptions(blocking=True): no mtlayer_gomp (its per-call
        gomp-thread-team leak killed multiphase marches; the 2-D system plans
        fast single-threaded).  nvmath/CUDA absent -> BackendError (Mac/CPU
        parity mode uses linsolver="splu" and never reaches here)."""
        A = A.tocsr()
        b = np.ascontiguousarray(b, np.float64)
        try:
            from nvmath.sparse.advanced import (DirectSolver,
                                                DirectSolverOptions)
        except Exception as exc:                       # nvmath / CUDA absent
            from ..errors import BackendError
            raise BackendError(
                "linsolver='cudss' requires nvmath-python[cu12] on a CUDA "
                f"device (import failed: {exc}). Use linsolver='splu' on "
                "CPU/Mac.") from exc
        if self._cudss is not None and self._cudss_nnz != A.nnz:
            # sparsity changed -> the cached plan is stale; release it
            # EXPLICITLY (GC finalizer double-frees device buffers under
            # pattern flapping — the multiphase 2026-07-10 lesson).
            try:
                self._cudss.free()
            except Exception:
                pass
            self._cudss = None
        try:
            if self._cudss is None:
                self._cudss = DirectSolver(
                    A, b, options=DirectSolverOptions(blocking=True))
                self._cudss.plan()
                self._cudss_nnz = A.nnz
            else:
                self._cudss.reset_operands(a=A, b=b)
            self._cudss.factorize()
            return np.asarray(self._cudss.solve())
        except Exception:
            # release the (possibly half-built) plan so the next attempt
            # re-plans cleanly rather than double-freeing on GC
            try:
                self._cudss.free()
            except Exception:
                pass
            self._cudss = None
            self._cudss_nnz = -1
            raise

    def _apply_increment(self, state, du, step):
        nf = self.n_free
        new = {f: state[f].copy() for f in range(NDOF)}
        for f in range(NDOF):
            duf = du[f * nf:(f + 1) * nf]
            if self._log_carriers and f in (IN, IP):
                # δu is a LOG-space increment: n̂ ← n̂·exp(step·δu).  Positive for
                # any δu — carriers can never cross zero.  The T-scatter of a
                # log increment then exp keeps hanging-node consistency because T
                # is nodal-injective on free dofs (uniform mesh here).  Clip the
                # exponent to the float64 exp range: a pathological (ill-
                # conditioned) δu that would overflow is capped to +inf-free
                # values so the positivity/backtracking guard can cleanly reject
                # the trial step without a spurious overflow warning.
                expo = np.clip(step * np.asarray(self.T @ duf), -230.0, 230.0)
                new[f] = np.asarray(state[f] * np.exp(expo))
            else:
                new[f] = np.asarray(state[f] + step * (self.T @ duf))
        return new

    def _positivity_ok(self, state, floor, x_floor):
        # only check free dofs (Dirichlet nodes are pinned to physical values)
        free = self.free
        if np.any(state[IN][free] < floor) or np.any(state[IP][free] < floor):
            return False
        if np.any(state[IXD][free] < x_floor) or np.any(state[IXA][free] < x_floor):
            return False
        return True

    # ── transient BDF wrapper ───────────────────────────────────────────────
    def step_bdf(self, state, dt, *, order=1, prev=None, prev2=None, **nk):
        """Advance one BDF step.  prev/prev2 are full-field state dicts.

        Sets self.sigma and self.hist (σ_BDF·u^n GP source) then Newton-solves.
        """
        if order == 1 or prev2 is None:
            self.sigma = 1.0 / dt
            hist_full = {f: self.sigma * prev[f] for f in range(NDOF)}
        else:
            self.sigma = 1.5 / dt
            hist_full = {f: (2.0 * prev[f] - 0.5 * prev2[f]) / dt
                         for f in range(NDOF)}
        # φ̂ has no time term: zero its history and its σ mass (Poisson steady).
        # The carrier/exciton blocks pick up σ via self.sigma; φ̂ block is pure
        # Poisson (no σ), so hist_φ is irrelevant — but zero it for clarity.
        hist_full[IPHI] = np.zeros(self.dm.n_nodes)
        self.hist = {f: _gp_value(self.dm, hist_full[f]) for f in range(NDOF)}
        st, info = self.solve_newton(state, **nk)
        # reset steady after step so residual/jacobian callers stay pure
        return st, info

    # ── Block-Gauss-Seidel mode (the CPU debugging path) ────────────────────
    def solve_block_gs(self, state, *, block_tol=1e-6, max_block=50,
                       inner_newton=4, floor=1e-30, x_floor=-1e-12,
                       verbose=False):
        """Block-GS: alternate (excitons | φ̂,n̂,p̂) and (φ̂,n̂,p̂ | excitons).

        Outer loop: (1) freeze φ̂,n̂,p̂, solve the two LINEAR exciton systems for
        X̂_D,X̂_A; (2) freeze X̂, run the carrier+Poisson Newton.  Iterate to
        ‖Δstate‖/‖state‖ < block_tol.  Monolithic ``solve_newton`` is the
        default; this is the parity-debugging path (plan B4).
        """
        self._current_state = state
        for it in range(max_block):
            prev = {f: state[f].copy() for f in range(NDOF)}

            # (1) exciton sub-solve (linear given φ̂,n̂,p̂ via the closures)
            state = self._exciton_subsolve(state)
            # (2) carrier+Poisson Newton sub-solve (excitons frozen)
            state = self._carrier_poisson_subsolve(
                state, inner_newton, floor, x_floor)

            num = np.sqrt(sum(np.linalg.norm(self.T.T @ (state[f] - prev[f])) ** 2
                              for f in range(NDOF)))
            den = np.sqrt(sum(np.linalg.norm(self.T.T @ state[f]) ** 2
                              for f in range(NDOF)))
            rel = num / max(den, 1e-30)
            if verbose:
                print(f"  block-GS it{it}: ‖Δ‖/‖u‖ = {rel:.3e}")
            if rel < block_tol:
                return state, dict(converged=True, iters=it + 1, rel=rel)
        return state, dict(converged=False, iters=max_block, rel=rel)

    def _exciton_subsolve(self, state):
        """Solve the two linear exciton systems given frozen φ̂,n̂,p̂."""
        dm = self.dm
        cl = self._closures(state)
        gd, ga = self._gen()
        sig_d = {pv: self.sigma + self.tau_inv_d + cl["kd"][pv] for pv in dm.bins}
        sig_a = {pv: self.sigma + self.tau_inv_a + cl["ka"][pv] for pv in dm.bins}
        z_aq = {pv: np.zeros((len(self.dist_gp[pv]), dm.dim)) for pv in dm.bins}
        for f, mu_gp, sig_gp, gsrc in ((IXD, self.mu_xd_gp, sig_d, gd),
                                       (IXA, self.mu_xa_gp, sig_a, ga)):
            K = _exciton_block(dm, mu_gp, sig_gp)
            fq = {pv: gsrc[pv] + cl["R"][pv] + self._hist_gp(f, pv)
                  for pv in dm.bins}
            F = _load_block(dm, z_aq, mu_gp, fq, 0.0, 0.0)
            if self.mms_source is not None and f in self.mms_source:
                F = F + self.mms_source[f]
            A = (self.T.T @ K @ self.T).tocsr()
            b = np.asarray(self.T.T @ F)
            A, b = self._apply_field_dirichlet(A, b, f, state)
            xf = self.linsolve(A, b)
            state = {**state, f: np.asarray(self.T @ xf)}
        return state

    def _apply_field_dirichlet(self, A, b, field, state):
        """Identity-row Dirichlet on a single scalar field (direct value form)."""
        if field not in self.dirichlet:
            return A.tocsr(), b
        A = A.tolil()
        nodes, vals = self.dirichlet[field]
        node_to_free = -np.ones(self.dm.n_nodes, np.int64)
        node_to_free[self.free] = np.arange(self.n_free)
        for k, nid in enumerate(nodes):
            fi = node_to_free[nid]
            if fi < 0:
                continue
            A.rows[fi] = [int(fi)]
            A.data[fi] = [1.0]
            b[fi] = vals[k]
        return A.tocsr(), b

    def _carrier_poisson_subsolve(self, state, max_iter, floor, x_floor):
        """Newton on (φ̂,n̂,p̂) with X̂_D,X̂_A frozen (3-field sub-Jacobian)."""
        sub = (IPHI, IN, IP)
        self._current_state = state
        for _ in range(max_iter):
            R = self.residual_full(state)
            blocks = self.jacobian_full(state)
            A, r = self._reduce_and_eliminate(blocks, R)
            # restrict to the (φ̂,n̂,p̂) sub-system by zeroing X couplings/rows
            nf = self.n_free
            keep = np.concatenate([np.arange(f * nf, (f + 1) * nf) for f in sub])
            Asub = A[np.ix_(keep, keep)].tocsc()
            rsub = r[keep]
            b = -rsub.copy()
            # Dirichlet rows in the sub-system carry (g−u): flip sign
            self._flip_dirichlet_rows(b, sub, state)
            du_sub = splu(Asub).solve(b)
            du = np.zeros(NDOF * nf)
            du[keep] = du_sub
            alpha = self._frac_to_boundary(state, du, floor, x_floor)
            state = self._apply_increment(state, du, alpha)
            self._current_state = state
            if np.linalg.norm(du_sub) < self.newton_tol * max(
                    np.linalg.norm(rsub), 1e-30):
                break
        return state

    def _flip_dirichlet_rows(self, b, sub, state):
        nf = self.n_free
        node_to_free = -np.ones(self.dm.n_nodes, np.int64)
        node_to_free[self.free] = np.arange(nf)
        for si, field in enumerate(sub):
            if field not in self.dirichlet:
                continue
            nodes, vals = self.dirichlet[field]
            for k, nid in enumerate(nodes):
                fi = node_to_free[nid]
                if fi < 0:
                    continue
                b[si * nf + fi] = vals[k] - state[field][nid]


def _safe_div(num, den, eps=1e-300):
    """num/den with 0 where den≈0 (the |∇φ̂|=0 b→0 dissociation limit)."""
    out = np.zeros_like(num)
    mask = np.abs(den) > 1e-30
    out[mask] = num[mask] / den[mask]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Bilayer electrode boundary conditions + continuation initial condition
# ══════════════════════════════════════════════════════════════════════════════

_E60 = float(np.exp(-60.0))   # e⁻⁶⁰ depleted-minority-carrier BC floor


def bilayer_electrode_bcs(sysm, mesh, cons, *, Eg_hat, V_app_hat=0.0,
                          h_axis=1, tol=1e-9, minority_ln=-60.0):
    """Apply the CPU bilayer electrode Dirichlet BCs to an XDDSystem.

    Anode  (ĥ=0, the h_axis-min wall):  φ̂ = +(Ê_g − V̂_app)/2, n̂ = 1, p̂ = e^m
    Cathode(ĥ=1, the h_axis-max wall):  φ̂ = −(Ê_g − V̂_app)/2, n̂ = e^m, p̂ = 1
    where m = minority_ln (the CPU value is −60; a milder floor keeps the
    PRIMAL Newton well-posed — the full e⁻⁶⁰ depletion is the B5 log-density
    regime per the SP-1 plan).  Excitons: natural.  Lateral walls: natural.
    ``h_axis`` selects the device-height coordinate (default y = axis 1).

    EQUILIBRIUM ALIGNMENT (post drift-sign fix): the majority carrier is n̂=1 at
    the +φ̂ anode, e^m at the cathode — i.e. n̂ ∝ e^{+(φ̂−φ̂a)}, the Boltzmann
    equilibrium of the CONSERVATIVE kernel (eqm n̂∝e^{+φ̂}).  With the corrected
    conservative drift these electrode values are the discrete kernel
    equilibrium (the e⁻⁶⁰-class depletion at the minority contact is the TRUE
    steady state — before the fix the kernel equilibrium was the OPPOSITE sign,
    n̂∝e^{−φ̂}, which is why Block C's dark bilayer march filled the bulk).
    """
    coords = mesh.node_coords
    hc = coords[:, h_axis]
    lo, hi = hc.min(), hc.max()
    anode = np.where(np.abs(hc - lo) < tol)[0]
    cathode = np.where(np.abs(hc - hi) < tol)[0]

    phi_a = +0.5 * (Eg_hat - V_app_hat)
    phi_c = -0.5 * (Eg_hat - V_app_hat)
    minority = float(np.exp(minority_ln))

    sysm.set_dirichlet(IPHI,
                       np.concatenate([anode, cathode]),
                       np.concatenate([np.full(len(anode), phi_a),
                                       np.full(len(cathode), phi_c)]))
    sysm.set_dirichlet(IN,
                       np.concatenate([anode, cathode]),
                       np.concatenate([np.full(len(anode), 1.0),
                                       np.full(len(cathode), minority)]))
    sysm.set_dirichlet(IP,
                       np.concatenate([anode, cathode]),
                       np.concatenate([np.full(len(anode), minority),
                                       np.full(len(cathode), 1.0)]))
    return dict(anode=anode, cathode=cathode, phi_a=phi_a, phi_c=phi_c)


def continuation_ic(sysm, mesh, *, Eg_hat, V_app_hat=0.0, h_axis=1,
                    minority_ln=-60.0):
    """Continuation-style initial guess (the dark, V̂=0 starting state).

    φ̂ linear between the electrode values; n̂,p̂ log-linear equilibrium
    profiles interpolating the electrode BC values; X̂ = 0.  ``minority_ln``
    matches the electrode-BC minority floor.
    """
    coords = mesh.node_coords
    hc = coords[:, h_axis]
    lo, hi = hc.min(), hc.max()
    xi = (hc - lo) / max(hi - lo, 1e-30)         # 0 at anode, 1 at cathode

    phi_a = +0.5 * (Eg_hat - V_app_hat)
    phi_c = -0.5 * (Eg_hat - V_app_hat)

    st = sysm.zero_state()
    phi_lin = phi_a + (phi_c - phi_a) * xi
    st[IPHI] = phi_lin
    # Boltzmann (drift-diffusion equilibrium) carrier profiles consistent with
    # the linear φ̂ and the anode/cathode Dirichlet values:
    #   n̂ = exp(φ̂ − φ̂_anode)   (electrons follow +φ̂; n̂=1 at the anode)
    #   p̂ = exp(φ̂_anode − φ̂)   (holes follow −φ̂; p̂=1 at the cathode)
    # POST drift-sign fix these ALIGN with the CONSERVATIVE kernel equilibrium
    # (eqm n̂∝e^{+φ̂}) — the IC is now the discrete near-solution (previously the
    # kernel equilibrium was the opposite sign, so this IC sat ~100× off in the
    # carrier-row residual; Block C's kernel-equilibrium residual test).  The
    # residual is small and Newton shows its quadratic tail; with
    # minority_ln = −Ê_g the cathode/anode floors match.
    st[IN] = np.exp(phi_lin - phi_a)
    st[IP] = np.exp(phi_a - phi_lin)
    st[IXD] = np.zeros(len(coords))
    st[IXA] = np.zeros(len(coords))
    return st
