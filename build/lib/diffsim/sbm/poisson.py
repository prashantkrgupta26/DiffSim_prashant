"""SBM Poisson on carved octree meshes: Dirichlet via the shifted Nitsche
form (Main & Scovazzi, JCP 2018), assembled-CSR path (M1a; matrix-free
arrives with NS in M1b).

Weak form (kappa-scaled; Su = u + grad_u . d + 1/2 d^T H(u) d the Taylor
shift AT BASIS ORDER — the second-order term is required for p2 third-order
convergence (measured: first-order shift caps L2 at order 2; Atallah-
Scovazzi high-order SBM) and vanishes exactly for linears, so P4 is
unchanged; gbar = g(x + d) the mapped Dirichlet data, n_tilde the
surrogate-face outward normal, h the face element size):

    int_Omega~ kappa grad_w . grad_u dV
  - int_Gamma~D kappa w (grad_u . n_tilde) dS          (consistency)
  - int_Gamma~D kappa (grad_w . n_tilde) Su dS         (adjoint consistency)
  + int_Gamma~D (alpha kappa / h) Sw Su dS             (penalty)
  = int_Omega~ w f dV
  - int_Gamma~D kappa (grad_w . n_tilde) gbar dS
  + int_Gamma~D (alpha kappa / h) Sw gbar dS

Exactness contract (the P4 keystone): linear u makes Su - gbar vanish
identically and the consistency term equal the true flux, so the patch test
is machine-exact at ANY alpha, lambda, and geometry rotation. The
-kappa (grad_w . n_tilde)(grad_u . d) piece has no transpose partner: the
operator is NONSYMMETRIC — solves use bicgstab, never cg.

kappa-linearity: A(kappa) = kappa A1, b = b_f + kappa b_g1; assemble()
returns the kappa=1 pieces in meta (the Task-10 kappa-gradient needs
dR/dkappa = A1 u - b_g1).

Symbols: a/b trial/test local indices; gna = grad(N_a) . n_tilde;
Sa/Sb = shifted basis values; dS = w_q (h/2)^(dim-1). Layout: face arrays
flat [Nf*nqf] in (face, q) order; conn is the p-bin's bin-local connectivity.

Kernels here are ASSEMBLY kernels, never taped => backward codegen off
(Task-1 rule); the taped face-residual kernels live in sbm/adjoint.py.
"""
import numpy as np
import scipy.sparse as sp
import warp as wp

wp.set_module_options({"enable_backward": False})

from ..assembly.femelm import FEMElm, fe_N
from ..assembly.operators import (_kernel_cache, CSROperator, volume_triplets)
from ..mesh.faces import face_tables
from ..solvers.krylov import bicgstab


@wp.func
def _shifted_basis(Nf: wp.array3d(dtype=wp.float64),
                   dNf: wp.array4d(dtype=wp.float64),
                   d2Nf: wp.array4d(dtype=wp.float64),
                   f: wp.int32, q: wp.int32, a: wp.int32, gp: wp.int32,
                   dvec: wp.array2d(dtype=wp.float64),
                   dscale: wp.float64, dim_: wp.int32) -> wp.float64:
    """Second-order Taylor shift of basis function a at face GP gp:
    S N_a = N_a + grad(N_a).d + 1/2 d^T H(N_a) d. The second-order term is
    REQUIRED for p2 optimal (third-order) convergence — the first-order
    shift's O(d^2) boundary consistency error caps L2 at order 2 (measured;
    Atallah-Scovazzi high-order SBM: shift order = basis order). Exact zero
    for linear fields, so the P4 patch contract is unchanged."""
    Sa = Nf[f, q, a]
    for dd in range(dim_):
        Sa += dNf[f, q, a, dd] * dscale * dvec[gp, dd]
    d2s = dscale * dscale
    for ii in range(dim_):
        for jj in range(dim_):
            Sa += wp.float64(0.5) * d2Nf[f, q, a, ii * dim_ + jj] * d2s \
                  * dvec[gp, ii] * dvec[gp, jj]
    return Sa


@wp.func
def _shifted_basis_o1(Nf: wp.array3d(dtype=wp.float64),
                      dNf: wp.array4d(dtype=wp.float64),
                      d2Nf: wp.array4d(dtype=wp.float64),
                      f: wp.int32, q: wp.int32, a: wp.int32, gp: wp.int32,
                      dvec: wp.array2d(dtype=wp.float64),
                      dscale: wp.float64, dim_: wp.int32) -> wp.float64:
    """First-order Taylor shift (p1 elements). Q1's Hessian is INCOMPLETE
    (mixed partials nonzero, diagonals identically zero), and including the
    mixed-only terms pollutes the consistent first-order shift — measured:
    p1 MMS order 2.00 -> 1.85. Production gates the second-order term on
    elemOrder == 2 for the same reason (Dendrite NSEquation.h SBM_Ae).
    d2Nf is accepted and ignored so both shift variants share a signature."""
    Sa = Nf[f, q, a]
    for dd in range(dim_):
        Sa += dNf[f, q, a, dd] * dscale * dvec[gp, dd]
    return Sa


def _shift_fn_for(nbf: int, dim: int):
    """Shift order = basis order: p2 gets the full second-order shift, p1
    the first-order one (see _shifted_basis_o1)."""
    p = round(nbf ** (1.0 / dim)) - 1
    return _shifted_basis if p == 2 else _shifted_basis_o1


@wp.func
def _flux_shift(dNf: wp.array4d(dtype=wp.float64),
                d2Nf: wp.array4d(dtype=wp.float64),
                f: wp.int32, q: wp.int32, b: wp.int32, gp: wp.int32,
                dvec: wp.array2d(dtype=wp.float64),
                nvec: wp.array2d(dtype=wp.float64),
                dscale: wp.float64, dim_: wp.int32) -> wp.float64:
    """Shifted TRUE-normal flux of basis b at face GP gp (p2 faces):
    (grad N_b + H(N_b) d) . n_true — the Eq.-21 Hessian flux correction."""
    v = wp.float64(0.0)
    d2s = dscale * dscale
    for ii in range(dim_):
        gi = dNf[f, q, b, ii] * dscale
        for jj in range(dim_):
            gi += d2Nf[f, q, b, ii * dim_ + jj] * d2s * dvec[gp, jj]
        v += gi * nvec[gp, ii]
    return v


@wp.func
def _flux_shift_o1(dNf: wp.array4d(dtype=wp.float64),
                   d2Nf: wp.array4d(dtype=wp.float64),
                   f: wp.int32, q: wp.int32, b: wp.int32, gp: wp.int32,
                   dvec: wp.array2d(dtype=wp.float64),
                   nvec: wp.array2d(dtype=wp.float64),
                   dscale: wp.float64, dim_: wp.int32) -> wp.float64:
    """grad N_b . n_true (p1 faces: the Hessian correction is
    unrepresentable — the S13.1 failure the p2 band exists to fix;
    d2Nf accepted and ignored for signature uniformity)."""
    v = wp.float64(0.0)
    for ii in range(dim_):
        v += dNf[f, q, b, ii] * dscale * nvec[gp, ii]
    return v


def _flux_shift_fn_for(nbf: int, dim: int):
    p = round(nbf ** (1.0 / dim)) - 1
    return _flux_shift if p == 2 else _flux_shift_o1


@wp.func
def _flux_shift_ax(dNf: wp.array4d(dtype=wp.float64),
                   d2Nf: wp.array4d(dtype=wp.float64),
                   f: wp.int32, q: wp.int32, b: wp.int32, gp: wp.int32,
                   dvec: wp.array2d(dtype=wp.float64),
                   ax: wp.int32, sgn: wp.float64,
                   dscale: wp.float64, dim_: wp.int32) -> wp.float64:
    """Shifted SURROGATE-normal flux of basis b: (grad N_b + H(N_b) d).n_tilde
    (n_tilde = sgn e_ax). MEASURED REQUIREMENT (2026-07-04 overnight): the
    surrogate-normal term must be shifted too — leaving the tangential
    component unshifted (the local-p draft's Eq. 21 as literally written)
    truncates tau.H.d = O(h) with an O(1) staircase weight b = tau.n_tilde,
    producing a clean first-order L2 floor (measured orders 1.04/1.04). The
    fully-shifted-gradient (Atallah-Scovazzi S-grad) variant restores order
    2 — finding recorded for the draft's authors."""
    v = dNf[f, q, b, ax] * dscale
    d2s = dscale * dscale
    for jj in range(dim_):
        v += d2Nf[f, q, b, ax * dim_ + jj] * d2s * dvec[gp, jj]
    return sgn * v


@wp.func
def _flux_shift_ax_o1(dNf: wp.array4d(dtype=wp.float64),
                      d2Nf: wp.array4d(dtype=wp.float64),
                      f: wp.int32, q: wp.int32, b: wp.int32, gp: wp.int32,
                      dvec: wp.array2d(dtype=wp.float64),
                      ax: wp.int32, sgn: wp.float64,
                      dscale: wp.float64, dim_: wp.int32) -> wp.float64:
    """p1 twin of _flux_shift_ax: unshifted grad N_b . n_tilde."""
    return sgn * dNf[f, q, b, ax] * dscale


def _flux_shift_ax_fn_for(nbf: int, dim: int):
    p = round(nbf ** (1.0 / dim)) - 1
    return _flux_shift_ax if p == 2 else _flux_shift_ax_o1


def make_sbm_neumann_Ae(nbf: int, nqf: int, dim: int):
    """Neumann boundary bilinear terms — Eq. 21 of the local-p-refinement
    draft == production HTEquation.h:1928-37 (their residual form translated
    to A u = b): A[a,b] += kappa N_a [corr (Sflux_b.n) - grad N_b.n_tilde] dS
    plus the optional beta penalty kappa beta Sflux_a Sflux_b corr^2 dS
    (production BetaForSBMNeumann, default 0). Gamma~ -> Gamma limit: corr=1,
    d=0 makes the bracket vanish — classical Neumann recovered."""
    key = ("sbm_neu_Ae", nbf, nqf, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    fluxshift_fn = _flux_shift_fn_for(nbf, dim)

    @wp.kernel(module="unique", enable_backward=False)
    def sbm_neu_Ae(felem: wp.array(dtype=wp.int32),
                   fface: wp.array(dtype=wp.int32),
                   h: wp.array(dtype=wp.float64),
                   Nf: wp.array3d(dtype=wp.float64),
                   dNf: wp.array4d(dtype=wp.float64),
                   d2Nf: wp.array4d(dtype=wp.float64),
                   wf: wp.array(dtype=wp.float64),
                   dvec: wp.array2d(dtype=wp.float64),
                   nvec: wp.array2d(dtype=wp.float64),   # true normal, out of Omega
                   corr: wp.array(dtype=wp.float64),     # a = n . n_tilde
                   beta: wp.float64, kappa: wp.float64,
                   Ae: wp.array3d(dtype=wp.float64)):
        fi = wp.tid()
        e = felem[fi]
        f = fface[fi]
        he = h[e]
        half = he * wp.float64(0.5)
        jacS = wp.float64(1.0)
        for _ in range(dim - 1):
            jacS = jacS * half
        dscale = wp.float64(2.0) / he
        ax = f / 2
        sgn = wp.float64(1.0)
        if f % 2 == 0:
            sgn = wp.float64(-1.0)
        for q in range(nqf):
            gp = fi * nqf + q
            dS = wf[q] * jacS
            a_corr = corr[gp]
            for a in range(nbf):
                Na = Nf[f, q, a]
                sfa = fluxshift_fn(dNf, d2Nf, f, q, a, gp, dvec, nvec, dscale, dim)
                for b in range(nbf):
                    gnb_surr = sgn * dNf[f, q, b, ax] * dscale
                    sfb = fluxshift_fn(dNf, d2Nf, f, q, b, gp, dvec, nvec, dscale, dim)
                    Ae[fi, a, b] += kappa * (Na * (a_corr * sfb - gnb_surr)
                                             + beta * sfa * sfb
                                             * a_corr * a_corr) * dS

    _kernel_cache[key] = sbm_neu_Ae
    return sbm_neu_Ae


def make_sbm_neumann_be(nbf: int, nqf: int, dim: int):
    """Neumann RHS: b[a] += kappa (N_a + beta Sflux_a corr) corr qbar dS,
    qbar = kappa-NORMALIZED flux grad(u).n at the mapped point (production
    default convention; the kernel multiplies by kappa)."""
    key = ("sbm_neu_be", nbf, nqf, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    fluxshift_fn = _flux_shift_fn_for(nbf, dim)

    @wp.kernel(module="unique", enable_backward=False)
    def sbm_neu_be(felem: wp.array(dtype=wp.int32),
                   fface: wp.array(dtype=wp.int32),
                   conn: wp.array2d(dtype=wp.int32),
                   h: wp.array(dtype=wp.float64),
                   Nf: wp.array3d(dtype=wp.float64),
                   dNf: wp.array4d(dtype=wp.float64),
                   d2Nf: wp.array4d(dtype=wp.float64),
                   wf: wp.array(dtype=wp.float64),
                   dvec: wp.array2d(dtype=wp.float64),
                   nvec: wp.array2d(dtype=wp.float64),
                   corr: wp.array(dtype=wp.float64),
                   qbar: wp.array(dtype=wp.float64),     # [Nf*nqf]
                   beta: wp.float64, kappa: wp.float64,
                   be: wp.array(dtype=wp.float64)):
        fi = wp.tid()
        e = felem[fi]
        f = fface[fi]
        he = h[e]
        half = he * wp.float64(0.5)
        jacS = wp.float64(1.0)
        for _ in range(dim - 1):
            jacS = jacS * half
        dscale = wp.float64(2.0) / he
        for q in range(nqf):
            gp = fi * nqf + q
            dS = wf[q] * jacS
            a_corr = corr[gp]
            qv = qbar[gp]
            for a in range(nbf):
                Na = Nf[f, q, a]
                sfa = fluxshift_fn(dNf, d2Nf, f, q, a, gp, dvec, nvec, dscale, dim)
                wp.atomic_add(be, conn[e, a],
                              kappa * (Na + beta * sfa * a_corr)
                              * a_corr * qv * dS)

    _kernel_cache[key] = sbm_neu_be
    return sbm_neu_be


def make_sbm_dirichlet_Ae(nbf: int, nqf: int, dim: int):
    """Per-face element matrices of the three Dirichlet boundary terms."""
    key = ("sbm_dir_Ae", nbf, nqf, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    shift_fn = _shift_fn_for(nbf, dim)

    @wp.kernel(module="unique", enable_backward=False)
    def sbm_dir_Ae(felem: wp.array(dtype=wp.int32),    # [Nf] bin-local rows
                   fface: wp.array(dtype=wp.int32),    # [Nf] face ids
                   h: wp.array(dtype=wp.float64),      # bin-local element h
                   Nf: wp.array3d(dtype=wp.float64),   # [2*dim, nqf, nbf]
                   dNf: wp.array4d(dtype=wp.float64),  # [2*dim, nqf, nbf, dim]
                   d2Nf: wp.array4d(dtype=wp.float64), # [2*dim, nqf, nbf, dim*dim]
                   wf: wp.array(dtype=wp.float64),     # [nqf]
                   dvec: wp.array2d(dtype=wp.float64), # [Nf*nqf, dim]
                   alpha: wp.float64, kappa: wp.float64,
                   Ae: wp.array3d(dtype=wp.float64)):  # [Nf, nbf, nbf]
        fi = wp.tid()
        e = felem[fi]
        f = fface[fi]
        he = h[e]
        half = he * wp.float64(0.5)
        jacS = wp.float64(1.0)
        for _ in range(dim - 1):
            jacS = jacS * half                         # dS = w * (h/2)^(dim-1)
        dscale = wp.float64(2.0) / he
        ax = f / 2
        sgn = wp.float64(1.0)
        if f % 2 == 0:
            sgn = wp.float64(-1.0)                     # n_tilde = sgn * e_ax
        for q in range(nqf):
            dS = wf[q] * jacS
            gp = fi * nqf + q
            for a in range(nbf):
                Na = Nf[f, q, a]
                gna = sgn * dNf[f, q, a, ax] * dscale
                Sa = shift_fn(Nf, dNf, d2Nf, f, q, a, gp, dvec, dscale, dim)
                for b in range(nbf):
                    gnb = sgn * dNf[f, q, b, ax] * dscale
                    Sb = shift_fn(Nf, dNf, d2Nf, f, q, b, gp, dvec, dscale, dim)
                    Ae[fi, a, b] += kappa * (-Na * gnb - gna * Sb
                                             + alpha / he * Sa * Sb) * dS

    _kernel_cache[key] = sbm_dir_Ae
    return sbm_dir_Ae


def make_sbm_dirichlet_be(nbf: int, nqf: int, dim: int):
    """RHS boundary terms: be_a += kappa (-gna + alpha/h Sa) gbar dS."""
    key = ("sbm_dir_be", nbf, nqf, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    shift_fn = _shift_fn_for(nbf, dim)

    @wp.kernel(module="unique", enable_backward=False)
    def sbm_dir_be(felem: wp.array(dtype=wp.int32),
                   fface: wp.array(dtype=wp.int32),
                   conn: wp.array2d(dtype=wp.int32),   # bin-local [nb, nbf]
                   h: wp.array(dtype=wp.float64),
                   Nf: wp.array3d(dtype=wp.float64),
                   dNf: wp.array4d(dtype=wp.float64),
                   d2Nf: wp.array4d(dtype=wp.float64),
                   wf: wp.array(dtype=wp.float64),
                   dvec: wp.array2d(dtype=wp.float64),
                   gbar: wp.array(dtype=wp.float64),   # [Nf*nqf] mapped data
                   alpha: wp.float64, kappa: wp.float64,
                   be: wp.array(dtype=wp.float64)):    # [n_nodes]
        fi = wp.tid()
        e = felem[fi]
        f = fface[fi]
        he = h[e]
        half = he * wp.float64(0.5)
        jacS = wp.float64(1.0)
        for _ in range(dim - 1):
            jacS = jacS * half
        dscale = wp.float64(2.0) / he
        ax = f / 2
        sgn = wp.float64(1.0)
        if f % 2 == 0:
            sgn = wp.float64(-1.0)
        for q in range(nqf):
            dS = wf[q] * jacS
            gp = fi * nqf + q
            gq = gbar[gp]
            for a in range(nbf):
                gna = sgn * dNf[f, q, a, ax] * dscale
                Sa = shift_fn(Nf, dNf, d2Nf, f, q, a, gp, dvec, dscale, dim)
                wp.atomic_add(be, conn[e, a],
                              kappa * (-gna + alpha / he * Sa) * gq * dS)

    _kernel_cache[key] = sbm_dir_be
    return sbm_dir_be


def make_sbm_dirichlet_Ae_var(nbf: int, nqf: int, dim: int):
    """Var-kappa variant of sbm_dir_Ae: kq[fi*nqf + q] replaces the scalar
    (spatially-varying conductivity at the surrogate face GP — the local
    PDE coefficient, spec S6.3). Separate factory: scalar path untouched."""
    key = ("sbm_dir_Ae_var", nbf, nqf, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    shift_fn = _shift_fn_for(nbf, dim)

    @wp.kernel(module="unique", enable_backward=False)
    def sbm_dir_Ae_var(felem: wp.array(dtype=wp.int32),
                       fface: wp.array(dtype=wp.int32),
                       h: wp.array(dtype=wp.float64),
                       Nf: wp.array3d(dtype=wp.float64),
                       dNf: wp.array4d(dtype=wp.float64),
                       d2Nf: wp.array4d(dtype=wp.float64),
                       wf: wp.array(dtype=wp.float64),
                       dvec: wp.array2d(dtype=wp.float64),
                       kq: wp.array(dtype=wp.float64),      # [Nf*nqf]
                       alpha: wp.float64,
                       Ae: wp.array3d(dtype=wp.float64)):
        fi = wp.tid()
        e = felem[fi]
        f = fface[fi]
        he = h[e]
        half = he * wp.float64(0.5)
        jacS = wp.float64(1.0)
        for _ in range(dim - 1):
            jacS = jacS * half
        dscale = wp.float64(2.0) / he
        ax = f / 2
        sgn = wp.float64(1.0)
        if f % 2 == 0:
            sgn = wp.float64(-1.0)
        for q in range(nqf):
            gp = fi * nqf + q
            dS = wf[q] * jacS * kq[gp]
            for a in range(nbf):
                Na = Nf[f, q, a]
                gna = sgn * dNf[f, q, a, ax] * dscale
                Sa = shift_fn(Nf, dNf, d2Nf, f, q, a, gp, dvec, dscale, dim)
                for b in range(nbf):
                    gnb = sgn * dNf[f, q, b, ax] * dscale
                    Sb = shift_fn(Nf, dNf, d2Nf, f, q, b, gp, dvec, dscale, dim)
                    Ae[fi, a, b] += (-Na * gnb - gna * Sb
                                     + alpha / he * Sa * Sb) * dS

    _kernel_cache[key] = sbm_dir_Ae_var
    return sbm_dir_Ae_var


def make_sbm_dirichlet_be_var(nbf: int, nqf: int, dim: int):
    """Var-kappa variant of sbm_dir_be (see make_sbm_dirichlet_Ae_var)."""
    key = ("sbm_dir_be_var", nbf, nqf, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    shift_fn = _shift_fn_for(nbf, dim)

    @wp.kernel(module="unique", enable_backward=False)
    def sbm_dir_be_var(felem: wp.array(dtype=wp.int32),
                       fface: wp.array(dtype=wp.int32),
                       conn: wp.array2d(dtype=wp.int32),
                       h: wp.array(dtype=wp.float64),
                       Nf: wp.array3d(dtype=wp.float64),
                       dNf: wp.array4d(dtype=wp.float64),
                       d2Nf: wp.array4d(dtype=wp.float64),
                       wf: wp.array(dtype=wp.float64),
                       dvec: wp.array2d(dtype=wp.float64),
                       kq: wp.array(dtype=wp.float64),      # [Nf*nqf]
                       gbar: wp.array(dtype=wp.float64),
                       alpha: wp.float64,
                       be: wp.array(dtype=wp.float64)):
        fi = wp.tid()
        e = felem[fi]
        f = fface[fi]
        he = h[e]
        half = he * wp.float64(0.5)
        jacS = wp.float64(1.0)
        for _ in range(dim - 1):
            jacS = jacS * half
        dscale = wp.float64(2.0) / he
        ax = f / 2
        sgn = wp.float64(1.0)
        if f % 2 == 0:
            sgn = wp.float64(-1.0)
        for q in range(nqf):
            gp = fi * nqf + q
            dS = wf[q] * jacS * kq[gp]
            gq = gbar[gp]
            for a in range(nbf):
                gna = sgn * dNf[f, q, a, ax] * dscale
                Sa = shift_fn(Nf, dNf, d2Nf, f, q, a, gp, dvec, dscale, dim)
                wp.atomic_add(be, conn[e, a],
                              (-gna + alpha / he * Sa) * gq * dS)

    _kernel_cache[key] = sbm_dir_be_var
    return sbm_dir_be_var


class _FaceSet:
    """Prepared surrogate face set: p-bin resolution (M1a single-face-p
    rule), bin-local rows, face tables, and device uploads. need_normals
    adds the true-normal and area-correction arrays (Neumann sets)."""

    def __init__(self, dm, sf, geo, need_normals=False):
        self.sf, self.geo = sf, geo
        d = dm.device
        p_face = np.unique(np.asarray(dm.mesh.p_elem)[sf.elem])
        if len(p_face) != 1:
            raise ValueError(
                f"surrogate faces span p-bins {p_face.tolist()}; M1a requires "
                "one face order (put the whole band at one p)")
        self.pv = int(p_face[0])
        eids = dm.mesh.bins[self.pv]
        row_of = np.full(len(dm.mesh.tree), -1, np.int64)
        row_of[eids] = np.arange(len(eids))
        self.felem_rows = row_of[sf.elem]
        if (self.felem_rows < 0).any():
            raise RuntimeError("face element missing from its p-bin")
        self.ftab = face_tables(self.pv, dm.dim)
        if len(geo.corr) != len(sf.elem) * self.ftab.nqf:
            raise ValueError(
                "GeometryData was evaluated with a different face table "
                f"(expected {len(sf.elem) * self.ftab.nqf} face GPs, got "
                f"{len(geo.corr)})")
        self.felem_d = wp.array(self.felem_rows.astype(np.int32),
                                dtype=wp.int32, device=d)
        self.fface_d = wp.array(sf.face.astype(np.int32),
                                dtype=wp.int32, device=d)
        self.Nf_d = wp.array(np.ascontiguousarray(self.ftab.N),
                             dtype=wp.float64, device=d)
        self.dNf_d = wp.array(np.ascontiguousarray(self.ftab.dN),
                              dtype=wp.float64, device=d)
        # Hessian face tables, flattened to 4-D for warp (femelm.fe_d2N_s)
        self.d2Nf_d = wp.array(
            np.ascontiguousarray(self.ftab.d2N.reshape(
                2 * dm.dim, self.ftab.nqf, self.ftab.nbf, dm.dim * dm.dim)),
            dtype=wp.float64, device=d)
        self.wf_d = wp.array(np.ascontiguousarray(self.ftab.w),
                             dtype=wp.float64, device=d)
        self.dvec_d = wp.array(np.ascontiguousarray(geo.d),
                               dtype=wp.float64, device=d)
        if need_normals:
            self.nvec_d = wp.array(np.ascontiguousarray(geo.n),
                                   dtype=wp.float64, device=d)
            self.corr_d = wp.array(np.ascontiguousarray(geo.corr),
                                   dtype=wp.float64, device=d)

    def scatter_conn(self, dm):
        return dm.mesh.conn_of[self.pv][self.felem_rows]      # [Nf, nbf]


def surrogate_flux(dm, sf, geo, u_all, kappa=1.0):
    """Area-corrected shifted flux over a surrogate face set:
    int kappa (S grad u . n)(n . n_tilde) dS~ — the true-boundary flux
    estimate (production Surrogate2True Nu_0). S grad u includes the Hessian
    term on p2 faces (shift order = basis order). Host-side: a scalar
    observable at M1a sizes; kernelized with the observables layer (spec
    S14) in M1b."""
    mesh = dm.mesh
    p_face = np.unique(np.asarray(mesh.p_elem)[sf.elem])
    assert len(p_face) == 1
    pv = int(p_face[0])
    ftab = face_tables(pv, dm.dim)
    eids = mesh.bins[pv]
    row_of = np.full(len(mesh.tree), -1, np.int64)
    row_of[eids] = np.arange(len(eids))
    rows = row_of[sf.elem]
    conn = mesh.conn_of[pv][rows]
    uc = np.asarray(u_all)[conn]                       # [Nf, nbf]
    h = mesh.tree.h()[sf.elem]
    nqf, dim = ftab.nqf, dm.dim
    dvec = geo.d.reshape(len(sf.elem), nqf, dim)
    nvec = geo.n.reshape(len(sf.elem), nqf, dim)
    corr = geo.corr.reshape(len(sf.elem), nqf)
    total = 0.0
    for i in range(len(sf.elem)):
        f = int(sf.face[i])
        dscale = 2.0 / h[i]
        grad = np.einsum("qad,a->qd", ftab.dN[f], uc[i]) * dscale
        if pv == 2:
            H = np.einsum("qaij,a->qij", ftab.d2N[f], uc[i]) * dscale ** 2
            grad = grad + np.einsum("qij,qj->qi", H, dvec[i])
        sflux = np.einsum("qd,qd->q", grad, nvec[i])
        dS = ftab.w * (h[i] / 2.0) ** (dim - 1)
        total += kappa * float((sflux * corr[i] * dS).sum())
    return total


class SBMPoisson:
    """Octree-SBM Poisson problem: volume stiffness + optional shifted
    Dirichlet face set (geo/sf/g_fn) + optional shifted Neumann face set
    (neumann=(sf_N, geo_N, q_fn), Eq. 21 / production HTEquation form) +
    optional strong Dirichlet on the outer box (g_outer_fn at solve time).

    M1a face-bin rule: each face SET must carry ONE polynomial order
    (uniform-p meshes trivially; the mixed-p Neumann band puts all faces in
    the p2 bin by construction — spec S13.1's hard rule).

    q_fn contract: kappa-NORMALIZED true flux grad(u).n at the MAPPED point
    (kernels multiply by kappa; production DividePeOutForFlux=False)."""

    def __init__(self, dm, geo=None, sf=None, g_fn=None, kappa=1.0,
                 alpha=10.0, neumann=None, beta_neumann=0.0):
        self.dm = dm
        self.g_fn, self.alpha = g_fn, float(alpha)
        self.beta_neumann = float(beta_neumann)
        # kappa: scalar, or callable kappa(x [M,dim]) -> [M] evaluated per
        # Gauss point (the spec S6.3 closure-hook pathway). Field kappa
        # forfeits the kappa-linearity meta (A1/bg1) and the M1a scalar
        # kappa-gradient; the field-kappa gradient is M2's closure interface.
        self.kappa_fn = kappa if callable(kappa) else None
        self.kappa = 1.0 if callable(kappa) else float(kappa)

        self.dir = _FaceSet(dm, sf, geo) if sf is not None else None
        if neumann is not None:
            sf_n, geo_n, q_fn = neumann
            if self.kappa_fn is not None:
                raise NotImplementedError(
                    "field kappa + SBM Neumann arrives with the M2 closure "
                    "interface")
            self.neu = _FaceSet(dm, sf_n, geo_n, need_normals=True)
            self.q_fn = q_fn
        else:
            self.neu, self.q_fn = None, None
        # back-compat attributes (Task 6 tests introspect these)
        self.geo, self.sf = geo, sf
        if self.dir is not None:
            self.pv, self.felem_rows, self.ftab = (self.dir.pv,
                                                   self.dir.felem_rows,
                                                   self.dir.ftab)

    # ------------------------------------------------------------------
    def _volume_load(self, f_fn) -> np.ndarray:
        """T^T (volume load vector) — free-node RHS from f."""
        from ..physics.poisson import gauss_points, make_load_kernel
        dm, d = self.dm, self.dm.device
        xq_by_bin = gauss_points(dm.mesh, dm.tables_by_p)
        F_full = wp.zeros(dm.n_nodes, dtype=wp.float64, device=d)
        for pv, b in dm.bins.items():
            fq = wp.array(np.ascontiguousarray(f_fn(xq_by_bin[pv]), np.float64),
                          dtype=wp.float64, device=d)
            lk = make_load_kernel(b["nbf"], b["nqp"], dm.dim)
            wp.launch(lk, dim=len(b["eids"]),
                      inputs=[b["conn"], b["h"], b["N"], b["w"], fq, F_full],
                      device=d)
        return np.asarray(self.dm.constraints.T.T @ F_full.numpy())

    def _dirichlet_rhs_kappa1(self, kq_face_d=None) -> np.ndarray:
        """T^T (Dirichlet face RHS): kappa = 1 (or per-GP kq for field
        kappa). Zeros when there is no Dirichlet set or no g_fn."""
        dm, d = self.dm, self.dm.device
        if self.dir is None or self.g_fn is None:
            return np.zeros(dm.n_free)
        fs = self.dir
        gbar = np.ascontiguousarray(
            self.g_fn(fs.geo.xq + fs.geo.d), np.float64)
        gbar_d = wp.array(gbar, dtype=wp.float64, device=d)
        be_full = wp.zeros(dm.n_nodes, dtype=wp.float64, device=d)
        b = dm.bins[fs.pv]
        if kq_face_d is None:
            k = make_sbm_dirichlet_be(fs.ftab.nbf, fs.ftab.nqf, dm.dim)
            wp.launch(k, dim=len(fs.sf.elem),
                      inputs=[fs.felem_d, fs.fface_d, b["conn"], b["h"],
                              fs.Nf_d, fs.dNf_d, fs.d2Nf_d, fs.wf_d,
                              fs.dvec_d, gbar_d, wp.float64(self.alpha),
                              wp.float64(1.0), be_full],
                      device=d)
        else:
            k = make_sbm_dirichlet_be_var(fs.ftab.nbf, fs.ftab.nqf, dm.dim)
            wp.launch(k, dim=len(fs.sf.elem),
                      inputs=[fs.felem_d, fs.fface_d, b["conn"], b["h"],
                              fs.Nf_d, fs.dNf_d, fs.d2Nf_d, fs.wf_d,
                              fs.dvec_d, kq_face_d, gbar_d,
                              wp.float64(self.alpha), be_full],
                      device=d)
        return np.asarray(self.dm.constraints.T.T @ be_full.numpy())

    def _neumann_rhs_kappa1(self) -> np.ndarray:
        """T^T (Neumann face RHS at kappa = 1)."""
        dm, d = self.dm, self.dm.device
        if self.neu is None:
            return np.zeros(dm.n_free)
        fs = self.neu
        qbar = np.ascontiguousarray(
            self.q_fn(fs.geo.xq + fs.geo.d), np.float64)
        qbar_d = wp.array(qbar, dtype=wp.float64, device=d)
        be_full = wp.zeros(dm.n_nodes, dtype=wp.float64, device=d)
        b = dm.bins[fs.pv]
        k = make_sbm_neumann_be(fs.ftab.nbf, fs.ftab.nqf, dm.dim)
        wp.launch(k, dim=len(fs.sf.elem),
                  inputs=[fs.felem_d, fs.fface_d, b["conn"], b["h"],
                          fs.Nf_d, fs.dNf_d, fs.d2Nf_d, fs.wf_d,
                          fs.dvec_d, fs.nvec_d, fs.corr_d, qbar_d,
                          wp.float64(self.beta_neumann), wp.float64(1.0),
                          be_full],
                  device=d)
        return np.asarray(self.dm.constraints.T.T @ be_full.numpy())

    def _face_triplets_kappa1(self, kq_face_d=None):
        """COO triplets of all face bilinear terms at kappa = 1."""
        dm, d = self.dm, self.dm.device
        rows, cols, vals = [], [], []
        if self.dir is not None:
            fs = self.dir
            nbf, nqf = fs.ftab.nbf, fs.ftab.nqf
            nfc = len(fs.sf.elem)
            Ae = wp.zeros((nfc, nbf, nbf), dtype=wp.float64, device=d)
            b = dm.bins[fs.pv]
            if kq_face_d is None:
                k = make_sbm_dirichlet_Ae(nbf, nqf, dm.dim)
                wp.launch(k, dim=nfc,
                          inputs=[fs.felem_d, fs.fface_d, b["h"], fs.Nf_d,
                                  fs.dNf_d, fs.d2Nf_d, fs.wf_d, fs.dvec_d,
                                  wp.float64(self.alpha), wp.float64(1.0),
                                  Ae],
                          device=d)
            else:
                k = make_sbm_dirichlet_Ae_var(nbf, nqf, dm.dim)
                wp.launch(k, dim=nfc,
                          inputs=[fs.felem_d, fs.fface_d, b["h"], fs.Nf_d,
                                  fs.dNf_d, fs.d2Nf_d, fs.wf_d, fs.dvec_d,
                                  kq_face_d, wp.float64(self.alpha), Ae],
                          device=d)
            conn = fs.scatter_conn(dm)
            rows.append(np.repeat(conn, nbf, axis=1).ravel())
            cols.append(np.tile(conn, (1, nbf)).ravel())
            vals.append(Ae.numpy().ravel())
        if self.neu is not None:
            fs = self.neu
            nbf, nqf = fs.ftab.nbf, fs.ftab.nqf
            nfc = len(fs.sf.elem)
            Ae = wp.zeros((nfc, nbf, nbf), dtype=wp.float64, device=d)
            b = dm.bins[fs.pv]
            k = make_sbm_neumann_Ae(nbf, nqf, dm.dim)
            wp.launch(k, dim=nfc,
                      inputs=[fs.felem_d, fs.fface_d, b["h"], fs.Nf_d,
                              fs.dNf_d, fs.d2Nf_d, fs.wf_d, fs.dvec_d,
                              fs.nvec_d, fs.corr_d,
                              wp.float64(self.beta_neumann), wp.float64(1.0),
                              Ae],
                      device=d)
            conn = fs.scatter_conn(dm)
            rows.append(np.repeat(conn, nbf, axis=1).ravel())
            cols.append(np.tile(conn, (1, nbf)).ravel())
            vals.append(Ae.numpy().ravel())
        return rows, cols, vals

    def assemble(self, f_fn, g_outer_fn=None):
        """Returns (A csr, b, meta). Scalar kappa: meta carries the kappa=1
        pieces A1/bg1 (pre-row-replacement; bg1 = ALL kappa-proportional RHS,
        Dirichlet + Neumann) for the Task-10 kappa gradient. Field kappa
        (callable): coefficients baked in per Gauss point;
        meta["field_kappa"]=True with A1=bg1=None (no linearity trick)."""
        dm, d = self.dm, self.dm.device
        field = self.kappa_fn is not None
        kq_face_d = None
        if field:
            from ..physics.poisson import gauss_points
            xq_by_bin = gauss_points(dm.mesh, dm.tables_by_p)
            kq_by_bin = {pv: np.ascontiguousarray(self.kappa_fn(xq), np.float64)
                         for pv, xq in xq_by_bin.items()}
            kmin = min(kq.min() for kq in kq_by_bin.values())
            if self.dir is not None:
                kq_face = np.ascontiguousarray(
                    self.kappa_fn(self.dir.geo.xq), np.float64)
                kmin = min(kmin, kq_face.min())
                kq_face_d = wp.array(kq_face, dtype=wp.float64, device=d)
            if kmin <= 0.0:
                raise ValueError(f"kappa field must be positive (min {kmin})")
            rows, cols, vals = volume_triplets(dm, kq_by_bin)
        else:
            rows, cols, vals = volume_triplets(dm)

        frows, fcols, fvals = self._face_triplets_kappa1(kq_face_d)
        K1 = sp.coo_matrix(
            (np.concatenate([vals, *fvals]),
             (np.concatenate([rows, *frows]), np.concatenate([cols, *fcols]))),
            shape=(dm.n_nodes, dm.n_nodes)).tocsr()
        T = dm.constraints.T.tocsr()
        A1 = (T.T @ K1 @ T).tocsr()

        b_f = self._volume_load(f_fn)
        if field:
            bg = self._dirichlet_rhs_kappa1(kq_face_d)
            A = A1
            rhs = b_f + bg
            meta = {"A1": None, "bg1": None, "bf": b_f, "dir_rows": None,
                    "field_kappa": True}
        else:
            bg1 = self._dirichlet_rhs_kappa1() + self._neumann_rhs_kappa1()
            A = (self.kappa * A1).tocsr()
            rhs = b_f + self.kappa * bg1
            meta = {"A1": A1, "bg1": bg1, "bf": b_f, "dir_rows": None,
                    "field_kappa": False}

        if g_outer_fn is not None:
            dirf = dm.mesh.boundary_nodes[dm.constraints.free_nodes]
            idx = np.where(dirf)[0]
            if len(idx):
                # strong outer Dirichlet by row replacement (nonsym solver
                # anyway; identity rows keep the Jacobi diagonal sane).
                # Direct LIL row surgery — NOT A[idx, :] = 0, which makes
                # scipy broadcast a DENSE (len(idx) x N) zero block
                # (measured: 1.71 TiB attempted at 3D L7, 98k rows x 2.4M).
                A = A.tolil()
                for i in idx:
                    A.rows[i] = [int(i)]
                    A.data[i] = [1.0]
                A = A.tocsr()
                coords = dm.mesh.node_coords[dm.constraints.free_nodes][idx]
                rhs = rhs.copy()
                rhs[idx] = g_outer_fn(coords)
                meta["dir_rows"] = idx
        return A, rhs, meta

    def solve(self, f_fn, g_outer_fn=None, tol=1e-12, maxiter=20000,
              solver="direct"):
        """Assemble + solve + expand: returns u at ALL nodes.

        solver="direct" (default): scipy splu on the host — the prototype
        analogue of the cuDSS direct path in the solver reuse map (spec
        S5.4); exact, so `tol` is ignored. Appropriate for the M1a
        assembled-CSR path at prototype sizes. solver="bicgstab": in-framework
        Krylov (Jacobi), the path that generalizes to matrix-free in M1b —
        measured 2026-07-04: host-sync dot() latency on WSL2 makes it
        ~100-400x slower than splu at these sizes (the recorded M0/M1a
        deferred perf item), so it is opt-in here and exercised by a
        dedicated patch test to stay honest.
        """
        A, rhs, meta = self.assemble(f_fn, g_outer_fn)
        if solver == "direct":
            from scipy.sparse.linalg import splu
            x = splu(A.tocsc()).solve(rhs)
        elif solver == "bicgstab":
            op = CSROperator(A, self.dm.device)
            x, info = bicgstab(op, rhs, tol=tol, maxiter=maxiter,
                               diag=np.asarray(A.diagonal()))
            assert info["converged"], info
        else:
            raise ValueError(f"unknown solver {solver!r}")
        return np.asarray(self.dm.constraints.T @ x)
