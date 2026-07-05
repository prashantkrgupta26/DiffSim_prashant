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


class SBMPoisson:
    """Octree-SBM Poisson problem: volume stiffness + Dirichlet face terms
    on the surrogate boundary, optional strong Dirichlet on the outer box.

    M1a face-bin rule: all surrogate faces must carry ONE polynomial order
    (uniform-p meshes trivially; the mixed-p Neumann band puts all faces in
    the p2 bin by construction — spec S13.1's hard rule)."""

    def __init__(self, dm, geo, sf, g_fn=None, kappa=1.0, alpha=10.0):
        self.dm, self.geo, self.sf = dm, geo, sf
        self.g_fn, self.alpha = g_fn, float(alpha)
        # kappa: scalar, or callable kappa(x [M,dim]) -> [M] evaluated per
        # Gauss point (the spec S6.3 closure-hook pathway). Field kappa
        # forfeits the kappa-linearity meta (A1/bg1) and the M1a scalar
        # kappa-gradient; the field-kappa gradient is M2's closure interface.
        self.kappa_fn = kappa if callable(kappa) else None
        self.kappa = 1.0 if callable(kappa) else float(kappa)
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

        self._felem_d = wp.array(self.felem_rows.astype(np.int32),
                                 dtype=wp.int32, device=d)
        self._fface_d = wp.array(sf.face.astype(np.int32),
                                 dtype=wp.int32, device=d)
        self._Nf_d = wp.array(np.ascontiguousarray(self.ftab.N),
                              dtype=wp.float64, device=d)
        self._dNf_d = wp.array(np.ascontiguousarray(self.ftab.dN),
                               dtype=wp.float64, device=d)
        # Hessian face tables, flattened to 4-D for warp (see femelm.fe_d2N_s)
        self._d2Nf_d = wp.array(
            np.ascontiguousarray(self.ftab.d2N.reshape(
                2 * dm.dim, self.ftab.nqf, self.ftab.nbf, dm.dim * dm.dim)),
            dtype=wp.float64, device=d)
        self._wf_d = wp.array(np.ascontiguousarray(self.ftab.w),
                              dtype=wp.float64, device=d)
        self._dvec_d = wp.array(np.ascontiguousarray(self.geo.d),
                                dtype=wp.float64, device=d)

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

    def _face_rhs_kappa1(self, kq_face_d=None) -> np.ndarray:
        """T^T (Dirichlet face RHS): at kappa = 1 for scalar kappa, or with
        per-GP kq for a field kappa. Zeros when g_fn is None."""
        dm, d = self.dm, self.dm.device
        if self.g_fn is None:
            return np.zeros(dm.n_free)
        gbar = np.ascontiguousarray(
            self.g_fn(self.geo.xq + self.geo.d), np.float64)
        gbar_d = wp.array(gbar, dtype=wp.float64, device=d)
        be_full = wp.zeros(dm.n_nodes, dtype=wp.float64, device=d)
        b = dm.bins[self.pv]
        if kq_face_d is None:
            k = make_sbm_dirichlet_be(self.ftab.nbf, self.ftab.nqf, dm.dim)
            wp.launch(k, dim=len(self.sf.elem),
                      inputs=[self._felem_d, self._fface_d, b["conn"], b["h"],
                              self._Nf_d, self._dNf_d, self._d2Nf_d, self._wf_d,
                              self._dvec_d, gbar_d, wp.float64(self.alpha),
                              wp.float64(1.0), be_full],
                      device=d)
        else:
            k = make_sbm_dirichlet_be_var(self.ftab.nbf, self.ftab.nqf, dm.dim)
            wp.launch(k, dim=len(self.sf.elem),
                      inputs=[self._felem_d, self._fface_d, b["conn"], b["h"],
                              self._Nf_d, self._dNf_d, self._d2Nf_d, self._wf_d,
                              self._dvec_d, kq_face_d, gbar_d,
                              wp.float64(self.alpha), be_full],
                      device=d)
        return np.asarray(self.dm.constraints.T.T @ be_full.numpy())

    def assemble(self, f_fn, g_outer_fn=None):
        """Returns (A csr, b, meta). Scalar kappa: meta carries the kappa=1
        pieces A1/bg1 (pre-row-replacement) for the Task-10 kappa gradient.
        Field kappa (callable): coefficients baked in per Gauss point;
        meta["field_kappa"]=True with A1=bg1=None (no linearity trick)."""
        dm, d = self.dm, self.dm.device
        field = self.kappa_fn is not None
        kq_face_d = None
        if field:
            from ..physics.poisson import gauss_points
            xq_by_bin = gauss_points(dm.mesh, dm.tables_by_p)
            kq_by_bin = {pv: np.ascontiguousarray(self.kappa_fn(xq), np.float64)
                         for pv, xq in xq_by_bin.items()}
            kq_face = np.ascontiguousarray(self.kappa_fn(self.geo.xq), np.float64)
            kmin = min(min(kq.min() for kq in kq_by_bin.values()), kq_face.min())
            if kmin <= 0.0:
                raise ValueError(f"kappa field must be positive (min {kmin})")
            kq_face_d = wp.array(kq_face, dtype=wp.float64, device=d)
            rows, cols, vals = volume_triplets(dm, kq_by_bin)
        else:
            rows, cols, vals = volume_triplets(dm)

        nbf, nqf = self.ftab.nbf, self.ftab.nqf
        nfc = len(self.sf.elem)
        Ae = wp.zeros((nfc, nbf, nbf), dtype=wp.float64, device=d)
        b = dm.bins[self.pv]
        if field:
            k = make_sbm_dirichlet_Ae_var(nbf, nqf, dm.dim)
            wp.launch(k, dim=nfc,
                      inputs=[self._felem_d, self._fface_d, b["h"], self._Nf_d,
                              self._dNf_d, self._d2Nf_d, self._wf_d,
                              self._dvec_d, kq_face_d, wp.float64(self.alpha),
                              Ae],
                      device=d)
        else:
            k = make_sbm_dirichlet_Ae(nbf, nqf, dm.dim)
            wp.launch(k, dim=nfc,
                      inputs=[self._felem_d, self._fface_d, b["h"], self._Nf_d,
                              self._dNf_d, self._d2Nf_d, self._wf_d,
                              self._dvec_d, wp.float64(self.alpha),
                              wp.float64(1.0), Ae],
                      device=d)
        conn = dm.mesh.conn_of[self.pv][self.felem_rows]      # [Nf, nbf]
        frows = np.repeat(conn, nbf, axis=1).ravel()
        fcols = np.tile(conn, (1, nbf)).ravel()
        K1 = sp.coo_matrix(
            (np.concatenate([vals, Ae.numpy().ravel()]),
             (np.concatenate([rows, frows]), np.concatenate([cols, fcols]))),
            shape=(dm.n_nodes, dm.n_nodes)).tocsr()
        T = dm.constraints.T.tocsr()
        A1 = (T.T @ K1 @ T).tocsr()

        b_f = self._volume_load(f_fn)
        if field:
            bg = self._face_rhs_kappa1(kq_face_d)
            A = A1
            rhs = b_f + bg
            meta = {"A1": None, "bg1": None, "bf": b_f, "dir_rows": None,
                    "field_kappa": True}
        else:
            bg1 = self._face_rhs_kappa1()
            A = (self.kappa * A1).tocsr()
            rhs = b_f + self.kappa * bg1
            meta = {"A1": A1, "bg1": bg1, "bf": b_f, "dir_rows": None,
                    "field_kappa": False}

        if g_outer_fn is not None:
            dirf = dm.mesh.boundary_nodes[dm.constraints.free_nodes]
            idx = np.where(dirf)[0]
            if len(idx):
                # strong outer Dirichlet by row replacement (nonsym solver
                # anyway; identity rows keep the Jacobi diagonal sane)
                A = A.tolil()
                A[idx, :] = 0.0
                A[idx, idx] = 1.0
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
