import numpy as np
import scipy.sparse as sp
import warp as wp
from .femelm import FEMElm, fe_dN_s, fe_detJxW_s

# Volume kernels are never differentiated through wp.Tape (adjoints go through
# the assembled-CSR transpose; spec S5.2). Skipping backward codegen halves the
# module compile cost, which matters: dim=4 p2 kernels (nbf=81, unrolled) took
# ~280 s/module WITH backward on this workstation (measured 2026-07-04, RTX
# 6000 Ada, warp 1.14). M1b's matrix-free tape path must flip this per-kernel.
wp.set_module_options({"enable_backward": False})

# Dim-keyed float64 vector types for gradient accumulation.
_VEC = {2: wp.vec2d, 3: wp.vec3d, 4: wp.vec4d}


def _csr_to_device(A: sp.csr_matrix, device):
    return (
        wp.array(np.ascontiguousarray(A.indptr.astype(np.int32)),
                 dtype=wp.int32, device=device),
        wp.array(np.ascontiguousarray(A.indices.astype(np.int32)),
                 dtype=wp.int32, device=device),
        wp.array(np.ascontiguousarray(A.data.astype(np.float64)),
                 dtype=wp.float64, device=device),
    )


@wp.kernel
def csr_spmv(
    indptr: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    data: wp.array(dtype=wp.float64),
    x: wp.array(dtype=wp.float64),
    y: wp.array(dtype=wp.float64),
):
    row = wp.tid()
    acc = wp.float64(0.0)
    for j in range(indptr[row], indptr[row + 1]):
        acc += data[j] * x[indices[j]]
    y[row] = acc


_kernel_cache: dict = {}


def make_poisson_matvec(nbf: int, nqp: int, dim: int = 3):
    """Poisson matvec kernel: y += A x for Poisson stiffness.

    Cache key includes dim so kernels for different space dimensions are
    compiled and stored independently.
    Primary approach: accumulate gradient into a VEC() (wp.vec{dim}d),
    then dot with per-row gradients. dim is a Python compile-time constant
    closed over from the factory; jac = (he/2)^dim via a power loop.
    """
    key = ("poisson_mv", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    VEC = _VEC[dim]

    @wp.kernel
    def poisson_mv(
        conn: wp.array2d(dtype=wp.int32),
        h: wp.array(dtype=wp.float64),
        Ntab: wp.array2d(dtype=wp.float64),
        dNtab: wp.array3d(dtype=wp.float64),
        wtab: wp.array(dtype=wp.float64),
        x: wp.array(dtype=wp.float64),
        y: wp.array(dtype=wp.float64),
    ):
        e = wp.tid()
        fe = FEMElm()
        fe.e = e
        fe.he = h[e]
        half = fe.he * wp.float64(0.5)
        # jac = (he/2)^dim  — power loop; dim is compile-time constant
        jac = wp.float64(1.0)
        for _ in range(dim):
            jac = jac * half
        dscale = wp.float64(2.0) / fe.he
        for q in range(nqp):
            fe.q = q
            dJxW = fe_detJxW_s(wtab, fe, jac)
            # Build gradient g = sum_b dN_b * x_b  using VEC for dim-generic storage.
            g = VEC()
            for b in range(nbf):
                xb = x[conn[e, b]]
                for d in range(dim):
                    g[d] = g[d] + fe_dN_s(dNtab, fe, b, d, dscale) * xb
            # Accumulate: y_a += (grad N_a) . g * dJxW
            for a in range(nbf):
                val = wp.float64(0.0)
                for d in range(dim):
                    val = val + fe_dN_s(dNtab, fe, a, d, dscale) * g[d]
                wp.atomic_add(y, conn[e, a], val * dJxW)

    _kernel_cache[key] = poisson_mv
    return poisson_mv


def make_volume_kernel(nqp: int, dim: int = 3):
    """Volume integration kernel: accumulates element volumes into out[0].

    jac = (he/2)^dim computed via power loop (dim compile-time constant).
    """
    key = ("volume", nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel
    def volume_k(
        h: wp.array(dtype=wp.float64),
        wtab: wp.array(dtype=wp.float64),
        out: wp.array(dtype=wp.float64),
    ):
        e = wp.tid()
        acc = wp.float64(0.0)
        half = h[e] * wp.float64(0.5)
        jac = wp.float64(1.0)
        for _ in range(dim):
            jac = jac * half
        for q in range(nqp):
            acc += wtab[q] * jac
        wp.atomic_add(out, 0, acc)

    _kernel_cache[key] = volume_k
    return volume_k


class DeviceMesh:
    """Device-side mesh with per-polynomial-degree (per-bin) arrays.

    Accepts ``tables_by_p`` as either a single ``Tables`` object (uniform
    back-compat, wrapped internally as ``{mesh.p: tables}``) or a
    ``dict[int, Tables]`` for mixed-p meshes.

    Back-compat accessor asymmetry (deliberate): ``conn``/``h``/``N``/``dN``/
    ``w`` assert a single-bin (uniform) mesh, because silently returning one
    bin's arrays for a mixed mesh would drop elements. ``tables`` does NOT
    assert — it returns the max-p Tables, which is well-defined for any mesh
    (quadrature metadata, not per-element data).
    """

    def __init__(self, mesh, constraints, tables_by_p, device):
        if not isinstance(tables_by_p, dict):
            tables_by_p = {mesh.p: tables_by_p}
        self.mesh = mesh
        self.constraints = constraints
        self.device = device
        self.dim = mesh.dim
        self.tables_by_p = tables_by_p

        h_all = mesh.tree.h()
        self.bins: dict = {}
        for pv, eids in mesh.bins.items():
            tb = tables_by_p[pv]
            self.bins[pv] = dict(
                eids=eids,
                conn=wp.array(np.ascontiguousarray(mesh.conn_of[pv]),
                              dtype=wp.int32, device=device),
                h=wp.array(np.ascontiguousarray(h_all[eids].astype(np.float64)),
                           dtype=wp.float64, device=device),
                N=wp.array(np.ascontiguousarray(tb.N.astype(np.float64)),
                           dtype=wp.float64, device=device),
                dN=wp.array(np.ascontiguousarray(tb.dN.astype(np.float64)),
                            dtype=wp.float64, device=device),
                w=wp.array(np.ascontiguousarray(tb.w.astype(np.float64)),
                           dtype=wp.float64, device=device),
                nbf=tb.nbf, nqp=tb.nqp,
            )

        T = constraints.T.tocsr()
        self.T_dev = _csr_to_device(T, device)
        self.Tt_dev = _csr_to_device(T.T.tocsr(), device)
        self.n_nodes = len(mesh.node_coords)
        self.n_free = T.shape[1]

    @classmethod
    def from_mesh(cls, m, c, t, d):
        return cls(m, c, t, d)

    # ------------------------------------------------------------------
    # Uniform-mesh back-compat accessors (M0 tests use dm.conn, dm.h, …)
    # conn/h/N/dN/w assert single-bin; tables returns max-p Tables (no assert).
    # ------------------------------------------------------------------
    def _only_bin(self):
        assert len(self.bins) == 1, (
            f"Back-compat property requires a single-bin mesh; "
            f"got bins={list(self.bins.keys())}"
        )
        return next(iter(self.bins.values()))

    conn   = property(lambda s: s._only_bin()["conn"])
    h      = property(lambda s: s._only_bin()["h"])
    N      = property(lambda s: s._only_bin()["N"])
    dN     = property(lambda s: s._only_bin()["dN"])
    w      = property(lambda s: s._only_bin()["w"])
    tables = property(lambda s: s.tables_by_p[max(s.tables_by_p)])


def integrate_volume(dm: DeviceMesh) -> float:
    """Total mesh volume. Uniform mesh only (single-bin DeviceMesh); mixed-p
    meshes raise at the first back-compat accessor (``dm.h``)."""
    out = wp.zeros(1, dtype=wp.float64, device=dm.device)
    kernel = make_volume_kernel(dm.tables.nqp, dm.dim)
    wp.launch(kernel, dim=len(dm.mesh.tree),
              inputs=[dm.h, dm.w, out], device=dm.device)
    return float(out.numpy()[0])


class ConstrainedOperator:
    """y_free = T^T (A (T x_free)), where A is the Poisson stiffness action (no BCs).

    Supports mixed-p meshes: one kernel launch per bin, all accumulating into
    the same y_full via atomic_add (safe on both CPU and CUDA).
    """

    def __init__(self, dm: DeviceMesh):
        self.dm = dm
        # Pre-compile one kernel per (nbf, nqp, dim) combination.
        self._kernels = {
            pv: make_poisson_matvec(b["nbf"], b["nqp"], dm.dim)
            for pv, b in dm.bins.items()
        }
        d = dm.device
        self.x_full = wp.zeros(dm.n_nodes, dtype=wp.float64, device=d)
        self.y_full = wp.zeros(dm.n_nodes, dtype=wp.float64, device=d)

    def matvec(self, x_free: wp.array, y_free: wp.array):
        dm, d = self.dm, self.dm.device
        # x_full = T @ x_free
        wp.launch(csr_spmv, dim=dm.n_nodes,
                  inputs=[*dm.T_dev, x_free, self.x_full], device=d)
        # y_full = A @ x_full  (zero once, then per-bin accumulate)
        self.y_full.zero_()
        for pv, b in dm.bins.items():
            wp.launch(self._kernels[pv], dim=len(b["eids"]),
                      inputs=[b["conn"], b["h"], b["N"], b["dN"], b["w"],
                              self.x_full, self.y_full],
                      device=d)
        # y_free = T^T @ y_full
        wp.launch(csr_spmv, dim=dm.n_free,
                  inputs=[*dm.Tt_dev, self.y_full, y_free], device=d)

    def matvec_numpy(self, x: np.ndarray) -> np.ndarray:
        xd = wp.array(np.ascontiguousarray(x.astype(np.float64)),
                      dtype=wp.float64, device=self.dm.device)
        yd = wp.zeros(self.dm.n_free, dtype=wp.float64, device=self.dm.device)
        self.matvec(xd, yd)
        return yd.numpy()


def make_poisson_element_matrices(nbf: int, nqp: int, dim: int = 3):
    """Assemble per-element stiffness matrices Ke[e, a, b].

    dim-generic: gradient loop runs range(dim), jac = (he/2)^dim.
    """
    key = ("poisson_Ke", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel
    def poisson_Ke(h: wp.array(dtype=wp.float64), dNtab: wp.array3d(dtype=wp.float64),
                   wtab: wp.array(dtype=wp.float64),
                   Ke: wp.array3d(dtype=wp.float64)):        # [Ne, nbf, nbf]
        e = wp.tid()
        fe = FEMElm(); fe.e = e; fe.he = h[e]
        half = fe.he * wp.float64(0.5)
        jac = wp.float64(1.0)
        for _ in range(dim):
            jac = jac * half
        dscale = wp.float64(2.0) / fe.he
        for q in range(nqp):
            fe.q = q
            dJxW = fe_detJxW_s(wtab, fe, jac)
            for a in range(nbf):
                for b in range(nbf):
                    v = wp.float64(0.0)
                    for d in range(dim):
                        v = v + fe_dN_s(dNtab, fe, a, d, dscale) * fe_dN_s(dNtab, fe, b, d, dscale)
                    Ke[e, a, b] = Ke[e, a, b] + v * dJxW

    _kernel_cache[key] = poisson_Ke
    return poisson_Ke


def volume_triplets(dm):
    """Unconstrained COO triplets (rows, cols, vals) of the volume Poisson
    stiffness at kappa = 1, over all per-degree bins. Callers concatenate
    additional (e.g. SBM face) triplets before the single tocsr so shared
    nodes are summed once."""
    all_rows, all_cols, all_vals = [], [], []
    for pv, b in dm.bins.items():
        ne_bin = len(b["eids"])
        nbf = b["nbf"]
        nqp = b["nqp"]
        Ke = wp.zeros((ne_bin, nbf, nbf), dtype=wp.float64, device=dm.device)
        k = make_poisson_element_matrices(nbf, nqp, dm.dim)
        wp.launch(k, dim=ne_bin, inputs=[b["h"], b["dN"], b["w"], Ke],
                  device=dm.device)
        Keh = Ke.numpy()
        conn = dm.mesh.conn_of[pv]                      # int32 [ne_bin, nbf]
        all_rows.append(np.repeat(conn, nbf, axis=1).ravel())
        all_cols.append(np.tile(conn, (1, nbf)).ravel())
        all_vals.append(Keh.ravel())
    return (np.concatenate(all_rows), np.concatenate(all_cols),
            np.concatenate(all_vals))


def assemble_csr(dm):
    """Assemble global constrained stiffness matrix T^T K T."""
    rows, cols, vals = volume_triplets(dm)
    K = sp.coo_matrix((vals, (rows, cols)),
                      shape=(dm.n_nodes, dm.n_nodes)).tocsr()
    T = dm.constraints.T.tocsr()
    return (T.T @ K @ T).tocsr()


class CSROperator:
    """Operator-protocol wrapper for an assembled scipy CSR on device:
    .matvec(x_wp, y_wp), .matvec_numpy(x), .n_free, .device. The adjoint
    operator is simply CSROperator(A.T.tocsr(), device) — Tier-2 VJP #1's
    assembled-path form (spec S5.2)."""

    def __init__(self, A: sp.csr_matrix, device):
        A = A.tocsr()
        self.device = device
        self.n_free = A.shape[0]
        self._dev = _csr_to_device(A, device)

    def matvec(self, x: wp.array, y: wp.array):
        wp.launch(csr_spmv, dim=self.n_free,
                  inputs=[*self._dev, x, y], device=self.device)

    def matvec_numpy(self, x: np.ndarray) -> np.ndarray:
        xd = wp.array(np.ascontiguousarray(x, np.float64),
                      dtype=wp.float64, device=self.device)
        yd = wp.zeros(self.n_free, dtype=wp.float64, device=self.device)
        self.matvec(xd, yd)
        return yd.numpy()


def operator_diagonal(dm) -> np.ndarray:
    return np.asarray(assemble_csr(dm).diagonal())
