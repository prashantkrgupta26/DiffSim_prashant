import numpy as np
import scipy.sparse as sp
import warp as wp
from .femelm import FEMElm, fe_N, fe_dN, fe_detJxW, fe_dN_s, fe_detJxW_s

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
    def __init__(self, mesh, constraints, tables, device):
        self.mesh = mesh
        self.constraints = constraints
        self.tables = tables
        self.device = device
        self.dim = mesh.dim          # space dimension (2, 3, or 4)
        self.conn = wp.array(
            np.ascontiguousarray(mesh.conn),
            dtype=wp.int32,
            device=device,
        )
        self.h = wp.array(
            np.ascontiguousarray(mesh.tree.h().astype(np.float64)),
            dtype=wp.float64,
            device=device,
        )
        self.N = wp.array(
            np.ascontiguousarray(tables.N.astype(np.float64)),
            dtype=wp.float64,
            device=device,
        )
        self.dN = wp.array(
            np.ascontiguousarray(tables.dN.astype(np.float64)),
            dtype=wp.float64,
            device=device,
        )
        self.w = wp.array(
            np.ascontiguousarray(tables.w.astype(np.float64)),
            dtype=wp.float64,
            device=device,
        )
        T = constraints.T.tocsr()
        self.T_dev = _csr_to_device(T, device)
        self.Tt_dev = _csr_to_device(T.T.tocsr(), device)
        self.n_nodes = len(mesh.node_coords)
        self.n_free = T.shape[1]

    @classmethod
    def from_mesh(cls, m, c, t, d):
        return cls(m, c, t, d)


def integrate_volume(dm: DeviceMesh) -> float:
    out = wp.zeros(1, dtype=wp.float64, device=dm.device)
    kernel = make_volume_kernel(dm.tables.nqp, dm.dim)
    wp.launch(kernel, dim=len(dm.mesh.tree),
              inputs=[dm.h, dm.w, out], device=dm.device)
    return float(out.numpy()[0])


class ConstrainedOperator:
    """y_free = T^T (A (T x_free)), where A is the Poisson stiffness action (no BCs)."""

    def __init__(self, dm: DeviceMesh):
        self.dm = dm
        self.kernel = make_poisson_matvec(dm.tables.nbf, dm.tables.nqp, dm.dim)
        d = dm.device
        self.x_full = wp.zeros(dm.n_nodes, dtype=wp.float64, device=d)
        self.y_full = wp.zeros(dm.n_nodes, dtype=wp.float64, device=d)

    def matvec(self, x_free: wp.array, y_free: wp.array):
        dm, d = self.dm, self.dm.device
        # x_full = T @ x_free
        wp.launch(csr_spmv, dim=dm.n_nodes,
                  inputs=[*dm.T_dev, x_free, self.x_full], device=d)
        # y_full = A @ x_full
        self.y_full.zero_()
        wp.launch(self.kernel, dim=len(dm.mesh.tree),
                  inputs=[dm.conn, dm.h, dm.N, dm.dN, dm.w,
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


def assemble_csr(dm):
    ne, nbf, nqp = len(dm.mesh.tree), dm.tables.nbf, dm.tables.nqp
    Ke = wp.zeros((ne, nbf, nbf), dtype=wp.float64, device=dm.device)
    k = make_poisson_element_matrices(nbf, nqp, dm.dim)
    wp.launch(k, dim=ne, inputs=[dm.h, dm.dN, dm.w, Ke], device=dm.device)
    Keh = Ke.numpy()
    conn = dm.mesh.conn
    rows = np.repeat(conn, nbf, axis=1).ravel()
    cols = np.tile(conn, (1, nbf)).ravel()
    K = sp.coo_matrix((Keh.ravel(), (rows, cols)),
                      shape=(dm.n_nodes, dm.n_nodes)).tocsr()
    T = dm.constraints.T.tocsr()
    return (T.T @ K @ T).tocsr()


def operator_diagonal(dm) -> np.ndarray:
    return np.asarray(assemble_csr(dm).diagonal())
