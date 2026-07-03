import numpy as np
import warp as wp
from ..assembly.femelm import FEMElm, fe_N, fe_detJxW_s
from ..assembly.operators import _kernel_cache


def gauss_points(mesh, tables):
    """Physical Gauss-point coords, [Ne*nqp, dim], ordering (e, q).

    Reference lattice uses the same x-fastest reversed-itertools idiom as
    basis_tables so (e,q) flat ordering matches the kernels' q loop.
    Physical scale: 2^{-lmax(mesh.dim)} per anchor unit.
    """
    from itertools import product as iproduct
    from ..octree import morton
    from ..mesh.basis import gauss_1d
    dim = mesh.dim
    pts, _ = gauss_1d(tables.p)
    nq1 = len(pts)
    # x-fastest ordering: iproduct varies last factor fastest, [::-1] reverses
    qidx = np.array(list(iproduct(*[range(nq1)] * dim)), np.int64)[:, ::-1]
    ref = pts[qidx]                                      # [nqp, dim]
    lo = mesh.tree.anchors() * 2.0 ** (-morton.lmax(dim))
    h = mesh.tree.h()
    xq = lo[:, None, :] + (ref[None, :, :] + 1.0) * 0.5 * h[:, None, None]
    return xq.reshape(-1, dim)


def make_load_kernel(nbf: int, nqp: int, dim: int = 3):
    """Assemble RHS load vector: be_a += f(x_q) N_a(x_q) dJxW.

    dim-generic: jac = (he/2)^dim via power loop (dim compile-time constant).
    """
    key = ("load", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel
    def load(conn: wp.array2d(dtype=wp.int32), h: wp.array(dtype=wp.float64),
             Ntab: wp.array2d(dtype=wp.float64), wtab: wp.array(dtype=wp.float64),
             fq: wp.array(dtype=wp.float64),          # f at Gauss points, [Ne*nqp]
             be: wp.array(dtype=wp.float64)):
        e = wp.tid()
        fe = FEMElm(); fe.e = e; fe.he = h[e]
        half = fe.he * wp.float64(0.5)
        jac = wp.float64(1.0)
        for _ in range(dim):
            jac = jac * half
        for q in range(nqp):
            fe.q = q
            dJxW = fe_detJxW_s(wtab, fe, jac)
            fv = fq[e * nqp + q]
            for a in range(nbf):
                wp.atomic_add(be, conn[e, a], fe_N(Ntab, fe, a) * fv * dJxW)

    _kernel_cache[key] = load
    return load


def make_l2_kernel(nbf: int, nqp: int, dim: int = 3):
    """L2 error integrator: accumulates ||u_h - u_exact||^2 into out[0].

    dim-generic: jac = (he/2)^dim via power loop (dim compile-time constant).
    """
    key = ("l2", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel
    def l2(conn: wp.array2d(dtype=wp.int32), h: wp.array(dtype=wp.float64),
           Ntab: wp.array2d(dtype=wp.float64), wtab: wp.array(dtype=wp.float64),
           u: wp.array(dtype=wp.float64), uq_exact: wp.array(dtype=wp.float64),
           out: wp.array(dtype=wp.float64)):
        e = wp.tid()
        fe = FEMElm(); fe.e = e; fe.he = h[e]
        half = fe.he * wp.float64(0.5)
        jac = wp.float64(1.0)
        for _ in range(dim):
            jac = jac * half
        acc = wp.float64(0.0)
        for q in range(nqp):
            fe.q = q
            uh = wp.float64(0.0)
            for a in range(nbf):
                uh += fe_N(Ntab, fe, a) * u[conn[e, a]]
            diff = uh - uq_exact[e * nqp + q]
            acc += diff * diff * fe_detJxW_s(wtab, fe, jac)
        wp.atomic_add(out, 0, acc)

    _kernel_cache[key] = l2
    return l2


def l2_error(dm, u_all: np.ndarray, u_exact_fn) -> float:
    xq = gauss_points(dm.mesh, dm.tables)
    uq = wp.array(u_exact_fn(xq), dtype=wp.float64, device=dm.device)
    ud = wp.array(u_all.astype(np.float64), dtype=wp.float64, device=dm.device)
    out = wp.zeros(1, dtype=wp.float64, device=dm.device)
    k = make_l2_kernel(dm.tables.nbf, dm.tables.nqp, dm.dim)
    wp.launch(k, dim=len(dm.mesh.tree),
              inputs=[dm.conn, dm.h, dm.N, dm.w, ud, uq, out], device=dm.device)
    return float(np.sqrt(out.numpy()[0]))
