import numpy as np
import warp as wp
from ..assembly.femelm import FEMElm, fe_N, fe_detJxW_s
from ..assembly.operators import _kernel_cache

# Load/L2 kernels are never taped (see operators.py note); skip backward codegen.
wp.set_module_options({"enable_backward": False})


def gauss_points(mesh, tables_by_p):
    """Physical Gauss-point coords per polynomial degree bin.

    Returns ``dict[int, np.ndarray]`` mapping each degree ``pv`` to an array
    of shape ``[ne_pv * nqp_pv, dim]``.  Ordering within each bin is
    (element, quadrature-point) flat, matching the kernels' ``e * nqp + q``
    indexing.

    ``tables_by_p`` may be a single ``Tables`` object (uniform mesh back-compat,
    wrapped internally as ``{mesh.p: tables_by_p}``).

    Reference lattice uses the same x-fastest reversed-itertools idiom as
    ``basis_tables`` so the (e, q) flat ordering is consistent with the kernels.
    Physical scale: 2^{-lmax(mesh.dim)} per anchor unit.
    """
    from itertools import product as iproduct
    from ..octree import morton
    from ..mesh.basis import gauss_1d

    if not isinstance(tables_by_p, dict):
        tables_by_p = {mesh.p: tables_by_p}

    dim = mesh.dim
    lo_all = mesh.tree.anchors() * 2.0 ** (-morton.lmax(dim))
    h_all = mesh.tree.h()
    result: dict = {}
    for pv in sorted(tables_by_p):
        pts, _ = gauss_1d(pv)
        nq1 = len(pts)
        qidx = np.array(list(iproduct(*[range(nq1)] * dim)), np.int64)[:, ::-1]
        ref = pts[qidx]                        # [nqp_pv, dim]
        eids = mesh.bins[pv]
        lo = lo_all[eids]                      # [ne_pv, dim]
        h = h_all[eids]                        # [ne_pv]
        xq = lo[:, None, :] + (ref[None, :, :] + 1.0) * 0.5 * h[:, None, None]
        result[pv] = xq.reshape(-1, dim)
    return result


def make_load_kernel(nbf: int, nqp: int, dim: int = 3):
    """Assemble RHS load vector: be_a += f(x_q) N_a(x_q) dJxW.

    dim-generic: jac = (he/2)^dim via power loop (dim compile-time constant).
    """
    key = ("load", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=False)
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

    @wp.kernel(module="unique", enable_backward=False)
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


def make_l2_masked_kernel(nbf: int, nqp: int, dim: int = 3):
    """L2 error with a per-Gauss-point FP64 mask: accumulates
    ||u_h - u_exact||^2 over masked points only. SBM uses this to measure
    error on the TRUE domain Omega, not the surrogate Omega~ (the S13.3.3
    warning: Omega~ \\ Omega carries extrapolation error that is not part of
    the method's accuracy claim)."""
    key = ("l2m", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=False)
    def l2m(conn: wp.array2d(dtype=wp.int32), h: wp.array(dtype=wp.float64),
            Ntab: wp.array2d(dtype=wp.float64), wtab: wp.array(dtype=wp.float64),
            u: wp.array(dtype=wp.float64), uq_exact: wp.array(dtype=wp.float64),
            mask: wp.array(dtype=wp.float64),        # [Ne*nqp] 0/1
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
            acc += diff * diff * mask[e * nqp + q] * fe_detJxW_s(wtab, fe, jac)
        wp.atomic_add(out, 0, acc)

    _kernel_cache[key] = l2m
    return l2m


def l2_error_masked(dm, u_all: np.ndarray, u_exact_fn, mask_fn) -> float:
    """L2 error over {x : mask_fn(x)} — mask_fn maps [M, dim] -> bool [M]."""
    xq_by_bin = gauss_points(dm.mesh, dm.tables_by_p)
    ud = wp.array(u_all.astype(np.float64), dtype=wp.float64, device=dm.device)
    out = wp.zeros(1, dtype=wp.float64, device=dm.device)
    for pv, b in dm.bins.items():
        xq = xq_by_bin[pv]
        uq = wp.array(np.ascontiguousarray(u_exact_fn(xq), np.float64),
                      dtype=wp.float64, device=dm.device)
        mq = wp.array(np.ascontiguousarray(mask_fn(xq), np.float64),
                      dtype=wp.float64, device=dm.device)
        k = make_l2_masked_kernel(b["nbf"], b["nqp"], dm.dim)
        wp.launch(k, dim=len(b["eids"]),
                  inputs=[b["conn"], b["h"], b["N"], b["w"], ud, uq, mq, out],
                  device=dm.device)
    return float(np.sqrt(out.numpy()[0]))


def l2_error(dm, u_all: np.ndarray, u_exact_fn) -> float:
    """L2 error || u_h - u_exact ||, summed over all per-degree bins."""
    xq_by_bin = gauss_points(dm.mesh, dm.tables_by_p)
    ud = wp.array(u_all.astype(np.float64), dtype=wp.float64, device=dm.device)
    out = wp.zeros(1, dtype=wp.float64, device=dm.device)
    for pv, b in dm.bins.items():
        xq = xq_by_bin[pv]
        uq = wp.array(u_exact_fn(xq), dtype=wp.float64, device=dm.device)
        k = make_l2_kernel(b["nbf"], b["nqp"], dm.dim)
        wp.launch(k, dim=len(b["eids"]),
                  inputs=[b["conn"], b["h"], b["N"], b["w"], ud, uq, out],
                  device=dm.device)
    return float(np.sqrt(out.numpy()[0]))
