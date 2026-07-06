"""The Integrands brick API v1 (M1b Task 2; spec S3 — the zero-relearning
promise for Dendrite/TalyFEM users).

A physics brick subclasses CEquation and provides warp functions:

    class PoissonEquation(CEquation):
        ndof = 1

        @staticmethod
        @wp.func
        def Integrands_Ae(fe: FEMElm,
                          Ntab: wp.array2d(dtype=wp.float64),
                          dNtab: wp.array3d(dtype=wp.float64),
                          detJxW: wp.float64, dscale: wp.float64,
                          nbf: wp.int32, dim: wp.int32, ndof: wp.int32,
                          Ae: wp.array3d(dtype=wp.float64), e: wp.int32):
            for a in range(nbf):
                for b in range(nbf):
                    K = wp.float64(0.0)
                    for k in range(dim):
                        K += fe_dN_s(dNtab, fe, a, k, dscale) \
                             * fe_dN_s(dNtab, fe, b, k, dscale)
                    Ae[e, ndof * a, ndof * b] += K * detJxW

The framework owns the element loop and the Gauss-point loop (spec S3.1);
the brick is called once per integration point with the FEMElm view and the
Hughes-nomenclature accessors (fe_N, fe_dN_s, node-major Ae blocks).

v1 deltas from TalyFEM, documented per spec S3.2: (i) basis tables and
compile-time sizes are explicit arguments (warp structs cannot carry
arrays); (ii) Ae is indexed [e, ndof*a+i, ndof*b+j] with the element id
passed through. Field access (fe.value / NodeData) and Integrands4side wiring
arrive with the NS bricks (M1b Tasks 5-7). Purity contract unchanged: no
hidden state, everything through arguments => tape/Enzyme-ready.

Kernels: module="unique"; enable_backward defaults False (assembly), a brick
may set `taped = True` to opt its kernels into backward codegen (finding 4c
patterns are then MANDATORY: no loop-reassigned locals — wp.pow computes the
jacobian factor here for exactly that reason).
"""
import numpy as np
import scipy.sparse as sp
import warp as wp

from ..assembly.femelm import FEMElm, fe_N, fe_dN_s, fe_detJxW_s  # noqa: F401 (re-export for bricks)
from ..assembly.operators import _kernel_cache


class CEquation:
    """Base physics brick. Subclasses define ndof and the Integrands_* warp
    functions (staticmethod wp.func). taped=True opts kernels into
    enable_backward."""
    ndof: int = 1
    taped: bool = False

    # subclasses provide:
    #   Integrands_Ae(fe, Ntab, dNtab, detJxW, dscale, nbf, dim, ndof, Ae, e)
    #   Integrands_be(fe, Ntab, dNtab, detJxW, dscale, nbf, dim, ndof,
    #                 fq, nqp, be, e)          [optional]


def _brick_Ae_kernel(brick, nbf: int, nqp: int, dim: int):
    key = ("brick_Ae", brick, nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]
    integrands = brick.Integrands_Ae
    ndof = brick.ndof
    dim_pow = float(dim)

    @wp.kernel(module="unique", enable_backward=bool(brick.taped))
    def brick_Ae(conn: wp.array2d(dtype=wp.int32),
                 h: wp.array(dtype=wp.float64),
                 Ntab: wp.array2d(dtype=wp.float64),
                 dNtab: wp.array3d(dtype=wp.float64),
                 wtab: wp.array(dtype=wp.float64),
                 Ae: wp.array3d(dtype=wp.float64)):
        e = wp.tid()
        fe = FEMElm()
        fe.e = e
        fe.he = h[e]
        jac = wp.pow(fe.he * wp.float64(0.5), wp.float64(dim_pow))
        dscale = wp.float64(2.0) / fe.he
        for q in range(nqp):
            fe.q = q
            detJxW = fe_detJxW_s(wtab, fe, jac)
            integrands(fe, Ntab, dNtab, detJxW, dscale, nbf, dim, ndof, Ae, e)

    _kernel_cache[key] = brick_Ae
    return brick_Ae


def _brick_be_kernel(brick, nbf: int, nqp: int, dim: int):
    key = ("brick_be", brick, nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]
    integrands = brick.Integrands_be
    ndof = brick.ndof
    dim_pow = float(dim)

    @wp.kernel(module="unique", enable_backward=bool(brick.taped))
    def brick_be(conn: wp.array2d(dtype=wp.int32),
                 h: wp.array(dtype=wp.float64),
                 Ntab: wp.array2d(dtype=wp.float64),
                 dNtab: wp.array3d(dtype=wp.float64),
                 wtab: wp.array(dtype=wp.float64),
                 fq: wp.array(dtype=wp.float64),
                 be: wp.array2d(dtype=wp.float64)):
        e = wp.tid()
        fe = FEMElm()
        fe.e = e
        fe.he = h[e]
        jac = wp.pow(fe.he * wp.float64(0.5), wp.float64(dim_pow))
        dscale = wp.float64(2.0) / fe.he
        for q in range(nqp):
            fe.q = q
            detJxW = fe_detJxW_s(wtab, fe, jac)
            integrands(fe, Ntab, dNtab, detJxW, dscale, nbf, dim, ndof,
                       fq, nqp, be, e)

    _kernel_cache[key] = brick_be
    return brick_be


def assemble_brick_csr(dm, brick) -> sp.csr_matrix:
    """Constrained CSR T_vec^T K T_vec from the brick's Integrands_Ae —
    mixed-p aware (one launch per bin), node-major ndof blocks."""
    ndof = brick.ndof
    rows, cols, vals = [], [], []
    for pv, b in dm.bins.items():
        ne = len(b["eids"])
        nbf, nqp = b["nbf"], b["nqp"]
        Ae = wp.zeros((ne, nbf * ndof, nbf * ndof), dtype=wp.float64,
                      device=dm.device)
        k = _brick_Ae_kernel(brick, nbf, nqp, dm.dim)
        wp.launch(k, dim=ne, inputs=[b["conn"], b["h"], b["N"], b["dN"],
                                     b["w"], Ae], device=dm.device)
        Aeh = Ae.numpy()
        conn = dm.mesh.conn_of[pv].astype(np.int64)
        gdof = (conn[:, :, None] * ndof
                + np.arange(ndof)[None, None, :]).reshape(ne, nbf * ndof)
        rows.append(np.repeat(gdof, nbf * ndof, axis=1).ravel())
        cols.append(np.tile(gdof, (1, nbf * ndof)).ravel())
        vals.append(Aeh.ravel())
    Nn = dm.n_nodes * ndof
    K = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(Nn, Nn)).tocsr()
    T = dm.constraints.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr") \
        if ndof > 1 else T
    return (T_vec.T @ K @ T_vec).tocsr()


def brick_load_vector(dm, brick, f_fn) -> np.ndarray:
    """T_vec^T load vector from the brick's Integrands_be; f_fn evaluated at
    Gauss points per bin (scalar per (gp, dof) handled by the brick)."""
    from ..physics.poisson import gauss_points
    ndof = brick.ndof
    d = dm.device
    F_full = np.zeros(dm.n_nodes * ndof)
    xq_by_bin = gauss_points(dm.mesh, dm.tables_by_p)
    for pv, b in dm.bins.items():
        ne = len(b["eids"])
        nbf, nqp = b["nbf"], b["nqp"]
        fq = wp.array(np.ascontiguousarray(f_fn(xq_by_bin[pv]), np.float64),
                      dtype=wp.float64, device=d)
        be = wp.zeros((ne, nbf * ndof), dtype=wp.float64, device=d)
        k = _brick_be_kernel(brick, nbf, nqp, dm.dim)
        wp.launch(k, dim=ne, inputs=[b["conn"], b["h"], b["N"], b["dN"],
                                     b["w"], fq, be], device=d)
        beh = be.numpy()
        conn = dm.mesh.conn_of[pv].astype(np.int64)
        gdof = (conn[:, :, None] * ndof
                + np.arange(ndof)[None, None, :]).reshape(ne, nbf * ndof)
        np.add.at(F_full, gdof.ravel(), beh.ravel())
    T = dm.constraints.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr") \
        if ndof > 1 else T
    return np.asarray(T_vec.T @ F_full)
