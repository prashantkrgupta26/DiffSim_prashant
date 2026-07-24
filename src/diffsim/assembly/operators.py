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
    # P0-2 mixed-width: row offsets index nnz-space -> stay int64 once
    # nnz >= 2^31 (near-zero memory tax vs the values); column indices
    # only address rows/cols (< 2^31 at our scales) -> always int32.
    wide = A.nnz >= 2 ** 31
    off_np, off_wp = (np.int64, wp.int64) if wide else (np.int32, wp.int32)
    return (
        wp.array(np.ascontiguousarray(A.indptr.astype(off_np)),
                 dtype=off_wp, device=device),
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


# ---------------------------------------------------------------------
# Task #38 ChunkedCSR: warp's array_t ABI carries int32 shapes AND int32
# byte-strides, so NO single array dimension may reach 2^31 elements
# (types.py check_array_shape) — any nnz-length buffer dies at
# construction past 2.15B nnz regardless of index dtype (#33 widened the
# arithmetic, not the arrays).  Fix: block-ROW chunked storage — one 2-D
# [nchunks, cap] array whose per-chunk byte stride stays < 2^31; the
# int64 global slot is located with a small binary search over the
# chunk-base table and addressed [c, int32(slot - bases[c])].
# ---------------------------------------------------------------------
@wp.func
def _chunk_of(bases: wp.array(dtype=wp.int64), nc: wp.int32,
              slot: wp.int64) -> wp.int32:
    """Largest c in [0, nc) with bases[c] <= slot (bases ascending,
    length nc+1) — the chunk holding nnz-space position `slot`."""
    lo = wp.int32(0)
    hi = nc - wp.int32(1)
    while lo < hi:
        mid = (lo + hi + wp.int32(1)) / wp.int32(2)
        if bases[mid] <= slot:
            lo = mid
        else:
            hi = mid - wp.int32(1)
    return lo


def make_csr_spmv_chunked():
    """CSR SpMV over CHUNKED (block-row 2-D) data/column arrays: row
    offsets int64, columns int32.  Chunk boundaries are row-aligned
    (ChunkTable contract), so one chunk lookup per row suffices and the
    inner walk is int32 within-chunk."""
    key = ("csr_spmv_chunked",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def spmv_ch(
        indptr: wp.array(dtype=wp.int64),
        indices: wp.array2d(dtype=wp.int32),
        data: wp.array2d(dtype=wp.float64),
        bases: wp.array(dtype=wp.int64),
        nc: wp.int32,
        x: wp.array(dtype=wp.float64),
        y: wp.array(dtype=wp.float64),
    ):
        row = wp.tid()
        j0 = indptr[row]
        cnt = wp.int32(indptr[row + 1] - j0)
        acc = wp.float64(0.0)
        if cnt > 0:
            c = _chunk_of(bases, nc, j0)
            lj = wp.int32(j0 - bases[c])
            for k in range(cnt):
                acc += data[c, lj + k] * x[indices[c, lj + k]]
        y[row] = acc

    _kernel_cache[key] = spmv_ch
    return spmv_ch


def make_csr_spmv(idx_dtype=wp.int32):
    """CSR SpMV specialized on the OFFSET (row-pointer) dtype — the P0-2
    mixed-width path.  Narrow (int32) returns the module-level csr_spmv
    unchanged (bit-for-bit).  Wide (int64) compiles a variant whose row-
    offset loop bound is int64 so the range never wraps past 2^31; the
    COLUMN indices stay int32 (dof-space)."""
    if idx_dtype is wp.int32:
        return csr_spmv
    key = ("csr_spmv64",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique")
    def spmv64(
        indptr: wp.array(dtype=wp.int64),
        indices: wp.array(dtype=wp.int32),
        data: wp.array(dtype=wp.float64),
        x: wp.array(dtype=wp.float64),
        y: wp.array(dtype=wp.float64),
    ):
        row = wp.tid()
        acc = wp.float64(0.0)
        # warp's range() is int32-only; the row-offset bounds are int64
        # (nnz-space) -> walk them with an explicit int64 counter so the
        # index into data/indices never wraps past 2^31.
        j = indptr[row]
        end = indptr[row + 1]
        while j < end:
            acc += data[j] * x[indices[j]]
            j += wp.int64(1)
        y[row] = acc

    _kernel_cache[key] = spmv64
    return spmv64


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

    @wp.kernel(module="unique", enable_backward=False)
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

    @wp.kernel(module="unique", enable_backward=False)
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
                lapN=wp.array(np.ascontiguousarray(
                    tb.lapN.astype(np.float64)), dtype=wp.float64,
                    device=device),
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

    @wp.kernel(module="unique", enable_backward=False)
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


def make_poisson_element_matrices_var(nbf: int, nqp: int, dim: int = 3):
    """Var-kappa variant of poisson_Ke: stiffness weighted by a per-Gauss-
    point coefficient kq[e*nqp + q] (spatially-varying / field-dependent
    conductivity — the closure-hook pathway, spec S6.3). Kept separate from
    the scalar kernel so the scalar path's module hash is untouched; unify
    in the M1b matrix-free rewrite."""
    key = ("poisson_Ke_var", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=False)
    def poisson_Ke_var(h: wp.array(dtype=wp.float64),
                       dNtab: wp.array3d(dtype=wp.float64),
                       wtab: wp.array(dtype=wp.float64),
                       kq: wp.array(dtype=wp.float64),      # [Ne*nqp]
                       Ke: wp.array3d(dtype=wp.float64)):
        e = wp.tid()
        fe = FEMElm(); fe.e = e; fe.he = h[e]
        half = fe.he * wp.float64(0.5)
        jac = wp.float64(1.0)
        for _ in range(dim):
            jac = jac * half
        dscale = wp.float64(2.0) / fe.he
        for q in range(nqp):
            fe.q = q
            dJxW = fe_detJxW_s(wtab, fe, jac) * kq[e * nqp + q]
            for a in range(nbf):
                for b in range(nbf):
                    v = wp.float64(0.0)
                    for d in range(dim):
                        v = v + fe_dN_s(dNtab, fe, a, d, dscale) * fe_dN_s(dNtab, fe, b, d, dscale)
                    Ke[e, a, b] = Ke[e, a, b] + v * dJxW

    _kernel_cache[key] = poisson_Ke_var
    return poisson_Ke_var


def volume_triplets(dm, kq_by_bin=None):
    """Unconstrained COO triplets (rows, cols, vals) of the volume Poisson
    stiffness, over all per-degree bins. kappa = 1 by default; pass
    kq_by_bin (dict pv -> FP64 [ne_bin*nqp] Gauss-point coefficients) for a
    spatially-varying coefficient. Callers concatenate additional (e.g. SBM
    face) triplets before the single tocsr so shared nodes are summed once."""
    all_rows, all_cols, all_vals = [], [], []
    for pv, b in dm.bins.items():
        ne_bin = len(b["eids"])
        nbf = b["nbf"]
        nqp = b["nqp"]
        Ke = wp.zeros((ne_bin, nbf, nbf), dtype=wp.float64, device=dm.device)
        if kq_by_bin is None:
            k = make_poisson_element_matrices(nbf, nqp, dm.dim)
            wp.launch(k, dim=ne_bin, inputs=[b["h"], b["dN"], b["w"], Ke],
                      device=dm.device)
        else:
            kq = wp.array(np.ascontiguousarray(kq_by_bin[pv], np.float64),
                          dtype=wp.float64, device=dm.device)
            k = make_poisson_element_matrices_var(nbf, nqp, dm.dim)
            wp.launch(k, dim=ne_bin, inputs=[b["h"], b["dN"], b["w"], kq, Ke],
                      device=dm.device)
        Keh = Ke.numpy()
        conn = dm.mesh.conn_of[pv]                      # int32 [ne_bin, nbf]
        all_rows.append(np.repeat(conn, nbf, axis=1).ravel())
        all_cols.append(np.tile(conn, (1, nbf)).ravel())
        all_vals.append(Keh.ravel())
    return (np.concatenate(all_rows), np.concatenate(all_cols),
            np.concatenate(all_vals))


def assemble_csr(dm):
    """Assemble global constrained scalar stiffness matrix T^T K T (host).

    The host COO->CSR path: GPU element matrices Ke are pulled to host
    (.numpy()), Python builds COO triplets, and scipy runs a single-
    threaded coo_matrix(...).tocsr() + the T^T(.)T sparse triple product.
    Measured wall: ~1011 s at 3-D L8 (17M dofs) vs a 2.6 s AMGX solve.
    For the device-resident scatter path that eliminates this wall see
    ``DeviceScalarPoissonAssembler`` / ``assemble_csr_device``."""
    rows, cols, vals = volume_triplets(dm)
    K = sp.coo_matrix((vals, (rows, cols)),
                      shape=(dm.n_nodes, dm.n_nodes)).tocsr()
    T = dm.constraints.T.tocsr()
    return (T.T @ K @ T).tocsr()


class DeviceScalarPoissonAssembler:
    """Device-resident scalar (1-dof/node) Poisson stiffness assembler —
    the K_p analogue of DeviceNSAssembler, sized for the 100M PPE.

    The host ``assemble_csr`` wall is the .numpy() pull of the element
    matrices + the single-threaded scipy COO->CSR + the T^T(.)T sparse
    triple product (~1011 s at 3-D L8).  This mirrors DeviceNSAssembler's
    slot-map scatter with ndof=1:

      * ONE-TIME symbolic sparsity + slot map (host, per epoch): the
        constrained pattern of ``T^T K T`` and, per element pair, the CSR
        value-index it lands in.  For a uniform mesh (no hanging nodes) T
        is the identity, so ``T^T K T == K`` and the slot map is the plain
        element->CSR scatter; for a mesh WITH hanging nodes the element
        entries are expanded through the constraint weights host-once
        (w_r*w_c at (master_r, master_c)) exactly as DeviceNSAssembler's
        weighted path does — so ``T^T K T`` is built DIRECTLY in the free-
        dof space with NO host triple product.

      * PER-BUILD numeric fill (device): the Poisson element matrices Ke
        are computed by the existing warp kernel and scatter-atomic-added
        into the preallocated device CSR values buffer — NO host pull, NO
        COO->CSR.

    Correctness contract: the device CSR equals host ``assemble_csr(dm)``
    to fp tolerance (gated ~1e-12).  Reuse across builds (var-kappa,
    changing geometry on a fixed pattern) via ``fill()`` (device scatter
    only) + ``to_csr()`` (one download) or the zero-copy ``device_op()``.
    """

    def __init__(self, dm, index_width="auto", chunking="auto",
                 chunk_cap=None, node_pattern=None):
        from .device_assembly import DeviceNSAssembler
        self.dm = dm
        # ndof=1 scalar system.  DeviceNSAssembler already builds T^T K T
        # in the free-dof space (identity-T -> plain K; hanging-T ->
        # weighted expansion), which is EXACTLY the scalar K_p pattern.
        #
        # node_pattern (the 100M-critical choice): the DEFAULT COO symbolic
        # path still builds a HOST COO + the K2[rr,cc] fancy-index slot map
        # — the SAME single-threaded host wall.  The node-graph pattern
        # instead builds indptr/indices in CLOSED FORM on device and
        # computes element slots IN-KERNEL, so NO host COO exists.  It
        # requires identity constraints (uniform meshes — exactly the PPE
        # scaling case), so we auto-select it there and fall back to the
        # COO path (still device SCATTER, host symbolic) for hanging-node
        # meshes.  node_pattern=None => auto by identity-T; True/False
        # force.  Node pattern uses conn-order columns (unsorted), so
        # to_csr() sorts before returning.
        T = dm.constraints.T.tocsr()
        identity_T = (T.shape[0] == T.shape[1]) and (
            T != sp.identity(T.shape[0], format="csr")).nnz == 0
        if node_pattern is None:
            node_pattern = identity_T
        self._node_pattern = bool(node_pattern)
        self._asm = DeviceNSAssembler(
            dm, ndof=1, coloring=False, index_width=index_width,
            chunking=chunking, chunk_cap=chunk_cap,
            node_pattern=node_pattern)
        self.nnz = self._asm.nnz
        self.Nfull = self._asm.Nfull

    def fill(self, kq_by_bin=None):
        """Device numeric fill: compute the Poisson element matrices Ke
        and scatter them into the device CSR values (no host round-trip).
        kq_by_bin (dict pv -> FP64 [ne*nqp]) opts a spatially-varying
        coefficient (the var-kappa kernel); None => kappa=1."""
        dm = self._asm.dm
        self._asm.zero_fill()
        for k_bin, (pv, b, ne, nbf, _g) in enumerate(self._asm._bins):
            nqp = b["nqp"]
            Ke = wp.zeros((ne, nbf, nbf), dtype=wp.float64, device=dm.device)
            if kq_by_bin is None:
                kK = make_poisson_element_matrices(nbf, nqp, dm.dim)
                wp.launch(kK, dim=ne,
                          inputs=[b["h"], b["dN"], b["w"], Ke],
                          device=dm.device)
            else:
                kq = wp.array(np.ascontiguousarray(kq_by_bin[pv], np.float64),
                              dtype=wp.float64, device=dm.device)
                kK = make_poisson_element_matrices_var(nbf, nqp, dm.dim)
                wp.launch(kK, dim=ne,
                          inputs=[b["h"], b["dN"], b["w"], kq, Ke],
                          device=dm.device)
            # scalar has no rhs; feed a zero be so the shared scatter_bin
            # (which also scatters be into F_d) is a harmless no-op there.
            be = wp.zeros((ne, nbf), dtype=wp.float64, device=dm.device)
            self._asm.scatter_bin(k_bin, Ke, be)
        return self

    def to_csr(self):
        """Download the current device CSR to a scipy csr_matrix (the
        single host pull — replaces the whole host COO->CSR + T^T(.)T
        wall with one values copy).  The node-graph pattern stores columns
        in conn (node) order within a row, so the CSR is sorted before
        return to match the canonical host layout (cheap vs assembly)."""
        a = self._asm
        vals = a.vals_d.numpy()
        K = sp.csr_matrix((vals, a.indices, a.indptr),
                          shape=(a.Nfull, a.Nfull))
        if self._node_pattern:
            K.sort_indices()
        return K

    def device_op(self):
        """Zero-copy CSROperator-protocol view over the device CSR (no
        host round-trip) — for a device-resident solve."""
        return self._asm.device_operator()


def assemble_csr_device(dm, kq_by_bin=None, index_width="auto",
                        chunking="auto"):
    """Device-resident equivalent of ``assemble_csr(dm)`` (host COO->CSR
    eliminated).  Returns a scipy csr_matrix equal to ``assemble_csr(dm)``
    to fp tolerance.  For repeated fills on a fixed mesh build a
    ``DeviceScalarPoissonAssembler`` once and reuse it."""
    asm = DeviceScalarPoissonAssembler(dm, index_width=index_width,
                                       chunking=chunking)
    asm.fill(kq_by_bin)
    return asm.to_csr()


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
        # P0-2: pick the SpMV whose offset dtype matches the indptr.  A
        # host scipy CSR here indexes nnz-space through indptr; if that
        # exceeds int32 (>2^31 nnz) _csr_to_device keeps int64 offsets
        # and this selects the wide kernel.  <2^31 -> narrow (unchanged).
        self._spmv = make_csr_spmv(self._dev[0].dtype)

    @classmethod
    def from_device_arrays(cls, indptr_d, indices_d, data_d, n, device):
        """Wrap an ALREADY-DEVICE CSR (wp.int32|int64 offsets, int32
        columns, float64 data) without any host round-trip — the G5
        device-resident blockch setup consumes assembler-owned value
        buffers directly.  The SpMV specializes on the offset dtype so a
        wide (int64-offset) full-A operator never wraps.

        Task #38: indices_d/data_d may be ChunkedArray (block-row 2-D
        past warp's 2^31-element array ceiling) — the chunked SpMV then
        rides the shared chunk table; matvec() is unchanged because the
        _dev tuple mirrors the chunked kernel's argument order."""
        op = cls.__new__(cls)
        op.device = device
        op.n_free = n
        if not isinstance(data_d, wp.array):        # ChunkedArray pair
            t = data_d.table
            op._dev = (indptr_d, indices_d.data, data_d.data,
                       t.bases_d, wp.int32(t.nchunks))
            op._spmv = make_csr_spmv_chunked()
        else:
            op._dev = (indptr_d, indices_d, data_d)
            op._spmv = make_csr_spmv(indptr_d.dtype)
        return op

    def matvec(self, x: wp.array, y: wp.array):
        wp.launch(self._spmv, dim=self.n_free,
                  inputs=[*self._dev, x, y], device=self.device)

    def matvec_numpy(self, x: np.ndarray) -> np.ndarray:
        xd = wp.array(np.ascontiguousarray(x, np.float64),
                      dtype=wp.float64, device=self.device)
        yd = wp.zeros(self.n_free, dtype=wp.float64, device=self.device)
        self.matvec(xd, yd)
        return yd.numpy()


def operator_diagonal(dm) -> np.ndarray:
    return np.asarray(assemble_csr(dm).diagonal())
