"""Unified linear-solve dispatch for the steppers (M1b GPU-residency work).

Backends:
  "splu"   — scipy SuperLU on the host (prototype default; unbeatable small,
             pays factorization on EVERY new matrix).
  "fused"  — the single-sync device Krylov (solvers/krylov_dev): BiCGStab for
             nonsymmetric systems, CG for SPD; Jacobi preconditioned; the
             iterations live on the GPU, one scalar readback per
             check_every iterations (m1a finding 3 / P2 measurements).
  "gpu_cg" — single-GPU resident PCG via dist_cg.pcg + SerialComm.  SPD
             systems only (sym=True required; raises ValueError otherwise).
             Builds a torch.sparse_csr_tensor on `device`, runs Jacobi-
             preconditioned CG, returns a host numpy array.  First step
             toward the 100M PPE path on GPU (the direct-solver cuDSS wall
             blocks the 100M scale; iterative CG on the SPD Laplacian is
             the escape route).  No factorization; no matrix caching.
  "amgx"  — NVIDIA AMGX (algebraic multigrid) through pyamgx, when built:
            AMG-preconditioned Krylov — flattens the O(h^-1) Jacobi
            iteration growth (P2 Explore (b)). Falls back with a clear
            error if pyamgx is unavailable.
  "cudss" — NVIDIA cuDSS (GPU direct sparse solver) via the official
            nvmath-python package: factorization AND triangular solves on
            the device — the GPU analogue of splu, right fit for the
            per-step-new-matrix stepper pattern (spec S5.4's "cuDSS
            analogue" made literal).

The dispatch is deliberately per-solve and stateless except for optional
operator caching by `cache` (a dict the caller owns): constant matrices
(mass, PPE stiffness) upload/factorize once.
"""
import numpy as np

from ..device import default_device
from ..errors import BackendError, ConvergenceError

_CUDSS_OPTS = ...          # lazily built by cudss_options()

# Iteration-count sentinel written by backends that track iteration counts and
# read by the return_result wrapper (allows cacheless callers to get iters).
# Single-element list so it is mutable from nested call frames.
_LAST_ITERS = [None]

# Inner-stats sentinel written by fgmres_pcd after each solve and read by the
# return_result wrapper.  Schema: {"F": {...}, "Ap": {...}, "Mp": {...}} where
# each dict has keys applies/iters_total/cap_hits/max_exit_relres.
# None when the last solve was not fgmres_pcd or stats were not collected.
_LAST_INNER_STATS = [None]



def cudss_options():
    """DirectSolverOptions with multithreaded host planning
    (libcudss_mtlayer_gomp) when the layer ships with nvmath — measured
    NECESSARY for per-step refactorization loops (M3 bunny v2: a
    single-threaded plan() was a 3.5 h mostly-idle stall). Shared by
    solve_linear and the device-resident stepper/film solve paths."""
    global _CUDSS_OPTS
    if _CUDSS_OPTS is ...:
        from nvmath.sparse.advanced import DirectSolverOptions
        import glob as _glob
        import os as _os
        try:
            import nvidia
            _roots = list(nvidia.__path__)
        except ImportError:
            _roots = []
        mt = []
        for _r in _roots:
            mt = _glob.glob(_os.path.join(
                _r, "cu12", "lib", "libcudss_mtlayer_gomp.so*"))
            if mt:
                break
        _CUDSS_OPTS = (DirectSolverOptions(multithreading_lib=mt[0])
                       if mt else None)
    return _CUDSS_OPTS


def _blockch_pairs(A, b, meta, tol, device):
    """blockch generalized to ndof-node-major systems carrying several
    (phi, mu) pairs (G4: the ternary 4-dof CH block (phi1,mu1,phi2,mu2)
    and the Wodo film). Each pair p = {"off", "m", "kappa"} gets the
    SAME two-factor recipe as the binary branch (design laws 1-4 apply
    unchanged; see the blockch branch of solve_linear):
        K_p  = Acm_p / m_p,   F_p = -Amc_p - kap_p K_p     (SIGNED, law 3)
        W1_p = (sqrt(sig)/sig) Acc_p + sqrt(m_p kap_p) K_p
        W2_p = W1_p + (m_p/sqrt(sig)) F_p
    with Acc/Acm/Amc/Amm extracted at dof offsets (ndof*k+off,
    ndof*k+off+1). The cross blocks (M12 transport, d12 FH coupling;
    the film's advection and top-flux rows live INSIDE the extracted
    blocks) are dropped in the PRECONDITIONER only — the outer FGMRES
    carries them. Escalation on outer stall switches the WHOLE apply to
    per-pair exact-Schur.

    B-track (M, K) extension: meta may carry "ac" = [{"off": o}, ...],
    one entry per Allen-Cahn (psi, theta) dof pair at node offsets
    (o, o+1). Each AC block is preconditioned by its OWN extracted
    diagonal blocks, lower-triangular within the pair (the theta row's
    psi column — the KWC p'(psi) torque — is carried; the psi row has
    no theta column by construction):
        z_s = Ass^{-1} r_s;  z_t = Att^{-1} (r_t - Ats z_s)
    Ass = sigma M + L_psi (f'' M + eps2 K) class (mass-dominated at
    production dt; indefinite f'' and film advection possible -> GMRES
    inner, not CG); Att = sigma M (frozen bookkeeping / marker
    advection) or the KWC (p+pf)-weighted SPD row. The phi/mu <-> psi
    couplings are dropped in the preconditioner; the outer FGMRES
    carries them. AC blocks are exact-block solves already, so the
    escalation only upgrades the CH pairs. Returns (x, (outer(+1000 on
    fallback), inner_total))."""
    from scipy.sparse.linalg import (LinearOperator, cg as _cg,
                                     gmres as _gmres, lgmres as _lgmres)
    sig = meta["sigma"]
    ndof = meta["ndof"]
    n = A.shape[0] // ndof
    base = np.arange(n) * ndof
    inner_it = [0]
    _cb = lambda *_: inner_it.__setitem__(0, inner_it[0] + 1)

    def _host_solvers(Amm, W1, W2):
        def _jacobi(W):
            d = W.diagonal().copy()
            d[d == 0] = 1.0
            return LinearOperator(W.shape, lambda v: v / d)

        MjM, Mj1, Mj2 = _jacobi(Amm), _jacobi(W1), _jacobi(W2)

        def msolve(y):
            z, info = _cg(Amm, y, M=MjM, rtol=1e-10, atol=0.0,
                          maxiter=1000, callback=_cb)
            if info != 0:
                raise ConvergenceError(f"blockch mass CG not converged: {info}")
            return z

        def w1solve(y):
            z, info = _cg(W1, y, M=Mj1, rtol=1e-8, atol=0.0,
                          maxiter=3000, callback=_cb)
            if info != 0:
                raise ConvergenceError(f"blockch W1 CG not converged: {info}")
            return z

        def w2solve(y):
            z, info = _gmres(W2, y, M=Mj2, rtol=1e-8, atol=0.0,
                             maxiter=3000, restart=100, callback=_cb,
                             callback_type="legacy")
            if info != 0:
                raise ConvergenceError(f"blockch W2 GMRES not converged: {info}")
            return z

        return msolve, w1solve, w2solve

    def _device_solvers(Amm, W1, W2):
        # fused single-sync Krylov stack; W2 indefinite -> bicgstab_dev
        from ..assembly.operators import CSROperator
        from .krylov_dev import cg_dev, bicgstab_dev
        ops = [CSROperator(X, device) for X in (Amm, W1, W2)]
        dgs = [np.asarray(X.diagonal()).copy() for X in (Amm, W1, W2)]
        for dg in dgs:
            dg[dg == 0] = 1.0
        dgs[2] = np.abs(dgs[2])         # Jacobi sign-guard (indefinite)

        def _dev(op, y, dg, rtol, krylov, label):
            if not np.any(y):
                return np.zeros_like(y)   # zero rhs: exact, and the
                # device BiCGStab 0/0-breaks down on it (frozen-theta
                # rows carry an exactly-zero residual)
            x_, info = krylov(op, y, tol=rtol, atol=1e-13,
                              maxiter=4000, diag=dg, check_every=50,
                              graph=meta.get("krylov_graph"))
            if not info.get("converged"):
                raise ConvergenceError(f"blockch {label} device solve: {info}")
            inner_it[0] += info.get("iters", 0)
            return x_

        return (lambda y: _dev(ops[0], y, dgs[0], 1e-10, cg_dev, "mass"),
                lambda y: _dev(ops[1], y, dgs[1], 1e-8, cg_dev, "W1"),
                lambda y: _dev(ops[2], y, dgs[2], 1e-8, bicgstab_dev,
                               "W2"))

    def _host_block(W, label):
        # Jacobi-GMRES on one extracted scalar block (possibly
        # indefinite / advective — GMRES, not CG; abs-diag sign guard)
        d = np.abs(W.diagonal().copy())
        d[d == 0] = 1.0
        Mj = LinearOperator(W.shape, lambda v: v / d)

        def solve(y):
            z, info = _gmres(W, y, M=Mj, rtol=1e-8, atol=0.0,
                             maxiter=3000, restart=100, callback=_cb,
                             callback_type="legacy")
            if info != 0:
                raise ConvergenceError(
                    f"blockch {label} GMRES not converged: {info}")
            return z

        return solve

    def _dev_block(W, label):
        from ..assembly.operators import CSROperator
        from .krylov_dev import bicgstab_dev
        op = CSROperator(W, device)
        dg = np.abs(np.asarray(W.diagonal()).copy())
        dg[dg == 0] = 1.0

        def solve(y):
            if not np.any(y):
                return np.zeros_like(y)   # zero rhs (see _dev above)
            x_, info = bicgstab_dev(op, y, tol=1e-8, atol=1e-13,
                                    maxiter=4000, diag=dg,
                                    check_every=50,
                                    graph=meta.get("krylov_graph"))
            if not info.get("converged"):
                raise ConvergenceError(
                    f"blockch {label} device solve: {info}")
            inner_it[0] += info.get("iters", 0)
            return x_

        return solve

    dev_inners = meta.get("inners") == "device"
    mk = _device_solvers if dev_inners else _host_solvers
    mkb = _dev_block if dev_inners else _host_block
    acs = []
    for a_ in meta.get("ac", ()):
        si = base + a_["off"]
        ti = si + 1
        Ass = A[si][:, si].tocsr()
        Att = A[ti][:, ti].tocsr()
        Ats = A[ti][:, si].tocsr()      # KWC torque col (frozen: empty)
        acs.append(dict(si=si, ti=ti, Ats=Ats,
                        ssolve=mkb(Ass, "acS"),
                        tsolve=mkb(Att, "acT")))
    pairs = []
    for p in meta["pairs"]:
        mmo, kap = p["m"], p["kappa"]
        ci = base + p["off"]
        mi = ci + 1
        Acm = A[ci][:, mi].tocsr()
        Amc = A[mi][:, ci].tocsr()
        Amm = A[mi][:, mi].tocsr()
        Acc = A[ci][:, ci].tocsr()
        K = (Acm / mmo).tocsr()
        F = (-Amc - kap * K).tocsr()        # signed curvature (law 3)
        W1 = ((np.sqrt(sig) / sig) * Acc + np.sqrt(mmo * kap) * K).tocsr()
        W2 = (W1 + (mmo / np.sqrt(sig)) * F).tocsr()
        msolve, w1solve, w2solve = mk(Amm, W1, W2)
        pairs.append(dict(ci=ci, mi=mi, m=mmo, Acm=Acm, Amc=Amc,
                          Amm=Amm, Acc=Acc, K=K, msolve=msolve,
                          w1solve=w1solve, w2solve=w2solve))

    def _apply_ac(r, z):
        for B in acs:
            zs = B["ssolve"](r[B["si"]])
            zt = B["tsolve"](r[B["ti"]] - B["Ats"] @ zs)
            z[B["si"]] = zs
            z[B["ti"]] = zt

    def apply(r):
        z = r.copy()            # identity on any dof no block covers
        for P in pairs:
            rc, rm = r[P["ci"]], r[P["mi"]]
            a = P["w1solve"](rc - P["Acm"] @ P["msolve"](rm))
            zc = P["w2solve"](P["Amm"] @ a)
            zm = P["msolve"](rm - P["Amc"] @ zc)
            z[P["ci"]] = zc
            z[P["mi"]] = zm
        _apply_ac(r, z)
        return z

    it = [0]
    x, info = _lgmres(A, b, M=LinearOperator(A.shape, apply),
                      rtol=tol, atol=1e-13, maxiter=100,
                      callback=lambda _: it.__setitem__(0, it[0] + 1))
    if info != 0:
        # ESCALATE the whole apply: per-pair exact Schur (matrix-free),
        # W1-form preconditioned — the binary escape hatch, pairwise.
        def _mk_schur(P):
            H = (-P["Amc"]).tocsr()
            Sc = LinearOperator(
                (n, n), lambda v: P["Acc"] @ v
                + P["m"] * (P["K"] @ P["msolve"](H @ v)))
            Mpre = LinearOperator(
                (n, n),
                lambda v: P["w1solve"](P["Amm"] @ P["w1solve"](v)))

            def schur(y):
                zz, sinfo = _gmres(Sc, y, M=Mpre, rtol=1e-8, atol=0.0,
                                   maxiter=800, restart=160, callback=_cb,
                                   callback_type="legacy")
                if sinfo != 0:
                    raise ConvergenceError(
                        f"blockch fallback Schur GMRES: {sinfo}")
                return zz

            return schur

        for P in pairs:
            P["schur"] = _mk_schur(P)

        def apply_fb(r):
            z = r.copy()
            for P in pairs:
                rc, rm = r[P["ci"]], r[P["mi"]]
                zc = P["schur"](rc - P["Acm"] @ P["msolve"](rm))
                zm = P["msolve"](rm - P["Amc"] @ zc)
                z[P["ci"]] = zc
                z[P["mi"]] = zm
            _apply_ac(r, z)
            return z

        it[0] = 0
        x, info = _lgmres(A, b, M=LinearOperator(A.shape, apply_fb),
                          rtol=tol, atol=1e-13, maxiter=40,
                          callback=lambda _: it.__setitem__(0, it[0] + 1))
        if info != 0:
            raise ConvergenceError(
                f"blockch fallback FGMRES not converged: {info}")
        it[0] += 1000           # mark fallback path in the iters record
    return x, (it[0], inner_it[0])


def _block_maps(indptr, indices, ndof, row_off, col_offs):
    """Gather maps for the node-blocks (row_off, c) for c in col_offs:
    positions in A.data of every entry whose dof row is ndof*k+row_off
    and dof col is ndof*j+c.  Returns (rowptr, colnodes, {c: pos});
    a block ABSENT from the pattern (block-masked kron(G, mask)
    patterns) yields an empty pos and does not participate in the
    shared-rowptr check.  All host, one row scan for all col_offs."""
    n = (len(indptr) - 1) // ndof
    nodes = np.arange(n, dtype=np.int64)
    rows = nodes * ndof + row_off
    starts = np.asarray(indptr)[rows].astype(np.int64)
    cnt = (np.asarray(indptr)[rows + 1] - starts).astype(np.int64)
    # positions of every entry in these rows (vectorized ranges)
    base = np.repeat(starts - np.concatenate(
        ([0], np.cumsum(cnt)[:-1])), cnt)
    idx = base + np.arange(int(cnt.sum()), dtype=np.int64)
    cols = np.asarray(indices)[idx].astype(np.int64)
    row_of = np.repeat(nodes, cnt)
    rowptr = colnodes = None
    pos = {}
    for col_off in col_offs:
        sel = (cols % ndof) == col_off
        pos[col_off] = idx[sel]
        if len(pos[col_off]) == 0:
            continue                    # structurally absent block
        cn = (cols[sel] // ndof).astype(np.int32)
        rp = np.zeros(n + 1, np.int64)
        np.cumsum(np.bincount(row_of[sel], minlength=n), out=rp[1:])
        if rowptr is None:
            rowptr, colnodes = rp, cn
        else:
            if not (len(cn) == len(colnodes) and (cn == colnodes).all()):
                raise BackendError("blocks do not share one node pattern")
    return rowptr, colnodes, pos


def _pair_pattern_maps(indptr, indices, ndof, off):
    """Host-once symbolic work for the DEVICE blockch setup (G5): for
    the (phi, mu) pair at dof offsets (ndof*k+off, ndof*k+off+1), gather
    maps from positions in A.data to the four pair blocks. All four
    blocks are live in every pattern (superset AND block-masked) and
    share ONE node-neighbor pattern (asserted) — W1/W2 live on it too.
    Returns (rowptr, colnodes, pos{cc,cm,mc,mm}, diag_slots), all host."""
    import scipy.sparse as _sp
    rp1, cn1, posr1 = _block_maps(indptr, indices, ndof, off,
                                  (off, off + 1))
    rp2, cn2, posr2 = _block_maps(indptr, indices, ndof, off + 1,
                                  (off, off + 1))
    if not (cn1 is not None and cn2 is not None
            and len(cn1) == len(cn2) and (cn1 == cn2).all()
            and all(len(p) == len(cn1)
                    for p in (*posr1.values(), *posr2.values()))):
        raise BackendError("pair blocks do not share one node pattern")
    rowptr, colnodes = rp1, cn1
    pos = {"cc": posr1[off], "cm": posr1[off + 1],
           "mc": posr2[off], "mm": posr2[off + 1]}
    probe = _sp.csr_matrix(
        (np.arange(len(colnodes), dtype=np.float64), colnodes, rowptr),
        shape=((len(indptr) - 1) // ndof,) * 2)
    diag = probe.diagonal().astype(np.int64)
    return rowptr, colnodes, pos, diag


def _ac_pattern_maps(indptr, indices, ndof, off):
    """AC-block analogue of _pair_pattern_maps for the (psi, theta)
    group at (off, off+1): ss = (psi, psi), tt = (theta, theta) —
    always live, shared node pattern asserted — and ts = (theta, psi),
    the KWC torque column, which a block-masked frozen-theta pattern
    legitimately DROPS (returned as None then).  Returns (rowptr,
    colnodes, {ss, ts|None, tt}, diag_slots)."""
    import scipy.sparse as _sp
    rp1, cn1, posr1 = _block_maps(indptr, indices, ndof, off, (off,))
    rp2, cn2, posr2 = _block_maps(indptr, indices, ndof, off + 1,
                                  (off, off + 1))
    if not (cn1 is not None and cn2 is not None
            and len(cn1) == len(cn2) and (cn1 == cn2).all()):
        raise BackendError("AC ss/tt blocks do not share one node pattern")
    ts = posr2[off] if len(posr2[off]) else None
    if ts is not None and len(ts) != len(cn1):
        raise BackendError("AC ts block pattern differs from ss/tt")
    pos = {"ss": posr1[off], "ts": ts, "tt": posr2[off + 1]}
    probe = _sp.csr_matrix(
        (np.arange(len(cn1), dtype=np.float64), cn1, rp1),
        shape=((len(indptr) - 1) // ndof,) * 2)
    diag = probe.diagonal().astype(np.int64)
    return rp1, cn1, pos, diag


def _blockch_apply_kernels():
    """Task #42: node-strided gather/scatter + fused axpy for the
    DEVICE-RESIDENT blockch apply.  A dof block lives at node-major
    positions base + i*ndof + off (i in [0, n)); these keep r/z on the
    device so the per-inner-solve rhs upload / solution download and the
    per-outer-matvec host round-trip collapse to ONE upload of r + ONE
    download of z per apply (the outer FGMRES check cadence)."""
    from ..assembly.operators import _kernel_cache
    import warp as wp
    key = ("blockch_apply_gs",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=False)
    def gather_stride(src: wp.array(dtype=wp.float64),
                      off: wp.int32, ndof: wp.int32,
                      out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        out[i] = src[i * ndof + off]

    @wp.kernel(module="unique", enable_backward=False)
    def scatter_stride(vals: wp.array(dtype=wp.float64),
                       off: wp.int32, ndof: wp.int32,
                       dst: wp.array(dtype=wp.float64)):
        i = wp.tid()
        dst[i * ndof + off] = vals[i]

    @wp.kernel(module="unique", enable_backward=False)
    def axpy_into(a: wp.array(dtype=wp.float64), c: wp.float64,
                  b: wp.array(dtype=wp.float64),
                  out: wp.array(dtype=wp.float64)):
        # out = a + c*b   (c=-1 gives a-b); a/b/out may alias out=a
        i = wp.tid()
        out[i] = a[i] + c * b[i]

    ks = (gather_stride, scatter_stride, axpy_into)
    _kernel_cache[key] = ks
    return ks


def _blockch_pair_fill_kernel():
    from ..assembly.operators import _kernel_cache
    import warp as wp
    key = ("blockch_pair_fill",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=False)
    def pf(Ad: wp.array(dtype=wp.float64),
           pcc: wp.array(dtype=wp.int32),
           pcm: wp.array(dtype=wp.int32),
           pmc: wp.array(dtype=wp.int32),
           pmm: wp.array(dtype=wp.int32),
           c_acc: wp.float64, c_acm: wp.float64,
           c_f_acm: wp.float64, c_w2f: wp.float64,
           amm: wp.array(dtype=wp.float64),
           acm: wp.array(dtype=wp.float64),
           amc: wp.array(dtype=wp.float64),
           w1: wp.array(dtype=wp.float64),
           w2: wp.array(dtype=wp.float64)):
        # W1 = c_acc*Acc + c_acm*Acm;  F = -Amc + c_f_acm*Acm (SIGNED);
        # W2 = W1 + c_w2f*F — one pass over the pair's node pattern
        i = wp.tid()
        vcm = Ad[pcm[i]]
        vmc = Ad[pmc[i]]
        amm[i] = Ad[pmm[i]]
        acm[i] = vcm
        amc[i] = vmc
        w1i = c_acc * Ad[pcc[i]] + c_acm * vcm
        w1[i] = w1i
        w2[i] = w1i + c_w2f * (-vmc + c_f_acm * vcm)

    _kernel_cache[key] = pf
    return pf


def _blockch_pair_fill_kernel_chunked():
    """Task #38 chunked variant of _blockch_pair_fill_kernel: A's
    values live in the block-row chunked 2-D buffer, the pos maps carry
    int64 nnz-space positions; each read locates its chunk (binary
    search over the int64 base table) — same W1/W2/F recipe."""
    from ..assembly.operators import _kernel_cache, _chunk_of
    import warp as wp
    key = ("blockch_pair_fill_ch",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=False)
    def pfc(Ad: wp.array2d(dtype=wp.float64),
            bases: wp.array(dtype=wp.int64),
            nc: wp.int32,
            pcc: wp.array(dtype=wp.int64),
            pcm: wp.array(dtype=wp.int64),
            pmc: wp.array(dtype=wp.int64),
            pmm: wp.array(dtype=wp.int64),
            c_acc: wp.float64, c_acm: wp.float64,
            c_f_acm: wp.float64, c_w2f: wp.float64,
            amm: wp.array(dtype=wp.float64),
            acm: wp.array(dtype=wp.float64),
            amc: wp.array(dtype=wp.float64),
            w1: wp.array(dtype=wp.float64),
            w2: wp.array(dtype=wp.float64)):
        i = wp.tid()
        s = pcm[i]
        c = _chunk_of(bases, nc, s)
        vcm = Ad[c, wp.int32(s - bases[c])]
        s = pmc[i]
        c = _chunk_of(bases, nc, s)
        vmc = Ad[c, wp.int32(s - bases[c])]
        s = pmm[i]
        c = _chunk_of(bases, nc, s)
        amm[i] = Ad[c, wp.int32(s - bases[c])]
        s = pcc[i]
        c = _chunk_of(bases, nc, s)
        vcc = Ad[c, wp.int32(s - bases[c])]
        acm[i] = vcm
        amc[i] = vmc
        w1i = c_acc * vcc + c_acm * vcm
        w1[i] = w1i
        w2[i] = w1i + c_w2f * (-vmc + c_f_acm * vcm)

    _kernel_cache[key] = pfc
    return pfc


def blockch_pairs_device(indptr, indices, vals_d, b, meta, tol=1e-10,
                         device=None, cache=None, cache_key=None,
                         idx_dev=None, jv=None):
    """G5: _blockch_pairs with a DEVICE-RESIDENT setup. A's values live
    on the GPU (warp array vals_d, e.g. DeviceNSAssembler.vals_d);
    indptr/indices are the assembler's HOST pattern mirrors. Symbolic
    work (pair gather maps, diag slots, full-A index upload, derived
    value buffers) is host-once per pattern, cached under
    ('blockch_dev_setup', cache_key). Per Newton iterate: one fill
    kernel per pair builds Amm/Acm/Amc/W1/W2 values on the shared node
    pattern (no host matrix ever exists), inners run on the fused
    device Krylov stack, and the outer host FGMRES sees A only through
    a device-spmv closure — 2 vector transfers per outer matvec + the
    inner-solve rhs/solution transfers are the recorded host cost of
    this rung. Same recipe/laws as _blockch_pairs; escalation gathers
    Acc lazily and runs the per-pair exact Schur via device matvecs.
    B-track: meta["ac"] AC blocks ride the same machinery — value
    GATHERS (no W combination) into per-block ss/ts/tt buffers on the
    shared node pattern, device BiCGStab inners, lower-triangular
    within the (psi, theta) pair (_blockch_pairs docstring).
    Returns host x; records ('blockch_iters', cache_key)."""
    device = default_device() if device is None else device
    import warp as wp
    from scipy.sparse.linalg import (LinearOperator, gmres as _gmres,
                                     lgmres as _lgmres)
    from ..assembly.operators import CSROperator
    from ..assembly.device_assembly import (_gather_kernel, ChunkedArray,
                                            _gather_kernel_chunked)
    from .krylov_dev import cg_dev, bicgstab_dev
    sig = meta["sigma"]
    ndof = meta["ndof"]
    N = len(indptr) - 1
    n = N // ndof
    nnz = len(indices)
    # Task #38: a ChunkedArray vals_d (block-row 2-D storage past warp's
    # 2^31-element array ceiling) switches the A-value GATHER/FILL
    # kernels and the outer full-A SpMV to their chunk-aware variants;
    # the pair/AC BLOCK machinery (node-space, < 2^31 by construction)
    # is unchanged either way.
    chunked = isinstance(vals_d, ChunkedArray)
    tbl = vals_d.table if chunked else None
    # P0-2: pos maps (positions in A.data) and the full-A row offsets
    # index nnz-space -> widen the GATHER index dtype past 2^31 (and
    # always in chunked mode: chunked kernels carry int64 positions).
    # The pair/AC BLOCK operators are node-space (always int32).  Node
    # cols (colnodes) and diag are node-space; only the A.data positions
    # (pos_d) and the outer full-A indptr widen.
    wide_pos = chunked or nnz >= 2 ** 31
    idx_np = np.int64 if wide_pos else np.int32
    idx_dt = wp.int64 if wide_pos else wp.int32
    fp = (N, nnz, len(meta["pairs"]), len(meta.get("ac", ())), chunked)
    setup = (cache or {}).get(("blockch_dev_setup", cache_key))
    if setup is None or setup["fp"] != fp:
        setup = {"fp": fp, "pairs": [], "ac": []}
        dev = lambda a_, dt: wp.array(np.ascontiguousarray(a_),
                                      dtype=dt, device=device)
        for p in meta["pairs"]:
            rowptr, colnodes, pos, diag = _pair_pattern_maps(
                indptr, indices, ndof, p["off"])
            nnzp = len(colnodes)
            setup["pairs"].append(dict(
                nnzp=nnzp,
                rowptr_d=dev(rowptr.astype(np.int32), wp.int32),
                colnodes_d=dev(colnodes, wp.int32),
                pos_d={k: dev(v.astype(idx_np), idx_dt)
                       for k, v in pos.items()},
                diag_d=dev(diag.astype(np.int32), wp.int32),
                amm_d=wp.zeros(nnzp, dtype=wp.float64, device=device),
                acm_d=wp.zeros(nnzp, dtype=wp.float64, device=device),
                amc_d=wp.zeros(nnzp, dtype=wp.float64, device=device),
                w1_d=wp.zeros(nnzp, dtype=wp.float64, device=device),
                w2_d=wp.zeros(nnzp, dtype=wp.float64, device=device),
                dg_d=wp.zeros(n, dtype=wp.float64, device=device)))
        for a_blk in meta.get("ac", ()):
            rowptr, colnodes, pos, diag = _ac_pattern_maps(
                indptr, indices, ndof, a_blk["off"])
            nnzp = len(colnodes)
            setup["ac"].append(dict(
                nnzp=nnzp,
                rowptr_d=dev(rowptr.astype(np.int32), wp.int32),
                colnodes_d=dev(colnodes, wp.int32),
                # ss = psi-psi, ts = theta-psi (KWC torque; None when
                # the pattern drops it), tt = theta-theta
                pos_d={k: (None if pos[k] is None
                           else dev(pos[k].astype(idx_np), idx_dt))
                       for k in ("ss", "ts", "tt")},
                diag_d=dev(diag.astype(np.int32), wp.int32),
                ss_d=wp.zeros(nnzp, dtype=wp.float64, device=device),
                ts_d=(wp.zeros(nnzp, dtype=wp.float64, device=device)
                      if pos["ts"] is not None else None),
                tt_d=wp.zeros(nnzp, dtype=wp.float64, device=device),
                dg_d=wp.zeros(n, dtype=wp.float64, device=device)))
        # idx_dev: caller-provided device (indptr, indices) int32 pair
        # (e.g. DeviceNSAssembler._op_idx) — avoids a DUPLICATE device
        # copy of the full-A indices (4-8 GB at the B4/B5 sizes)
        # full-A row offsets index nnz-space (widen); column indices are
        # dof-space (int32).  idx_dev, when provided, already carries the
        # assembler's chosen width (DeviceNSAssembler._op_idx).
        # Chunked (#38): the column indices CANNOT exist as one 1-D
        # >2^31-element array — they ride a ChunkedArray on vals_d's
        # chunk table (uploaded once here, cached with the setup).
        if idx_dev is not None:
            setup["A_idx_d"] = idx_dev
        elif chunked:
            ind_ch = ChunkedArray(tbl, wp.int32, device)
            ind_ch.upload(np.asarray(indices, np.int32))
            setup["A_idx_d"] = (
                wp.array(np.ascontiguousarray(
                    np.asarray(indptr).astype(np.int64)),
                    dtype=wp.int64, device=device),
                ind_ch)
        else:
            setup["A_idx_d"] = (
                wp.array(np.ascontiguousarray(
                    np.asarray(indptr).astype(idx_np)),
                    dtype=idx_dt, device=device),
                wp.array(np.ascontiguousarray(
                    np.asarray(indices).astype(np.int32)),
                    dtype=wp.int32, device=device))
        if cache is not None:
            cache[("blockch_dev_setup", cache_key)] = setup
    inner_it = [0]
    _cb = lambda *_: inner_it.__setitem__(0, inner_it[0] + 1)

    # Task #42: device-resident apply engages on CUDA whenever the fused
    # inner path is active (the same knob #40 introduced); "off" keeps
    # today's host-transfer apply bit-for-bit.  It requires no AC blocks
    # (the AC lower-triangular chain stays on the host transfer path,
    # which the film production config never exercises) and a stored
    # (non-matrix-free) outer.  The pair operators are node-space CSR
    # (always flat), so a chunked full-A vals_d is fine — the apply never
    # touches the chunked outer spmv.
    from .krylov_dev import _resolve_path
    _kg = meta.get("krylov_graph")
    _fused_on, _ = _resolve_path(_kg, device)
    # DEFAULT OFF (measured regression on the Ada box — the host<->device
    # transfers overlapped the GPU queue, so residency adds blocking
    # device copies for no sync-latency saving; see the wodo_film knob
    # docstring and the task-42 report G3).  Opt-in via meta.
    # Task #49: the device-resident OUTER FGMRES (precond_dev_outer)
    # subsumes the device-resident apply — with the whole outer solve on
    # device, the r/z copies at the apply boundary become intra-device, so
    # the outer implies the apply.  The #42 apply-only knob
    # (precond_dev_apply) stays for the host-lgmres-outer + device-apply
    # A/B; the outer knob overrides it (device apply is mandatory under a
    # device outer).  Both require the fused CUDA path, no AC blocks, and a
    # stored (non-matrix-free) outer.
    dev_outer = (_fused_on and str(device).startswith("cuda")
                 and not meta.get("ac") and jv is None
                 and meta.get("precond_dev_outer", False))
    dev_apply = dev_outer or (
        _fused_on and str(device).startswith("cuda")
        and not meta.get("ac") and jv is None
        and meta.get("precond_dev_apply", False))

    def _dev_solve(op, y, dg, rtol, krylov, label):
        if not np.any(y):
            return np.zeros_like(y)       # zero rhs (see _blockch_pairs)
        x_, info = krylov(op, y, tol=rtol, atol=1e-13, maxiter=4000,
                          diag=dg, check_every=50,
                          graph=meta.get("krylov_graph"))
        if not info.get("converged"):
            raise ConvergenceError(f"blockch {label} device solve: {info}")
        inner_it[0] += info.get("iters", 0)
        return x_

    # Task #42: OPTIONAL fixed inner-iteration budget (readback-free inner
    # solves — collapses the per-check_every scal.numpy() to nothing, so
    # the ONLY sync per apply is the single r upload + z download).  The
    # inners are a PRECONDITIONER, so a fixed budget makes the M^-1 action
    # inexact-but-deterministic and the outer FGMRES residual is the true
    # gate.  MEASURED (film config): a budget that does not fully converge
    # the mass/W1 solves weakens the preconditioner enough to trip the
    # exact-Schur escalation (10x-cost fallback), so the DEFAULT is
    # convergence-checked inners (fixed disabled) — the device residency
    # already removes the dominant per-apply wp.copy transfers.  meta
    # override "precond_fixed_iters" (>0) opts into the readback-free
    # budget where a regime tolerates it.
    _fixed = meta.get("precond_fixed_iters", 0)
    _fixed = None if not _fixed else int(_fixed)

    # Task #49: device |.|_inf for the zero-rhs guard (one scalar readback)
    _amax_scratch = {}

    def _amax_dev(y_d):
        from ..assembly.operators import _kernel_cache
        k = _kernel_cache.get(("blockch_amax",))
        if k is None:
            @wp.kernel(module="unique", enable_backward=False)
            def amax(a: wp.array(dtype=wp.float64),
                     out: wp.array(dtype=wp.float64)):
                i = wp.tid()
                wp.atomic_max(out, 0, wp.abs(a[i]))
            _kernel_cache[("blockch_amax",)] = k = amax
        s = _amax_scratch.get("s")
        if s is None:
            s = _amax_scratch["s"] = wp.zeros(1, dtype=wp.float64,
                                              device=device)
        s.zero_()
        wp.launch(k, dim=y_d.shape[0], inputs=[y_d, s], device=device)
        return float(s.numpy()[0])

    def _dev_solve_resident(op, y_d, minv_d, rtol, krylov, out_d, label):
        # device-in / device-out: NO host upload of y, NO .numpy() of the
        # solution — the whole inner solve stays on device (Task #42).
        #
        # Task #49 ZERO-RHS GUARD (the #42-review latent bug): the host
        # _dev_solve short-circuits `if not np.any(y): return zeros` — a
        # structurally-zero rhs sub-vector is exact-zero and the device
        # BiCGStab 0/0-breaks down on it (frozen rows carry an exactly-zero
        # residual).  The resident path lacked this, so on a zero block the
        # device outer would DIVERGE in inner-iteration count from host.
        # We test ||y_d||_inf on device (one scalar readback, negligible vs
        # a whole inner solve) and write an exact zero solution when it is
        # structurally zero, matching the host iterate exactly.
        if _amax_dev(y_d) == 0.0:
            out_d.zero_()
            return
        _x, info = krylov(op, None, tol=rtol, atol=1e-13, maxiter=4000,
                          check_every=50, graph=meta.get("krylov_graph"),
                          b_dev=y_d, x_out=out_d, diag_dev=minv_d,
                          fixed_iters=_fixed)
        if not info.get("converged"):
            raise ConvergenceError(f"blockch {label} device solve: {info}")
        inner_it[0] += info.get("iters", 0)

    # two gather widths: pos_d index nnz-space (P0-2 wide / #38
    # chunked); diag_d is node-space (always int32).  _gather_A is the
    # one dispatch point for A-value gathers at nnz-space positions.
    gat_diag = _gather_kernel(wp.int32)
    if chunked:
        gat_ch = _gather_kernel_chunked()
        fill_ch = _blockch_pair_fill_kernel_chunked()

        def _gather_A(pos, out, n_):
            wp.launch(gat_ch, dim=n_,
                      inputs=[vals_d.data, pos, tbl.bases_d,
                              wp.int32(tbl.nchunks), out], device=device)
    else:
        gat = _gather_kernel(idx_dt)
        fill = _blockch_pair_fill_kernel()

        def _gather_A(pos, out, n_):
            wp.launch(gat, dim=n_, inputs=[vals_d, pos, out],
                      device=device)
    pairs = []
    for p, Pd in zip(meta["pairs"], setup["pairs"]):
        mmo, kap = p["m"], p["kappa"]
        coefs = [wp.float64(np.sqrt(sig) / sig),
                 wp.float64(np.sqrt(mmo * kap) / mmo),
                 wp.float64(-kap / mmo), wp.float64(mmo / np.sqrt(sig))]
        outs = [Pd["amm_d"], Pd["acm_d"], Pd["amc_d"], Pd["w1_d"],
                Pd["w2_d"]]
        if chunked:
            wp.launch(fill_ch, dim=Pd["nnzp"], inputs=[
                vals_d.data, tbl.bases_d, wp.int32(tbl.nchunks),
                Pd["pos_d"]["cc"], Pd["pos_d"]["cm"],
                Pd["pos_d"]["mc"], Pd["pos_d"]["mm"],
                *coefs, *outs], device=device)
        else:
            wp.launch(fill, dim=Pd["nnzp"], inputs=[
                vals_d, Pd["pos_d"]["cc"], Pd["pos_d"]["cm"],
                Pd["pos_d"]["mc"], Pd["pos_d"]["mm"],
                *coefs, *outs], device=device)
        dgs = []
        minv_ds = []
        for arr in (Pd["amm_d"], Pd["w1_d"], Pd["w2_d"]):
            wp.launch(gat_diag, dim=n,
                      inputs=[arr, Pd["diag_d"], Pd["dg_d"]],
                      device=device)
            dg = Pd["dg_d"].numpy()
            dg[dg == 0] = 1.0
            dgs.append(dg)
        dgs[2] = np.abs(dgs[2])         # Jacobi sign-guard (indefinite)
        if dev_apply:
            # device-resident: 1/diag lives on device (uploaded ONCE per
            # Newton fill, not per apply).  The host dg above is still
            # computed for the zero-fix / sign-guard (a one-time setup
            # readback, negligible vs the per-apply transfers removed).
            for dg in dgs:
                minv_ds.append(wp.array(
                    np.ascontiguousarray(1.0 / dg, np.float64),
                    dtype=wp.float64, device=device))
        mkop = lambda a_, Pd=Pd: CSROperator.from_device_arrays(
            Pd["rowptr_d"], Pd["colnodes_d"], a_, n, device)
        opM = mkop(Pd["amm_d"])
        opW1, opW2 = mkop(Pd["w1_d"]), mkop(Pd["w2_d"])
        base = np.arange(n, dtype=np.int64) * ndof + p["off"]
        pd = dict(
            ci=base, mi=base + 1, m=mmo, Pd=Pd, off=int(p["off"]),
            opAcm=mkop(Pd["acm_d"]), opAmc=mkop(Pd["amc_d"]), opM=opM,
            opW1=opW1, opW2=opW2,
            msolve=lambda y, o=opM, d_=dgs[0]:
                _dev_solve(o, y, d_, 1e-10, cg_dev, "mass"),
            w1solve=lambda y, o=opW1, d_=dgs[1]:
                _dev_solve(o, y, d_, 1e-8, cg_dev, "W1"),
            w2solve=lambda y, o=opW2, d_=dgs[2]:
                _dev_solve(o, y, d_, 1e-8, bicgstab_dev, "W2"))
        if dev_apply:
            pd["minv_M"], pd["minv_W1"], pd["minv_W2"] = minv_ds
        pairs.append(pd)

    acs = []
    for a_blk, Bd in zip(meta.get("ac", ()), setup["ac"]):
        for name, buf in (("ss", Bd["ss_d"]), ("ts", Bd["ts_d"]),
                          ("tt", Bd["tt_d"])):
            if buf is None:
                continue                # masked-out KWC torque block
            _gather_A(Bd["pos_d"][name], buf, Bd["nnzp"])
        dgs = []
        for arr in (Bd["ss_d"], Bd["tt_d"]):
            wp.launch(gat_diag, dim=n, inputs=[arr, Bd["diag_d"],
                                               Bd["dg_d"]], device=device)
            dg = np.abs(Bd["dg_d"].numpy())
            dg[dg == 0] = 1.0
            dgs.append(dg)
        mkop = lambda a_, Bd=Bd: CSROperator.from_device_arrays(
            Bd["rowptr_d"], Bd["colnodes_d"], a_, n, device)
        si = np.arange(n, dtype=np.int64) * ndof + a_blk["off"]
        acs.append(dict(
            si=si, ti=si + 1,
            opTS=(mkop(Bd["ts_d"]) if Bd["ts_d"] is not None
                  else None),
            ssolve=lambda y, o=mkop(Bd["ss_d"]), d_=dgs[0]:
                _dev_solve(o, y, d_, 1e-8, bicgstab_dev, "acS"),
            tsolve=lambda y, o=mkop(Bd["tt_d"]), d_=dgs[1]:
                _dev_solve(o, y, d_, 1e-8, bicgstab_dev, "acT")))

    def _apply_ac(r, z):
        for B in acs:
            zs = B["ssolve"](r[B["si"]])
            rt = r[B["ti"]]
            if B["opTS"] is not None:
                rt = rt - B["opTS"].matvec_numpy(zs)
            zt = B["tsolve"](rt)
            z[B["si"]] = zs
            z[B["ti"]] = zt

    def apply(r):
        z = r.copy()
        for P in pairs:
            rc, rm = r[P["ci"]], r[P["mi"]]
            a = P["w1solve"](rc - P["opAcm"].matvec_numpy(
                P["msolve"](rm)))
            zc = P["w2solve"](P["opM"].matvec_numpy(a))
            zm = P["msolve"](rm - P["opAmc"].matvec_numpy(zc))
            z[P["ci"]] = zc
            z[P["mi"]] = zm
        _apply_ac(r, z)
        return z

    # ---- Task #42: device-resident apply -----------------------------
    # r/z stay on device across the WHOLE preconditioner chain: r is
    # uploaded ONCE per apply, z downloaded ONCE.  Node-strided gather/
    # scatter slice the (phi, mu) blocks in place; the inner solves run
    # device-in/device-out (no per-solve rhs upload or .numpy()); the
    # opAcm/opM/opAmc matvecs write into device scratch (no matvec_numpy
    # round-trip).  Same arithmetic and iterate sequence as apply() —
    # the inner solves still converge to their own tolerances, so the
    # outer FGMRES sees an identical M^{-1} action (few-ULP; G1).
    if dev_apply:
        gk, sk, axpy = _blockch_apply_kernels()
        # apply scratch cached with the setup (allocated once per pattern,
        # not per Newton iterate): r_d/z_d (N) + 6 node buffers per pair.
        sc = setup.get("apply_scratch")
        if sc is None:
            sc = {"r_d": wp.zeros(N, dtype=wp.float64, device=device),
                  "z_d": wp.zeros(N, dtype=wp.float64, device=device),
                  "pairs": [[wp.zeros(n, dtype=wp.float64, device=device)
                             for _ in range(6)]
                            for _ in pairs]}
            setup["apply_scratch"] = sc
        r_d, z_d = sc["r_d"], sc["z_d"]
        for P, psc in zip(pairs, sc["pairs"]):
            P["_sc"] = psc

        def _apply_resident(rin_d, zout_d):
            # device-in/device-out preconditioner apply: the WHOLE M^{-1}
            # chain on device.  Task #49's device outer calls this directly
            # (rin_d = the Arnoldi basis vector, zout_d = its preconditioned
            # image) so the r/z NEVER touch the host — the point of the task.
            wp.copy(zout_d, rin_d)        # identity on any uncovered dof
            for P in pairs:
                rc, rm, a, zc, zm, tmp = P["_sc"]
                off = P["off"]
                wp.launch(gk, dim=n, inputs=[rin_d, off, ndof, rc],
                          device=device)
                wp.launch(gk, dim=n, inputs=[rin_d, off + 1, ndof, rm],
                          device=device)
                # a = W1^{-1} (rc - Acm M^{-1} rm)
                _dev_solve_resident(P["opM"], rm, P["minv_M"], 1e-10,
                                    cg_dev, tmp, "mass")       # tmp=M^{-1}rm
                P["opAcm"].matvec(tmp, zm)                     # zm=Acm tmp
                wp.launch(axpy, dim=n, inputs=[rc, wp.float64(-1.0), zm,
                                               tmp], device=device)
                _dev_solve_resident(P["opW1"], tmp, P["minv_W1"], 1e-8,
                                    cg_dev, a, "W1")            # a=W1^{-1}..
                # zc = W2^{-1} (M a)
                P["opM"].matvec(a, tmp)                        # tmp=M a
                _dev_solve_resident(P["opW2"], tmp, P["minv_W2"], 1e-8,
                                    bicgstab_dev, zc, "W2")     # zc
                # zm = M^{-1} (rm - Amc zc)
                P["opAmc"].matvec(zc, tmp)                     # tmp=Amc zc
                wp.launch(axpy, dim=n, inputs=[rm, wp.float64(-1.0), tmp,
                                               tmp], device=device)
                _dev_solve_resident(P["opM"], tmp, P["minv_M"], 1e-10,
                                    cg_dev, zm, "mass")         # zm
                wp.launch(sk, dim=n, inputs=[zc, off, ndof, zout_d],
                          device=device)
                wp.launch(sk, dim=n, inputs=[zm, off + 1, ndof, zout_d],
                          device=device)

        def apply_dev(r):
            wp.copy(r_d, wp.array(np.ascontiguousarray(r, np.float64),
                                  dtype=wp.float64, device=device))
            _apply_resident(r_d, z_d)
            return z_d.numpy()            # ONE download per apply

        apply = apply_dev

    opA = CSROperator.from_device_arrays(*setup["A_idx_d"], vals_d, N,
                                         device)
    # MATRIX-FREE OUTER (2026-07-14): when jv is supplied the outer
    # FGMRES sees A ONLY through the caller's matrix-free J @ v closure
    # (the device element-block apply — no monolithic CSR); the blockch
    # W-factor inners still ride the stored pair-block gathers (the
    # "inners stored, outer matrix-free" design).  Without jv the stored
    # full-A device spmv is used (the B2 path, unchanged).
    A_matvec = jv if jv is not None else opA.matvec_numpy
    it = [0]

    # ---- Task #49: device-resident OUTER FGMRES ----------------------
    # The lever #40/#42 converged on: replace the host scipy lgmres outer
    # with an on-device flexible GMRES so the ENTIRE preconditioned solve
    # is device-resident — the outer Krylov vecops, the preconditioner
    # apply (_apply_resident), and the matvec (opA.matvec) all stay on the
    # GPU.  The r/z host<->device copies at the apply boundaries become
    # intra-device (free); the only host contact is the periodic residual
    # readback for the convergence test.  On non-convergence the SAME
    # exact-Schur escalation runs (host lgmres over apply_fb, below), so a
    # degraded preconditioner is caught identically.  The device FGMRES is
    # right-preconditioned flexible GMRES(restart=30) — the flexible
    # analogue of scipy lgmres with the LGMRES augmentation off (see
    # fgmres_dev docstring); the iterate matches host to few-ULP.
    if dev_outer:
        from .fgmres_dev import fgmres_dev
        b_dev = wp.array(np.ascontiguousarray(b, np.float64),
                         dtype=wp.float64, device=device)
        x_dev, finfo = fgmres_dev(
            opA.matvec, b_dev, _apply_resident, N, device,
            tol=tol, atol=1e-13, restart=30, maxiter=100)
        it[0] = finfo["outer"]
        if finfo["converged"]:
            if cache is not None:
                cache[("blockch_iters", cache_key)] = (it[0], inner_it[0])
            return x_dev.numpy()
        # not converged on the device outer: fall through to the host
        # exact-Schur escalation (identical to the lgmres info!=0 branch),
        # which the two-factor form's escape hatch handles.
        info = -1
    else:
        x, info = _lgmres(LinearOperator((N, N), A_matvec), b,
                          M=LinearOperator((N, N), apply),
                          rtol=tol, atol=1e-13, maxiter=100,
                          callback=lambda _: it.__setitem__(0, it[0] + 1))
    if info != 0:
        # ESCALATE: per-pair exact Schur; Acc gathered lazily on device
        def _mk_schur(P):
            Pd = P["Pd"]
            acc_d = wp.zeros(Pd["nnzp"], dtype=wp.float64, device=device)
            _gather_A(Pd["pos_d"]["cc"], acc_d, Pd["nnzp"])
            opAcc = CSROperator.from_device_arrays(
                Pd["rowptr_d"], Pd["colnodes_d"], acc_d, n, device)
            Sc = LinearOperator(
                (n, n), lambda v: opAcc.matvec_numpy(v)
                + P["opAcm"].matvec_numpy(
                    P["msolve"](-P["opAmc"].matvec_numpy(v))))
            Mpre = LinearOperator(
                (n, n), lambda v: P["w1solve"](
                    P["opM"].matvec_numpy(P["w1solve"](v))))

            def schur(y):
                zz, sinfo = _gmres(Sc, y, M=Mpre, rtol=1e-8, atol=0.0,
                                   maxiter=800, restart=160, callback=_cb,
                                   callback_type="legacy")
                if sinfo != 0:
                    raise ConvergenceError(
                        f"blockch fallback Schur GMRES: {sinfo}")
                return zz

            return schur

        for P in pairs:
            P["schur"] = _mk_schur(P)

        def apply_fb(r):
            z = r.copy()
            for P in pairs:
                rc, rm = r[P["ci"]], r[P["mi"]]
                zc = P["schur"](rc - P["opAcm"].matvec_numpy(
                    P["msolve"](rm)))
                zm = P["msolve"](rm - P["opAmc"].matvec_numpy(zc))
                z[P["ci"]] = zc
                z[P["mi"]] = zm
            _apply_ac(r, z)
            return z

        it[0] = 0
        x, info = _lgmres(LinearOperator((N, N), A_matvec), b,
                          M=LinearOperator((N, N), apply_fb),
                          rtol=tol, atol=1e-13, maxiter=40,
                          callback=lambda _:
                          it.__setitem__(0, it[0] + 1))
        if info != 0:
            raise ConvergenceError(
                f"blockch fallback FGMRES not converged: {info}")
        it[0] += 1000
    if cache is not None:
        cache[("blockch_iters", cache_key)] = (it[0], inner_it[0])
    return x


def solve_linear(A, b, solver="splu", sym=False, tol=1e-10, maxiter=40000,
                 device=None, cache=None, cache_key=None,
                 return_result=False):
    """Solve A x = b (scipy CSR A, host b). Returns host x.

    sym=True routes to CG/SPD paths. cache/cache_key: reuse device uploads
    or factorizations for constant matrices across steps.

    return_result=True wraps the solution in a
    :class:`diffsim.solvers.result.LinearSolveResult` (converged/iterations/
    backend telemetry) instead of returning a bare array — non-invasive opt-in
    (critical-eval P2.1).  A returned result always has ``converged=True``: the
    iterative/direct backends raise :class:`~diffsim.errors.ConvergenceError`
    on failure rather than returning an unconverged vector."""
    device = default_device() if device is None else device
    if return_result:
        from .result import LinearSolveResult
        _LAST_ITERS[0] = None          # cleared before every call
        _LAST_INNER_STATS[0] = None    # cleared before every call
        x = solve_linear(A, b, solver=solver, sym=sym, tol=tol,
                         maxiter=maxiter, device=device, cache=cache,
                         cache_key=cache_key)
        iters = _LAST_ITERS[0]         # written by backends that track iters
        if iters is None and cache is not None and cache_key is not None:
            rec = cache.get(("blockch_iters", cache_key))
            if rec is not None:
                iters = rec[0]
        inner_stats = _LAST_INNER_STATS[0]   # written by fgmres_pcd
        return LinearSolveResult(x=x, converged=True, iterations=iters,
                                 backend=solver, reason="converged",
                                 inner_stats=inner_stats)
    # W2c: device-resident CSR handoff.  A DeviceSaddleCSR keeps the assembled
    # values on device; the saddle iterative backends (fgmres_bdiag /
    # fused_bdiag) consume them via a Warp SpMV + device-gathered diagonal, so
    # we must NOT call A.tocsr() (the 19 GB pull we are eliminating).  Any other
    # solver still gets the host CSR through .tocsr() (transparent fallback).
    from ..assembly.device_assembly import DeviceSaddleCSR
    _dev_handoff = A if isinstance(A, DeviceSaddleCSR) else None
    if _dev_handoff is None:
        A = A.tocsr()
    elif solver not in ("fgmres_bdiag", "fused_bdiag"):
        A = A.tocsr()          # unsupported solver: fall back to host pull
    if cache is not None and cache_key is not None \
            and solver not in ("blockch", "blockamgx",
                               "fgmres_bdiag", "fgmres_pcd", "fused_bdiag"):
        # cheap staleness guard (evaluation solver-review item): cached
        # factorizations are for CONSTANT matrices — catch reuse of a key
        # after the matrix changed shape/pattern (values are the caller's
        # contract; a full value check would defeat the cache's purpose).
        # blockch is EXEMPT: it caches meta/iteration records only and
        # rebuilds its factors per call — the host T^T K T pattern
        # legitimately flaps under multiphase noise (the S2 finding).
        # fgmres_bdiag / fgmres_pcd are EXEMPT: the cache carries only
        # preconditioner META (pcd_meta, ndof) assembled from the mesh
        # once per BDF order — NOT matrix factorizations; the A changes
        # every step (Picard convection) and that is expected.
        fp = (A.shape, A.nnz, str(A.dtype))
        old = cache.get(("fingerprint", cache_key))
        if old is None:
            cache[("fingerprint", cache_key)] = fp
        elif old != fp:
            from ..errors import ConfigError
            raise ConfigError(
                f"solve_linear cache_key={cache_key!r} reused with a "
                f"different matrix (was {old}, now {fp}) — cached "
                f"factorizations are for constant matrices")
    if solver == "splu":
        from scipy.sparse.linalg import splu
        if cache is not None and cache_key is not None:
            lu = cache.get(("splu", cache_key))
            if lu is None:
                lu = splu(A.tocsc())
                cache[("splu", cache_key)] = lu
            return lu.solve(b)
        return splu(A.tocsc()).solve(b)

    if solver == "fused":
        from ..assembly.operators import CSROperator
        from .krylov_dev import cg_dev, bicgstab_dev
        if cache is not None and cache_key is not None:
            op = cache.get(("fusedop", cache_key))
            if op is None:
                op = CSROperator(A, device)
                cache[("fusedop", cache_key)] = op
        else:
            op = CSROperator(A, device)
        diag = np.asarray(A.diagonal())
        krylov = cg_dev if sym else bicgstab_dev
        x, info = krylov(op, b, tol=tol, atol=1e-13, maxiter=maxiter,
                         diag=diag, check_every=100)
        if not info.get("converged"):
            raise ConvergenceError(f"fused solve failed: {info}")
        return x

    if solver == "fused_bdiag":
        # W5a: block-diagonal preconditioned BiCGStab for the (u, p) saddle.
        # CSROperator + make_bdiag_apply feed bicgstab_dev via the apply_dev
        # hook (general right-preconditioner, legacy loop — no Krylov basis).
        # ndof recovered from blocktri_meta (same convention as fgmres_bdiag).
        from ..assembly.operators import CSROperator
        from .krylov_dev import bicgstab_dev
        from .saddle_precond import make_bdiag_apply

        meta = (cache or {}).get(("blocktri_meta", cache_key), {})
        ndof = meta.get("ndof", 3)

        # T4b equilibration knob (same semantics as fgmres_bdiag).
        _equilibrate = bool(meta.get("saddle_equilibrate", False))

        _bdiag_block = meta.get("bdiag_block", "scalar")
        _diag_host = None
        if _dev_handoff is not None and _bdiag_block == "scalar":
            # W2c device-resident fast path (see fgmres_bdiag for rationale).
            from .saddle_precond import make_bdiag_apply_from_diag
            op = _dev_handoff.device_operator()
            _diag_dev = _dev_handoff.diagonal_device()
            apply_bdiag = make_bdiag_apply_from_diag(_diag_dev, ndof, device)
            if _equilibrate:
                _diag_host = (_diag_dev.numpy() if hasattr(_diag_dev, "numpy")
                              else np.asarray(_diag_dev))
        else:
            if _dev_handoff is not None:
                A = _dev_handoff.tocsr()   # node-block: needs host CSR blocks
            op = CSROperator(A, device)
            # W5d knob (same convention as fgmres_bdiag): opt-in node-block Jacobi.
            apply_bdiag = make_bdiag_apply(A, ndof, device, block=_bdiag_block)
            if _equilibrate:
                _diag_host = np.asarray(A.diagonal()).copy()

        if _equilibrate:
            # T4b: run BiCGStab on the equilibrated system A_hat = D^{-1/2} A
            # D^{-1/2}.  The wrapped operator carries the scaled matvec; the
            # apply is built from the scaled diagonal (~identity) — the same
            # algebra as fgmres_bdiag.  The wrapper is not a CSROperator so
            # this takes the legacy (apply_dev) loop, which is correct here.
            import warp as wp
            from .saddle_precond import make_equilibrated_solve
            N = op.n_free
            _mvh, _bhat, _recover, apply_bdiag = make_equilibrated_solve(
                op.matvec, b, _diag_host, N, device, ndof)

            class _EqOp:
                device = op.device
                n_free = N
                def matvec(self, x, y):
                    _mvh(x, y)
            x_hat, info = bicgstab_dev(_EqOp(), _bhat.numpy(), tol=tol,
                                       atol=1e-13, maxiter=maxiter,
                                       check_every=100, apply_dev=apply_bdiag)
            x = _recover(x_hat)
        else:
            x, info = bicgstab_dev(op, b, tol=tol, atol=1e-13, maxiter=maxiter,
                                   check_every=100, apply_dev=apply_bdiag)
        _LAST_ITERS[0] = info.get("iters")
        if not info.get("converged"):
            raise ConvergenceError(
                f"fused_bdiag: not converged after {info.get('iters')} "
                f"iterations; relres={info.get('relres', float('inf')):.3e}")
        return x

    if solver == "gpu_cg":
        # Single-GPU resident PCG via dist_cg.pcg + SerialComm.
        # Only valid for SPD systems (sym=True); the PPE Laplacian is the
        # canonical caller.  Uses torch.sparse_csr_tensor for the on-device
        # SpMV (same pattern as make_partitioned_spmv in nccl_cg_proof.py)
        # and a Jacobi (diagonal) preconditioner.  Returns a host numpy
        # array to match the existing solver contract.
        if not sym:
            raise ValueError(
                "gpu_cg requires sym=True (SPD systems only); "
                "use 'fused' for non-symmetric systems")
        import torch
        from .dist_cg import pcg, SerialComm
        torch_device = torch.device(str(device))
        # Build torch.sparse_csr_tensor from the scipy CSR matrix (A is
        # already .tocsr() from the top of solve_linear).
        crow = torch.tensor(A.indptr.astype(np.int64),
                            dtype=torch.int64, device=torch_device)
        col = torch.tensor(A.indices.astype(np.int64),
                           dtype=torch.int64, device=torch_device)
        val = torch.tensor(np.ascontiguousarray(A.data, dtype=np.float64),
                           dtype=torch.float64, device=torch_device)
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            A_t = torch.sparse_csr_tensor(crow, col, val,
                                          size=tuple(A.shape),
                                          dtype=torch.float64,
                                          device=torch_device)
        # On-device SpMV: no halo (SerialComm) so p is owned-only.
        def spmv(p):
            return torch.mv(A_t, p)
        # Jacobi preconditioner: extract diagonal once, guard zeros.
        diag_np = np.asarray(A.diagonal(), dtype=np.float64).copy()
        diag_np[diag_np == 0.0] = 1.0
        diag_t = torch.tensor(diag_np, dtype=torch.float64,
                              device=torch_device)
        def precond(r):
            return r / diag_t
        # Convert b to a device tensor.
        b_dev = torch.tensor(np.ascontiguousarray(b, dtype=np.float64),
                             dtype=torch.float64, device=torch_device)
        comm = SerialComm()
        rtol_cg = tol if tol > 0.0 else 1e-8
        maxit_cg = maxiter if maxiter > 0 else 5000
        x_dev, info = pcg(spmv, precond, b_dev, comm,
                          rtol=rtol_cg, maxit=maxit_cg)
        if not info.get("converged"):
            raise ConvergenceError(
                f"gpu_cg solve failed: iters={info.get('iters')}, "
                f"final_resid={info.get('resid_history', [None])[-1]}")
        # Return host numpy array (same contract as splu / cudss).
        if isinstance(x_dev, torch.Tensor):
            return x_dev.cpu().numpy()
        return np.asarray(x_dev)

    if solver == "amgx":
        from .amgx import amgx_solve
        return amgx_solve(A, b, sym=sym, tol=tol, maxiter=maxiter,
                          cache=cache, cache_key=cache_key)

    if solver == "blocktri":
        # task-#6 production recipe (findings 8f-i): FGMRES + block-
        # triangular preconditioner with EXACT F (cuDSS velocity block)
        # + diagC Schur — 2-3 iterations at sigma=0 where AMG diverged.
        # kwargs via cache: caller stores {"ndof": d+1} under
        # ("blocktri_meta", cache_key).
        import scipy.sparse as _sp
        from scipy.sparse.linalg import LinearOperator, lgmres
        meta = (cache or {}).get(("blocktri_meta", cache_key), {})
        ndof = meta.get("ndof", 3)
        n = A.shape[0] // ndof
        dim = ndof - 1
        u_ids = (np.arange(n)[:, None] * ndof
                 + np.arange(dim)[None, :]).ravel()
        p_ids = np.arange(n) * ndof + dim
        F = A[u_ids][:, u_ids].tocsr()
        G = A[u_ids][:, p_ids].tocsr()
        Cd = np.asarray(A[p_ids][:, p_ids].diagonal())
        Cd[Cd == 0] = 1.0
        from nvmath.sparse.advanced import DirectSolver
        slvF = DirectSolver(F, np.zeros(F.shape[0]))
        slvF.plan()
        slvF.factorize()

        def apply(r):
            r_u, r_p = r[u_ids], r[p_ids]
            z_p = r_p / np.abs(Cd)
            slvF.reset_operands(b=np.ascontiguousarray(r_u - G @ z_p))
            z_u = np.asarray(slvF.solve())
            z = np.empty_like(r)
            z[u_ids] = z_u
            z[p_ids] = z_p
            return z

        it = [0]
        x, info = lgmres(A, b, M=LinearOperator(A.shape, apply),
                         rtol=tol, atol=1e-13, maxiter=100,
                         callback=lambda _: it.__setitem__(0, it[0] + 1))
        if info != 0:
            raise ConvergenceError(f"blocktri FGMRES not converged: {info}")
        return x

    if solver == "blockch":
        # Two-factor Schur block preconditioner for the mixed CH Newton
        # system  J = [[sigma M, m K], [-(f'' M + kap K), M]]  (node-major
        # 2-dof (c, mu)), Pearson-Wathen-matching class. Schur onto c:
        #     S = sigma M + m K M^{-1} (F + kap K),   F = Int f'' N N,
        # approximated by the NONSYMMETRIC two-factor form
        #     S~ = W1 M^{-1} W2
        #     W1 = sqrt(sigma) M + sqrt(m kap) K
        #     W2 = W1 + (m/sqrt(sigma)) F        (F SIGNED, not clipped)
        # Everything is recovered from J's own constrained blocks:
        # K = J_cmu/m, F = -J_muc - kap K.
        # MEASURED DESIGN LAWS (2026-07-09, three dumped offender
        # systems: FH quench sigma=500 with f'' to 9984; poly sigma=50
        # biharmonic-dominated; poly sigma=50 Newton-transient with
        # f'' to 4100):
        # (1) the classical single-factor form (F dropped, the proven
        #     [1/2,1]-bound preconditioner) DIVERGES for the logarithmic
        #     potential even with exact mass solves;
        # (2) dropping W1's sqrt(m kap)K from the second factor DIVERGES
        #     in the biharmonic-dominated regime;
        # (3) CLIPPING F to its positive part DIVERGES on large-
        #     curvature Newton transients (200 its where signed F takes
        #     88) — carry the sign, solve W2 with GMRES not CG;
        # (4) lumped mass in the pre/back-substitution DIVERGES at large
        #     f'' (the lumping error is amplified by F): consistent-mass
        #     CG solves are load-bearing.
        # Signed-F verdict across the three offenders: 32 / 53 / 88 its
        # (1-4 typical away from pathologies). The exact-Schur fallback
        # (inner GMRES on S, W-preconditioned) is 2-3 outer its
        # everywhere at ~5-10x the cost per apply — the recorded escape
        # hatch if a regime defeats the two-factor form.
        # (The published OSC pipeline, Bergermann et al. CiCP 2023,
        # sidesteps all of this by going semi-implicit AND replacing the
        # log with a polynomial; this form keeps the fully-implicit
        # Newton on the true regularized log.)
        # Apply, given r = (r_c, r_mu):
        #     a    = W1^{-1} (r_c - J_cmu M^{-1} r_mu)
        #     z_c  = W2^{-1} (M a)
        #     z_mu = M^{-1} (r_mu - J_muc z_c)
        # W1: Jacobi-CG (SPD); W2: Jacobi-GMRES (indefinite where f''<0);
        # M: Jacobi-CG. Outer: lgmres. meta via cache:
        # {"sigma","m","kappa"} under ("blockch_meta", cache_key) —
        # refresh when sigma (dt/BDF) changes.
        from scipy.sparse.linalg import (LinearOperator, cg as _cg,
                                         gmres as _gmres,
                                         lgmres as _lgmres)
        meta = (cache or {}).get(("blockch_meta", cache_key))
        if meta is None:
            raise ValueError("blockch requires ('blockch_meta', cache_key) "
                             "= {'sigma','m','kappa'} in cache")
        if "pairs" in meta:
            # G4: multi-pair generalization (meta {'sigma','ndof','pairs'})
            # — the ternary 4-dof block and the Wodo film; the binary
            # path below is untouched.
            x, iters = _blockch_pairs(A, b, meta, tol, device)
            if cache is not None:
                cache[("blockch_iters", cache_key)] = iters
            return x
        sig, mmo, kap = meta["sigma"], meta["m"], meta["kappa"]
        n = A.shape[0] // 2
        ci = np.arange(n) * 2
        mi = ci + 1
        Acm = A[ci][:, mi].tocsr()
        Amc = A[mi][:, ci].tocsr()
        Amm = A[mi][:, mi].tocsr()          # the (constrained) mass matrix
        Acc = A[ci][:, ci].tocsr()          # sigma * mass (+ Dirichlet rows)
        K = (Acm / mmo).tocsr()
        F = (-Amc - kap * K).tocsr()        # signed curvature (law 3)
        W1 = ((np.sqrt(sig) / sig) * Acc + np.sqrt(mmo * kap) * K).tocsr()
        W2 = (W1 + (mmo / np.sqrt(sig)) * F).tocsr()
        inner_it = [0]

        def _jacobi(W):
            d = W.diagonal().copy()
            d[d == 0] = 1.0
            return LinearOperator(W.shape, lambda v: v / d)

        _cb = lambda *_: inner_it.__setitem__(0, inner_it[0] + 1)
        if meta.get("inners") == "device":
            # DEVICE inners: the fused single-sync Krylov stack. W1/M are
            # SPD -> cg_dev; W2 is indefinite where f'' < 0 -> bicgstab_dev.
            # Matrices ride to the device once per Newton solve.
            from ..assembly.operators import CSROperator
            from .krylov_dev import cg_dev, bicgstab_dev
            opM = CSROperator(Amm, device)
            opW1 = CSROperator(W1, device)
            opW2 = CSROperator(W2, device)
            dgM, dg1, dg2 = (np.asarray(X.diagonal()).copy()
                             for X in (Amm, W1, W2))
            for dg in (dgM, dg1, dg2):
                dg[dg == 0] = 1.0
            dg2 = np.abs(dg2)               # Jacobi sign-guard (indefinite)

            def _dev(op, y, dg, rtol, krylov, label):
                if not np.any(y):
                    return np.zeros_like(y)   # zero rhs (see above)
                x_, info = krylov(op, y, tol=rtol, atol=1e-13,
                                  maxiter=4000, diag=dg, check_every=50,
                                  graph=meta.get("krylov_graph"))
                if not info.get("converged"):
                    raise ConvergenceError(f"blockch {label} device solve: "
                                       f"{info}")
                inner_it[0] += info.get("iters", 0)
                return x_

            _msolve = lambda y: _dev(opM, y, dgM, 1e-10, cg_dev, "mass")
            _w1solve = lambda y: _dev(opW1, y, dg1, 1e-8, cg_dev, "W1")
            _w2solve = lambda y: _dev(opW2, y, dg2, 1e-8, bicgstab_dev,
                                      "W2")
        else:
            MjM, Mj1, Mj2 = _jacobi(Amm), _jacobi(W1), _jacobi(W2)

            def _msolve(y):
                z, info = _cg(Amm, y, M=MjM, rtol=1e-10, atol=0.0,
                              maxiter=1000, callback=_cb)
                if info != 0:
                    raise ConvergenceError(
                        f"blockch mass CG not converged: {info}")
                return z

            def _w1solve(y):
                z, info = _cg(W1, y, M=Mj1, rtol=1e-8, atol=0.0,
                              maxiter=3000, callback=_cb)
                if info != 0:
                    raise ConvergenceError(
                        f"blockch W1 CG not converged: {info}")
                return z

            def _w2solve(y):
                z, info = _gmres(W2, y, M=Mj2, rtol=1e-8, atol=0.0,
                                 maxiter=3000, restart=100, callback=_cb,
                                 callback_type="legacy")
                if info != 0:
                    raise ConvergenceError(
                        f"blockch W2 GMRES not converged: {info}")
                return z

        def apply(r):
            rc, rm = r[ci], r[mi]
            a = _w1solve(rc - Acm @ _msolve(rm))
            zc = _w2solve(Amm @ a)
            zm = _msolve(rm - Amc @ zc)
            z = np.empty_like(r)
            z[ci] = zc
            z[mi] = zm
            return z

        it = [0]
        x, info = _lgmres(A, b, M=LinearOperator(A.shape, apply),
                          rtol=tol, atol=1e-13, maxiter=100,
                          callback=lambda _: it.__setitem__(0, it[0] + 1))
        if info != 0:
            # ESCALATE to the exact-Schur fallback: inner GMRES on the
            # true S = J_cc + m K M^{-1} H (matrix-free), preconditioned
            # by the W1 square-root form — measured 2-3 outer its on
            # every offender the two-factor form has ever lost.
            H = (-Amc).tocsr()
            Sc = LinearOperator((n, n), lambda v:
                                Acc @ v + mmo * (K @ _msolve(H @ v)))

            def _schur(y):
                zz, sinfo = _gmres(Sc, y,
                                   M=LinearOperator(
                                       (n, n),
                                       lambda v: _w1solve(Amm @ _w1solve(v))),
                                   rtol=1e-8, atol=0.0, maxiter=800,
                                   restart=160, callback=_cb,
                                   callback_type="legacy")
                if sinfo != 0:
                    raise ConvergenceError(
                        f"blockch fallback Schur GMRES: {sinfo}")
                return zz

            def apply_fb(r):
                rc, rm = r[ci], r[mi]
                zc = _schur(rc - Acm @ _msolve(rm))
                zm = _msolve(rm - Amc @ zc)
                z = np.empty_like(r)
                z[ci] = zc
                z[mi] = zm
                return z

            it[0] = 0
            x, info = _lgmres(A, b, M=LinearOperator(A.shape, apply_fb),
                              rtol=tol, atol=1e-13, maxiter=40,
                              callback=lambda _:
                              it.__setitem__(0, it[0] + 1))
            if info != 0:
                raise ConvergenceError(
                    f"blockch fallback FGMRES not converged: {info}")
            it[0] += 1000        # mark fallback path in the iters record
        if cache is not None:
            cache[("blockch_iters", cache_key)] = (it[0], inner_it[0])
        return x

    if solver == "cudss":
        # constant-matrix reuse: keep the factorized DirectSolver per key
        from nvmath.sparse.advanced import DirectSolver, direct_solver
        _opts = cudss_options()
        if cache is not None and cache_key is not None:
            slv = cache.get(("cudss", cache_key))
            if slv is None:
                slv = DirectSolver(A, np.ascontiguousarray(b, np.float64),
                                   options=_opts)
                slv.plan()
                slv.factorize()
                cache[("cudss", cache_key)] = slv
            slv.reset_operands(b=np.ascontiguousarray(b, np.float64))
            return np.asarray(slv.solve())
        return np.asarray(direct_solver(
            A, np.ascontiguousarray(b, np.float64),
            options=_opts))

    if solver == "blockamgx":
        # P2-R2b.1: host-orchestrated block-preconditioned FGMRES for the
        # monolithic SBM-NS saddle. The Cahouet-Chabard Schur + AMG-on-F
        # preconditioner (block_precond.BlockAMGPreconditioner) runs AMGX
        # inner V-cycles on the GPU; the outer flexible GMRES is on the
        # host (few 10s of iterations, so the per-iter sync amortizes).
        # meta via cache: ("blockamgx_meta", cache_key) =
        #   {"n_nodes","ndof","Kp","Mp_diag","sigma","nu","dir_rows"}.
        from .block_precond import (BlockAMGPreconditioner,
                                    solve_block_preconditioned)
        meta = (cache or {}).get(("blockamgx_meta", cache_key))
        if meta is None:
            raise ValueError("blockamgx requires ('blockamgx_meta', "
                             "cache_key) = {'n_nodes','ndof','Kp',"
                             "'Mp_diag','sigma','nu','dir_rows'} in cache")
        # optional inner-solve tuning knobs (absent -> current defaults):
        #   f_iters/f_tol   -> velocity-block AMG solve strength
        #   kp_iters/kp_tol -> Schur pressure-stiffness solve strength
        #   f_cycles/kp_cycles -> persistent AMGX V-cycle counts
        #   gmres_restart/gmres_maxiter -> outer FGMRES restart/maxiter
        #   schur_mode -> Schur approximation ("cahouet_chabard" default, or
        #                 "pspg_c" = C^-1 with C the monolithic p-p block).
        _pre_kw = {}
        for _k in ("f_iters", "f_tol", "kp_iters", "kp_tol",
                   "f_cycles", "kp_cycles", "schur_mode", "f_solver"):
            if _k in meta:
                _pre_kw[_k] = meta[_k]
        # per-step rebuild: free the previous step's GPU state first (AMGX
        # Resources + cuDSS F factor) — an 80-step march must not leak 80x.
        _old = (cache or {}).get(("blockamgx_pre", cache_key))
        if _old is not None:
            _old.destroy()
        pre = BlockAMGPreconditioner(
            A, meta["n_nodes"], meta["ndof"], meta["Kp"], meta["Mp_diag"],
            meta["sigma"], meta["nu"], dir_rows=meta.get("dir_rows"),
            **_pre_kw)
        if cache is not None and cache_key is not None:
            cache[("blockamgx_pre", cache_key)] = pre
        x, iters = solve_block_preconditioned(
            A, b, pre, tol=tol,
            maxiter=meta.get("gmres_maxiter", 200),
            restart=meta.get("gmres_restart", 50))
        if cache is not None and cache_key is not None:
            cache[("blockamgx_iters", cache_key)] = (iters,)
        return x

    if solver == "fgmres_bdiag":
        # Task A1: block-diagonal (Jacobi-by-block) preconditioned FGMRES for
        # the monolithic (u, p) saddle system.  The preconditioner applies
        # independent Jacobi scaling to the velocity block and the
        # PSPG-stabilized pressure block (with a 1e-12 relative floor on |d_p|
        # so near-zero pressure pivots do not amplify noise).
        # CSR construction mirrors the "fused" backend pattern; the outer
        # Krylov is fgmres_dev (device-resident flexible GMRES, restart configurable via saddle_restart meta (default 60)).
        #
        # A3 knob A — restart: opt-in via meta["saddle_restart"] (set by the
        # driver from SADDLE_RESTART env) or the "saddle_restart" cache entry.
        # Default 60 is bit-for-bit identical to the prior hardcoded value.
        # cycles divisor always equals restart so the total-iteration cap is
        # consistent regardless of restart length.
        #
        # A3 knob B — warm-start x0: opt-in via meta["saddle_x0"] = "extrap".
        # The driver stores the last two solution vectors in the cache under
        # ("bdiag_x_prev", cache_key) and ("bdiag_x_prev2", cache_key).
        # x0 = 2*x^n - x^{n-1} (linear extrapolation); first two steps fall
        # back to x^n or zero.  Default None = cold start, byte-identical.
        from ..assembly.operators import CSROperator
        from .fgmres_dev import fgmres_dev
        from .saddle_precond import make_bdiag_apply
        import warp as wp

        # ndof: caller's contract for the monolithic saddle is ndof = dim+1.
        # We recover ndof from the blocktri_meta cache slot when available
        # (same convention used by "blocktri"), otherwise default to 3 (2-D).
        meta = (cache or {}).get(("blocktri_meta", cache_key), {})
        ndof = meta.get("ndof", 3)

        # A3 knob A: restart length (opt-in, default 60)
        _restart = int(meta.get("saddle_restart", 60))

        # Krylov restart advisor: when running on a CUDA device, estimate the
        # flexible-FGMRES V+Z storage (2·restart·8·N bytes) and warn once per
        # cache key if it exceeds 50% of free VRAM.  LOG-ONLY: never changes
        # _restart.  No-op on CPU paths (guarded by is_cuda) and safe against
        # API mismatches (free_memory query wrapped in try/except).
        _advisor_key = ("_bdiag_restart_advised", cache_key)
        if (cache is not None and cache_key is not None
                and not cache.get(_advisor_key)):
            try:
                _wd = wp.get_device(device)
                if _wd.is_cuda:
                    _N_adv = len(b)
                    _krylov_bytes = 2 * _restart * 8 * _N_adv
                    _free_vram = _wd.free_memory
                    if _free_vram > 0 and _krylov_bytes > 0.5 * _free_vram:
                        _suggested = max(
                            1, int(0.4 * _free_vram / (8 * _N_adv)))
                        print(
                            f"[saddle] RESTART-ADVISOR: restart={_restart} "
                            f"needs V+Z≈{_krylov_bytes/2**30:.2f} GiB "
                            f"(>50% of {_free_vram/2**30:.2f} GiB free VRAM "
                            f"at N={_N_adv}); "
                            f"suggested restart≤{_suggested}",
                            flush=True)
            except Exception:
                pass   # API mismatch or non-CUDA build — skip silently
            cache[_advisor_key] = True

        # T4b knob — symmetric diagonal EQUILIBRATION (opt-in via
        # meta["saddle_equilibrate"], set by the driver from SADDLE_EQUILIBRATE).
        # Solve (D^{-1/2} A D^{-1/2}) y = D^{-1/2} b, x = D^{-1/2} y with
        # D = |diag(A)| (floored).  This collapses a many-order diagonal span
        # (Cb_f Nitsche penalties on fine surrogate faces vs coarse bulk) to
        # O(1), changing the metric the unpreconditioned-residual FGMRES
        # minimizes and breaking the scalar-Jacobi relres floor.  After scaling
        # diag(A_hat) = +-1, so the bdiag apply on the scaled system is
        # ~identity — equilibration REPLACES the Jacobi preconditioner (the
        # apply is built from the scaled diagonal, which keeps the pressure
        # floor exact).  Default off = byte-identical.
        _equilibrate = bool(meta.get("saddle_equilibrate", False))

        _bdiag_block = meta.get("bdiag_block", "scalar")
        _diag_host = None                 # populated for the equilibrate path
        if _dev_handoff is not None and _bdiag_block == "scalar":
            # W2c device-resident fast path: SpMV over the resident CSR values
            # (no re-upload) + preconditioner diagonal from a device gather
            # (73 MB download vs the 19 GB values pull).  Only the scalar
            # bdiag needs the diagonal alone; node-block Jacobi needs full CSR
            # sub-blocks and takes the host fallback below.
            from .saddle_precond import make_bdiag_apply_from_diag
            op = _dev_handoff.device_operator()
            _diag_dev = _dev_handoff.diagonal_device()
            apply_dev = make_bdiag_apply_from_diag(_diag_dev, ndof, device)
            N = _dev_handoff.shape[0]
            if _equilibrate:
                _diag_host = (_diag_dev.numpy() if hasattr(_diag_dev, "numpy")
                              else np.asarray(_diag_dev))
        else:
            if _dev_handoff is not None:
                A = _dev_handoff.tocsr()   # node-block: needs host CSR blocks
            op = CSROperator(A, device)
            # W5d knob: bdiag_block="node" upgrades scalar Jacobi to per-node
            # ndof x ndof block Jacobi (opt-in via blocktri_meta; default scalar).
            apply_dev = make_bdiag_apply(A, ndof, device, block=_bdiag_block)
            N = A.shape[0]
            if _equilibrate:
                _diag_host = np.asarray(A.diagonal()).copy()

        # T4b: replace op.matvec/rhs/apply with the equilibrated forms (the
        # solution is recovered after fgmres via `recover`).
        _matvec = op.matvec
        _recover = None
        if _equilibrate:
            from .saddle_precond import make_equilibrated_solve
            _matvec, b_dev_eq, _recover, apply_dev = make_equilibrated_solve(
                op.matvec, b, _diag_host, N, device, ndof)

        b_dev = wp.array(np.ascontiguousarray(b, np.float64),
                         dtype=wp.float64, device=device)
        if _equilibrate:
            b_dev = b_dev_eq
        # cycles × restart bounds total inner iterations; divisor == restart
        cycles = min(200, max(1, maxiter // _restart))

        # A3 knob B: warm-start initial guess (opt-in via saddle_x0="extrap")
        _x0_mode = meta.get("saddle_x0")
        x0_dev = None
        if _x0_mode == "extrap" and cache is not None and cache_key is not None:
            _xn = cache.get(("bdiag_x_prev", cache_key))   # x^n (last step)
            _xn1 = cache.get(("bdiag_x_prev2", cache_key)) # x^{n-1}
            if _xn is not None and _xn1 is not None:
                # linear extrapolation: x0 = 2*x^n - x^{n-1}
                x0_np = 2.0 * _xn - _xn1
                x0_dev = wp.array(np.ascontiguousarray(x0_np, np.float64),
                                  dtype=wp.float64, device=device)
            elif _xn is not None:
                # only one prior step: use x^n as initial guess
                x0_dev = wp.array(np.ascontiguousarray(_xn, np.float64),
                                  dtype=wp.float64, device=device)
            # else: first step — cold start (x0_dev stays None)

        # T4b: under equilibration the outer solve is in the SCALED space
        # (y = D^{1/2} x); the warm-start guess must be pre-scaled the same
        # way: y0 = x0 / s  (s = D^{-1/2}).  We recover x = s .* y after.
        if _equilibrate and x0_dev is not None:
            _x0h = x0_dev.numpy() / _recover(np.ones(N))  # = x0 * D^{1/2}
            x0_dev = wp.array(np.ascontiguousarray(_x0h, np.float64),
                              dtype=wp.float64, device=device)

        x_dev, finfo = fgmres_dev(
            _matvec, b_dev, apply_dev, N, device,
            tol=tol, atol=1e-13, restart=_restart, maxiter=cycles,
            x0_dev=x0_dev)

        # min-work drift-guard (truck five-leg forensics; meta knob
        # saddle_min_work like its siblings): under a
        # loose tol + warm start, entry residuals below tol get accepted
        # with ZERO iterations, and the unsolved drift compounds across
        # steps until the march collapses (the tol=1e-3 failure mode).
        # When enabled, an iters==0 acceptance is followed by ONE polishing
        # restart cycle targeting a 4x residual reduction; its result is
        # accepted regardless of the convergence flag (it is a polish, not
        # a gate).  Default off = byte-identical.
        if (meta.get("saddle_min_work")
                and finfo.get("converged") and finfo.get("inner", 0) == 0):
            _r0 = float(finfo.get("relres", 0.0))
            if _r0 > 0.0:
                x_dev, _finfo2 = fgmres_dev(
                    _matvec, b_dev, apply_dev, N, device,
                    tol=0.25 * _r0, atol=1e-13, restart=_restart,
                    maxiter=1, x0_dev=x_dev)
                finfo = dict(finfo)
                finfo["inner"] = int(_finfo2.get("inner", 0))
                finfo["relres"] = _finfo2.get("relres", _r0)
                finfo["converged"] = True   # polish never gates

        # Per-solve miss telemetry for the driver, via the same meta side
        # channel as the accept-miss flag: True iff this solve ended
        # non-converged (the driver clears it if a fallback then converges).
        meta["last_solve_miss"] = not bool(finfo["converged"])

        if not finfo["converged"]:
            # Baskar directive (T5): solve misses during the initial
            # transient are acceptable — accept the truncated iterate, LOG
            # the achieved relres, and march on (the caller's blow-up
            # sentinel guards against drift masquerading as progress).
            # Windowing is the caller's job: it sets/clears the meta flag
            # per step.  Default (flag absent) = strict raise, byte-
            # identical.
            if meta.get("saddle_accept_miss"):
                print(f"[saddle] ACCEPT-MISS: relres={finfo['relres']:.3e} "
                      f"after {finfo['inner']} inner ({finfo['outer']} "
                      f"restarts) vs tol={tol:.1e}", flush=True)
            else:
                print(f"[saddle] BUDGET-EXHAUSTED fgmres_bdiag: "
                      f"relres={finfo['relres']:.3e} after {finfo['inner']} "
                      f"inner ({finfo['outer']} restarts) — primary budget "
                      f"exhausted, raising for caller fallback", flush=True)
                raise ConvergenceError(
                    f"fgmres_bdiag: not converged after {finfo['inner']} inner "
                    f"iterations ({finfo['outer']} restarts); "
                    f"relres={finfo['relres']:.3e}")

        # T4b: recover x = D^{-1/2} y from the scaled solution.
        if _equilibrate:
            _x_np = _recover(x_dev)
            x_dev = wp.array(np.ascontiguousarray(_x_np, np.float64),
                             dtype=wp.float64, device=device)

        # Publish iteration count to the module sentinel so the return_result
        # wrapper (which calls us without return_result=True) can surface it.
        _LAST_ITERS[0] = finfo["inner"]

        # A3 knob B: store solution for next-step warm start (no-op when
        # saddle_x0 is not "extrap" — cache keys unused in that case).
        if _x0_mode == "extrap" and cache is not None and cache_key is not None:
            _xn_cur = x_dev.numpy()
            _xn_old = cache.get(("bdiag_x_prev", cache_key))
            if _xn_old is not None:
                cache[("bdiag_x_prev2", cache_key)] = _xn_old
            cache[("bdiag_x_prev", cache_key)] = _xn_cur

        return x_dev.numpy()

    if solver == "fgmres_pcd":
        # Task A3: PCD (pressure convection-diffusion) Schur-complement
        # preconditioned FGMRES for the monolithic (u, p) saddle.  The
        # preconditioner is upper-block-triangular:
        #     z_p = S^{-1} r_p,  S^{-1} ~ sigma Ap^{-1} + nu Mp^{-1}
        #     z_u = F^{-1} (r_u - G z_p)
        # with F the velocity block extracted from A, G the pressure-gradient
        # block, and Ap/Mp the pressure-space stiffness/mass assembled ONCE per
        # mesh from dm (build_pcd_meta) and passed via the cache under
        # ("pcd_meta", cache_key) — the backend cannot see dm.  Inner solves are
        # loose Jacobi-CG (preconditioner strength; the outer FGMRES gates).
        # Mirrors the fgmres_bdiag branch (same fgmres_dev, restart, cycles,
        # iteration-sentinel plumbing).
        from ..assembly.operators import CSROperator
        from .fgmres_dev import fgmres_dev
        from .saddle_precond import make_pcd_apply
        import warp as wp

        meta = (cache or {}).get(("pcd_meta", cache_key))
        if meta is None:
            raise ValueError(
                "fgmres_pcd requires ('pcd_meta', cache_key) in cache — build "
                "it once per mesh via saddle_precond.build_pcd_meta(dm, nu, "
                "sigma) and pass cache=/cache_key=")

        # Truck-fallback memory fix: when A is a DeviceSaddleCSR (device-
        # resident handoff), reuse its resident SpMV for the outer matvec
        # instead of uploading a SECOND full device CSR (the ~19 GB duplicate
        # that OOMed the truck fallback at 77 GB committed).  The
        # preconditioner blocks (F/G/Ap/Mp extraction) still need one host
        # pull — only F is re-uploaded to device (~half the saddle nnz).
        if hasattr(A, "device_operator"):
            op = A.device_operator()          # resident CSR, no re-upload
            A_pc = A.tocsr()                  # host pull for block extraction
        else:
            op = CSROperator(A, device)
            A_pc = A
        # T1: per-block inner-solve telemetry — create the accumulator dict
        # and pass it to make_pcd_apply; after the solve, publish it to the
        # module sentinel so return_result=True can copy it to inner_stats.
        _inner_stats: dict = {
            blk: {"applies": 0, "iters_total": 0,
                  "cap_hits": 0, "max_exit_relres": 0.0}
            for blk in ("F", "Ap", "Mp")
        }
        apply_dev = make_pcd_apply(A_pc, meta, device, stats=_inner_stats)
        N = A.shape[0]

        b_dev = wp.array(np.ascontiguousarray(b, np.float64),
                         dtype=wp.float64, device=device)
        # pcd_restart (meta knob, default 60 = byte-identical): flexible
        # FGMRES stores V+Z = 2*restart*8N bytes — at 10M+ DOF alongside an
        # AMGX F-hierarchy this is the difference between fitting in HBM and
        # an AMGX "CUDA kernel launch error" (OOM in disguise; sesc Rung 1).
        _pcd_restart = int(meta.get("pcd_restart", 60))
        cycles = min(200, max(1, maxiter // _pcd_restart))
        x_dev, finfo = fgmres_dev(
            op.matvec, b_dev, apply_dev, N, device,
            tol=tol, atol=1e-13, restart=_pcd_restart, maxiter=cycles)

        if not finfo["converged"]:
            raise ConvergenceError(
                f"fgmres_pcd: not converged after {finfo['inner']} inner "
                f"iterations ({finfo['outer']} restarts); "
                f"relres={finfo['relres']:.3e}")

        _LAST_ITERS[0] = finfo["inner"]
        _LAST_INNER_STATS[0] = _inner_stats   # T1: publish for return_result
        return x_dev.numpy()

    from ..errors import ConfigError
    raise ConfigError(f"unknown solver '{solver}'")
