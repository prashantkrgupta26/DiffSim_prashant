"""Unified linear-solve dispatch for the steppers (M1b GPU-residency work).

Backends:
  "splu"  — scipy SuperLU on the host (prototype default; unbeatable small,
            pays factorization on EVERY new matrix).
  "fused" — the single-sync device Krylov (solvers/krylov_dev): BiCGStab for
            nonsymmetric systems, CG for SPD; Jacobi preconditioned; the
            iterations live on the GPU, one scalar readback per
            check_every iterations (m1a finding 3 / P2 measurements).
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

from ..errors import BackendError, ConvergenceError

_CUDSS_OPTS = ...          # lazily built by cudss_options()


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
                              maxiter=4000, diag=dg, check_every=50)
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
                                    check_every=50)
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


def blockch_pairs_device(indptr, indices, vals_d, b, meta, tol=1e-10,
                         device="cuda:0", cache=None, cache_key=None,
                         idx_dev=None):
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
    import warp as wp
    from scipy.sparse.linalg import (LinearOperator, gmres as _gmres,
                                     lgmres as _lgmres)
    from ..assembly.operators import CSROperator
    from ..assembly.device_assembly import _gather_kernel
    from .krylov_dev import cg_dev, bicgstab_dev
    sig = meta["sigma"]
    ndof = meta["ndof"]
    N = len(indptr) - 1
    n = N // ndof
    nnz = len(indices)
    if nnz >= 2 ** 31:
        raise BackendError(
            f"blockch device nnz {nnz} >= 2^31: int32 device slot maps "
            f"overflow (add an int64 variant)")
    fp = (N, nnz, len(meta["pairs"]), len(meta.get("ac", ())))
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
                pos_d={k: dev(v.astype(np.int32), wp.int32)
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
                           else dev(pos[k].astype(np.int32), wp.int32))
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
        setup["A_idx_d"] = idx_dev if idx_dev is not None else (
            wp.array(np.ascontiguousarray(
                np.asarray(indptr).astype(np.int32)),
                dtype=wp.int32, device=device),
            wp.array(np.ascontiguousarray(
                np.asarray(indices).astype(np.int32)),
                dtype=wp.int32, device=device))
        if cache is not None:
            cache[("blockch_dev_setup", cache_key)] = setup
    inner_it = [0]
    _cb = lambda *_: inner_it.__setitem__(0, inner_it[0] + 1)

    def _dev_solve(op, y, dg, rtol, krylov, label):
        if not np.any(y):
            return np.zeros_like(y)       # zero rhs (see _blockch_pairs)
        x_, info = krylov(op, y, tol=rtol, atol=1e-13, maxiter=4000,
                          diag=dg, check_every=50)
        if not info.get("converged"):
            raise ConvergenceError(f"blockch {label} device solve: {info}")
        inner_it[0] += info.get("iters", 0)
        return x_

    gat = _gather_kernel()
    fill = _blockch_pair_fill_kernel()
    pairs = []
    for p, Pd in zip(meta["pairs"], setup["pairs"]):
        mmo, kap = p["m"], p["kappa"]
        wp.launch(fill, dim=Pd["nnzp"], inputs=[
            vals_d, Pd["pos_d"]["cc"], Pd["pos_d"]["cm"],
            Pd["pos_d"]["mc"], Pd["pos_d"]["mm"],
            wp.float64(np.sqrt(sig) / sig),
            wp.float64(np.sqrt(mmo * kap) / mmo),
            wp.float64(-kap / mmo), wp.float64(mmo / np.sqrt(sig)),
            Pd["amm_d"], Pd["acm_d"], Pd["amc_d"], Pd["w1_d"],
            Pd["w2_d"]], device=device)
        dgs = []
        for arr in (Pd["amm_d"], Pd["w1_d"], Pd["w2_d"]):
            wp.launch(gat, dim=n, inputs=[arr, Pd["diag_d"], Pd["dg_d"]],
                      device=device)
            dg = Pd["dg_d"].numpy()
            dg[dg == 0] = 1.0
            dgs.append(dg)
        dgs[2] = np.abs(dgs[2])         # Jacobi sign-guard (indefinite)
        mkop = lambda a_, Pd=Pd: CSROperator.from_device_arrays(
            Pd["rowptr_d"], Pd["colnodes_d"], a_, n, device)
        opM = mkop(Pd["amm_d"])
        opW1, opW2 = mkop(Pd["w1_d"]), mkop(Pd["w2_d"])
        base = np.arange(n, dtype=np.int64) * ndof + p["off"]
        pairs.append(dict(
            ci=base, mi=base + 1, m=mmo, Pd=Pd,
            opAcm=mkop(Pd["acm_d"]), opAmc=mkop(Pd["amc_d"]), opM=opM,
            msolve=lambda y, o=opM, d_=dgs[0]:
                _dev_solve(o, y, d_, 1e-10, cg_dev, "mass"),
            w1solve=lambda y, o=opW1, d_=dgs[1]:
                _dev_solve(o, y, d_, 1e-8, cg_dev, "W1"),
            w2solve=lambda y, o=opW2, d_=dgs[2]:
                _dev_solve(o, y, d_, 1e-8, bicgstab_dev, "W2")))

    acs = []
    for a_blk, Bd in zip(meta.get("ac", ()), setup["ac"]):
        for name, buf in (("ss", Bd["ss_d"]), ("ts", Bd["ts_d"]),
                          ("tt", Bd["tt_d"])):
            if buf is None:
                continue                # masked-out KWC torque block
            wp.launch(gat, dim=Bd["nnzp"],
                      inputs=[vals_d, Bd["pos_d"][name], buf],
                      device=device)
        dgs = []
        for arr in (Bd["ss_d"], Bd["tt_d"]):
            wp.launch(gat, dim=n, inputs=[arr, Bd["diag_d"],
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

    opA = CSROperator.from_device_arrays(*setup["A_idx_d"], vals_d, N,
                                         device)
    it = [0]
    x, info = _lgmres(LinearOperator((N, N), opA.matvec_numpy), b,
                      M=LinearOperator((N, N), apply),
                      rtol=tol, atol=1e-13, maxiter=100,
                      callback=lambda _: it.__setitem__(0, it[0] + 1))
    if info != 0:
        # ESCALATE: per-pair exact Schur; Acc gathered lazily on device
        def _mk_schur(P):
            Pd = P["Pd"]
            acc_d = wp.zeros(Pd["nnzp"], dtype=wp.float64, device=device)
            wp.launch(gat, dim=Pd["nnzp"],
                      inputs=[vals_d, Pd["pos_d"]["cc"], acc_d],
                      device=device)
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
        x, info = _lgmres(LinearOperator((N, N), opA.matvec_numpy), b,
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
                 device="cuda:0", cache=None, cache_key=None,
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
    if return_result:
        from .result import LinearSolveResult
        x = solve_linear(A, b, solver=solver, sym=sym, tol=tol,
                         maxiter=maxiter, device=device, cache=cache,
                         cache_key=cache_key)
        iters = None
        if cache is not None and cache_key is not None:
            rec = cache.get(("blockch_iters", cache_key))
            if rec is not None:
                iters = rec[0]
        return LinearSolveResult(x=x, converged=True, iterations=iters,
                                 backend=solver, reason="converged")
    A = A.tocsr()
    if cache is not None and cache_key is not None \
            and solver not in ("blockch",):
        # cheap staleness guard (evaluation solver-review item): cached
        # factorizations are for CONSTANT matrices — catch reuse of a key
        # after the matrix changed shape/pattern (values are the caller's
        # contract; a full value check would defeat the cache's purpose).
        # blockch is EXEMPT: it caches meta/iteration records only and
        # rebuilds its factors per call — the host T^T K T pattern
        # legitimately flaps under multiphase noise (the S2 finding).
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
                                  maxiter=4000, diag=dg, check_every=50)
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

    from ..errors import ConfigError
    raise ConfigError(f"unknown solver '{solver}'")
