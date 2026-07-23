"""P2-R2b.1 Task 1 — solver="blockamgx" wiring tests.

Two groups:

* gpubox NUMERICS GATE (needs AMGX): `test_blockamgx_matches_splu_on_small_saddle`
  — the block-preconditioned FGMRES matches a splu direct solve on a small
  saddle. SKIPPED on the Mac (no pyamgx).
* LOCAL off-box unit tests (no AMGX): the `blockamgx_meta` plumbing, the
  `dir_rows` passthrough into `BlockAMGPreconditioner`, and the block DOF-index
  split — all exercised WITHOUT touching AMGX by stubbing `_AMGXCycle`.
"""
import os
import sys

import numpy as np
import scipy.sparse as sp
import pytest

sys.path.insert(0, os.path.dirname(__file__))


def _small_saddle(n_nodes=40, dim=2, seed=0):
    """A small node-major interleaved (u,p) saddle system with a mass-
    dominated velocity block (sigma large) and a Cahouet-Chabard-friendly
    pressure block, so the block preconditioner is a good match."""
    rng = np.random.default_rng(seed)
    ndof = dim + 1
    N = n_nodes * ndof
    sigma, nu = 50.0, 0.02
    # scalar pressure stiffness Kp (SPD, pinned last row) and mass diag
    Kp = sp.diags([2.1] * n_nodes) - sp.eye(n_nodes, k=1) * 0.5 \
        - sp.eye(n_nodes, k=-1) * 0.5
    Kp = sp.csr_matrix(Kp)
    Mp_diag = np.full(n_nodes, 0.7)
    # assemble a monolithic A on interleaved layout
    idx = np.arange(N).reshape(n_nodes, ndof)
    u_ids = idx[:, :dim].ravel()
    p_ids = idx[:, dim].ravel()
    A = sp.lil_matrix((N, N))
    F = sigma * sp.eye(len(u_ids)) + 0.05 * sp.random(
        len(u_ids), len(u_ids), density=0.02, random_state=rng)
    G = 0.3 * sp.random(len(u_ids), n_nodes, density=0.05, random_state=rng)
    A[np.ix_(u_ids, u_ids)] = F.toarray()
    A[np.ix_(u_ids, p_ids)] = G.toarray()
    A[np.ix_(p_ids, u_ids)] = -G.T.toarray()
    A[np.ix_(p_ids, p_ids)] = (nu * sp.diags(Mp_diag)).toarray()
    A = A.tocsr()
    b = rng.standard_normal(N)
    return dict(A=A, b=b, n_nodes=n_nodes, ndof=ndof, Kp=Kp,
                Mp_diag=Mp_diag, sigma=sigma, nu=nu)


# --------------------------------------------------------------------------
# gpubox NUMERICS GATE (needs AMGX) — skips on the Mac.
# --------------------------------------------------------------------------
def test_blockamgx_matches_splu_on_small_saddle():
    pytest.importorskip("pyamgx")  # gpubox-only; skips on the Mac
    from diffsim.solvers.linsolve import solve_linear
    s = _small_saddle()
    x_direct = solve_linear(s["A"], s["b"], solver="splu")
    cache, key = {}, "t1"
    cache[("blockamgx_meta", key)] = dict(
        n_nodes=s["n_nodes"], ndof=s["ndof"], Kp=s["Kp"],
        Mp_diag=s["Mp_diag"], sigma=s["sigma"], nu=s["nu"], dir_rows=None)
    x_block = solve_linear(s["A"], s["b"], solver="blockamgx", tol=1e-8,
                           cache=cache, cache_key=key)
    rel = np.linalg.norm(x_block - x_direct) / np.linalg.norm(x_direct)
    assert rel < 1e-6, f"blockamgx vs splu rel err {rel:.2e}"
    outer, = cache[("blockamgx_iters", key)]
    assert 0 < outer < 200


# --------------------------------------------------------------------------
# LOCAL off-box unit tests — no AMGX. `_AMGXCycle` is stubbed to an exact
# dense inverse so the preconditioner runs on the CPU and we can assert on
# plumbing / index math / dir_rows without a GPU.
# --------------------------------------------------------------------------
class _ExactCycleStub:
    """Drop-in for `_AMGXCycle`: a dense-LU exact block solve. Lets the block
    preconditioner run on the CPU so the OUTER FGMRES converges in one step —
    which makes the plumbing testable off-box."""

    def __init__(self, A, sym, cycles=1, tol=0.0, pre_cycles=1):
        from scipy.sparse.linalg import splu
        self._lu = splu(sp.csc_matrix(A))

    def solve(self, b, **_ignored):
        return self._lu.solve(np.ascontiguousarray(b, np.float64))


@pytest.fixture
def _stub_amgx(monkeypatch):
    import diffsim.solvers.block_precond as bp
    monkeypatch.setattr(bp, "_AMGXCycle", _ExactCycleStub)
    return bp


def test_blockamgx_meta_missing_raises():
    """The branch must raise a clear error if the caller forgot the meta."""
    from diffsim.solvers.linsolve import solve_linear
    s = _small_saddle(n_nodes=8)
    with pytest.raises(ValueError, match="blockamgx_meta"):
        solve_linear(s["A"], s["b"], solver="blockamgx",
                     cache={}, cache_key="missing")


def test_blockamgx_block_dof_split():
    """u_ids/p_ids split of a node-major interleaved CSR: velocity DOFs are
    the first `dim` per node, pressure is the last."""
    from diffsim.solvers.block_precond import BlockAMGPreconditioner
    n_nodes, ndof = 5, 3
    A = sp.eye(n_nodes * ndof, format="csr")
    # build with the stub-free path: only the index math runs before the
    # _AMGXCycle build, so patch it locally here too.
    import diffsim.solvers.block_precond as bp
    orig = bp._AMGXCycle
    bp._AMGXCycle = _ExactCycleStub
    try:
        pre = BlockAMGPreconditioner(A, n_nodes, ndof, sp.eye(n_nodes,
                                     format="csr"), np.ones(n_nodes), 1.0, 1.0)
    finally:
        bp._AMGXCycle = orig
    idx = np.arange(n_nodes * ndof).reshape(n_nodes, ndof)
    assert np.array_equal(pre.u_ids, idx[:, :2].ravel())
    assert np.array_equal(pre.p_ids, idx[:, 2].ravel())
    # every DOF is covered exactly once
    assert (sorted(np.concatenate([pre.u_ids, pre.p_ids]))
            == list(range(n_nodes * ndof)))


def test_dir_rows_passthrough_maps_to_velocity_local(_stub_amgx):
    """dir_rows (monolithic identity-row ids) must map to velocity-block-local
    indices in `_u_dir`, and pressure-DOF ids must be dropped."""
    from diffsim.solvers.block_precond import BlockAMGPreconditioner
    n_nodes, ndof, dim = 6, 3, 2
    A = sp.eye(n_nodes * ndof, format="csr")
    # node 0 velocity DOFs (rows 0,1), node 2 velocity DOF (row 6), plus a
    # PRESSURE DOF (row 5 = node1*3+2) that must be IGNORED.
    dir_rows = np.array([0, 1, 6, 5])
    pre = BlockAMGPreconditioner(
        A, n_nodes, ndof, sp.eye(n_nodes, format="csr"), np.ones(n_nodes),
        1.0, 1.0, dir_rows=dir_rows)
    # u_ids = [0,1, 3,4, 6,7, 9,10, 12,13, 15,16]; local pos of monolithic
    # rows 0->0, 1->1, 6->4. Row 5 is a pressure DOF -> dropped.
    assert pre._u_dir is not None
    assert sorted(pre._u_dir.tolist()) == [0, 1, 4]


def test_dir_rows_enforced_identity_in_apply(_stub_amgx):
    """apply() must reproduce the residual EXACTLY on the strong-Dirichlet
    velocity rows (identity action), regardless of the F cycle."""
    from diffsim.solvers.block_precond import BlockAMGPreconditioner
    s = _small_saddle(n_nodes=12, dim=2)
    # mark the first node's velocity DOFs as strong-Dirichlet identity rows
    dir_rows = np.array([0, 1])
    pre = BlockAMGPreconditioner(
        s["A"], s["n_nodes"], s["ndof"], s["Kp"], s["Mp_diag"],
        s["sigma"], s["nu"], dir_rows=dir_rows)
    r = np.random.default_rng(1).standard_normal(s["A"].shape[0])
    z = pre.apply(r)
    # on the Dirichlet rows the preconditioner is identity on the CORRECTED
    # residual r_u - G z_p; since these rows are in F, z[dir] == (r - G z_p)[dir]
    r_u = r[pre.u_ids]
    r_p = r[pre.p_ids]
    z_p = (s["sigma"] * pre._amg_Kp.solve(r_p)
           + s["nu"] * (r_p / pre.Mp_diag))
    expected = (r_u - pre.G @ z_p)[pre._u_dir]
    got = z[pre.u_ids][pre._u_dir]
    assert np.allclose(got, expected, atol=1e-12)


def test_blockamgx_local_solve_via_exact_stub(_stub_amgx):
    """End-to-end plumbing with the exact stub: solve_linear(solver=blockamgx)
    matches splu, and stashes the outer-iteration record. No AMGX involved."""
    from diffsim.solvers.linsolve import solve_linear
    s = _small_saddle(n_nodes=20, dim=2)
    x_direct = solve_linear(s["A"], s["b"], solver="splu")
    cache, key = {}, "local"
    cache[("blockamgx_meta", key)] = dict(
        n_nodes=s["n_nodes"], ndof=s["ndof"], Kp=s["Kp"],
        Mp_diag=s["Mp_diag"], sigma=s["sigma"], nu=s["nu"], dir_rows=None)
    x_block = solve_linear(s["A"], s["b"], solver="blockamgx", tol=1e-10,
                           cache=cache, cache_key=key)
    rel = np.linalg.norm(x_block - x_direct) / np.linalg.norm(x_direct)
    assert rel < 1e-6, f"exact-stub blockamgx vs splu rel err {rel:.2e}"
    assert ("blockamgx_iters", key) in cache
    outer, = cache[("blockamgx_iters", key)]
    assert 0 < outer < 200


# --------------------------------------------------------------------------
# Task 3 LOCAL test — monolithic_cd(solver="blockamgx") assembles the meta
# from the real 3-D sphere march and feeds it to solve_linear. AMGX is stubbed
# by the exact dense cycle so the whole march runs on the Mac (CPU). We assert
# the meta was built with the right keys/shapes/space AND that the blockamgx
# march matches the splu march step-for-step (the exact stub makes the block
# preconditioner an exact solve, so the two marches are identical to solver
# tolerance).
# --------------------------------------------------------------------------
def test_monolithic_cd_blockamgx_meta_and_march(_stub_amgx, monkeypatch):
    from p2r0_task10_sphere_derisk import build_sphere_3d, monolithic_cd

    fx = build_sphere_3d("cpu", 3, 100.0)
    dim, ndof = fx["dim"], fx["ndof"]
    nfree = fx["cons"].T.shape[1]
    strong = int(np.count_nonzero(fx["strong_mask"]))
    dt = 0.05

    # capture the meta solve_linear actually receives, and short-circuit the
    # AMGX solve path: with the exact-cycle stub the block preconditioner IS an
    # exact solve, so we can also let it run and compare marches. First: assert
    # the meta contract the blockamgx branch reads.
    import diffsim.solvers.linsolve as ls
    seen = {}
    orig = ls.solve_linear

    def _spy(A, b, **kw):
        if kw.get("solver") == "blockamgx":
            m = kw["cache"][("blockamgx_meta", kw["cache_key"])]
            seen.update(m)
            seen["_A_shape"] = A.shape
        return orig(A, b, **kw)

    monkeypatch.setattr(
        "p2r0_task10_sphere_derisk.solve_linear", _spy, raising=True)

    res_blk = monolithic_cd(fx, alpha=20.0, dt=dt, max_steps=2,
                            rate_tol=1e-9, solver="blockamgx")

    # ---- meta contract: exactly the keys the linsolve blockamgx branch reads
    assert set(seen) >= {"n_nodes", "ndof", "Kp", "Mp_diag", "sigma", "nu",
                         "dir_rows"}
    assert seen["n_nodes"] == nfree          # FREE-node pressure space
    assert seen["ndof"] == ndof
    # Kp/Mp_diag live on the FREE-node SCALAR pressure space (size nfree)
    assert seen["Kp"].shape == (nfree, nfree)
    assert seen["Mp_diag"].shape == (nfree,)
    assert np.isclose(seen["sigma"], 1.0 / dt)
    assert np.isclose(seen["nu"], fx["nu"])
    # dir_rows: strong velocity rows i*ndof+c (monolithic global dof ids)
    assert seen["dir_rows"].shape == (strong * dim,)
    assert seen["dir_rows"].max() < nfree * ndof
    assert (seen["dir_rows"] % ndof < dim).all()   # velocity components only
    # the pressure pin is a valid diagonal entry in Kp / Mp_diag
    p_pin = int(np.argmax(fx["coords"].sum(1)))
    assert np.isclose(seen["Kp"][p_pin, p_pin], 1.0)
    assert np.isclose(seen["Mp_diag"][p_pin], 1.0)
    assert seen["_A_shape"] == (nfree * ndof, nfree * ndof)

    # ---- march equivalence: exact-stub block precond == splu direct solve
    res_splu = monolithic_cd(fx, alpha=20.0, dt=dt, max_steps=2,
                             rate_tol=1e-9, solver="splu")
    assert np.isclose(res_blk["cd"], res_splu["cd"], rtol=1e-5, atol=1e-6), (
        f"blockamgx Cd {res_blk['cd']} vs splu {res_splu['cd']}")


def test_monolithic_cd_splu_path_takes_no_meta(monkeypatch):
    """The splu/cudss paths must NOT build or pass any blockamgx meta —
    guard the branch so the other solvers are untouched by Task 3."""
    from p2r0_task10_sphere_derisk import build_sphere_3d, monolithic_cd
    fx = build_sphere_3d("cpu", 3, 100.0)
    calls = {"cache_seen": False}
    import diffsim.solvers.linsolve as ls
    orig = ls.solve_linear

    def _spy(A, b, **kw):
        if kw.get("cache") is not None:
            calls["cache_seen"] = True
        return orig(A, b, **kw)

    # splu goes straight through scipy.splu (never solve_linear); cudss would
    # call solve_linear WITHOUT a cache. Only blockamgx passes a cache.
    monkeypatch.setattr(
        "p2r0_task10_sphere_derisk.solve_linear", _spy, raising=True)
    monolithic_cd(fx, alpha=20.0, dt=0.05, max_steps=1, rate_tol=1e-9,
                  solver="splu")
    assert calls["cache_seen"] is False


# --------------------------------------------------------------------------
# Task 6 LOCAL tests — tunable inner-solve strengths plumb through the meta
# into the preconditioner (attributes) and defaults reproduce current values.
# All exercised via the exact stub (no AMGX).
# --------------------------------------------------------------------------
def test_blockprecond_default_knobs_match_current_behavior():
    """With NO tuning args, the preconditioner attributes must equal the
    original hardcoded values (bit-for-bit current behavior)."""
    from diffsim.solvers.block_precond import BlockAMGPreconditioner
    import diffsim.solvers.block_precond as bp
    orig = bp._AMGXCycle
    bp._AMGXCycle = _ExactCycleStub
    try:
        s = _small_saddle(n_nodes=10)
        pre = BlockAMGPreconditioner(
            s["A"], s["n_nodes"], s["ndof"], s["Kp"], s["Mp_diag"],
            s["sigma"], s["nu"])
    finally:
        bp._AMGXCycle = orig
    assert pre.f_iters == 2 and pre.f_tol == 1e-2
    assert pre.kp_iters == 8 and pre.kp_tol == 1e-3
    assert pre.f_cycles == 1 and pre.kp_cycles == 3


def test_blockprecond_tuning_knobs_stored():
    """Explicit tuning args must land on the preconditioner attributes."""
    from diffsim.solvers.block_precond import BlockAMGPreconditioner
    import diffsim.solvers.block_precond as bp
    orig = bp._AMGXCycle
    bp._AMGXCycle = _ExactCycleStub
    try:
        s = _small_saddle(n_nodes=10)
        pre = BlockAMGPreconditioner(
            s["A"], s["n_nodes"], s["ndof"], s["Kp"], s["Mp_diag"],
            s["sigma"], s["nu"], f_iters=6, f_tol=5e-2, kp_iters=20,
            kp_tol=1e-4, f_cycles=2, kp_cycles=5)
    finally:
        bp._AMGXCycle = orig
    assert pre.f_iters == 6 and pre.f_tol == 5e-2
    assert pre.kp_iters == 20 and pre.kp_tol == 1e-4
    assert pre.f_cycles == 2 and pre.kp_cycles == 5


def test_blockamgx_meta_tuning_threads_to_preconditioner(_stub_amgx,
                                                          monkeypatch):
    """meta keys f_iters/f_tol/kp_iters/kp_tol/f_cycles/kp_cycles +
    gmres_restart/gmres_maxiter must be read from the meta and passed into
    the preconditioner ctor / solve_block_preconditioned."""
    import diffsim.solvers.block_precond as bp
    from diffsim.solvers.linsolve import solve_linear
    captured = {}
    orig_ctor = bp.BlockAMGPreconditioner.__init__

    def _spy_ctor(self, *a, **kw):
        captured["ctor_kw"] = dict(kw)
        return orig_ctor(self, *a, **kw)
    monkeypatch.setattr(bp.BlockAMGPreconditioner, "__init__", _spy_ctor)

    orig_solve = bp.solve_block_preconditioned

    def _spy_solve(A, b, pre, **kw):
        captured["solve_kw"] = dict(kw)
        return orig_solve(A, b, pre, **kw)
    # the branch does `from .block_precond import ...` at call time, so
    # patching the module attribute is what takes effect.
    monkeypatch.setattr(bp, "solve_block_preconditioned", _spy_solve)

    s = _small_saddle(n_nodes=16)
    cache, key = {}, "tune"
    cache[("blockamgx_meta", key)] = dict(
        n_nodes=s["n_nodes"], ndof=s["ndof"], Kp=s["Kp"],
        Mp_diag=s["Mp_diag"], sigma=s["sigma"], nu=s["nu"], dir_rows=None,
        f_iters=4, f_tol=3e-2, kp_iters=12, kp_tol=2e-4,
        f_cycles=2, kp_cycles=4, gmres_restart=30, gmres_maxiter=77)
    solve_linear(s["A"], s["b"], solver="blockamgx", tol=1e-8,
                 cache=cache, cache_key=key)
    ck = captured["ctor_kw"]
    assert ck["f_iters"] == 4 and ck["f_tol"] == 3e-2
    assert ck["kp_iters"] == 12 and ck["kp_tol"] == 2e-4
    assert ck["f_cycles"] == 2 and ck["kp_cycles"] == 4
    sk = captured["solve_kw"]
    assert sk["restart"] == 30 and sk["maxiter"] == 77


def test_blockamgx_meta_absent_tuning_uses_defaults(_stub_amgx, monkeypatch):
    """No tuning meta keys -> ctor gets no tuning kwargs and the outer solve
    uses restart=50/maxiter=200 (current behavior)."""
    import diffsim.solvers.block_precond as bp
    from diffsim.solvers.linsolve import solve_linear
    captured = {}
    orig_ctor = bp.BlockAMGPreconditioner.__init__

    def _spy_ctor(self, *a, **kw):
        captured["ctor_kw"] = dict(kw)
        return orig_ctor(self, *a, **kw)
    monkeypatch.setattr(bp.BlockAMGPreconditioner, "__init__", _spy_ctor)

    orig_solve = bp.solve_block_preconditioned

    def _spy_solve(A, b, pre, **kw):
        captured["solve_kw"] = dict(kw)
        return orig_solve(A, b, pre, **kw)
    monkeypatch.setattr(bp, "solve_block_preconditioned", _spy_solve)

    s = _small_saddle(n_nodes=16)
    cache, key = {}, "def"
    cache[("blockamgx_meta", key)] = dict(
        n_nodes=s["n_nodes"], ndof=s["ndof"], Kp=s["Kp"],
        Mp_diag=s["Mp_diag"], sigma=s["sigma"], nu=s["nu"], dir_rows=None)
    solve_linear(s["A"], s["b"], solver="blockamgx", tol=1e-8,
                 cache=cache, cache_key=key)
    # no tuning keys leaked into the ctor kwargs (only dir_rows present)
    assert set(captured["ctor_kw"]) == {"dir_rows"}
    # outer solve falls back to the current restart/maxiter defaults
    assert captured["solve_kw"]["restart"] == 50
    assert captured["solve_kw"]["maxiter"] == 200


def test_sphere_derisk_env_vars_populate_meta(monkeypatch):
    """The monolithic_cd blockamgx branch reads F_ITERS/... env vars into the
    meta dict; absent env => key omitted (default holds)."""
    import diffsim.solvers.block_precond as bp
    monkeypatch.setattr(bp, "_AMGXCycle", _ExactCycleStub)
    from p2r0_task10_sphere_derisk import build_sphere_3d, monolithic_cd
    import diffsim.solvers.linsolve as ls
    fx = build_sphere_3d("cpu", 3, 100.0)
    seen = {}
    orig = ls.solve_linear

    def _spy(A, b, **kw):
        c = kw.get("cache")
        if c is not None:
            for (tag, _), v in c.items():
                if tag == "blockamgx_meta":
                    seen["meta"] = v
        return orig(A, b, **kw)
    monkeypatch.setattr(
        "p2r0_task10_sphere_derisk.solve_linear", _spy, raising=True)

    for e in ("F_ITERS", "F_TOL", "KP_ITERS", "KP_TOL", "F_CYCLES",
              "KP_CYCLES", "GMRES_RESTART", "GMRES_MAXITER"):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv("F_ITERS", "5")
    monkeypatch.setenv("F_TOL", "0.05")
    monkeypatch.setenv("GMRES_RESTART", "40")

    monolithic_cd(fx, alpha=20.0, dt=0.05, max_steps=1, rate_tol=1e-9,
                  solver="blockamgx")
    m = seen["meta"]
    assert m["f_iters"] == 5 and m["f_tol"] == 0.05
    assert m["gmres_restart"] == 40
    # unset env vars must NOT create meta keys (defaults must hold)
    for k in ("kp_iters", "kp_tol", "f_cycles", "kp_cycles", "gmres_maxiter"):
        assert k not in m


# --------------------------------------------------------------------------
# Task 7 LOCAL tests — the PSPG C-block Schur mode (schur_mode="pspg_c").
# The diagnostic proved Cahouet-Chabard is WRONG for equal-order PSPG (a real
# pressure-pressure block C dominates the Schur), so the new mode approximates
# S^-1 ≈ C^-1 with C = A[p_ids][:, p_ids]. All exercised via a capturing stub
# for _AMGXCycle (no AMGX): assert C is extracted at the right shape/values,
# that the C-AMG cycle is built (and Kp is NOT), and that default/CC is
# bit-for-bit unchanged.
# --------------------------------------------------------------------------
class _CaptureCycleStub:
    """Drop-in for `_AMGXCycle` that RECORDS every matrix + (sym, cycles) it is
    built on into a shared registry, and solves via a dense LU so the
    preconditioner still runs on the CPU."""
    registry = []

    def __init__(self, A, sym, cycles=1, tol=0.0, pre_cycles=1):
        from scipy.sparse.linalg import splu
        A = sp.csr_matrix(A)
        _CaptureCycleStub.registry.append(
            dict(A=A, sym=sym, cycles=cycles, tol=tol,
                 pre_cycles=pre_cycles, shape=A.shape))
        self._lu = splu(sp.csc_matrix(A))

    def solve(self, b, **_ignored):
        return self._lu.solve(np.ascontiguousarray(b, np.float64))


def _build_pre(schur_mode=None, **kw):
    """Build a BlockAMGPreconditioner on the small saddle with _AMGXCycle
    swapped for the capturing stub; returns (pre, registry_snapshot)."""
    from diffsim.solvers.block_precond import BlockAMGPreconditioner
    import diffsim.solvers.block_precond as bp
    s = _small_saddle(n_nodes=12)
    _CaptureCycleStub.registry = []
    orig = bp._AMGXCycle
    bp._AMGXCycle = _CaptureCycleStub
    try:
        extra = {} if schur_mode is None else dict(schur_mode=schur_mode)
        pre = BlockAMGPreconditioner(
            s["A"], s["n_nodes"], s["ndof"], s["Kp"], s["Mp_diag"],
            s["sigma"], s["nu"], **extra, **kw)
    finally:
        bp._AMGXCycle = orig
    return pre, list(_CaptureCycleStub.registry), s


def test_schur_mode_default_is_cahouet_chabard_unchanged():
    """No schur_mode arg -> "cahouet_chabard": builds F + Kp cycles (Kp on the
    provided pressure stiffness), NO C block, no C attribute."""
    pre, reg, s = _build_pre(schur_mode=None)
    assert pre.schur_mode == "cahouet_chabard"
    assert pre._amg_Kp is not None and pre._amg_C is None
    assert not hasattr(pre, "C")
    # exactly two cycles built: F (nonsym) and Kp (sym) — same as before.
    assert len(reg) == 2
    kp_built = [r for r in reg if r["sym"]]
    assert len(kp_built) == 1
    # the sym cycle is Kp (the pressure stiffness), NOT the p-p block of A.
    Kp = s["Kp"].tocsr()
    assert kp_built[0]["shape"] == Kp.shape
    assert np.allclose(kp_built[0]["A"].toarray(), Kp.toarray())


def test_schur_mode_pspg_c_extracts_C_block():
    """schur_mode="pspg_c": extracts C = A[p_ids][:, p_ids] at shape
    (len(p_ids), len(p_ids)), builds a sym C-cycle on it, and does NOT build
    the Cahouet-Chabard Kp cycle (2 Resources total, not 3)."""
    pre, reg, s = _build_pre(schur_mode="pspg_c")
    assert pre.schur_mode == "pspg_c"
    assert pre._amg_C is not None and pre._amg_Kp is None
    np_ids = len(pre.p_ids)
    assert pre.C.shape == (np_ids, np_ids)
    # C must equal the pressure-pressure block of the monolithic A exactly.
    A = s["A"].tocsr()
    C_ref = A[pre.p_ids][:, pre.p_ids].toarray()
    assert np.allclose(pre.C.toarray(), C_ref)
    # in _small_saddle the p-p block is nu*diag(Mp_diag).
    assert np.allclose(pre.C.toarray(),
                       s["nu"] * np.diag(s["Mp_diag"]))
    # exactly two cycles built: F (nonsym) and C (sym) — 2 Resources, and the
    # sym cycle is C (not Kp).
    assert len(reg) == 2
    sym_built = [r for r in reg if r["sym"]]
    assert len(sym_built) == 1
    assert sym_built[0]["shape"] == (np_ids, np_ids)
    assert np.allclose(sym_built[0]["A"].toarray(), C_ref)


def test_schur_mode_pspg_c_apply_uses_C_only():
    """In pspg_c mode apply() sets z_p = C^-1 r_p (no sigma*Kp^-1 + nu*Mp^-1
    terms). With the exact-LU stub, z_p must equal solve(C, r_p)."""
    from scipy.sparse.linalg import splu
    pre, reg, s = _build_pre(schur_mode="pspg_c")
    rng = np.random.default_rng(1)
    r = rng.standard_normal(s["n_nodes"] * s["ndof"])
    z = pre.apply(r)
    r_p = r[pre.p_ids]
    z_p_ref = splu(sp.csc_matrix(pre.C)).solve(r_p)
    assert np.allclose(z[pre.p_ids], z_p_ref)


def test_schur_mode_unknown_raises():
    """An unrecognized schur_mode must raise a clear ValueError."""
    with pytest.raises(ValueError, match="schur_mode"):
        _build_pre(schur_mode="bogus")


def test_schur_mode_threads_through_linsolve_meta(monkeypatch):
    """linsolve's blockamgx branch forwards meta['schur_mode'] into the
    preconditioner ctor; absent -> not passed (default holds)."""
    import diffsim.solvers.block_precond as bp
    from diffsim.solvers.linsolve import solve_linear
    captured = {}

    class _CtorSpy:
        def __init__(self, *a, **kw):
            captured["ctor_kw"] = kw

        def as_linear_operator(self):
            from scipy.sparse.linalg import LinearOperator
            n = _small_saddle(n_nodes=8)["A"].shape[0]
            return LinearOperator((n, n), matvec=lambda x: x)
    s = _small_saddle(n_nodes=8)
    # linsolve does a LOCAL `from .block_precond import ...`, so patch the
    # source module (that import pulls the current attribute at call time).
    monkeypatch.setattr(bp, "BlockAMGPreconditioner", _CtorSpy, raising=False)
    monkeypatch.setattr(bp, "solve_block_preconditioned",
                        lambda A, b, pre, **kw: (b, 1), raising=False)
    cache, key = {}, "sm"
    cache[("blockamgx_meta", key)] = dict(
        n_nodes=s["n_nodes"], ndof=s["ndof"], Kp=s["Kp"],
        Mp_diag=s["Mp_diag"], sigma=s["sigma"], nu=s["nu"], dir_rows=None,
        schur_mode="pspg_c")
    solve_linear(s["A"], s["b"], solver="blockamgx", cache=cache,
                 cache_key=key)
    assert captured["ctor_kw"].get("schur_mode") == "pspg_c"


def test_sphere_derisk_schur_mode_env_populates_meta(monkeypatch):
    """monolithic_cd reads SCHUR_MODE env into the meta; absent -> key omitted
    (default cahouet_chabard holds)."""
    import diffsim.solvers.block_precond as bp
    monkeypatch.setattr(bp, "_AMGXCycle", _ExactCycleStub)
    from p2r0_task10_sphere_derisk import build_sphere_3d, monolithic_cd
    import diffsim.solvers.linsolve as ls
    fx = build_sphere_3d("cpu", 3, 100.0)
    seen = {}
    orig = ls.solve_linear

    def _spy(A, b, **kw):
        c = kw.get("cache")
        if c is not None:
            for (tag, _), v in c.items():
                if tag == "blockamgx_meta":
                    seen["meta"] = v
        return orig(A, b, **kw)
    monkeypatch.setattr(
        "p2r0_task10_sphere_derisk.solve_linear", _spy, raising=True)
    monkeypatch.setenv("SCHUR_MODE", "pspg_c")
    monolithic_cd(fx, alpha=20.0, dt=0.05, max_steps=1, rate_tol=1e-9,
                  solver="blockamgx")
    assert seen["meta"]["schur_mode"] == "pspg_c"

    # absent SCHUR_MODE -> key omitted
    monkeypatch.delenv("SCHUR_MODE", raising=False)
    seen.clear()
    monolithic_cd(fx, alpha=20.0, dt=0.05, max_steps=1, rate_tol=1e-9,
                  solver="blockamgx")
    assert "schur_mode" not in seen["meta"]
