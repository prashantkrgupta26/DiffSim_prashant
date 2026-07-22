"""P2-R2b.1 Task 1 — solver="blockamgx" wiring tests.

Two groups:

* gpubox NUMERICS GATE (needs AMGX): `test_blockamgx_matches_splu_on_small_saddle`
  — the block-preconditioned FGMRES matches a splu direct solve on a small
  saddle. SKIPPED on the Mac (no pyamgx).
* LOCAL off-box unit tests (no AMGX): the `blockamgx_meta` plumbing, the
  `dir_rows` passthrough into `BlockAMGPreconditioner`, and the block DOF-index
  split — all exercised WITHOUT touching AMGX by stubbing `_AMGXCycle`.
"""
import numpy as np
import scipy.sparse as sp
import pytest


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

    def __init__(self, A, sym, cycles=1):
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
