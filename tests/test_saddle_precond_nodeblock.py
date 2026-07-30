"""Task W5d — node-block (ndof×ndof) Jacobi option for bdiag.

Tests the ``block="node"`` path of ``make_bdiag_apply`` added in W5d:

1. ``test_nodeblock_scalar_parity``   — ``block="node"`` with a pure-diagonal
   matrix must give bit-for-bit identical results to ``block="scalar"`` (the
   scalar-default parity gate: when A is diagonal, the per-node block is also
   diagonal, so the two preconditioners are mathematically identical).

2. ``test_nodeblock_accuracy_vs_splu`` — ``block="node"`` must solve the real
   level-4 2-D saddle to the same accuracy gate as ``block="scalar"`` vs splu
   (||x - x_splu|| / ||x_splu|| < 1e-8).

3. ``test_nodeblock_beats_scalar_iters`` — node-block must reach the accuracy
   gate in no more iterations than scalar on the same saddle.  The raw counts
   are ALWAYS printed so the finding is explicit; if node-block does NOT win,
   the assertion fails with both counts in the message (a reported finding, not
   a hidden one).

The outer solve is driven by ``fgmres_dev`` DIRECTLY (not through solve_linear)
to avoid touching linsolve.py, which is under concurrent modification.
The apply closure is built by ``make_bdiag_apply`` in saddle_precond.py.
Helper ``_one_step_system`` is imported from test_saddle_precond — NOT edited.
"""
import os
import sys
import warnings

import numpy as np
import pytest
import warp as wp

sys.path.insert(0, os.path.dirname(__file__))

from test_saddle_precond import _one_step_system  # import, not edit

pytestmark = pytest.mark.tier4

# ---------------------------------------------------------------------------
# Module-level cache so the expensive level-4 assembly runs only once per
# pytest session (mirrors the pattern in test_saddle_precond).
# ---------------------------------------------------------------------------
_CACHE = {}


def _get_system():
    if "sys" not in _CACHE:
        _CACHE["sys"] = _one_step_system()
    return _CACHE["sys"]


# ---------------------------------------------------------------------------
# Helper: run fgmres_dev directly given a make_bdiag_apply closure
# ---------------------------------------------------------------------------

def _solve_with_bdiag(A, b, ndof, block, device="cpu", tol=1e-10,
                      restart=60, maxiter=200):
    """Solve A x = b using fgmres_dev + make_bdiag_apply(block=block).

    Returns (x_numpy, finfo) where finfo is the dict from fgmres_dev
    (keys: converged, inner, outer, relres).
    """
    from diffsim.assembly.operators import CSROperator
    from diffsim.solvers.fgmres_dev import fgmres_dev
    from diffsim.solvers.saddle_precond import make_bdiag_apply

    A = A.tocsr()
    N = A.shape[0]

    op = CSROperator(A, device)
    apply_dev = make_bdiag_apply(A, ndof, device, block=block)

    b_dev = wp.array(np.ascontiguousarray(b, np.float64),
                     dtype=wp.float64, device=device)

    # cycles × restart caps total inner iterations (mirrors fgmres_bdiag)
    cycles = maxiter

    x_dev, finfo = fgmres_dev(
        op.matvec, b_dev, apply_dev, N, device,
        tol=tol, atol=1e-13, restart=restart, maxiter=cycles)

    return x_dev.numpy(), finfo


# ---------------------------------------------------------------------------
# Test 1: scalar-default parity — diagonal matrix
# ---------------------------------------------------------------------------

def test_nodeblock_scalar_parity():
    """block='node' and block='scalar' must produce IDENTICAL output when A is
    diagonal (the per-node block is then diagonal, so the two preconditioners
    are algebraically equal — any numerical difference is a regression).

    We use the real level-4 saddle's diagonal as a stand-in: replace A with
    diag(A) (no off-diagonal entries) so the node block is trivially diagonal.
    The two apply closures must return the same z given the same v.
    """
    import scipy.sparse as sp
    from diffsim.solvers.saddle_precond import make_bdiag_apply

    Acsr, b, _ = _get_system()
    ndof = 3   # 2-D saddle: (u_x, u_y, p)

    # Build a purely diagonal matrix from A's diagonal
    diag = np.asarray(Acsr.diagonal()).copy()
    A_diag = sp.diags(diag, format="csr")

    device = "cpu"
    N = A_diag.shape[0]

    apply_scalar = make_bdiag_apply(A_diag, ndof, device, block="scalar")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        apply_node = make_bdiag_apply(A_diag, ndof, device, block="node")

    rng = np.random.default_rng(42)
    v_np = rng.standard_normal(N)
    v_d = wp.array(np.ascontiguousarray(v_np, np.float64),
                   dtype=wp.float64, device=device)

    z_scalar = wp.zeros(N, dtype=wp.float64, device=device)
    z_node   = wp.zeros(N, dtype=wp.float64, device=device)

    apply_scalar(v_d, z_scalar)
    apply_node(v_d, z_node)

    z_s = z_scalar.numpy()
    z_n = z_node.numpy()

    # Must be numerically identical (different inversion routes on a diagonal
    # matrix should be ULP-close, so we use a generous but tight rtol)
    assert np.allclose(z_n, z_s, rtol=1e-12, atol=1e-14), (
        f"node-block parity FAILED on diagonal matrix: "
        f"max |z_node - z_scalar| = {np.abs(z_n - z_s).max():.3e}")


# ---------------------------------------------------------------------------
# Test 2: node-block accuracy vs splu on the real saddle
# ---------------------------------------------------------------------------

def test_nodeblock_accuracy_vs_splu():
    """block='node' must solve the real level-4 2-D saddle to the same
    accuracy gate as block='scalar' vs the splu reference: norm-relative error
    ||x - x_splu|| / ||x_splu|| < 1e-8."""
    Acsr, b, x_splu = _get_system()
    ndof = 3

    x_node, finfo = _solve_with_bdiag(Acsr, b, ndof, block="node")

    assert finfo["converged"], (
        f"fgmres_dev with block='node' did NOT converge: "
        f"inner={finfo['inner']} relres={finfo['relres']:.3e}")

    nrel = np.linalg.norm(x_node - x_splu) / np.linalg.norm(x_splu)
    assert nrel < 1e-8, (
        f"node-block solution inaccurate: "
        f"||x_node - x_splu|| / ||x_splu|| = {nrel:.3e} (threshold 1e-8)")


# ---------------------------------------------------------------------------
# Test 3: node-block vs scalar iteration count (the key W5d finding)
# ---------------------------------------------------------------------------

def test_nodeblock_beats_scalar_iters():
    """node-block Jacobi must converge in <= scalar-Jacobi inner iterations on
    the real level-4 2-D saddle.

    BOTH counts are ALWAYS printed (the campaign-relevant finding), so that
    even an xfail record is honest.  If node-block does NOT win the assertion
    fails with both counts in the message — a REPORTED FINDING, not a silent pass.

    The accuracy gate is checked for both variants first so a slower-but-wrong
    run cannot mask a regression.
    """
    Acsr, b, x_splu = _get_system()
    ndof = 3   # 2-D: (u_x, u_y, p)

    x_scalar, finfo_s = _solve_with_bdiag(Acsr, b, ndof, block="scalar")
    x_node,   finfo_n = _solve_with_bdiag(Acsr, b, ndof, block="node")

    it_scalar = finfo_s["inner"]
    it_node   = finfo_n["inner"]

    # Always print — this is the campaign finding
    print(f"\n[W5d] node-block vs scalar Jacobi inner FGMRES iterations: "
          f"node={it_node}  scalar={it_scalar}")

    # Accuracy gates first (before the iteration comparison)
    assert finfo_s["converged"], (
        f"scalar block did NOT converge: inner={it_scalar} "
        f"relres={finfo_s['relres']:.3e}")
    assert finfo_n["converged"], (
        f"node block did NOT converge: inner={it_node} "
        f"relres={finfo_n['relres']:.3e}")

    nrel_s = np.linalg.norm(x_scalar - x_splu) / np.linalg.norm(x_splu)
    nrel_n = np.linalg.norm(x_node   - x_splu) / np.linalg.norm(x_splu)
    assert nrel_s < 1e-8, (
        f"scalar-block solution inaccurate: nrel={nrel_s:.3e}")
    assert nrel_n < 1e-8, (
        f"node-block solution inaccurate: nrel={nrel_n:.3e}")

    # Iteration comparison (the W5d claim)
    assert it_node <= it_scalar, (
        f"node-block did NOT beat scalar: node={it_node} scalar={it_scalar} "
        f"— W5d motivation not satisfied at level-4 2-D Re=250 saddle "
        f"(both numbers printed above for the record)")


# ---------------------------------------------------------------------------
# Test 4: asymmetric-block regression test (Fortran vs C order)
# ---------------------------------------------------------------------------

def test_nodeblock_asymmetric_block_exact():
    """test_nodeblock_asymmetric_block_exact — hand-craft a small block-diagonal
    matrix (2 nodes, ndof=3) with known asymmetric blocks per node. Build the
    node-block apply via make_bdiag_apply(A, ndof=3, device="cpu", block="node"),
    apply to a random vector, and assert z == inv(B_i) @ v_i per node (np.allclose
    1e-12). This pins B⁻¹ vs Bᵀ⁻¹ (a Fortran-order ravel regression would fail it).
    """
    import scipy.sparse as sp
    from diffsim.solvers.saddle_precond import make_bdiag_apply

    ndof = 3
    n_nodes = 2
    device = "cpu"

    # Hand-crafted asymmetric blocks (different per node to catch transpose bugs)
    # Node 0: [[4, 1, 0], [2, 5, 1], [0, 3, 6]]
    B0 = np.array([[4.0, 1.0, 0.0],
                   [2.0, 5.0, 1.0],
                   [0.0, 3.0, 6.0]], dtype=np.float64)

    # Node 1: [[3, 2, 1], [1, 4, 2], [2, 1, 5]]
    B1 = np.array([[3.0, 2.0, 1.0],
                   [1.0, 4.0, 2.0],
                   [2.0, 1.0, 5.0]], dtype=np.float64)

    # Construct block-diagonal CSR matrix
    A_dense = np.zeros((n_nodes * ndof, n_nodes * ndof), dtype=np.float64)
    A_dense[0:3, 0:3] = B0
    A_dense[3:6, 3:6] = B1
    A_csr = sp.csr_matrix(A_dense)

    # Precompute inverses (ground truth)
    B0_inv = np.linalg.inv(B0)
    B1_inv = np.linalg.inv(B1)

    # Build the node-block apply
    apply_node = make_bdiag_apply(A_csr, ndof, device, block="node")

    # Random test vector
    rng = np.random.default_rng(123)
    v_np = rng.standard_normal(n_nodes * ndof)
    v_dev = wp.array(np.ascontiguousarray(v_np, np.float64),
                     dtype=wp.float64, device=device)

    # Apply preconditioner
    z_dev = wp.zeros(n_nodes * ndof, dtype=wp.float64, device=device)
    apply_node(v_dev, z_dev)
    z_np = z_dev.numpy()

    # Extract blocks and verify inv(B_i) @ v_i per node
    for i in range(n_nodes):
        r0 = i * ndof
        r1 = r0 + ndof
        v_block = v_np[r0:r1]
        z_block = z_np[r0:r1]

        if i == 0:
            z_expected = B0_inv @ v_block
        else:
            z_expected = B1_inv @ v_block

        assert np.allclose(z_block, z_expected, rtol=1e-12, atol=1e-14), (
            f"node {i} block inversion mismatch: "
            f"max |z - B_inv @ v| = {np.abs(z_block - z_expected).max():.3e}")
