"""C0 regression tests -- the hands-on fail->pass gate.

The load-bearing test is ``test_jacobian_matches_fd``: it checks the analytic
Jacobian against a finite-difference of the residual.  Point it at the STARTER
module (reaction Jacobian omitted) and it FAILS; point it at the completed
``scalar_reaction`` and it PASSES::

    RD_MODULE=scalar_reaction_starter pytest -q test_reaction_jacobian.py   # RED
    pytest -q test_reaction_jacobian.py                                     # GREEN

The module under test is chosen by the ``RD_MODULE`` env var (default the
completed ``scalar_reaction``), so the same test file drives both states.
``test_stiffness_matches_production`` cross-checks the transparent stiffness
against the production Warp-kernel assembly (``assemble_brick_csr``), and
``test_mms_converges`` checks second-order accuracy -- both always use the
completed module.
"""
import importlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RD_MODULE = os.environ.get("RD_MODULE", "scalar_reaction")
rd = importlib.import_module(RD_MODULE)
import scalar_reaction as ref                      # the correct reference


def _fd_jacobian_rel_error(level=4, alpha=None, eps=1e-6, seed=0):
    """Relative L2 error between the analytic Jacobian action J v and the
    finite-difference (R(u+eps v) - R(u)) / eps, at a random state u."""
    alpha = ref.ALPHA_DEFAULT if alpha is None else alpha
    prob = rd.Problem(level=level, alpha=alpha, device="cpu")
    rng = np.random.default_rng(seed)
    u = 0.6 * rng.standard_normal(prob.n)
    u[prob.bnd] = prob.gvals[prob.bnd]             # consistent Dirichlet state
    v = rng.standard_normal(prob.n)
    J = rd.jacobian(prob, u)
    Jv = J @ v
    fd = (rd.residual(prob, u + eps * v) - rd.residual(prob, u)) / eps
    return float(np.linalg.norm(Jv - fd) / (np.linalg.norm(fd) + 1e-30))


def test_jacobian_matches_fd():
    """The analytic Jacobian must match the finite-difference of the residual.

    With the reaction Jacobian present this is limited only by FD truncation
    (~1e-6); with it MISSING (the starter) the mismatch is O(1).  Threshold
    1e-4 cleanly separates the two -- this is the hands-on red->green gate.
    """
    rel = _fd_jacobian_rel_error()
    assert rel < 1e-4, (
        f"analytic Jacobian disagrees with finite difference "
        f"(rel error {rel:.3e}); the reaction term 3*alpha*u^2 is likely "
        f"missing from jacobian() in module '{RD_MODULE}'")


def test_stiffness_matches_production():
    """The transparent diffusion stiffness equals the production Warp-kernel
    assembly (``assemble_brick_csr`` with ``PoissonBrick``) to round-off --
    proof the readable NumPy path IS the same math as the device kernel."""
    from diffsim.api.equation import assemble_brick_csr
    from diffsim.api import PoissonBrick
    prob = ref.Problem(level=4, device="cpu")
    K_transparent = ref.stiffness_csr(prob)
    K_prod = assemble_brick_csr(prob.dm, PoissonBrick())   # T^T K T, T = I here
    diff = abs(K_transparent - K_prod)
    assert diff.nnz == 0 or diff.max() < 1e-11, (
        f"transparent stiffness differs from production kernel "
        f"(max |dK| = {diff.max():.3e})")


def test_mms_converges():
    """Manufactured-solution L2 error falls at the p1 rate (~h^2)."""
    errs = []
    for level in (3, 4, 5):
        prob = ref.Problem(level=level, device="cpu")
        u, info = ref.newton_solve(prob, solver="splu", device="cpu")
        assert info["converged"], f"Newton did not converge at level {level}"
        errs.append(ref.l2_error(prob, u))
    order = np.log2(errs[0] / errs[1])
    order2 = np.log2(errs[1] / errs[2])
    assert 1.7 < order < 2.3 and 1.7 < order2 < 2.3, (
        f"MMS L2 orders {order:.2f}, {order2:.2f} not near 2 "
        f"(errors {errs})")


if __name__ == "__main__":       # convenience: `python test_reaction_jacobian.py`
    rel = _fd_jacobian_rel_error()
    status = "PASS" if rel < 1e-4 else "FAIL"
    print(f"[{status}] jacobian-vs-FD rel error = {rel:.3e}  "
          f"(module '{RD_MODULE}', threshold 1e-4)")
    sys.exit(0 if rel < 1e-4 else 1)
