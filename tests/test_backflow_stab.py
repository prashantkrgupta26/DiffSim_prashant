"""Unit tests for the outflow backflow stabilization (consistent-projection
change #6): -beta*rho int_{Gamma_out} (u.n)_- (u.v) dGamma, active only on
reverse flow. Bazilevs 2009 / Esmaily-Moghadam 2011 (directional-do-nothing).

The term must:
  * be identically ZERO when beta=0 (bit-for-bit OFF default);
  * be ZERO where the outflow-normal velocity is >= 0 (no backflow -> benign);
  * be a DISSIPATIVE (negative-semidefinite) contribution to the momentum
    operator where u.n < 0 (reverse flow), i.e. u^T A_bf u <= 0, so the
    energy balance loses the spurious inflow of convective energy.
"""
import numpy as np

from ladder_fixtures import build_square_channel_2d
from diffsim.api.ns_bricks import assemble_backflow_block, outflow_faces


LEVEL, RE, HALF = 4, 40, 0.125


def _fx():
    return build_square_channel_2d(LEVEL, RE, half=HALF, offset=0, device="cpu")


def test_outflow_faces_are_on_x1():
    fx = _fx()
    mesh = fx["mesh"]
    elem, face, ntilde = outflow_faces(mesh)
    assert len(elem) > 0
    assert np.allclose(ntilde, [1.0, 0.0])       # domain-outward +x normal
    # every reported face's corner nodes sit on x=1
    from diffsim.mesh.nodes import _local_offsets
    aidx = _local_offsets(1, mesh.dim)
    conn = mesh.conn_of[1]
    for e, f in zip(elem, face):
        ax, side = int(f) // 2, int(f) % 2
        assert (ax, side) == (0, 1)
        local = np.where(aidx[:, ax] == side)[0]
        nd = conn[e][local]
        assert np.allclose(mesh.node_coords[nd][:, 0], 1.0)


def test_beta_zero_is_exactly_empty():
    fx = _fx()
    dm, ndof = fx["dm"], fx["ndof"]
    u = np.random.default_rng(0).standard_normal((len(fx["coords"]), fx["dim"]))
    A = assemble_backflow_block(dm, u, beta=0.0, ndof=ndof)
    assert A.nnz == 0


def test_no_backflow_gives_zero_block():
    """Pure downstream flow (u.n = +u_x > 0 everywhere at outflow) -> the
    (u.n)_- factor is zero on every outflow GP -> the block is (numerically)
    empty, so the term does NOT corrupt the physics without backflow."""
    fx = _fx()
    dm, ndof, dim = fx["dm"], fx["ndof"], fx["dim"]
    u = np.zeros((len(fx["coords"]), dim))
    u[:, 0] = 1.0                                  # uniform +x, no reverse flow
    A = assemble_backflow_block(dm, u, beta=0.5, ndof=ndof)
    assert abs(A).sum() < 1e-14


def test_backflow_is_stabilizing():
    """Reverse flow at the outlet (u.n = u_x < 0) -> the operator block is
    POSITIVE-semidefinite: added to the momentum LHS it contributes
    -beta int (u.n)_- |u|^2 >= 0 (coercive), exactly counteracting the convective
    term's negative energy production on backflow. So u^T A_bf u >= 0 with equality
    only for u=0; this is the Bazilevs/Moghadam stabilizing sign."""
    fx = _fx()
    dm, ndof, dim = fx["dm"], fx["ndof"], fx["dim"]
    n_free = len(fx["coords"])
    u = np.zeros((n_free, dim))
    u[:, 0] = -1.0                                 # uniform reverse flow (-x)
    beta = 0.5
    A = assemble_backflow_block(dm, u, beta=beta, ndof=ndof)
    assert A.nnz > 0                               # the term is actually active
    # build the node-major velocity vector and evaluate u^T A u
    x = np.zeros(n_free * ndof)
    x.reshape(n_free, ndof)[:, :dim] = u
    quad = float(x @ (A @ x))
    assert quad > 0.0, f"backflow block not stabilizing: uAu={quad:+.3e}"
    # positive-semidefinite on the velocity block: test a batch of random fields
    rng = np.random.default_rng(1)
    for _ in range(5):
        y = np.zeros(n_free * ndof)
        y.reshape(n_free, ndof)[:, :dim] = rng.standard_normal((n_free, dim))
        assert float(y @ (A @ y)) >= -1e-12
    # sign/scale: doubling beta doubles the (linearized-in-beta) block
    A2 = assemble_backflow_block(dm, u, beta=2 * beta, ndof=ndof)
    assert np.allclose(A2.toarray(), 2.0 * A.toarray())
