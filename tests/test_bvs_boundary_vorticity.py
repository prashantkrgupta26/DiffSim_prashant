"""Change #5 — P1 boundary-vorticity stabilization unit tests.

The term ``delta*(grad q x n, nu curl u)_Gamma`` (Pacheco, Schussnig, Steinbach,
Fries, IJNME 2021, nme.6615; arXiv:2411.02100) is added on the outflow boundary
to BOTH the projection PPE (as a known-u_hat RHS source, ``bvs_ppe_source``) and
the monolithic PSPG continuity row (as a u-coupling C-block, ``assemble_bvs_block``)
so the same-mesh oracle stays exact.

These are the fast, deterministic guards:
  * CONSISTENCY: the monolithic C-block applied to a velocity field equals the
    negated projection RHS source at the SAME field (the two are the same term,
    just u-unknown vs u-known) — the same-mesh-oracle invariant.
  * ANTI-VACUITY: the term is non-zero on a field with outflow vorticity, and
    vanishes on an irrotational field (curl u == 0) — it is load-bearing exactly
    where the paper says (a real vorticity flux, not a numerical fudge).
"""
import numpy as np

from ladder_fixtures import build_square_channel_2d
from diffsim.api.ns_bricks import (bvs_ppe_source, assemble_bvs_block,
                                    outflow_faces)


def _fx():
    return build_square_channel_2d(5, 40, half=0.125, offset=0, device="cpu")


def test_bvs_ppe_and_monolithic_are_the_same_term():
    """The monolithic C-block row (continuity, pressure DOF) applied to a
    velocity vector must reproduce the NEGATED projection PPE source at the same
    velocity: both discretize delta*(grad q x n, nu curl u)_Gamma, one with u
    unknown (matrix) and one with u known (RHS moved to the other side). This is
    the exact same-mesh-oracle consistency the fix requires."""
    fx = _fx()
    dm, nu, ndof = fx["dm"], fx["nu"], fx["ndof"]
    dim = dm.dim
    dt = 0.01
    faces = outflow_faces(dm.mesh)
    rng = np.random.default_rng(0)
    u_free = rng.standard_normal((dm.n_free, dim))
    # projection source (u known); PPE assembles (...,q)=0 so the term is on the
    # RHS as -delta*nu*(grad N_a x n, curl u): source_free is that RHS vector.
    src_full = bvs_ppe_source(dm, u_free, nu, dt, faces=faces)
    src_free = np.asarray(dm.constraints.T.T @ src_full)     # [n_free]

    # monolithic C-block (u unknown): the pressure (continuity) rows, applied to
    # the full (u,p) vector with these velocities, give the term with the
    # OPPOSITE sign (it sits on the LHS as +delta*nu*(grad q x n, curl u)).
    C = assemble_bvs_block(dm, u_free, nu, dt, ndof, faces=faces)
    x = np.zeros(u_free.shape[0] * ndof)
    xv = x.reshape(u_free.shape[0], ndof)
    xv[:, :dim] = u_free
    Cx = (C @ x).reshape(u_free.shape[0], ndof)
    lhs_pressure = Cx[:, dim]                                 # continuity rows

    # C @ x is +delta*nu*(grad q x n, curl u); the PPE source is the negative.
    assert np.allclose(lhs_pressure, -src_free, atol=1e-12, rtol=1e-9), (
        f"BVS monolithic C-block and projection source disagree: "
        f"max|C@x + src|={np.abs(lhs_pressure + src_free).max():.3e}")


def test_bvs_load_bearing_and_irrotational_null():
    """ANTI-VACUITY. The term is NON-ZERO on a field with outflow vorticity and
    EXACTLY ZERO on an irrotational field (curl u == 0), as the vorticity-flux
    form demands. A shear field u=(y,0) has curl = -1 (non-zero); a uniform field
    u=(1,0) has curl = 0."""
    fx = _fx()
    dm, nu = fx["dm"], fx["nu"]
    dim = dm.dim
    dt = 0.01
    faces = outflow_faces(dm.mesh)
    coords = fx["coords"]                    # free-node coords [n_free, dim]

    # sheared field u = (y, 0): curl = du_y/dx - du_x/dy = -1 (non-zero).
    u_shear = np.zeros((dm.n_free, dim))
    u_shear[:, 0] = coords[:, 1]
    src_shear = bvs_ppe_source(dm, u_shear, nu, dt, faces=faces)
    assert np.linalg.norm(src_shear) > 1e-8, (
        "BVS source vanished on a sheared (rotational) field — term is not "
        "load-bearing")

    # uniform field u = (1, 0): curl == 0 -> source must be machine zero.
    u_unif = np.zeros((dm.n_free, dim))
    u_unif[:, 0] = 1.0
    src_unif = bvs_ppe_source(dm, u_unif, nu, dt, faces=faces)
    assert np.linalg.norm(src_unif) < 1e-12, (
        f"BVS source non-zero on an irrotational field (curl u == 0): "
        f"‖src‖={np.linalg.norm(src_unif):.3e}")
