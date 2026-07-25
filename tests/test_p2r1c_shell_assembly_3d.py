"""P2-R1c — 3-D thin-shell ASSEMBLY validation (T1a + T1b).

T1a: sbm_vector_dirichlet_twosided + surrogate_traction run without error in
dim=3 on a small uniform octree (level 4, 4096 cells). A planar shell at
x=0.5 blocks the flow: downstream through-flow -> 0, the two-sided coupling
is load-bearing (drop one side -> force collapses), and surrogate_traction
returns a 3-component force vector with dominant x-component.

T1b: shell surrogate + AMR hanging-node constraints compose without error.
A 3-D octree refined near x=0.5 produces hanging nodes; the shell
classification, two-sided extraction, and T-reduction all complete on the
graded mesh.

Spec: docs/superpowers/specs/2026-07-25-p2r1-thinshell-cfd-spec.md R1c T1a/T1b.
"""
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from diffsim.octree.build import build_uniform, refine_elements
from diffsim.octree.balance import balance2to1
from diffsim.octree import morton
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.csg import Plane
from diffsim.sbm.surrogate import (
    classify_shell_intercepted, extract_two_sided_surrogate)
from diffsim.sbm.vector import (
    sbm_vector_dirichlet, sbm_vector_dirichlet_twosided, surrogate_traction)
from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.physics.poisson import gauss_points

pytestmark = pytest.mark.tier5

PLATE_X = 0.5
LEVEL = 4          # 4096 cells uniform — completes in ~20-60s on warp-CPU
U = 1.0
NU = 0.05
ALPHA = 30.0
DT = 0.1
STEPS = 30         # pseudo-transient steps (early-exit on convergence)


# ---------------------------------------------------------------------------
# Shared setup helpers
# ---------------------------------------------------------------------------

def _build_3d_shell(level=LEVEL):
    """Build dim=3 uniform octree + two-sided shell surrogate at x=PLATE_X."""
    pl = Plane((PLATE_X, 0.0, 0.0), (1.0, 0.0, 0.0))
    tree = build_uniform(level, dim=3)
    ret, intercepted = classify_shell_intercepted(tree, pl)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), "cpu")
    ftab = face_tables(1, 3)
    (sfp, gp_geo), (sfm, gm_geo) = extract_two_sided_surrogate(ret, pl, ftab)
    return dict(dm=dm, mesh=mesh, cons=cons, pl=pl,
                sfp=sfp, gp=gp_geo, sfm=sfm, gm=gm_geo,
                n_excluded=int(intercepted.sum()))


def _outer_bc_3d(mesh, cons, ndof, dim, U):
    """Strong outer BCs: inflow u=(U,0,0) at x=0, no-slip on y/z walls.

    Returns (rows, vals, coords) in free-node space, matching the 2-D driver
    _outer_bc pattern lifted to 3-D.
    """
    coords = mesh.node_coords[cons.free_nodes]
    on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
    inflow = on(0.0, 0)
    walls = (on(0.0, 1) | on(1.0, 1) | on(0.0, 2) | on(1.0, 2))
    rows, vals = [], []
    for i in np.where(inflow | walls)[0]:
        gx = U if (inflow[i] and not walls[i]) else 0.0
        rows.append(i * ndof + 0); vals.append(gx)   # u_x
        rows.append(i * ndof + 1); vals.append(0.0)  # u_y
        rows.append(i * ndof + 2); vals.append(0.0)  # u_z
    return np.asarray(rows, np.int64), np.asarray(vals), coords


def _gp_field_3d(dm, mesh, T, u_node, dim):
    """Gauss-point advecting velocity + divergence for the linearized NS brick.

    Mirrors the 2-D driver _gp_field, lifted to dim=3 (and to any ndof).
    u_node: [n_free, dim] velocity in the free-node space.
    Returns aq (dict p->array [n_elem*nqp, dim]) and dq (dict p->array [n_elem*nqp]).
    """
    full = np.asarray(T @ u_node)            # [n_nodes, dim]
    aq, dq = {}, {}
    for pv in dm.bins:
        tb = dm.tables_by_p[pv]
        eids = dm.bins[pv]["eids"]           # element global ids in the tree
        vals = full[mesh.conn_of[pv]]        # [ne, nbf, dim]
        # velocity at GPs: einsum "qa,ead->eqd" with N[qa]=shape[nqp,nbf]
        aq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
        h = mesh.tree.h()[eids]
        # divergence proxy: einsum "qad,ead->eq" * (2/h) physical gradient scale
        dq[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                  * (2.0 / h)[:, None]).reshape(-1)
    return aq, dq


def _march_3d(dm, mesh, cons, face_fn, nu, dt, steps, U, dim, ndof):
    """Pseudo-transient monolithic march to steady (3-D analogue of _march).

    face_fn() -> (Af_c, bf_c): the constrained SBM face assembly.
    Pressure pinned at the node furthest downstream-low (outflow corner proxy).
    """
    T = cons.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    nfree = T.shape[1]
    rows, vals, coords = _outer_bc_3d(mesh, cons, ndof, dim, U)
    # pressure pin: max(x - y - z) in free-node space (outflow-low-back corner)
    p_pin = int(np.argmax(coords[:, 0] - coords[:, 1] - coords[:, 2]))
    xq = gauss_points(mesh, dm.tables_by_p)
    Af_c, bf_c = face_fn()
    x = np.zeros(nfree * ndof)
    sigma = 1.0 / dt
    prev_u = None
    for step in range(steps):
        u_node = x.reshape(nfree, ndof)[:, :dim]
        aq, dq = _gp_field_3d(dm, mesh, T, u_node, dim)
        fq = {pv: aq[pv] / dt for pv in xq}
        A, b = assemble_linear_ns(dm, aq, dq, fq, nu, sigma=sigma)
        A = (A + Af_c).tolil()
        b = b + bf_c
        for r, v in zip(rows, vals):
            A.rows[r] = [int(r)]; A.data[r] = [1.0]; b[r] = v
        pr = p_pin * ndof + dim
        A.rows[pr] = [pr]; A.data[pr] = [1.0]; b[pr] = 0.0
        x = splu(A.tocsr().tocsc()).solve(b)
        u_new = x.reshape(nfree, ndof)[:, :dim]
        if prev_u is not None and step > 5:
            if np.abs(u_new - prev_u).max() / dt < 1e-4:
                break
        prev_u = u_new.copy()
    u = x.reshape(nfree, ndof)[:, :dim]
    p = x.reshape(nfree, ndof)[:, dim]
    return x, u, p, coords, step + 1


def _through_flux_3d(u, coords, x_line, tol=0.05):
    """Mean |u_x| sampled on nodes near x=x_line (blockage proxy)."""
    band = np.abs(coords[:, 0] - x_line) < tol
    if not band.any():
        return float("nan")
    return float(np.abs(u[band, 0]).mean())


# ---------------------------------------------------------------------------
# T1a — 3-D uniform shell assembly + correctness
# ---------------------------------------------------------------------------

def test_t1a_3d_shell_assembly_runs_and_correct():
    """T1a: 3-D two-sided shell assembly runs + is correct.

    Spec: docs/superpowers/specs/2026-07-25-p2r1-thinshell-cfd-spec.md §R1c T1a.

    Asserts:
    (1) Assembly runs with dim=3 (no shape/dim error, no NaN in A/b).
    (2) surrogate_traction returns a finite 3-component force vector with the
        plate-normal (x) component dominant and in-plane (y, z) components ~0
        by symmetry.
    (3) Complete blockage: downstream through-flow < 5% of U (the 3-D analogue
        of the 2-D test_p2r1_thin_plate.py blockage assertion).
    (4) Two-sided coupling is LOAD-BEARING: dropping one side Gamma~+ -> net
        plate force collapses by > 50% (the 3-D analogue of the 2-D
        'loadbearing' verdict).
    """
    dim = 3
    ndof = dim + 1   # 4: (u_x, u_y, u_z, p)

    fx = _build_3d_shell(LEVEL)
    dm, mesh, cons = fx["dm"], fx["mesh"], fx["cons"]
    sfp, gp_geo = fx["sfp"], fx["gp"]
    sfm, gm_geo = fx["sfm"], fx["gm"]

    T = cons.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    noslip = lambda y: np.zeros((len(y), dim))

    # (1) assembly runs without error and produces finite entries
    Af, bf = sbm_vector_dirichlet_twosided(
        dm, sfp, gp_geo, sfm, gm_geo, noslip, NU, ndof, alpha=ALPHA)
    assert Af.shape == (dm.n_nodes * ndof, dm.n_nodes * ndof), (
        f"Wrong Af shape: {Af.shape}")
    assert np.isfinite(Af.data).all(), "NaN/Inf in two-sided Af matrix entries"
    assert np.isfinite(bf).all(), "NaN/Inf in two-sided bf RHS vector"

    # Full monolithic pseudo-transient solve
    def shell_face():
        A_f, b_f = sbm_vector_dirichlet_twosided(
            dm, sfp, gp_geo, sfm, gm_geo, noslip, NU, ndof, alpha=ALPHA)
        return (T_vec.T @ A_f @ T_vec).tocsr(), np.asarray(T_vec.T @ b_f)

    x, u, p, coords, nst = _march_3d(dm, mesh, cons, shell_face,
                                      NU, DT, STEPS, U, dim, ndof)
    assert np.isfinite(u).all(), "NaN in velocity after 3-D solve"
    assert np.isfinite(p).all(), "NaN in pressure after 3-D solve"

    xf = np.asarray(T_vec @ x)   # expand to full node-major DOFs

    # (2) surrogate_traction: 3-component force vector, x dominant
    Fp = surrogate_traction(dm, sfp, gp_geo, xf, NU, ndof)
    Fm = surrogate_traction(dm, sfm, gm_geo, xf, NU, ndof)
    assert Fp.shape == (3,), f"Force vector must be 3-D, got {Fp.shape}"
    assert Fm.shape == (3,), f"Force vector must be 3-D, got {Fm.shape}"
    assert np.isfinite(Fp).all() and np.isfinite(Fm).all(), \
        f"Non-finite force: Fp={Fp} Fm={Fm}"
    F_net = Fp + Fm
    # plate-normal (x) component must dominate both in-plane (y, z) components
    assert abs(F_net[0]) > abs(F_net[1]) + 1e-6, (
        f"x-force not dominant over y: F_net={F_net}")
    assert abs(F_net[0]) > abs(F_net[2]) + 1e-6, (
        f"x-force not dominant over z: F_net={F_net}")
    print(f"\n[T1a] 3-D force: Fp={Fp} Fm={Fm} F_net={F_net}")

    # (3) complete blockage: downstream through-flow < 5% of U
    u_down = _through_flux_3d(u, coords, 0.72)
    u_up = _through_flux_3d(u, coords, 0.28)
    print(f"[T1a] u_up={u_up:.4e}  u_down={u_down:.4e}  nst={nst}")
    assert u_down < 0.05 * U, (
        f"Downstream through-flow not blocked in 3-D: u_down={u_down:.4e} "
        f"(threshold={0.05*U:.4e})")

    # (4) load-bearing: drop Gamma~+ -> solve with ONLY Gamma~-
    def oneside_face():
        A_f, b_f = sbm_vector_dirichlet(dm, sfm, gm_geo, noslip, NU, ndof,
                                        alpha=ALPHA)
        return (T_vec.T @ A_f @ T_vec).tocsr(), np.asarray(T_vec.T @ b_f)

    x1, u1, p1, coords1, _ = _march_3d(dm, mesh, cons, oneside_face,
                                        NU, DT, STEPS, U, dim, ndof)
    xf1 = np.asarray(T_vec @ x1)
    F1p = surrogate_traction(dm, sfp, gp_geo, xf1, NU, ndof)
    F1m = surrogate_traction(dm, sfm, gm_geo, xf1, NU, ndof)
    F1_net = F1p + F1m
    print(f"[T1a] load-bearing: two-sided F_x={F_net[0]:.4f}  "
          f"one-sided F_x={F1_net[0]:.4f}")
    # dropping the loaded side must collapse net force by > 50%
    assert abs(F1_net[0]) < 0.5 * abs(F_net[0]) + 1e-6, (
        f"3-D load-bearing check failed: one-sided F_x={F1_net[0]:.4f} is "
        f"NOT < 50% of two-sided F_x={F_net[0]:.4f}")


# ---------------------------------------------------------------------------
# T1b — 3-D adaptive/graded mesh + hanging-node constraints
# ---------------------------------------------------------------------------

def test_t1b_3d_shell_adaptive_hanging_nodes():
    """T1b: shell surrogate + adaptive/graded 3-D mesh + hanging-node constraints.

    Spec: docs/superpowers/specs/2026-07-25-p2r1-thinshell-cfd-spec.md §R1c T1b.

    Builds a 3-D octree with cells near x=0.5 refined one extra level, applies
    balance2to1 (guaranteed to produce hanging nodes when a uniform L3 base is
    locally refined to L4 near x=0.5), then:
    (1) Confirms hanging nodes exist (cons.hanging.any()).
    (2) classify_shell_intercepted + extract_two_sided_surrogate run on the
        graded mesh without error, both sides non-empty.
    (3) sbm_vector_dirichlet_twosided assembles + T-reduces to a finite
        free-dof matrix of the right shape.
    """
    dim = 3
    ndof = dim + 1
    BASE_LEVEL = 3   # 512 cells uniform; local refinement -> ~700-1000 cells

    pl = Plane((PLATE_X, 0.0, 0.0), (1.0, 0.0, 0.0))
    tree = build_uniform(BASE_LEVEL, dim=3)

    # Refine cells near x=0.5 (center of the shell band) by one extra level.
    # Morton integers -> [0,1] physical: scale = 2^{-lmax(3)} = 2^{-21}.
    scale = 2.0 ** -morton.lmax(3)
    a = tree.anchors() * scale
    h = tree.h()
    near_plate = np.abs(a[:, 0] + h / 2 - PLATE_X) < 2 * h.max()
    tree_ref = refine_elements(tree, near_plate)
    tree_bal = balance2to1(tree_ref)

    # (1) shell classification on the graded tree
    ret, intercepted = classify_shell_intercepted(tree_bal, pl)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    assert cons.hanging.any(), (
        "Expected hanging nodes after local refinement near x=0.5 + balance2to1, "
        "but none found — the refinement may not have produced a graded mesh.")

    # (2) shell extraction runs without error; both sides non-empty
    ftab = face_tables(1, 3)
    (sfp, gp_geo), (sfm, gm_geo) = extract_two_sided_surrogate(ret, pl, ftab)
    assert len(sfp.elem) > 0, "Gamma~+ side is empty on adaptive mesh"
    assert len(sfm.elem) > 0, "Gamma~- side is empty on adaptive mesh"
    n_hanging = int(cons.hanging.sum())
    print(f"\n[T1b] adaptive mesh: {len(tree_bal)} cells, "
          f"{n_hanging} hanging nodes, "
          f"sfp={len(sfp.elem)} sfm={len(sfm.elem)} faces")

    # (3) assembly + T-reduction: matrix finite, right shape
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), "cpu")
    T = cons.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    noslip = lambda y: np.zeros((len(y), dim))
    Af, bf = sbm_vector_dirichlet_twosided(
        dm, sfp, gp_geo, sfm, gm_geo, noslip, NU, ndof, alpha=ALPHA)
    assert np.isfinite(Af.data).all(), "NaN/Inf in Af on adaptive 3-D mesh"
    assert np.isfinite(bf).all(), "NaN/Inf in bf on adaptive 3-D mesh"

    Af_free = (T_vec.T @ Af @ T_vec).tocsr()
    nfree = T.shape[1]
    assert Af_free.shape == (nfree * ndof, nfree * ndof), (
        f"Wrong Af_free shape: {Af_free.shape}")
    assert np.isfinite(Af_free.data).all(), "NaN/Inf in free-dof Af on adaptive mesh"
    print(f"[T1b] Af_free: {Af_free.shape}, nnz={Af_free.nnz}")
