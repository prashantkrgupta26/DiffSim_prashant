"""P2-R1c — 3-D flow PAST a finite thin plate: transient BDF2 Cd/Cl_y/Cl_z history.

3-D lift of tests/p2r1a_thin_plate_flow.py. The plate is a finite rectangular
patch in 3-D (FiniteSheet), normal to the flow (x-direction). The domain is a
unit cube [0,1]^3 (the octree's native domain with dim=3).

Plate (smoke default): center (0.375, 0.5, 0.5), half-widths (0.125, 0.125)
in the y-z plane, normal = (1, 0, 0). This is a 0.25 x 0.25 square patch
in the y-z cross-section.

Two-oracle strategy for the finite sheet:
  - `FiniteSheet` for BOTH `classify_shell_intercepted` AND
    `extract_two_sided_surrogate`. Unlike the 2-D driver which needed a
    Segment (unsigned psi) for classification and a Plane (signed) for
    extraction, FiniteSheet.distance_vector returns the fixed patch normal
    as n_grad for ALL surrogate GPs — the shell pipeline uses this to split
    Gamma~+/Gamma~- without needing a separate Plane oracle.

Flow-past BCs in 3-D:
  - Inflow (x=0): u = (U_inf, 0, 0)
  - 4 lateral walls (y=0, y=1, z=0, z=1): u = (U_inf, 0, 0) (freestream-like)
  - Outflow (x=1): do-nothing (no Dirichlet imposed)
  - Pressure pin at the outflow-low-back corner (max x, min y, min z free node)

BDF2 march: BDF1 bootstrap (step 0), then BDF2 (steps 1..nsteps-1).
sigma = b0/dt is passed to `assemble_linear_ns`; history part is `fq` forcing.

Force recovery:
    Cd   = F_x / (0.5 * U_inf**2 * plate_area)  [streamwise, should dominate]
    Cl_y = F_y / (0.5 * U_inf**2 * plate_area)  [transverse y, ~0 by symmetry]
    Cl_z = F_z / (0.5 * U_inf**2 * plate_area)  [transverse z, ~0 by symmetry]

where plate_area = 4 * half[0] * half[1] (area of the rectangular patch).

Quick smoke run:
    .venv/bin/python tests/p2r1c_thin_plate_flow_3d.py
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

sys.path.insert(0, os.path.dirname(__file__))

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.csg import FiniteSheet
from diffsim.sbm.surrogate import (
    classify_shell_intercepted, extract_two_sided_surrogate)
from diffsim.sbm.vector import (
    sbm_vector_dirichlet, sbm_vector_dirichlet_twosided, surrogate_traction)
from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.physics.poisson import gauss_points
from diffsim.solvers.timestepping import bdf_coeffs


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _make_sheet(x_c, y_c, z_c, half_y, half_z):
    """Return a FiniteSheet for a rectangular plate at x=x_c spanning
    [y_c-half_y, y_c+half_y] x [z_c-half_z, z_c+half_z], normal = +x.

    FiniteSheet works for BOTH classify_shell_intercepted (uses psi, which is
    unsigned distance) and extract_two_sided_surrogate (distance_vector returns
    the fixed patch normal as n_grad for all surrogate GPs — the shell pipeline
    then splits by sign(n_tilde . n_grad), giving Gamma~+ and Gamma~-).
    """
    return FiniteSheet(
        center=(x_c, y_c, z_c),
        half=(half_y, half_z),
        normal=(1.0, 0.0, 0.0),
    )


def _build_shell_3d(level, x_c, y_c, z_c, half_y, half_z):
    """Build dim=3 uniform octree + two-sided shell surrogate for the finite sheet.

    Returns a dict with dm, mesh, cons, sfp, gp, sfm, gm, n_excluded.
    """
    sheet = _make_sheet(x_c, y_c, z_c, half_y, half_z)
    tree = build_uniform(level, dim=3)
    ret, intercepted = classify_shell_intercepted(tree, sheet)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), "cpu")
    ftab = face_tables(1, 3)
    (sfp, gp), (sfm, gm) = extract_two_sided_surrogate(ret, sheet, ftab)
    return dict(dm=dm, mesh=mesh, cons=cons, sfp=sfp, gp=gp, sfm=sfm, gm=gm,
                n_excluded=int(intercepted.sum()))


def triangulate_finite_sheet(x_c, y_c, z_c, half_y, half_z):
    """Return (verts, tris) for the finite rectangular plate surface.

    The plate is a rectangle in the y-z plane at x=x_c.
    Two triangles cover the four corners.
    verts: float64 array [4, 3]
    tris:  int64 array [2, 3]
    """
    y0, y1 = y_c - half_y, y_c + half_y
    z0, z1 = z_c - half_z, z_c + half_z
    verts = np.array([
        [x_c, y0, z0],
        [x_c, y1, z0],
        [x_c, y1, z1],
        [x_c, y0, z1],
    ], dtype=np.float64)
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return verts, tris


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

def _outer_bc_3d(mesh, cons, ndof, dim, U_inf):
    """Strong outer BCs for 3-D flow past a plate:
      - inflow (x=0): u = (U_inf, 0, 0)
      - 4 lateral walls (y=0, y=1, z=0, z=1): u = (U_inf, 0, 0) [freestream]
      - outflow (x=x_max): do-nothing (not constrained here)

    Returns (rows, vals) in free-node DOF space.
    """
    coords = mesh.node_coords[cons.free_nodes]
    tol = 1e-10
    x_min = coords[:, 0].min()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()
    z_min = coords[:, 2].min()
    z_max = coords[:, 2].max()
    inflow = np.abs(coords[:, 0] - x_min) < tol
    walls = (
        (np.abs(coords[:, 1] - y_min) < tol) |
        (np.abs(coords[:, 1] - y_max) < tol) |
        (np.abs(coords[:, 2] - z_min) < tol) |
        (np.abs(coords[:, 2] - z_max) < tol)
    )
    forced = inflow | walls
    rows, vals = [], []
    for i in np.where(forced)[0]:
        rows.append(i * ndof + 0); vals.append(U_inf)   # u_x = U_inf
        rows.append(i * ndof + 1); vals.append(0.0)     # u_y = 0
        rows.append(i * ndof + 2); vals.append(0.0)     # u_z = 0
    return np.asarray(rows, np.int64), np.asarray(vals)


def _pressure_pin_3d(mesh, cons, ndof, dim):
    """Find outflow-low-back corner free node (max x - y - z) and return its
    pressure DOF index (to pin p=0 there)."""
    coords = mesh.node_coords[cons.free_nodes]
    corner = np.argmax(coords[:, 0] - coords[:, 1] - coords[:, 2])
    return int(corner * ndof + dim)   # pressure DOF


# ---------------------------------------------------------------------------
# Gauss-point field helpers
# ---------------------------------------------------------------------------

def _gp_field_3d(dm, mesh, T, u_node, dim):
    """Interpolate free-space velocity u_node [nfree, dim] to Gauss points.
    Returns (aq_by_bin, div_aq_by_bin) dicts. Mirrors _gp_field_3d in
    test_p2r1c_shell_assembly_3d.py."""
    full = np.asarray(T @ u_node)   # [n_nodes, dim]
    aq, dq = {}, {}
    for pv in dm.bins:
        tb = dm.tables_by_p[pv]
        eids = dm.bins[pv]["eids"]
        vals = full[mesh.conn_of[pv]]                     # [ne, nbf, dim]
        aq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
        h = mesh.tree.h()[eids]
        dq[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                  * (2.0 / h)[:, None]).reshape(-1)
    return aq, dq


def _gp_history_fq_3d(dm, mesh, T, u_pre1, u_pre2, b1, b2, dt, dim):
    """Build GP body-force field for BDF2 history RHS:
      fq = (-b1 * u^n - b2 * u^{n-1}) / dt
    Mirrors _gp_history_fq in the 2-D driver."""
    full1 = np.asarray(T @ u_pre1)
    full2 = np.asarray(T @ u_pre2)
    full_hist = (-b1 * full1 - b2 * full2) / dt
    fq = {}
    for pv in dm.bins:
        tb = dm.tables_by_p[pv]
        vals = full_hist[mesh.conn_of[pv]]
        fq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
    return fq


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_flow_past_3d(
    level=4,
    nsteps=5,
    dt=0.01,
    U_inf=1.0,
    nu=0.1,
    alpha=50.0,
    plate_xc=0.375,
    plate_yc=0.5,
    plate_zc=0.5,
    plate_half_y=0.125,
    plate_half_z=0.125,
    verbose=False,
    _two_sided=True,
    _return_fields=False,
):
    """Run 3-D flow past a finite thin plate with transient BDF2 march.

    Parameters
    ----------
    level : int
        Octree refinement level (each direction). Level 4 = 16^3 = 4096 cells.
    nsteps : int
        Number of BDF2 steps.
    dt : float
        Time step.
    U_inf : float
        Freestream velocity.
    nu : float
        Kinematic viscosity.
    alpha : float
        SBM penalty parameter.
    plate_xc, plate_yc, plate_zc : float
        Plate center in [0,1]^3.
    plate_half_y, plate_half_z : float
        In-plane half-widths of the rectangular plate.
    verbose : bool
        Print per-step Cd/Cl_y/Cl_z.
    _two_sided : bool
        Internal flag — False => one-sided (anti-vacuity test).
    _return_fields : bool
        Internal flag — True => also return mesh + node fields dict.

    Returns a dict with:
      'cd'         : np.ndarray [nsteps] — drag coefficient (x-direction)
      'cl_y'       : np.ndarray [nsteps] — transverse y force coefficient
      'cl_z'       : np.ndarray [nsteps] — transverse z force coefficient
      'n_excluded' : int — excluded cells (non-zero confirms plate active)
      'nsteps'     : int — steps taken

    When _return_fields=True, also returns 'mesh' and 'node_fields' dict with
    'velocity_magnitude' [Nn] and 'pressure' [Nn].
    """
    dim = 3
    ndof = dim + 1   # 4: (u_x, u_y, u_z, p)
    t0 = time.time()

    # ---- geometry + mesh ----------------------------------------------------
    fx = _build_shell_3d(level, plate_xc, plate_yc, plate_zc,
                         plate_half_y, plate_half_z)
    dm, mesh, cons = fx["dm"], fx["mesh"], fx["cons"]

    if verbose:
        print(f"[p2r1c-3d] level={level}  n_excluded={fx['n_excluded']}  "
              f"sfp={fx['sfp'].elem.size}  sfm={fx['sfm'].elem.size}  "
              f"nsteps={nsteps}  dt={dt}  nu={nu}", flush=True)

    T = cons.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    nfree = T.shape[1]

    # ---- BCs ----------------------------------------------------------------
    bc_rows, bc_vals = _outer_bc_3d(mesh, cons, ndof, dim, U_inf)
    p_pin = _pressure_pin_3d(mesh, cons, ndof, dim)

    # ---- SBM face terms (pre-assembled; geometry is fixed) ------------------
    noslip = lambda y: np.zeros((len(y), dim))
    if _two_sided:
        Af_raw, bf_raw = sbm_vector_dirichlet_twosided(
            dm, fx["sfp"], fx["gp"], fx["sfm"], fx["gm"],
            noslip, nu, ndof, alpha=alpha)
    else:
        Af_raw, bf_raw = sbm_vector_dirichlet(
            dm, fx["sfm"], fx["gm"], noslip, nu, ndof, alpha=alpha)
    Af_c = (T_vec.T @ Af_raw @ T_vec).tocsr()
    bf_c = np.asarray(T_vec.T @ bf_raw)

    xq = gauss_points(mesh, dm.tables_by_p)

    # ---- Reference force normalization (plate area) -------------------------
    plate_area = 4.0 * plate_half_y * plate_half_z
    ref_force = 0.5 * U_inf ** 2 * plate_area

    # ---- BDF2 march ---------------------------------------------------------
    x_cur = np.zeros(nfree * ndof)
    u_pre2 = np.zeros((nfree, dim))
    u_pre1 = np.zeros((nfree, dim))

    cd_hist = np.zeros(nsteps)
    cl_y_hist = np.zeros(nsteps)
    cl_z_hist = np.zeros(nsteps)

    for step in range(nsteps):
        order = 1 if step == 0 else 2
        b0, b1, b2 = bdf_coeffs(order, dt)
        sigma = b0 / dt

        # Advecting velocity at Gauss points
        aq, dq = _gp_field_3d(dm, mesh, T, u_pre1, dim)

        # History forcing
        if order == 1:
            fq_raw = {}
            for pv in dm.bins:
                tb = dm.tables_by_p[pv]
                vals = np.asarray(T @ u_pre1)[mesh.conn_of[pv]]
                fq_raw[pv] = (np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
                              / dt)
        else:
            fq_raw = _gp_history_fq_3d(dm, mesh, T, u_pre1, u_pre2,
                                        b1, b2, dt, dim)

        # Assemble monolithic NS
        A, b = assemble_linear_ns(dm, aq, dq, fq_raw, nu, sigma=sigma)
        A = (A + Af_c).tolil()
        b = b + bf_c

        # Apply strong Dirichlet BCs
        for r, v in zip(bc_rows, bc_vals):
            A.rows[r] = [int(r)]; A.data[r] = [1.0]; b[r] = v

        # Pressure pin
        A.rows[p_pin] = [p_pin]; A.data[p_pin] = [1.0]; b[p_pin] = 0.0

        # Solve
        x_cur = splu(A.tocsr().tocsc()).solve(b)

        # Extract velocity for next step
        u_new = x_cur.reshape(nfree, ndof)[:, :dim]

        # Full node-major vector for traction
        x_all = np.asarray(T_vec @ x_cur)

        # Force recovery (two-sided: F+ + F-)
        if _two_sided:
            Fp = surrogate_traction(dm, fx["sfp"], fx["gp"], x_all, nu, ndof)
            Fm = surrogate_traction(dm, fx["sfm"], fx["gm"], x_all, nu, ndof)
            F = Fp + Fm
        else:
            Fm = surrogate_traction(dm, fx["sfm"], fx["gm"], x_all, nu, ndof)
            F = Fm

        cd_hist[step]   = F[0] / ref_force   # drag (streamwise = x)
        cl_y_hist[step] = F[1] / ref_force   # transverse y
        cl_z_hist[step] = F[2] / ref_force   # transverse z

        # Rotate history
        u_pre2 = u_pre1.copy()
        u_pre1 = u_new.copy()

        if verbose:
            print(f"[p2r1c-3d] step {step:3d}  Cd={cd_hist[step]:+.4f}  "
                  f"Cl_y={cl_y_hist[step]:+.4f}  Cl_z={cl_z_hist[step]:+.4f}",
                  flush=True)

    elapsed = time.time() - t0
    if verbose:
        print(f"[p2r1c-3d] done in {elapsed:.1f}s  "
              f"Cd[-1]={cd_hist[-1]:+.4f}", flush=True)

    result = dict(
        cd=cd_hist,
        cl_y=cl_y_hist,
        cl_z=cl_z_hist,
        n_excluded=fx["n_excluded"],
        nsteps=nsteps,
    )

    if _return_fields:
        x_nodes = np.asarray(x_all).reshape(-1, ndof)
        u_node = x_nodes[:, :dim]
        p_node = x_nodes[:, dim]
        vel_mag = np.linalg.norm(u_node, axis=1)
        result["mesh"] = mesh
        result["node_fields"] = {
            "velocity_magnitude": vel_mag,
            "pressure": p_node,
        }

    return result


def run_flow_past_3d_one_sided(**kwargs):
    """Anti-vacuity: same setup but only Gamma~- assembled (drop Gamma~+).
    Used by the smoke gate to verify two-sided coupling is load-bearing."""
    kwargs["_two_sided"] = False
    return run_flow_past_3d(**kwargs)


# ---------------------------------------------------------------------------
# GH200 hero-run configuration (does NOT run in CI)
# ---------------------------------------------------------------------------
# Level 6 (64^3 = 262144 cells) + 200 steps — a real flow run for the GH200.
# Physical domain: unit cube [0,1]^3; plate at (0.375, 0.5, 0.5), 0.25x0.25.
# Re = U_inf * L_eff / nu where L_eff ~ sqrt(plate_area) = 0.25.
# Set nu=0.004 for Re~62.5 (laminar but convective — interesting wake in 3-D).
GH200_CONFIG = dict(
    level=6,
    nsteps=200,
    dt=0.005,
    U_inf=1.0,
    nu=0.004,
    alpha=50.0,
    plate_xc=0.375,
    plate_yc=0.5,
    plate_zc=0.5,
    plate_half_y=0.125,
    plate_half_z=0.125,
)


if __name__ == "__main__":
    from diffsim.viz.results import save_flow_run

    level        = int(os.environ.get("LEVEL", "4"))
    nsteps       = int(os.environ.get("NSTEPS", "5"))
    dt           = float(os.environ.get("DT", "0.01"))
    nu           = float(os.environ.get("NU", "0.1"))
    U_inf        = float(os.environ.get("U_INF", "1.0"))
    plate_xc     = float(os.environ.get("PLATE_XC", "0.375"))
    plate_yc     = float(os.environ.get("PLATE_YC", "0.5"))
    plate_zc     = float(os.environ.get("PLATE_ZC", "0.5"))
    plate_half_y = float(os.environ.get("PLATE_HALF_Y", "0.125"))
    plate_half_z = float(os.environ.get("PLATE_HALF_Z", "0.125"))

    re_approx = int(round(U_inf / nu)) if nu > 0 else 0
    case_name = f"p2r1c_3d_re{re_approx}_L{level}"

    res = run_flow_past_3d(
        level=level, nsteps=nsteps, dt=dt, nu=nu, U_inf=U_inf,
        plate_xc=plate_xc, plate_yc=plate_yc, plate_zc=plate_zc,
        plate_half_y=plate_half_y, plate_half_z=plate_half_z,
        verbose=True,
        _return_fields=True,
    )
    print(f"Cd={res['cd']}")
    print(f"Cl_y={res['cl_y']}")
    print(f"Cl_z={res['cl_z']}")

    t_arr = np.arange(1, nsteps + 1) * dt

    # Body: triangulate the finite rectangular plate (real 3-D surface mesh)
    body = triangulate_finite_sheet(plate_xc, plate_yc, plate_zc,
                                    plate_half_y, plate_half_z)

    written = save_flow_run(
        case_name,
        mesh=res.get("mesh"),
        node_fields=res.get("node_fields"),
        histories={
            "Cd":   (t_arr, res["cd"]),
            "Cl_y": (t_arr, res["cl_y"]),
            "Cl_z": (t_arr, res["cl_z"]),
        },
        body=body,
    )
    if written:
        print(f"[p2r1c-3d] Results written: {list(written.values())}")
    else:
        print("[p2r1c-3d] No results written (viz deps missing or export failed)")
