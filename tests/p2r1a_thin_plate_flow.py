"""P2-R1a — 2-D flow PAST a finite thin plate: transient BDF2 Cd/Cl history.

Flow past a vertical finite plate of length L centered at (x_c, y_c) in a
rectangular domain.  The plate is a FINITE line segment in the unit-square
domain [0,1]^2.

Two-oracle strategy for the finite plate:
  - `Segment(a, b)` for `classify_shell_intercepted` (exclusion of cells
    cut by the FINITE segment — the unsigned-psi fast path, already supported).
  - `Plane((xc, 0), (1, 0))` for `extract_two_sided_surrogate` (Newton
    projection needs a SIGNED psi; the Plane's signed distance gives correct
    normals ±x for the excluded band cells that the Segment classified).
    This is correct because for all excluded-band surrogate GPs (which are
    within O(h) of x=xc and within the plate's y-range), the Plane's normals
    are identical to the Segment's normals (both are ±x).

The two-sided SBM shell (`sbm_vector_dirichlet_twosided`) imposes no-slip on
both plate faces; `surrogate_traction` recovers force each step.

Transient march: BDF1 bootstrap (step 0), then BDF2 (steps 1..nsteps-1).
sigma = b0/dt is passed to `assemble_linear_ns`; the history part
(b1*u_n + b2*u_{n-1})/dt is the `fq` forcing field.

Drag/lift nondimensionalised:
    Cd = F_x / (0.5 * U_inf**2 * L)
    Cl = F_y / (0.5 * U_inf**2 * L)

Domain: unit square [0,1]^2 (the octree's native domain).
Plate (smoke default): vertical segment centered at (0.375, 0.5), L=0.25,
  i.e. from (0.375, 0.375) to (0.375, 0.625).

Quick smoke run:
    .venv/bin/python tests/p2r1a_thin_plate_flow.py
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
from diffsim.geometry.csg import Segment, Plane
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

def _make_plate(x_c, y_c, L):
    """Return (Segment, Plane) for a vertical finite plate of length L
    centered at (x_c, y_c).

    - Segment: used for `classify_shell_intercepted` (handles unsigned psi
      via the 'min-distance < reach' criterion — cells cut by the FINITE
      plate only).
    - Plane: used for `extract_two_sided_surrogate` (needs a SIGNED psi for
      Newton projection; for the excluded band's surrogate GPs the Plane and
      Segment normals are identical — both are ±x).
    """
    a = (x_c, y_c - L / 2.0)
    b = (x_c, y_c + L / 2.0)
    segment = Segment(a, b)
    plane = Plane((x_c, 0.0), (1.0, 0.0))  # same x, full-height (for normals)
    return segment, plane


def _build_shell(level, x_c, y_c, L, dim=2):
    """Build the two-sided shell surrogate for a finite vertical plate.

    Uses Segment for classification (finite plate extent) and Plane for
    surrogate face extraction (correct signed psi for Newton projection).
    Returns a dict with all mesh/shell data needed for the march."""
    segment, plane = _make_plate(x_c, y_c, L)
    tree = build_uniform(level, dim=dim)
    ret, intercepted = classify_shell_intercepted(tree, segment)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), "cpu")
    (sfp, gp), (sfm, gm) = extract_two_sided_surrogate(
        ret, plane, face_tables(1, dim))
    return dict(dm=dm, mesh=mesh, cons=cons, sfp=sfp, gp=gp, sfm=sfm, gm=gm,
                n_excluded=int(intercepted.sum()))


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

def _outer_bc(mesh, cons, ndof, dim, U_inf):
    """Strong outer BCs for flow past a plate:
      - inflow (x=0): u=(U_inf, 0)
      - top/bottom walls: u=(U_inf, 0)  [free-stream-like: u1=U_inf, u2=0]
      - outflow (x=x_max): do-nothing (not constrained here)
    Returns (rows, vals) in free-node DOF space."""
    coords = mesh.node_coords[cons.free_nodes]
    x_min = coords[:, 0].min()
    x_max = coords[:, 0].max()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()
    inflow = np.abs(coords[:, 0] - x_min) < 1e-10
    walls = (np.abs(coords[:, 1] - y_min) < 1e-10) | \
            (np.abs(coords[:, 1] - y_max) < 1e-10)
    forced = inflow | walls
    rows, vals = [], []
    for i in np.where(forced)[0]:
        rows.append(i * ndof + 0); vals.append(U_inf)   # u_x = U_inf
        rows.append(i * ndof + 1); vals.append(0.0)     # u_y = 0
    return np.asarray(rows, np.int64), np.asarray(vals)


def _pressure_pin(mesh, cons, ndof, dim):
    """Find the outflow-bottom-corner free node index and return its pressure
    DOF index (to pin p=0 there)."""
    coords = mesh.node_coords[cons.free_nodes]
    x_max = coords[:, 0].max()
    y_min = coords[:, 1].min()
    # outflow-bottom corner: maximise x, minimise y
    corner = np.argmax(coords[:, 0] - coords[:, 1])
    return int(corner * ndof + dim)   # pressure DOF


# ---------------------------------------------------------------------------
# Gauss-point field helpers (mirrors blocked-channel _gp_field)
# ---------------------------------------------------------------------------

def _gp_field(dm, mesh, T, u_node, dim):
    """Interpolate free-space velocity u_node [nfree, dim] to Gauss points.
    Returns (aq_by_bin, div_aq_by_bin) dicts."""
    full = np.asarray(T @ u_node)   # [n_nodes, dim]
    aq, dq = {}, {}
    for pv in dm.bins:
        tb = dm.tables_by_p[pv]
        vals = full[mesh.conn_of[pv]]                     # [ne, nbf, dim]
        aq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
        h = mesh.tree.h()[mesh.bins[pv]]
        dq[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                  * (2.0 / h)[:, None]).reshape(-1)
    return aq, dq


def _gp_history_fq(dm, mesh, T, u_pre1, u_pre2, b1, b2, dt, dim):
    """Build the body-force GP field for the BDF2 history RHS:
      fq = -(b1 * u^n + b2 * u^{n-1}) / dt   (negated because assemble_linear_ns
    puts sigma*u on the LHS and expects fq as the transient body-force on the RHS,
    i.e. the RHS contribution is +fq, and the BDF history contributes
    +(|b1| * u^n + |b2| * u^{n-1}) / dt with the signed BDF1/BDF2 tables
    where b1 < 0, b2 >= 0 for BDF2: {1.5, -2.0, 0.5}).

    The convention in _march (and blocked-channel): fq = u/dt for BDF1 (sigma=1/dt,
    fq acts as the u^n/dt source). For BDF2, generalizing:
      fq = (-b1 * u^n - b2 * u^{n-1}) / dt
    which equals (2*u^n - 0.5*u^{n-1})/dt for constant-step BDF2."""
    full1 = np.asarray(T @ u_pre1)   # [n_nodes, dim]
    full2 = np.asarray(T @ u_pre2)   # [n_nodes, dim]
    full_hist = (-b1 * full1 - b2 * full2) / dt   # [n_nodes, dim]
    fq = {}
    for pv in dm.bins:
        tb = dm.tables_by_p[pv]
        vals = full_hist[mesh.conn_of[pv]]
        fq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
    return fq


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_flow_past(
    level=5,
    nsteps=10,
    dt=0.01,
    U_inf=1.0,
    nu=0.1,
    alpha=50.0,
    plate_xc=0.375,
    plate_yc=0.5,
    plate_L=0.25,
    dim=2,
    verbose=False,
    _two_sided=True,   # internal flag: False => one-sided anti-vacuity test
):
    """Run flow past a finite thin plate with transient BDF2 march.

    Returns a dict with:
      'cd'       : np.ndarray [nsteps] — drag coefficient history
      'cl'       : np.ndarray [nsteps] — lift coefficient history
      'n_excluded': int — number of excluded octree cells (non-zero confirms plate active)
      'nsteps'   : int — number of steps actually taken
    """
    ndof = dim + 1
    t0 = time.time()

    # ---- geometry + mesh ----------------------------------------------------
    fx = _build_shell(level, plate_xc, plate_yc, plate_L, dim=dim)
    dm, mesh, cons = fx["dm"], fx["mesh"], fx["cons"]

    if verbose:
        print(f"[p2r1a] level={level}  n_excluded={fx['n_excluded']}  "
              f"sfp={fx['sfp'].elem.size}  sfm={fx['sfm'].elem.size}  "
              f"nsteps={nsteps}  dt={dt}  nu={nu}", flush=True)

    T = cons.T.tocsr()               # [n_nodes, nfree]
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    nfree = T.shape[1]

    # ---- BCs ----------------------------------------------------------------
    bc_rows, bc_vals = _outer_bc(mesh, cons, ndof, dim, U_inf)
    p_pin = _pressure_pin(mesh, cons, ndof, dim)

    # ---- SBM face terms (pre-assembled; geometry is fixed) ------------------
    noslip = lambda y: np.zeros((len(y), dim))
    if _two_sided:
        Af_raw, bf_raw = sbm_vector_dirichlet_twosided(
            dm, fx["sfp"], fx["gp"], fx["sfm"], fx["gm"],
            noslip, nu, ndof, alpha=alpha)
    else:
        # One-sided: only Gamma~- (drop Gamma~+, the upstream/loaded face)
        Af_raw, bf_raw = sbm_vector_dirichlet(
            dm, fx["sfm"], fx["gm"], noslip, nu, ndof, alpha=alpha)
    Af_c = (T_vec.T @ Af_raw @ T_vec).tocsr()
    bf_c = np.asarray(T_vec.T @ bf_raw)

    xq = gauss_points(mesh, dm.tables_by_p)

    # ---- BDF2 march ---------------------------------------------------------
    # Initialize: u=0 everywhere
    x_cur = np.zeros(nfree * ndof)
    u_pre2 = np.zeros((nfree, dim))    # u^{n-1} (only used for step >= 1)
    u_pre1 = np.zeros((nfree, dim))    # u^n

    cd_hist = np.zeros(nsteps)
    cl_hist = np.zeros(nsteps)

    ref_force = 0.5 * U_inf ** 2 * plate_L    # nondim denominator

    for step in range(nsteps):
        # BDF order: BDF1 for step 0 (bootstrap), BDF2 thereafter
        order = 1 if step == 0 else 2
        b0, b1, b2 = bdf_coeffs(order, dt)
        sigma = b0 / dt

        # Advecting velocity at Gauss points (linearization around u^n)
        aq, dq = _gp_field(dm, mesh, T, u_pre1, dim)

        # History forcing: -(b1 u^n + b2 u^{n-1})/dt
        if order == 1:
            # fq = u^n / dt  (equivalent to -(b1/dt)*u^n = 1/dt * u^n since b1=-1)
            fq_raw = {}
            for pv in dm.bins:
                tb = dm.tables_by_p[pv]
                vals = np.asarray(T @ u_pre1)[mesh.conn_of[pv]]
                fq_raw[pv] = (np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
                              / dt)
        else:
            fq_raw = _gp_history_fq(dm, mesh, T, u_pre1, u_pre2, b1, b2, dt, dim)

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

        cd_hist[step] = F[0] / ref_force   # drag (streamwise = x)
        cl_hist[step] = F[1] / ref_force   # lift (transverse = y)

        # Rotate history
        u_pre2 = u_pre1.copy()
        u_pre1 = u_new.copy()

        if verbose:
            print(f"[p2r1a] step {step:3d}  Cd={cd_hist[step]:+.4f}  "
                  f"Cl={cl_hist[step]:+.4f}", flush=True)

    elapsed = time.time() - t0
    if verbose:
        print(f"[p2r1a] done in {elapsed:.1f}s  "
              f"Cd[-1]={cd_hist[-1]:+.4f}  Cl[-1]={cl_hist[-1]:+.4f}", flush=True)

    return dict(
        cd=cd_hist,
        cl=cl_hist,
        n_excluded=fx["n_excluded"],
        nsteps=nsteps,
    )


def run_flow_past_one_sided(**kwargs):
    """Anti-vacuity run: same mesh/BCs but only Gamma~- assembled (drop Gamma~+).
    Used by the smoke gate to verify the two-sided coupling is load-bearing."""
    kwargs["_two_sided"] = False
    return run_flow_past(**kwargs)


# ---------------------------------------------------------------------------
# Re=250 full-resolution configuration (ThinShell.pdf §4.3)
# Run on gpubox ONLY — see docs/dev/p2r1a-thin-plate-runbook.md
# ---------------------------------------------------------------------------
#
# Domain: [0, 36] x [0, 16] (unit octree: all coords normalized to [0,1]).
# Note: octree native domain is [0,1]^2; physical coords = octree_coord * max_dim.
# The plate at physical (5, 8) => octree (5/36, 8/16) = (0.1389, 0.5).
# Plate length L = 1.0 (physical units); normalized L_norm = 1.0 / 16 = 0.0625.
# (L=1 matches Najjar & Balachandar 1995, domain H=16D with D=plate length=L).
# Re = U_inf * L / nu = 250 => nu = U_inf * L / 250 = 1.0 * 1.0 / 250 = 0.004.
# U_inf = 1.0, L = 1.0 (physical).
# Base mesh: level 7 (128^2 octree); wake refined to L9; plate cells to L9-11.
# Adaptive refinement: use refine_elements() + balance2to1() (not yet wired
# into run_flow_past; see runbook for the --adaptive flag).
# dt = 5e-5 (physical), march until t >= 50 (50+ shedding periods expected at St~0.15).
# t_start for averaging: 20.0 (post-transient).
# Expected (Table 1, ThinShell.pdf): Cd ~ 3.29-3.45, St ~ 0.15.
RE250_CONFIG = dict(
    level=7,
    nsteps=1_000_000,  # not actually run in CI; document only
    dt=5e-5,
    U_inf=1.0,
    nu=1.0 / 250.0,    # Re=U_inf*L/nu=250, L=1 (physical plate length)
    alpha=50.0,
    plate_xc=5.0 / 36.0,   # physical x=5 in [0,36] domain
    plate_yc=8.0 / 16.0,   # physical y=8 in [0,16] domain
    plate_L=1.0 / 16.0,    # L=1 physical, normalized by domain height 16
    dim=2,
)


if __name__ == "__main__":
    import os
    from diffsim.postproc.shedding import time_avg_cd, strouhal

    level = int(os.environ.get("LEVEL", "5"))
    nsteps = int(os.environ.get("NSTEPS", "10"))
    dt = float(os.environ.get("DT", "0.01"))
    nu = float(os.environ.get("NU", "0.1"))
    U_inf = float(os.environ.get("U_INF", "1.0"))
    plate_xc = float(os.environ.get("PLATE_XC", "0.375"))
    plate_yc = float(os.environ.get("PLATE_YC", "0.5"))
    plate_L = float(os.environ.get("PLATE_L", "0.25"))
    # Physical plate length: plate_L is the octree-normalized length (divided by
    # domain height H=16).  St = f*L/U uses the PHYSICAL plate length, so we
    # multiply back by 16 to denormalize.  For the default smoke run (plate_L=0.25,
    # unit-square domain) this gives plate_L_physical=4.0; for the RE250 gpubox
    # run (plate_L=1/16=0.0625) this gives plate_L_physical=1.0 (physical units).
    plate_L_physical = plate_L * 16.0  # *16: domain-height denormalization

    res = run_flow_past(
        level=level, nsteps=nsteps, dt=dt, nu=nu, U_inf=U_inf,
        plate_xc=plate_xc, plate_yc=plate_yc, plate_L=plate_L,
        verbose=True,
    )
    print(f"Cd={res['cd']}")
    print(f"Cl={res['cl']}")

    # t_arr: step k lands at time (k+1)*dt since step 0 is the first BDF step
    # (t=dt).  np.linspace(0, nsteps*dt, nsteps) gives wrong effective dt.
    t_arr = np.arange(1, nsteps + 1) * dt
    cd_mean = time_avg_cd(t_arr, res["cd"])
    try:
        St, freq = strouhal(t_arr, res["cl"], U_inf, plate_L_physical)
        print(f"Cd_mean={cd_mean:.4f}  St={St:.4f}  freq={freq:.4f}")
    except ValueError as exc:
        print(f"Cd_mean={cd_mean:.4f}  St=N/A (too short: {exc})")
