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

Symmetry-breaking perturbation
-------------------------------
A perfectly symmetric flow-past-a-symmetric-plate rides the unstable symmetric
branch and NEVER spontaneously sheds vortices on a symmetric mesh.  To trigger
shedding at Re >= ~100, a small transverse (cross-flow) velocity is injected at
the inflow nodes for early time t < pert_t_end:

    u_y|_{inflow} = pert_eps * U_inf    for t < pert_t_end
    u_y|_{inflow} = 0.0                 for t >= pert_t_end

This mirrors the cylinder shedding driver (tests/test_cylinder_strouhal.py:
``vkick = 0.05 * U_IN if t_new < 0.5 else 0.0``).

Parameters (passed to run_flow_past):
  pert_eps    : float | None  — transverse kick amplitude as fraction of U_inf.
                  None or 0.0 => no perturbation (default for CI smoke).
                  ~0.02–0.05 is sufficient for Re >= 100 on a plate.
  pert_t_end  : float         — time at which kick is switched off (default 1.0).

For the Re=250 gpubox run set pert_eps=0.03, pert_t_end=1.0.
For CI smoke (Re=10, 10 steps) leave pert_eps=None — deterministic & fast.

Outlet BC note
--------------
The outflow boundary (x=x_max) uses a "do-nothing" / natural outlet: no velocity
Dirichlet is imposed there, and the surface integral from IBP is simply dropped
(equivalent to a zero-traction, zero-stress condition sigma.n=0).  A single
pressure node is pinned (p=0 at the outflow–bottom corner) to remove the
pressure null-space.

Assessment: this is acceptable for the Re=250 shedding run PROVIDED the domain
is long enough that vortices are sufficiently diffused before reaching the
outlet.  For the unit-square smoke domain the plate-to-outlet distance is only
~0.625 plate lengths — marginal.  For the RE250_CONFIG the physical domain is
[0,36]x[0,16] with the plate at x=5, giving 31 plate-lengths of wake — well
beyond the ~20L recommended.

Reflection risk: do-nothing sets sigma.n=0, i.e. p - nu(grad u).n = 0.  When a
vortex convects through, this imposes a transient pressure that back-drives a
small spurious velocity.  In a long domain the vortex is weak by the time it
hits the outlet; in the unit-square smoke the wake never reaches the outlet in
10 steps (convection time L/U = 0.625 >> 0.1 s total march).  For Re=250 the
wake structures are coherent; if reflection artefacts appear (Cl oscillation
phase-locking to outlet convection time) the recommended fix is convective
(advective) outlet: u_t + U_inf * u_x = 0 applied weakly, or Robin BC
p - nu(grad u).n = U_inf * u.n.  This is a stepper rewrite and is deferred.
Baskar should watch for outlet-phase-locking signatures in the Re=250 Cl(t)
spectrum (a spurious peak at f = U_inf / L_wake).

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
from diffsim.steppers.leray_sbm import LeraySBMShellStepper


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

    Returns (rows, vals, inflow_vy_rows) in free-node DOF space.
      rows, vals         : base BCs (u_x=U_inf, u_y=0 everywhere forced)
      inflow_vy_rows     : np.int64 array of the u_y DOF rows for INFLOW nodes
                           only — used by the march loop to inject the
                           symmetry-breaking transverse kick (pert_eps * U_inf)
                           for early time.  Empty array if no inflow nodes.
    """
    coords = mesh.node_coords[cons.free_nodes]
    x_min = coords[:, 0].min()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()
    inflow_mask = np.abs(coords[:, 0] - x_min) < 1e-10
    walls = (np.abs(coords[:, 1] - y_min) < 1e-10) | \
            (np.abs(coords[:, 1] - y_max) < 1e-10)
    forced = inflow_mask | walls
    rows, vals = [], []
    for i in np.where(forced)[0]:
        rows.append(i * ndof + 0); vals.append(U_inf)   # u_x = U_inf
        rows.append(i * ndof + 1); vals.append(0.0)     # u_y = 0
    # Collect the u_y DOF indices for inflow nodes only (for the kick)
    inflow_vy_rows = np.array(
        [i * ndof + 1 for i in np.where(inflow_mask)[0]], dtype=np.int64)
    return np.asarray(rows, np.int64), np.asarray(vals), inflow_vy_rows


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
    _return_fields=False,  # internal flag: True => also return mesh + node fields
    pert_eps=None,     # symmetry-breaking kick: fraction of U_inf (None or 0.0 = off)
    pert_t_end=1.0,    # time (physical) at which the kick is switched off
):
    """Run flow past a finite thin plate with transient BDF2 march.

    Parameters
    ----------
    level, nsteps, dt, U_inf, nu, alpha, plate_xc, plate_yc, plate_L, dim :
        Standard geometry/physics parameters (see module docstring).
    verbose : bool
        Print per-step Cd/Cl.
    _two_sided : bool
        Internal flag — False => one-sided anti-vacuity test only.
    _return_fields : bool
        Internal flag — True => also return mesh + node fields dict.
    pert_eps : float | None
        Symmetry-breaking transverse kick amplitude as a fraction of U_inf.
        Applied at INFLOW nodes (u_y = pert_eps * U_inf) for t < pert_t_end,
        then switched off.  None or 0.0 => no perturbation (default for CI
        smoke so the smoke is deterministic and fast).  Use 0.03 for Re=250.
    pert_t_end : float
        Physical time at which the kick is switched off (default 1.0).
        After this the inflow reverts to pure streamwise (u_y = 0).

    Returns a dict with:
      'cd'       : np.ndarray [nsteps] — drag coefficient history
      'cl'       : np.ndarray [nsteps] — lift coefficient history
      'n_excluded': int — number of excluded octree cells (non-zero confirms plate active)
      'nsteps'   : int — number of steps actually taken

    When _return_fields=True, also returns:
      'mesh'     : the DiffSim Mesh object (full octree connectivity)
      'node_fields': dict with 'velocity_magnitude' [Nn] and 'pressure' [Nn]
                     extracted from the final time step's solution
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
    bc_rows, bc_vals, inflow_vy_rows = _outer_bc(mesh, cons, ndof, dim, U_inf)
    p_pin = _pressure_pin(mesh, cons, ndof, dim)

    # Symmetry-breaking perturbation setup
    # pert_eps=None or 0.0 => no kick (CI smoke path — deterministic)
    _pert_active = (pert_eps is not None) and (float(pert_eps) != 0.0)
    _pert_vkick = float(pert_eps) * U_inf if _pert_active else 0.0
    if _pert_active and verbose:
        print(f"[p2r1a] perturbation ON: eps={pert_eps}, v_kick={_pert_vkick:.4f}, "
              f"t_end={pert_t_end}, n_inflow_nodes={len(inflow_vy_rows)}", flush=True)

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
        t_new = (step + 1) * dt   # time at the END of this step

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

        # Symmetry-breaking perturbation: override inflow u_y for t < pert_t_end.
        # This mirrors test_cylinder_strouhal.py's kick:
        #   ``vkick = 0.05 * U_IN if t_new < 0.5 else 0.0``
        # The kick imposes a small constant transverse velocity at the inflow
        # for early time, breaking the perfect up-down symmetry so the wake
        # destabilises to the von Karman / bluff-body shedding branch.
        # After t >= pert_t_end the inflow reverts to pure streamwise (u_y = 0),
        # already set by the base bc_rows loop above (no additional action).
        if _pert_active and t_new < pert_t_end:
            vkick = _pert_vkick
            for r in inflow_vy_rows:
                ri = int(r)
                A.rows[ri] = [ri]; A.data[ri] = [1.0]; b[ri] = vkick

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

    result = dict(
        cd=cd_hist,
        cl=cl_hist,
        n_excluded=fx["n_excluded"],
        nsteps=nsteps,
    )

    if _return_fields:
        # Extract node-level fields from the final step's x_all.
        # x_all is [Nn * ndof] node-major: node 0 has [u_x, u_y, p],
        # node 1 has [u_x, u_y, p], etc.
        x_nodes = np.asarray(x_all).reshape(-1, ndof)   # [Nn, ndof]
        u_node = x_nodes[:, :dim]                        # [Nn, dim]
        p_node = x_nodes[:, dim]                         # [Nn]
        vel_mag = np.linalg.norm(u_node, axis=1)         # [Nn]
        result["mesh"] = mesh
        result["node_fields"] = {
            "velocity_magnitude": vel_mag,
            "pressure": p_node,
        }

    return result


def run_flow_past_one_sided(**kwargs):
    """Anti-vacuity run: same mesh/BCs but only Gamma~- assembled (drop Gamma~+).
    Used by the smoke gate to verify the two-sided coupling is load-bearing."""
    kwargs["_two_sided"] = False
    return run_flow_past(**kwargs)


# ---------------------------------------------------------------------------
# PROJECTION path (LeraySBMShellStepper) — 2-D mirror of the merged 3-D
# projection driver (tests/p2r1c_thin_plate_flow_3d_projection.py).
# ---------------------------------------------------------------------------
#
# WHY A SECOND SOLVER: the both-solver harness runs the SAME 2-D thin-plate
# case through (a) the monolithic VMS-saddle (run_flow_past above) and (b) this
# projection/PPE split, then compares BOTH Cd/St against the LITERATURE band
# (Najjar & Balachandar 1995: Cd 3.36, St 0.14; Table-1 band Cd 3.29-3.45,
# St ~0.15).  Per ns_projection_vms_paper §4.3+Fig.10 the monolithic VMS
# OVERpredicts Cd (~25%, the pressure-fine-scale/grad-div term) and the
# projection is the more literature-faithful one — so monolithic is NOT ground
# truth; the LITERATURE is.  This driver is the projection leg of that test.
#
# The wiring copies the merged 3-D projection driver EXACTLY where it is
# dim-generic: the (strong_mask, u_inf, outflow_nodes) BC convention, the
# lever defaults (consistent_projection=True, inner_iterate=True, inner_max=8,
# inner_relax=0.5, rotational_pin_wall=True), and the whole-outflow-face p'=0
# Dirichlet.  The ONLY 2-D specialization is the outlet node set: in 3-D the
# outlet is a 2-D FACE (a sheet of nodes) at x=x_max; in 2-D it is a 1-D LINE
# of nodes at x=x_max.  Both are just "the free nodes with x≈x_max", so the
# same np.abs(coords[:,0]-x_max)<tol mask works verbatim.
#
# Geometry/force use the SAME two-oracle signed-psi workaround as the
# monolithic path (Segment classify + Plane extract; _build_shell), the same
# finite Segment plate, two-sided shell, and perturbation kick.


def _outer_bc_masks_2d(mesh, cons, U_inf):
    """Strong outer BCs for 2-D flow past a plate, in the free-node-major
    (strong_mask, u_inf) convention LeraySBMShellStepper expects.

    Mirrors _outer_bc_masks_3d from the merged 3-D projection driver:
      - inflow (x=x_min) + top/bottom walls (y=y_min|y_max): u=(U_inf, 0)
        [freestream velocity-Dirichlet -> natural grad(phi).n=0 PPE BC]
      - outflow (x=x_max): do-nothing (velocity FREE); the WHOLE outflow LINE
        (a 1-D set of nodes in 2-D, the analogue of the 3-D outlet face) carries
        the PPE p'=0 Dirichlet (incremental van-Kan p'-scheme, Eq. 68d) via
        ``pressure_outflow_nodes`` — NOT the single enclosed-flow corner pin the
        monolithic saddle uses.  Pinning the whole outlet imposes the physical
        outlet pressure so the incremental p* does not drift (the 2026-07-25
        outflow-BC fix, dim-generic).  The plate/shell gets NO pressure
        Dirichlet (phi natural-Neumann there).

    Returns (strong_mask [n_free] bool, u_inf [n_free, 2] float,
             outflow_nodes [k] int, inflow_vy_rows [m] int):
      - strong_mask/u_inf : the strong velocity-Dirichlet set for the box.
      - outflow_nodes     : FREE-node indices on the outlet line x=x_max.
      - inflow_vy_rows    : FREE-node indices of the INFLOW nodes (used by the
        march to inject the symmetry-breaking transverse kick — the stepper
        overrides u_inf[inflow, 1] for early time, mirroring the monolithic path).
    """
    dim = 2
    coords = mesh.node_coords[cons.free_nodes]
    tol = 1e-10
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    inflow = np.abs(coords[:, 0] - x_min) < tol
    walls = (np.abs(coords[:, 1] - y_min) < tol) | \
            (np.abs(coords[:, 1] - y_max) < tol)
    strong_mask = inflow | walls
    u_inf = np.zeros((len(coords), dim))
    u_inf[strong_mask, 0] = U_inf                      # freestream u_x = U_inf
    # outflow LINE x=x_max: 1-D node set (2-D analogue of the 3-D outlet face)
    outflow_line = np.abs(coords[:, 0] - x_max) < tol
    outflow_nodes = np.where(outflow_line)[0].astype(np.int64)
    inflow_nodes = np.where(inflow)[0].astype(np.int64)
    return strong_mask, u_inf, outflow_nodes, inflow_nodes


def run_flow_past_projection(
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
    ppe_solver="splu",       # splu on CPU (Mac); gpu_cg on GPU (gpubox/GH200)
    predictor_solver="splu",
    picard_iters=2,
    order=2,
    consistent_projection=True,
    inner_iterate=True,
    inner_max=8,
    inner_relax=0.5,
    rotational_pin_wall=True,
    _two_sided=True,
    _return_fields=False,
    _return_stepper=False,
    pert_eps=None,           # symmetry-breaking kick: fraction of U_inf (None/0 = off)
    pert_t_end=1.0,          # physical time at which the kick is switched off
):
    """Run 2-D flow past a finite thin plate via the PROJECTION stepper.

    2-D mirror of tests/p2r1c_thin_plate_flow_3d_projection.run_flow_past_3d_projection,
    marched with LeraySBMShellStepper (two-sided shell SBM + Helmholtz-Leray
    projection split).  Same finite Segment plate + two-sided shell + two-oracle
    signed-psi workaround (Segment classify / Plane extract) + perturbation kick
    + Cd/Cl history as the monolithic run_flow_past.

    Parameters mirror run_flow_past plus the projection levers:
      ppe_solver : "splu" on CPU (Mac) / "gpu_cg" on GPU (gpubox/GH200).  The
        PPE Laplacian is SPD so gpu_cg is valid there; the Oseen predictor is
        NOT SPD so predictor_solver stays "splu" (or FGMRES/AMGX on device).
      consistent_projection / inner_iterate / inner_max / inner_relax /
      rotational_pin_wall : the merged 3-D projection driver's lever defaults —
        the outflow-BC p'-scheme set that makes the immersed-shell Cd
        POSITIVE + NON-DIVERGING (correct sign/shape).

    Returns a dict with:
      'cd'         : np.ndarray [nsteps] — drag coefficient (streamwise x)
      'cl'         : np.ndarray [nsteps] — lift coefficient (transverse y)
      'n_excluded' : int — excluded cells (non-zero confirms plate active)
      'nsteps'     : int — steps taken
      'ppe_solver' : str — the PPE solver actually requested (confirms path)
      + 'mesh'/'node_fields' when _return_fields, 'stepper' when _return_stepper.
    """
    ndof = dim + 1
    t0 = time.time()

    # ---- geometry + two-sided surrogate (SAME two-oracle workaround) --------
    fx = _build_shell(level, plate_xc, plate_yc, plate_L, dim=dim)
    dm, mesh, cons = fx["dm"], fx["mesh"], fx["cons"]

    if verbose:
        print(f"[p2r1a-proj] level={level}  n_excluded={fx['n_excluded']}  "
              f"sfp={fx['sfp'].elem.size}  sfm={fx['sfm'].elem.size}  "
              f"nsteps={nsteps}  dt={dt}  nu={nu}  "
              f"ppe_solver={ppe_solver}", flush=True)

    # ---- BCs in the (strong_mask, u_inf, outflow_nodes) free-node convention
    strong_mask, u_inf_arr, outflow_nodes, inflow_nodes = _outer_bc_masks_2d(
        mesh, cons, U_inf)

    def f_fn(x, t):
        return np.zeros((len(x), dim))

    # ---- one-sided anti-vacuity lever: drop Gamma~+ (fold onto minus side) --
    if _two_sided:
        sfp, gp = fx["sfp"], fx["gp"]
    else:
        # Empty Gamma~+ is illegal (one-face-order invariant); reuse sfm/gm as
        # the plus side too — the one-sided anti-vacuity assembly.
        sfp, gp = fx["sfm"], fx["gm"]

    # ---- projection stepper: two-sided shell + PPE -------------------------
    st = LeraySBMShellStepper(
        sfp, gp, fx["sfm"], fx["gm"],
        dm, nu, dt, f_fn,
        u_inf=u_inf_arr, strong_mask=strong_mask,
        order=order, picard_iters=picard_iters,
        solver=predictor_solver, ppe_solver=ppe_solver,
        alpha=alpha, beta_backflow=1.0,
        pressure_outflow_nodes=outflow_nodes,
        consistent_projection=consistent_projection,
        inner_iterate=inner_iterate, inner_max=inner_max,
        inner_relax=inner_relax,
        rotational_pin_wall=rotational_pin_wall,
        verbose=verbose,
    )
    st.set_initial(lambda coords: np.zeros((len(coords), dim)))

    # Symmetry-breaking perturbation setup (mirrors the monolithic path): a
    # small transverse kick at the INFLOW nodes for t < pert_t_end.  In the
    # projection path the box velocity is imposed strongly through st.u_inf, so
    # the kick is applied by mutating st.u_inf[inflow, 1] on/off per step.
    _pert_active = (pert_eps is not None) and (float(pert_eps) != 0.0)
    _pert_vkick = float(pert_eps) * U_inf if _pert_active else 0.0
    if _pert_active and verbose:
        print(f"[p2r1a-proj] perturbation ON: eps={pert_eps}, "
              f"v_kick={_pert_vkick:.4f}, t_end={pert_t_end}, "
              f"n_inflow_nodes={len(inflow_nodes)}", flush=True)

    ref_force = 0.5 * U_inf ** 2 * plate_L    # nondim denominator (matches mono)

    cd_hist = np.zeros(nsteps)
    cl_hist = np.zeros(nsteps)

    for step in range(nsteps):
        t_new = (step + 1) * dt
        # Apply/clear the transverse kick on the inflow u_y BEFORE the step.
        if _pert_active:
            st.u_inf[inflow_nodes, 1] = (_pert_vkick if t_new < pert_t_end
                                         else 0.0)
        st.step()
        F = st.surrogate_traction()
        cd_hist[step] = F[0] / ref_force   # drag (streamwise = x)
        cl_hist[step] = F[1] / ref_force   # lift (transverse = y)
        if verbose:
            print(f"[p2r1a-proj] step {step:3d}  Cd={cd_hist[step]:+.4f}  "
                  f"Cl={cl_hist[step]:+.4f}", flush=True)

    elapsed = time.time() - t0
    if verbose:
        print(f"[p2r1a-proj] done in {elapsed:.1f}s  "
              f"Cd[-1]={cd_hist[-1]:+.4f}  Cl[-1]={cl_hist[-1]:+.4f}", flush=True)

    result = dict(
        cd=cd_hist,
        cl=cl_hist,
        n_excluded=fx["n_excluded"],
        nsteps=nsteps,
        ppe_solver=ppe_solver,
    )

    if _return_fields:
        u = st.base._uvec(st.base.hist.pre1)
        xfree = np.zeros(st.n_free * ndof)
        xv = xfree.reshape(st.n_free, ndof)
        xv[:, :dim] = u
        xv[:, dim] = st.base.p_star
        x_all = np.asarray(st._T_vec @ xfree).reshape(-1, ndof)
        result["mesh"] = mesh
        result["node_fields"] = {
            "velocity_magnitude": np.linalg.norm(x_all[:, :dim], axis=1),
            "pressure": x_all[:, dim],
        }
    if _return_stepper:
        result["stepper"] = st

    return result


def run_flow_past_projection_one_sided(**kwargs):
    """Anti-vacuity run for the projection path: only Gamma~- assembled (drop
    Gamma~+).  Used by the smoke gate to verify the two-sided coupling is
    load-bearing (mirrors the monolithic run_flow_past_one_sided)."""
    kwargs["_two_sided"] = False
    return run_flow_past_projection(**kwargs)


# ---------------------------------------------------------------------------
# Both-solver comparison vs the LITERATURE
# ---------------------------------------------------------------------------
# Literature band (ThinShell.pdf §4.3 Table 1 + Najjar & Balachandar 1995):
LIT_CD_LOW, LIT_CD_HIGH = 3.29, 3.45     # Cd band
LIT_CD_REF = 3.36                        # Najjar & Balachandar center
LIT_ST_REF = 0.15                        # Strouhal (Najjar 0.14; band ~0.15)


def compare_solvers(t_start=None, plate_L_physical=None, **cfg):
    """Run the SAME 2-D thin-plate case through BOTH solvers (monolithic and
    projection) and report each solver's Cd (time-average) + St (FFT of Cl)
    against the LITERATURE band (Cd 3.29-3.45, St ~0.15; Najjar 3.36/0.14).

    The deliverable is WHICH SOLVER matches the literature — NOT which matches
    the other.  Per ns_projection_vms_paper §4.3 the monolithic VMS
    OVERpredicts Cd (~25%) and the projection is the more literature-faithful
    one, so this comparison settles whether the projection's lower Cd is a
    FEATURE (paper-consistent) or an immersed-shell deficiency.

    Parameters
    ----------
    t_start : float or None
        Post-transient time at which Cd time-averaging begins (defaults to
        halfway through the march).
    plate_L_physical : float or None
        PHYSICAL plate length used for St = f*L/U (defaults to cfg['plate_L'],
        i.e. the octree-normalized length; for the RE250 config denormalize by
        the domain height, see __main__).
    **cfg : forwarded to BOTH run_flow_past and run_flow_past_projection
        (level, nsteps, dt, U_inf, nu, alpha, plate_xc, plate_yc, plate_L,
        pert_eps, pert_t_end, ...).

    Returns a dict:
      {'mono': {'cd_mean','St','freq','cd','cl'},
       'proj': {'cd_mean','St','freq','cd','cl','ppe_solver'},
       'literature': {'cd_low','cd_high','cd_ref','st_ref'}}
    and prints the comparison table.
    """
    from diffsim.postproc.shedding import time_avg_cd, strouhal

    U_inf = cfg.get("U_inf", 1.0)
    dt = cfg.get("dt", 0.01)
    nsteps = cfg.get("nsteps", 10)
    plate_L = cfg.get("plate_L", 0.25)
    L_phys = plate_L if plate_L_physical is None else plate_L_physical

    # Which projection knobs to pull out of cfg (leave the rest for both).
    proj_only = {}
    for k in ("ppe_solver", "predictor_solver", "picard_iters", "order",
              "consistent_projection", "inner_iterate", "inner_max",
              "inner_relax", "rotational_pin_wall"):
        if k in cfg:
            proj_only[k] = cfg.pop(k)

    t_arr = np.arange(1, nsteps + 1) * dt
    ts = t_arr[len(t_arr) // 2] if t_start is None else t_start

    def _reduce(res):
        cd_mean = time_avg_cd(t_arr, res["cd"], t_start=ts)
        try:
            St, freq = strouhal(t_arr, res["cl"], U_inf, L_phys)
        except ValueError:
            St, freq = float("nan"), float("nan")
        return cd_mean, St, freq

    res_m = run_flow_past(**cfg)
    cd_m, st_m, f_m = _reduce(res_m)

    res_p = run_flow_past_projection(**cfg, **proj_only)
    cd_p, st_p, f_p = _reduce(res_p)

    def _cd_dist(cd):
        # signed absolute distance to the nearest band edge (0 if inside band)
        if cd < LIT_CD_LOW:
            return cd - LIT_CD_LOW
        if cd > LIT_CD_HIGH:
            return cd - LIT_CD_HIGH
        return 0.0

    print("\n" + "=" * 72)
    print("BOTH-SOLVER vs LITERATURE — 2-D thin plate")
    print(f"  Literature: Cd {LIT_CD_LOW}-{LIT_CD_HIGH} (ref {LIT_CD_REF}), "
          f"St ~{LIT_ST_REF} (Najjar & Balachandar 3.36/0.14)")
    print("-" * 72)
    hdr = f"{'solver':<12}{'Cd_mean':>10}{'|dCd_ref|':>11}{'relCd_ref':>11}" \
          f"{'St':>8}{'|dSt|':>8}"
    print(hdr)
    for name, cd, st in (("monolithic", cd_m, st_m), ("projection", cd_p, st_p)):
        dcd = abs(cd - LIT_CD_REF)
        rel = dcd / LIT_CD_REF
        dst = abs(st - LIT_ST_REF)
        band = "" if _cd_dist(cd) == 0.0 else "  (out-of-band)"
        print(f"{name:<12}{cd:>10.4f}{dcd:>11.4f}{rel:>10.1%}"
              f"{st:>8.4f}{dst:>8.4f}{band}")
    print("=" * 72 + "\n")

    return dict(
        mono=dict(cd_mean=cd_m, St=st_m, freq=f_m,
                  cd=res_m["cd"], cl=res_m["cl"]),
        proj=dict(cd_mean=cd_p, St=st_p, freq=f_p,
                  cd=res_p["cd"], cl=res_p["cl"],
                  ppe_solver=res_p.get("ppe_solver")),
        literature=dict(cd_low=LIT_CD_LOW, cd_high=LIT_CD_HIGH,
                        cd_ref=LIT_CD_REF, st_ref=LIT_ST_REF),
    )


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
    # --- Symmetry-breaking perturbation (ON for Re=250 gpubox run) -----------
    # A small transverse kick at the inflow for t < pert_t_end breaks the
    # perfect up-down symmetry so the wake can destabilise to vortex shedding.
    # eps=0.03 => v_kick = 0.03 * U_inf = 0.03.  Duration: 1.0 physical time
    # unit = 20,000 steps at dt=5e-5.  After that, pure streamwise inflow.
    # This mirrors the cylinder shedding driver (test_cylinder_strouhal.py).
    pert_eps=0.03,
    pert_t_end=1.0,
)


if __name__ == "__main__":
    import os
    from diffsim.postproc.shedding import time_avg_cd, strouhal
    from diffsim.viz.results import save_flow_run

    level = int(os.environ.get("LEVEL", "5"))
    nsteps = int(os.environ.get("NSTEPS", "10"))
    dt = float(os.environ.get("DT", "0.01"))
    nu = float(os.environ.get("NU", "0.1"))
    U_inf = float(os.environ.get("U_INF", "1.0"))
    plate_xc = float(os.environ.get("PLATE_XC", "0.375"))
    plate_yc = float(os.environ.get("PLATE_YC", "0.5"))
    plate_L = float(os.environ.get("PLATE_L", "0.25"))
    # Symmetry-breaking perturbation (env-var knob for CLI runs):
    #   PERT_EPS=0.03 PERT_T_END=1.0  (default: no perturbation for smoke)
    _pert_eps_env = os.environ.get("PERT_EPS", "")
    pert_eps = float(_pert_eps_env) if _pert_eps_env else None
    pert_t_end = float(os.environ.get("PERT_T_END", "1.0"))
    # Physical plate length: plate_L is the octree-normalized length (divided by
    # domain height H=16).  St = f*L/U uses the PHYSICAL plate length, so we
    # multiply back by 16 to denormalize.  For the default smoke run (plate_L=0.25,
    # unit-square domain) this gives plate_L_physical=4.0; for the RE250 gpubox
    # run (plate_L=1/16=0.0625) this gives plate_L_physical=1.0 (physical units).
    plate_L_physical = plate_L * 16.0  # *16: domain-height denormalization

    # Re number for the case name (dimensionless: Re = U_inf / nu for L=1)
    re_approx = int(round(U_inf / nu)) if nu > 0 else 0
    case_name = f"p2r1a_re{re_approx}_L{level}"

    # ---- BOTH-SOLVER comparison mode (SOLVER=both or COMPARE=1) ------------
    # Runs monolithic AND projection on the SAME config and prints the
    # Cd/St-vs-literature table.  The projection PPE solver is PPE_SOLVER
    # (default splu on CPU; set gpu_cg on gpubox/GH200).
    _compare = (os.environ.get("SOLVER", "").lower() == "both"
                or os.environ.get("COMPARE", "") == "1")
    if _compare:
        ppe_solver = os.environ.get("PPE_SOLVER", "splu")
        _t_start_env = os.environ.get("T_START", "")
        t_start = float(_t_start_env) if _t_start_env else None
        out = compare_solvers(
            t_start=t_start, plate_L_physical=plate_L_physical,
            level=level, nsteps=nsteps, dt=dt, nu=nu, U_inf=U_inf,
            plate_xc=plate_xc, plate_yc=plate_yc, plate_L=plate_L,
            pert_eps=pert_eps, pert_t_end=pert_t_end,
            ppe_solver=ppe_solver,
        )
        print(f"[p2r1a] monolithic: Cd_mean={out['mono']['cd_mean']:.4f}  "
              f"St={out['mono']['St']:.4f}")
        print(f"[p2r1a] projection: Cd_mean={out['proj']['cd_mean']:.4f}  "
              f"St={out['proj']['St']:.4f}  (ppe={out['proj']['ppe_solver']})")
        sys.exit(0)

    # Run with _return_fields=True so we get mesh + node fields for VTU export.
    # The CI smoke test calls run_flow_past() directly without _return_fields,
    # so it is completely unaffected by this flag.
    res = run_flow_past(
        level=level, nsteps=nsteps, dt=dt, nu=nu, U_inf=U_inf,
        plate_xc=plate_xc, plate_yc=plate_yc, plate_L=plate_L,
        verbose=True,
        _return_fields=True,
        pert_eps=pert_eps,
        pert_t_end=pert_t_end,
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

    # ---- Export visualization artifacts to results/ (Dropbox-synced) ----------
    # Body: the thin plate is a 2-D line segment, which cannot be represented as
    # a valid triangle mesh (needs >= 3 non-collinear vertices with area > 0).
    # save_flow_run detects this and skips the body export with a clear message.
    t_hist = t_arr
    written = save_flow_run(
        case_name,
        mesh=res.get("mesh"),
        node_fields=res.get("node_fields"),
        histories={"Cd": (t_hist, res["cd"]), "Cl": (t_hist, res["cl"])},
        body=None,   # 2-D line plate: no valid surface mesh — skip body export
    )
    if written:
        print(f"[p2r1a] Results written: {list(written.values())}")
    else:
        print("[p2r1a] No results written (viz deps missing or export failed)")
