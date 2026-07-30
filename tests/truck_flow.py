"""TRUCK case driver — volumetric one-sided SBM on a merged-STL triangle soup.

This is the ThinShell.pdf truck case on the incomplete-octree + direct-STL
design (task TU2).  The truck is a SOLID (not a shell): the one-sided
volumetric SBM pipeline is used (mirrors tests/p2r1_thin_plate_blocked_channel
``build_carved``), NOT the two-sided shell machinery of p2r1c.

Mesh recipe (incomplete-octree, ThinShell C++ prior art):
  1. build_uniform(base_level, dim=3) on the octree unit cube [0,1]^3.
  2. SLAB carve: retain only the channel slab (physical [0,16]x[0,2]x[0,2] ->
     unit [0,1]x[0,1/8]x[0,1/8]) via classify_lambda(..., domain="inside").
     The slab bounds y=1/8, z=1/8 are DYADIC, so no cell is cut — asserted
     exactly (zero intercepted slab cells).
  3. region-refine boxes (refine_elements + balance2to1).
  4. truck-band refine: refine cells whose merged-mesh |distance| < band.
  5. TRUCK carve: classify_lambda(..., domain="outside") on the MergedTriMesh
     -> retain fluid around the solid; extract the exposed surrogate faces.

Forces (BOTH observables recorded per step, task amendment 4):
  * consistent REACTION (canonical): variational identity on a truck-enclosing
    u_x indicator, Cd_reaction = -w^T (A_vol x - b_vol) / ref_force.
  * surrogate-surface TRACTION integral (-p n + viscous), Cd_surr.

Strong BCs (C++ TRUCK branch, amendment 3), on the dyadic domain planes:
  inlet x=0 : u_x = min(y_phys/slopeNearGround, 1), u_y=u_z=0
  ground y=0: no-slip (u=0)
  ceiling y=y_max: SLIP (u_y=0 only)
  side z=0,z=z_max: SLIP (u_z=0 only)
  outlet x=1: do-nothing + pressure pin

Run (tiny smoke):  .venv/bin/python tests/truck_flow.py
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

sys.path.insert(0, os.path.dirname(__file__))

from diffsim.octree.build import build_uniform, refine_elements
from diffsim.octree.balance import balance2to1
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.csg import Box
from diffsim.geometry.merged_trimesh import MergedTriMesh
from diffsim.sbm.surrogate import classify_lambda, extract_surrogate, GeometryData
from diffsim.sbm.vector import sbm_vector_dirichlet, surrogate_traction
from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.physics.poisson import gauss_points
from diffsim.solvers.timestepping import bdf_coeffs
from diffsim.solvers.linsolve import solve_linear, _LAST_ITERS


# ---------------------------------------------------------------------------
# Mesh build (incomplete octree: slab carve + region refine + truck carve)
# ---------------------------------------------------------------------------

def _channel_box(domain_min, domain_max, scale):
    """Box oracle for the channel slab in unit-cube coords.  psi<0 inside."""
    lo = np.asarray(domain_min, np.float64) * scale
    hi = np.asarray(domain_max, np.float64) * scale
    center = tuple((lo + hi) / 2.0)
    half = tuple((hi - lo) / 2.0)
    return Box(center, half)


def slab_carve(tree, channel_box):
    """Retain only cells inside the channel slab.  Returns (ret, n_intercepted).

    Dyadic slab bounds => NO cell is cut: we assert every retained cell is
    FULLY inside (frac == 1.0) and no cell is partially intercepted.  This is
    the dyadic-exactness guarantee (zero intercepted slab cells).
    """
    # lipschitz_bound=inf forces dense sampling so a straddling cell WOULD show
    # a fractional frac; dyadic bounds guarantee it does not.
    ret, frac = classify_lambda(tree, channel_box, lam=0.0, domain="inside",
                                lipschitz_bound=np.inf)
    # With lam=0.0, only fully-inside cells (frac==1) are retained; any cut cell
    # (0<frac<1) is DROPPED.  Dyadic exactness => the set of cut cells is empty,
    # which we verify by re-classifying with lam=1.0 (retains cut cells too) and
    # asserting the two retained counts are identical (no cut cells exist).
    ret_all, frac_all = classify_lambda(tree, channel_box, lam=1.0,
                                        domain="inside", lipschitz_bound=np.inf)
    n_cut = len(ret_all) - len(ret)
    assert n_cut == 0, (
        f"slab carve NOT dyadic-exact: {n_cut} intercepted (cut) slab cells; "
        f"channel bounds must land on cell boundaries.")
    assert np.all(frac == 1.0), "retained slab cells must be fully interior"
    return ret, n_cut


def refine_region_boxes(tree, regions, scale):
    """Refine cells whose center is inside each region box, up to its level."""
    import torch
    for r in regions:
        lo = np.asarray(r.min_c, np.float64) * scale
        hi = np.asarray(r.max_c, np.float64) * scale
        target = int(r.refine_region_lvl)
        # iterate levels: refine cells inside the box below target level
        for _ in range(64):
            centers = tree.centers()
            lvl = tree.levels.astype(np.int64)
            inside = np.all((centers >= lo) & (centers <= hi), axis=1)
            mask = inside & (lvl < target)
            if not mask.any():
                break
            tree = refine_elements(tree, mask)
            tree = balance2to1(tree)
    return tree


def refine_truck_band(tree, merged, band_cells, refine_to):
    """Refine cells within band_cells*h of the merged truck surface, to
    refine_to.  Mirrors the C++ 'distance < RefineElementNumber x (h)' band."""
    for _ in range(64):
        centers = tree.centers()
        lvl = tree.levels.astype(np.int64)
        h = tree.h()
        cand = lvl < refine_to
        if not cand.any():
            break
        dist = np.abs(merged.classify(centers))
        mask = cand & (dist < band_cells * h)
        if not mask.any():
            break
        tree = refine_elements(tree, mask)
        tree = balance2to1(tree)
    return tree


def build_truck_mesh(cfg, base_level, region_refine=True, truck_band_to=None,
                     band_cells=3, device="cpu", merged=None,
                     bodies=None):
    """Build the incomplete-octree mesh + volumetric surrogate for the truck.

    Returns a dict: dm, mesh, cons, sf, geo, merged, scale, n_excluded,
    n_slab_cut, n_cells.
    """
    scale = cfg.domain_scale
    bodies = cfg.bodies if bodies is None else bodies
    if merged is None:
        merged = MergedTriMesh.from_bodies(bodies, cfg.config_dir, scale=scale,
                                           device=device)

    tree = build_uniform(base_level, dim=3)

    # 1. slab carve (dyadic-exact)
    cbox = _channel_box(cfg.domain_min, cfg.domain_max, scale)
    tree, n_slab_cut = slab_carve(tree, cbox)

    # 2. region refine
    if region_refine and cfg.region_refine:
        tree = refine_region_boxes(tree, cfg.region_refine, scale)

    # 3. truck-band refine
    if truck_band_to is not None:
        tree = refine_truck_band(tree, merged, band_cells, truck_band_to)

    n_before = len(tree)

    # 4. truck carve (flow AROUND the solid: domain="outside"), lam=1.0 keeps
    # intercepted cells so the surrogate hugs Gamma from OUTSIDE (production
    # RatioGPSBM default).
    ret, _frac = classify_lambda(tree, merged, lam=1.0, domain="outside")
    n_excluded = n_before - len(ret)

    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), device)
    geo = GeometryData.evaluate(merged, ret, sf, face_tables(1, 3),
                                domain="outside")
    return dict(dm=dm, mesh=mesh, cons=cons, sf=sf, geo=geo, merged=merged,
                scale=scale, n_excluded=int(n_excluded), n_slab_cut=int(n_slab_cut),
                n_cells=len(ret))


# ---------------------------------------------------------------------------
# Boundary conditions (C++ TRUCK branch, task amendment 3)
# ---------------------------------------------------------------------------

def truck_bc_masks(mesh, cons, scale, slope_near_ground, domain_max):
    """Return a dict of boolean masks over FREE nodes for each dyadic plane:
    inflow, ground, ceiling, side_zlo, side_zhi, outlet.  Coordinates are in
    unit-cube; physical y for the ramp is y_unit / scale.
    """
    coords = mesh.node_coords[cons.free_nodes]
    tol = 1e-10
    x = coords[:, 0]; y = coords[:, 1]; z = coords[:, 2]
    x_min, x_max = x.min(), x.max()
    y_min = 0.0
    y_max = float(domain_max[1]) * scale
    z_min = 0.0
    z_max = float(domain_max[2]) * scale
    masks = dict(
        inflow=np.abs(x - x_min) < tol,
        outlet=np.abs(x - x_max) < tol,
        ground=np.abs(y - y_min) < tol,
        ceiling=np.abs(y - y_max) < tol,
        side_zlo=np.abs(z - z_min) < tol,
        side_zhi=np.abs(z - z_max) < tol,
    )
    return masks, coords


def truck_strong_bc(mesh, cons, ndof, dim, scale, slope_near_ground,
                    domain_max):
    """Build strong-BC (rows, vals) for the TRUCK branch.

    inlet   x=0     : u_x = min(y_phys/slope, 1), u_y=u_z=0     (ramped)
    ground  y=0     : u = 0                                     (no-slip)
    ceiling y=y_max : u_y = 0                                   (slip)
    sides   z=0,zmax: u_z = 0                                   (slip)
    outlet  x=1     : do-nothing (no rows)

    Precedence at shared edges: no-slip (ground) wins over slip; inflow ramp
    sets all 3 components at x=0.  Returns (rows, vals, masks, coords).
    """
    masks, coords = truck_bc_masks(mesh, cons, scale, slope_near_ground,
                                   domain_max)
    y_phys = coords[:, 1] / scale
    ramp = np.minimum(y_phys / slope_near_ground, 1.0)

    # per-node, per-component value dict keyed by DOF row (later wins)
    val_of = {}

    def set_dof(node_idx, comp, value):
        val_of[int(node_idx) * ndof + comp] = float(value)

    idx = np.arange(len(coords))
    # slip planes first (lowest precedence)
    for i in idx[masks["ceiling"]]:
        set_dof(i, 1, 0.0)               # u_y = 0
    for i in idx[masks["side_zlo"] | masks["side_zhi"]]:
        set_dof(i, 2, 0.0)               # u_z = 0
    # ground no-slip (overrides slip components on shared edges)
    for i in idx[masks["ground"]]:
        set_dof(i, 0, 0.0); set_dof(i, 1, 0.0); set_dof(i, 2, 0.0)
    # inflow ramp (highest precedence at x=0)
    for i in idx[masks["inflow"]]:
        set_dof(i, 0, ramp[i]); set_dof(i, 1, 0.0); set_dof(i, 2, 0.0)

    rows = np.array(sorted(val_of.keys()), np.int64)
    vals = np.array([val_of[int(r)] for r in rows], np.float64)
    return rows, vals, masks, coords


def _pressure_pin(coords, ndof, dim):
    """Outflow-low-back corner free node (max x - y - z); its pressure DOF."""
    corner = int(np.argmax(coords[:, 0] - coords[:, 1] - coords[:, 2]))
    return corner * ndof + dim


def soft_start_amp(t_unit, soft_start, dt):
    """Inlet amplitude ramp multiplier for the impulsive-start transient.

    Physical motivation (TU5R): the campaign's impulsive step-function inlet
    (full amplitude at t=0) drives a spurious startup pressure spike — the
    incompressible pressure is elliptic, so switching the inlet on instantly
    excites the whole field at once (the +201/-87 cd_react class transient seen
    post-onset in the T5 gate leg).  A short linear amplitude ramp

        amp(t) = min(1, t / (soft_start * dt))

    turns the inlet on over ``soft_start`` time-units (in dt) and lets the
    startup pressure response develop smoothly.  ``soft_start=None`` (default)
    or <= 0 returns 1.0 (off; byte-identical to prior behaviour).
    """
    if soft_start is None or soft_start <= 0:
        return 1.0
    return float(min(1.0, t_unit / (float(soft_start) * dt)))


# ---------------------------------------------------------------------------
# Truck-enclosing reaction indicator
# ---------------------------------------------------------------------------

def truck_reaction_set(mesh, cons, merged, ndof, margin):
    """Free-node indices for the reaction arbiter (canonical force observable).

    The truck-enclosing indicator is w=1 on every free node INSIDE a box that
    tightly bounds the merged truck plus ``margin`` (in unit-cube units) on
    each side.  By the discrete NS variational identity, w^T (A_vol x - b_vol)
    then equals the net x-momentum flux through that enclosing surface plus the
    SBM reaction on the carved body — i.e. the force ON the truck (LD-5 arbiter,
    ported from the 2-D driver with the plate indicator replaced by this
    truck-enclosing box).  The caller drops any surgery-row DOFs so w stays
    orthogonal to strong rows (identity holds exactly).
    """
    coords = mesh.node_coords[cons.free_nodes]
    v = merged.verts.numpy()
    lo = v.min(0) - margin
    hi = v.max(0) + margin
    inside = np.all((coords >= lo) & (coords <= hi), axis=1)
    return np.where(inside)[0]


# ---------------------------------------------------------------------------
# Gauss-point field helpers (mirror the p2r1c driver)
# ---------------------------------------------------------------------------

def _gp_field(dm, mesh, T, u_node, dim):
    full = np.asarray(T @ u_node)
    aq, dq = {}, {}
    for pv in dm.bins:
        tb = dm.tables_by_p[pv]
        vals = full[mesh.conn_of[pv]]
        aq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
        h = mesh.tree.h()[dm.bins[pv]["eids"]]
        dq[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                  * (2.0 / h)[:, None]).reshape(-1)
    return aq, dq


def _gp_history_fq(dm, mesh, T, u_pre1, u_pre2, b1, b2, dt, dim):
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

def run_truck(cfg, nsteps, device="cpu", assembly="host", mono_solver="splu",
              saddle_x0=None, nu_schedule=None, on_step=None, verbose=False,
              base_level=6, dt=0.01, nu=None, U_inf=1.0, alpha=None,
              truck_band_to=None, band_cells=3, region_refine=True,
              L_ref=None, bodies=None, merged=None, reaction_band=None,
              viz_interval=None, viz_dir=None,
              viz_checkpoint_interval=None, viz_Q_thresh=0.5, viz_roi=None,
              mesh_only=False, linsolve_tol=1e-10, linsolve_tol_schedule=None,
              saddle_restart=None,
              saddle_equilibrate=False, soft_start=None, dt_schedule=None,
              saddle_fallback=None, pcd_f_inner="amgx", pcd_ap_inner="amgx",
              tau_dt=None):
    """Run the truck case: transient BDF2 monolithic march.

    Returns a history dict with keys:
      'cd'          [nsteps]  reaction Cd (canonical, truck-enclosing indicator)
      'cd_surr'     [nsteps]  surrogate-traction Cd (-p n + viscous)
      'cl_y','cl_z' [nsteps]  reaction transverse coeffs (surrogate)
      'cl_y_surr','cl_z_surr'
      'n_excluded'  int       carved (truck-intercepted) cells
      'n_slab_cut'  int       intercepted slab cells (must be 0)
      'n_cells'     int
      'nsteps'      int

    Optional viz kwargs (all default to None = no-op, byte-identical march):
      viz_interval : int or None
          Write per-frame extracts every this many steps (Q-isosurface .vtp,
          centerline slice .vtp, surface-Cp .vtp).  None disables all viz.
      viz_dir : str or pathlib.Path or None
          Output directory for viz artifacts.  Required when viz_interval
          is not None.
      viz_checkpoint_interval : int or None
          Write full .vtu every this many steps.  Defaults to
          max(1, nsteps // 5) when viz_interval is set.
      viz_Q_thresh : float
          Q-criterion isosurface threshold (default 0.5).
      viz_roi : tuple or None
          [x0,x1,y0,y1,z0,z1] ROI clip box in unit-cube coords.
      mesh_only : bool
          If True, build the mesh/carve/surrogate only and return immediately
          (no time-stepping).  Returns the same dict as the full driver with
          the mesh stats filled in and all force arrays empty/zero.  Used for
          the T4 mesh+carve probe leg.
      linsolve_tol : float
          Convergence tolerance for iterative linear solvers (default 1e-10 —
          matches solve_linear default, byte-identical to prior behaviour).
          For production transient NS at large DOF counts, 1e-5 or 1e-6 is
          sufficient and dramatically reduces iteration counts; set via the
          T4 smoke gate.
      linsolve_tol_schedule : callable or None
          t_unit -> tol callable (Baskar directive, T4b).  When given it
          OVERRIDES linsolve_tol per step: loose (e.g. 5e-4) during the Re ramp,
          tight (e.g. 1e-6) post-ramp — trivial analogue of nu_schedule.  Called
          with t_new (unit time of the step just being solved).  None (default)
          = fixed linsolve_tol, byte-identical.
      saddle_equilibrate : bool
          T4b: opt-in symmetric diagonal equilibration of the saddle inside the
          fgmres_bdiag / fused_bdiag backend (breaks the scalar-Jacobi relres
          floor caused by the ~10-order diagonal span of the Cb_f Nitsche
          penalties on fine surrogate faces).  Default False = byte-identical.
      soft_start : float or None
          TU5R: inlet amplitude ramp length in dt-units.  When set, the inflow
          x-velocity strong-BC value is scaled per step by
          soft_start_amp(t_new, soft_start, dt) = min(1, t_new/(soft_start*dt)),
          ramping the inlet from 0 to full over the first ``soft_start`` steps.
          Mitigates the impulsive-start startup pressure spike (the +201/-87
          cd_react transient).  None (default) or <=0 = off, byte-identical.
      dt_schedule : callable or None
          step index -> dt_unit for that step (the C++ dt_V startup ladder).
          A smaller startup dt raises sigma = b0/dt, restoring the
          mass-dominance that keeps plain block-Jacobi convergent through the
          developing-flow transient (T4b forensics).  The dt change is handled
          with the variable-step BDF2 table (bdf_coeffs dt_prev branch), no
          BDF1 restart; unit time accumulates (t_new = sum of per-step dt), so
          all t-based schedules (nu, tol, soft_start) stay physical-time
          consistent.  None (default) = fixed dt, byte-identical (t_new keeps
          the (step+1)*dt closed form).
      saddle_fallback : str or None
          "pcd": per-step solver fallback (five-leg Jacobi-class verdict).
          When the primary bdiag-family solve raises ConvergenceError, the
          SAME step is re-solved with fgmres_pcd (Track-A Cahouet-Chabard
          Schur preconditioner, build_pcd_meta assembled once per mesh, sigma/
          nu refreshed per use).  Easy steps keep the fast bdiag path; only
          budget-exhausted steps pay the PCD cost (plus the device->host CSR
          pull under SADDLE_DEVICE_CSR).  None (default) = byte-identical
          (failure raises as before).
      pcd_f_inner, pcd_ap_inner : str
          Inner-solve backends for the PCD fallback F/Ap blocks ("amgx"
          default — the A4/A2-validated AMGX inners; "jacobi" = dependency-
          free CG).  Only read when saddle_fallback="pcd".
      tau_dt : float or None
          Baskar directive (dt-ladder rescue): decouple tau_m's transient
          term from the marching dt.  The dt-ladder leg showed that at small
          dt the 4/dt^2 term collapses tau_m — and tau is the ONLY p-p
          coupling in the stabilized form, so the pressure block goes
          near-singular for diagonal preconditioners.  Other groups drop the
          transient term at small dt; this knob generalizes:
            None (default) : sig2tau = (2 sigma)^2 as today, byte-identical.
            > 0            : sig2tau = (2 b0/tau_dt)^2 — tau frozen at the
                             given dt (e.g. base dt during a dt-ladder
                             startup: exact once the march reaches full dt).
            0              : sig2tau = 0 — steady tau, transient term
                             dropped entirely.
          Approximates tau only; the discrete time derivative (sigma, BDF
          history) always uses the TRUE marching dt.
    """
    dim = 3
    ndof = dim + 1
    t0 = time.time()

    if alpha is None:
        alpha = cfg.cb_f              # Cb_f=20 -> Nitsche alpha (documented)
    if nu is None:
        nu = 1.0 / 100.0             # tiny-gate default Re~100

    fx = build_truck_mesh(cfg, base_level, region_refine=region_refine,
                          truck_band_to=truck_band_to, band_cells=band_cells,
                          device=device, merged=merged, bodies=bodies)
    dm, mesh, cons = fx["dm"], fx["mesh"], fx["cons"]
    scale = fx["scale"]
    merged = fx["merged"]

    if mesh_only:
        # Return mesh stats without time-stepping.
        if verbose:
            nn = len(mesh.node_coords)
            nfree_nodes = cons.T.shape[1]
            ndof_free = nfree_nodes * (dim + 1)
            print(f"[truck][mesh_only] cells={fx['n_cells']} carved={fx['n_excluded']} "
                  f"slab_cut={fx['n_slab_cut']} sf_faces={fx['sf'].elem.size} "
                  f"nodes={nn} free_nodes={nfree_nodes} free_dofs={ndof_free} "
                  f"mesh_time={time.time()-t0:.1f}s", flush=True)
        return dict(cd=np.zeros(0), cd_surr=np.zeros(0),
                    cl_y=np.zeros(0), cl_z=np.zeros(0),
                    cl_y_surr=np.zeros(0), cl_z_surr=np.zeros(0),
                    n_excluded=fx["n_excluded"], n_slab_cut=fx["n_slab_cut"],
                    n_cells=fx["n_cells"], n_rxn_nodes=0,
                    sf_faces=int(fx["sf"].elem.size), L_ref=0.0,
                    nsteps=0, viz_hook=None,
                    mesh=mesh, cons=cons, dm=dm, merged=merged, sf=fx["sf"],
                    mesh_time=time.time() - t0)

    if verbose:
        print(f"[truck] cells={fx['n_cells']} carved={fx['n_excluded']} "
              f"slab_cut={fx['n_slab_cut']} sf={fx['sf'].elem.size} "
              f"nsteps={nsteps} dt={dt} nu={nu} alpha={alpha}", flush=True)

    T = cons.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    nfree = T.shape[1]

    # ---- BCs ----------------------------------------------------------------
    bc_rows, bc_vals, masks, coords = truck_strong_bc(
        mesh, cons, ndof, dim, scale, cfg.slope_near_ground, cfg.domain_max)
    p_pin = _pressure_pin(coords, ndof, dim)

    # ---- soft-start inlet ramp bookkeeping (TU5R; opt-in) -------------------
    # Which entries of (bc_rows, bc_vals) are inflow x-velocity DOFs with a
    # nonzero target (u_x = ramp)?  Those are the only rows scaled by the
    # per-step amplitude; every other strong row (no-slip 0, slip 0, u_y/u_z=0
    # at the inlet) stays fixed.  Precomputed ONCE; None/off => empty mask.
    _inflow_x_rows = None
    if soft_start is not None and soft_start > 0:
        _inflow_nodes = np.nonzero(masks["inflow"])[0]
        _inflow_x_dofs = set(int(n) * ndof + 0 for n in _inflow_nodes)
        _bc_rows_arr = np.asarray(bc_rows, np.int64)
        _mask = np.array([int(r) in _inflow_x_dofs and abs(v) > 0.0
                          for r, v in zip(_bc_rows_arr, bc_vals)], bool)
        _inflow_x_rows = np.nonzero(_mask)[0]   # indices INTO bc_rows/bc_vals

    # ---- SBM face terms (geometry fixed; pre-assembled) ---------------------
    noslip = lambda y: np.zeros((len(y), dim))
    Af_raw, bf_raw = sbm_vector_dirichlet(
        dm, fx["sf"], fx["geo"], noslip, nu, ndof, alpha=alpha)
    Af_c = (T_vec.T @ Af_raw @ T_vec).tocsr()
    bf_c = np.asarray(T_vec.T @ bf_raw)

    _ = gauss_points(mesh, dm.tables_by_p)   # warm the GP cache

    # ---- reference force ----------------------------------------------------
    # L_ref: frontal reference length.  Default = merged truck frontal area
    # (y-extent * z-extent in unit coords) as a projected area.
    if L_ref is None:
        v = merged.verts.numpy()
        L_ref = float((v[:, 1].max() - v[:, 1].min())
                      * (v[:, 2].max() - v[:, 2].min()))
    ref_force = 0.5 * U_inf ** 2 * L_ref

    # ---- reaction arbiter setup ---------------------------------------------
    # margin: a few FINE cells around the truck bbox (fine level = band level
    # if refined, else base).  Keeps the enclosing box tight to the body.
    if reaction_band is None:
        fine_lvl = truck_band_to if truck_band_to is not None else base_level
        reaction_band = 3.0 / (1 << fine_lvl)
    rset = truck_reaction_set(mesh, cons, merged, ndof, reaction_band)
    surgery = set(int(r) for r in bc_rows); surgery.add(int(p_pin))
    w_rxn = np.zeros(nfree * ndof)
    n_rxn_nodes = 0
    for ni in rset:
        dof_ux = int(ni) * ndof + 0
        if dof_ux in surgery:
            continue                # keep w orthogonal to surgery rows
        w_rxn[dof_ux] = 1.0
        n_rxn_nodes += 1

    # ---- device-resident CSR handoff setup (opt-in: assembly="device") ------
    # P1 (T5): mirror run_flow_past_3d's device path.  The host march does a
    # per-step (A + Af).tolil() row-surgery then .tocsr() -> ~200 GB RSS at
    # 8 M DOF (LIL of a 500 M-nnz matrix).  The device path builds A_vol on
    # the GPU, atomically adds the geometry-cached SBM face system (Af_c) at
    # fixed CSR slots, applies the static strong rows on device, and hands off
    # a DeviceSaddleCSR (values stay resident) — NO host CSR, NO LIL surgery.
    #
    # BC surgery slots are STATIC (BC rows + pressure pin are fixed for the
    # mesh), so set_strong_rows() precomputes the zero-span + unit-diag plan
    # ONCE here (the W2c pattern the brief calls out) — per step is just a
    # device kernel that re-applies the plan to the fresh fill.
    _dev_asm = None
    _af_slots_d = _af_vals_d = _bf_dofs_d = _bf_vals_d = None
    _strong_rows = _Af_csr = _Af_csr_nnz = None
    _w_rxn_dofs = _w_rxn_vals = None       # sparse reaction indicator (host)
    if assembly == "device":
        from diffsim.assembly.device_assembly import DeviceNSAssembler
        from diffsim.errors import BackendError
        import warp as wp

        _dev_asm = DeviceNSAssembler(dm, ndof=ndof)   # symbolic pattern once

        # SBM face system -> fixed device slots + values (geometry-cached).
        _Af_csr = Af_c.tocsr()
        _Af_csr_nnz = _Af_csr.nnz
        _af_rows, _af_cols = _Af_csr.nonzero()
        try:
            _af_slots = _dev_asm.csr_slots(_af_rows, _af_cols)
        except BackendError as _e:
            raise ValueError(
                f"assembly='device': SBM face system (Af_c) has entries "
                f"absent from the device CSR pattern — the constraint-aware "
                f"face entries exceed the element-pair graph on this mesh. "
                f"Use assembly='host'. Original: {_e}") from _e
        _af_slots_d = wp.array(_af_slots.astype(_dev_asm._idx_np),
                               dtype=_dev_asm._idx_dtype, device=dm.device)
        _af_vals_d = wp.array(np.ascontiguousarray(_Af_csr.data, np.float64),
                              dtype=wp.float64, device=dm.device)
        _bf_nz = np.nonzero(bf_c)[0]
        _bf_dofs_d = wp.array(_bf_nz.astype(np.int32), dtype=wp.int32,
                              device=dm.device)
        _bf_vals_d = wp.array(np.ascontiguousarray(bf_c[_bf_nz], np.float64),
                              dtype=wp.float64, device=dm.device)

        # Static strong rows = BC rows + pressure pin (fixed for the mesh).
        # np.unique sorts -> canonical order for the per-step strong_b_vals.
        _strong_rows = np.unique(np.concatenate(
            [np.asarray(bc_rows, np.int64), np.array([int(p_pin)], np.int64)]))
        _dev_asm.set_strong_rows(_strong_rows)     # precompute surgery slots

        # Reaction arbiter on device: F_raw = w^T(A_vol x - b_vol).  The device
        # fill gives A_full = A_vol + Af + surgery.  w is orthogonal to surgery
        # rows, so on its (non-surgery) rows A_full = A_vol + Af and b = b_vol +
        # bf, giving F_raw = w^T(A_full x) - w^T(Af x) - w^T b + w^T bf.  Store w
        # as a sparse (dofs, vals) pair for cheap host dots against the pulled
        # A_full x and the host Af_c @ x / bf_c.
        _w_nz = np.nonzero(w_rxn)[0]
        _w_rxn_dofs = _w_nz.astype(np.int64)
        _w_rxn_vals = w_rxn[_w_nz]

    # ---- viz hook setup (opt-in; None = no-op, byte-identical march) --------
    _viz_hook = None
    if viz_interval is not None and viz_dir is not None:
        import pathlib as _pathlib
        from diffsim.viz.truck_viz import TruckVizHook as _TruckVizHook
        _ckpt_interval = (viz_checkpoint_interval
                          if viz_checkpoint_interval is not None
                          else max(1, nsteps // 5))
        _viz_hook = _TruckVizHook(
            mesh, cons, merged, fx["sf"],
            _pathlib.Path(viz_dir),
            viz_interval=int(viz_interval),
            checkpoint_interval=_ckpt_interval,
            Q_thresh=viz_Q_thresh,
            roi=viz_roi,
            U_inf=U_inf,
        )

    # ---- march --------------------------------------------------------------
    x_cur = np.zeros(nfree * ndof)
    u_pre2 = np.zeros((nfree, dim))
    u_pre1 = np.zeros((nfree, dim))

    cd = np.zeros(nsteps); cd_surr = np.zeros(nsteps)
    cl_y = np.zeros(nsteps); cl_z = np.zeros(nsteps)
    cl_y_surr = np.zeros(nsteps); cl_z_surr = np.zeros(nsteps)

    _pcd_cache = {"ndof": ndof}
    if mono_solver in ("fgmres_bdiag", "fused_bdiag"):
        _bd = {"ndof": ndof}
        if saddle_x0 is not None:
            _bd["saddle_x0"] = saddle_x0
        if saddle_restart is not None:
            _bd["saddle_restart"] = int(saddle_restart)
        if saddle_equilibrate:
            _bd["saddle_equilibrate"] = True     # T4b diagonal equilibration
        _pcd_cache[("blocktri_meta", "truck")] = _bd

    # ---- per-step PCD fallback (five-leg Jacobi-class verdict) --------------
    _fb_meta = None
    if saddle_fallback == "pcd" and mono_solver in ("fgmres_bdiag",
                                                    "fused_bdiag"):
        from diffsim.solvers.saddle_precond import build_pcd_meta
        _t_fb = time.time()
        # sigma/nu here are placeholders — refreshed from the failing step's
        # actual values before every fallback solve (Mp/Ap are geometry-only).
        _fb_meta = build_pcd_meta(dm, nu if nu is not None else 1.0, 1.0 / dt,
                                  p_pin=int(p_pin), inner=pcd_f_inner,
                                  ap_inner=pcd_ap_inner)
        _pcd_cache[("pcd_meta", "truck")] = _fb_meta
        print(f"[truck] PCD fallback armed (F={pcd_f_inner}, "
              f"Ap={pcd_ap_inner}, meta {time.time()-_t_fb:.1f}s)", flush=True)

    def _iter_solve(Acsr, b, _tol, sigma, nu_step, step):
        """Primary iterative solve with optional one-shot PCD re-solve."""
        _slv_cache = (_pcd_cache if mono_solver in
                      ("fgmres_pcd", "fgmres_bdiag", "fused_bdiag")
                      else None)
        try:
            return solve_linear(Acsr, b, solver=mono_solver, sym=False,
                                tol=_tol, device=device,
                                cache=_slv_cache, cache_key="truck")
        except Exception as e:
            if _fb_meta is None or type(e).__name__ != "ConvergenceError":
                raise
            print(f"[truck] step {step}: {mono_solver} exhausted ({e}) -> "
                  f"fgmres_pcd fallback", flush=True)
            _fb_meta["sigma"] = float(sigma)
            _fb_meta["nu"] = float(nu_step)
            _t0 = time.time()
            A_host = Acsr if sp.issparse(Acsr) else Acsr.tocsr()
            x = solve_linear(A_host, b, solver="fgmres_pcd", sym=False,
                             tol=_tol, device=device,
                             cache=_pcd_cache, cache_key="truck")
            print(f"[truck] step {step}: PCD fallback CONVERGED "
                  f"(iters={_LAST_ITERS[0]}, {time.time()-_t0:.1f}s)",
                  flush=True)
            return x

    t_cur = 0.0
    dt_prev_step = None
    for step in range(nsteps):
        dt_step = (float(dt_schedule(step)) if dt_schedule is not None
                   else dt)
        order = 1 if step == 0 else 2
        b0, b1, b2 = bdf_coeffs(order, dt_step,
                                dt_prev=(dt_prev_step if order == 2 else None))
        sigma = b0 / dt_step
        # tau_dt: decouple tau_m's transient term from the marching dt
        # (None -> assemblers default to (2 sigma)^2, byte-identical)
        _sig2tau = None
        if tau_dt is not None:
            _sig2tau = 0.0 if tau_dt <= 0 else (2.0 * b0 / tau_dt) ** 2
        # closed form when no schedule (byte-identical to prior behaviour);
        # accumulated sum under a schedule (physical time stays consistent)
        t_new = ((step + 1) * dt if dt_schedule is None
                 else t_cur + dt_step)
        nu_step = float(nu_schedule(t_new)) if nu_schedule is not None else nu

        aq, dq = _gp_field(dm, mesh, T, u_pre1, dim)
        if order == 1:
            fq_raw = {}
            for pv in dm.bins:
                tb = dm.tables_by_p[pv]
                vals = np.asarray(T @ u_pre1)[mesh.conn_of[pv]]
                fq_raw[pv] = (np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
                              / dt_step)
        else:
            fq_raw = _gp_history_fq(dm, mesh, T, u_pre1, u_pre2, b1, b2,
                                    dt_step, dim)

        # If nu changes, the SBM face system must be reassembled (it scales nu).
        if nu_schedule is not None:
            Af_raw, bf_raw = sbm_vector_dirichlet(
                dm, fx["sf"], fx["geo"], noslip, nu_step, ndof, alpha=alpha)
            Af_c = (T_vec.T @ Af_raw @ T_vec).tocsr()
            bf_c = np.asarray(T_vec.T @ bf_raw)
            if _dev_asm is not None:
                # refresh geometry-cached device SBM values/rhs (pattern fixed)
                _Af_csr = Af_c.tocsr()
                assert _Af_csr.nnz == _Af_csr_nnz, (
                    f"Af_c sparsity changed under nu ramp: {_Af_csr.nnz} vs "
                    f"{_Af_csr_nnz}")
                _af_vals_d = wp.array(
                    np.ascontiguousarray(_Af_csr.data, np.float64),
                    dtype=wp.float64, device=dm.device)
                _bf_nz = np.nonzero(bf_c)[0]
                _bf_dofs_d = wp.array(_bf_nz.astype(np.int32),
                                      dtype=wp.int32, device=dm.device)
                _bf_vals_d = wp.array(
                    np.ascontiguousarray(bf_c[_bf_nz], np.float64),
                    dtype=wp.float64, device=dm.device)

        _tol = (float(linsolve_tol_schedule(t_new))
                if linsolve_tol_schedule is not None else linsolve_tol)

        # Soft-start: scale the inflow x-velocity strong-BC values this step.
        # bc_vals_step == bc_vals when soft_start is off (byte-identical).
        if _inflow_x_rows is not None:
            _amp = soft_start_amp(t_new, soft_start, dt)
            bc_vals_step = np.asarray(bc_vals, np.float64).copy()
            bc_vals_step[_inflow_x_rows] *= _amp
        else:
            bc_vals_step = bc_vals

        if assembly == "device":
            # ---- Device-resident CSR handoff path (P1) ----------------------
            # A_vol on device + atomic-add Af_c at fixed slots + static strong
            # rows.  SADDLE_DEVICE_CSR=1 keeps the ~19 GB values resident
            # (assemble_handoff); the fgmres_bdiag / fused_bdiag saddle paths
            # in solve_linear consume vals_d directly.  Read the env live so
            # parity harnesses can toggle it (default byte-identical to host).
            _val_of = {int(r): float(v) for r, v in zip(bc_rows, bc_vals_step)}
            _val_of[int(p_pin)] = 0.0
            _sb = np.array([_val_of[int(r)] for r in _strong_rows])
            _dev_csr = os.environ.get(
                "SADDLE_DEVICE_CSR", "0").strip() not in ("", "0")
            _asm_call = (_dev_asm.assemble_handoff if _dev_csr
                         else _dev_asm.assemble)
            Acsr, b = _asm_call(
                aq, dq, fq_raw, nu_step, sigma,
                sig2tau=_sig2tau,
                strong_b_vals=_sb,
                extra_matrix=(_af_slots_d, _af_vals_d),
                extra_rhs=(_bf_dofs_d, _bf_vals_d))
            if mono_solver == "splu":
                x_cur = splu(Acsr.tocsc()).solve(b)
            else:
                _LAST_ITERS[0] = None
                x_cur = _iter_solve(Acsr, b, _tol, sigma, nu_step, step)

            # Reaction arbiter (device): F_raw = w^T(A_full x) - w^T(Af x)
            #                                    - w^T b + w^T bf
            _op = Acsr.device_operator() if hasattr(Acsr, "device_operator") \
                else _dev_asm.device_operator()
            _x_d = wp.array(np.ascontiguousarray(x_cur, np.float64),
                            dtype=wp.float64, device=dm.device)
            _Ax_d = wp.zeros(_dev_asm.Nfull, dtype=wp.float64,
                             device=dm.device)
            _op.matvec(_x_d, _Ax_d)
            _Ax = _Ax_d.numpy()
            _wAx = float(_w_rxn_vals @ _Ax[_w_rxn_dofs])
            _wAfx = float(_w_rxn_vals @ (Af_c @ x_cur)[_w_rxn_dofs])
            _wb = float(_w_rxn_vals @ b[_w_rxn_dofs])
            _wbf = float(_w_rxn_vals @ bf_c[_w_rxn_dofs])
            F_raw = _wAx - _wAfx - _wb + _wbf
        else:
            # ---- Host assembly path (default; bit-for-bit unchanged) --------
            A, b = assemble_linear_ns(dm, aq, dq, fq_raw, nu_step, sigma=sigma,
                                      sig2tau=_sig2tau)
            _A_vol = A.tocsr()      # pre-SBM, pre-surgery (reaction arbiter)
            _b_vol = b.copy()
            A = (A + Af_c).tolil()
            b = b + bf_c
            for r, v in zip(bc_rows, bc_vals_step):
                A.rows[r] = [int(r)]; A.data[r] = [1.0]; b[r] = v
            A.rows[p_pin] = [p_pin]; A.data[p_pin] = [1.0]; b[p_pin] = 0.0

            Acsr = A.tocsr()
            if mono_solver == "splu":
                x_cur = splu(Acsr.tocsc()).solve(b)
            else:
                _LAST_ITERS[0] = None
                x_cur = _iter_solve(Acsr, b, _tol, sigma, nu_step, step)
            # consistent reaction (x-drag): F_raw = w^T (A_vol x - b_vol)
            F_raw = (float(np.asarray(_A_vol.T @ w_rxn) @ x_cur)
                     - float(w_rxn @ _b_vol))

        u_new = x_cur.reshape(nfree, ndof)[:, :dim]
        x_all = np.asarray(T_vec @ x_cur)

        # surrogate-traction force
        Fs = surrogate_traction(dm, fx["sf"], fx["geo"], x_all, nu_step, ndof)
        cd_surr[step] = Fs[0] / ref_force
        cl_y_surr[step] = Fs[1] / ref_force
        cl_z_surr[step] = Fs[2] / ref_force

        cd[step] = -F_raw / ref_force
        # reuse surrogate transverse for cl (reaction indicator is x-only)
        cl_y[step] = cl_y_surr[step]
        cl_z[step] = cl_z_surr[step]

        u_pre2 = u_pre1.copy()
        u_pre1 = u_new.copy()
        t_cur = t_new
        dt_prev_step = dt_step

        if verbose:
            print(f"[truck] step {step:3d}  Cd_react={cd[step]:+.4f}  "
                  f"Cd_surr={cd_surr[step]:+.4f}", flush=True)
        if on_step is not None:
            # T4b: expose the RAW (un-normalized) forces + the shared ref_force
            # so the harness can arithmetically diagnose the cd_surr magnitude
            # (Fs[0] is the raw surrogate x-traction; F_raw is the raw reaction;
            # both are divided by the SAME ref_force to form cd).
            on_step(step, dict(cd=cd[step], cd_surr=cd_surr[step],
                               F_surr_raw=float(Fs[0]),
                               F_react_raw=float(-F_raw),
                               ref_force=float(ref_force),
                               x=x_cur))
        if _viz_hook is not None:
            _viz_hook(step, dict(x=x_cur))

    if _viz_hook is not None:
        _viz_hook.write_time_average()

    if verbose:
        print(f"[truck] done in {time.time()-t0:.1f}s", flush=True)

    return dict(cd=cd, cd_surr=cd_surr, cl_y=cl_y, cl_z=cl_z,
                cl_y_surr=cl_y_surr, cl_z_surr=cl_z_surr,
                n_excluded=fx["n_excluded"], n_slab_cut=fx["n_slab_cut"],
                n_cells=fx["n_cells"], n_rxn_nodes=n_rxn_nodes,
                sf_faces=int(fx["sf"].elem.size), L_ref=float(L_ref),
                nsteps=nsteps, viz_hook=_viz_hook)


def make_nu_schedule(cfg, U_inf=1.0, L_ref=1.0, scale=None):
    """Time-based Re ramp -> nu(t) callable, from Re_V / Re_ramping.

    Piecewise-linear Re(t) through (Re_ramping[i], Re_V[i]); nu = U*L/Re.

    UNIT-SYSTEM MAPPING (T4b — the recurring #1 bug class; see the campaign
    doc's unit table).  The C++ prior art computes on the PHYSICAL channel
    frame domain=[16,2,2] with Coe_diff = 1/Re, dt=0.01, ramp times in that
    same frame (Re_ramping = physical seconds).  Our octree remaps coordinates
    to the unit cube via an ISOTROPIC scale s = domain_scale = 1/16
    (x_unit = s * x_phys).  Under this pure spatial rescaling with U held at 1:

        nu_unit = s * nu_phys = s / Re      (Re preserved: Re = U*L/nu, and the
                                             feature length s*L keeps Re fixed)
        t_unit  = s * t_phys                (convection term consistency)

    So when ``scale`` is given, this returns nu in UNIT-CUBE units as a function
    of UNIT time: it maps t_unit -> t_phys = t_unit/s to interpolate Re, then
    returns nu_unit = s * L_ref / Re.  The driver must correspondingly pass
    dt in unit time (dt_unit = s * dt_phys).  ``scale=None`` (legacy) keeps the
    old physical-frame mapping (nu = U*L/Re, t interpreted directly) — kept only
    for back-compat; the truck smoke passes scale=cfg.domain_scale.
    """
    ts = np.asarray(cfg.re_ramping, np.float64)
    res = np.asarray(cfg.re_v, np.float64)
    s = 1.0 if scale is None else float(scale)

    def nu_of_t(t):
        # t is UNIT time when scale given; map back to physical for the ramp.
        t_phys = t / s
        Re = float(np.interp(t_phys, ts, res))
        Re = max(Re, 1e-12)
        return s * U_inf * L_ref / Re

    return nu_of_t


if __name__ == "__main__":
    from diffsim.cases.truck_config import load_truck_config
    cfgp = os.environ.get("TRUCK_CONFIG", os.path.join(
        os.path.dirname(__file__), "..", "local_code_old",
        "truck_4case_fresh_inputs", "NewRun-no-shell-slope0p25", "config.txt"))
    cfg = load_truck_config(cfgp)
    base_level = int(os.environ.get("BASE_LEVEL", "6"))
    nsteps = int(os.environ.get("NSTEPS", "3"))
    res = run_truck(cfg, nsteps=nsteps, base_level=base_level,
                    truck_band_to=int(os.environ.get("BAND_TO", str(base_level + 2))),
                    verbose=True)
    print("Cd_react", res["cd"])
    print("Cd_surr", res["cd_surr"])
