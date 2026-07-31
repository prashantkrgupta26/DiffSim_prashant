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

from diffsim.octree.build import build_uniform, refine_elements, Octree
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



def refine_ground(tree, target_lvl, height):
    """Refine cells whose center lies within ``height`` of the ground (y=0)
    up to ``target_lvl`` — the C++ refine_walls heritage (Baskar: refine near
    the floor).  Resolves the near-ground shear layer (the sloped-inlet
    profile rises over y<0.0156 = TWO base-level cells unrefined — the
    u-probe wave nursery)."""
    for _ in range(64):
        centers = tree.centers()
        lvl = tree.levels.astype(np.int64)
        mask = (centers[:, 1] < height) & (lvl < target_lvl)
        if not mask.any():
            break
        tree = refine_elements(tree, mask)
        tree = balance2to1(tree)
    return tree


def _face_components(tree):
    """Cell connected components under FACE adjacency (Baskar: two cells are
    the same fluid domain only if they SHARE A FACE — corner/edge contact is
    not a flow passage).  Mirrors extract_surrogate's neighbor probe: for
    each face, sub-face quarter probes catch finer neighbors; the reverse
    direction catches coarser ones."""
    from itertools import product as _iproduct
    from diffsim.octree import morton as _morton
    from diffsim.octree.lookup import LeafLookup as _LeafLookup, \
        face_offsets as _face_offsets
    dim = tree.dim
    lk = _LeafLookup(tree)
    L = _morton.lmax(dim)
    anchors = tree.anchors()
    size = (1 << (L - tree.levels.astype(np.int64)))
    center = anchors + size[:, None] // 2
    offs = _face_offsets(dim)
    tang = np.array(list(_iproduct((-1, 1), repeat=dim - 1)), np.int64)
    n = len(tree)
    src, dst = [], []
    for f in range(2 * dim):
        ax = f // 2
        off = offs[f]
        tang_axes = [d for d in range(dim) if d != ax]
        base = center + off[None, :] * (size[:, None] // 2 + 1)
        for combo in tang:
            probe = base.copy()
            for j, d in enumerate(tang_axes):
                probe[:, d] += combo[j] * (size // 4)
            nb = lk.find(probe)
            m = nb >= 0
            src.append(np.where(m)[0])
            dst.append(nb[m])
    g = sp.coo_matrix((np.ones(sum(len(x) for x in src)),
                       (np.concatenate(src), np.concatenate(dst))),
                      shape=(n, n))
    ncomp, labels = sp.csgraph.connected_components(g, directed=False)
    return ncomp, labels


def flood_fill_retain(ret):
    """Single-fluid-domain retention (Baskar directive): drop every face-
    connected component except the largest.  The 22-body truck assembly
    encloses internal cavities (engine bay, cab, tank gaps) that the carve
    otherwise retains as isolated "fluid" pockets — each carries its own
    pressure nullspace (only one global pin exists), making the saddle
    singular (the measured underbody instability + solver floor).  Removing
    a pocket exposes no new main-domain faces (no shared face by
    definition), so the surrogate extraction is unaffected."""
    ncomp, labels = _face_components(ret)
    if ncomp <= 1:
        return ret, 0, 0
    sizes = np.bincount(labels)
    main = int(np.argmax(sizes))
    keepm = labels == main
    n_dropped = int((~keepm).sum())
    print(f"[truck] flood-fill: dropped {n_dropped} cells in {ncomp - 1} "
          f"enclosed pocket(s) (face-adjacency); single fluid domain = "
          f"{int(sizes[main])} cells", flush=True)
    return (Octree(ret.keys[keepm], ret.levels[keepm], dim=ret.dim,
                   periodic=ret.periodic), ncomp - 1, n_dropped)


def build_truck_mesh(cfg, base_level, region_refine=True, truck_band_to=None,
                     band_cells=3, device="cpu", merged=None,
                     bodies=None, carve_lam=1.0,
                     ground_refine_to=None, ground_band=0.0156):
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

    # 2b. ground refine (C++ refine_walls; Baskar directive)
    if ground_refine_to is not None:
        n0 = len(tree)
        tree = refine_ground(tree, int(ground_refine_to), float(ground_band))
        print(f"[truck] ground refine: lvl>={ground_refine_to} within "
              f"y<{ground_band} ({n0} -> {len(tree)} cells)", flush=True)

    # 3. truck-band refine
    if truck_band_to is not None:
        tree = refine_truck_band(tree, merged, band_cells, truck_band_to)

    n_before = len(tree)

    # 4. truck carve (flow AROUND the solid: domain="outside").  carve_lam
    # picks the intercepted-cell convention: 1.0 KEEPS cut cells (surrogate
    # hugs Gamma from outside — RatioGPSBM default); 0.0 REMOVES them (the
    # ThinShell paper's T~h := {T : T cap Gamma = 0} and the C++ carve).
    # u-probe forensics: with the truck touching the ground (position clips
    # y to 0.000), lam=1.0 retains PINCHED SLIVER cells along the whole
    # contact line — mostly-inside-solid cells squeezed between strong truck
    # dofs and strong ground dofs — the measured epicenter of the underbody
    # instability.  lam=0.0 removes them (paper-faithful).
    ret, _frac = classify_lambda(tree, merged, lam=float(carve_lam),
                                 domain="outside")

    # 5. flood-fill: single fluid domain (face-adjacency components)
    ret, n_pockets, n_pocket_cells = flood_fill_retain(ret)
    n_excluded = n_before - len(ret)

    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), device)
    geo = GeometryData.evaluate(merged, ret, sf, face_tables(1, 3),
                                domain="outside")
    return dict(dm=dm, mesh=mesh, cons=cons, sf=sf, geo=geo, merged=merged,
                scale=scale, n_excluded=int(n_excluded), n_slab_cut=int(n_slab_cut),
                n_cells=len(ret), n_pockets=int(n_pockets),
                n_pocket_cells=int(n_pocket_cells))


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
              tau_dt=None, accept_miss_until=None, u_cap=50.0,
              sbm_start_step=None, tau_m_scale=1.0, carve_lam=1.0,
              nonlin_iters=1, nonlin_tol=1e-3,
              slope_near_ground=None,
              ground_refine_to=None, ground_band=0.0156,
              checkpoint_interval=None, checkpoint_dir=None,
              resume=False):
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
      accept_miss_until : int or None
          Baskar directive (T5): solver misses during the initial transient
          are acceptable.  For steps < this value, a budget-exhausted primary
          solve ACCEPTS the truncated iterate (logged "[saddle] ACCEPT-MISS"
          with achieved relres) instead of raising / falling back; from this
          step on, strict semantics (raise -> optional PCD fallback) return.
          Guarded by ``u_cap`` so drift cannot masquerade as progress.
          None (default) = strict everywhere, byte-identical.
      u_cap : float
          March blow-up sentinel (STATE-based, leg-6 lesson): after every
          step, |u|_inf must be finite and below this cap (cd_react must be
          finite).  cd_react itself is NOT capped — it is a residual
          functional and conflates solve error with physics under an
          accepted miss.  U_inf-normalized flow: legitimate startup peaks
          are O(1-5); default 50.
      sbm_start_step : int or None
          Baskar staged-BC strategy: for steps < this value, the truck is
          the CARVED-OUT geometry with STRONG no-slip (identity rows on
          every velocity dof supporting the surrogate boundary = the
          nonzero rows of Af_c; the paper's 4.8 treatment) and the SBM
          face system is NOT assembled; from this step on, the strong
          truck rows are released and the true-geometry SBM Nitsche
          system takes over.  The CSR pattern is the superset (built once);
          the device strong-row plan is re-set once at the switch.  Put
          the switch inside the accept_miss window so the O(h) boundary
          shift's transition kick is absorbed.  Phase-1 cd_react/cd_surr
          are DIAGNOSTIC ONLY (the reaction indicator overlaps the
          phase-1 surgery rows).  None (default) = SBM from step 0,
          byte-identical.
      tau_m_scale : float
          C++ tauM_scale heritage (NSEquation.h:530; config key the loader
          previously warned-ignored — the truck case runs 0.1): direct
          multiplier on tauM in the volume kernels; tauC = 1/(tauM*gg)
          computed from the SCALED tauM inherits the inverse (C++-exact).
          The u-probe forensics motivated honoring it: a spurious
          near-ground wave born in the 2-cell sloped-inlet shear layer
          grows ~x1.2/step under tau_m_scale=1 and detonates at the truck;
          the C++ ran this exact case at 0.1.  Default 1.0 byte-identical.
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
                          device=device, merged=merged, bodies=bodies,
                          carve_lam=carve_lam,
                          ground_refine_to=ground_refine_to,
                          ground_band=ground_band)
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
    # slope_near_ground=None -> config value; 0/inf-like -> UNIFORM inlet
    # (the paper 4.8 BC: u_inf=(1,0,0); the slope is this config variant's
    # addition and its 2-base-cell shear layer is the u-probe wave nursery).
    _slope = (cfg.slope_near_ground if slope_near_ground is None
              else float(slope_near_ground))
    if _slope <= 0:
        _slope = 1e-12          # min(y/slope,1) -> 1 everywhere: uniform
    bc_rows, bc_vals, masks, coords = truck_strong_bc(
        mesh, cons, ndof, dim, scale, _slope, cfg.domain_max)
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

    # ---- staged BC (Baskar): phase-1 strong truck rows ----------------------
    # The velocity dofs supporting the surrogate boundary are exactly the
    # nonzero rows of the (condensed) SBM face system.  Pressure rows are NOT
    # strongified (only velocity no-slip).
    _truck_rows = None
    if sbm_start_step is not None and int(sbm_start_step) > 0:
        _af_sup = np.unique(Af_c.nonzero()[0])
        _truck_rows = _af_sup[_af_sup % ndof != dim].astype(np.int64)
        print(f"[truck] staged BC: strong carved-out no-slip on "
              f"{len(_truck_rows)} velocity dofs until step "
              f"{int(sbm_start_step)}, then SBM", flush=True)

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
        _strong_rows_base = np.unique(np.concatenate(
            [np.asarray(bc_rows, np.int64), np.array([int(p_pin)], np.int64)]))
        # Staged BC: phase 1 additionally strongifies the truck's surrogate-
        # boundary velocity dofs; the plan is re-set once at the switch step.
        if _truck_rows is not None:
            _strong_rows = np.unique(np.concatenate(
                [_strong_rows_base, _truck_rows]))
        else:
            _strong_rows = _strong_rows_base
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
        """Primary iterative solve with optional one-shot PCD re-solve.

        The fallback solve runs OUTSIDE the except block: solving inside it
        keeps the caught exception's traceback alive, which pins the failed
        primary solve's entire Krylov workspace (~7.6 GB at restart=120 on
        the 7.96M truck) through the fallback — the measured cause of the
        fallback-time VRAM OOM.  gc.collect() then actually releases it
        before the PCD allocations."""
        _slv_cache = (_pcd_cache if mono_solver in
                      ("fgmres_pcd", "fgmres_bdiag", "fused_bdiag")
                      else None)
        _fb_msg = None
        try:
            return solve_linear(Acsr, b, solver=mono_solver, sym=False,
                                tol=_tol, device=device,
                                cache=_slv_cache, cache_key="truck")
        except Exception as e:
            if _fb_meta is None or type(e).__name__ != "ConvergenceError":
                raise
            _fb_msg = str(e)
        # ---- fallback path (traceback released, workspace reclaimable) ----
        import gc
        gc.collect()
        print(f"[truck] step {step}: {mono_solver} exhausted ({_fb_msg}) -> "
              f"fgmres_pcd fallback", flush=True)
        _fb_meta["sigma"] = float(sigma)
        _fb_meta["nu"] = float(nu_step)
        _t0 = time.time()
        # Pass Acsr as-is: the fgmres_pcd branch reuses a DeviceSaddleCSR's
        # resident SpMV for the outer matvec (no 19 GB duplicate upload) and
        # pulls host values only for the preconditioner block extraction.
        # Bounded rescue: cap the fallback budget so a non-converging rescue
        # fails in ~1 h, not an unbounded grind (leg-3 lesson: jacobi-inner
        # PCD at the transient crest ran >5 h toward a 12000-iter default).
        _fb_maxiter = int(os.environ.get("PCD_FALLBACK_MAXITER", "3000"))
        x = solve_linear(Acsr, b, solver="fgmres_pcd", sym=False,
                         tol=_tol, device=device, maxiter=_fb_maxiter,
                         cache=_pcd_cache, cache_key="truck")
        print(f"[truck] step {step}: PCD fallback CONVERGED "
              f"(iters={_LAST_ITERS[0]}, {time.time()-_t0:.1f}s)",
              flush=True)
        return x

    t_cur = 0.0
    dt_prev_step = None
    _step0 = 0
    _ckpt_dir = None
    if checkpoint_dir is not None:
        import pathlib as _pl
        _ckpt_dir = _pl.Path(checkpoint_dir)
        _ckpt_dir.mkdir(parents=True, exist_ok=True)
    if resume and _ckpt_dir is not None:
        _cands = sorted(_ckpt_dir.glob("march_ckpt_*.npz"),
                        key=lambda q: q.stat().st_mtime)
        if _cands:
            _ck = np.load(_cands[-1])
            _step0 = int(_ck["step"]) + 1
            t_cur = float(_ck["t_cur"])
            dt_prev_step = (float(_ck["dt_prev"])
                            if np.isfinite(_ck["dt_prev"]) else None)
            x_cur = _ck["x_cur"].copy()
            u_pre1 = _ck["u_pre1"].copy()
            u_pre2 = _ck["u_pre2"].copy()
            _n0 = min(_step0, nsteps)
            cd[:_n0] = _ck["cd"][:_n0]; cd_surr[:_n0] = _ck["cd_surr"][:_n0]
            cl_y[:_n0] = _ck["cl_y"][:_n0]; cl_z[:_n0] = _ck["cl_z"][:_n0]
            cl_y_surr[:_n0] = _ck["cl_y_surr"][:_n0]
            cl_z_surr[:_n0] = _ck["cl_z_surr"][:_n0]
            # seed the warm-start cache so the first resumed solve is warm
            _pcd_cache[("bdiag_x_prev", "truck")] = x_cur.copy()
            print(f"[truck] RESUME from {_cands[-1].name}: step {_step0}, "
                  f"t={t_cur:.6f}", flush=True)
        else:
            print("[truck] resume requested but no checkpoint found — "
                  "starting from step 0", flush=True)
    for step in range(_step0, nsteps):
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

        # Accept-miss window (Baskar T5): per-step flag on the bdiag meta.
        if accept_miss_until is not None and mono_solver in (
                "fgmres_bdiag", "fused_bdiag"):
            _pcd_cache[("blocktri_meta", "truck")]["saddle_accept_miss"] = \
                bool(step < int(accept_miss_until))

        # Staged BC (Baskar): phase flag + one-time switch to SBM.
        _sbm_on = (sbm_start_step is None) or (step >= int(sbm_start_step))
        if (_truck_rows is not None and step == int(sbm_start_step)):
            if _dev_asm is not None:
                _strong_rows = _strong_rows_base
                _dev_asm.set_strong_rows(_strong_rows)
            print(f"[truck] staged BC: SWITCH to SBM at step {step} "
                  f"(strong truck rows released)", flush=True)

        # Soft-start: scale the inflow x-velocity strong-BC values this step.
        # bc_vals_step == bc_vals when soft_start is off (byte-identical).
        if _inflow_x_rows is not None:
            _amp = soft_start_amp(t_new, soft_start, dt)
            bc_vals_step = np.asarray(bc_vals, np.float64).copy()
            bc_vals_step[_inflow_x_rows] *= _amp
        else:
            bc_vals_step = bc_vals

        # C++ iterMaxBlock heritage: Picard sub-iterations — the
        # advection field re-linearized about the current iterate.
        u_iter = u_pre1
        _nl_done = 1
        for _nl in range(max(1, int(nonlin_iters))):
            aq, dq = _gp_field(dm, mesh, T, u_iter, dim)
            if assembly == "device":
                # ---- Device-resident CSR handoff path (P1) ----------------------
                # A_vol on device + atomic-add Af_c at fixed slots + static strong
                # rows.  SADDLE_DEVICE_CSR=1 keeps the ~19 GB values resident
                # (assemble_handoff); the fgmres_bdiag / fused_bdiag saddle paths
                # in solve_linear consume vals_d directly.  Read the env live so
                # parity harnesses can toggle it (default byte-identical to host).
                _val_of = {int(r): float(v) for r, v in zip(bc_rows, bc_vals_step)}
                _val_of[int(p_pin)] = 0.0
                # staged BC phase 1: truck surrogate-boundary rows -> 0.0 (the
                # .get default); phase 2 uses the base row set (no truck rows).
                _sb = np.array([_val_of.get(int(r), 0.0) for r in _strong_rows])
                _dev_csr = os.environ.get(
                    "SADDLE_DEVICE_CSR", "0").strip() not in ("", "0")
                _asm_call = (_dev_asm.assemble_handoff if _dev_csr
                             else _dev_asm.assemble)
                Acsr, b = _asm_call(
                    aq, dq, fq_raw, nu_step, sigma,
                    sig2tau=_sig2tau, tau_scale=float(tau_m_scale),
                    strong_b_vals=_sb,
                    extra_matrix=((_af_slots_d, _af_vals_d) if _sbm_on else None),
                    extra_rhs=((_bf_dofs_d, _bf_vals_d) if _sbm_on else None))
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
                _wb = float(_w_rxn_vals @ b[_w_rxn_dofs])
                if _sbm_on:
                    _wAfx = float(_w_rxn_vals @ (Af_c @ x_cur)[_w_rxn_dofs])
                    _wbf = float(_w_rxn_vals @ bf_c[_w_rxn_dofs])
                    F_raw = _wAx - _wAfx - _wb + _wbf
                else:
                    # staged phase 1: no Af/bf in A_full; DIAGNOSTIC only (the
                    # indicator overlaps the phase-1 truck surgery rows).
                    F_raw = _wAx - _wb
            else:
                # ---- Host assembly path (default; bit-for-bit unchanged) --------
                A, b = assemble_linear_ns(dm, aq, dq, fq_raw, nu_step, sigma=sigma,
                                          sig2tau=_sig2tau,
                                          tau_scale=float(tau_m_scale))
                _A_vol = A.tocsr()      # pre-SBM, pre-surgery (reaction arbiter)
                _b_vol = b.copy()
                if _sbm_on:
                    A = (A + Af_c).tolil()
                    b = b + bf_c
                else:
                    A = A.tolil()       # staged phase 1: no SBM face system
                for r, v in zip(bc_rows, bc_vals_step):
                    A.rows[r] = [int(r)]; A.data[r] = [1.0]; b[r] = v
                A.rows[p_pin] = [p_pin]; A.data[p_pin] = [1.0]; b[p_pin] = 0.0
                if not _sbm_on:
                    # staged phase 1: strong no-slip on the carved-out boundary
                    for r in _truck_rows:
                        A.rows[r] = [int(r)]; A.data[r] = [1.0]; b[r] = 0.0

                Acsr = A.tocsr()
                if mono_solver == "splu":
                    x_cur = splu(Acsr.tocsc()).solve(b)
                else:
                    _LAST_ITERS[0] = None
                    x_cur = _iter_solve(Acsr, b, _tol, sigma, nu_step, step)
                # consistent reaction (x-drag): F_raw = w^T (A_vol x - b_vol)
                F_raw = (float(np.asarray(_A_vol.T @ w_rxn) @ x_cur)
                         - float(w_rxn @ _b_vol))


            _u_it = x_cur.reshape(nfree, ndof)[:, :dim]
            _nl_done = _nl + 1
            if int(nonlin_iters) <= 1:
                break
            _dn = float(np.linalg.norm(_u_it - u_iter))
            _un = max(float(np.linalg.norm(_u_it)), 1e-300)
            u_iter = _u_it
            if _dn / _un < float(nonlin_tol):
                break

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

        # Blow-up sentinel (guards the accept-miss window; always active):
        # drift/instability must fail LOUDLY, never masquerade as progress.
        # STATE-based (leg-6 lesson): cd_react is a residual functional and
        # conflates solve error with physics under a miss — the honest
        # measure is the velocity magnitude itself (U_inf-normalized flow;
        # legitimate startup peaks are O(1-5), instabilities run away).
        _umax = float(np.abs(u_new).max())
        if not np.isfinite(cd[step]) or not np.isfinite(_umax) or \
                _umax > u_cap:
            raise RuntimeError(
                f"[truck] MARCH BLOW-UP at step {step}: |u|_inf={_umax:.3e} "
                f"(cap {u_cap}), cd_react={cd[step]} — aborting (accept-miss "
                f"window is not a license to drift)")

        u_pre2 = u_pre1.copy()
        u_pre1 = u_new.copy()
        t_cur = t_new
        dt_prev_step = dt_step

        if (_ckpt_dir is not None and checkpoint_interval
                and (step + 1) % int(checkpoint_interval) == 0):
            _slot = (step // int(checkpoint_interval)) % 2
            _tmp = _ckpt_dir / f".march_ckpt_{_slot}.tmp.npz"
            np.savez(_tmp, step=step, t_cur=t_cur,
                     dt_prev=(dt_prev_step if dt_prev_step is not None
                              else np.nan),
                     x_cur=x_cur, u_pre1=u_pre1, u_pre2=u_pre2,
                     cd=cd, cd_surr=cd_surr, cl_y=cl_y, cl_z=cl_z,
                     cl_y_surr=cl_y_surr, cl_z_surr=cl_z_surr)
            _tmp.rename(_ckpt_dir / f"march_ckpt_{_slot}.npz")
            print(f"[truck] checkpoint @ step {step} -> "
                  f"march_ckpt_{_slot}.npz", flush=True)

        if verbose:
            print(f"[truck] step {step:3d}  Cd_react={cd[step]:+.4f}  "
                  f"Cd_surr={cd_surr[step]:+.4f}", flush=True)
        if on_step is not None:
            # T4b: expose the RAW (un-normalized) forces + the shared ref_force
            # so the harness can arithmetically diagnose the cd_surr magnitude
            # (Fs[0] is the raw surrogate x-traction; F_raw is the raw reaction;
            # both are divided by the SAME ref_force to form cd).
            # state telemetry (instability forensics): |u|_inf + its location
            _uarg = int(np.argmax(np.abs(u_new)))
            _unode = _uarg // dim
            try:
                _uloc = tuple(float(c) for c in coords[_unode])
            except Exception:
                _uloc = None
            on_step(step, dict(cd=cd[step], cd_surr=cd_surr[step],
                               F_surr_raw=float(Fs[0]),
                               F_react_raw=float(-F_raw),
                               ref_force=float(ref_force),
                               umax=_umax, umax_loc=_uloc,
                               nonlin=_nl_done,
                               coords=coords, u=u_new,
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
