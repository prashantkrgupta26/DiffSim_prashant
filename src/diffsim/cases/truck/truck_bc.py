"""TRUCK case boundary conditions (C++ TRUCK branch, task amendment 3).

Moved verbatim from tests/truck_flow.py.  Functions: truck_bc_masks,
truck_strong_bc, _pressure_pin, soft_start_amp, truck_reaction_set.
"""
import numpy as np


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
