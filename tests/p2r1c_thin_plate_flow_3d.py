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

from diffsim.octree.build import build_uniform, refine_elements
from diffsim.octree.balance import balance2to1
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
from diffsim.solvers.linsolve import solve_linear, _LAST_ITERS


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


def _build_shell_3d(level, x_c, y_c, z_c, half_y, half_z,
                    refine_to=None, band_cells=2, device="cpu"):
    """Build dim=3 octree (uniform or adaptive) + two-sided shell surrogate.

    refine_to=None  -> uniform octree at ``level`` (current behavior, unchanged).
    refine_to=int   -> adaptive octree: uniform base at ``level``, progressively
                       refined near the FiniteSheet plate to ``refine_to``.

    Returns a dict with dm, mesh, cons, sfp, gp, sfm, gm, n_excluded,
    and (when adaptive) n_nodes, n_hanging, build_time.
    """
    sheet = _make_sheet(x_c, y_c, z_c, half_y, half_z)
    extra = {}
    if refine_to is None:
        tree = build_uniform(level, dim=3)
        ret, intercepted = classify_shell_intercepted(tree, sheet)
        mesh = build_mesh(ret, p=1)
        cons = build_constraints(mesh)
        n_excluded = int(intercepted.sum())
    else:
        amr = build_adaptive_plate_mesh(level, refine_to, sheet,
                                        band_cells=band_cells)
        mesh = amr["mesh"]
        cons = amr["cons"]
        ret = amr["ret"]
        n_excluded = amr["n_excluded"]
        extra = dict(n_nodes=amr["n_nodes"], n_hanging=amr["n_hanging"],
                     build_time=amr["build_time"])
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), device)
    ftab = face_tables(1, 3)
    (sfp, gp), (sfm, gm) = extract_two_sided_surrogate(ret, sheet, ftab)
    return dict(dm=dm, mesh=mesh, cons=cons, sfp=sfp, gp=gp, sfm=sfm, gm=gm,
                n_excluded=n_excluded, **extra)


def build_adaptive_plate_mesh(base_level, refine_to, plate_geom, band_cells=2):
    """Build an adaptive 3-D octree mesh refined near a FiniteSheet plate.

    Starting from a uniform octree at ``base_level``, iterates level by level
    from ``base_level+1`` up to ``refine_to``, each time refining cells whose
    center is within ``band_cells * h_local`` of the plate (measured by
    ``plate_geom.psi()`` on element centers), then applies ``balance2to1``.
    After all refinement passes, runs ``classify_shell_intercepted`` to exclude
    plate-intercepted cells, then builds the FEM mesh and constraints on the
    post-exclusion tree.

    Parameters
    ----------
    base_level : int
        Starting uniform octree level (e.g. 4 = 16^3 cells).
    refine_to : int
        Target refinement level near the plate.  Must be >= base_level.
    plate_geom : FiniteSheet
        Plate geometry; ``psi()`` is called on element centers to measure
        distance.
    band_cells : int, optional
        Half-width of the refinement band in units of the *local* cell size
        at the refinement level being applied.  Default 2 keeps the band thin.

    Returns
    -------
    dict with keys:
      'mesh'        : the built FEM mesh (on post-exclusion tree)
      'cons'        : hanging-node constraints (on post-exclusion tree)
      'ret'         : classified tree (post cell-exclusion) from classify_shell_intercepted
      'n_nodes'     : total node count (post-exclusion — actual solver mesh)
      'n_hanging'   : number of hanging nodes (post-exclusion)
      'n_excluded'  : number of plate-intercepted cells excluded
      'build_time'  : wall-clock seconds for the entire mesh build
    """
    import torch
    t0 = time.time()
    tree = build_uniform(base_level, dim=3)
    for _ in range(base_level + 1, refine_to + 1):
        centers = tree.centers()                        # [N, 3] in [0,1]^3
        # psi() expects a torch tensor
        pts_t = torch.tensor(centers, dtype=torch.float64)
        psi_vals = plate_geom.psi(pts_t).detach().numpy()  # [N] unsigned distance
        h_local = tree.h()                              # [N] current cell sizes
        # refine cells close enough to the plate
        mask = psi_vals < band_cells * h_local
        if not mask.any():
            break
        tree = refine_elements(tree, mask)
        tree = balance2to1(tree)
    # Exclude plate-intercepted cells so n_nodes/n_hanging reflect the actual solver mesh
    ret, intercepted = classify_shell_intercepted(tree, plate_geom)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    n_nodes = len(mesh.node_coords)
    n_hanging = int(cons.hanging.sum())
    build_time = time.time() - t0
    return dict(
        mesh=mesh,
        cons=cons,
        ret=ret,
        n_nodes=n_nodes,
        n_hanging=n_hanging,
        n_excluded=int(intercepted.sum()),
        build_time=build_time,
    )


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
    refine_to=None,
    band_cells=2,
    _two_sided=True,
    _return_fields=False,
    mono_solver="splu",
    device="cpu",
    assembly="host",   # assembly backend: "host" (default, bit-for-bit) | "device"
    solver_stats=None,  # optional list to append per-step iteration counts (A2 ladder)
    pcd_inner="jacobi",  # (T4) PCD F-block inner-solve backend: "jacobi" | "amgx"
    pcd_ap_inner="jacobi",  # (T5) PCD Ap-block inner-solve backend: "jacobi" | "amgx"
    saddle_restart=None,  # A3 knob A: FGMRES restart length; None => default 60
    saddle_x0=None,       # A3 knob B: warm-start mode; "extrap" | None (cold)
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
    mono_solver : str
        Monolithic solve backend: "splu" (host LU), "cudss" (GPU direct),
        or "fused" (GPU BiCGStab). Default "splu" preserves legacy behavior.
    device : str
        Device for non-splu backends: "cpu" or "cuda"/"hip" for GPU.
        Default "cpu".
    assembly : str
        Assembly backend: "host" (default, bit-for-bit host path) or
        "device" (DeviceNSAssembler; symbolic pattern once per mesh epoch,
        numeric fill on device per step, two-sided SBM face system via cached slots).
        "host" default keeps all existing tests bit-for-bit unchanged.
    solver_stats : list or None
        Optional list to which per-step iteration counts are appended when the
        backend provides them (i.e. when mono_solver is an iterative backend such
        as "fgmres_bdiag" that writes to ``_LAST_ITERS``).  One integer is
        appended per step.  Default None => no collection, byte-for-byte identical
        march (no overhead).  Used by the A2 iteration-ladder harness.

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
                         plate_half_y, plate_half_z,
                         refine_to=refine_to, band_cells=band_cells, device=device)
    dm, mesh, cons = fx["dm"], fx["mesh"], fx["cons"]

    if verbose:
        mode = f"adaptive(base={level},refine_to={refine_to})" if refine_to else f"uniform(L{level})"
        print(f"[p2r1c-3d] mesh={mode}  n_excluded={fx['n_excluded']}  "
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

    # ---- Device assembler setup (once per mesh epoch) -----------------------
    # Builds symbolic pattern + slot maps; caches SBM face system slots so only
    # VALUE arrays need refreshing per step (pattern fixed; geometry is static).
    # The two-sided SBM face system (Af_c/bf_c) is just a host CSR assembled
    # before the loop — identical slot treatment to the 2-D driver.
    _dev_asm = None           # DeviceNSAssembler (None => host path)
    _af_slots_d = None        # device CSR slots for Af_c
    _af_vals_d = None         # device values array for Af_c (geometry-cached)
    _bf_dofs_d = None         # device dof indices for bf_c nonzeros
    _bf_vals_d = None         # device values array for bf_c nonzeros
    _strong_rows = None       # sorted unique strong rows (set once per epoch)
    _Af_csr = None            # CSR form of Af_c (cached for value refresh)
    _Af_csr_nnz = None        # nnz of Af_csr (to assert pattern unchanged)

    if assembly == "device":
        from diffsim.assembly.device_assembly import DeviceNSAssembler
        from diffsim.errors import BackendError
        import warp as wp

        # No pre-emptive gate for adaptive (hanging-node) meshes: the
        # DeviceNSAssembler's constraint-aware weighted scatter expands
        # element entries THROUGH the constraint weights, producing the same
        # free-dof pattern as T^T K T.  The try/except below is the honesty
        # net: if a future mesh or Af_c variant genuinely exceeds the pattern,
        # it is reported clearly rather than silently skipped.

        _dev_asm = DeviceNSAssembler(dm)    # symbolic pattern once per epoch

        # --- SBM face system -> fixed-pattern device slots (cached) ----------
        # Af_c is geometry-cached (assembled once before the loop);
        # upload slots + values once; no per-step reallocation needed.
        _Af_csr = Af_c.tocsr()
        _Af_csr_nnz = _Af_csr.nnz
        _af_rows, _af_cols = _Af_csr.nonzero()
        try:
            _af_slots = _dev_asm.csr_slots(_af_rows, _af_cols)
        except BackendError as _e:
            raise ValueError(
                f"assembly='device': SBM face system (Af_c) contains entries "
                f"absent from the device assembler's CSR pattern. This is a "
                f"REAL finding: the constraint-aware face-system entries on this "
                f"mesh exceed the element-pair graph. Use assembly='host'. "
                f"Original error: {_e}") from _e

        # Cast slots to the assembler's index dtype (int32 for small meshes)
        _af_slots_np = _af_slots.astype(_dev_asm._idx_np)
        _af_slots_d = wp.array(_af_slots_np, dtype=_dev_asm._idx_dtype,
                               device=dm.device)
        _af_vals_d = wp.array(
            np.ascontiguousarray(_Af_csr.data, np.float64),
            dtype=wp.float64, device=dm.device)

        # bf_c sparse: only nonzero dofs uploaded.
        # int32 dof indices: _scatter_vec_kernel requires gdof: wp.array(dtype=wp.int32)
        # (API contract in device_assembly.py).  On meshes where Nfull >= 2^31
        # the assembler itself refuses at construction, so int32 is always safe here.
        _bf_nz = np.nonzero(bf_c)[0]
        _bf_dofs_d = wp.array(_bf_nz.astype(np.int32), dtype=wp.int32,
                              device=dm.device)
        _bf_vals_d = wp.array(
            np.ascontiguousarray(bf_c[_bf_nz], np.float64),
            dtype=wp.float64, device=dm.device)

        # --- strong rows: BCs + pressure pin (no inflow kick in 3-D) --------
        # np.unique sorts the rows, giving a canonical order we'll use for
        # strong_b_vals per step.  set_strong_rows preserves input order, so
        # the sorted unique array is consistent with the per-step value lookup.
        # 3-D has NO symmetry-breaking perturbation kick — strong rows are
        # outer BC rows + pressure pin only.
        _strong_rows = np.unique(
            np.concatenate([np.asarray(bc_rows, np.int64),
                            np.array([int(p_pin)], np.int64)]))
        _dev_asm.set_strong_rows(_strong_rows)

    # ---- Reference force normalization (plate area) -------------------------
    plate_area = 4.0 * plate_half_y * plate_half_z
    ref_force = 0.5 * U_inf ** 2 * plate_area

    # ---- PCD meta cache (fgmres_pcd only) -----------------------------------
    # build_pcd_meta is called once per BDF order (sigma changes at step 0->1).
    # The cache dict is passed to solve_linear so make_pcd_apply sees pcd_meta.
    # fgmres_bdiag also reads ndof from ("blocktri_meta", key); wire it here too
    # so the 3-D bdiag backend gets ndof=4 (not the default 3).
    _pcd_cache = {"ndof": ndof}   # carry ndof for fgmres_bdiag (blocktri_meta slot)
    _pcd_last_order = None        # track when to rebuild pcd_meta (sigma change)
    if mono_solver == "fgmres_bdiag":
        # fgmres_bdiag reads ndof (and A3 knobs) via ("blocktri_meta", cache_key)
        _bdiag_meta = {"ndof": ndof}
        if saddle_restart is not None:
            _bdiag_meta["saddle_restart"] = int(saddle_restart)
        if saddle_x0 is not None:
            _bdiag_meta["saddle_x0"] = saddle_x0
        _pcd_cache[("blocktri_meta", "ns3d")] = _bdiag_meta

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

        # Rebuild pcd_meta when BDF order (and thus sigma) changes.
        # This is at most 2 builds per run (BDF1 -> BDF2 at step 1).
        if mono_solver == "fgmres_pcd" and order != _pcd_last_order:
            from diffsim.solvers.saddle_precond import build_pcd_meta
            _pcd_cache[("pcd_meta", "ns3d")] = build_pcd_meta(
                dm, nu, sigma, p_pin=p_pin, inner=pcd_inner,
                ap_inner=pcd_ap_inner)
            _pcd_last_order = order

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

        if assembly == "device":
            # ---- Device assembly path ---------------------------------------
            # Strong values in _strong_rows ORDER (np.unique-sorted order).
            # Build a dict from all strong-row values, then index by _strong_rows.
            _val_of = {}
            for _r, _v in zip(bc_rows, bc_vals):
                _val_of[int(_r)] = float(_v)
            # Pressure pin
            _val_of[int(p_pin)] = 0.0
            # Build strong_b_vals in sorted unique row order (_strong_rows)
            # every strong row is bc_rows or the pin — fail loudly if not
            _sb = np.array([_val_of[int(_r)] for _r in _strong_rows])

            # Af_c is geometry-cached (assembled once before the loop); its
            # value array is constant each step.  Use the pre-uploaded
            # _af_vals_d directly — no per-step reallocation needed.
            # (The pattern-unchanged assert is a safety guard for future
            # callers that might pass a per-step Af_c.)
            assert _Af_csr.nnz == _Af_csr_nnz, (
                f"Af_c sparsity changed mid-march: {_Af_csr.nnz} vs "
                f"{_Af_csr_nnz}. Cannot refresh device values safely.")

            # assemble: volume fill + extra_matrix(Af) + extra_rhs(bf)
            # + strong rows — order mirrors host: A_vol + Af_c, b + bf_c,
            # then LIL surgery.  Oracle: aq/dq/fq as flat pv-keyed dicts.
            Acsr, b = _dev_asm.assemble(
                aq, dq, fq_raw, nu, sigma,
                strong_b_vals=_sb,
                extra_matrix=(_af_slots_d, _af_vals_d),
                extra_rhs=(_bf_dofs_d, _bf_vals_d))

            # Solve (same routing as host path)
            if mono_solver == "splu":
                x_cur = splu(Acsr.tocsc()).solve(b)
            else:
                _LAST_ITERS[0] = None
                _slv_cache = (_pcd_cache
                              if mono_solver in ("fgmres_pcd", "fgmres_bdiag")
                              else None)
                x_cur = solve_linear(Acsr, b, solver=mono_solver, sym=False,
                                     device=device,
                                     cache=_slv_cache, cache_key="ns3d")
                if solver_stats is not None and _LAST_ITERS[0] is not None:
                    solver_stats.append(int(_LAST_ITERS[0]))

        else:
            # ---- Host assembly path (default; bit-for-bit unchanged) --------
            # Assemble monolithic NS
            A, b = assemble_linear_ns(dm, aq, dq, fq_raw, nu, sigma=sigma)
            A = (A + Af_c).tolil()
            b = b + bf_c

            # Apply strong Dirichlet BCs
            for r, v in zip(bc_rows, bc_vals):
                A.rows[r] = [int(r)]; A.data[r] = [1.0]; b[r] = v

            # Pressure pin
            A.rows[p_pin] = [p_pin]; A.data[p_pin] = [1.0]; b[p_pin] = 0.0

            # Solve — routed through solve_linear so MONO_SOLVER/DEVICE select
            # the backend (splu host | cudss GPU-direct | fused GPU-BiCGStab).
            # Matrix changes every step (Picard convection).
            # For fgmres_pcd the pcd_meta is constant per BDF order (mesh ops);
            # for fgmres_bdiag the blocktri_meta carries ndof=4 for the 3-D case.
            Acsr = A.tocsr()
            if mono_solver == "splu":
                x_cur = splu(Acsr.tocsc()).solve(b)      # legacy path, bit-for-bit
            else:
                _LAST_ITERS[0] = None
                _slv_cache = (_pcd_cache
                              if mono_solver in ("fgmres_pcd", "fgmres_bdiag")
                              else None)
                x_cur = solve_linear(Acsr, b, solver=mono_solver, sym=False,
                                     device=device,
                                     cache=_slv_cache, cache_key="ns3d")
                if solver_stats is not None and _LAST_ITERS[0] is not None:
                    solver_stats.append(int(_LAST_ITERS[0]))

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


def report_adaptive_sizes(
    base_level=4,
    refine_levels=(5, 6, 7, 8, 9),
    plate_xc=0.375,
    plate_yc=0.5,
    plate_zc=0.5,
    plate_half_y=0.125,
    plate_half_z=0.125,
    band_cells=2,
):
    """Build-only node-count probe for adaptive meshes near the plate.

    For each refine_to in refine_levels, builds the adaptive mesh (no solve)
    and prints a table of refine_to, n_nodes, n_hanging, build_time.
    Returns a list of dicts with those fields.

    Example::

        python -c "from p2r1c_thin_plate_flow_3d import report_adaptive_sizes; report_adaptive_sizes()"
    """
    sheet = _make_sheet(plate_xc, plate_yc, plate_zc, plate_half_y, plate_half_z)
    print(f"{'refine_to':>10}  {'n_nodes':>10}  {'n_hanging':>10}  {'build_time(s)':>14}")
    print("-" * 52)
    rows = []
    for refine_to in refine_levels:
        try:
            r = build_adaptive_plate_mesh(base_level, refine_to, sheet,
                                          band_cells=band_cells)
            print(f"{refine_to:>10}  {r['n_nodes']:>10}  {r['n_hanging']:>10}  "
                  f"{r['build_time']:>14.2f}")
            rows.append(dict(refine_to=refine_to, n_nodes=r["n_nodes"],
                             n_hanging=r["n_hanging"],
                             build_time=r["build_time"]))
        except Exception as exc:
            print(f"{refine_to:>10}  ERROR: {exc}")
            rows.append(dict(refine_to=refine_to, error=str(exc)))
    return rows


if __name__ == "__main__":
    from diffsim.viz.results import save_flow_run

    level        = int(os.environ.get("LEVEL", "4"))
    base_level   = int(os.environ.get("BASE_LEVEL", str(level)))
    refine_level_str = os.environ.get("REFINE_LEVEL", "")
    refine_level = int(refine_level_str) if refine_level_str else None
    nsteps       = int(os.environ.get("NSTEPS", "5"))
    dt           = float(os.environ.get("DT", "0.01"))
    nu           = float(os.environ.get("NU", "0.1"))
    U_inf        = float(os.environ.get("U_INF", "1.0"))
    plate_xc     = float(os.environ.get("PLATE_XC", "0.375"))
    plate_yc     = float(os.environ.get("PLATE_YC", "0.5"))
    plate_zc     = float(os.environ.get("PLATE_ZC", "0.5"))
    plate_half_y = float(os.environ.get("PLATE_HALF_Y", "0.125"))
    plate_half_z = float(os.environ.get("PLATE_HALF_Z", "0.125"))
    mono_solver = os.environ.get("MONO_SOLVER", "splu")
    device      = os.environ.get("DEVICE", "cpu")
    assembly    = os.environ.get("ASSEMBLY", "host")

    re_approx = int(round(U_inf / nu)) if nu > 0 else 0
    if refine_level:
        case_name = f"p2r1c_3d_re{re_approx}_L{base_level}_r{refine_level}"
    else:
        case_name = f"p2r1c_3d_re{re_approx}_L{base_level}"

    res = run_flow_past_3d(
        level=base_level, nsteps=nsteps, dt=dt, nu=nu, U_inf=U_inf,
        plate_xc=plate_xc, plate_yc=plate_yc, plate_zc=plate_zc,
        plate_half_y=plate_half_y, plate_half_z=plate_half_z,
        verbose=True,
        refine_to=refine_level,
        _return_fields=True,
        mono_solver=mono_solver,
        device=device,
        assembly=assembly,
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
