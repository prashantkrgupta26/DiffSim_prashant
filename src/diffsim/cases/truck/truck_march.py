"""TRUCK case march loop: transient BDF2 monolithic driver.

Moved verbatim from tests/truck_flow.py.  Functions: _gp_field,
_gp_history_fq, run_truck, make_nu_schedule.
"""
import os
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from diffsim.sbm.vector import sbm_vector_dirichlet, surrogate_traction
from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.physics.poisson import gauss_points
from diffsim.solvers.timestepping import bdf_coeffs
from diffsim.solvers.linsolve import solve_linear, _LAST_ITERS
from diffsim.errors import ConvergenceError

from .truck_mesh import build_truck_mesh
from .truck_bc import (truck_strong_bc, _pressure_pin, soft_start_amp,
                       truck_reaction_set)


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
              saddle_restart=None, saddle_min_work=False,
              saddle_equilibrate=False, soft_start=None, dt_schedule=None,
              saddle_fallback=None, pcd_f_inner="amgx", pcd_ap_inner="amgx",
              tau_dt=None, accept_miss_until=None, u_cap=50.0,
              sbm_start_step=None, tau_m_scale=1.0, carve_lam=1.0,
              nonlin_iters=1, nonlin_tol=1e-3,
              slope_near_ground=None,
              ground_refine_to=None, ground_band=0.0156,
              walls_refine_to=None, carve_delta=None,
              seal_underbody=False, seal_y=0.003, seal_boxes=None,
              backflow_stab=False,
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
          RESTRICTION: only supported with ``mono_solver="fgmres_bdiag"``; any
          other solver raises ``ValueError`` at march start.
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
                          ground_band=ground_band,
                          carve_delta=carve_delta,
                          walls_refine_to=walls_refine_to,
                          seal_underbody=seal_underbody, seal_y=seal_y,
                          seal_boxes=seal_boxes)
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

    # ---- outlet backflow stabilization (C++ BACKFLOW_STAB, lumped) ----------
    # NSEquation.h:2589: Ae += -0.5 N_a min(0, u.n) N_b on outlet faces,
    # Picard-linearized about the current state.  Lumped first cut: an
    # approximate nodal area A_i (kNN spacing squared, see below); per step add
    # -0.5*min(0, a_x)*A_i to the three velocity diagonals of outlet nodes
    # (outlet normal = +x, so u.n = u_x).  Off by default = byte-identical.
    _obf_rows = _obf_area = None
    if backflow_stab:
        _coords_all = mesh.node_coords[cons.free_nodes]
        _x_max = _coords_all[:, 0].max()
        _on_outlet = np.abs(_coords_all[:, 0] - _x_max) < 1e-10
        _out_nodes = np.where(_on_outlet)[0]
        # robust: use per-node local h from nearest outlet cell size — approximate
        # with the finest wall cell area (walls-refined): h_loc via node spacing
        _area = np.zeros(len(_out_nodes))
        if len(_out_nodes):
            # per-node area ~ (kNN nearest-neighbor spacing)^2: use the y/z grid
            # of outlet nodes to estimate local spacing per node.  This is an
            # O(1) approximation — ~2x off at 2:1 interface nodes — and is
            # adequate as a stabilization coefficient (not a geometric area).
            _yz = _coords_all[_out_nodes][:, 1:]
            from scipy.spatial import cKDTree as _KD
            _kd = _KD(_yz)
            _dd, _ = _kd.query(_yz, k=2)
            _hloc = _dd[:, 1]
            _area = _hloc ** 2
        _obf_rows = (_out_nodes[:, None] * ndof
                     + np.arange(dim)[None, :]).ravel().astype(np.int64)
        _obf_area = np.repeat(_area, dim)
        # host/device cd_react parity: the reaction indicator w_rxn must be zero
        # on all backflow rows — the host path captures A_vol pre-backflow while
        # the device path subtracts Af and bf but not the backflow diagonal.
        # Correctness requires that w_rxn has zero support on outlet nodes.
        assert np.all(w_rxn[_obf_rows] == 0.0), (
            "w_rxn is nonzero on backflow rows — reaction bbox overlaps outlet; "
            "host/device cd_react parity violated")
        print(f"[truck] backflow stabilization armed: {len(_out_nodes)} "
              f"outlet nodes", flush=True)

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

        # Backflow-stab diagonal slots (outlet velocity dofs), device path.
        _obf_slots_d = _obf_comb_slots_d = None
        if _obf_rows is not None:
            _obf_slots = _dev_asm.csr_slots(_obf_rows, _obf_rows)
            _obf_slots_d = wp.array(_obf_slots.astype(_dev_asm._idx_np),
                                    dtype=_dev_asm._idx_dtype, device=dm.device)
            _obf_comb_slots = np.concatenate([_af_slots, _obf_slots])
            _obf_comb_slots_d = wp.array(
                _obf_comb_slots.astype(_dev_asm._idx_np),
                dtype=_dev_asm._idx_dtype, device=dm.device)

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

    if accept_miss_until is not None and mono_solver != "fgmres_bdiag":
        raise ValueError(
            "accept_miss_until is only supported with mono_solver="
            "'fgmres_bdiag' (the fused_bdiag branch raises strictly; "
            "final-review I-1)")
    _pcd_cache = {"ndof": ndof}
    # _bd always exists (on_step reads last_solve_miss from it); it is only
    # installed as the solver meta for the bdiag solvers below.
    _bd = {"ndof": ndof}
    if mono_solver in ("fgmres_bdiag", "fused_bdiag"):
        if saddle_x0 is not None:
            _bd["saddle_x0"] = saddle_x0
        if saddle_restart is not None:
            _bd["saddle_restart"] = int(saddle_restart)
        if saddle_equilibrate:
            _bd["saddle_equilibrate"] = True     # T4b diagonal equilibration
        if saddle_min_work:
            _bd["saddle_min_work"] = True        # drift-guard polish cycle
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
            if _fb_meta is None or not isinstance(e, ConvergenceError):
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
        # the step's final state DID meet tol — clear the primary's miss flag
        _bd["last_solve_miss"] = False
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
            # mesh-compatibility guard (final-review I-2): plain resume is
            # STRICT — a changed mesh must go through interpolate_checkpoint
            if "nfree" in _ck and int(_ck["nfree"]) != nfree:
                raise ValueError(
                    f"resume checkpoint mesh mismatch: ckpt nfree="
                    f"{int(_ck['nfree'])} vs current {nfree} — mesh knobs "
                    "changed; use interpolate_checkpoint for mesh-sequenced "
                    "restarts")
            if "n_cells" in _ck and int(_ck["n_cells"]) != fx["n_cells"]:
                raise ValueError(
                    f"resume checkpoint mesh mismatch: ckpt n_cells="
                    f"{int(_ck['n_cells'])} vs current {fx['n_cells']} — "
                    "mesh knobs changed; use interpolate_checkpoint for "
                    "mesh-sequenced restarts")
            if _ck["x_cur"].shape != (nfree * ndof,):
                raise ValueError(
                    f"resume checkpoint shape mismatch: {_ck['x_cur'].shape}"
                    f" vs {(nfree * ndof,)}")
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
            # seed the warm-start cache (both slots when checkpointed —
            # final-review I-4 — so the first resumed solve extrapolates)
            if "x_prev" in _ck and _ck["x_prev"].size:
                _pcd_cache[("bdiag_x_prev", "truck")] = _ck["x_prev"].copy()
                if _ck["x_prev2"].size:
                    _pcd_cache[("bdiag_x_prev2", "truck")] = \
                        _ck["x_prev2"].copy()
            else:
                _pcd_cache[("bdiag_x_prev", "truck")] = x_cur.copy()
            # time-base guard (final-review I-3): with dt_schedule=None the
            # loop uses the closed form t=(step+1)*dt, which silently
            # discards the checkpointed t_cur — require consistency so a
            # dt/ladder mismatch between legs fails loudly
            if dt_schedule is None and abs(t_cur - _step0 * dt) > 1e-9:
                raise ValueError(
                    f"resume time-base mismatch: checkpoint t={t_cur} but "
                    f"(step0={_step0})*dt={_step0 * dt} — the original leg "
                    "used a different dt/dt_schedule; rerun with matching "
                    "knobs")
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
        # ONLY the fgmres_bdiag branch honors it (final-review I-1).
        if accept_miss_until is not None and mono_solver == "fgmres_bdiag":
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
                # outlet backflow stabilization (Picard: u.n = u_x from
                # the advection iterate at outlet nodes)
                _extra_m = (_af_slots_d, _af_vals_d) if _sbm_on else None
                if _obf_rows is not None:
                    _ax = u_iter[(_obf_rows[::dim] // ndof), 0]
                    _bfv = (np.repeat(-0.5 * np.minimum(0.0, _ax), dim)
                            * _obf_area)
                    if _sbm_on:
                        _cv = np.concatenate([_Af_csr.data, _bfv])
                        _extra_m = (_obf_comb_slots_d, wp.array(
                            np.ascontiguousarray(_cv, np.float64),
                            dtype=wp.float64, device=dm.device))
                    else:
                        _extra_m = (_obf_slots_d, wp.array(
                            np.ascontiguousarray(_bfv, np.float64),
                            dtype=wp.float64, device=dm.device))
                Acsr, b = _asm_call(
                    aq, dq, fq_raw, nu_step, sigma,
                    sig2tau=_sig2tau, tau_scale=float(tau_m_scale),
                    strong_b_vals=_sb,
                    extra_matrix=_extra_m,
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
                if _obf_rows is not None:
                    _ax = u_iter[(_obf_rows[::dim] // ndof), 0]
                    _bfv = (np.repeat(-0.5 * np.minimum(0.0, _ax), dim)
                            * _obf_area)
                    A = A + sp.csr_matrix((_bfv, (_obf_rows, _obf_rows)),
                                          shape=A.shape)
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
            _xp = _pcd_cache.get(("bdiag_x_prev", "truck"))
            _xp2 = _pcd_cache.get(("bdiag_x_prev2", "truck"))
            np.savez(_tmp, step=step, t_cur=t_cur,
                     dt_prev=dt_prev_step,
                     x_cur=x_cur, u_pre1=u_pre1, u_pre2=u_pre2,
                     cd=cd, cd_surr=cd_surr, cl_y=cl_y, cl_z=cl_z,
                     cl_y_surr=cl_y_surr, cl_z_surr=cl_z_surr,
                     nfree=nfree, n_cells=fx["n_cells"],
                     # warm-start history (final-review I-4): both slots so
                     # the first resumed iterative solve extrapolates
                     x_prev=(_xp if _xp is not None else np.zeros(0)),
                     x_prev2=(_xp2 if _xp2 is not None else np.zeros(0)))
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
                               accepted_miss=bool(_bd.get("last_solve_miss",
                                                          False)),
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
