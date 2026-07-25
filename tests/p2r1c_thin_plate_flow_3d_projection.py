"""P2-R1c — 3-D flow past a finite thin plate: PROJECTION stepper + gpu_cg PPE.

Mirror of tests/p2r1c_thin_plate_flow_3d.py (the MONOLITHIC driver) but
marched via LeraySBMShellStepper (two-sided shell SBM + projection split)
with solver="gpu_cg" for the SPD pressure-Poisson sub-problem.

SCALABLE PATH: the PPE Laplacian K_p is SPD; gpu_cg (dist_cg.pcg + SerialComm,
torch.sparse_csr CG with Jacobi preconditioner) solves it iteratively without
factorization — the escape from the host splu wall at L9-near-plate scale
(~186k nodes) that the monolithic host-splu cannot reach. The predictor
(nonsymmetric Oseen) still uses splu; for full L9 scaling, predictor_solver
can be "fused" or AMGX (deferred follow-on).

⚠️ 3-D PHYSICS CAVEAT (read before trusting the Cd):
    This driver lands SCALABLE INFRASTRUCTURE, not yet faithful 3-D physics.
    The lagged-pressure projection split has a DOCUMENTED 3-D defect: with an
    open outflow, the momentum predictor cannot build the driving stagnation
    pressure from rest, so the split settles into a weak/wrong steady state
    (the monolithic steady state is NOT a fixed point of the lagged-p* split;
    ‖L - K_p‖/‖K_p‖ ≈ 0.67 — projection K_p and monolithic PSPG enforce
    different discrete incompressibility). The 3-D Cd produced here is FINITE
    but NOT PHYSICALLY FAITHFUL (it can be wrong-signed and grow under the
    startup transient). The correct 3-D engine is the MONOLITHIC SBM-NS saddle
    (tests/p2r1c_thin_plate_flow_3d.py). The projection path's physics fix
    (consistent PPE operator + outflow-BC / pressure-correction p' co-design)
    is a SEPARATE research track — see the `p2-r2a-monolithic-pivot` memory
    verdict and docs/dev/2026-07-23-projection-ladder-verdict.md. The gpu_cg
    PPE lever wired here is exactly what that fix will scale on.

WHAT THIS DRIVER GUARANTEES (and its smoke gate asserts):
    - runs end-to-end on a two-sided thin-plate shell,
    - produces a FINITE Cd,
    - the PPE ran on gpu_cg and converged,
    - the two-sided shell coupling is load-bearing (two-sided ≠ one-sided).

Quick smoke run (L3, 3 steps):
    .venv/bin/python tests/p2r1c_thin_plate_flow_3d_projection.py

GH200 run (L6, 200 steps, gpu_cg on the PPE):
    LEVEL=6 NSTEPS=200 DT=0.005 NU=0.004 PPE_SOLVER=gpu_cg \\
        .venv/bin/python tests/p2r1c_thin_plate_flow_3d_projection.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.sbm.surrogate import (
    classify_shell_intercepted, extract_two_sided_surrogate)
from diffsim.steppers.leray_sbm import LeraySBMShellStepper

# Reuse the monolithic driver's geometry/BC helpers (single source of truth)
from p2r1c_thin_plate_flow_3d import (
    _make_sheet, build_adaptive_plate_mesh, triangulate_finite_sheet,
)


def _build_shell_proj_3d(level, x_c, y_c, z_c, half_y, half_z,
                         refine_to=None, band_cells=2):
    """Build dim=3 octree + two-sided shell surrogate for the projection driver.

    Mirrors _build_shell_3d from the monolithic driver: uniform octree at
    ``level`` (refine_to=None) or adaptive plate refinement (refine_to=int).
    Returns dict with dm, mesh, cons, sfp, gp, sfm, gm, n_excluded.
    """
    sheet = _make_sheet(x_c, y_c, z_c, half_y, half_z)
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
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), "cpu")
    ftab = face_tables(1, 3)
    (sfp, gp), (sfm, gm) = extract_two_sided_surrogate(ret, sheet, ftab)
    return dict(dm=dm, mesh=mesh, cons=cons, sfp=sfp, gp=gp, sfm=sfm, gm=gm,
                n_excluded=n_excluded)


def _outer_bc_masks_3d(mesh, cons, U_inf):
    """Strong outer BCs for 3-D flow past a plate, in the free-node-major
    (strong_mask, u_inf) convention LeraySBMShellStepper expects:
      - inflow (x=x_min) + 4 lateral walls: u = (U_inf, 0, 0) [freestream]
      - outflow (x=x_max): do-nothing (not in the strong set)

    Returns (strong_mask [n_free] bool, u_inf [n_free, 3] float, pin_node int)
    where pin_node is the free-node index of the outflow-low-back corner used
    to pin the pressure correction (p'=0) in the PPE.
    """
    dim = 3
    coords = mesh.node_coords[cons.free_nodes]
    tol = 1e-10
    x_min = coords[:, 0].min()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    inflow = np.abs(coords[:, 0] - x_min) < tol
    walls = (
        (np.abs(coords[:, 1] - y_min) < tol) |
        (np.abs(coords[:, 1] - y_max) < tol) |
        (np.abs(coords[:, 2] - z_min) < tol) |
        (np.abs(coords[:, 2] - z_max) < tol)
    )
    strong_mask = inflow | walls
    u_inf = np.zeros((len(coords), dim))
    u_inf[strong_mask, 0] = U_inf                    # freestream u_x = U_inf
    # outflow-low-back corner (max x, min y, min z) for the pressure pin
    pin_node = int(np.argmax(coords[:, 0] - coords[:, 1] - coords[:, 2]))
    return strong_mask, u_inf, pin_node


def run_flow_past_3d_projection(
    level=3,
    nsteps=3,
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
    ppe_solver="gpu_cg",
    predictor_solver="splu",
    picard_iters=2,
    order=2,
    _two_sided=True,
    _return_fields=False,
    _return_stepper=False,
):
    """Run 3-D flow past a finite thin plate via the PROJECTION stepper.

    Uses LeraySBMShellStepper (two-sided shell SBM) with ppe_solver
    (default "gpu_cg") for the SPD pressure-Poisson sub-problem and
    predictor_solver (default "splu") for the nonsymmetric Oseen predictor.

    Parameters
    ----------
    level : int
        Octree refinement level (each direction). Level 3 = 8^3 = 512 cells.
    nsteps : int
        Number of BDF steps (BDF1 bootstrap then BDF2 when order=2).
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
        Print per-step Cd/Cl_y/Cl_z and the PPE solver in use.
    refine_to : int or None
        If set, adaptive octree refinement to this level near the plate.
    band_cells : int
        Half-width of the refinement band in local cell sizes (default 2).
    ppe_solver : str
        Solver for the SPD pressure-Poisson sub-problem (default "gpu_cg").
    predictor_solver : str
        Solver for the nonsymmetric Oseen predictor (default "splu").
        The predictor is NOT SPD, so gpu_cg is invalid here.
    picard_iters : int
        Picard iterations for the predictor (default 2).
    order : int
        BDF order target (default 2).
    _two_sided : bool
        Internal — False assembles only Gamma~- (drops Gamma~+); the
        anti-vacuity lever for the load-bearing smoke check.
    _return_fields : bool
        Internal — True also returns 'mesh' and 'node_fields'.
    _return_stepper : bool
        Internal — True also returns the live 'stepper' (for the smoke
        gate to introspect the solver / convergence).

    Returns a dict with:
      'cd'         : np.ndarray [nsteps] — drag coefficient (x-direction)
      'cl_y'       : np.ndarray [nsteps] — transverse y force coefficient
      'cl_z'       : np.ndarray [nsteps] — transverse z force coefficient
      'n_excluded' : int — excluded cells (non-zero confirms plate active)
      'nsteps'     : int — steps taken
      'ppe_solver' : str — the PPE solver used (confirms gpu_cg was requested)

    ⚠️ The Cd here is FINITE but NOT physically faithful in 3-D — see the
    module docstring's projection-split caveat.
    """
    dim = 3
    ndof = dim + 1
    t0 = time.time()

    # ---- geometry + two-sided surrogate ------------------------------------
    fx = _build_shell_proj_3d(level, plate_xc, plate_yc, plate_zc,
                              plate_half_y, plate_half_z,
                              refine_to=refine_to, band_cells=band_cells)
    dm, mesh, cons = fx["dm"], fx["mesh"], fx["cons"]

    if verbose:
        mode = (f"adaptive(base={level},refine_to={refine_to})"
                if refine_to else f"uniform(L{level})")
        print(f"[p2r1c-3d-proj] mesh={mode}  n_excluded={fx['n_excluded']}  "
              f"sfp={fx['sfp'].elem.size}  sfm={fx['sfm'].elem.size}  "
              f"nsteps={nsteps}  dt={dt}  nu={nu}  "
              f"ppe_solver={ppe_solver}", flush=True)

    # ---- BCs in the (strong_mask, u_inf) free-node convention --------------
    strong_mask, u_inf_arr, pin_node = _outer_bc_masks_3d(mesh, cons, U_inf)

    def f_fn(x, t):
        return np.zeros((len(x), dim))

    # ---- one-sided anti-vacuity lever: drop Gamma~+ ------------------------
    if _two_sided:
        sfp, gp = fx["sfp"], fx["gp"]
    else:
        # Empty Gamma~+ side: the assembler cannot take a zero-face set
        # (M1a one-face-order invariant), so fold BOTH sides' faces onto the
        # minus side by reusing sfm/gm alone — the one-sided assembly the
        # monolithic driver uses for its own anti-vacuity check.
        sfp, gp = fx["sfm"], fx["gm"]

    # ---- projection stepper: two-sided shell + gpu_cg PPE ------------------
    st = LeraySBMShellStepper(
        sfp, gp, fx["sfm"], fx["gm"],
        dm, nu, dt, f_fn,
        u_inf=u_inf_arr, strong_mask=strong_mask,
        order=order, picard_iters=picard_iters,
        solver=predictor_solver, ppe_solver=ppe_solver,
        alpha=alpha, beta_backflow=1.0,
        pressure_outflow_nodes=np.array([pin_node]),
        verbose=verbose,
    )
    st.set_initial(lambda coords: np.zeros((len(coords), dim)))

    plate_area = 4.0 * plate_half_y * plate_half_z
    ref_force = 0.5 * U_inf ** 2 * plate_area

    cd_hist = np.zeros(nsteps)
    cl_y_hist = np.zeros(nsteps)
    cl_z_hist = np.zeros(nsteps)

    for step in range(nsteps):
        st.step()
        F = st.surrogate_traction()
        cd_hist[step] = F[0] / ref_force
        cl_y_hist[step] = F[1] / ref_force
        cl_z_hist[step] = F[2] / ref_force
        if verbose:
            print(f"[p2r1c-3d-proj] step {step:3d}  Cd={cd_hist[step]:+.4f}  "
                  f"Cl_y={cl_y_hist[step]:+.4f}  Cl_z={cl_z_hist[step]:+.4f}",
                  flush=True)

    elapsed = time.time() - t0
    if verbose:
        print(f"[p2r1c-3d-proj] done in {elapsed:.1f}s  "
              f"Cd[-1]={cd_hist[-1]:+.4f}  (FINITE but not physically "
              f"faithful — see module caveat)", flush=True)

    result = dict(
        cd=cd_hist,
        cl_y=cl_y_hist,
        cl_z=cl_z_hist,
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
        vel_mag = np.linalg.norm(x_all[:, :dim], axis=1)
        result["mesh"] = mesh
        result["node_fields"] = {
            "velocity_magnitude": vel_mag,
            "pressure": x_all[:, dim],
        }
    if _return_stepper:
        result["stepper"] = st

    return result


def run_flow_past_3d_projection_one_sided(**kwargs):
    """Anti-vacuity: same setup but only Gamma~- assembled (drop Gamma~+).
    Used by the smoke gate to verify the two-sided coupling is load-bearing."""
    kwargs["_two_sided"] = False
    return run_flow_past_3d_projection(**kwargs)


# ---------------------------------------------------------------------------
# GH200 hero-run configuration (does NOT run in CI)
# ---------------------------------------------------------------------------
# Same config as the monolithic driver's GH200_CONFIG, but marched via the
# projection stepper with gpu_cg on the PPE — the scalable path the monolithic
# host-splu cannot reach at L9-near-plate. Physics caveat still applies.
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
    ppe_solver="gpu_cg",
)


if __name__ == "__main__":
    from diffsim.viz.results import save_flow_run

    level        = int(os.environ.get("LEVEL", "3"))
    base_level   = int(os.environ.get("BASE_LEVEL", str(level)))
    refine_level_str = os.environ.get("REFINE_LEVEL", "")
    refine_level = int(refine_level_str) if refine_level_str else None
    nsteps       = int(os.environ.get("NSTEPS", "3"))
    dt           = float(os.environ.get("DT", "0.01"))
    nu           = float(os.environ.get("NU", "0.1"))
    U_inf        = float(os.environ.get("U_INF", "1.0"))
    plate_xc     = float(os.environ.get("PLATE_XC", "0.375"))
    plate_yc     = float(os.environ.get("PLATE_YC", "0.5"))
    plate_zc     = float(os.environ.get("PLATE_ZC", "0.5"))
    plate_half_y = float(os.environ.get("PLATE_HALF_Y", "0.125"))
    plate_half_z = float(os.environ.get("PLATE_HALF_Z", "0.125"))
    ppe_solver   = os.environ.get("PPE_SOLVER", "gpu_cg")

    re_approx = int(round(U_inf / nu)) if nu > 0 else 0
    if refine_level:
        case_name = f"p2r1c_3d_proj_re{re_approx}_L{base_level}_r{refine_level}"
    else:
        case_name = f"p2r1c_3d_proj_re{re_approx}_L{base_level}"

    res = run_flow_past_3d_projection(
        level=base_level, nsteps=nsteps, dt=dt, nu=nu, U_inf=U_inf,
        plate_xc=plate_xc, plate_yc=plate_yc, plate_zc=plate_zc,
        plate_half_y=plate_half_y, plate_half_z=plate_half_z,
        verbose=True, refine_to=refine_level, ppe_solver=ppe_solver,
        _return_fields=True,
    )
    print(f"Cd={res['cd']}")
    print(f"Cl_y={res['cl_y']}")
    print(f"Cl_z={res['cl_z']}")
    print(f"PPE solver used: {res['ppe_solver']}")
    print("NOTE: 3-D projection Cd is FINITE but not physically faithful "
          "(projection-split defect — see module docstring caveat).")

    t_arr = np.arange(1, nsteps + 1) * dt
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
        print(f"[p2r1c-3d-proj] Results written: {list(written.values())}")
    else:
        print("[p2r1c-3d-proj] No results written (viz deps missing)")
