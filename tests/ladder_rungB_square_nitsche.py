"""Projection Validation Ladder — Rung B driver: body-fitted square, WEAK
NITSCHE no-slip (the SECOND scaffolded concept, in isolation).

Rung A (strong Dirichlet on the EXACT body-fitted square) PASSED with the
consistent-projection scheme: the split is faithful to the same-mesh monolithic
at Re=40 and Re=100. Rung B keeps the SAME body-fitted mesh (aligned `Box`,
`offset=0` => `dmax==0`) but swaps the obstacle BC treatment: instead of adding
the obstacle nodes to the STRONG Dirichlet set, the no-slip is imposed WEAKLY
via the SBM shifted-Nitsche vector-Dirichlet block. Because `dmax==0` (`d=0`,
`geo.corr==1`), the shifted-Nitsche form `S N_a = N_a` DEGENERATES TO STANDARD
NITSCHE EXACTLY — so rung B isolates the Nitsche coupling with no shift confound.

If B passes, projection+Nitsche works and faithfulness survives weak imposition;
if it breaks, the defect isolates to the Nitsche coupling (not the base
projection — rung A cleared it — and not the shift — d=0).

Two solves march on the SAME mesh (`build_square_channel_2d`, aligned `half`):

  * PROJECTION — the base `LerayProjectionStepper` in consistent-projection mode
    (the rung-A-validated scheme: collocated-divergence PPE + fine-scale +
    disjoint outflow BCs + rotational + rotational_pin_outflow + backflow
    beta=0.5), with the SBM shifted-Nitsche vector-Dirichlet face block injected
    into the momentum predictor via the `extra_block`/`sbm_nodes` hook (mirrors
    `LeraySBMStepper`), assembled ONCE from the fixture's `sf`/`geo` (+ a per-step
    backflow increment on the surrogate faces). Box strong Dirichlet (inflow +
    channel walls) unchanged from rung A; the OBSTACLE is NOT in the strong set —
    its surrogate-face nodes are the `sbm_nodes` that skip the strong overwrite.
    PPE surrogate BC is HOMOGENEOUS Neumann `grad(phi).n_hat=0` (the default
    `ppe_surrogate_flux=None`; Suresh octree-SBM Remark 3.9), `p'=0` Dirichlet at
    the TRUE outflow (`pressure_outflow_nodes`, unchanged).
  * MONOLITHIC — the same-mesh saddle NS solve (inline, mirrors
    p2r0_task10_sphere_derisk::monolithic_cd) with the IDENTICAL SBM Nitsche
    block (`sbm_vector_dirichlet`, no-slip body) added to the momentum system,
    box strong Dirichlet on inflow+walls, single outflow pressure pin. This is
    the primary, box-free bar (the weak-Nitsche same-mesh oracle).

Drag: `surrogate_traction(dm, sf, geo, x_full, nu, ndof)`; with `dmax==0`
(aligned) the surrogate faces == true box faces, so this is the EXACT body-fitted
traction (`geo.corr==1`). `Cd = F_x/qref`, `qref = 0.5 U_IN^2 D`, `D = 2*half`
(same qref as rung A). Lift `Cl = F_y/qref` for the Re=100 Strouhal signal.

Anti-vacuity: the Nitsche penalty must be LOAD-BEARING — zeroing it (`alpha=0`,
no penalty) BREAKS the match (the obstacle stops being felt). And rung B must
reproduce rung A (weak Nitsche with d=0 ~= strong Dirichlet, to within the
Nitsche consistency error) — the key faithfulness check.

Run (gpubox, CPU-splu — 2-D is small):
    PYTHONPATH=src:tests .venv/bin/python tests/ladder_rungB_square_nitsche.py
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from diffsim.steppers.leray import LerayProjectionStepper
from diffsim.sbm.vector import (sbm_vector_dirichlet, sbm_vector_penalty,
                                sbm_wall_pressure_neumann,
                                sbm_wall_pressure_kio_vms, surrogate_traction)
from diffsim.mesh.faces import face_tables
from diffsim.api.ns_bricks import (assemble_linear_ns, assemble_backflow_block,
                                   assemble_bvs_block, outflow_faces)
from diffsim.physics.poisson import gauss_points

from ladder_fixtures import build_square_channel_2d, U_IN

# rung A shares the qref / mean_speed / strouhal definitions — reuse verbatim so
# the B-vs-A comparison is on identical observables.
from ladder_rungA_square_strong import qref, mean_speed, strouhal

ALPHA = 10.0          # Nitsche penalty scale (matches the monolithic SBM default)


# --------------------------------------------------------------------------
# box-only strong BC: inflow + channel walls (obstacle is WEAK Nitsche now)
# --------------------------------------------------------------------------
def build_box_strong_bc(fx):
    """The strong Dirichlet set is inflow + channel walls ONLY (the box).
    Unlike rung A, the obstacle nodes are NOT added — they carry the WEAK
    Nitsche no-slip. Returns (strong_nodes, g_strong) in FREE-node space:
    inflow rows carry U_IN, walls carry 0."""
    strong_nodes = np.where(fx["strong_mask"])[0]
    g_strong = fx["u_inf"][strong_nodes]           # U_IN@inflow, 0@walls
    return strong_nodes, g_strong


# --------------------------------------------------------------------------
# SBM shifted-Nitsche block on the surrogate (obstacle) faces
# --------------------------------------------------------------------------
def build_sbm_block(fx, alpha=ALPHA, beta_backflow=0.0):
    """Assemble the geometry-only SBM shifted-Nitsche vector-Dirichlet block
    for a no-slip (g=0) body on the fixture's surrogate faces, CONSTRAINED to
    free-node-major space. With `dmax==0` this is standard Nitsche exactly.

    Returns (Af_c, bf_c, T_vec, sbm_nodes) where `sbm_nodes` are the free-node
    indices governed weakly by the block (surrogate-face nodes that must skip
    the strong box overwrite in the predictor)."""
    dm, sf, geo = fx["dm"], fx["sf"], fx["geo"]
    ndof, nu, dim = fx["ndof"], fx["nu"], fx["dim"]
    T = dm.constraints.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    g_body = lambda y: np.zeros((len(y), dim))
    Af, bf = sbm_vector_dirichlet(dm, sf, geo, g_body, nu, ndof, alpha=alpha,
                                  a_face=None, beta_backflow=beta_backflow)
    Af_c = (T_vec.T @ Af @ T_vec).tocsr()
    bf_c = np.asarray(T_vec.T @ bf)

    # SBM-governed free nodes: surrogate-face nodes (skip the box overwrite).
    # Read the surrogate faces' element order from p_elem (p-generic: p=1 for
    # uniform-P1, p=2 for the P2/P2-band fixtures) instead of hardcoding bin 1.
    mesh = dm.mesh
    pv_sf = int(np.unique(np.asarray(mesh.p_elem)[sf.elem])[0])
    conn = mesh.conn_of[pv_sf][np.searchsorted(mesh.bins[pv_sf], sf.elem)]
    glob = np.unique(conn.ravel())
    free_idx = dm.constraints.free_nodes
    free_of = np.full(dm.n_nodes, -1, dtype=np.int64)
    free_of[free_idx] = np.arange(len(free_idx))
    fn = free_of[glob]
    sbm_nodes = fn[fn >= 0]
    return Af_c, bf_c, T_vec, sbm_nodes


def build_correction_penalty(fx, alpha=ALPHA):
    """FN1 velocity-update re-pin: the PENALTY-ONLY viscous Nitsche block on the
    surrogate faces, reduced to the SCALAR constrained free-node space.

    Returns ``(N_scalar_c, g_pen)`` for the stepper's ``correction_penalty``
    hook: ``N_scalar_c`` is the component-diagonal penalty matrix (n_free x
    n_free, the same for every velocity component — the block is
    component-diagonal by construction) and ``g_pen`` is the RHS
    ``alpha(nu/h)<S v, g>`` (n_free x dim; zero for a no-slip g=0 body). The
    stepper adds ``N_scalar_c`` to the consistent mass and solves
    ``(M + N_scalar_c) u_new[:,c] = M-rhs[:,c] + g_pen[:,c]`` per component,
    RE-PINNING the surrogate wall trace AFTER the pressure correction (the weak
    analog of the strong-node overwrite)."""
    dm, sf, geo = fx["dm"], fx["sf"], fx["geo"]
    ndof, nu, dim = fx["ndof"], fx["nu"], fx["dim"]
    g_body = lambda y: np.zeros((len(y), dim))
    Apen, bpen = sbm_vector_penalty(dm, sf, geo, g_body, nu, ndof, alpha=alpha)
    T = dm.constraints.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    Apen_c = (T_vec.T @ Apen @ T_vec).tocsr()
    bpen_c = np.asarray(T_vec.T @ bpen)
    nfree = T.shape[1]
    bpen_v = bpen_c.reshape(nfree, ndof)
    # component-diagonal: extract the c=0 scalar sub-block (every component
    # carries the SAME scalar penalty), and the per-component RHS (velocity dofs).
    idx0 = np.arange(nfree) * ndof + 0
    N_scalar_c = Apen_c[np.ix_(idx0, idx0)].tocsr()
    g_pen = bpen_v[:, :dim].copy()
    return N_scalar_c, g_pen


def _bf_conn(fx):
    """Cache the surrogate-face p-value / face-table / connectivity for the
    per-step backflow advecting field (mirrors LeraySBMStepper._bf_setup)."""
    dm, sf = fx["dm"], fx["sf"]
    mesh = dm.mesh
    pv = int(np.unique(np.asarray(mesh.p_elem)[sf.elem])[0])
    ftab = face_tables(pv, fx["dim"])
    conn = mesh.conn_of[pv][np.searchsorted(mesh.bins[pv], sf.elem)]
    return pv, ftab, conn


def _a_face(fx, u_free, ftab, conn):
    """Advecting field at surrogate-face GPs from a free-node velocity,
    flattened (fi, q) -> [ne_f*nqf, dim] (matches geo.n GP layout)."""
    dm, sf = fx["dm"], fx["sf"]
    dim = fx["dim"]
    u_full = np.asarray(dm.constraints.T @ u_free)
    nqf = ftab.nqf
    af = np.empty((len(sf.elem) * nqf, dim))
    for fi in range(len(sf.elem)):
        f = int(sf.face[fi])
        un = u_full[conn[fi]]
        for q in range(nqf):
            af[fi * nqf + q] = ftab.N[f][q] @ un
    return af


# --------------------------------------------------------------------------
# PROJECTION march (base LerayProjectionStepper + SBM Nitsche extra-block)
# --------------------------------------------------------------------------
# FN1 fix: the grad-div (LSIC) penalty magnitude that stabilizes the growing
# interior-divergence mode the WEAK wall excites in the projection split. The
# seed-monolithic-one-step probe (ladder_rungB_seed_probe) localizes the rung-B
# divergence to this mode (NOT wall penetration — the weak monolithic itself has
# n.u~0.8 at the wall and is stable). grad-div makes the seeded monolithic a
# BOUNDED fixed point (|p*| ~O(10) vs mono ~7, mean|u| within a few %) where the
# bare split runs |p*| past 1e6 and blows up. The required gamma scales with Re:
# ~20 stabilizes Re=40 (levels 4-5); Re=100 needs ~50 (at gamma=20 it reaches
# only ~step 36). gamma=50 covers both Re=40 and Re=100 here. The correction
# re-pin (correction_repin) holds the weak wall trace at u_hat (the weak analog
# of the strong overwrite) and is ~neutral-to-slightly-helpful on top. RESIDUAL
# (honest): the drag Cd (wall pressure-traction) is NOT recovered — the split's
# steady wall pressure differs from the monolithic saddle, and MORE grad-div
# drives Cd more negative. The lagged wall pressure-traction (fix 2) helps Cd
# when SEEDED but destabilizes from rest; the PPE no-penetration coupling
# (fix ii) was not the stabilizer. The constant gamma is Re/mesh-dependent (a
# tau_C-scaled grad-div would be more robust). See task-FN1-report.md.
# FN4 (2026-07-23): the drag residual is FIXED by rot_pin_wall (the rotational
# -nu*q wall pin) — see march_projection's docstring and task-FN4-report.md.
FN1_GRADDIV_GAMMA = 50.0


def march_projection(fx, dt=0.02, nsteps=400, rate_tol=None, order=2,
                     log_every=0, consistent_projection=True, solver="splu",
                     alpha=ALPHA, beta_backflow=0.5, correction_repin=True,
                     graddiv_gamma=FN1_GRADDIV_GAMMA, wall_pneumann=False,
                     tauc_graddiv=False, graddiv_scale=1.0,
                     rot_pin_wall=False):
    """March the base projection stepper (consistent mode) with WEAK Nitsche
    no-slip on the obstacle, injected via the extra_block/sbm_nodes hook.
    Returns dict(cd, cl, mean_u, div, steps, cd_hist, cl_hist, ...).

    `alpha` is the Nitsche penalty scale (anti-vacuity knob — 0 removes the
    penalty). `beta_backflow` stabilizes inflow through the immersed body
    during the transient (mirrors LeraySBMStepper); it defaults to 0.5 under
    the consistent scheme (benign at steady state).

    FN4 (2026-07-23): `rot_pin_wall` is THE drag fix — pin the rotational
    -nu*q pressure correction to 0 on the immersed-wall (SBM) nodes (the wall
    analog of the F3b outflow pin). At a weak-Nitsche wall the predictor's
    div(u_hat) is O(1) penetration garbage, and -nu*q writes a refinement-
    growing O(0.5) wall-pressure bias every step (measured by the seeded-
    monolithic step-1 decomposition — predictor and PPE are exact at the
    seed). With the pin: Cd matches the same-mesh mono at L4 (3.0%) AND L5
    (8.7%), same settings, no tuning. `wall_pneumann` keeps the REJECTED
    KIO-style PPE wall sources for diagnostics (dict of
    sbm_wall_pressure_neumann kwargs, or "vms" for the fine-scale-weighted
    form — both mesh-dependent/destabilizing; see their docstrings).
    `tauc_graddiv` swaps the fixed-gamma grad-div for the tau_C (VMS) dynamic
    grad-div (`velocity_update="graddiv"` + `graddiv_dynamic`, scaled by
    `graddiv_scale`)."""
    dim = fx["dim"]
    dm = fx["dm"]
    sf, geo = fx["sf"], fx["geo"]
    ndof, nu = fx["ndof"], fx["nu"]
    strong_nodes, g_strong = build_box_strong_bc(fx)
    Af_c, bf_c, T_vec, sbm_nodes = build_sbm_block(fx, alpha=alpha,
                                                   beta_backflow=0.0)
    bf_pv, bf_ftab, bf_conn = _bf_conn(fx)
    # FN1 velocity-update re-pin (the fix): the penalty-only Nitsche block folded
    # into the correction mass solve so the surrogate wall trace is re-imposed
    # AFTER the pressure correction. None => plain update (the old FAILING path).
    correction_penalty = (build_correction_penalty(fx, alpha=alpha)
                          if correction_repin else None)

    def f_fn(x, t):
        return np.zeros((len(x), dim))

    def g_fn(coords_at_dir, t):
        return g_strong

    st = LerayProjectionStepper(
        dm, nu, dt, f_fn=f_fn, g_fn=g_fn, order=order, picard_iters=1,
        solver=solver, pressure_outflow_nodes=fx["outflow_nodes"],
        consistent_projection=consistent_projection,
        velocity_update=("graddiv" if tauc_graddiv else "consistent"),
        graddiv_scale=graddiv_scale, graddiv_dynamic=tauc_graddiv,
        graddiv_gamma=(graddiv_gamma if graddiv_gamma else None),
        rotational_pin_wall=rot_pin_wall)
    st.dir_nodes = strong_nodes                    # box strong; obstacle weak
    st.set_initial(lambda c: np.zeros((len(c), dim)))

    # FN4 consistent (KIO) wall-pressure PPE hook (the drag fix). None =>
    # homogeneous-Neumann wall (the FN1 drag-losing default).
    #   True / "vms" -> sbm_wall_pressure_kio_vms, the MESH-INDEPENDENT
    #     fine-scale-weighted form: total wall flux dphi/dn =
    #     sigma tau_m (dp_KIO/dn - dp*/dn), scale=1.0 at every level.
    #   dict -> sbm_wall_pressure_neumann kwargs (the raw oint g q source;
    #     term-level DIAGNOSTIC variants — mesh-dependent, kept to document
    #     the FN4 scaling bug).
    def wall_pn_vms(uhat):
        from diffsim.solvers.timestepping import bdf_coeffs, bdf_order_now
        o = bdf_order_now(st.t + st.dt, st.dt, st.order,
                          have_history=st.hist.have(2))
        bdf = bdf_coeffs(o, st.dt)
        u1 = st._uvec(st.hist.pre1)
        u2 = st._uvec(st.hist.pre2) if st.hist.have(2) else None
        return sbm_wall_pressure_kio_vms(dm, sf, geo, uhat, u1, u2, bdf,
                                         st.dt, nu, ndof,
                                         timestab=st.timestab)

    kio_kw = wall_pneumann if isinstance(wall_pneumann, dict) else {}

    def wall_pn_raw(uhat):
        return sbm_wall_pressure_neumann(dm, sf, geo, uhat, st.p_star, nu,
                                         ndof, **kio_kw)

    wall_pn_hook = None
    if wall_pneumann:
        wall_pn_hook = wall_pn_raw if isinstance(wall_pneumann, dict) \
            else wall_pn_vms

    def extra_block(u_free):
        """Cached geometry SBM block + per-step backflow increment on the
        surrogate faces (velocity-dependent; zero from rest)."""
        A, b = Af_c, bf_c
        if beta_backflow != 0.0 and u_free is not None and np.any(u_free):
            a_face = _a_face(fx, u_free, bf_ftab, bf_conn)
            if np.any(a_face):
                Af_bf, _ = sbm_vector_dirichlet(
                    dm, sf, geo, lambda y: np.zeros((len(y), dim)), nu, ndof,
                    alpha=alpha, a_face=a_face, beta_backflow=beta_backflow)
                Ab = (T_vec.T @ Af_bf @ T_vec).tocsr() - Af_c
                A = (Af_c + Ab).tocsr()
        return (A, b)

    def cur_u_free():
        if st.hist.pre1 is None:
            return np.zeros((st.n_free, dim))
        return st._uvec(st.hist.pre1)

    q = qref(fx)
    prev = None
    steps = 0
    cd = cl = np.nan
    cd_hist, cl_hist, mu_hist = [], [], []
    u = None
    blew_up = False
    for steps in range(1, nsteps + 1):
        # PPE surrogate BC is the DEFAULT homogeneous-Neumann grad(phi).n=0
        # (ppe_surrogate_flux=None) — Suresh Remark 3.9; obstacle is a no-slip
        # wall so no through-flux is imposed on the pressure correction.
        u, p = st.step(extra_block=extra_block(cur_u_free()),
                       sbm_nodes=sbm_nodes, ppe_surrogate_flux=None,
                       correction_penalty=correction_penalty,
                       wall_pressure_neumann=wall_pn_hook)
        if not np.isfinite(u).all() or not np.isfinite(p).all() \
                or np.abs(u).max() > 1e4:
            blew_up = True
            if log_every:
                print(f"[rungB proj] BLOW-UP at step{steps} "
                      f"(max|u|={np.abs(u).max():.3e})", flush=True)
            break
        xfree = np.zeros(st.n_free * ndof)
        xv = xfree.reshape(st.n_free, ndof)
        xv[:, :dim] = u
        xv[:, dim] = p
        F = surrogate_traction(dm, sf, geo, np.asarray(T_vec @ xfree), nu, ndof)
        cd = float(F[0] / q)
        cl = float(F[1] / q)
        mu = mean_speed(u)
        cd_hist.append(cd)
        cl_hist.append(cl)
        mu_hist.append(mu)
        if log_every and (steps <= 3 or steps % log_every == 0):
            print(f"[rungB proj] step{steps:4d}  Cd={cd:+.4f}  Cl={cl:+.4f}  "
                  f"mean|u|={mu:.4f}", flush=True)
        if rate_tol is not None and prev is not None:
            if np.abs(u - prev).max() / dt < rate_tol:
                break
        prev = u.copy()
    div = float(st.divergence_l2())
    # FN2 diagnostic hook: expose the final FULL node-major (u,p) field so the
    # drag helpers (surrogate_traction / sbm_consistent_flux) can be re-evaluated
    # off the same steady state. Cheap (one reshape); does not affect the march.
    x_full_final = None
    if u is not None and not blew_up:
        xfree = np.zeros(st.n_free * ndof)
        xv = xfree.reshape(st.n_free, ndof)
        xv[:, :dim] = u
        xv[:, dim] = p
        x_full_final = np.asarray(T_vec @ xfree)
    return dict(cd=cd, cl=cl, mean_u=mean_speed(u), div=div, steps=steps,
                cd_hist=cd_hist, cl_hist=cl_hist, mu_hist=mu_hist,
                pnorm=float(np.linalg.norm(st.p_star)), blew_up=blew_up,
                x_full=x_full_final)


# --------------------------------------------------------------------------
# MONOLITHIC march (inline saddle, same mesh, SBM Nitsche block)
# --------------------------------------------------------------------------
def march_monolithic(fx, dt=0.02, nsteps=400, rate_tol=2e-4, log_every=0,
                     backflow_beta=0.0, boundary_vorticity=False,
                     alpha=ALPHA, graddiv_gamma=FN1_GRADDIV_GAMMA):
    """March the same-mesh monolithic saddle NS (inline; mirrors
    p2r0_task10_sphere_derisk::monolithic_cd) with the SBM Nitsche block for a
    no-slip obstacle. Box strong Dirichlet on inflow+walls; single outflow
    pressure pin. Returns dict(cd, cl, mean_u, div, steps, ...).

    `alpha` is the SAME Nitsche penalty the projection uses, so the same-mesh
    oracle is exact. `backflow_beta`/`boundary_vorticity` mirror the rung-A
    outflow stabilization (kept matched at Re=100)."""
    dim = fx["dim"]
    dm, mesh, cons = fx["dm"], fx["mesh"], fx["cons"]
    sf, geo = fx["sf"], fx["geo"]
    ndof, nu = fx["ndof"], fx["nu"]
    coords = fx["coords"]
    strong_nodes, g_strong = build_box_strong_bc(fx)

    T = cons.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    nfree = T.shape[1]
    xq = gauss_points(mesh, dm.tables_by_p)

    # SBM Nitsche block (geometry-only, no-slip body), constrained. Assembled
    # ONCE (geometry-static) — same block the projection predictor carries.
    Af, bf = sbm_vector_dirichlet(dm, sf, geo,
                                  lambda y: np.zeros((len(y), dim)),
                                  nu, ndof, alpha=alpha)
    Af_c = (T_vec.T @ Af @ T_vec).tocsr()
    bf_c = np.asarray(T_vec.T @ bf)

    # Same grad-div (LSIC) block as the projection predictor, so the same-mesh
    # oracle carries the IDENTICAL consistent stabilization (it vanishes as
    # div u -> 0, so it barely shifts the steady state but keeps the comparison
    # term-for-term fair). Built via the stepper's cached assembler.
    gd_block = None
    if graddiv_gamma:
        _st = LerayProjectionStepper(
            dm, nu, dt, f_fn=lambda x, t: np.zeros((len(x), dim)),
            g_fn=lambda c, t: None, order=1, graddiv_gamma=graddiv_gamma)
        gd_block = _st._graddiv_gamma_block

    def gp_field(node_vec):
        full = np.asarray(T @ node_vec)
        aq, dq = {}, {}
        for pv in dm.bins:
            tb = dm.tables_by_p[pv]
            vals = full[mesh.conn_of[pv]]
            aq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
            h = mesh.tree.h()[mesh.bins[pv]]
            dq[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                      * (2.0 / h)[:, None]).reshape(-1)
        return aq, dq

    p_pin = int(np.argmax(coords[:, 0]))
    q = qref(fx)
    x = np.zeros(nfree * ndof)
    sigma = 1.0 / dt
    bf_faces = (outflow_faces(mesh)
                if (backflow_beta != 0.0 or boundary_vorticity) else None)
    prev_u = None
    steps = 0
    cd = cl = np.nan
    cd_hist, cl_hist, mu_hist = [], [], []
    u_new = None
    for step in range(nsteps):
        u_node = x.reshape(nfree, ndof)[:, :dim]
        aq, dq = gp_field(u_node)
        fq = {pv: aq[pv] / dt for pv in xq}
        A, b = assemble_linear_ns(dm, aq, dq, fq, nu, sigma=sigma)
        A = (A + Af_c)                             # SBM Nitsche block
        b = b + bf_c
        if gd_block is not None:                    # same grad-div as projection
            A = A + gd_block
        if backflow_beta != 0.0:
            A = A + assemble_backflow_block(dm, u_node, backflow_beta, ndof,
                                            faces=bf_faces)
        if boundary_vorticity:
            A = A + assemble_bvs_block(dm, u_node, nu, dt, ndof, faces=bf_faces)
        A = A.tolil()
        for k, i in enumerate(strong_nodes):
            for c in range(dim):
                r = i * ndof + c
                A.rows[r] = [int(r)]
                A.data[r] = [1.0]
                b[r] = g_strong[k, c]
        pin = p_pin * ndof + dim
        A.rows[pin] = [pin]
        A.data[pin] = [1.0]
        b[pin] = 0.0
        x = splu(A.tocsr().tocsc()).solve(b)
        u_new = x.reshape(nfree, ndof)[:, :dim]
        F = surrogate_traction(dm, sf, geo, np.asarray(T_vec @ x), nu, ndof)
        cd = float(F[0] / q)
        cl = float(F[1] / q)
        mu = mean_speed(u_new)
        cd_hist.append(cd)
        cl_hist.append(cl)
        mu_hist.append(mu)
        steps = step + 1
        if log_every and (steps <= 3 or steps % log_every == 0):
            print(f"[rungB mono] step{steps:4d}  Cd={cd:+.4f}  Cl={cl:+.4f}  "
                  f"mean|u|={mu:.4f}", flush=True)
        if rate_tol is not None and prev_u is not None and step > 5:
            if np.abs(u_new - prev_u).max() / dt < rate_tol:
                break
        prev_u = u_new.copy()
    _, dq = gp_field(u_new)
    tot, vol = 0.0, 0.0
    for pv in dm.bins:
        tb = dm.tables_by_p[pv]
        h = mesh.tree.h()[mesh.bins[pv]]
        jac = (h / 2.0) ** dim
        w = np.tile(tb.w, len(h)) * np.repeat(jac, tb.nqp)
        tot += (dq[pv] ** 2 * w).sum()
        vol += w.sum()
    div = float(np.sqrt(tot / vol))
    return dict(cd=cd, cl=cl, mean_u=mean_speed(u_new), div=div, steps=steps,
                cd_hist=cd_hist, cl_hist=cl_hist, mu_hist=mu_hist, p_pin=p_pin,
                x_full=np.asarray(T_vec @ x))


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------
TOL_CD_REL = 0.15      # projection Cd vs monolithic Cd (relative), as rung A
TOL_MU_REL = 0.20      # projection mean|u| vs monolithic mean|u| (relative)
WEAK_PLATEAU_FRAC = 0.5


def run_rungB(level=5, half=0.125, res=(40, 100), device="cpu",
              dt40=0.02, nsteps40=600, dt100=0.01, nsteps100=2000,
              log_every=25, alpha=ALPHA, solver="splu"):
    """Full rung-B driver. Re=40 steady (PRIMARY): projection (weak Nitsche) vs
    monolithic (weak Nitsche, same mesh) steady Cd + mean|u|. Re=100: mean Cd +
    Strouhal. Prints PASS/FAIL vs the same-mesh oracle.

    THE decisive read: does weak-Nitsche projection (consistent scheme) match
    the same-mesh weak-Nitsche monolithic Cd AND develop mean|u|, i.e. does the
    Nitsche layer preserve faithfulness on the working projection?"""
    results = {}
    all_pass = True
    D = 2.0 * half
    for Re in res:
        steady = (Re < 50)
        dt = dt40 if steady else dt100
        nsteps = nsteps40 if steady else nsteps100
        rate_tol_p = 5e-4 if steady else None
        rate_tol_m = 2e-4 if steady else None

        print(f"\n{'='*70}\n Rung B — body-fitted square WEAK NITSCHE  "
              f"Re={Re}  level={level}  half={half} (D={D})  alpha={alpha}"
              f"\n{'='*70}", flush=True)

        fx = build_square_channel_2d(level, Re, half=half, offset=0,
                                     device=device)
        dmax = fx["dmax"]
        # BODY-FITTED GUARD (anti-vacuity): weak Nitsche on an EXACT body-fitted
        # mesh => d=0 => the shifted-Nitsche form degenerates to standard
        # Nitsche exactly. Assert dmax==0 (no SBM shift active — rung C only).
        assert dmax == 0, (f"rung B requires an EXACT body-fitted carve "
                           f"(dmax==0); got dmax={dmax}.")
        print(f" body-fitted guard: dmax={dmax}  (OK, standard Nitsche)  "
              f"n_fluid_cells={fx['n_fluid_cells']}  "
              f"n_obstacle_nodes={int(fx['obstacle_node_mask'].sum())}",
              flush=True)

        mono_beta = 0.5
        mono_bvs = True
        # FN4: the verdict projection carries the rotational WALL PIN (the
        # drag fix). rot_pin_wall=False reproduces the FN1-era drag defect.
        pr = march_projection(fx, dt=dt, nsteps=nsteps, rate_tol=rate_tol_p,
                              log_every=log_every, alpha=alpha, solver=solver,
                              rot_pin_wall=True)
        mo = march_monolithic(fx, dt=dt, nsteps=nsteps, rate_tol=rate_tol_m,
                              log_every=log_every, backflow_beta=mono_beta,
                              boundary_vorticity=mono_bvs, alpha=alpha)

        cd_rel = (abs(pr["cd"] - mo["cd"]) / abs(mo["cd"])
                  if mo["cd"] != 0 else float("inf"))
        mu_rel = (abs(pr["mean_u"] - mo["mean_u"]) / abs(mo["mean_u"])
                  if mo["mean_u"] != 0 else float("inf"))
        mu_frac = pr["mean_u"] / mo["mean_u"] if mo["mean_u"] != 0 else 0.0
        weak_pin = mu_frac < WEAK_PLATEAU_FRAC
        blew = pr.get("blew_up", False)

        if steady:
            cd_ok = (not blew) and cd_rel < TOL_CD_REL
            mu_ok = (not blew) and (mu_rel < TOL_MU_REL) and (not weak_pin)
            verdict = cd_ok and mu_ok
            print(f"\n --- Re={Re} STEADY verdict ---")
            print(f"  projection : Cd={pr['cd']:+.4f}  mean|u|={pr['mean_u']:.4f}"
                  f"  ‖div‖={pr['div']:.3e}  ‖p‖={pr['pnorm']:.3e}  "
                  f"steps={pr['steps']}")
            print(f"  monolithic : Cd={mo['cd']:+.4f}  mean|u|={mo['mean_u']:.4f}"
                  f"  ‖div‖={mo['div']:.3e}  steps={mo['steps']}")
            print(f"  Cd rel-diff = {cd_rel:.3%} (tol {TOL_CD_REL:.0%})  "
                  f"-> {'OK' if cd_ok else 'FAIL'}")
            print(f"  mean|u| proj/mono = {mu_frac:.3f}  rel-diff={mu_rel:.3%} "
                  f"(tol {TOL_MU_REL:.0%})  weak_pin={weak_pin}  "
                  f"-> {'OK' if mu_ok else 'FAIL'}")
        else:
            def tail_mean(h):
                a = np.asarray(h)
                return float(a[int(0.4 * len(a)):].mean())
            pr_cdm = tail_mean(pr["cd_hist"])
            mo_cdm = tail_mean(mo["cd_hist"])
            cd_rel = (abs(pr_cdm - mo_cdm) / abs(mo_cdm)
                      if mo_cdm != 0 else float("inf"))
            st_p, _ = strouhal(pr["cl_hist"], dt, D)
            st_m, _ = strouhal(mo["cl_hist"], dt, D)
            cd_ok = (not blew) and cd_rel < TOL_CD_REL
            mu_ok = (not blew) and (mu_rel < TOL_MU_REL) and (not weak_pin)
            verdict = cd_ok and mu_ok
            pr["cd_mean_period"], mo["cd_mean_period"] = pr_cdm, mo_cdm
            pr["St"], mo["St"] = st_p, st_m
            print(f"\n --- Re={Re} SHEDDING verdict ---")
            print(f"  projection : mean Cd={pr_cdm:+.4f}  St={st_p:.4f}  "
                  f"mean|u|={pr['mean_u']:.4f}  ‖div‖={pr['div']:.3e}  "
                  f"steps={pr['steps']}")
            print(f"  monolithic : mean Cd={mo_cdm:+.4f}  St={st_m:.4f}  "
                  f"mean|u|={mo['mean_u']:.4f}  ‖div‖={mo['div']:.3e}  "
                  f"steps={mo['steps']}")
            print(f"  mean-Cd rel-diff = {cd_rel:.3%} (tol {TOL_CD_REL:.0%})  "
                  f"-> {'OK' if cd_ok else 'FAIL'}")
            print(f"  mean|u| proj/mono = {mu_frac:.3f}  weak_pin={weak_pin}  "
                  f"-> {'OK' if mu_ok else 'FAIL'}")

        all_pass = all_pass and verdict
        print(f" ===> Re={Re}  {'PASS' if verdict else 'FAIL'}", flush=True)
        results[Re] = dict(
            cd_proj=pr["cd"], cd_mono=mo["cd"], cd_rel=cd_rel,
            mean_u_proj=pr["mean_u"], mean_u_mono=mo["mean_u"],
            mu_frac=mu_frac, weak_pin=weak_pin,
            div_proj=pr["div"], div_mono=mo["div"],
            steps_proj=pr["steps"], steps_mono=mo["steps"],
            St_proj=pr.get("St"), St_mono=mo.get("St"),
            cd_mean_period_proj=pr.get("cd_mean_period"),
            cd_mean_period_mono=mo.get("cd_mean_period"),
            pnorm_proj=pr["pnorm"], blew_up=blew, dmax=dmax, verdict=verdict)

    print(f"\n{'#'*70}\n RUNG B VERDICT: "
          f"{'PASS — weak-Nitsche projection matches the same-mesh weak-Nitsche monolithic (the Nitsche layer preserves faithfulness on the working projection)' if all_pass else 'FAIL — weak-Nitsche projection does NOT match the same-mesh monolithic; the defect isolates to the Nitsche coupling.'}"
          f"\n{'#'*70}", flush=True)
    results["all_pass"] = all_pass
    return results


if __name__ == "__main__":
    import json
    import os
    import sys
    lvl = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    res = (40,) if (len(sys.argv) > 2 and sys.argv[2] == "re40") else (40, 100)
    solver = os.environ.get("PROJ_SOLVER", "splu")
    device = os.environ.get("DEVICE", "cpu" if solver == "splu" else "cuda:0")
    out = run_rungB(level=lvl, res=res, solver=solver, device=device)
    print("\n[json]", json.dumps(out, default=lambda o: None))
