"""OrgElMorph course - Computational C3: octree refinement + temporal adaptivity.

Importable core for the adaptivity concept.  Resolution should follow the
physics.  This module measures:

  1. OCTREE REFINEMENT + HANGING-NODE CONSTRAINTS (spatial).  build_adaptive
     puts small elements only where a field is sharp and leaves the bulk
     coarse.  IMPORTANT HONESTY: the refinement here is along a STATIC
     circle -- it is octree refinement to a *fixed* geometric criterion, NOT
     solution-adaptive AMR (which would estimate the error of the running
     solution, mark, refine/coarsen, and transfer state between meshes every
     few steps).  We count the dofs, confirm the CH brick runs on the
     hanging-node mesh, spell out what TRUE dynamic AMR additionally requires,
     and measure one piece of it that is cheap: the CONSERVATIVE-TRANSFER
     error (a naive restriction loses mass; a conservative one does not).
     Full dynamic AMR is a Phase-3 deliverable.

  2. TEMPORAL (LTE-controlled step ladder).  adaptive_march estimates the
     local truncation error by step-doubling (one dt-step vs two dt/2-steps),
     accepts when it is below tolerance, and rescales dt.  dt SHRINKS through
     the violent onset and GROWS through slow coarsening.  We report the REAL
     cost accounting -- accepted/rejected steps, full+half solves, total
     Newton iterations, wall time -- and a MATCHED-ACCURACY fixed-dt sweep
     (NOT the meaningless horizon/min-dt ratio).

  3. WHY VARIABLE-COEFFICIENT BDF2.  Growing dt safely is not free: the
     textbook BDF2 coefficients 3/2, [2, -1/2] are derived for a CONSTANT
     step.  Feed them a varying dt and the order collapses toward 1.  We
     MEASURE this locally (a tutorial-local constant-coefficient march, by
     forcing the coefficient ratio r=1 while the history spacing varies) and
     compare with the brick's variable-coefficient form (order ~2).

Everything drives the production brick src/diffsim/physics/cahn_hilliard.py
and its adaptive_march.
"""
import time

import numpy as np

from diffsim.octree.build import build_uniform, build_adaptive
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.cahn_hilliard import CahnHilliardStepper, adaptive_march


# --- 1. spatial octree refinement ------------------------------------

def _interface_refine(radius=0.30, band=1.5):
    """Refine any element whose center is within `band`*h of the circle
    of the given radius (centered in the unit box) -- i.e. resolve the
    interface, coarsen the bulk."""
    def refine_fn(centers, h):
        r = np.sqrt(((centers - 0.5) ** 2).sum(1))
        return np.abs(r - radius) < h[:, 0] * band
    return refine_fn


def octree_refinement(max_level=6, device="cuda:0"):
    """Compare an interface-refined octree mesh to a uniform mesh at the
    same finest level, and confirm the CH brick steps on the adaptive
    (hanging-node) mesh.  Returns node/element counts, the savings, and
    the adaptive element centers+levels for plotting."""
    atree = build_adaptive(_interface_refine(), max_level=max_level, dim=2)
    amesh = build_mesh(atree, p=1)
    acons = build_constraints(amesh)
    utree = build_uniform(max_level, dim=2)
    umesh = build_mesh(utree, p=1)

    # correctness: a CH step must run on the adaptive mesh (T != identity;
    # hanging nodes are constrained by build_constraints).
    adm = DeviceMesh.from_mesh(amesh, acons, basis_tables(1, dim=2), device)
    st = CahnHilliardStepper(adm, 1.0, 5e-4, 0.02, order=1)
    rng = np.random.default_rng(3)
    st.set_initial(lambda x: 0.05 * rng.standard_normal(len(x)),
                   mu_init="consistent")
    c, _ = st.step()

    return dict(
        max_level=max_level,
        adaptive_nodes=len(amesh.node_coords),
        uniform_nodes=len(umesh.node_coords),
        adaptive_elems=len(atree.keys),
        uniform_elems=len(utree.keys),
        node_savings=len(umesh.node_coords) / len(amesh.node_coords),
        elem_savings=len(utree.keys) / len(atree.keys),
        free_dofs=int(acons.T.shape[1]),
        step_ok=bool(np.isfinite(c).all()),
        centers=atree.centers(), levels=np.asarray(atree.levels),
        hs=atree.h())


# --- 1b. what TRUE AMR needs: the conservative-transfer error --------
# Static octree refinement is the EASY half of AMR.  Dynamic AMR must, every
# few steps: (a) estimate the running solution's error, (b) MARK elements,
# (c) refine/coarsen, (d) enforce 2:1 balance, (e) rebuild the mesh + T, (f)
# TRANSFER the state (and the BDF history!) to the new mesh, (g) continue.
# Step (f) is where conservation is won or lost: the transfer operator must
# preserve Int c.  We demonstrate the failure mode of a NAIVE transfer and
# the fix, on a structured grid (pure numpy -- no marching needed).

def transfer_error_sweep(radii=(0.5, 0.7, 1.0, 1.5, 2.5), n_drops=12,
                         seed=3):
    """How the transfer error scales with feature size: injection loses
    mass sharply once the feature drops below the coarse cell; averaging is
    exact at every size.  Returns the radius sweep for the figure."""
    return [transfer_error(n_drops=n_drops, radius=r, seed=seed)
            for r in radii]


def transfer_error(nf=64, n_drops=12, radius=0.5, seed=3):
    """Restrict a fine field with sub-coarse-cell structure to a 2x-coarser
    grid two ways and measure the mass change of each.  The field is a set
    of small droplets (radius ~1 fine cell) at random positions -- the kind
    of sub-cell feature dynamic AMR must not lose when it coarsens a region:
      - INJECTION (sample every other node): NOT conservative -- droplets
        sitting between the sampled nodes are missed, so Int c changes.
      - CELL AVERAGING (mean of each 2x2 block): conservative -- every block
        integrates its droplet content, so Int c is preserved exactly.
    Returns the relative mass error of each transfer."""
    rng = np.random.default_rng(seed)
    ii, jj = np.mgrid[0:nf, 0:nf]
    c = np.zeros((nf, nf))
    for _ in range(n_drops):
        ci, cj = rng.uniform(radius, nf - radius, size=2)
        c += np.exp(-((ii - ci) ** 2 + (jj - cj) ** 2) / (2 * radius ** 2))
    m_fine = float(c.mean())                       # Int c / area (uniform)
    inj = c[::2, ::2]                              # nodal injection
    avg = c.reshape(nf // 2, 2, nf // 2, 2).mean((1, 3))   # 2x2 average
    return dict(nf=nf, n_drops=n_drops, mass_fine=m_fine,
                inj_err=abs(float(inj.mean()) - m_fine) / abs(m_fine),
                avg_err=abs(float(avg.mean()) - m_fine) / abs(m_fine),
                c_fine=c, c_inj=inj, c_avg=avg)


# --- 2. temporal LTE-controlled step ladder --------------------------

def _quench_stepper(dt0, level=5, device="cuda:0"):
    dm, _, _ = _dm(level, device)
    st = CahnHilliardStepper(dm, 1.0, 5e-4, dt0, order=2)
    rng = np.random.default_rng(3)
    st.set_initial(lambda x: 0.05 * rng.standard_normal(len(x)),
                   mu_init="consistent")
    return st


def _dm(level, device):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    return DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                                device), mesh, cons


def instrumented_march(st, t_end, tol=5e-4, dt_min=1e-5, dt_max=0.5,
                       safety=0.85):
    """A tutorial-local copy of adaptive_march that RECORDS the real work:
    accepted / rejected steps, full+half Newton solves, total Newton
    iterations, and wall time.  Same step-doubling controller as the brick's
    adaptive_march -- the point is the accounting, not a new algorithm."""
    p_ord = st.order
    ts, dts = [], []
    acc = rej = full_solves = half_solves = newton_iters = 0
    t0 = time.perf_counter()
    while st.t < t_end - 1e-12:
        state = (st.x.copy(), [h.copy() for h in st.hist], st.t, st.dt_prev)
        st.step(); full_solves += 1; newton_iters += st.last_newton["iters"]
        x1 = st.x.copy()
        st.x, st.hist, st.t, st.dt_prev = (state[0].copy(),
                                           [h.copy() for h in state[1]],
                                           state[2], state[3])
        dt_full = st.dt
        st.dt = dt_full / 2
        st.step(); half_solves += 1; newton_iters += st.last_newton["iters"]
        st.step(); half_solves += 1; newton_iters += st.last_newton["iters"]
        x2 = st.x.copy()
        num = float(np.linalg.norm(x1[0::2] - x2[0::2]))
        den = max(float(np.linalg.norm(x2[0::2])), 1e-30)
        lte = (num / den) / (2 ** p_ord - 1)
        if lte < tol or dt_full <= dt_min * 2:
            acc += 1
            dt_new = min(dt_max, max(dt_min, dt_full * min(
                2.0, max(0.5, safety * (tol / max(lte, 1e-30))
                         ** (1.0 / (p_ord + 1))))))
            st.dt = dt_new
            ts.append(st.t); dts.append(dt_full)
        else:
            rej += 1
            st.x, st.hist, st.t, st.dt_prev = (state[0].copy(),
                                               [h.copy() for h in state[1]],
                                               state[2], state[3])
            st.dt = max(dt_min, dt_full / 2)
    wall = time.perf_counter() - t0
    return dict(ts=np.asarray(ts), dts=np.asarray(dts), accepted=acc,
                rejected=rej, full_solves=full_solves,
                half_solves=half_solves, newton_iters=newton_iters,
                wall=wall)


def _fixed_cost(dm, dt, t_end):
    """March fixed-dt to t_end recording steps / Newton iters / wall and the
    final field."""
    st = CahnHilliardStepper(dm, 1.0, 5e-4, dt, order=2)
    rng = np.random.default_rng(3)
    st.set_initial(lambda x: 0.05 * rng.standard_normal(len(x)),
                   mu_init="consistent")
    n = int(round(t_end / dt))
    newton = 0
    t0 = time.perf_counter()
    for _ in range(n):
        st.step(); newton += st.last_newton["iters"]
    return dict(dt=dt, steps=n, newton_iters=newton,
                wall=time.perf_counter() - t0, field=st.x[0::2].copy())


def adaptive_time_stepping(t_end=0.8, tol=5e-4, dt0=0.005, level=5,
                           device="cuda:0"):
    """Run the LTE-controlled ladder over a quench and report its dt growth
    (for the figure).  Cost accounting lives in adaptive_cost (below)."""
    st = _quench_stepper(dt0, level, device)
    ts, dts = adaptive_march(st, t_end=t_end, tol=tol)
    dts = np.asarray(dts); ts = np.asarray(ts)
    c = st.hist[0]
    return dict(ts=ts, dts=dts, n_adaptive=len(dts),
                growth=float(dts[-1] / dts[0]),
                dt_min=float(dts.min()), dt_max=float(dts.max()),
                span=float(dts.max() / dts.min()),
                c_min=float(c.min()), c_max=float(c.max()),
                t_end=t_end, tol=tol)


def adaptive_cost(t_end=0.6, tol=5e-4, dt0=0.005, level=5, device="cuda:0",
                  ref_dt=2e-4, fixed_dts=(4e-3, 2e-3, 1e-3, 5e-4)):
    """REAL cost accounting for the adaptive ladder, and a MATCHED-ACCURACY
    fixed-dt sweep (not the horizon/min-dt fiction).  Marches the adaptive
    controller (instrumented) and a set of fixed-dt runs to the SAME t_end,
    scores each against a fine reference, and reports the coarsest fixed dt
    that matches the adaptive accuracy along with its real work."""
    dm, _, _ = _dm(level, device)
    ref = _fixed_cost(dm, ref_dt, t_end)["field"]
    den = max(float(np.linalg.norm(ref)), 1e-30)
    # adaptive
    st = _quench_stepper(dt0, level, device)
    acc = instrumented_march(st, t_end=t_end, tol=tol)
    err_ad = float(np.linalg.norm(st.x[0::2] - ref) / den)
    # fixed sweep
    fixed = []
    for dt in fixed_dts:
        fc = _fixed_cost(dm, dt, t_end)
        fc["err"] = float(np.linalg.norm(fc["field"] - ref) / den)
        fixed.append(fc)
    # matched: coarsest fixed dt no less accurate than adaptive
    matched = min((f for f in fixed if f["err"] <= err_ad),
                  key=lambda f: f["steps"], default=fixed[-1])
    return dict(
        t_end=t_end, tol=tol, ref_dt=ref_dt,
        adaptive=dict(accepted=acc["accepted"], rejected=acc["rejected"],
                      full_solves=acc["full_solves"],
                      half_solves=acc["half_solves"],
                      newton_iters=acc["newton_iters"], wall=acc["wall"],
                      err=err_ad),
        fixed=[dict(dt=f["dt"], steps=f["steps"],
                    newton_iters=f["newton_iters"], wall=f["wall"],
                    err=f["err"]) for f in fixed],
        matched=dict(dt=matched["dt"], steps=matched["steps"],
                     newton_iters=matched["newton_iters"],
                     wall=matched["wall"], err=matched["err"]),
        newton_savings=matched["newton_iters"] / max(acc["newton_iters"], 1),
        wall_savings=matched["wall"] / max(acc["wall"], 1e-9))


# --- 3. variable-coefficient BDF2 order ------------------------------

_G3_IC = lambda x: (0.8 + 0.05 * np.cos(np.pi * x[:, 0])
                    * np.cos(np.pi * x[:, 1]))


def _var_march(dm, dt0, T=0.096):
    """March with the ALTERNATING (dt0, dt0/2) sequence: every step sees
    a fresh dt/dt_prev ratio (r = 2 then r = 1/2), exercising the
    variable-coefficient BDF2 on every step."""
    st = CahnHilliardStepper(dm, 1.0, 5e-4, dt0, order=2)
    st.set_initial(_G3_IC, mu_init="consistent")
    for _ in range(round(T / (1.5 * dt0))):
        st.dt = dt0
        st.step()
        st.dt = dt0 / 2
        st.step()
    assert abs(st.t - T) < 1e-12, st.t
    return st.x[0::2].copy()


def _const_march(dm, dt0, T=0.096):
    """Tutorial-LOCAL constant-coefficient BDF2 on the alternating sequence.
    We keep the real (alternating) history spacing but FORCE the coefficient
    ratio r = dt/dt_prev to 1 every step (by overwriting dt_prev = dt before
    each step).  That makes the stepper use the textbook constant-step
    coefficients 3/2, [2, -1/2] on a NON-uniform history -- exactly the bug
    the variable-coefficient form fixes -- so we can MEASURE the degraded
    order here instead of citing an inaccessible dev note."""
    st = CahnHilliardStepper(dm, 1.0, 5e-4, dt0, order=2)
    st.set_initial(_G3_IC, mu_init="consistent")
    for _ in range(round(T / (1.5 * dt0))):
        st.dt = dt0
        if st.dt_prev is not None:
            st.dt_prev = st.dt          # force r = 1 -> constant coeffs
        st.step()
        st.dt = dt0 / 2
        st.dt_prev = st.dt              # force r = 1 -> constant coeffs
        st.step()
    assert abs(st.t - T) < 1e-12, st.t
    return st.x[0::2].copy()


def _fixed_march(dm, dt0, T=0.096):
    st = CahnHilliardStepper(dm, 1.0, 5e-4, dt0, order=2)
    st.set_initial(_G3_IC, mu_init="consistent")
    for _ in range(round(T / dt0)):
        st.step()
    assert abs(st.t - T) < 1e-12, st.t
    return st.x[0::2].copy()


def _order(dm, march, dts, ref):
    errs = [float(np.abs(march(dm, d) - ref).max()) for d in dts]
    orders = [float(np.log2(errs[i] / errs[i + 1]))
              for i in range(len(errs) - 1)]
    return errs, orders


def bdf2_variable_order(level=4, device="cuda:0"):
    """Measure BOTH the VARIABLE-coefficient (brick) and the tutorial-local
    CONSTANT-coefficient BDF2 order on the same alternating-dt sequence
    against a fine fixed-dt reference.  Variable ~2 (order preserved);
    constant ~0.9 (order collapses toward 1) -- both MEASURED here."""
    dm, _, _ = _dm(level, device)
    ref = _fixed_march(dm, 2e-4)
    dts = (8e-3, 4e-3, 2e-3)
    v_errs, v_orders = _order(dm, _var_march, dts, ref)
    c_errs, c_orders = _order(dm, _const_march, dts, ref)
    return dict(dts=list(dts), errs=v_errs, orders=v_orders,
                order=float(np.mean(v_orders)),
                const_errs=c_errs, const_orders=c_orders,
                const_order=float(np.mean(c_orders)),
                const_coeff_order=tuple(round(o, 2) for o in c_orders))
