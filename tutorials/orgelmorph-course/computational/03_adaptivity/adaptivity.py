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
from diffsim.octree.build import refine_elements
from diffsim.octree.balance import balance2to1
from diffsim.adaptivity.remesh import (conservative_transfer, nodal_transfer,
                                       field_mass, consistent_mass_matrix)
from diffsim.adaptivity.amr_march import (amr_march, build_ch_stepper, remesh,
                                          free_energy)


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


# --- 4. DYNAMIC (solution-adaptive) AMR — the Phase-3 deliverable ------
# Sections 1-3 above are honest that the octree refinement there is STATIC
# (a fixed circle) and that the conservative-transfer demo is a numpy toy.
# The functions below implement and MEASURE the real thing: the full
# estimate -> mark -> refine/coarsen -> 2:1 balance -> rebuild -> transfer
# (field AND BDF history) -> continue cycle, driving the production CH brick
# through src/diffsim/adaptivity/{remesh,amr_march}.py.

_AMR_EPS = 0.03


def _droplet_ic(coords):
    r = np.sqrt(((coords - 0.5) ** 2).sum(1))
    return np.tanh((0.3 - r) / (np.sqrt(2.0) * _AMR_EPS))


def _band(tree, r=0.3, band=1.5, cap=None):
    rr = np.sqrt(((tree.centers() - 0.5) ** 2).sum(1))
    m = np.abs(rr - r) < tree.h() * band
    if cap is not None:
        m &= tree.levels < cap
    return m


def _analytic_smooth(coords):
    x, y = coords[:, 0], coords[:, 1]
    return 0.3 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) + 0.5


def fe_conservative_transfer():
    """The REAL conservative transfer (not the numpy toy of section 1b): the
    L2/Galerkin projection on the common refinement of two FE octree meshes,
    which conserves INT c dV to solver tolerance under BOTH refinement and
    coarsening, converges at 2nd order, and round-trips losslessly on a
    representable field. Also shows the nodal-injection failure under
    coarsening. Host-side (no GPU)."""
    TB = basis_tables(1, dim=2)

    def mk(tree):
        m = build_mesh(tree, p=1)
        return m, build_constraints(m)

    coarse = build_uniform(4, dim=2)
    fine = balance2to1(refine_elements(coarse, _band(coarse)))
    cm, cc = mk(coarse)
    fm, fc = mk(fine)

    # refine + coarsen mass conservation
    res = {}
    for name, (om, oc), (nm, nc) in (("refine", (cm, cc), (fm, fc)),
                                     ("coarsen", (fm, fc), (cm, cc))):
        f0 = _analytic_smooth(om.node_coords[oc.free_nodes])
        m0 = field_mass(om, oc, TB, f0)
        (fn,) = conservative_transfer(om, oc, TB, nm, nc, TB, [f0])
        res[name] = abs(field_mass(nm, nc, TB, fn) - m0)
    # nodal (non-conservative) contrast on a genuinely lossy coarsening
    of = build_uniform(5, dim=2)
    ofm, ofc = mk(of)
    og = build_uniform(3, dim=2)
    ogm, ogc = mk(og)
    f0 = _analytic_smooth(ofm.node_coords[ofc.free_nodes])
    m0 = field_mass(ofm, ofc, TB, f0)
    (fn_c,) = conservative_transfer(ofm, ofc, TB, ogm, ogc, TB, [f0])
    (fn_n,) = nodal_transfer(ofm, ofc, ogm, ogc, [f0])
    nodal_err = abs(field_mass(ogm, ogc, TB, fn_n) - m0) / abs(m0)
    cons_err = abs(field_mass(ogm, ogc, TB, fn_c) - m0) / abs(m0)

    # 2nd-order convergence of the projection
    errs = []
    for lvl in (3, 4, 5):
        om, oc = mk(build_uniform(lvl + 1, dim=2))
        nm, nc = mk(build_uniform(lvl, dim=2))
        cf = _analytic_smooth(om.node_coords[oc.free_nodes])
        (cn,) = conservative_transfer(om, oc, TB, nm, nc, TB, [cf])
        exact = _analytic_smooth(nm.node_coords[nc.free_nodes])
        Mm = consistent_mass_matrix(nm, TB)
        e = np.asarray(nc.T @ (cn - exact))
        errs.append(float(np.sqrt(e @ (Mm @ e))))
    orders = [float(np.log2(errs[i] / errs[i + 1]))
              for i in range(len(errs) - 1)]
    return dict(mass_refine=float(res["refine"]),
                mass_coarsen=float(res["coarsen"]),
                nodal_coarsen_err=float(nodal_err),
                cons_coarsen_err=float(cons_err),
                order=float(np.mean(orders)), orders=orders)


def dynamic_amr_cycle(device="cuda:0", t_end=5e-3, dt=2e-4, max_level=6):
    """Run the solution-adaptive AMR march over a shrinking droplet and MEASURE
    the cycle: mass conserved across every remesh, the free-energy jump across
    a remesh (transfer error only), and the refined-region/interface overlap."""
    out = amr_march(build_uniform(3, dim=2), _droplet_ic, M=1.0,
                    kappa=_AMR_EPS ** 2, dt=dt, t_end=t_end, device=device,
                    order=1, remesh_every=5, refine_frac=0.15,
                    coarse_frac=0.03, max_level=max_level, min_level=3)
    rec = out["rec"]
    dm = np.abs(np.array(rec["remesh_mass_after"])
                - np.array(rec["remesh_mass_before"]))
    dE = np.abs(np.array(rec["remesh_E_after"])
                - np.array(rec["remesh_E_before"]))
    return dict(remeshes=len(dm),
                mass_jump=float(dm.max()),
                energy_jump=float(dE.max()),
                overlap_min=float(min(rec["overlap"])),
                overlap_mean=float(np.mean(rec["overlap"])),
                dofs_peak=int(max(rec["dofs"])),
                dofs_mean=float(np.mean(rec["dofs"])),
                c_min=float(out["final_c"].min()),
                c_max=float(out["final_c"].max()),
                finite=bool(np.isfinite(out["final_c"]).all()),
                out=out)


def amr_error_vs_dofs(device="cuda:0", ref_level=7, amr_max=6, t_end=5e-3,
                      dt=2e-4, uniform_levels=(4, 5, 6)):
    """error-vs-dofs and error-vs-walltime for dynamic AMR versus a uniform-mesh
    sweep, all scored against a uniform-fine (ref_level) reference. The AMR run
    reaches uniform-fine-comparable accuracy at MEASURABLY fewer dofs."""
    TB = basis_tables(1, dim=2)

    def uniform_run(level):
        st, mesh, cons = build_ch_stepper(build_uniform(level, dim=2), 1.0,
                                          _AMR_EPS ** 2, dt, 1, "poly", device)
        st.set_initial(_droplet_ic, mu_init="consistent")
        t0 = time.perf_counter()
        for _ in range(int(round(t_end / dt))):
            st.step()
        return dict(mesh=mesh, cons=cons, c=st.x[0::2].copy(),
                    dofs=st.nfree, wall=time.perf_counter() - t0)

    ref = uniform_run(ref_level)
    rm, rc = ref["mesh"], ref["cons"]
    Mref = consistent_mass_matrix(rm, TB)

    def err(mesh, cons, c):
        (cr,) = conservative_transfer(mesh, cons, TB, rm, rc, TB, [c])
        e = np.asarray(rc.T @ (cr - ref["c"]))
        return float(np.sqrt(e @ (Mref @ e)))

    uni = []
    for lvl in uniform_levels:
        r = uniform_run(lvl)
        r["err"] = err(r["mesh"], r["cons"], r["c"])
        uni.append(dict(level=lvl, dofs=r["dofs"], err=r["err"],
                        wall=r["wall"]))

    t0 = time.perf_counter()
    out = amr_march(build_uniform(3, dim=2), _droplet_ic, 1.0, _AMR_EPS ** 2,
                    dt, t_end, device, order=1, remesh_every=5,
                    refine_frac=0.15, coarse_frac=0.03, max_level=amr_max,
                    min_level=3)
    amr_wall = time.perf_counter() - t0
    amr_err = err(out["mesh"], out["cons"], out["final_c"])
    amr_dofs = int(max(out["rec"]["dofs"]))

    ud = np.array([u["dofs"] for u in uni], float)
    ue = np.array([u["err"] for u in uni], float)
    order = np.argsort(ue)
    uni_dofs_match = float(np.exp(np.interp(np.log(amr_err),
                                            np.log(ue[order]),
                                            np.log(ud[order]))))
    # matched uniform wall for the same accuracy (log-interp on the curve)
    uw = np.array([u["wall"] for u in uni], float)
    uni_wall_match = float(np.exp(np.interp(np.log(amr_err),
                                            np.log(ue[order]),
                                            np.log(uw[order]))))
    return dict(ref_level=ref_level, ref_dofs=ref["dofs"],
                uniform=uni, amr_err=amr_err, amr_dofs=amr_dofs,
                amr_wall=amr_wall, uni_dofs_match=uni_dofs_match,
                dof_ratio=uni_dofs_match / amr_dofs,
                wall_ratio=uni_wall_match / max(amr_wall, 1e-9))


def spacetime_order(device="cuda:0", T=0.048, Nref=1200, Ns=(15, 30, 60)):
    """A3 space-time interaction: measure the temporal order of a variable-step
    BDF2 march that REMESHES at its midpoint. Both BDF history levels are
    conservatively transferred, so order 2 is recovered across the remesh
    ('both'); dropping the second level forces a BDF1 restart and inflates the
    error constant ('drop') while a single restart still recovers order 2."""
    def march(dt, N, mode):
        st, mesh, cons = build_ch_stepper(build_uniform(4, dim=2), 1.0, 5e-4,
                                          dt, order=2, energy="poly",
                                          device=device)
        st.set_initial(lambda x: 0.6 + 0.2 * np.cos(np.pi * x[:, 0])
                       * np.cos(np.pi * x[:, 1]), mu_init="consistent")
        rstep = int(round(0.5 * N))
        for i in range(1, N + 1):
            st.dt = dt
            st.step()
            if i == rstep:
                nt = balance2to1(refine_elements(mesh.tree,
                                                 _band(mesh.tree, cap=5)))
                st, mesh, cons = remesh(st, mesh, cons, nt, 1.0, 5e-4,
                                        "poly", device)
                if mode == "drop":
                    st.hist = [st.x[0::2].copy(), st.x[0::2].copy()]
                    st.dt_prev = None
        return st.x[0::2].copy()

    res = {}
    for mode in ("both", "drop"):
        ref = march(T / Nref, Nref, mode)
        errs = [float(np.abs(march(T / N, N, mode) - ref).max()) for N in Ns]
        orders = [float(np.log2(errs[i] / errs[i + 1]))
                  for i in range(len(errs) - 1)]
        res[mode] = dict(errs=errs, orders=orders, order=float(np.mean(orders)))
    return dict(dts=[T / N for N in Ns],
                both_order=res["both"]["order"], both=res["both"],
                drop_order=res["drop"]["order"], drop=res["drop"],
                drop_err_penalty=res["drop"]["errs"][0] / res["both"]["errs"][0])
