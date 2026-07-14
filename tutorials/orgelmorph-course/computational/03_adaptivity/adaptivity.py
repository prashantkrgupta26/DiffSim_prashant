"""OrgElMorph course - Computational C3: spatial and temporal adaptivity.

Importable core for the adaptivity concept.  Resolution should follow the
physics.  This module measures the two adaptivities the simulator uses,
plus the one subtlety that makes the temporal one correct:

  1. SPATIAL (octree refinement).  build_adaptive puts small elements
     only where a field is sharp -- here, near an interface -- and leaves
     the bulk coarse.  We count the degrees of freedom of an adaptive
     mesh refined along a circle against a uniform mesh at the same
     finest resolution, and confirm the CH brick runs on the (hanging-
     node) adaptive mesh.

  2. TEMPORAL (LTE-controlled step ladder).  adaptive_march estimates the
     local truncation error by step-doubling (one dt-step vs two
     dt/2-steps), accepts when it is below tolerance, and rescales dt.
     dt SHRINKS through the violent spinodal onset and GROWS through slow
     coarsening -- the property that makes long phase-field horizons
     affordable.  We measure the growth and the step savings over a
     fixed-dt march.

  3. WHY VARIABLE-COEFFICIENT BDF2.  Growing dt safely is not free: the
     textbook BDF2 coefficients 3/2, [2, -1/2] are derived for a CONSTANT
     step.  Feed them a varying dt and the scheme is only formally
     consistent -- its order collapses toward 1.  The retrofit measured
     exactly that (constant-coefficient BDF2 on an alternating-dt
     sequence: order 0.90/0.95; audit doc 2026-07-13, G3).  The brick now
     uses the VARIABLE-coefficient form (coefficients from the actual
     (dt, dt_prev)), restoring order 2.  We re-measure the fixed form's
     order here (~2.0) and cite the degraded baseline.

Everything drives the production brick
src/diffsim/physics/cahn_hilliard.py and its adaptive_march; the gates
promoted are test_ch_adaptive_dt and test_ch_bdf2_variable_dt_order.
"""
import numpy as np

from diffsim.octree.build import build_uniform, build_adaptive
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.cahn_hilliard import CahnHilliardStepper, adaptive_march

# The constant-coefficient BDF2 baseline MEASURED at the G3 retrofit on an
# alternating-dt sequence (docs/dev/2026-07-13-m0m5-retrofit-audit.md, G3):
# order 0.90 / 0.95 with errors 4-20x the fixed-dt run.  Cited, not re-run
# (the brick no longer exposes the constant-coefficient path).
CONST_COEFF_BDF2_ORDER = (0.90, 0.95)


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


def adaptive_time_stepping(t_end=0.8, tol=5e-4, dt0=0.005, level=5,
                           device="cuda:0"):
    """Run the LTE-controlled ladder over a quench and measure how dt
    grows and how many steps it saves against a fixed-dt march that would
    have to use the ladder's SMALLEST step throughout to be equally safe.
    Returns the (t, dt) history and the step-count comparison."""
    st = _quench_stepper(dt0, level, device)
    ts, dts = adaptive_march(st, t_end=t_end, tol=tol)
    dts = np.asarray(dts)
    ts = np.asarray(ts)
    c = st.hist[0]
    n_adaptive = len(dts)
    # a fixed-step run safe for the whole horizon must use the smallest dt
    # the ladder ever needed (its onset step); count how many that takes.
    n_fixed = int(np.ceil(t_end / dts.min()))
    return dict(ts=ts, dts=dts, n_adaptive=n_adaptive, n_fixed=n_fixed,
                growth=float(dts[-1] / dts[0]),
                dt_min=float(dts.min()), dt_max=float(dts.max()),
                span=float(dts.max() / dts.min()),
                savings=n_fixed / n_adaptive,
                c_min=float(c.min()), c_max=float(c.max()),
                t_end=t_end, tol=tol)


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


def _fixed_march(dm, dt0, T=0.096):
    st = CahnHilliardStepper(dm, 1.0, 5e-4, dt0, order=2)
    st.set_initial(_G3_IC, mu_init="consistent")
    for _ in range(round(T / dt0)):
        st.step()
    assert abs(st.t - T) < 1e-12, st.t
    return st.x[0::2].copy()


def bdf2_variable_order(level=4, device="cuda:0"):
    """Measure the VARIABLE-coefficient BDF2 order on the alternating-dt
    sequence (a fine fixed-dt march is the reference).  Should be ~2 --
    the fix for the 0.90/0.95 constant-coefficient degradation cited in
    CONST_COEFF_BDF2_ORDER."""
    dm, _, _ = _dm(level, device)
    ref = _fixed_march(dm, 2e-4)
    dts = (8e-3, 4e-3, 2e-3)
    errs = [float(np.abs(_var_march(dm, d) - ref).max()) for d in dts]
    orders = [float(np.log2(errs[i] / errs[i + 1]))
              for i in range(len(errs) - 1)]
    return dict(dts=list(dts), errs=errs, orders=orders,
                order=float(np.mean(orders)),
                const_coeff_order=CONST_COEFF_BDF2_ORDER)
