"""OrgElMorph course - Computational C1: convergence, measured honestly.

A discretization is only trustworthy if it converges at the rate the theory
predicts, and the *only* honest way to know is to measure it -- WITH the two
error sources separated and the algebraic (solver) error controlled.  This
module drives the production Cahn-Hilliard brick
(``src/diffsim/physics/cahn_hilliard.py`` -- the same brick as Physics
Chapter P1 and the shipping gates ``test_ch_mms_orders`` /
``test_ch_bdf2_variable_dt_order``) into a taught verification study:

  1. SPATIAL (method of manufactured solutions, STEADY).  Plug a known
     *time-independent* field into the equations, add whatever residual it
     leaves as a source, and solve to the discrete steady state.  Because the
     exact field is steady, the time term contributes NOTHING at steady state
     -- the measured error is pure SPATIAL discretization error, independent
     of dt.  This is the clean separation the previous (transient) MMS lacked.
     For degree-p elements the L2 error falls like h^{p+1} and the H1
     seminorm like h^{p}.  We measure both, for BOTH fields c and mu.

  2. TEMPORAL (self-convergence).  Freeze an over-resolved mesh, march the
     same smooth initial state with a shrinking time step, and compare each
     march against a very fine reference march on the identical mesh -- so the
     spatial error cancels and only the time error remains.  BDF1 is O(dt);
     BDF2 is O(dt^2).  We VERIFY the reference (halve its dt, show the order
     is stable, Richardson-estimate its residual error) before trusting it.

  3. ALGEBRAIC control.  A convergence plot contaminated by solver error is
     worthless.  We report the Newton and linear-solver tolerances, and show
     that tightening them does NOT move the measured discretization error --
     the discretization error is what we claim to measure.

  4. DELIBERATE FAILURES.  Four ways to get a wrong-but-plausible order
     (loose Newton, wrong BC, wrong source, under-resolved reference) for the
     student to diagnose.

The mixed (c, mu) Cahn-Hilliard system solved each step is
    c_t = M div(grad mu) + f_c,        (conserved transport)
    mu  = f'(c) - kappa lap c + f_m,   (chemical potential)
with f(c) = 1/4 (c^2 - 1)^2 the polynomial double well.
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.cahn_hilliard import CahnHilliardStepper
from diffsim.diagnostics import convergence as cvg

# Model constants shared by both studies (the spatial-gate values).
M, KAP = 1.0, 0.02


def build_dm(level, p, device="cuda:0"):
    """Uniform 2-D box, 2^level cells per side, degree-p elements."""
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), device)
    return dm, mesh, cons


# --- the STEADY manufactured solution --------------------------------
# c*(x) = cos(pi x) cos(pi y)   (smooth, time-INDEPENDENT)
# mu*(x)= sin(pi x) sin(pi y)   (manufactured independently)
# Both have Laplacian -2 pi^2 (field).  Because the fields are steady,
# c_t* = 0, so the source f_c carries only the spatial residual and the
# measured error is dt-independent (the clean-separation fix).

# The field wavenumber (k*pi) is a knob: k=1 is smooth and well resolved on
# coarse meshes; a large k makes a field that a coarse mesh CANNOT resolve, so
# its convergence study sits in the pre-asymptotic regime (the
# under-resolved-feature failure demo below).

def c_star_k(x, k=1.0):
    return np.cos(k * np.pi * x[:, 0]) * np.cos(k * np.pi * x[:, 1])


def mu_star_k(x, k=1.0):
    return np.sin(k * np.pi * x[:, 0]) * np.sin(k * np.pi * x[:, 1])


def grad_c_star_k(x, k=1.0):
    p = k * np.pi
    return np.stack([-p * np.sin(p * x[:, 0]) * np.cos(p * x[:, 1]),
                     -p * np.cos(p * x[:, 0]) * np.sin(p * x[:, 1])], axis=1)


def grad_mu_star_k(x, k=1.0):
    p = k * np.pi
    return np.stack([p * np.cos(p * x[:, 0]) * np.sin(p * x[:, 1]),
                     p * np.sin(p * x[:, 0]) * np.cos(p * x[:, 1])], axis=1)


def f_c_k(x, t, k=1.0):
    # R_c = c_t - M lap(mu) - f_c = 0.  STEADY: c_t = 0.  lap(mu*)=-2(k pi)^2 mu*
    return -M * (-2 * (k * np.pi) ** 2 * mu_star_k(x, k))


def f_m_k(x, t, k=1.0):
    # R_m = mu - f'(c) + kap lap(c) - f_m = 0.  f'(c)=c^3-c.
    c = c_star_k(x, k)
    return mu_star_k(x, k) - (c ** 3 - c) + KAP * (-2 * (k * np.pi) ** 2 * c)


# The default smooth (k=1) manufactured solution used by the main studies.
def c_star(x):
    return c_star_k(x, 1.0)


def mu_star(x):
    return mu_star_k(x, 1.0)


def grad_c_star(x):
    return grad_c_star_k(x, 1.0)


def grad_mu_star(x):
    return grad_mu_star_k(x, 1.0)


def f_c(x, t):
    return f_c_k(x, t, 1.0)


def f_m(x, t):
    return f_m_k(x, t, 1.0)


# --- quadrature error integrator (numpy, same quadrature as the solver)

def _quad_errors(st, c_free, mu_free, k=1.0):
    """L2 and H1-seminorm errors of BOTH fields, plus discrete mass error,
    integrated with the SOLVER's Gauss quadrature (not a nodal proxy).

    Returns dict: l2_c, h1_c, l2_mu, h1_mu, mass_err.  Units: field units
    (L2) and field/length (H1 seminorm); mass_err is the field integral.
    """
    cv, cg = st._gp_scalar(c_free)
    mv, mg = st._gp_scalar(mu_free)
    l2c = h1c = l2m = h1m = mass_h = mass_ex = 0.0
    for pv, b in st.dm.bins.items():
        tb = st.dm.tables_by_p[pv]
        h = st.mesh.tree.h()[st.mesh.bins[pv]]
        jac = (h / 2.0) ** st.dm.dim
        w = (tb.w[None, :] * jac[:, None]).reshape(-1)          # [ne*nqp]
        xq = st.xq[pv]
        ec, emu = c_star_k(xq, k), mu_star_k(xq, k)
        egc, egm = grad_c_star_k(xq, k), grad_mu_star_k(xq, k)
        l2c += float(np.sum(w * (cv[pv] - ec) ** 2))
        l2m += float(np.sum(w * (mv[pv] - emu) ** 2))
        h1c += float(np.sum(w * np.sum((cg[pv] - egc) ** 2, axis=1)))
        h1m += float(np.sum(w * np.sum((mg[pv] - egm) ** 2, axis=1)))
        mass_h += float(np.sum(w * cv[pv]))
        mass_ex += float(np.sum(w * ec))
    return dict(l2_c=np.sqrt(l2c), h1_c=np.sqrt(h1c),
                l2_mu=np.sqrt(l2m), h1_mu=np.sqrt(h1m),
                mass_err=abs(mass_h - mass_ex))


def _boundary_free(mesh, cons):
    """Indices (into free-node ordering) of the box boundary nodes."""
    coords = mesh.node_coords[cons.free_nodes]
    bdry = np.zeros(len(coords), bool)
    for cc in range(2):
        bdry |= (np.abs(coords[:, cc]) < 1e-12) | \
                (np.abs(coords[:, cc] - 1.0) < 1e-12)
    return np.where(bdry)[0]


def _solve_steady(p, level, device, dt=1e3, nsteps=3,
                  newton_tol=1e-11, newton_max=25, pin_bc=True,
                  fm=None, fc=None, k=1.0):
    """March the STEADY manufactured problem to its discrete steady state.

    Returns (stepper, cons, c_free, mu_free).  We take a few very large-dt
    steps from the exact-field initial guess: the time term (sigma ~ 1/dt)
    then vanishes and each Newton solve lands on the discrete STEADY FE
    system, staying in the correct branch of the non-monotone cubic f'(c)
    (small-dt marching can drift to a spurious steady branch -- a real CH
    pitfall).  The measured error is pure spatial discretization error,
    independent of dt (Newton residual reported)."""
    fc = fc or (lambda x, t: f_c_k(x, t, k))
    fm = fm or (lambda x, t: f_m_k(x, t, k))
    dm, mesh, cons = build_dm(level, p, device)
    dirich = _boundary_free(mesh, cons) if pin_bc else None
    st = CahnHilliardStepper(
        dm, M, KAP, dt, order=2, fc_fn=fc, fm_fn=fm,
        dirichlet=dirich,
        gc_fn=(lambda x, t: c_star_k(x, k)) if pin_bc else None,
        gm_fn=(lambda x, t: mu_star_k(x, k)) if pin_bc else None,
        newton_tol=newton_tol, newton_max=newton_max)
    st.set_initial(lambda x: c_star_k(x, k), mu_init="consistent")
    st.x[1::2] = mu_star_k(st.free_coords, k)   # seed mu at exact (steady)
    for _ in range(nsteps):
        st.step()
    return st, cons, st.x[0::2].copy(), st.x[1::2].copy()


def spatial_mms(p, levels, dt=1e3, nsteps=3, device="cuda:0",
                newton_tol=1e-11, pin_bc=True, fm=None, k=1.0):
    """Measure the SPATIAL order of accuracy by STEADY MMS at degree p.

    Reports L2 and H1 errors of BOTH c and mu, the discrete mass error, and
    the worst final-step Newton increment (the algebraic residual -- proof the
    plot is not solver-contaminated).  L2 error ~ h^{p+1}, H1 ~ h^{p}."""
    hs = [1.0 / (1 << lv) for lv in levels]
    rows, dxinf = [], []
    for lv in levels:
        st, cons, cf, mf = _solve_steady(p, lv, device, dt, nsteps,
                                         newton_tol=newton_tol,
                                         pin_bc=pin_bc, fm=fm, k=k)
        rows.append(_quad_errors(st, cf, mf, k=k))
        dxinf.append(st.last_newton["dx_inf"])

    def col(key):
        return [r[key] for r in rows]
    l2c, h1c = col("l2_c"), col("h1_c")
    return dict(
        p=p, levels=list(levels), h=hs,
        l2_c=l2c, h1_c=h1c, l2_mu=col("l2_mu"), h1_mu=col("h1_mu"),
        mass_err=col("mass_err"), newton_dx_inf=dxinf, newton_tol=newton_tol,
        errs=l2c,                                  # back-compat alias (L2 c)
        orders=list(cvg.pairwise_orders(hs, l2c)),
        order=float(cvg.observed_order(hs, l2c)),
        h1_orders=list(cvg.pairwise_orders(hs, h1c)),
        h1_order=float(cvg.observed_order(hs, h1c)),
        l2_mu_order=float(cvg.observed_order(hs, col("l2_mu"))),
    )


def algebraic_control(p=1, level=5, device="cuda:0",
                      tols=(1e-4, 1e-8, 1e-12)):
    """Show the DISCRETIZATION error is invariant to the ALGEBRAIC (Newton)
    tolerance -- the load-bearing check that the convergence plot measures
    discretization, not solver, error.  The linear solve is a direct sparse
    LU (splu): exact to round-off, so the only algebraic knob is Newton."""
    out = []
    for tol in tols:
        st, cons, cf, mf = _solve_steady(p, level, device,
                                         newton_tol=tol, newton_max=40)
        e = _quad_errors(st, cf, mf)
        out.append(dict(newton_tol=tol, l2_c=e["l2_c"],
                        dx_inf=st.last_newton["dx_inf"],
                        iters=st.last_newton["iters"]))
    ref = out[-1]["l2_c"]                       # tightest tol = truth
    spread = max(abs(o["l2_c"] - ref) for o in out[:-1])
    return dict(p=p, level=level, rows=out, l2_ref=ref,
                spread=spread, spread_frac=spread / ref)


# --- temporal self-convergence ---------------------------------------
# A smooth cosine perturbation of a uniform blend.  No manufactured source:
# we compare each march to a very fine-dt reference march on the SAME
# (over-resolved) mesh, so the spatial error cancels and only the time error
# remains.  BDF1 is O(dt); BDF2 is O(dt^2).

TEMP_KAP = 5e-4


def _temporal_IC(x):
    return 0.8 + 0.05 * np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])


def _march_to(dm, order, dt, T, newton_tol=1e-10):
    """March a fixed-dt spinodal relaxation to time T; return c."""
    st = CahnHilliardStepper(dm, M, TEMP_KAP, dt, order=order,
                             newton_tol=newton_tol)
    st.set_initial(_temporal_IC, mu_init="consistent")
    nsteps = round(T / dt)
    for _ in range(nsteps):
        st.step()
    assert abs(st.t - T) < 1e-9, (st.t, T)
    return st.x[0::2].copy()


def temporal_convergence(order, dts, ref_dt=1e-4, T=0.096, level=4,
                         device="cuda:0"):
    """Measure the TEMPORAL order for a BDF scheme (order=1 BDF1, order=2
    BDF2).  A fine-dt BDF2 march is the reference; each coarse march's error
    is the relative L2 difference from it.  Constant dt throughout, so BDF2
    runs its exact 1.5/[2,-0.5] coefficients (r=1)."""
    dm, _, _ = build_dm(level, 1, device)
    ref = _march_to(dm, 2, ref_dt, T)              # high-accuracy target
    den = float(np.linalg.norm(ref))
    errs = [float(np.linalg.norm(_march_to(dm, order, d, T) - ref) / den)
            for d in dts]
    return dict(order=order, dts=list(dts), errs=errs, ref_dt=ref_dt,
                orders=list(cvg.pairwise_orders(dts, errs)),
                order_est=float(cvg.observed_order(dts, errs)))


def verify_reference(dts=(4e-3, 2e-3, 1e-3), level=4, device="cuda:0",
                     ref_dts=(2e-4, 1e-4)):
    """VERIFY the temporal reference before trusting it (spec C1).

    Re-measure the BDF2 order against two references whose dt differ by 2x.
    If the order is stable across them, the reference is fine enough to be
    'truth'.  Also Richardson-estimate the reference's own residual error
    from the two fine marches."""
    dm, _, _ = build_dm(level, 1, device)
    T = 0.096
    marches = {d: _march_to(dm, 2, d, T) for d in set(ref_dts)}
    orders = {}
    for rd in ref_dts:
        ref = marches[rd]
        den = float(np.linalg.norm(ref))
        errs = [float(np.linalg.norm(_march_to(dm, 2, d, T) - ref) / den)
                for d in dts]
        orders[rd] = float(cvg.observed_order(dts, errs))
    # Richardson error of the FINER reference from the two ref marches
    # (relative L2 norm of the difference, scaled by 1/(r^p - 1)):
    r = ref_dts[0] / ref_dts[1]                # coarse/fine ratio (>1)
    diff = float(np.linalg.norm(marches[ref_dts[1]] - marches[ref_dts[0]]))
    den = float(np.linalg.norm(marches[ref_dts[1]]))
    rich = (diff / den) / (r ** 2.0 - 1.0)
    return dict(ref_dts=list(ref_dts), orders=orders,
                order_stable=abs(orders[ref_dts[0]] - orders[ref_dts[1]]),
                richardson_ref_err=float(rich))


# --- deliberate-failure demos ----------------------------------------
# Each returns the WRONG-but-plausible order the student must diagnose.

def fail_underresolved_feature(p=1, levels=(2, 3, 4), k=6.0,
                               device="cuda:0"):
    """Under-resolved feature: a sharp manufactured field (k=6, i.e. six
    half-waves across the box) that the COARSE meshes cannot represent.  On
    levels 2-4 the study is PRE-ASYMPTOTIC -- the error is huge and the
    observed 'order' is meaningless (well below p+1) until the mesh finally
    resolves the feature.  The lesson: a two- or three-point study on an
    under-resolved feature reports a fictitious order."""
    r = spatial_mms(p, levels, device=device, k=k)
    return dict(name="under-resolved feature (k=6, coarse mesh)",
                order=r["order"], errs=r["l2_c"],
                diagnosis="pre-asymptotic: feature not resolved on the mesh")


def fail_wrong_bc(p=1, levels=(4, 5, 6), device="cuda:0"):
    """Wrong BC: the manufactured field is NOT no-flux, but we impose the
    solver's natural (no-flux) BC instead of pinning c*, mu*.  The boundary
    error dominates and the order collapses."""
    r = spatial_mms(p, levels, device=device, pin_bc=False)
    return dict(name="wrong BC (natural, not pinned)", order=r["order"],
                errs=r["l2_c"],
                diagnosis="boundary mismatch, not interior discretization")


def fail_wrong_source(p=1, levels=(4, 5, 6), device="cuda:0"):
    """Wrong source: drop the kappa*lap(c) term from f_m so the source no
    longer matches the manufactured field.  The scheme converges to the
    WRONG steady state -- a nonzero but sub-theoretical order."""
    def fm_bad(x, t):
        c = c_star(x)
        return mu_star(x) - (c ** 3 - c)       # missing kappa*lap(c)
    r = spatial_mms(p, levels, device=device, fm=fm_bad)
    return dict(name="wrong source (kappa term dropped)", order=r["order"],
                errs=r["l2_c"],
                diagnosis="source does not match manufactured field")


def fail_temporal_contamination(order=2, dts=(8e-3, 4e-3, 2e-3),
                                ref_dt=6e-3, level=4, device="cuda:0"):
    """Under-resolved reference: a BDF2 reference at dt=6e-3 is barely finer
    than the coarsest sampled dt, so the 'error' saturates and the measured
    order is meaningless (the self-convergence method's one pitfall)."""
    r = temporal_convergence(order, dts, ref_dt=ref_dt, level=level,
                             device=device)
    return dict(name="under-resolved reference (ref dt 6e-3)",
                order=r["order_est"], errs=r["errs"],
                diagnosis="reference not 'exact enough' vs sampled dt")


DELIBERATE_FAILURES = [fail_wrong_bc, fail_wrong_source,
                       fail_underresolved_feature,
                       fail_temporal_contamination]
