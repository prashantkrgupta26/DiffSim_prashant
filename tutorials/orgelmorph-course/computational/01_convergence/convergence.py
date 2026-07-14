"""OrgElMorph course - Computational C1: convergence, measured two ways.

Importable core for the flagship computational concept.  A discretization
is only trustworthy if it converges at the rate the theory predicts, and
the *only* honest way to know is to measure it.  This module promotes the
two convergence gates that ship with the Cahn-Hilliard brick into a
taught study:

  1. SPATIAL (method of manufactured solutions).  Plug a KNOWN field into
     the equations, add whatever residual it leaves as a source term, and
     solve.  The discrete solution must approach the known one; for a
     finite-element space of polynomial degree p the L2 error falls like
     h^{p+1}.  We measure that exponent for p=1 and p=2.

  2. TEMPORAL (self-convergence).  Freeze the mesh, march the same smooth
     initial state with a shrinking time step, and compare against a very
     fine reference march.  Backward Euler (BDF1) is first order in dt;
     BDF2 is second order.  We measure both exponents.

Nothing here is tutorial-only: `CahnHilliardStepper` is the same
production brick the research code and the gates in
tests/test_cahn_hilliard.py use (test_ch_mms_orders,
test_ch_bdf2_variable_dt_order).  The manufactured source terms below are
the ones the spatial gate uses verbatim.

The mixed (c, mu) Cahn-Hilliard system solved each step is
    c_t = M div(grad mu) + f_c,        (conserved transport)
    mu  = f'(c) - kappa lap c + f_m,   (chemical potential)
with f(c) = 1/4 (c^2 - 1)^2 the polynomial double well.  For the MMS we
manufacture c* and mu* INDEPENDENTLY (the cleanest way to avoid taking a
Laplacian of a cubic by hand) and let the two source terms f_c, f_m carry
whatever residual the chosen pair leaves.
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.cahn_hilliard import CahnHilliardStepper
from diffsim.physics.poisson import l2_error

# Model constants shared by both studies (the spatial-gate values).
M, KAP = 1.0, 0.02


def build_dm(level, p, device="cuda:0"):
    """Uniform 2-D box, 2^level cells per side, degree-p elements."""
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), device)
    return dm, mesh, cons


# --- the manufactured solution ---------------------------------------
# c*(x, t) = cos(pi x) cos(pi y) e^{-t}   (a smooth, mean-zero field)
# mu*(x,t) = sin(pi x) sin(pi y) e^{-t}   (manufactured independently)
# Both have Laplacian -2 pi^2 (field), which is why the sources are
# closed-form.  The two source functions below are exactly what make the
# pair (c*, mu*) an EXACT solution of the discrete residual, so any
# leftover error is pure discretization error - the thing we measure.

def c_star(x, t):
    return np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1]) * np.exp(-t)


def mu_star(x, t):
    return np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) * np.exp(-t)


def f_c(x, t):
    # R_c = c_t - M lap(mu) - f_c = 0.  c_t = -c*, lap(mu*) = -2pi^2 mu*.
    c = c_star(x, t)
    lap_mu = -2 * np.pi ** 2 * mu_star(x, t)
    return -c - M * lap_mu


def f_m(x, t):
    # R_m = mu - f'(c) + kap lap(c) - f_m = 0.  f'(c)=c^3-c.
    c = c_star(x, t)
    lap_c = -2 * np.pi ** 2 * c
    return mu_star(x, t) - (c ** 3 - c) + KAP * lap_c


def spatial_mms(p, levels, dt=0.002, nsteps=6, device="cuda:0"):
    """Measure the SPATIAL order of accuracy by MMS at degree p.

    For each mesh level we march a few steps of the manufactured problem
    (Dirichlet data on the boundary from c*, mu*) and measure the L2
    error against c* at the final time.  The error should fall like
    h^{p+1}; the exponent between successive levels is the observed
    order.  Returns h, errs, and the pairwise orders."""
    hs, errs = [], []
    for lv in levels:
        dm, mesh, cons = build_dm(lv, p, device)
        # boundary nodes carry the exact (c*, mu*) as strong data - the
        # manufactured field is not no-flux, so we pin it.
        coords = mesh.node_coords[cons.free_nodes]
        bdry = np.zeros(len(coords), bool)
        for cc in range(2):
            bdry |= (np.abs(coords[:, cc]) < 1e-12) | \
                    (np.abs(coords[:, cc] - 1.0) < 1e-12)
        st = CahnHilliardStepper(dm, M, KAP, dt, order=2,
                                 fc_fn=f_c, fm_fn=f_m,
                                 dirichlet=np.where(bdry)[0],
                                 gc_fn=c_star, gm_fn=mu_star)
        st.set_initial(lambda x: c_star(x, 0.0))
        for _ in range(nsteps):
            c, _ = st.step()
        err = l2_error(dm, np.asarray(cons.T @ c),
                       lambda x: c_star(x, st.t))
        hs.append(1.0 / (1 << lv))
        errs.append(err)
    orders = [float(np.log2(errs[i] / errs[i + 1]))
              for i in range(len(errs) - 1)]
    return dict(p=p, levels=list(levels), h=hs, errs=errs, orders=orders,
                order=float(np.mean(orders)))


# --- temporal self-convergence ---------------------------------------
# A smooth cosine perturbation of a uniform blend (the retrofit's G3
# initial condition).  No manufactured source: we compare each march to a
# very fine-dt reference march on the SAME mesh, so only the time error
# remains.  BDF1 (backward Euler) is O(dt); BDF2 is O(dt^2).

def _temporal_IC(x):
    return 0.8 + 0.05 * np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])


def _march_to(dm, order, dt, T):
    """March a fixed-dt spinodal relaxation to time T; return c."""
    st = CahnHilliardStepper(dm, M, 5e-4, dt, order=order)
    st.set_initial(_temporal_IC, mu_init="consistent")
    nsteps = round(T / dt)
    for _ in range(nsteps):
        st.step()
    assert abs(st.t - T) < 1e-9, (st.t, T)
    return st.x[0::2].copy()


def temporal_convergence(order, dts, ref_dt=2e-4, T=0.096, level=4,
                         device="cuda:0"):
    """Measure the TEMPORAL order for a BDF scheme (order=1 BDF1,
    order=2 BDF2).  A fine-dt BDF2 march is the reference; the error of
    each coarse march is |c(dt) - c_ref|_inf.  Constant dt throughout, so
    BDF2 runs its exact 1.5/[2,-0.5] coefficients (r=1).  Returns dts,
    errs, and the pairwise orders."""
    dm, _, _ = build_dm(level, 1, device)
    ref = _march_to(dm, 2, ref_dt, T)              # high-accuracy target
    errs = [float(np.abs(_march_to(dm, order, d, T) - ref).max())
            for d in dts]
    orders = [float(np.log2(errs[i] / errs[i + 1]))
              for i in range(len(errs) - 1)]
    return dict(order=order, dts=list(dts), errs=errs, orders=orders,
                order_est=float(np.mean(orders)))
