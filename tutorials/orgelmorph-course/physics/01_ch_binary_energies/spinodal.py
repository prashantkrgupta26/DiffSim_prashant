"""OrgElMorph course - Physics P1: binary Cahn-Hilliard, two free energies.

Importable core for the tutorial: run a binary spinodal-decomposition
march with either the polynomial double-well or the Flory-Huggins
(logarithmic) bulk free energy, and measure the Ginzburg-Landau energy
budget (total = bulk + interfacial) at every step.

This is the object the student reads FIRST, then run.py drives it and
gen_figures.py renders the figures the LaTeX shows.  Nothing here is
tutorial-only: `CahnHilliardStepper` is the same production brick the
research code uses (src/diffsim/physics/cahn_hilliard.py) - the reason
the interface below is small is that the physics lives in the brick,
not in a toy re-implementation.

Free energy (Ginzburg-Landau):  F[c] = INT [ f(c) + (kap/2)|grad c|^2 ] dV
  bulk        f(c)                       - the homogeneous mixing energy
  interfacial (kap/2)|grad c|^2          - the gradient penalty on
                                           composition variation (sets
                                           the interface width ~ sqrt(kap))

  poly:  f = (1/4)(c^2 - 1)^2,  c in [-1, 1], symmetric double well.
  fh:    f = A[c ln c + (1-c) ln(1-c)] + B c(1-c),  c in (0, 1);
         A = entropic scale (~1/N), B = enthalpic (~chi).  Spinodal
         where the entropic curvature A(1/c + 1/(1-c)) drops below 2B.

Conserved (Model-B) dynamics: c_t = div(M grad mu), mu = dF/dc, so F is
a Lyapunov functional - it can only DECREASE (the numbers prove it).
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.cahn_hilliard import CahnHilliardStepper


def build_mesh_dm(level=7, p=1, device="cuda:0"):
    """Uniform 2-D box, 2^level cells per side (level 7 = 128x128).
    Natural (no-flux) boundaries: grad c . n = grad mu . n = 0, so
    total mass is conserved and no composition leaves the box."""
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), device)
    return dm, mesh, cons


# --- the Ginzburg-Landau energy budget, computed by quadrature --------
# We evaluate c and grad c at the Gauss points (the stepper exposes the
# same helper it uses internally) and integrate the two energy densities
# separately, so the student sees bulk and interfacial energy on the
# same axes.

def _bulk_density(c, energy, A, B):
    if energy == "fh":
        eps = 1e-4
        rl = lambda x: np.where(x < eps, np.log(eps) + (x - eps) / eps,
                                np.log(np.maximum(x, eps)))
        return A * (c * rl(c) + (1 - c) * rl(1 - c)) + B * c * (1 - c)
    return 0.25 * (c ** 2 - 1.0) ** 2


def energy_budget(st, dm, mesh, cf, energy, A, B):
    """Return (F_total, F_bulk, F_interface) for the free-field vector
    cf, by Gauss-point quadrature of the two energy densities."""
    v, g = st._gp_scalar(cf)               # c and grad c at Gauss pts
    F_bulk = F_int = 0.0
    for pv, b in dm.bins.items():
        h = mesh.tree.h()[mesh.bins[pv]]
        ne = len(mesh.conn_of[pv])
        wq = np.tile(dm.tables_by_p[pv].w, ne) \
            * np.repeat((h / 2.0) ** dm.dim, b["nqp"])
        F_bulk += float((wq * _bulk_density(v[pv], energy, A, B)).sum())
        F_int += float((wq * 0.5 * st.kappa * (g[pv] ** 2).sum(1)).sum())
    return F_bulk + F_int, F_bulk, F_int


def total_mass(st, dm, mesh, cf):
    v, _ = st._gp_scalar(cf)
    m = 0.0
    for pv, b in dm.bins.items():
        h = mesh.tree.h()[mesh.bins[pv]]
        ne = len(mesh.conn_of[pv])
        wq = np.tile(dm.tables_by_p[pv].w, ne) \
            * np.repeat((h / 2.0) ** dm.dim, b["nqp"])
        m += float((wq * v[pv]).sum())
    return m


def run_spinodal(energy="poly", level=6, steps=300, dt=0.02,
                 M=1.0, kappa=5e-4, fh_A=1.0, fh_B=2.5,
                 c_avg=None, amp=0.05, seed=3, device="cuda:0",
                 snap_steps=None):
    """March a binary spinodal decomposition and record the energy
    budget each step.  Returns a dict with the time series + snapshots.

    c_avg default: 0.0 for poly (symmetric wells), 0.5 for fh (the
    symmetric point of the log entropy).  amp is the initial random
    perturbation that seeds the instability (small: the linear
    spinodal analysis governs the early growth)."""
    if c_avg is None:
        c_avg = 0.0 if energy == "poly" else 0.5
    if snap_steps is None:
        snap_steps = [0, steps // 10, steps // 3, steps]
    dm, mesh, cons = build_mesh_dm(level, p=1, device=device)
    st = CahnHilliardStepper(dm, M, kappa, dt, order=1, energy=energy,
                             fh_A=fh_A, fh_B=fh_B)
    rng = np.random.default_rng(seed)
    st.set_initial(lambda x: c_avg + amp * rng.standard_normal(len(x)),
                   mu_init="consistent")

    def budget(cf):
        return energy_budget(st, dm, mesh, cf, energy, fh_A, fh_B)

    full = lambda cf: np.asarray(cons.T @ cf)
    side = int(round(np.sqrt(len(mesh.node_coords))))
    m0 = total_mass(st, dm, mesh, st.hist[0])
    Ft, Fb, Fi = budget(st.hist[0])
    rec = dict(t=[0.0], F_total=[Ft], F_bulk=[Fb], F_interface=[Fi],
               mass=[m0], snaps={}, side=side, energy=energy,
               kappa=kappa, c_avg=c_avg, params=dict(
                   M=M, dt=dt, fh_A=fh_A, fh_B=fh_B, level=level))
    if 0 in snap_steps:
        rec["snaps"][0] = full(st.x[0::2]).reshape(side, side).copy()
    for n in range(1, steps + 1):
        c, mu = st.step()
        Ft, Fb, Fi = budget(c)
        rec["t"].append(st.t)
        rec["F_total"].append(Ft)
        rec["F_bulk"].append(Fb)
        rec["F_interface"].append(Fi)
        rec["mass"].append(total_mass(st, dm, mesh, c))
        if n in snap_steps:
            rec["snaps"][n] = full(c).reshape(side, side).copy()
    for k in ("t", "F_total", "F_bulk", "F_interface", "mass"):
        rec[k] = np.asarray(rec[k])
    return rec
