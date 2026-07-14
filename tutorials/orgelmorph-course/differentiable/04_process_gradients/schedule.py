"""OrgElMorph course - Differentiable D4: process gradients (the quench schedule).

Importable core for the fourth differentiable concept.  Material
parameters (D1-D3) are what the blend IS; PROCESS parameters are what we
DO to it.  The headline process control is the temperature schedule
T(t) - the quench - applied while the film crystallises.  This concept
computes the gradient of a morphology objective with respect to the WHOLE
schedule at once, as a TIME SERIES

    dJ/dT_n ,   n = 0 .. N-1

- one number per step, telling us how much the temperature at each moment
  of the process matters to the outcome.  Getting all N of them from a
  single reverse sweep (not N separate finite-difference probes) is what
  makes process design tractable.

The physics is the coupled Cahn-Hilliard x Allen-Cahn crystallisation
system (retained-species core, M=K=1): a conserved composition phi and a
non-conserved crystallinity psi.  The temperature enters each step's
residual through TWO channels:
  * the Turnbull crystallisation drive  drive = dh (T/Tm - 1), and
  * the Flory interaction  B(T) = B0 + bT (T - Tref).
The adjoint's temperature_gradient() accumulates both.  Verified against
central finite differences step by step.
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.adjoint.phasefield import FHEnergy
from diffsim.adjoint.crystallization import CACHForward, CACHAdjoint

NF = 3   # dofs per node: (phi, mu, psi)

# base material/process constants (the G3 gate setting - interior & stable)
BASE = dict(M=1.0, kappa=0.02, eps2=0.02, L=1.0, dsig=1.0, dh=1.0, Tm=2.0,
            A=1.0, B=2.5)
BT, TREF = 0.8, 0.5              # T couples into the Flory chi B(T)


def build_dm(level=3, device="cuda:0"):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    return dm, mesh


def _ic(coords):
    cc = np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])
    return 0.5 + 0.1 * cc, 0.3 + 0.1 * cc          # phi0, psi0


def march(dm, coords, Tsched, dt=0.01, order=2):
    """Forward crystallisation march driven by the temperature schedule
    Tsched (one control temperature per step).  Returns the recorder."""
    phi0, psi0 = _ic(coords)
    fwd = CACHForward(dm, FHEnergy(BASE["A"], BASE["B"]), M=BASE["M"],
                      kappa=BASE["kappa"], eps2=BASE["eps2"], L=BASE["L"],
                      dsig=BASE["dsig"], dh=BASE["dh"], Tm=BASE["Tm"],
                      T=Tsched[0], dt=dt, order=order)
    fwd.set_initial(phi0, psi0)
    fwd.set_schedule(list(Tsched), bT=BT, Tref=TREF)
    fwd.run(len(Tsched))
    return fwd


def objective(fwd, tgt_phi, tgt_psi):
    """J = 1/2 (|phi_N - tgt_phi|^2 + |psi_N - tgt_psi|^2) on the final
    fields."""
    xN = fwd.steps[-1]["x"]
    return 0.5 * float(((xN[0::NF] - tgt_phi) ** 2).sum()
                       + ((xN[2::NF] - tgt_psi) ** 2).sum())


def crystallinity_trace(fwd):
    """Mean crystallinity <psi> at each recorded step (the process
    output we watch respond to the schedule)."""
    return np.array([float(np.mean(s["x"][2::NF])) for s in fwd.steps])


def schedule_gradient(level=3, dt=0.01, order=2, device="cuda:0",
                      Tsched=(0.5, 0.6, 0.7, 0.55),
                      tgt_phi_val=0.5, tgt_psi_val=0.45):
    """Compute the time-series schedule gradient dJ/dT_n by adjoint, and
    verify it step by step against central finite differences.  Returns
    everything the tutorial cites (schedule, crystallinity trace,
    gradient, FD gradient, agreements)."""
    dm, mesh = build_dm(level, device)
    coords = mesh.node_coords
    nn = dm.n_nodes
    Tsched = np.asarray(Tsched, float)
    NS = len(Tsched)
    tgt_phi = np.full(nn, tgt_phi_val)
    tgt_psi = np.full(nn, tgt_psi_val)

    fwd = march(dm, coords, Tsched, dt, order)
    J = objective(fwd, tgt_phi, tgt_psi)
    cryst = crystallinity_trace(fwd)

    # adjoint: ONE reverse sweep -> the whole dJ/dT_n time series
    dJdx = [np.zeros(NF * nn) for _ in range(NS)]
    xN = fwd.steps[-1]["x"]
    dJdx[-1][0::NF] = xN[0::NF] - tgt_phi
    dJdx[-1][2::NF] = xN[2::NF] - tgt_psi
    gT_adj = CACHAdjoint(fwd).temperature_gradient(dJdx)

    # finite-difference verification, one step at a time
    def J_of(Ts):
        f = march(dm, coords, Ts, dt, order)
        return objective(f, tgt_phi, tgt_psi)

    gT_fd = np.zeros(NS)
    for i in range(NS):
        e = 1e-6
        hi = Tsched.copy(); hi[i] += e
        lo = Tsched.copy(); lo[i] -= e
        gT_fd[i] = (J_of(hi) - J_of(lo)) / (2 * e)

    rel = np.abs(gT_adj - gT_fd) / np.maximum(np.abs(gT_fd), 1e-14)
    return dict(Tsched=Tsched, cryst=cryst, J=J, gT_adj=gT_adj,
                gT_fd=gT_fd, rel=rel, NS=NS, nn=nn,
                side=int(round(np.sqrt(nn))),
                tgt_phi=tgt_phi_val, tgt_psi=tgt_psi_val)
