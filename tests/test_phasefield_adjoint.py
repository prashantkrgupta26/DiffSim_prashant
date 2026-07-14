"""Differentiable phase-field adjoint gates — the THREE-WAY verification
(hand IFT adjoint == torch autograd twin == central finite difference) that
is the non-negotiable milestone gate.

G1: Cahn-Hilliard BDF1, material params dJ/d{M, kappa, chi(=B), A}.
G2a: variable-coefficient BDF2 (history cotangent gains the second slot).
Plus: the numpy discrete forward is bit-parity with the production
CahnHilliardStepper (so the adjoint differentiates the REAL brick)."""
import numpy as np
import pytest

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.adjoint.phasefield import (CHForward, CHAdjoint, FHEnergy,
                                        PolyEnergy)
from diffsim.adjoint.torch_twin import CHTwin
from diffsim.physics.cahn_hilliard import CahnHilliardStepper

pytestmark = pytest.mark.ad


def _dm(level, device):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    return dm, mesh


def _ic(coords):
    return 0.5 + 0.1 * np.cos(np.pi * coords[:, 0]) \
        * np.cos(np.pi * coords[:, 1])


def _three_way(dm, coords, order, n_steps, dt=0.01,
               M0=1.0, kap0=0.01, A0=1.0, B0=2.5):
    """Return {param: (adj, twin, fd)} for an FH CH march, J = 0.5|c_N-0.5|^2."""
    import torch
    nn = dm.n_nodes
    target = np.full(nn, 0.5)
    names = ["M", "kappa", "B", "A"]

    def run(M, kappa, A, B, record=False):
        fwd = CHForward(dm, FHEnergy(A, B), M=M, kappa=kappa, dt=dt,
                        order=order)
        fwd.set_initial(_ic(coords))
        fwd.run(n_steps)
        cN = fwd.steps[-1]["c"]
        J = 0.5 * float(((cN - target) ** 2).sum())
        return (fwd, J) if record else J

    fwd, _ = run(M0, kap0, A0, B0, record=True)
    adj = CHAdjoint(fwd)
    dJdx = [np.zeros(2 * nn) for _ in range(n_steps)]
    dJdx[-1][0::2] = fwd.steps[-1]["c"] - target
    g_adj = adj.gradient(dJdx, names)

    def fd(p):
        base = dict(M=M0, kappa=kap0, A=A0, B=B0)
        eps = 1e-6 * max(1.0, abs(base[p]))
        hi = dict(base); hi[p] += eps
        lo = dict(base); lo[p] -= eps
        return (run(hi['M'], hi['kappa'], hi['A'], hi['B'])
                - run(lo['M'], lo['kappa'], lo['A'], lo['B'])) / (2 * eps)
    g_fd = {p: fd(p) for p in names}

    twin = CHTwin(dm, energy="fh", dt=dt, order=order, device="cpu")
    c0 = torch.tensor(_ic(coords))
    leaves = {p: torch.tensor(v, requires_grad=True)
              for p, v in dict(M=M0, kappa=kap0, A=A0, B=B0).items()}
    out = twin.march(c0, None, leaves["M"], leaves["kappa"],
                     dict(A=leaves["A"], B=leaves["B"]), n_steps)
    loss = 0.5 * ((out[-1][0] - torch.tensor(target)) ** 2).sum()
    loss.backward()
    g_tw = {p: float(leaves[p].grad) for p in names}
    return {p: (g_adj[p], g_tw[p], g_fd[p]) for p in names}


def test_g1_ch_bdf1_material_params(device):
    """G1: dJ/d{M, kappa, chi} through a 3-step CH BDF1 march, three ways."""
    dm, mesh = _dm(3, device)
    res = _three_way(dm, mesh.node_coords, order=1, n_steps=3)
    for p, (a, t, f) in res.items():
        r_t = abs(a - t) / max(abs(t), 1e-14)
        r_f = abs(a - f) / max(abs(f), 1e-14)
        print(f"G1 {p:6s} adj={a:+.6e} twin={t:+.6e} fd={f:+.6e} "
              f"adj/twin={r_t:.2e} adj/fd={r_f:.2e}")
        # locked with >=2x headroom (measured adj/twin ~1e-16, adj/fd ~1e-9)
        assert r_t < 1e-10, (p, a, t)
        assert r_f < 1e-6, (p, a, f)


def test_g2a_ch_bdf2_material_params(device):
    """G2a: variable-coefficient BDF2 — the second history slot's cotangent
    must be exact.  4-step march (steps 2-4 are BDF2)."""
    dm, mesh = _dm(3, device)
    res = _three_way(dm, mesh.node_coords, order=2, n_steps=4)
    for p, (a, t, f) in res.items():
        r_t = abs(a - t) / max(abs(t), 1e-14)
        r_f = abs(a - f) / max(abs(f), 1e-14)
        print(f"G2a {p:6s} adj={a:+.6e} twin={t:+.6e} fd={f:+.6e} "
              f"adj/twin={r_t:.2e} adj/fd={r_f:.2e}")
        assert r_t < 1e-10, (p, a, t)
        assert r_f < 1e-6, (p, a, f)


def _cryst_three_way(dm, coords, order, n_steps, dt=0.01):
    """Coupled CH x AC (M=K=1) crystallisation: dJ/d{M,kappa,eps2,L,dsig,
    dh,Tm,A,B} three ways.  J = 0.5(|phi_N-0.5|^2 + |psi_N-0.4|^2)."""
    import torch
    from diffsim.adjoint.crystallization import CACHForward, CACHAdjoint
    from diffsim.adjoint.torch_twin import CACHTwin
    nn = dm.n_nodes
    NF = 3
    cc = np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])
    phi0 = 0.5 + 0.1 * cc
    psi0 = 0.3 + 0.1 * cc
    tgt_phi = np.full(nn, 0.5)
    tgt_psi = np.full(nn, 0.4)
    base = dict(M=1.0, kappa=0.02, eps2=0.02, L=1.0, dsig=1.0, dh=1.0,
                Tm=2.0, T=0.5, A=1.0, B=2.5)
    names = ["M", "kappa", "eps2", "L", "dsig", "dh", "Tm", "A", "B"]

    def run(P, record=False):
        fwd = CACHForward(dm, FHEnergy(P["A"], P["B"]), M=P["M"],
                          kappa=P["kappa"], eps2=P["eps2"], L=P["L"],
                          dsig=P["dsig"], dh=P["dh"], Tm=P["Tm"], T=P["T"],
                          dt=dt, order=order)
        fwd.set_initial(phi0, psi0)
        fwd.run(n_steps)
        xN = fwd.steps[-1]["x"]
        J = 0.5 * float(((xN[0::NF] - tgt_phi) ** 2).sum()
                        + ((xN[2::NF] - tgt_psi) ** 2).sum())
        return (fwd, J) if record else J

    fwd, _ = run(base, record=True)
    adj = CACHAdjoint(fwd)
    dJdx = [np.zeros(NF * nn) for _ in range(n_steps)]
    xN = fwd.steps[-1]["x"]
    dJdx[-1][0::NF] = xN[0::NF] - tgt_phi
    dJdx[-1][2::NF] = xN[2::NF] - tgt_psi
    g_adj = adj.gradient(dJdx, names)

    def fd(p):
        eps = 1e-6 * max(1.0, abs(base[p]))
        hi = dict(base); hi[p] += eps
        lo = dict(base); lo[p] -= eps
        return (run(hi) - run(lo)) / (2 * eps)
    g_fd = {p: fd(p) for p in names}

    twin = CACHTwin(dm, dt=dt, order=order, device="cpu")
    leaves = {k: torch.tensor(float(v), requires_grad=True)
              for k, v in base.items()}
    out = twin.march(torch.tensor(phi0), torch.tensor(psi0), leaves, n_steps)
    xNt = out[-1]
    loss = 0.5 * (((xNt[0::NF] - torch.tensor(tgt_phi)) ** 2).sum()
                  + ((xNt[2::NF] - torch.tensor(tgt_psi)) ** 2).sum())
    loss.backward()
    g_tw = {p: float(leaves[p].grad) for p in names}
    return {p: (g_adj[p], g_tw[p], g_fd[p]) for p in names}


def test_g2b_crystallization_bdf1(device):
    """G2b: coupled CH x AC crystallisation params (dh, Tm, dsigma, eps2, L)
    three-way verified, BDF1, 3 steps.  Capability (a) = complete."""
    dm, mesh = _dm(3, device)
    res = _cryst_three_way(dm, mesh.node_coords, order=1, n_steps=3)
    for p, (a, t, f) in res.items():
        r_t = abs(a - t) / max(abs(t), 1e-14)
        r_f = abs(a - f) / max(abs(f), 1e-14)
        print(f"G2b {p:6s} adj={a:+.6e} adj/twin={r_t:.2e} adj/fd={r_f:.2e}")
        assert r_t < 1e-10, (p, a, t)
        assert r_f < 1e-6, (p, a, f)


def test_g2c_crystallization_bdf2(device):
    """G2c: crystallisation params, variable-coefficient BDF2, 4 steps —
    both conserved-in-time fields (phi AND psi) carry history cotangents."""
    dm, mesh = _dm(3, device)
    res = _cryst_three_way(dm, mesh.node_coords, order=2, n_steps=4)
    for p, (a, t, f) in res.items():
        r_t = abs(a - t) / max(abs(t), 1e-14)
        r_f = abs(a - f) / max(abs(f), 1e-14)
        print(f"G2c {p:6s} adj={a:+.6e} adj/twin={r_t:.2e} adj/fd={r_f:.2e}")
        assert r_t < 1e-10, (p, a, t)
        assert r_f < 1e-6, (p, a, f)


def test_forward_parity_production(device):
    """The numpy discrete forward must reproduce the production
    CahnHilliardStepper to Newton tolerance — otherwise the adjoint is not
    differentiating the real brick.  (measured ~1e-16, machine precision.)"""
    dm, mesh = _dm(3, device)
    coords = mesh.node_coords

    # poly, BDF1
    ic = lambda x: 0.3 * np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
    myf = CHForward(dm, PolyEnergy(), M=1.0, kappa=0.01, dt=0.005, order=1)
    myf.set_initial(ic(coords))
    prod = CahnHilliardStepper(dm, 1.0, 0.01, 0.005, order=1)
    prod.set_initial(lambda x: ic(x))
    for _ in range(5):
        myf.step(); prod.step()
    dev_c = np.abs(myf.c - prod.x[0::2]).max()
    print(f"parity poly-BDF1: max|dc|={dev_c:.2e}")
    assert dev_c < 1e-12, dev_c

    # FH, BDF2
    ic2 = lambda x: 0.5 + 0.1 * np.cos(np.pi * x[:, 0]) \
        * np.cos(np.pi * x[:, 1])
    myf2 = CHForward(dm, FHEnergy(1.0, 2.5), M=1.0, kappa=0.01, dt=0.005,
                     order=2)
    myf2.set_initial(ic2(coords))
    prod2 = CahnHilliardStepper(dm, 1.0, 0.01, 0.005, order=2, energy="fh",
                                fh_A=1.0, fh_B=2.5)
    prod2.set_initial(lambda x: ic2(x))
    for _ in range(5):
        myf2.step(); prod2.step()
    dev_c2 = np.abs(myf2.c - prod2.x[0::2]).max()
    print(f"parity FH-BDF2: max|dc|={dev_c2:.2e}")
    assert dev_c2 < 1e-12, dev_c2
