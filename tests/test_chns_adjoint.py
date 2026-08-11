"""tests/test_chns_adjoint.py — Task 11 (SP-0): discrete-IFT adjoint through
the coupled CHNS BDF1 march, three-way verified (hand adjoint = torch twin =
central FD) for the 5 scalar params x 2 objectives.

CH interface only (the SP-0 adjoint exit gate; CAC adjoint is SP-1+).

Config: coarse bubble rise, level 4 (16x16), 5 BDF1 steps, rho_ratio 10,
Cn_override="2h" for resolvability (matches the forward gates).

Debug order (crystallization discipline): twin-vs-FD FIRST (validates the twin
independently of the hand adjoint), then hand-vs-twin (isolates transposition
bugs).  Tolerances: hand-vs-FD rel <= 1e-4 (FD to its best plateau over
eps in {1e-4,1e-5,1e-6}); hand-vs-twin rel <= 1e-6.
"""
import os as _os
import sys as _sys

import numpy as np
import pytest

_BENCH_DIR = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..",
                  "benchmarks"))
if _BENCH_DIR not in _sys.path:
    _sys.path.insert(0, _BENCH_DIR)
_REPO_DIR = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
if _REPO_DIR not in _sys.path:
    _sys.path.insert(0, _REPO_DIR)

from chns.cases import BUBBLE_RISE_RE35_WE10  # noqa: E402

PARAMS = ("rho_ratio", "eta_ratio", "We", "mobility", "Fr")
OBJECTIVES = ("terminal_phi_mismatch", "centroid_y")
N_STEPS = 5
LEVEL = 4


def _make_dm(level, dim=2):
    import warp as wp
    wp.init()
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    from diffsim import default_device

    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim),
                              default_device())
    return dm, mesh, cons


def _setup():
    """Shared coarse bubble-rise config; returns (dm, coords, phi0, Cn,
    params0, phi_target)."""
    dm, mesh, cons = _make_dm(LEVEL, dim=2)
    coords = mesh.node_coords
    h = float(dm.mesh.tree.h().min())
    Cn = 2.0 * h
    r = np.sqrt((coords[:, 0] - 0.5) ** 2 + (coords[:, 1] - 0.35) ** 2)
    phi0 = -np.tanh((r - 0.2) / (Cn * np.sqrt(2.0)))
    case = BUBBLE_RISE_RE35_WE10
    params0 = dict(rho_ratio=10.0, eta_ratio=10.0, We=float(case.We),
                   mobility=1.0 / float(case.Pe), Fr=float(case.Fr))
    # phi_target for the mismatch objective: a slightly-shifted bubble so the
    # gradient is nonzero (target generated at a different bubble centre).
    rt = np.sqrt((coords[:, 0] - 0.5) ** 2 + (coords[:, 1] - 0.45) ** 2)
    phi_target = -np.tanh((rt - 0.2) / (Cn * np.sqrt(2.0)))
    return dm, coords, phi0, Cn, params0, phi_target


def _make_mirror(dm, Cn, params0):
    from dataclasses import replace
    from diffsim.adjoint.chns import CHNSDiscrete
    case = replace(BUBBLE_RISE_RE35_WE10, rho_ratio=params0["rho_ratio"],
                   eta_ratio=params0["eta_ratio"], We=params0["We"],
                   Pe=1.0 / params0["mobility"], Fr=params0["Fr"])
    m = CHNSDiscrete(level=LEVEL, dim=2, case=case, dt=case.dt0, dm=dm,
                     gravity=True, Cn_override=Cn, newton_tol=1e-12)
    return m


def _hand_grads(dm, coords, phi0, Cn, params0, objective, phi_target):
    from diffsim.adjoint.chns import CHNSAdjoint
    m = _make_mirror(dm, Cn, params0)
    m.set_initial(phi0)
    adj = CHNSAdjoint(m, objective=objective, phi_target=phi_target,
                      coords=coords)
    adj.march(N_STEPS)
    return adj.gradients(PARAMS)


def _twin(dm, Cn, params0):
    from diffsim.adjoint.torch_twin import CHNSTwin
    case = BUBBLE_RISE_RE35_WE10
    return CHNSTwin(dm, case, dt=case.dt0, Cn=Cn, gravity=True)


# ---------------------------------------------------------------------------
# Step 1 (debug order): twin vs central FD — validates the twin independently.
# ---------------------------------------------------------------------------
@pytest.mark.ad
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_twin_vs_fd(objective):
    dm, coords, phi0, Cn, params0, phi_target = _setup()
    tw = _twin(dm, Cn, params0)
    g_twin = tw.grads(phi0, params0, list(PARAMS), N_STEPS, objective,
                      phi_target=phi_target, coords=coords)

    worst = 0.0
    for p in PARAMS:
        p0 = params0[p]
        scale = abs(p0) if abs(p0) > 0 else 1.0
        best = np.inf
        for eps_rel in (1e-4, 1e-5, 1e-6):
            eps = eps_rel * scale
            pp = dict(params0); pp[p] = p0 + eps
            pm = dict(params0); pm[p] = p0 - eps
            Jp = tw.loss(phi0, pp, N_STEPS, objective,
                         phi_target=phi_target, coords=coords)
            Jm = tw.loss(phi0, pm, N_STEPS, objective,
                         phi_target=phi_target, coords=coords)
            fd = (Jp - Jm) / (2.0 * eps)
            rel = abs(fd - g_twin[p]) / max(abs(g_twin[p]), 1e-14)
            best = min(best, rel)
        worst = max(worst, best)
        assert best <= 1e-4, (
            f"[{objective}] twin-vs-FD {p}: rel {best:.3e} > 1e-4 "
            f"(twin={g_twin[p]:.6e})")
    print(f"[twin-vs-FD {objective}] worst rel = {worst:.3e}")


# ---------------------------------------------------------------------------
# Step 2 (debug order): hand adjoint vs twin — isolates transposition bugs.
# ---------------------------------------------------------------------------
@pytest.mark.ad
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_hand_vs_twin(objective):
    dm, coords, phi0, Cn, params0, phi_target = _setup()
    g_hand = _hand_grads(dm, coords, phi0, Cn, params0, objective, phi_target)
    tw = _twin(dm, Cn, params0)
    g_twin = tw.grads(phi0, params0, list(PARAMS), N_STEPS, objective,
                      phi_target=phi_target, coords=coords)
    worst = 0.0
    for p in PARAMS:
        rel = abs(g_hand[p] - g_twin[p]) / max(abs(g_twin[p]), 1e-14)
        worst = max(worst, rel)
        assert rel <= 1e-6, (
            f"[{objective}] hand-vs-twin {p}: rel {rel:.3e} > 1e-6 "
            f"(hand={g_hand[p]:.6e} twin={g_twin[p]:.6e})")
    print(f"[hand-vs-twin {objective}] worst rel = {worst:.3e}")


# ---------------------------------------------------------------------------
# Step 3: hand adjoint vs central FD (through the mirror march directly).
# ---------------------------------------------------------------------------
@pytest.mark.ad
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_hand_vs_fd(objective):
    from diffsim.adjoint.chns import CHNSAdjoint
    dm, coords, phi0, Cn, params0, phi_target = _setup()
    g_hand = _hand_grads(dm, coords, phi0, Cn, params0, objective, phi_target)

    def loss_at(pp):
        m = _make_mirror(dm, Cn, pp)
        m.set_initial(phi0)
        adj = CHNSAdjoint(m, objective=objective, phi_target=phi_target,
                          coords=coords)
        adj.march(N_STEPS)
        return adj.objective_value()

    worst = 0.0
    for p in PARAMS:
        p0 = params0[p]
        scale = abs(p0) if abs(p0) > 0 else 1.0
        best = np.inf
        for eps_rel in (1e-4, 1e-5, 1e-6):
            eps = eps_rel * scale
            pp = dict(params0); pp[p] = p0 + eps
            pm = dict(params0); pm[p] = p0 - eps
            fd = (loss_at(pp) - loss_at(pm)) / (2.0 * eps)
            rel = abs(fd - g_hand[p]) / max(abs(g_hand[p]), 1e-14)
            best = min(best, rel)
        worst = max(worst, best)
        assert best <= 1e-4, (
            f"[{objective}] hand-vs-FD {p}: rel {best:.3e} > 1e-4 "
            f"(hand={g_hand[p]:.6e})")
    print(f"[hand-vs-FD {objective}] worst rel = {worst:.3e}")


# ---------------------------------------------------------------------------
# Step 4: gradient-descent smoke — 3 iters on We to reduce the mismatch
# objective toward a target generated at a DIFFERENT We; J must decrease
# monotonically ("gradients actually usable").
# ---------------------------------------------------------------------------
@pytest.mark.ad
def test_descent_smoke_We():
    from diffsim.adjoint.chns import CHNSAdjoint
    dm, coords, phi0, Cn, params0, _ = _setup()

    # target = terminal phi from a march at a DIFFERENT We
    p_tgt = dict(params0); p_tgt["We"] = params0["We"] * 1.5
    m = _make_mirror(dm, Cn, p_tgt)
    m.set_initial(phi0)
    adj_t = CHNSAdjoint(m, objective="centroid_y", coords=coords)
    adj_t.march(N_STEPS)
    phi_target = m.unpack(adj_t.steps[-1]["x"])[2].copy()

    We = params0["We"]
    lr = None
    Js = []
    for it in range(3):
        pp = dict(params0); pp["We"] = We
        m = _make_mirror(dm, Cn, pp)
        m.set_initial(phi0)
        adj = CHNSAdjoint(m, objective="terminal_phi_mismatch",
                          phi_target=phi_target, coords=coords)
        adj.march(N_STEPS)
        J = adj.objective_value()
        g = adj.gradients(("We",))["We"]
        Js.append(J)
        if lr is None:
            # first-step learning rate: normalise so the We step is ~ 0.1*We
            lr = 0.1 * We / max(abs(g), 1e-30)
        We = We - lr * g
    print(f"[descent We] J trajectory = {['%.4e' % j for j in Js]}, "
          f"We_final = {We:.4f}")
    assert Js[1] < Js[0] and Js[2] < Js[1], (
        f"J not monotonically decreasing: {Js}")
