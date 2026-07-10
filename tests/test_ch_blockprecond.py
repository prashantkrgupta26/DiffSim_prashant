"""G1 gate: the blockch two-factor Schur preconditioner (linsolve.py).

Contract (measured 2026-07-09): on Newton-CONVERGED regimes the blockch
trajectory matches splu and outer iteration counts are small and flat in
h (L5: 3/1 its poly/FH at dt = 2e-3; L6: 1 it both). Fixed LARGE dt
slammed into a quench onset is OUT of contract: Newton itself does not
converge there (15-iteration cap, chaotic wells), so any two linear
solvers land in different wells and trajectory parity is ill-posed
(measured 2.7e-2 'parity' between splu and blockch at dt = 2e-2 onset —
the wodo device-parity lesson at the nonlinear level). Production
marches are LTE-adaptive: onset gets small dt automatically; the
adaptive-march gate below sweeps sigma continuously over PHYSICAL
states, which is the robustness claim that matters.

TOLERANCE NOTE (the parity-gate lesson): parity asserted at 1e-7, three
decades above the measured 1e-10..5e-13; iteration bounds carry >2x
headroom over measured maxima."""
import numpy as np
import pytest

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.cahn_hilliard import CahnHilliardStepper

pytestmark = pytest.mark.tier3


def _dm(level, device):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    return DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)


def _stepper(dm, energy, solver, dt, order=1):
    st = CahnHilliardStepper(dm, M=1.0, kappa=2e-3, dt=dt, order=order,
                             linsolver=solver, energy=energy)
    rng = np.random.default_rng(3)
    ic = ((lambda x: 0.05 * rng.standard_normal(len(x)))
          if energy == "poly" else
          (lambda x: 0.5 + 0.05 * rng.standard_normal(len(x))))
    st.set_initial(ic, mu_init="consistent")
    return st


@pytest.mark.parametrize("energy", ["poly", "fh"])
def test_blockch_parity_and_iterations(energy, device):
    """Newton-converged regime (dt = 2e-3 quench, L5): trajectory parity
    vs splu + small outer counts. Measured: poly 3 its / 1.0e-10 parity;
    FH 1 it / 5.5e-13."""
    dm = _dm(5, device)
    sts = _stepper(dm, energy, "splu", 0.002)
    stb = _stepper(dm, energy, "blockch", 0.002)
    its = []
    for _ in range(6):
        c_ref, _ = sts.step()
        c_bp, _ = stb.step()
        its.append(stb._solver_cache[("blockch_iters", "ch")][0])
    rel = np.abs(c_ref - c_bp).max() / max(np.abs(c_ref).max(), 1e-30)
    print(f"blockch {energy}: parity {rel:.2e}, outer its {its}")
    assert rel < 1e-7, rel
    assert max(i % 1000 for i in its) <= 12, its
    assert not any(i >= 1000 for i in its), ("fallback engaged in the "
                                             "converged regime", its)


@pytest.mark.parametrize("energy", ["poly", "fh"])
def test_blockch_adaptive_march(energy, device):
    """The production-shaped robustness gate: an LTE-adaptive march from
    quench into early coarsening completes with blockch, sweeping sigma
    continuously over physical states; dt grows; fields stay physical."""
    from diffsim.physics.cahn_hilliard import adaptive_march
    dm = _dm(5, device)
    st = _stepper(dm, energy, "blockch", 0.002, order=2)
    its = []
    orig = st.step

    def rec():
        r = orig()
        its.append(st._solver_cache[("blockch_iters", "ch")][0])
        return r

    st.step = rec
    ts, dts = adaptive_march(st, t_end=0.15, tol=5e-4)
    c = st.hist[0]
    mx = max(i % 1000 for i in its)
    fb = sum(1 for i in its if i >= 1000)
    print(f"blockch {energy} adaptive: {len(dts)} accepted steps, dt "
          f"{dts[0]:.4f}->{dts[-1]:.4f}, max outer {mx}, fallbacks {fb}")
    assert len(dts) > 0 and np.isfinite(c).all()
    if energy == "fh":
        assert c.min() > 0.0 and c.max() < 1.0
    # measured over t=0.5 marches: poly max 3 (4026 solves), FH max 4
    # (1983 solves), zero fallbacks; bound at 10x the measured max
    assert mx <= 40, mx
    assert fb == 0, ("fallback engaged on a physical adaptive "
                     "trajectory", its)
