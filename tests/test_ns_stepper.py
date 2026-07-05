"""M1b Task 5b gates: the linearized monolithic stepper on a transient
manufactured vortex — temporal order 2 (BDF2, spatial-error-free via a
fine-dt reference on the SAME mesh), bootstrap correctness, divergence
sentinel. Forcing uses the EXACT convection u*.grad u*; the stepper's
linearization at the extrapolated field differs by O(dt^2) — consistent."""
import numpy as np
import pytest
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.steppers.linearized import LinearizedMonolithicStepper
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from test_ns_bricks import u_star, p_star, f_star

pytestmark = pytest.mark.tier5

PI = np.pi
OM = 2 * PI          # temporal frequency: strong dt^2 signal
NU = 0.01


def F(t):
    return np.cos(OM * t)


def Fp(t):
    return -OM * np.sin(OM * t)


def u_ex(x, t):
    return F(t) * u_star(x)


def f_ex(x, t):
    # d/dt + exact convection + grad p - nu lap, from the steady pieces:
    # f_star(sigma=0, oseen) at F=1 gives conv + grad p - nu lap linearly
    # in each factor; scale: conv ~ F^2, grad p ~ F, lap ~ F.
    base_lin = f_star(x, NU, 0.0, False)          # grad p - nu lap (F=1)
    conv = f_star(x, NU, 0.0, True) - base_lin    # pure convection (F=1)
    return Fp(t) * u_star(x) + F(t) ** 2 * conv + F(t) * base_lin


def _make_stepper(level, dt, device, order=2, timestab=True):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    st = LinearizedMonolithicStepper(
        dm, NU, dt, f_fn=f_ex,
        g_fn=lambda x, t: np.zeros((len(x), 2)),
        order=order, timestab=timestab,
        p_pin_value_fn=lambda x0, t: F(t) * p_star(x0[None, :])[0])
    st.set_initial(lambda x: u_ex(x, 0.0))
    return st


def _run(level, nsteps, T, device, order=2, timestab=True):
    st = _make_stepper(level, T / nsteps, device, order, timestab)
    for _ in range(nsteps):
        x = st.step()
    return x[:, :2], st


def test_temporal_order2(device):
    # fixed mesh (level 4); errors vs a fine-dt reference on the SAME mesh
    # timestab OFF: tau must be dt-independent or its variation floors
    # the ladder at ~2e-3 (measured; the production timeStab toggle exists
    # for exactly this)
    T = 0.25
    ref, _ = _run(4, 64, T, device, timestab=False)
    errs = []
    for n in (8, 16, 32):
        u, _ = _run(4, n, T, device, timestab=False)
        errs.append(np.sqrt(((u - ref) ** 2).sum(1).mean()))
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
    assert rates[-1] > 1.7, (errs, rates)
    assert errs[-1] < 1e-3, errs


def test_accuracy_and_divergence_sentinel(device):
    # absolute accuracy vs exact + the mass sentinel
    T = 0.2
    u, st = _run(5, 20, T, device)
    err = np.sqrt(((u - u_ex(st.free_coords, T)) ** 2).sum(1).mean())
    assert err < 2.5e-2, err
    assert st.divergence_l2() < 0.5, st.divergence_l2()


def test_bdf1_vs_bdf2(device):
    # order=1 stepping must be clearly worse at equal dt (mechanism live)
    T = 0.25
    ref, _ = _run(4, 64, T, device, timestab=False)
    u2, _ = _run(4, 16, T, device, order=2, timestab=False)
    u1, _ = _run(4, 16, T, device, order=1, timestab=False)
    e2 = np.sqrt(((u2 - ref) ** 2).sum(1).mean())
    e1 = np.sqrt(((u1 - ref) ** 2).sum(1).mean())
    assert e1 > 2.5 * e2, (e1, e2)
