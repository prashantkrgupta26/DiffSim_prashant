"""M1c gate: transient adjoint chain — dJ/dnu over a 4-step BDF2 cavity
run vs central FD (full rerun). All cotangents taped; no FD inside the
adjoint."""
import numpy as np
import pytest

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.steppers.linearized import LinearizedMonolithicStepper
from diffsim.sbm.transient_adjoint import TransientAdjoint

pytestmark = pytest.mark.ad


def _lid(x, t):
    g = np.zeros((len(x), 2))
    g[np.abs(x[:, 1] - 1.0) < 1e-12, 0] = 1.0
    return g


def _run(nu, device, n_steps=4, record=False):
    tree = build_uniform(4, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    st = LinearizedMonolithicStepper(
        dm, nu, 0.05, f_fn=lambda x, t: np.zeros((len(x), 2)),
        g_fn=_lid, order=2)
    st.finescale_extrap = False
    st.set_initial(lambda x: np.zeros((len(x), 2)))
    if record:
        ta = TransientAdjoint(st)
        xs = ta.run(n_steps)
        return ta, xs
    xs = []
    for _ in range(n_steps):
        xs.append(st.step().reshape(-1).copy())
    return None, xs


def test_transient_nu_gradient(device):
    nu0 = 0.02
    ta, xs = _run(nu0, device, record=True)
    J0 = 0.5 * sum(float(x @ x) for x in xs)
    dJdx = [x.copy() for x in xs]
    g_adj = ta.nu_gradient(dJdx)

    eps = 1e-6
    Js = []
    for dnu in (+eps, -eps):
        _, xs2 = _run(nu0 + dnu, device)
        Js.append(0.5 * sum(float(x @ x) for x in xs2))
    fd = (Js[0] - Js[1]) / (2 * eps)
    print(f"transient dJ/dnu: adjoint = {g_adj:.8e}  FD = {fd:.8e}  "
          f"rel = {abs(g_adj - fd) / max(abs(fd), 1e-14):.2e}")
    assert abs(g_adj - fd) < 5e-5 * max(abs(fd), 1e-12), (g_adj, fd, J0)


def test_dim3_nu_gradient(device):
    """dim-3 composition gate: single-step dJ/dnu on a small 3-D cavity
    (the transient machinery is dim-generic; the kernels now exist)."""
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    from diffsim.steppers.linearized import LinearizedMonolithicStepper
    from diffsim.sbm.transient_adjoint import TransientAdjoint

    def lid3(x, t):
        g = np.zeros((len(x), 3))
        g[np.abs(x[:, 2] - 1.0) < 1e-12, 0] = 1.0
        return g

    def run(nu, record):
        tree = build_uniform(3, dim=3)
        mesh = build_mesh(tree, p=1)
        cons = build_constraints(mesh)
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3),
                                  device)
        st = LinearizedMonolithicStepper(
            dm, nu, 0.05, f_fn=lambda x, t: np.zeros((len(x), 3)),
            g_fn=lid3, order=2)
        st.finescale_extrap = False
        st.set_initial(lambda x: np.zeros((len(x), 3)))
        if record:
            ta = TransientAdjoint(st)
            xs = ta.run(2)
            return ta, xs
        return None, [st.step().reshape(-1).copy() for _ in range(2)]

    nu0 = 0.05
    ta, xs = run(nu0, True)
    g_adj = ta.nu_gradient([x.copy() for x in xs])
    eps = 1e-6
    Js = []
    for dnu in (+eps, -eps):
        _, xs2 = run(nu0 + dnu, False)
        Js.append(0.5 * sum(float(x @ x) for x in xs2))
    fd = (Js[0] - Js[1]) / (2 * eps)
    print(f"dim-3 transient dJ/dnu: adj={g_adj:.6e} fd={fd:.6e} "
          f"rel={abs(g_adj-fd)/max(abs(fd),1e-14):.2e}")
    assert abs(fd - g_adj) < 5e-5 * max(abs(fd), 1e-12), (g_adj, fd)


@pytest.mark.ad
def test_transient_shape_gradient(device):
    """Transient SHAPE gradient: d(sum_n 0.5|x_n|^2)/d(cylinder center)
    over a 3-step BDF2 start-up flow — face cotangents accumulated over
    the reverse sweep, composed through the oracle torch chain; vs FD."""
    from diffsim.geometry.csg import Sphere
    from diffsim.mesh.faces import face_tables
    from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                       GeometryData)
    from diffsim.sbm.transient_adjoint import TransientShapeAdjoint

    R_, U_IN, NU = 0.07, 1.0, 0.014

    def make(center):
        oracle = Sphere(tuple(center), R_)
        tree = build_uniform(5, dim=2)
        ret, _ = classify_lambda(tree, oracle, 0.5, domain="outside")
        sf = extract_surrogate(ret)
        mesh = build_mesh(ret, p=1)
        cons = build_constraints(mesh)
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                                  device)
        geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 2),
                                    domain="outside")
        coords = mesh.node_coords[cons.free_nodes]
        on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
        strong = np.where(on(0.0, 0) | on(0.0, 1) | on(1.0, 1))[0]
        g_strong = np.zeros((len(strong), 2))
        g_strong[np.abs(coords[strong, 0]) < 1e-12, 0] = U_IN
        return TransientShapeAdjoint(dm, sf, geo, oracle, NU, 0.05,
                                     10.0, strong, g_strong), ret

    c0 = np.array([0.3, 0.5])
    tsa, ret0 = make(c0)
    xs = tsa.run(3)
    tsa.shape_gradient([x.copy() for x in xs])
    g_adj = tsa.oracle.center.grad.numpy().copy()

    eps = 1e-6
    scale = max(np.abs(g_adj).max(), 1e-12)
    for i in range(2):
        Js = []
        for s_ in (+eps, -eps):
            c = c0.copy(); c[i] += s_
            t2, ret2 = make(c)
            assert np.array_equal(ret2.keys, ret0.keys)
            xs2 = t2.run(3)
            Js.append(0.5 * sum(float(x @ x) for x in xs2))
        fd = (Js[0] - Js[1]) / (2 * eps)
        print(f"transient shape grad c[{i}]: adj={g_adj[i]:+.6e} "
              f"fd={fd:+.6e} rel={abs(g_adj[i]-fd)/max(abs(fd),1e-14):.2e}")
        assert abs(fd - g_adj[i]) < 5e-4 * max(abs(fd), scale), (
            i, fd, g_adj[i])
