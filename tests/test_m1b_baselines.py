"""M1b locked benchmark baselines (plan Task 8). Two tiers:

- RECOMPUTED here (cheap, every suite run): cavity Re=100 monolithic du
  and cylinder Re=20 Cd — rtol 1e-4 (pseudo-time marches accumulate GPU
  atomic-ordering noise ~1e-8..1e-6; 1e-4 is ~100x above it and ~100x
  below regression signal).
- REFERENCE entries (measured once, guarded by their own tests' bands, too
  long to recompute in CI): cavity Re=100 leray, Re=1000, cylinder Re=100
  Cd, sphere Re=100 Cd. Each carries its config string.

Regenerate deliberately: .venv/bin/python tests/test_m1b_baselines.py
"""
import json
import os
import sys

import numpy as np
import pytest

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "baselines",
                             "m1b_baselines.json")
pytestmark = pytest.mark.tier5

REFERENCE = {
    "cavity_re100_leray_du": {
        "value": 0.0035, "rtol_note": "test_cavity leray lock du<0.02",
        "config": "level 5 p1, 200 fixed steps, dt=0.05"},
    "cavity_re1000_mono_du": {
        "value": 0.0727, "rtol_note": "nightly driver",
        "config": "level 6 p1, BDF1 pseudo-time, 900 steps, dt=0.05"},
    "cylinder_re100_cd": {
        "value": 1.352, "rtol_note": "lit ~1.33 unbounded; 14% blockage",
        "config": "level 6, BDF2 dt=0.02, steady wake (no shedding at this "
                  "blockage/dissipation; St = level-7 nightly)"},
    "sphere_re100_cd": {
        "value": 0.381, "rtol_note": "pipeline lock; D/h=3.8 preasymptotic",
        "config": "level 4 3D, BDF1 pseudo-time, 56 steps"},
}


def compute_recomputed(device):
    sys.path.insert(0, os.path.dirname(__file__))
    out = {}
    # cavity Re=100 monolithic centerline deviation
    from test_cavity import GHIA_Y, GHIA_U, _lid_g
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    from diffsim.steppers.linearized import LinearizedMonolithicStepper
    from diffsim.mesh.pointeval import point_eval_weights
    tree = build_uniform(5, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    st = LinearizedMonolithicStepper(
        dm, 0.01, 0.05, f_fn=lambda x, t: np.zeros((len(x), 2)),
        g_fn=_lid_g, order=1)
    st.set_initial(lambda x: np.zeros((len(x), 2)))
    for _ in range(200):
        u = st.step()[:, :2]
    W = point_eval_weights(mesh, np.stack(
        [np.full_like(GHIA_Y, 0.5), GHIA_Y], axis=1))
    u_c = np.asarray(W @ np.asarray(cons.T.tocsr() @ u[:, 0]))
    out["cavity_re100_mono_du"] = float(np.abs(u_c - GHIA_U).max())

    # cylinder Re=20 drag (the Task-8b configuration, fixed 60 steps)
    import test_cylinder as tc
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
    from diffsim.geometry.csg import Sphere
    from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                       GeometryData)
    from diffsim.sbm.vector import sbm_vector_dirichlet, surrogate_traction
    from diffsim.api.ns_bricks import assemble_linear_ns
    from diffsim.physics.poisson import gauss_points
    from diffsim.mesh.faces import face_tables
    ndof, dim, dt = 3, 2, 0.05
    oracle = Sphere(tc.CTR, tc.R)
    tree = build_uniform(5, dim=2)
    ret, _ = classify_lambda(tree, oracle, 0.5, domain="outside")
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 2),
                                domain="outside")
    T = cons.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    nfree = T.shape[1]
    coords = mesh.node_coords[cons.free_nodes]
    xq = gauss_points(mesh, dm.tables_by_p)
    on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
    strong = np.where(on(0.0, 0) | on(0.0, 1) | on(1.0, 1))[0]
    g_strong = np.zeros((len(strong), 2))
    g_strong[np.abs(coords[strong, 0]) < 1e-12, 0] = tc.U_IN
    Af, bf = sbm_vector_dirichlet(
        dm, sf, geo, lambda y: np.zeros((len(y), 2)), tc.NU, ndof)
    Af_c = (T_vec.T @ Af @ T_vec).tocsr()
    bf_c = np.asarray(T_vec.T @ bf)

    def gp_field(u_node):
        full = np.asarray(T @ u_node)
        aq, dq = {}, {}
        for pv in dm.bins:
            tb = dm.tables_by_p[pv]
            vals = full[mesh.conn_of[pv]]
            aq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
            h = mesh.tree.h()[mesh.bins[pv]]
            dq[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                      * (2.0 / h)[:, None]).reshape(-1)
        return aq, dq

    x = np.zeros(nfree * ndof)
    for _ in range(60):
        u_node = x.reshape(nfree, ndof)[:, :dim]
        aq, dq = gp_field(u_node)
        fq = {pv: aq[pv] / dt for pv in xq}
        A, b = assemble_linear_ns(dm, aq, dq, fq, tc.NU, sigma=1.0 / dt)
        A = (A + Af_c).tolil()
        b = b + bf_c
        for k, i in enumerate(strong):
            for c in range(dim):
                r = i * ndof + c
                A.rows[r] = [int(r)]
                A.data[r] = [1.0]
                b[r] = g_strong[k, c]
        pin = int(np.argmax(coords[:, 0] + coords[:, 1])) * ndof + dim
        A.rows[pin] = [pin]
        A.data[pin] = [1.0]
        b[pin] = 0.0
        x = splu(A.tocsr().tocsc()).solve(b)
    F = surrogate_traction(dm, sf, geo, np.asarray(T_vec @ x), tc.NU, ndof)
    out["cylinder_re20_cd"] = float(F[0] / (0.5 * tc.U_IN ** 2 * 2 * tc.R))
    return out


def test_m1b_baselines_locked(device):
    assert os.path.exists(BASELINE_PATH), (
        "generate with `python tests/test_m1b_baselines.py`")
    with open(BASELINE_PATH) as fh:
        ref = json.load(fh)
    cur = compute_recomputed(device)
    bad = []
    for k, v in cur.items():
        r = ref["recomputed"][k]
        rel = abs(v - r) / max(abs(r), 1e-300)
        if rel > 1e-4:
            bad.append(f"  {k}: locked {r:.8e} current {v:.8e} rel {rel:.2e}")
    assert not bad, "M1b baseline regression:\n" + "\n".join(bad)


if __name__ == "__main__":
    vals = compute_recomputed("cuda:0")
    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    with open(BASELINE_PATH, "w") as fh:
        json.dump({"recomputed": vals, "reference": REFERENCE}, fh,
                  indent=2, sort_keys=True)
    print(f"wrote {BASELINE_PATH}")
    for k, v in sorted(vals.items()):
        print(f"  recomputed {k} = {v:.8e}")
    for k, v in sorted(REFERENCE.items()):
        print(f"  reference  {k} = {v['value']}")
