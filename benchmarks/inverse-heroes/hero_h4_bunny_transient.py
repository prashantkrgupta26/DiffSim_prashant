"""Hero H4 (M2 headline): recover a hidden GENIE edit of the STANFORD
BUNNY from TRANSIENT flow probes — the H2 story on the real two-head
checkpoint. Steady H3 plateaued (observation-limited: findings
4f-addendum); the transient wake is the informative measurement.

J = sum_n sum_probes |u_n(probe) - u*_n(probe)|^2 over an N-step BDF2
start-up flow. Frozen epoch + anchored feet (the H1 lessons); recovery
via Gauss-Newton with FD Jacobian columns (k transient runs per epoch);
the ADJOINT gradient from the reverse sweep is the per-epoch consistency
check (cos printed) — and the scalable path at large k.

Run: python benchmarks/hero_h2_sphere_transient.py [n_steps] [n_epochs]
"""

import os as _bos, sys as _bsys  # noqa: E402  (benchmark import bootstrap)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import _bench_bootstrap  # noqa: E402,F401
import sys

import numpy as np
import torch

sys.path.insert(0, "tests")
sys.path.insert(0, "benchmarks")
from hero_h1_sphere_steady import K, U_IN, NU, ALPHA_F
from hero_h3_bunny_steady import make_bunny_oracle as make_oracle
from hero_h3_bunny_steady import BUNNY_JSON
from diffsim.geometry.provided_inr import ProvidedINROracle, extract_modes
from diffsim import default_device
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                   GeometryData)
from diffsim.sbm.transient_adjoint import TransientShapeAdjoint
from diffsim.mesh.pointeval import point_eval_weights

LEVEL, N_STEPS = 4, 8
DT = 0.05
# bunny wake plane (bbox [0.28,0.79]; H3 diagnosis)
_py, _pz = np.meshgrid(np.linspace(0.35, 0.65, 4),
                       np.linspace(0.35, 0.65, 4))
PROBES = np.column_stack([np.full(16, 0.90), _py.ravel(), _pz.ravel()])

_EPOCH = {}


def build_epoch(V):
    o0 = make_oracle(np.zeros(K), V)
    tree = build_uniform(LEVEL, dim=3)
    ret, _ = classify_lambda(tree, o0, 1.0, domain="outside")
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), default_device())
    coords = mesh.node_coords[cons.free_nodes]
    on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
    strong = np.where(on(0.0, 0) | on(0.0, 1) | on(1.0, 1)
                      | on(0.0, 2) | on(1.0, 2))[0]
    g_strong = np.zeros((len(strong), 3))
    g_strong[np.abs(coords[strong, 0]) < 1e-12, 0] = U_IN
    W = point_eval_weights(mesh, PROBES)
    _EPOCH.update(ret=ret, sf=sf, mesh=mesh, cons=cons, dm=dm,
                  strong=strong, g_strong=g_strong, W=W)


def run_transient(alpha_np, V, n_steps=N_STEPS):
    """Forward N-step run; returns (tsa, xs, probe series [n, np, 3])."""
    if not _EPOCH:
        build_epoch(V)
    E = _EPOCH
    o = make_oracle(alpha_np, V)
    geo = GeometryData.evaluate(o, E["ret"], E["sf"], face_tables(1, 3),
                                domain="outside",
                                warm_feet=E.get("feet"),
                                max_fail_frac=0.005)
    if "feet" not in E:
        E["feet"] = geo.d.copy()
    tsa = TransientShapeAdjoint(E["dm"], E["sf"], geo, o, NU, DT,
                                ALPHA_F, E["strong"], E["g_strong"])
    xs = tsa.run(n_steps)
    ndof = 4
    series = []
    for x in xs:
        full = np.asarray(tsa.T_vec @ x).reshape(-1, ndof)
        series.append(np.stack([E["W"] @ full[:, c] for c in range(3)],
                               axis=1))
    return tsa, xs, np.array(series)


def main(n_steps=N_STEPS, n_epochs=6):
    o0 = ProvidedINROracle.from_genie_json(BUNNY_JSON, head=0)
    g = np.linspace(0.05, 0.95, 40)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    with torch.no_grad():
        v = o0.psi(torch.tensor(P)).numpy()
    band = P[np.abs(v) < 0.04]
    rng = np.random.default_rng(7)
    band = band[rng.permutation(len(band))[:900]]
    band += rng.uniform(-0.01, 0.01, band.shape)
    V, evals, stab = extract_modes(o0, band, k=K,
                                   check_pts01=band[::-1].copy())
    print(f"[modes] stability {stab:.3f}", flush=True)

    # EAR-DOMINANT edit: mode 0 (Gram eigenvalue 107k, 200x the rest) IS
    # the ear-movement direction this checkpoint was trained for —
    # semantically large geometry, hence an observable wake signature
    # (distributed small-ripple edits measured sub-grid at L4: v2/v3)
    alpha_star = np.array([0.012, -0.002, 0.0, 0.0])
    _, _, target = run_transient(alpha_star, V, n_steps)
    print(f"[setup] hidden alpha* = {alpha_star}", flush=True)

    alpha = np.zeros(K)
    print(f"{'ep':>3} {'J':>12} {'|alpha-alpha*|':>15} {'|adj cos|':>10}",
          flush=True)
    for ep in range(n_epochs):
        tsa, xs, series = run_transient(alpha, V, n_steps)
        r0 = (series - target).reshape(-1)
        J = 0.5 * float(r0 @ r0)
        # adjoint gradient (consistency check): dJ/dx_n from probe resid
        dJdx = []
        E = _EPOCH
        ndof = 4
        for n in range(n_steps):
            resid_n = series[n] - target[n]
            full_bar = np.zeros((E["dm"].n_nodes, ndof))
            for c in range(3):
                full_bar[:, c] = E["W"].T @ resid_n[:, c]
            dJdx.append(np.asarray(tsa.T_vec.T @ full_bar.reshape(-1)))
        tsa.shape_gradient(dJdx)
        g_adj = tsa.oracle.alpha.grad.numpy().copy()
        # FD Jacobian (k transient runs)
        eps_j = 1e-5
        Jac = np.zeros((len(r0), K))
        for k_ in range(K):
            a2 = alpha.copy(); a2[k_] += eps_j
            _, _, s2 = run_transient(a2, V, n_steps)
            Jac[:, k_] = ((s2 - target).reshape(-1) - r0) / eps_j
        g_jac = Jac.T @ r0
        cos = float(g_adj @ g_jac / max(
            np.linalg.norm(g_adj) * np.linalg.norm(g_jac), 1e-30))
        err = np.linalg.norm(alpha - alpha_star)
        print(f"{ep:>3} {J:12.6e} {err:15.6e} {cos:10.5f}", flush=True)
        if J < 1e-14:
            break
        # GUARDED step (H4 lesson: weak signal -> near-singular Jac ->
        # unguarded GN escaped along a null direction to a spurious
        # J=0 at |alpha|~6.6). LM damping scaled to the gradient + a
        # trust-region cap at the epoch's own displacement bound.
        lamb = 1e-3 * np.trace(Jac.T @ Jac) / K
        step = np.linalg.solve(Jac.T @ Jac + lamb * np.eye(K),
                               -(Jac.T @ r0))
        MAX_STEP = 0.004               # displacement units << 0.35 h
        n_ = np.linalg.norm(step)
        if n_ > MAX_STEP:
            step *= MAX_STEP / n_
        alpha = alpha + step
    print(f"[result] recovered {np.round(alpha, 5)} vs {alpha_star}",
          flush=True)
    print(f"[result] recovery err = "
          f"{np.linalg.norm(alpha - alpha_star):.3e}", flush=True)


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else N_STEPS
    ne = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    main(ns, ne)
