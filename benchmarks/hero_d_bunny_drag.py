"""M2-D FINALE (p1, per Baskar 13:45): recover the hidden ear-dominant
GENIE edit of the STANFORD BUNNY from the TRANSIENT DRAG HISTORY —
whole-body integrated observable (the H4-final recipe: L5 + drag +
ear edits). QoI per step: F_n = drag; J = 0.5 sum (F_n - F*_n)^2;
dJ/dx_n = (F_n - F*_n) * w_traction (linear QoI).

J = sum_n sum_probes |u_n(probe) - u*_n(probe)|^2 over an N-step BDF2
start-up flow. Frozen epoch + anchored feet (the H1 lessons); recovery
via Gauss-Newton with FD Jacobian columns (k transient runs per epoch);
the ADJOINT gradient from the reverse sweep is the per-epoch consistency
check (cos printed) — and the scalable path at large k.

Run: python benchmarks/hero_h2_sphere_transient.py [n_steps] [n_epochs]
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "tests")
sys.path.insert(0, "benchmarks")
from hero_h1_sphere_steady import K, U_IN, NU, ALPHA_F
from hero_h3_bunny_steady import make_bunny_oracle as make_oracle
from hero_h3_bunny_steady import BUNNY_JSON
from diffsim.geometry.provided_inr import ProvidedINROracle, extract_modes
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

LEVEL, N_STEPS = 5, 8
DT = 0.05
# bunny wake plane (bbox [0.28,0.79]; H3 diagnosis)
_py, _pz = np.meshgrid(np.linspace(0.35, 0.65, 4),
                       np.linspace(0.35, 0.65, 4))
PROBES = np.column_stack([np.full(16, 0.90), _py.ravel(), _pz.ravel()])

_EPOCH = {}


def build_epoch(V, alpha_np=None):
    o0 = make_oracle(alpha_np if alpha_np is not None
                     else np.zeros(K), V)
    tree = build_uniform(LEVEL, dim=3)
    ret, _ = classify_lambda(tree, o0, 1.0, domain="outside")
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), "cuda:0")
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
    """Forward N-step run; returns (tsa, xs, DRAG series [n])."""
    if not _EPOCH:
        build_epoch(V, alpha_np)
    E = _EPOCH
    o = make_oracle(alpha_np, V)
    geo = GeometryData.evaluate(o, E["ret"], E["sf"], face_tables(1, 3),
                                domain="outside",
                                warm_feet=E.get("feet"),
                                max_fail_frac=0.012)
    # EPOCH HOMOTOPY (M3 rung 2 preview): per-alpha re-carve when the
    # surface drifts past the trust region — cell-scale edits become
    # representable; each alpha gets a deterministic carve + cold
    # anchored feet (objective well-defined per-alpha; cross-epoch
    # J-jumps are part of the landscape, guarded GN absorbs them)
    DRIFT = float('inf')  # M3 continuation owns re-anchoring
    if np.linalg.norm(geo.d, axis=1).max() > DRIFT:
        _EPOCH.clear()
        build_epoch(V, alpha_np)
        E = _EPOCH
        geo = GeometryData.evaluate(o, E["ret"], E["sf"],
                                    face_tables(1, 3),
                                    domain="outside",
                                    max_fail_frac=0.012)
    if "feet" not in E:
        E["feet"] = geo.d.copy()
    _EPOCH["_geo_last"] = geo
    if "Wwake" not in E:
        from diffsim.mesh.pointeval import point_eval_weights
        py, pz = np.meshgrid(np.linspace(0.40, 0.62, 3),
                             np.linspace(0.40, 0.62, 3))
        wpts = np.column_stack([np.full(9, 0.86), py.ravel(), pz.ravel()])
        E["Wwake"] = point_eval_weights(E["mesh"], wpts)
    tsa = TransientShapeAdjoint(E["dm"], E["sf"], geo, o, NU, DT,
                                ALPHA_F, E["strong"], E["g_strong"])
    xs = tsa.run(n_steps)
    from diffsim.sbm.vector import surrogate_traction
    series = []
    for x in xs:
        x_full = np.asarray(tsa.T_vec @ x)
        F = surrogate_traction(E["dm"], E["sf"], geo, x_full, NU, 4)
        ux = E["Wwake"] @ x_full.reshape(-1, 4)[:, 0]
        series.append(np.concatenate([F[:3], ux]))
    return tsa, xs, np.array(series).reshape(-1)


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
    alpha_star = np.array([0.03, -0.005, 0.0, 0.0])
    _, _, target = run_transient(alpha_star, V, n_steps)
    print(f"[setup] hidden alpha* = {alpha_star}", flush=True)

    alpha = np.zeros(K)
    print(f"{'ep':>3} {'J':>12} {'|alpha-alpha*|':>15} {'|adj cos|':>10}",
          flush=True)
    for ep in range(n_epochs):
        tsa, xs, series = run_transient(alpha, V, n_steps)
        r0 = (series - target)
        J = 0.5 * float(r0 @ r0)
        # adjoint gradient: dJ/dx_n = resid_n * w_traction (linear QoI)
        from diffsim.sbm.ns_shape import traction_functional
        E = _EPOCH
        geo_now = tsa.geo if hasattr(tsa, "geo") else None
        w_tr = traction_functional(E["dm"], E["sf"], _EPOCH["_geo_last"],
                                   NU, 4, direction=0)
        dJdx = []
        for n in range(n_steps):
            dJdx.append(np.asarray(tsa.T_vec.T @ (
                (series[n] - target[n]) * w_tr)))
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
