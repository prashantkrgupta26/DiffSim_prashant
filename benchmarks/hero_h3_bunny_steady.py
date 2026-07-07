"""Hero H3 (M2 ladder): recover a hidden GENIE edit of the STANFORD
BUNNY from steady-flow probes — the H1 story on the real two-head
checkpoint (bunny_ear_movement_two_head.json, head 0).

Same recipe as H1-sphere with the full framework stack (frozen epoch,
anchored feet, displacement-normalized Gram modes, adjoint-Picard
gradient, Gauss-Newton with FD Jacobian + adjoint consistency check) but
with the bunny's own near-surface band driving mode extraction, and
cuDSS in the Picard loop.

Run: python benchmarks/hero_h3_bunny_steady.py [level] [n_epochs]
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "tests")
sys.path.insert(0, "benchmarks")
from diffsim.geometry.provided_inr import ProvidedINROracle, extract_modes
from hero_h1_sphere_steady import (steady, alpha_gradient,
                                   probe_values, K)
import hero_h1_sphere_steady as h1

BUNNY_JSON = os.path.join("SDF examples",
                          "bunny_ear_movement_two_head.json")


def make_bunny_oracle(alpha_np, V):
    o = ProvidedINROracle.from_genie_json(BUNNY_JSON, head=0)
    if V is not None:
        o._V = torch.tensor(V[0])
        o._Vscale = torch.tensor(V[1])
        o.alpha = torch.tensor(np.asarray(alpha_np, np.float64))
    return o


def main(level=4, n_epochs=8):
    h1.make_oracle = make_bunny_oracle          # swap the geometry source
    # WAKE probes (measured fix): the bunny bbox is [0.28,0.79]^ish and
    # the sphere-inherited plane x=0.78 sat ON the body's rear edge
    # (no-slip vicinity -> J ~600x weaker than the sphere case, GN
    # plateaued flat). Mode activity spans the whole body, so a
    # downstream wake plane sees every mode through the flow.
    _py, _pz = np.meshgrid(np.linspace(0.35, 0.65, 4),
                           np.linspace(0.35, 0.65, 4))
    h1.PROBES = np.column_stack([np.full(16, 0.90), _py.ravel(),
                                 _pz.ravel()])

    o0 = ProvidedINROracle.from_genie_json(BUNNY_JSON, head=0)
    # near-surface band by rejection sampling (the bunny has no closed
    # form: sample a grid, keep |psi| small, jitter for thickness)
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
    print(f"[modes] bunny head-0 top-{K} evals {np.round(evals, 1)}, "
          f"stability {stab:.3f}", flush=True)
    Vpack = (V, o0._Vscale.numpy())

    alpha_star = np.array([0.008, -0.006, 0.005, -0.006])
    st_star = steady(alpha_star, Vpack, level)
    u_target = probe_values(st_star)
    print(f"[setup] hidden alpha* = {alpha_star} (displacement units)",
          flush=True)

    alpha = np.zeros(K)
    print(f"{'ep':>3} {'J':>13} {'|a-a*|':>13} {'adjcos':>9}", flush=True)
    for ep in range(n_epochs):
        st = steady(alpha, Vpack, level)
        r0 = (probe_values(st) - u_target).reshape(-1)
        J = 0.5 * float(r0 @ r0)
        _, g_adj = alpha_gradient(st, u_target)
        eps_j = 1e-5
        Jac = np.zeros((len(r0), K))
        for k_ in range(K):
            a2 = alpha.copy(); a2[k_] += eps_j
            st2 = steady(a2, Vpack, level)
            Jac[:, k_] = ((probe_values(st2) - u_target).reshape(-1)
                          - r0) / eps_j
        g_jac = Jac.T @ r0
        cos = float(g_adj @ g_jac / max(
            np.linalg.norm(g_adj) * np.linalg.norm(g_jac), 1e-30))
        err = np.linalg.norm(alpha - alpha_star)
        print(f"{ep:>3} {J:13.6e} {err:13.6e} {cos:9.4f}", flush=True)
        if J < 1e-14:
            break
        lamb = 1e-8 * np.trace(Jac.T @ Jac) / K
        alpha = alpha + np.linalg.solve(Jac.T @ Jac + lamb * np.eye(K),
                                        -(Jac.T @ r0))
    print(f"[result] recovered {np.round(alpha, 5)} vs {alpha_star}",
          flush=True)
    print(f"[result] recovery err = "
          f"{np.linalg.norm(alpha - alpha_star):.3e}", flush=True)


if __name__ == "__main__":
    lv = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    ne = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    main(lv, ne)
