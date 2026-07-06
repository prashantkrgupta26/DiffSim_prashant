"""E2 — Differentiable SBM Poisson with a GENIE INR: edit recovery from
sparse measurements.

LEARNING OUTCOME
    Connect a trained INR (your GENIE checkpoint) to the differentiable
    Octree-SBM pipeline: load it as a geometry oracle, extract the
    Gram-eigenmode design space, solve a 3-D SBM Poisson problem on the
    implicit geometry, differentiate a probe QoI w.r.t. the mode
    coefficients through the FEM adjoint, and recover a hidden last-layer
    edit from probe data alone — the full "differentiable simulation
    meets INR editing" loop, at Poisson cost (seconds per solve).

BACKGROUND
    GENIE (Karki et al.): for linear-last-layer INRs, edits live in
    Delta-psi = h(x)^T Delta-theta; the thick-band Gram operator's
    eigenmodes span the realizable, well-posed edits. DiffSim treats the
    provided INR as an SDFOracle (psi -> classify; Newton+IFT -> distance
    vectors d, n) and the mode coefficients alpha as design variables:
    within a frozen classification epoch, dJ/dalpha flows through the
    SBM face terms' taped residual into psi's autograd graph — exactly
    the m1a shape-gradient machinery, with YOUR network as the geometry.
    Three measured INR facts are baked into the framework (see
    docs/superpowers/specs/2026-07-06-neuralsdf.md): the computational
    WINDOW contract (pick the sub-box so the feature spans >= ~8 cells),
    domain-BOUNDED projections (sine INRs have spurious far zero sheets),
    and the BRANCH-STABILITY rule for FD checks (bistable closest points
    make J genuinely non-smooth at isolated knife-edge points).

    Pipeline (each step is one function below):
      1. load INR -> ProvidedINROracle (explicit per-model conventions!)
      2. extract_modes: thick-band Gram eigenmodes -> alpha design space
      3. carve ONCE (frozen epoch), solve SBM Poisson at any alpha
      4. adjoint: dJ/dalpha for a probe QoI (+ FD cross-check)
      5. Gauss-Newton recovery of a hidden alpha* from probes

EXPECTED RESULTS (measured, sphere checkpoint, L5, k=4 modes)
    mode stability = 1.000 (thick-band sampling; thin bands degrade it)
    adjoint-vs-FD on dJ/dalpha: rel ~ 1e-2 at L5 (the objective is
      LOCALLY very steep — see below; at L4 with gentler structure the
      same check measures 1e-6-1e-8: the pipeline itself is exact)
    objective consistency: J(alpha*) = O(1e-30) (the sentinel)
    recovery: J drops ~1000x but PLATEAUS at a local minimum near
      alpha=0 — an honest, measured limitation: this objective has steep
      local structure at scales BELOW |alpha*| (sensitivity at 0 is
      ~100x the mean slope to the target), so descent methods stall.
      Understanding and taming this landscape (probe choice? penalty
      interplay with the 7% surface wobble? multiscale continuation?) is
      an OPEN RESEARCH QUESTION this chapter hands to you — see
      EXPLORE 6-7. The NS version of the same story (hero demo H1)
      converges: advective transport enriches the measurement.

Run:  python tutorials/E_differentiable/E2_genie_diffsbm.py
      (expects SDF examples/model_single_head0.pt; swap in your own
      checkpoint via ProvidedINROracle.from_genie_json)
"""
import os
import sys

import numpy as np
import torch
from scipy.sparse.linalg import splu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.provided_inr import ProvidedINROracle, extract_modes
from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                   GeometryData)
from diffsim.sbm.poisson import SBMPoisson
from diffsim.sbm.adjoint import solve_adjoint, shape_gradient, probe_qoi

SPHERE_PT = os.path.join(os.path.dirname(__file__), "..", "..",
                         "SDF examples", "model_single_head0.pt")
K, LEVEL = 4, 5
C = 0.47                          # window_center=0.03 -> sphere center
# probes on a NEAR SHELL (r ~ 0.36 about the sphere at C=0.47): Poisson
# information decays with distance — corner probes measured J0 ~ 4e-4
# (weak observability, wild Gauss-Newton); the near shell fixes it.
_g = np.array([-1.0, 0.0, 1.0])
_D = np.array([[x, y, z] for x in _g for y in _g for z in _g
               if (x, y, z) != (0, 0, 0)])
_D = _D / np.linalg.norm(_D, axis=1, keepdims=True)
PROBES = 0.47 + 0.42 * _D  # 26-pt shell OUTSIDE the wobble+edit
#   envelope (surface reaches r~0.31 in wobble directions; r=0.36 probes
#   landed in alpha*-dropped cells — measured)
U_BC = lambda x: x[:, 0] + 2 * x[:, 1] - x[:, 2]     # outer Dirichlet


def load_oracle(V=None, alpha=None):
    """Step 1: the checkpoint, with EXPLICIT conventions (w0=1.0 was
    determined numerically for this .pt — the admissibility gate guards
    it) and the window contract (half=0.5: sphere spans ~8 cells at L4)."""
    o = ProvidedINROracle.from_state_dict(SPHERE_PT, n_layers=7, w0=1.0,
                                          window_half=0.5,
                                          window_center=0.03)
    if V is not None:
        o._V = torch.tensor(V[0])
        o._Vscale = torch.tensor(V[1])
        o.alpha = torch.zeros(K, dtype=torch.float64)
        if alpha is not None:
            with torch.no_grad():
                o.alpha += torch.tensor(alpha)
    return o


def get_modes(o):
    """Step 2: thick-band Gram eigenmodes (GENIE computeTopKBasis with
    the paper's thick-band sampling rule) + reproducibility check."""
    rng = np.random.default_rng(11)
    d = rng.standard_normal((800, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    band = C + (0.262 + rng.uniform(-0.05, 0.05, 800))[:, None] * d
    V, evals, stab = extract_modes(o, band, k=K,
                                   check_pts01=band[::-1].copy())
    print(f"[modes] top-{K} eigenvalues {np.round(evals, 2)}, "
          f"subspace stability {stab:.3f}")
    return (V, o._Vscale.numpy())


_EPOCH = {}
H_MIN = 1.0 / 2 ** LEVEL
DRIFT = 0.35 * H_MIN     # re-carve when the surface drifts this far from
#                          the frozen surrogate (epoch trust region). The
#                          line-probe measurement that motivated this: a
#                          J = 183 barrier at mid-path alphas whose shapes
#                          sat too far from the alpha=0 surrogate — frozen
#                          epochs are smooth but walled; re-carving on
#                          drift gives smooth-within, valid-across.


def build_epoch(V, alpha):
    o0 = load_oracle(V, alpha)
    tree = build_uniform(LEVEL, dim=3)
    ret, _ = classify_lambda(tree, o0, 1.0, domain="outside")
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3),
                              "cuda:0")
    _EPOCH.clear()
    _EPOCH.update(ret=ret, sf=sf, mesh=mesh, cons=cons, dm=dm)


def solve_poisson(V, alpha):
    """Step 3: EPOCH-MANAGED solve — carve lazily, re-carve when the
    edit drifts the surface beyond DRIFT of the frozen surrogate."""
    if not _EPOCH:
        build_epoch(V, alpha)
    E = _EPOCH
    o = load_oracle(V, alpha)
    geo = GeometryData.evaluate(o, E["ret"], E["sf"], face_tables(1, 3),
                                domain="outside",
                                warm_feet=E.get("feet"))
    if "feet" not in E:
        # ANCHOR feet ONCE per epoch: every alpha warm-starts from the
        # SAME feet -> deterministic objective (a cache updated per-call
        # made J depend on evaluation ORDER — measured as an inconsistent
        # target) + continuous branch selection.
        E["feet"] = geo.d.copy()
    if np.linalg.norm(geo.d, axis=1).max() > DRIFT:
        build_epoch(V, alpha)                     # new epoch at THIS alpha
        E = _EPOCH
        geo = GeometryData.evaluate(o, E["ret"], E["sf"],
                                    face_tables(1, 3), domain="outside")
    prob = SBMPoisson(E["dm"], geo, E["sf"], g_fn=U_BC, kappa=1.0)
    A, b, meta = prob.assemble(lambda x: np.ones(len(x)), g_outer_fn=U_BC)
    u_free = splu(A.tocsc()).solve(b)
    u_all = np.asarray(E["cons"].T @ u_free)
    evalJ, dJdu_fn = probe_qoi(E["dm"], PROBES, np.zeros(len(PROBES)))
    return dict(A=A, meta=meta, u_all=u_all, prob=prob, oracle=o,
                probes=np.array([u_all @ w for w in
                                 _probe_rows(E, len(PROBES))]),
                evalJ=evalJ, dJdu_fn=dJdu_fn)


def _probe_rows(E, npr):
    from diffsim.mesh.pointeval import point_eval_weights
    W = point_eval_weights(E["mesh"], PROBES)
    return [np.asarray(W[i].todense()).ravel() for i in range(npr)]


def alpha_gradient(st, target):
    """Step 4: adjoint dJ/dalpha, J = 0.5 sum (u(probe) - target)^2."""
    resid = st["probes"] - target
    rows = _probe_rows(_EPOCH, len(PROBES))
    dJdu_all = sum(r_ * w for r_, w in zip(resid, rows))
    dJdu = np.asarray(_EPOCH["cons"].T.T @ dJdu_all)
    lam = solve_adjoint(st["A"], dJdu)
    shape_gradient(st["prob"], st["u_all"], lam, st["oracle"], st["meta"])
    return (0.5 * float(resid @ resid),
            st["oracle"].alpha.grad.numpy().copy())


def main():
    o0 = load_oracle()
    V = get_modes(o0)

    # hidden edit + measurements
    alpha_star = np.array([0.004, -0.0035, 0.0025, -0.003])
    # alpha in SURFACE-DISPLACEMENT units (normalized modes). RULE
    # (measured the hard way): |alpha*| must fit INSIDE one epoch's trust
    # region DRIFT = 0.35 h — here |alpha*| ~ 0.0066 vs DRIFT(L5) = 0.011.
    # At L5 an earlier 0.016-edit made the TARGET re-carve into its own
    # epoch while trials stayed in the alpha=0 epoch: the J-floor was the
    # carve-difference signature, unrecoverable by ANY optimizer. Edits
    # larger than the trust region need epoch-homotopy (EXPLORE 6).
    target = solve_poisson(V, alpha_star)["probes"]
    print(f"[setup] hidden alpha* = {alpha_star}")

    # gradient check (FD, branch-stability implicitly fine at Poisson/L4)
    st0 = solve_poisson(V, np.zeros(K))
    J0, g0 = alpha_gradient(st0, target)
    eps = 1e-6
    ap = np.zeros(K); ap[0] = eps
    am = np.zeros(K); am[0] = -eps
    Jp = 0.5 * float(((solve_poisson(V, ap)["probes"] - target) ** 2).sum())
    Jm = 0.5 * float(((solve_poisson(V, am)["probes"] - target) ** 2).sum())
    fd = (Jp - Jm) / (2 * eps)
    print(f"[check] dJ/dalpha_0: adjoint {g0[0]:+.6e} vs FD {fd:+.6e} "
          f"(rel {abs(g0[0]-fd)/max(abs(fd), 1e-14):.1e})")

    # Gauss-Newton recovery
    alpha = np.zeros(K)
    for ep in range(8):
        st = solve_poisson(V, alpha)
        r0 = st["probes"] - target
        J = 0.5 * float(r0 @ r0)
        print(f"[recover] ep {ep}: J = {J:.3e}, "
              f"|alpha - alpha*| = {np.linalg.norm(alpha-alpha_star):.3e}")
        if J < 1e-16:
            break
        Jac = np.zeros((len(r0), K))
        for k in range(K):
            a2 = alpha.copy(); a2[k] += 1e-4
            Jac[:, k] = (solve_poisson(V, a2)["probes"] - target
                         - r0) / 1e-4
        # Levenberg-Marquardt: adaptive damping, step accepted only if J
        # decreases (undamped GN measured to overshoot on weak signals)
        if ep == 0:
            lm = 1e-3 * np.trace(Jac.T @ Jac) / K
        for _try in range(8):
            step = np.linalg.solve(Jac.T @ Jac + lm * np.eye(K),
                                   -(Jac.T @ r0))
            r_new = (solve_poisson(V, alpha + step)["probes"] - target)
            if 0.5 * float(r_new @ r_new) < J:
                alpha = alpha + step
                lm = max(lm * 0.3, 1e-12)
                break
            lm *= 4.0
    r_star = solve_poisson(V, alpha_star)["probes"] - target
    print(f"[sanity] J(alpha*) = {0.5*float(r_star@r_star):.3e} "
          f"(must be ~0: objective consistency)")
    print(f"[result] recovered {np.round(alpha, 5)} "
          f"vs hidden {alpha_star}")


if __name__ == "__main__":
    main()
    print("""
EXPLORE
 1. Thin the mode-extraction band (0.262 +- 0.005 instead of +- 0.05).
    How does the subspace stability change? Reproduce the GENIE paper's
    thick-band finding on YOUR checkpoint.
 2. Reduce the probes to 2. When does recovery become ill-posed, and how
    does the Gauss-Newton Jacobian's condition number warn you first?
 3. Swap in your own two-head bunny json (from_genie_json, head=0) with
    an appropriate window. What does the admissibility report say, and
    at which level does the carve resolve the ears (band-study rule)?
 4. Break it on purpose: set w0=30 for the sphere checkpoint. Which gate
    catches it, and what does the failure look like?
 5. alpha_star outside the mode span: add a perturbation orthogonal to
    V's span to the target's last layer. What does recovery converge to,
    and why is that exactly GENIE's well-posedness theorem in action?
 6. THE OPEN QUESTION: map J along random 1-D alpha rays at resolutions
    1e-6..1e-2. Where does the steep local structure live, and does its
    scale track h, the surface wobble, or the Nitsche penalty weight?
 7. Try recovering with a multiscale strategy: first optimize on probe
    data SMOOTHED over several nearby evaluation points (or with a
    coarser L4 forward), then refine at L5. Does continuation over
    smoothness beat the local minimum that stops plain Gauss-Newton?
""")
