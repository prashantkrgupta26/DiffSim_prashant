"""M4 track (e) OPENER: first learned free energy — recover hidden
f_mix parameters from a synthetic evaporating-film trajectory.

TRUTH: one Fig-3-class 1-D film march (Bi = 1, 4 x 128 strip, blend
0.2/0.2/0.6 + 1% IC noise, seed 7) with a PERTURBED f_mix:
chi_pf = 1.35 (base 1.0), chi_ps = 0.45 (base 0.3). DATA = laterally
averaged phi_p AND phi_f profiles (all 129 theta rows) at 6 evenly
spaced film heights h in [0.62, 0.97], linearly interpolated in h
between accepted steps (the dt heuristic makes step-h values
param-dependent; interpolation keeps residuals smooth in the params).

DATA-WINDOW + CONTINUATION (measured, first attempt recorded): the
perturbed truth is spinodal-unstable at t = 0 (FH curvature det < 0
at the blend), separation saturates by h ~= 0.94 while the base
params separate only at h ~= 0.83. With snaps at h in [0.55, 0.90]
every snapshot compares SATURATED layer patterns: a base->truth line
scan measured J = 84 -> 100 -> 51 -> 22 -> 68 -> 73 -> 51 -> 11 ->
2.7 -> 0.4 -> 0 (rugged, smooth basin only within ~30% of truth),
and plain GN from base stalled at J = 61 in a wrong well
(chi_ps/chi_fs -> 1.2-1.3) with a WELL-CONDITIONED Gramian
(cond ~50) — nonconvexity, not rank deficiency: the chi information
lives in the onset/growth phase (same IC noise pattern, different
growth factors => residuals smooth in the params), while saturated
patterns carry rugged layer-phase information. Fix = the house
continuation discipline in h: snap 1 placed in the truth growth
window (h = 0.97) and GN staged over snap prefixes (1, 2, 4, all 6),
warm-started — early snaps pull the params into the basin, the late
saturated snaps then refine sharply.

LEARNABLE SET (documented): the stepper's FH f_mix exposes exactly
3 chi's + k_e as scalar knobs — v1 mobility M0 is frozen at the known
initial composition (mobility degeneracy is track-(e) data-window
work, not this opener), kappa/N are known. A composition-dependent
bump IS supported via the v1.3 Chebyshev delta-f'(phi_p)
(sum_{k=2..4} c_k T_k(2 phi_p - 1)). GAUGE ANCHORING (the stretch's
own measured finding): T0 is a constant in mu — invisible to grad-mu
fluxes; T1 integrates to a QUADRATIC free energy, which the chi terms
already span — (chi_pf, chi_ps, c_1) += t is an EXACT model symmetry
(verified: trajectory difference 5e-15 at t = 0.1). Stretch v1/v2
with T1 learnable descended 4.5 decades in J and then slid along
exactly that gauge direction (recovered-minus-truth = (0.210, 0.209,
0.147) on (chi_pf, chi_ps, c_1) — the two chi offsets EQUAL to 3
digits), failing the 5% gate: STRUCTURAL unidentifiability no data
window can fix. Beyond-FH content starts at cubic f, so the
learnable basis is T2..T4. So:
  default  : 4 params  (chi_pf, chi_ps, chi_fs, k_e)
  --stretch: 7 params  (+ c2, c3, c4 Chebyshev truth perturbation)
             — the first FUNCTIONAL (beyond-FH) correction.

RECOVERY: Gauss-Newton from the BASE parameters, forward-FD Jacobian
columns (n_p marches / iterate), absolute Levenberg damping with a
lambda floor anchoring non-final stages (see gauss_newton docstring)
plus geometric extension of accepted steps (valley acceleration),
signal-scale check J0 >> 1e-8 before burning compute, Jacobian SVD +
Gramian spectrum reported EVERY iterate (house identifiability
discipline). GATE: every parameter within 5% of truth.

RESULTS (2026-07-08, RTX 6000 Ada, tests/baselines/m4_learned_fmix
.json): FH-4 gate PASS — recovery EXACT (J 52.7 -> 1.5e-12, every
parameter to 7+ digits; 112 marches, 147 s). CHEB-7 STRETCH VERDICT:
the FH block still recovers within 5% (chi_pf 2.6%, chi_ps 2.9%,
chi_fs 2.8%, k_e 4.2%) but the functional coefficients do NOT
(J stalls at 0.23 vs J0 = 43, truth at J = 0); three independent
protocols (staged, staged + acceleration, FH-first ladder restart
with cheb re-anchored at 0) landed in three DISTINCT genuine local
minima (J = 0.195 / 0.228 / 0.051, the last re-verified with
FD_STEP = 1e-3: gradient zero, not an FD artifact). Mechanism:
(chi_pf, chi_ps) += t injects exactly a T1 term into mu_1, and over
the narrow composition range the film visits, T2..T4 combinations
approximate it (smallest Gramian eig ~1e-5 of the top) — the
profile-misfit landscape in the functional subspace is
near-degenerate AND multi-basin. The beyond-FH functional correction
needs richer observables (instrument-space S(q,t), h(t)-resolved
losses) or broader composition coverage — the track-(e) next rung.
Identifiability, not mechanism.

Run:  python benchmarks/m4_learn_fmix.py [--stretch] [--iters N]
"""

import os as _bos, sys as _bsys  # noqa: E402  (benchmark import bootstrap)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import _bench_bootstrap  # noqa: E402,F401
import argparse
import json
import time

import numpy as np
import warp as wp

from diffsim import default_device
from diffsim.octree.build import build_uniform, Octree
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.wodo_film import WodoFilmStepper

PHI_P0, PHI_F0 = 0.2, 0.2
NCHAIN = (5.0, 5.0, 1.0)
KAPPA = (2e-4, 2e-4)
IC_SEED = 7                       # SAME IC for truth and every candidate
H_SNAP = np.linspace(0.97, 0.62, 6)
STAGES = (1, 2, 4, 6)             # h-continuation: snap-prefix ladder

NAMES4 = ("chi_pf", "chi_ps", "chi_fs", "k_e")
BASE4 = np.array([1.0, 0.30, 0.30, 1.0])
TRUTH4 = np.array([1.35, 0.45, 0.30, 1.0])
NAMES7 = NAMES4 + ("cheb_c2", "cheb_c3", "cheb_c4")
BASE7 = np.concatenate([BASE4, [0.0, 0.0, 0.0]])
TRUTH7 = np.concatenate([TRUTH4, [-0.06, 0.04, 0.03]])
FD_STEP = 1e-2                    # absolute; params are O(0.04-1.4)


def build_strip(level=7, nx=4):
    tree0 = build_uniform(level, dim=2)
    keep = tree0.centers()[:, 0] < nx * 2.0 ** (-level)
    tree = Octree(tree0.keys[keep], tree0.levels[keep], dim=2,
                  periodic=tree0.periodic)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    return mesh, cons


def m0_frozen():
    """v1 constant-mobility mapping at the KNOWN initial composition
    (chi-independent, so frozen across all candidate runs)."""
    phis0 = 1.0 - PHI_P0 - PHI_F0
    D0 = phis0 * 1.0 + (1.0 - phis0) * 1e-3
    fpp = 1.0 / (NCHAIN[0] * PHI_P0) + 1.0 / (NCHAIN[2] * phis0)
    return D0 / fpp


class ProfileRecorder:
    """Laterally averaged phi_p/phi_f row profiles, linearly
    interpolated in h at the (decreasing) snapshot heights."""

    def __init__(self, st, mesh, h_snap):
        self.st = st
        yv = mesh.node_coords[:, 1]
        _, self.rowinv = np.unique(np.round(yv, 12), return_inverse=True)
        self.rowcnt = np.bincount(self.rowinv)
        self.h_snap = list(h_snap)
        p1n, p2n = st.hist[0]
        self.prev = (st.h_curr, self._rows(p1n), self._rows(p2n))
        self.snaps = []                       # (h, rows_p, rows_f)

    def _rows(self, free_vec):
        full = np.asarray(self.st.Tc @ free_vec)
        return np.bincount(self.rowinv, weights=full) / self.rowcnt

    def __call__(self, st, K, dt, iters):
        h = st.h_curr
        if not self.h_snap or h > self.h_snap[0]:
            if self.h_snap:
                self.prev = (h, self._rows(st.x[0::4]),
                             self._rows(st.x[2::4]))
            return
        r1 = self._rows(st.x[0::4])
        r2 = self._rows(st.x[2::4])
        hp, p1p, p2p = self.prev
        while self.h_snap and h <= self.h_snap[0]:
            ht = self.h_snap.pop(0)
            w = 1.0 if hp <= h else (hp - ht) / (hp - h)
            self.snaps.append((ht, (1 - w) * p1p + w * r1,
                               (1 - w) * p2p + w * r2))
        self.prev = (h, r1, r2)


def forward_run(params, dm, mesh):
    """One film march at candidate params -> stacked profile vector
    (6 snaps x [phi_p rows, phi_f rows]). None on solver failure."""
    chi = tuple(float(c) for c in params[:3])
    k_e = float(params[3])
    f_cheb = (tuple(float(c) for c in params[4:7])
              if len(params) > 4 else (0.0, 0.0, 0.0))
    M0 = m0_frozen()
    st = WodoFilmStepper(dm, chi=chi, N=NCHAIN, M=(M0, 0.0, M0),
                         kappa=KAPPA, k_e=k_e, dt=1e-4, f_cheb=f_cheb,
                         use_device_assembly=True)
    rng = np.random.default_rng(IC_SEED)
    st.set_initial(
        lambda x: PHI_P0 + 0.01 * rng.standard_normal(len(x)),
        lambda x: PHI_F0 + 0.01 * rng.standard_normal(len(x)))
    rec = ProfileRecorder(st, mesh, H_SNAP)
    reason = st.march(h_min=H_SNAP[-1] - 0.02, phis_stop=0.01,
                      max_steps=5000, callback=rec)
    if len(rec.snaps) != len(H_SNAP) or reason == "dt_underflow":
        return None
    return np.concatenate([np.concatenate([s[1], s[2]])
                           for s in rec.snaps])


def gauss_newton(data, p0, names, dm, mesh, n_active, max_iter=12,
                 tol_step=1e-5, tag="", lam_floor=1e-9):
    """LM-damped GN with forward-FD Jacobian on the first n_active
    observations (h-continuation snap prefix); SVD + Gramian spectrum
    reported every iterate (identifiability discipline).

    Damping is ABSOLUTE Levenberg (lam * gmax * I, gmax = max diag G),
    not Marquardt diag-scaling: weakly determined columns must be
    damped HARDER, not proportionally to their own small norms.
    lam_floor ANCHORS non-final continuation stages: directions with
    Gramian eigenvalue below ~lam_floor*gmax cannot step, so the
    degenerate manifold of an early data window (measured, 7-param
    stretch v1: stage-1 Gramian split 3 strong / 4 weak eigs over 7
    decades and diag-scaled LM ran cheb_c1 to -0.15 against truth
    +0.10 while fitting snap 1 to J = 2e-10 — a wrong point on a real
    degenerate manifold that stranded the final stage at J ~ 2e-3)
    stays parked at the warm start until later snaps constrain it."""
    p = p0.copy()
    y = forward_run(p, dm, mesh)
    assert y is not None, "initial march failed"
    r = (y - data)[:n_active]
    cost = 0.5 * float(r @ r)
    lam = max(1e-3, lam_floor)
    n_march = 1
    sv_last = None
    print(f"[gn{tag}] it  0: J={cost:.6e}  " +
          "  ".join(f"{n}={v:.4f}" for n, v in zip(names, p)),
          flush=True)
    for it in range(1, max_iter + 1):
        Jac = np.empty((n_active, len(p)))
        for i in range(len(p)):
            pp = p.copy()
            pp[i] += FD_STEP
            yi = forward_run(pp, dm, mesh)
            n_march += 1
            assert yi is not None, f"FD march failed on {names[i]}"
            Jac[:, i] = ((yi - data)[:n_active] - r) / FD_STEP
        sv = np.linalg.svd(Jac, compute_uv=False)
        sv_last = sv
        G = Jac.T @ Jac
        gev = np.linalg.eigvalsh(G)[::-1]
        print(f"[gn{tag}]   J sv: "
              f"[{', '.join(f'{s:.3e}' for s in sv)}]  "
              f"cond={sv[0] / sv[-1]:.2e}  Gramian eig: "
              f"[{', '.join(f'{e:.2e}' for e in gev)}]", flush=True)
        g = Jac.T @ r
        gmax = float(G.diagonal().max())
        accepted = False
        for _ in range(10):
            dp = np.linalg.solve(G + lam * gmax * np.eye(len(p)), -g)
            p_try = p + dp
            y_try = (forward_run(p_try, dm, mesh)
                     if np.isfinite(p_try).all() and p_try[3] > 0.05
                     else None)
            n_march += 1
            c_try = (0.5 * float((((y_try - data)[:n_active]) ** 2).sum())
                     if y_try is not None else np.inf)
            if c_try < cost:
                p_pre = p
                p, cost = p_try, c_try
                r = (y_try - data)[:n_active]
                lam = max(lam / 3.0, lam_floor)
                accepted = True
                # geometric step extension (valley acceleration): a
                # near-degenerate direction (measured, 7-param stretch:
                # T2..T4 alias the quadratic-f gauge over the RESTRICTED
                # composition range the film visits; smallest Gramian
                # eig ~1e-5 of the top) makes plain LM crawl along the
                # valley at ~0.01/iterate; extending the ACCEPTED step
                # while the cost keeps dropping covers it at 1 march
                # per doubling.
                k = 2.0
                while k <= 64.0:
                    p_ext = p_pre + k * dp
                    y_ext = (forward_run(p_ext, dm, mesh)
                             if p_ext[3] > 0.05 else None)
                    n_march += 1
                    if y_ext is None:
                        break
                    c_ext = 0.5 * float(
                        (((y_ext - data)[:n_active]) ** 2).sum())
                    if c_ext >= cost:
                        break
                    p, cost = p_ext, c_ext
                    r = (y_ext - data)[:n_active]
                    k *= 2.0
                break
            lam *= 10.0
        print(f"[gn{tag}] it {it:2d}: J={cost:.6e}  lam={lam:.1e}  " +
              "  ".join(f"{n}={v:.4f}" for n, v in zip(names, p)),
              flush=True)
        if not accepted:
            print(f"[gn{tag}] LM could not find a descent step; "
                  f"stopping stage.", flush=True)
            break
        if np.abs(dp).max() < tol_step:
            print(f"[gn{tag}] step below tol; stage converged.",
                  flush=True)
            break
    return p, cost, sv_last, n_march


def staged_recovery(data, p0, names, dm, mesh, max_iter, nrow):
    """h-continuation: GN over growing snap prefixes (STAGES),
    warm-started — the early growth-phase snaps pull the params into
    the truth basin, the saturated snaps refine (module docstring)."""
    p = p0.copy()
    n_march = 0
    sv = None
    cost = None
    for ks, k in enumerate(STAGES):
        n_active = k * 2 * nrow
        iters = max_iter if k == STAGES[-1] else 4
        print(f"\n[stage {ks + 1}/{len(STAGES)}] snaps 1..{k} "
              f"(h >= {H_SNAP[k - 1]:.2f}, {n_active} obs)", flush=True)
        p, cost, sv, nm = gauss_newton(
            data, p, names, dm, mesh, n_active, max_iter=iters,
            tag=f" s{ks + 1}",
            lam_floor=1e-9 if k == STAGES[-1] else 1e-2)
        n_march += nm
    return p, cost, sv, n_march


def recover(tag, truth, base, names, dm, mesh, max_iter):
    print(f"\n===== {tag}: {len(names)} params =====", flush=True)
    t0 = time.time()
    data = forward_run(truth, dm, mesh)
    assert data is not None, "truth march failed"
    print(f"[truth] {len(data)} observations "
          f"({len(H_SNAP)} snaps x 2 fields x {len(data) // (2 * len(H_SNAP))}"
          f" rows), march {time.time() - t0:.1f}s", flush=True)
    y0 = forward_run(base, dm, mesh)
    J0 = 0.5 * float((y0 - data) @ (y0 - data))
    nrow = len(data) // (2 * len(H_SNAP))
    J0_s1 = 0.5 * float(((y0 - data)[:2 * nrow] ** 2).sum())
    print(f"[signal] J0 = {J0:.6e} (stage-1 window {J0_s1:.3e}; "
          f"floor ~1e-8) -> "
          f"{'OK' if min(J0, J0_s1) > 1e-8 else 'TOO WEAK'}", flush=True)
    assert J0_s1 > 1e-8, "signal too weak to optimize"
    p, cost, sv, n_march = staged_recovery(data, base, names, dm, mesh,
                                           max_iter, nrow)
    wall = time.time() - t0
    print(f"\n--- {tag} recovery table ({n_march + 2} marches, "
          f"{wall:.0f}s wall) ---", flush=True)
    print(f"{'param':>8s} {'truth':>9s} {'base':>9s} {'recovered':>11s} "
          f"{'abs err':>10s} {'rel err':>8s}  gate(5%)")
    ok_all = True
    rows = {}
    for i, n in enumerate(names):
        ae = abs(p[i] - truth[i])
        re = ae / max(abs(truth[i]), 1e-12)
        ok = re < 0.05
        ok_all &= ok
        rows[n] = {"truth": float(truth[i]), "base": float(base[i]),
                   "recovered": float(p[i]), "abs_err": float(ae),
                   "rel_err": float(re)}
        print(f"{n:>8s} {truth[i]:9.4f} {base[i]:9.4f} {p[i]:11.6f} "
              f"{ae:10.2e} {100 * re:7.2f}%  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"[gate] all params within 5%: "
          f"{'PASS' if ok_all else 'FAIL'}   J: {J0:.3e} -> {cost:.3e}",
          flush=True)
    return {"params": rows, "J0": J0, "J_final": cost,
            "gate_5pct": bool(ok_all), "n_marches": n_march + 2,
            "wall_s": round(wall, 1),
            "final_sv": ([float(s) for s in sv]
                         if sv is not None else None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stretch", action="store_true",
                    help="7-param: + 3-coeff Chebyshev delta-f'(phi_p)")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--level", type=int, default=7)
    ap.add_argument("--json", type=str, default=None,
                    help="write results JSON here")
    args = ap.parse_args()
    wp.init()
    device = default_device()
    mesh, cons = build_strip(level=args.level)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    print(f"strip: {len(mesh.tree)} elements, {len(mesh.node_coords)} "
          f"nodes, device={device}, M0={m0_frozen():.4f}, Bi=1, "
          f"h_snap={np.round(H_SNAP, 3).tolist()}")
    out = {}
    out["fh4"] = recover("FH-4", TRUTH4, BASE4, NAMES4, dm, mesh,
                         args.iters)
    if args.stretch:
        out["cheb7"] = recover("CHEB-7", TRUTH7, BASE7, NAMES7, dm,
                               mesh, args.iters)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[json] wrote {args.json}")


if __name__ == "__main__":
    main()
