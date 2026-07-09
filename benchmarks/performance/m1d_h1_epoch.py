"""M1d closure item 3: time ONE Gauss-Newton epoch of the H1 hero
config (hero_h1_sphere_steady.py, L4: one steady + adjoint gradient +
4 forward-FD steadies) on the CURRENT stack, vs the M1c-era baseline.

Baseline (documented): ~20 min/epoch with host splu in the Picard loops
at the time of the m1c hero runs — commit b0a7b73 ("perf(m1d): measured
per-stage profile"): "the hero scripts' Picard loops use splu, so their
20-min epochs are a ONE-LINE cudss switch away from ~1-2 min"
(splu_factor 686 s at 3-D L5 vs cuDSS 2.26 s = 300x was the driver).

Run: python benchmarks/m1d_h1_epoch.py [level]
"""

import os as _bos, sys as _bsys  # noqa: E402  (benchmark import bootstrap)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import _bench_bootstrap  # noqa: E402,F401
import sys
import time

import numpy as np

sys.path.insert(0, "tests")
sys.path.insert(0, "benchmarks")

import hero_h1_sphere_steady as h1

BASELINE_S = 20 * 60.0          # the documented M1c-era epoch cost


def main(level=4):
    o0 = h1.ProvidedINROracle.from_state_dict(
        h1.SPHERE_PT, n_layers=7, w0=1.0, window_half=0.5,
        window_center=0.03)
    rng = np.random.default_rng(11)
    band_dirs = rng.standard_normal((800, 3))
    band_dirs /= np.linalg.norm(band_dirs, axis=1, keepdims=True)
    band = 0.47 + (0.262 + rng.uniform(-0.05, 0.05, 800))[:, None] \
        * band_dirs
    V, evals, stab = h1.extract_modes(o0, band, k=h1.K,
                                      check_pts01=band[::-1].copy())

    # target probes; ALSO warms the frozen epoch + anchored feet +
    # kernel compiles (one-time, excluded — per-epoch cost is the metric)
    alpha_star = np.array([0.015, -0.010, 0.008, -0.012])
    t0 = time.perf_counter()
    st_star = h1.steady(alpha_star, V, level)
    u_target = h1.probe_values(st_star)
    print(f"setup (epoch build + target steady): "
          f"{time.perf_counter() - t0:.1f} s")

    # ---- ONE GN epoch, timed (mirrors hero main()'s epoch body) ------
    alpha = np.zeros(h1.K)
    t0 = time.perf_counter()
    st = h1.steady(alpha, V, level)
    t_steady = time.perf_counter() - t0
    r0 = (h1.probe_values(st) - u_target).reshape(-1)
    J = 0.5 * float(r0 @ r0)
    t1 = time.perf_counter()
    _, g_adj = h1.alpha_gradient(st, u_target)
    t_adj = time.perf_counter() - t1
    eps_j = 1e-5
    Jac = np.zeros((len(r0), h1.K))
    t1 = time.perf_counter()
    for k_ in range(h1.K):
        ap = alpha.copy()
        ap[k_] += eps_j
        rp = (h1.probe_values(h1.steady(ap, V, level))
              - u_target).reshape(-1)
        Jac[:, k_] = (rp - r0) / eps_j
    t_fd = time.perf_counter() - t1
    lamb = 1e-8 * np.trace(Jac.T @ Jac) / h1.K
    step = np.linalg.solve(Jac.T @ Jac + lamb * np.eye(h1.K),
                           -(Jac.T @ r0))
    t_epoch = time.perf_counter() - t0

    g_jac = Jac.T @ r0
    cos = float(g_adj @ g_jac / max(
        np.linalg.norm(g_adj) * np.linalg.norm(g_jac), 1e-30))
    print(f"epoch sanity: J={J:.4e}  |GN step|={np.linalg.norm(step):.3e}"
          f"  adj-vs-Jac cos={cos:.6f}")
    print(f"\nONE H1 GN EPOCH (L{level}, current stack: cuDSS Picard + "
          f"adjoint):")
    print(f"  steady solve      {t_steady:7.1f} s")
    print(f"  adjoint gradient  {t_adj:7.1f} s")
    print(f"  4 FD steadies     {t_fd:7.1f} s")
    print(f"  TOTAL             {t_epoch:7.1f} s "
          f"({t_epoch/60:.2f} min)")
    print(f"\nM1c-era baseline (host splu, commit b0a7b73): "
          f"{BASELINE_S:.0f} s (~20 min)")
    print(f"H1-EPOCH SPEEDUP: {BASELINE_S/t_epoch:.1f}x  "
          f"(D3 exit bar: >= 10x -> "
          f"{'PASS' if BASELINE_S/t_epoch >= 10 else 'FAIL'})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
