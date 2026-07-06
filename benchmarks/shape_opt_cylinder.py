"""M1c demo: drag-driven shape optimization of the immersed cylinder at
Re=20 (confined channel) — gradient descent on the center using the
adjoint-Picard shape gradient (gate: rel 1.25e-4 vs FD).

Each step is a full epoch: reclassify -> steady Picard (sigma=0) ->
adjoint-Picard gradient -> move center. The frozen-classification
assumption holds WITHIN a step; the epoch boundary is the step.

Run: python benchmarks/shape_opt_cylinder.py [n_steps] [lr]
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "tests")
from test_ns_shape_gradient import _steady, NU, ALPHA, R, U_IN  # noqa: E402
from diffsim.sbm.ns_shape import drag_shape_gradient  # noqa: E402


def main(n_steps=12, lr=0.02, level=5):
    """MEASURED (L5, 15 epochs): per-epoch gradients are exact
    (gate rel 1.25e-4) but Cd carries O(10%) reclassification noise —
    at L5 the cylinder spans only ~2.2 cells, so the surrogate staircase
    reshapes as the center crosses cell boundaries (the band-study
    preasymptotic rule, in optimization form).

    L6 MEASURED (15 epochs): within-epoch noise is much reduced, but the
    honest headline is sharper — the within-epoch gradient (FD-gated to 4
    digits) is NOT descent-stable across epochs: Cd RISES 2.63->3.74 along
    the descent path because reclassification reshapes the surrogate
    discontinuously between epochs, and those jumps dominate the smooth
    local decrease. CONCLUSION (recorded for M1c): naive epoch-wise SBM
    shape descent needs either feature-resolving resolution (band-study
    rule), objective smoothing, or the differentiable-classification path
    (NeuralSDF) — this demo is the measured motivation for the latter."""
    c = np.array([0.30, 0.42])          # start off-center: does it center?
    print(f"{'step':>4} {'c_x':>8} {'c_y':>8} {'Cd':>8} "
          f"{'dCd/dcx':>10} {'dCd/dcy':>10}")
    for it in range(n_steps):
        st = _steady(c, "cuda:0", level=level)
        F = drag_shape_gradient(st["dm"], st["sf"], st["geo"], st["oracle"],
                                st["A"], st["x_full"], NU, ALPHA,
                                st["strong_rows"], ndof=3, direction=0,
                                fixed_point=True)
        g = st["oracle"].params[0].grad.detach().numpy().copy()
        cd = F / (0.5 * U_IN ** 2 * 2 * R)
        print(f"{it:>4} {c[0]:8.4f} {c[1]:8.4f} {cd:8.4f} "
              f"{g[0]:10.4f} {g[1]:10.4f}", flush=True)
        # trust-region step: fixed length lr along -grad(Cd), clipped to
        # a safe interior box (classification admissibility)
        g_cd = g / (0.5 * U_IN ** 2 * 2 * R)
        step = -lr * g_cd / max(np.linalg.norm(g_cd), 1e-12)
        c = np.clip(c + step, 0.18, 0.82)
    print("final center:", c)


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    lr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    lv = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    main(ns, lr, lv)
