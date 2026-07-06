"""Interaction probe: tau wp.func + consumed mat-accumulator."""
import numpy as np, warp as wp
from diffsim.physics.vms import tau_m_metric

@wp.kernel(module="unique", enable_backward=True)
def k_tm(a: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.float64),
         nu_arr: wp.array(dtype=wp.float64),
         out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    nu = nu_arr[0]
    av = wp.vec2d(a[2 * i], a[2 * i + 1])
    gu = wp.mat22d()
    for b in range(3):
        xv = wp.vec2d(x[6 * i + 2 * b], x[6 * i + 2 * b + 1])
        gu += wp.outer(xv, wp.vec2d(wp.float64(1.0 + float(b)),
                                    wp.float64(2.0)))
    tauM = tau_m_metric(wp.sqrt(wp.dot(av, av)), wp.float64(0.125), nu,
                        wp.float64(1600.0), wp.float64(2.0))
    w = gu * av
    out[i] = tauM * wp.dot(w, w)

rng = np.random.default_rng(0)
tape = wp.Tape()
a = wp.array(rng.standard_normal(4), dtype=wp.float64, device="cuda:0",
             requires_grad=True)
x = wp.array(rng.standard_normal(12), dtype=wp.float64, device="cuda:0")
nua = wp.array(np.array([0.05]), dtype=wp.float64, device="cuda:0",
               requires_grad=True)
out = wp.zeros(2, dtype=wp.float64, device="cuda:0", requires_grad=True)
with tape:
    wp.launch(k_tm, dim=2, inputs=[a, x, nua, out], device="cuda:0")
tape.backward(grads={out: wp.array(np.ones(2), dtype=wp.float64,
                                   device="cuda:0")})
print("tau+mat: a-grad nan =", np.isnan(tape.gradients[a].numpy()).any(),
      "| nu-grad =", float(tape.gradients[nua].numpy()[0]), flush=True)
