"""Replicate L6's resm expression shape exactly, minimally."""
import numpy as np, warp as wp

@wp.kernel(module="unique", enable_backward=True)
def k_resm(a: wp.array(dtype=wp.float64),      # grad (like aq)
           dv: wp.array(dtype=wp.float64),     # grad (like div_aq)
           x: wp.array(dtype=wp.float64),      # frozen
           out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    av = wp.vec2d(a[2 * i], a[2 * i + 1])
    uval = wp.vec2d(x[2 * i], x[2 * i + 1])            # non-grad
    gu = wp.outer(uval, wp.vec2d(wp.float64(1.0), wp.float64(2.0)))
    gradp = wp.vec2d(x[2 * i + 1], x[2 * i])
    sigma = wp.float64(20.0)
    s_skew = wp.float64(0.5)
    resm = (sigma + s_skew * dv[i]) * uval + gu * av + gradp
    out[i] = wp.dot(resm, resm) * wp.float64(1e-3)

tape = wp.Tape()
a = wp.array(np.array([1.0, 2.0, 3.0, 4.0]), dtype=wp.float64,
             device="cuda:0", requires_grad=True)
dv = wp.array(np.array([0.3, -0.2]), dtype=wp.float64, device="cuda:0",
              requires_grad=True)
x = wp.array(np.array([0.5, -1.0, 2.0, 0.7]), dtype=wp.float64,
             device="cuda:0")
out = wp.zeros(2, dtype=wp.float64, device="cuda:0", requires_grad=True)
with tape:
    wp.launch(k_resm, dim=2, inputs=[a, dv, x, out], device="cuda:0")
tape.backward(grads={out: wp.array(np.ones(2), dtype=wp.float64,
                                   device="cuda:0")})
ga = tape.gradients[a].numpy()
gd = tape.gradients[dv].numpy()
print(f"a-grad nan={np.isnan(ga).any()} {ga}", flush=True)
print(f"dv-grad nan={np.isnan(gd).any()} {gd}", flush=True)
