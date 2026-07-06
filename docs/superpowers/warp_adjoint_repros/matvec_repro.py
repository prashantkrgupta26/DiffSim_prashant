"""Minimal repro candidate: wp.mat22d * wp.vec2d adjoint."""
import numpy as np, warp as wp

@wp.kernel(module="unique", enable_backward=True)
def k_mv(a: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    v = wp.vec2d(a[2 * i], a[2 * i + 1])
    m = wp.outer(v, wp.vec2d(wp.float64(1.0), wp.float64(2.0)))
    w = m * v                                  # mat-vec: the suspect
    out[i] = wp.dot(w, w)

@wp.kernel(module="unique", enable_backward=True)
def k_rows(a: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    v = wp.vec2d(a[2 * i], a[2 * i + 1])
    m = wp.outer(v, wp.vec2d(wp.float64(1.0), wp.float64(2.0)))
    w = wp.vec2d(wp.dot(wp.vec2d(m[0, 0], m[0, 1]), v),
                 wp.dot(wp.vec2d(m[1, 0], m[1, 1]), v))   # manual rows
    out[i] = wp.dot(w, w)

for name, k in (("matvec", k_mv), ("manual-rows", k_rows)):
    tape = wp.Tape()
    a = wp.array(np.array([1.0, 2.0, 3.0, 4.0]), dtype=wp.float64,
                 device="cuda:0", requires_grad=True)
    out = wp.zeros(2, dtype=wp.float64, device="cuda:0", requires_grad=True)
    with tape:
        wp.launch(k, dim=2, inputs=[a, out], device="cuda:0")
    tape.backward(grads={out: wp.array(np.ones(2), dtype=wp.float64,
                                       device="cuda:0")})
    g = tape.gradients[a].numpy()
    print(f"{name}: grad = {g}  nan = {np.isnan(g).any()}", flush=True)
