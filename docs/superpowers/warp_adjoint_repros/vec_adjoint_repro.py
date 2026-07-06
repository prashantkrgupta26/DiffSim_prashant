"""Minimal repro: are wp.vec2d op adjoints NaN in warp 1.14 kernels?"""
import numpy as np, warp as wp

@wp.kernel(module="unique", enable_backward=True)
def k_vec(a: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    v = wp.vec2d(a[2 * i], a[2 * i + 1])
    w = wp.vec2d(wp.float64(2.0), wp.float64(3.0))
    out[i] = wp.dot(v, w) + wp.dot(v, v)

@wp.kernel(module="unique", enable_backward=True)
def k_scal(a: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    x = a[2 * i]
    y = a[2 * i + 1]
    out[i] = (2.0 * x + 3.0 * y) + (x * x + y * y)

for name, k in (("vec2d", k_vec), ("scalar", k_scal)):
    tape = wp.Tape()
    a = wp.array(np.array([1.0, 2.0, 3.0, 4.0]), dtype=wp.float64,
                 device="cuda:0", requires_grad=True)
    out = wp.zeros(2, dtype=wp.float64, device="cuda:0", requires_grad=True)
    with tape:
        wp.launch(k, dim=2, inputs=[a, out], device="cuda:0")
    tape.backward(grads={out: wp.array(np.ones(2), dtype=wp.float64,
                                       device="cuda:0")})
    g = tape.gradients[a].numpy()
    # analytic: d/dx = 2 + 2x, d/dy = 3 + 2y
    exact = np.array([2 + 2 * 1, 3 + 2 * 2, 2 + 2 * 3, 3 + 2 * 4], float)
    print(f"{name}: grad = {g}  exact = {exact}  "
          f"{'OK' if np.allclose(g, exact) else 'MISMATCH/NAN'}", flush=True)
