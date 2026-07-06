"""Final suspect: loop-ACCUMULATED mat consumed in a grad expression."""
import numpy as np, warp as wp

@wp.kernel(module="unique", enable_backward=True)
def k_ma(a: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.float64),
         out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    av = wp.vec2d(a[2 * i], a[2 * i + 1])
    gu = wp.mat22d()
    for b in range(3):                          # unrolled accumulation
        xv = wp.vec2d(x[6 * i + 2 * b], x[6 * i + 2 * b + 1])
        dn = wp.vec2d(wp.float64(1.0 + float(b)), wp.float64(2.0))
        gu += wp.outer(xv, dn)
    w = gu * av
    out[i] = wp.dot(w, w)

@wp.kernel(module="unique", enable_backward=True)
def k_sc(a: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.float64),
         out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    av = wp.vec2d(a[2 * i], a[2 * i + 1])
    g00 = wp.float64(0.0); g01 = wp.float64(0.0)
    g10 = wp.float64(0.0); g11 = wp.float64(0.0)
    for b in range(3):
        x0 = x[6 * i + 2 * b]
        x1 = x[6 * i + 2 * b + 1]
        d0 = wp.float64(1.0 + float(b)); d1 = wp.float64(2.0)
        g00 += x0 * d0; g01 += x0 * d1
        g10 += x1 * d0; g11 += x1 * d1
    w = wp.vec2d(g00 * av[0] + g01 * av[1], g10 * av[0] + g11 * av[1])
    out[i] = wp.dot(w, w)

rng = np.random.default_rng(0)
for name, k in (("mat-accum", k_ma), ("scalar-accum", k_sc)):
    tape = wp.Tape()
    a = wp.array(rng.standard_normal(4), dtype=wp.float64, device="cuda:0",
                 requires_grad=True)
    x = wp.array(rng.standard_normal(12), dtype=wp.float64, device="cuda:0")
    out = wp.zeros(2, dtype=wp.float64, device="cuda:0", requires_grad=True)
    with tape:
        wp.launch(k, dim=2, inputs=[a, x, out], device="cuda:0")
    tape.backward(grads={out: wp.array(np.ones(2), dtype=wp.float64,
                                       device="cuda:0")})
    g = tape.gradients[a].numpy()
    print(f"{name}: nan={np.isnan(g).any()} grad={g}", flush=True)
