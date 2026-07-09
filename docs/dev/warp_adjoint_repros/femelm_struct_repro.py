"""MINIMAL UPSTREAM REPRO (warp 1.14, bug #3 of the DiffSim 4c family):
mutating a struct field inside a kernel loop silently breaks the backward
pass. MEASURED OUTPUT of this script: the struct-mutation kernel returns
ALL-ZERO gradients with no error (silent wrongness — worse than NaN); the
identical math with plain locals returns correct gradients. In larger
kernels (DiffSim's NS residual) the same root cause manifested as
all-NaN tapes. Rule: NO struct field mutation in enable_backward
kernels."""
import numpy as np
import warp as wp


@wp.struct
class Ctx:
    e: wp.int32
    q: wp.int32
    he: wp.float64


@wp.kernel(module="unique", enable_backward=True)
def k_struct(a: wp.array(dtype=wp.float64), w: wp.array2d(dtype=wp.float64),
             out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    c = Ctx()
    c.e = i
    c.he = wp.float64(0.5)
    s = wp.float64(0.0)
    for q in range(4):
        c.q = q                              # struct field mutated per iter
        s += a[i * 4 + q] * w[c.q, 0] * c.he
    wp.atomic_add(out, i, s * s)


@wp.kernel(module="unique", enable_backward=True)
def k_local(a: wp.array(dtype=wp.float64), w: wp.array2d(dtype=wp.float64),
            out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    he = wp.float64(0.5)
    s = wp.float64(0.0)
    for q in range(4):
        s += a[i * 4 + q] * w[q, 0] * he     # identical math, plain locals
    wp.atomic_add(out, i, s * s)


rng = np.random.default_rng(0)
for name, k in (("struct-mutation", k_struct), ("plain-local", k_local)):
    tape = wp.Tape()
    a = wp.array(rng.standard_normal(8), dtype=wp.float64, device="cuda:0",
                 requires_grad=True)
    w = wp.array(rng.standard_normal((4, 2)), dtype=wp.float64,
                 device="cuda:0")
    out = wp.zeros(2, dtype=wp.float64, device="cuda:0", requires_grad=True)
    with tape:
        wp.launch(k, dim=2, inputs=[a, w, out], device="cuda:0")
    tape.backward(grads={out: wp.array(np.ones(2), dtype=wp.float64,
                                       device="cuda:0")})
    g = tape.gradients[a].numpy()
    print(f"{name}: nan = {np.isnan(g).any()}  grad[:2] = {g[:2]}")
