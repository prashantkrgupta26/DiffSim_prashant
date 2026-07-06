# PASTE-READY ISSUE for github.com/NVIDIA/warp

*Verified standalone against warp 1.14.0 / CUDA (RTX 6000 Ada, sm_89,
driver via WSL2) on 2026-07-06. Paste title + body below; attach or
inline the repro (also at `docs/superpowers/warp_adjoint_repros/femelm_struct_repro.py`).*

---

**Title:**

`[BUG] Mutating a struct field inside a kernel loop silently zeroes gradients (enable_backward)`

**Body:**

## Description

When a `wp.struct` local's field is assigned inside a kernel loop (e.g. a
per-iteration index stored on a struct), the backward pass of an
`enable_backward=True` kernel produces **silently wrong gradients** — in
the minimal repro below they come back **all zeros**, with no warning or
error. The identical computation using a plain local variable instead of
the struct field differentiates correctly.

In larger kernels (an FEM residual kernel where a quadrature-context
struct carried element/point indices) the same root cause manifested as
**all-NaN gradients**, which is how we found it. The silent-zero minimal
form seems more dangerous: nothing signals that the tape is broken.

Forward values are correct in all cases — only the adjoint is affected.

## Repro (30 lines, self-contained)

```python
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
    print(f"{name}: grad[:2] = {g[:2]}")
```

## Output (warp 1.14.0, CUDA, float64)

```
struct-mutation: grad[:2] = [0. 0.]        <-- silently wrong (all zeros)
plain-local:     grad[:2] = [-0.67647656  0.55669775]   <-- correct
```

The two kernels compute the same function of `a`; only the struct-field
indirection differs.

## Expected

Either correct gradients through the struct-field indirection, or a
compile-time error/warning that struct field mutation inside loops is
unsupported under `enable_backward=True`.

## Environment

- warp-lang 1.14.0 (PyPI)
- Python 3.12, CUDA 12.x, NVIDIA RTX 6000 Ada (sm_89), WSL2 Linux
- float64 kernels; reproduces with `module="unique"` and default module
  mode

## Related note (separate lead, same hunt)

While isolating this we also observed that the documented
dynamic-loop-not-replayed limitation interacts confusingly with adjoint
debugging: fully-unrolled kernels (verified in the generated `.cu`) were
initially suspected because gradient corruption looked identical. A
runtime diagnostic ("this backward kernel contains non-replayed dynamic
loops") would make both failure modes much easier to attribute. Happy to
open that as a separate feature request if useful.
