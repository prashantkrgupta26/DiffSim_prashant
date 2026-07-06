import warp as wp

@wp.kernel
def _dot_kernel(a: wp.array(dtype=wp.float64), b: wp.array(dtype=wp.float64),
                out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    wp.atomic_add(out, 0, a[i] * b[i])

@wp.kernel
def _axpy_kernel(alpha: wp.float64, x: wp.array(dtype=wp.float64),
                 y: wp.array(dtype=wp.float64)):
    i = wp.tid()
    y[i] = y[i] + alpha * x[i]

@wp.kernel
def _xpay_kernel(alpha: wp.float64, x: wp.array(dtype=wp.float64),
                 y: wp.array(dtype=wp.float64)):
    i = wp.tid()
    y[i] = x[i] + alpha * y[i]

@wp.kernel
def _mult_kernel(a: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.float64),
                 y: wp.array(dtype=wp.float64)):
    i = wp.tid()
    y[i] = a[i] * x[i]

def dot(a, b, device):
    out = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(_dot_kernel, dim=len(a), inputs=[a, b, out], device=device)
    return float(out.numpy()[0])

def axpy(alpha, x, y, device):   # y += alpha x
    wp.launch(_axpy_kernel, dim=len(x), inputs=[wp.float64(alpha), x, y], device=device)

def xpay(alpha, x, y, device):   # y = x + alpha y
    wp.launch(_xpay_kernel, dim=len(x), inputs=[wp.float64(alpha), x, y], device=device)

def hadamard(a, x, y, device):   # y = a * x elementwise
    wp.launch(_mult_kernel, dim=len(x), inputs=[a, x, y], device=device)
