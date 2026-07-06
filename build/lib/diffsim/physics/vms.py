"""VMS machinery for M1b (Task 3): stabilization parameters and the
generalized advection operator M_{a,s}.

Two tau_M forms behind one interface (production-code-conventions.md):

- METRIC form (monolithic steppers; NSEquationLinear.h:332-403 calc_tau_alt):
      tauM = 1/sqrt((2 b0/dt)^2 + u.G u + Ci_f nu^2 G:G),   Ci_f = 36
      tauC = 1/(tauM g.g)
  On our axis-aligned cubes the element metric is diagonal:
      G = (2/h)^2 I,  u.Gu = 4|u|^2/h^2,  G:G = dim (2/h)^4,  g.g = dim (2/h)^2.
  `timestab=False` drops the transient (2 b0/dt)^2 term (steady tau).

- H-BASED form (Leray projection stepper ONLY; the Helmholtz-Leray VMS
  draft Eq. 45 — user directive):
      tau_m = [sigma^2 + c1 |u|^2/h^2 + c2 C_I nu^2/h^4]^(-1/2),  sigma = b0/dt
  Draft-faithful defaults c1=4, c2*C_I = 36*16*dim reproduce the metric form
  on cubes — the two coincide here by construction; they differ on general
  meshes (recorded, not asserted).

Generalized advection (Biswajit's linearized-NSE draft):
      M_{a,s} u = (a.grad) u + s (div a) u,       s in {0, 1/2, 1}
Integration by parts gives the exact-adjoint structure
      <M_{a,s} u, w> = <u, -M_{a,1-s} w> + <(a.n) u, w>_boundary
so with a.n = 0 on the boundary:  A_{a,s} = -A_{a,1-s}^T  discretely
(exact when quadrature integrates the forms exactly). s = 1/2 is the
self-adjoint (energy-stable) choice — the M1b default and the M1c adjoint
mechanism; locked by test before any NS kernel exists.

Device wp.funcs mirror the host functions 1:1 (host mirrors are the gates).
"""
import numpy as np
import warp as wp

CI_F = 36.0


# ------------------------------ host mirrors ------------------------------
def tau_metric_host(u_mag, h, nu, dt=None, b0=1.0, dim=2, timestab=True):
    """(tauM, tauC) — metric-tensor form on axis-aligned cubes."""
    uGu = 4.0 * u_mag ** 2 / h ** 2
    GG = dim * (2.0 / h) ** 4
    gg = dim * (2.0 / h) ** 2
    trans = (2.0 * b0 / dt) ** 2 if (timestab and dt is not None) else 0.0
    tauM = 1.0 / np.sqrt(trans + uGu + CI_F * nu ** 2 * GG)
    return tauM, 1.0 / (tauM * gg)


def tau_hbased_host(u_mag, h, nu, dt=None, b0=1.0, dim=2,
                    c1=4.0, c2CI=None):
    """tau_m — the projection draft's Eq. 45 (h-based). Defaults reproduce
    the metric form on cubes (c2*C_I = 36*16*dim)."""
    if c2CI is None:
        c2CI = CI_F * 16.0 * dim
    sigma = (b0 / dt) if dt is not None else 0.0
    return 1.0 / np.sqrt((2.0 * sigma) ** 2 + c1 * u_mag ** 2 / h ** 2
                         + c2CI * nu ** 2 / h ** 4)


# ------------------------------ device funcs ------------------------------
@wp.func
def tau_m_metric(u_mag: wp.float64, h: wp.float64, nu: wp.float64,
                 sig2: wp.float64, dim_f: wp.float64) -> wp.float64:
    # sig2 = (2 b0/dt)^2 precomputed on host; 0 for steady/timestab off
    uGu = wp.float64(4.0) * u_mag * u_mag / (h * h)
    GG = dim_f * wp.pow(wp.float64(2.0) / h, wp.float64(4.0))
    return wp.float64(1.0) / wp.sqrt(sig2 + uGu
                                     + wp.float64(CI_F) * nu * nu * GG)


@wp.func
def tau_c_metric(tauM: wp.float64, h: wp.float64,
                 dim_f: wp.float64) -> wp.float64:
    gg = dim_f * wp.float64(4.0) / (h * h)
    return wp.float64(1.0) / (tauM * gg)


@wp.func
def tau_m_hbased(u_mag: wp.float64, h: wp.float64, nu: wp.float64,
                 sig2: wp.float64, c1: wp.float64,
                 c2CI: wp.float64) -> wp.float64:
    return wp.float64(1.0) / wp.sqrt(
        sig2 + c1 * u_mag * u_mag / (h * h)
        + c2CI * nu * nu / wp.pow(h, wp.float64(4.0)))


# --------------------- M_{a,s}: host dense form builder --------------------
def advection_matrix_dense(nodes_1d, dim, a_fn, div_a_fn, s, nq=4):
    """Dense scalar advection matrix A_ab = int N_a [a.grad N_b
    + s (div a) N_b] dV on a uniform p1 grid over [0,1]^dim with n
    elements per axis — the FORM gate (device kernels arrive fused inside
    the NS momentum brick, Task 5). nq Gauss points/axis => exact for
    polynomial a up to degree 2nq-3."""
    n = nodes_1d - 1
    h = 1.0 / n
    xg, wg = np.polynomial.legendre.leggauss(nq)
    xg = 0.5 * (xg + 1.0)          # [0,1] reference
    wg = 0.5 * wg
    if dim != 2:
        raise NotImplementedError("form gate is 2D (the identity is dim-blind)")
    Nn = nodes_1d ** 2
    A = np.zeros((Nn, Nn))

    def nid(i, j):
        return i * nodes_1d + j

    # p1 shape functions on [0,1]^2: (1-x)(1-y), x(1-y), (1-x)y, xy
    for ei in range(n):
        for ej in range(n):
            conn = [nid(ei, ej), nid(ei + 1, ej), nid(ei, ej + 1),
                    nid(ei + 1, ej + 1)]
            for gx, wx in zip(xg, wg):
                for gy, wy in zip(xg, wg):
                    x = (ei + gx) * h
                    y = (ej + gy) * h
                    N = np.array([(1 - gx) * (1 - gy), gx * (1 - gy),
                                  (1 - gx) * gy, gx * gy])
                    dNx = np.array([-(1 - gy), (1 - gy), -gy, gy]) / h
                    dNy = np.array([-(1 - gx), -gx, (1 - gx), gx]) / h
                    av = a_fn(x, y)
                    da = div_a_fn(x, y)
                    conv = av[0] * dNx + av[1] * dNy + s * da * N
                    w = wx * wy * h * h
                    A[np.ix_(conn, conn)] += np.outer(N, conv) * w
    return A
