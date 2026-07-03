from dataclasses import dataclass
import numpy as np

def gauss_1d(p: int):
    if p == 1:
        a = 1.0 / np.sqrt(3.0)
        return np.array([-a, a]), np.array([1.0, 1.0])
    b = np.sqrt(3.0 / 5.0)
    return np.array([-b, 0.0, b]), np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

def lagrange_1d(p: int, xi: float):
    if p == 1:
        N = np.array([0.5 * (1 - xi), 0.5 * (1 + xi)])
        dN = np.array([-0.5, 0.5])
    else:
        N = np.array([0.5 * xi * (xi - 1.0), 1.0 - xi * xi, 0.5 * xi * (xi + 1.0)])
        dN = np.array([xi - 0.5, -2.0 * xi, xi + 0.5])
    return N, dN

@dataclass(frozen=True)
class Tables:
    p: int
    nbf: int
    nqp: int
    N: np.ndarray    # [nqp, nbf]
    dN: np.ndarray   # [nqp, nbf, 3] (reference derivatives)
    w: np.ndarray    # [nqp]

def basis_tables(p: int) -> Tables:
    pts, wts = gauss_1d(p)
    npe, nq1 = p + 1, len(pts)
    nbf, nqp = npe**3, nq1**3
    N = np.zeros((nqp, nbf)); dN = np.zeros((nqp, nbf, 3)); w = np.zeros(nqp)
    q = 0
    for qk in range(nq1):
        for qj in range(nq1):
            for qi in range(nq1):                     # x fastest
                Nx, dNx = lagrange_1d(p, pts[qi])
                Ny, dNy = lagrange_1d(p, pts[qj])
                Nz, dNz = lagrange_1d(p, pts[qk])
                w[q] = wts[qi] * wts[qj] * wts[qk]
                for k in range(npe):
                    for j in range(npe):
                        for i in range(npe):
                            a = i + npe * j + npe * npe * k
                            N[q, a] = Nx[i] * Ny[j] * Nz[k]
                            dN[q, a, 0] = dNx[i] * Ny[j] * Nz[k]
                            dN[q, a, 1] = Nx[i] * dNy[j] * Nz[k]
                            dN[q, a, 2] = Nx[i] * Ny[j] * dNz[k]
                q += 1
    return Tables(p, nbf, nqp, N, dN, w)
