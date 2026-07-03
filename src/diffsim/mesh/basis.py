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
    dim: int
    nbf: int
    nqp: int
    N: np.ndarray    # [nqp, nbf]
    dN: np.ndarray   # [nqp, nbf, dim] (reference derivatives)
    w: np.ndarray    # [nqp]

def basis_tables(p: int, dim: int = 3) -> Tables:
    from itertools import product as iproduct
    pts, wts = gauss_1d(p)
    npe, nq1 = p + 1, len(pts)
    nbf, nqp = npe**dim, nq1**dim
    # x-fastest multi-indices for both qp and nodes (product varies last
    # factor fastest -> reverse columns; matches nodes._local_offsets)
    qidx = np.array(list(iproduct(*[range(nq1)] * dim)), np.int64)[:, ::-1]
    aidx = np.array(list(iproduct(*[range(npe)] * dim)), np.int64)[:, ::-1]
    N = np.zeros((nqp, nbf)); dN = np.zeros((nqp, nbf, dim)); w = np.zeros(nqp)
    N1 = np.zeros((nq1, npe)); dN1 = np.zeros((nq1, npe))
    for q in range(nq1):
        N1[q], dN1[q] = lagrange_1d(p, pts[q])
    for q in range(nqp):
        w[q] = np.prod(wts[qidx[q]])
        for a in range(nbf):
            vals = N1[qidx[q], aidx[a]]                    # per-axis 1D values
            N[q, a] = np.prod(vals)
            for d in range(dim):
                terms = vals.copy()
                terms[d] = dN1[qidx[q, d], aidx[a, d]]
                dN[q, a, d] = np.prod(terms)
    return Tables(p, dim, nbf, nqp, N, dN, w)
