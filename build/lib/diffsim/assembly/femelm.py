import warp as wp


@wp.struct
class FEMElm:
    e: wp.int32
    q: wp.int32
    he: wp.float64


@wp.func
def fe_N(Ntab: wp.array2d(dtype=wp.float64), fe: FEMElm, a: wp.int32) -> wp.float64:
    return Ntab[fe.q, a]


# --- Scaled accessors: precomputed dscale and jac supplied by the kernel factory ---

@wp.func
def fe_dN_s(dNtab: wp.array3d(dtype=wp.float64), fe: FEMElm, a: wp.int32,
            k: wp.int32, dscale: wp.float64) -> wp.float64:
    """Physical derivative: reference dN times dscale = 2/he."""
    return dNtab[fe.q, a, k] * dscale


@wp.func
def fe_detJxW_s(wtab: wp.array(dtype=wp.float64), fe: FEMElm,
                jac: wp.float64) -> wp.float64:
    """Quadrature weight times jac = (he/2)^dim (dim folded in by caller)."""
    return wtab[fe.q] * jac


@wp.func
def fe_d2N_s(d2Ntab: wp.array4d(dtype=wp.float64), f: wp.int32, fe: FEMElm,
             a: wp.int32, i: wp.int32, j: wp.int32, dim_: wp.int32,
             d2scale: wp.float64) -> wp.float64:
    """Physical second derivative at a FACE Gauss point, from face tables
    flattened to [2*dim, nqf, nbf, dim*dim] (warp's 4-dim array cap);
    d2scale = (2/he)^2."""
    return d2Ntab[f, fe.q, a, i * dim_ + j] * d2scale
