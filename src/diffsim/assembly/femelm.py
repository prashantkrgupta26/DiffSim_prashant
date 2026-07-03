import warp as wp


@wp.struct
class FEMElm:
    e: wp.int32
    q: wp.int32
    he: wp.float64


@wp.func
def fe_N(Ntab: wp.array2d(dtype=wp.float64), fe: FEMElm, a: wp.int32) -> wp.float64:
    return Ntab[fe.q, a]


@wp.func
def fe_dN(dNtab: wp.array3d(dtype=wp.float64), fe: FEMElm, a: wp.int32, k: wp.int32) -> wp.float64:
    return dNtab[fe.q, a, k] * (wp.float64(2.0) / fe.he)


@wp.func
def fe_detJxW(wtab: wp.array(dtype=wp.float64), fe: FEMElm) -> wp.float64:
    half = fe.he * wp.float64(0.5)
    return wtab[fe.q] * half * half * half
