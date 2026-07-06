

def test_same_name_bricks_get_distinct_kernels(device):
    """Evaluation finding 3 (CONFIRMED by design review, fixed): kernel
    cache was keyed on brick.__name__, so two bricks with the same class
    name silently shared one compiled kernel — numerically plausible,
    wrong physics. Keys now use class identity."""
    from diffsim.api.equation import _brick_Ae_kernel, CEquation

    def make_brick(coeff):
        import warp as wp
        from diffsim.assembly.femelm import FEMElm, fe_N, fe_dN_s

        @wp.func
        def integrands(fe: FEMElm, Ntab: wp.array2d(dtype=wp.float64),
                       dNtab: wp.array3d(dtype=wp.float64),
                       detJxW: wp.float64, dscale: wp.float64,
                       nbf: int, dim: int, ndof: int,
                       Ae: wp.array3d(dtype=wp.float64), e: int):
            for a in range(nbf):
                for b in range(nbf):
                    v = wp.float64(0.0)
                    for d_ in range(dim):
                        v += fe_dN_s(dNtab, fe, a, d_, dscale) \
                            * fe_dN_s(dNtab, fe, b, d_, dscale)
                    Ae[e, a, b] += wp.float64(coeff) * v * detJxW

        class SameName(CEquation):
            Integrands_Ae = integrands
        return SameName

    B1 = make_brick(1.0)
    B2 = make_brick(2.0)          # SAME __name__, different integrand
    assert B1.__name__ == B2.__name__
    k1 = _brick_Ae_kernel(B1, 4, 4, 2)
    k2 = _brick_Ae_kernel(B2, 4, 4, 2)
    assert k1 is not k2, "same-name bricks shared one compiled kernel"
