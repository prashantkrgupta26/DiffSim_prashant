r"""Worked example bricks — read these to learn the Integrands API.

A *brick* is a physics term written the way TalyFEM/Dendrite/DiffPack users
already know: you state what happens at ONE integration point, and the
framework owns everything else (the element loop, the Gauss-point loop,
quadrature and the Jacobian, the hanging-node constraints, assembly into the
global sparse matrix, and the solve). If you can write the weak form, you can
write a brick.

`PoissonBrick` below is the canonical example, annotated so you can trace every
line back to the variational form. It is the exact brick the "lego gate"
regression test (`tests/test_brick_api.py`) checks against the hand-written
framework path bit-for-bit, so the comments describe running code.

-------------------------------------------------------------------------------
The Poisson problem, from strong form to code
-------------------------------------------------------------------------------

STRONG FORM.  Find u(x) on a domain Omega with

        -div( grad u ) = f      in Omega
                     u = g      on the Dirichlet boundary Gamma_D
            grad u . n = h      on the Neumann boundary Gamma_N

WEAK FORM.  Multiply by a test function w (with w = 0 on Gamma_D), integrate
over Omega, and integrate the second-derivative term by parts to move one
derivative onto w:

    -Int_Omega  w div(grad u) dV
        = Int_Omega grad w . grad u dV  -  Int_Gamma w (grad u . n) dS

So the problem becomes: find u such that for every admissible w,

    Int_Omega grad w . grad u dV  =  Int_Omega w f dV  +  Int_{Gamma_N} w h dS
    \___________________________/     \_____________________________________/
            a(w, u)                                 l(w)
        the BILINEAR FORM                        the LOAD (linear) FORM

    a(w, u) -> the stiffness matrix  -> Integrands_Ae   (this brick)
    l(w)    -> the load vector        -> Integrands_be   (this brick; the
                                         Neumann surface term is a separate
                                         side-integrand, omitted here)

DISCRETIZE.  Expand u_h = sum_b N_b u_b and take w = N_a (Galerkin). On one
element, with quadrature points x_q and weights w_q, the element stiffness and
load are

    K^e_{ab} = Int_{Omega_e} grad N_a . grad N_b dV
             = sum_q ( sum_k dN_a/dx_k * dN_b/dx_k ) * |J| w_q

    f^e_a    = Int_{Omega_e} N_a f dV
             = sum_q  N_a(x_q) * f(x_q) * |J| w_q

MAP TO CODE.  The framework calls the integrand once per (element e, quadrature
point q) and hands you the pieces:

    fe_dN_s(dNtab, fe, a, k, dscale)  =  dN_a/dx_k   (PHYSICAL-space derivative;
                                         dscale = 2/h is the reference->physical
                                         chain-rule factor for the octree cube)
    fe_N(Ntab, fe, a)                 =  N_a(x_q)
    detJxW                            =  |J| w_q     (Jacobian * quad weight)
    Ae[e, ndof*a, ndof*b] += ...      =  scatter into the element matrix block
                                         (ndof = 1 for Poisson, so the block is
                                          a single scalar per (a, b) node pair)

The two nested loops over `a` and `b` are exactly the test/trial basis pair
(w_a, u_b); the inner loop over `k` is the dot product grad N_a . grad N_b.
That is the whole correspondence: one line of math, one line of code.
"""
import warp as wp

from ..assembly.femelm import FEMElm, fe_N, fe_dN_s  # noqa: F401 (bricks use these)
from .equation import CEquation


class PoissonBrick(CEquation):
    r"""-div(grad u) = f, the Hughes-form volume brick (spec S3.1 example).

    Bilinear form  a(w, u) = Int grad w . grad u dV   -> Integrands_Ae
    Load form      l(w)    = Int w f dV               -> Integrands_be
    """
    ndof = 1                       # one unknown per node (a scalar field u)

    @staticmethod
    @wp.func
    def Integrands_Ae(fe: FEMElm,
                      Ntab: wp.array2d(dtype=wp.float64),
                      dNtab: wp.array3d(dtype=wp.float64),
                      detJxW: wp.float64, dscale: wp.float64,
                      nbf: wp.int32, dim: wp.int32, ndof: wp.int32,
                      Ae: wp.array3d(dtype=wp.float64), e: wp.int32):
        # a(w, u) = Int grad w . grad u dV, evaluated at ONE Gauss point.
        # K^e_{ab} += ( grad N_a . grad N_b ) * |J| w_q
        for a in range(nbf):                       # test function  w = N_a
            for b in range(nbf):                   # trial function u = N_b
                K = wp.float64(0.0)
                for k in range(dim):               # the dot product over x_k
                    # dN_a/dx_k * dN_b/dx_k  (physical-space gradients)
                    K += fe_dN_s(dNtab, fe, a, k, dscale) \
                         * fe_dN_s(dNtab, fe, b, k, dscale)
                # accumulate this Gauss point's contribution (ndof = 1, so the
                # node-major block index ndof*a is just a)
                Ae[e, ndof * a, ndof * b] += K * detJxW

    @staticmethod
    @wp.func
    def Integrands_be(fe: FEMElm,
                      Ntab: wp.array2d(dtype=wp.float64),
                      dNtab: wp.array3d(dtype=wp.float64),
                      detJxW: wp.float64, dscale: wp.float64,
                      nbf: wp.int32, dim: wp.int32, ndof: wp.int32,
                      fq: wp.array(dtype=wp.float64), nqp: wp.int32,
                      be: wp.array2d(dtype=wp.float64), e: wp.int32):
        # l(w) = Int w f dV, evaluated at ONE Gauss point.
        # f^e_a += N_a(x_q) * f(x_q) * |J| w_q
        fv = fq[e * nqp + fe.q]                     # source f sampled at x_q
        for a in range(nbf):                       # test function w = N_a
            be[e, ndof * a] += fe_N(Ntab, fe, a) * fv * detJxW
