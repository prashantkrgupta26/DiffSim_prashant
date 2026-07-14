r"""Gauge-anchored neural free energy for beyond-Flory-Huggins learning
(roadmap "Learning rung 2").

WHY THIS EXISTS.  The M4 result learned the *parametric* Flory-Huggins
interaction (chi_pf/chi_ps/chi_fs, k_e) to 1e-7, and in doing so mapped a
hard gauge/identifiability structure (benchmarks/phase-field/m4_learn_fmix.py
docstring; m4-milestone-report.md Sec 2):

  * A CONSTANT in f'(c) shifts mu by a constant -> invisible to the conserved
    c-dynamics (T0 gauge, structurally unfixable by any c data).
  * A LINEAR-in-c term in f'(c) is a *quadratic* free energy already spanned by
    the FH chi term: the map (chi_pf, chi_ps, c1) -> (+t,+t,+t) cancels
    identically in mu and the Jacobian (measured trajectory difference 5e-15).
    An optimizer with this mode free slides exactly along it.

So a learnable free-energy correction that is to mean anything BEYOND FH must
live in the subspace ORTHOGONAL to {1, c} -- i.e. start at quadratic-in-c in
f' (cubic-and-up in f).  This module builds exactly that: a small MLP produces
a raw correction to f'(c), and we L2-project the {1, c} content out of it over
the composition domain every evaluation.  The projection coefficients depend on
the network weights, so the gauge fix is differentiable and the optimizer never
sees the aliased directions.

  f'(c)  = f'_FH(c; A, B)  +  [ g(c; theta) - a0 - a1 c ]
  (a0, a1) = argmin_{a0,a1}  || g - a0 - a1 c ||^2_{L2(Omega)}

The base FH part keeps the identifiable quadratic physics; the bracket is the
gauge-anchored beyond-FH functional the instrument-space / composition-diverse
losses can actually pin down (see benchmarks/phase-field/beyond_fh_*).

INTERFACE.  Exposes .fp(c) and .fpp(c) on torch tensors of Gauss-point
compositions, the contract adjoint/torch_twin.CHTwin consumes when its
``energy`` is an object rather than the "fh"/"poly" string.  The MLP carries an
analytic input-derivative (chain rule through tanh) so f''(c) is closed-form and
we avoid nested autograd inside the Newton solve; gradients w.r.t. {A, B, theta}
flow through the twin's autograd-through-convergence exactly as for the scalar
FH params, and are FD-gated in tests/test_neural_energy.py."""
import numpy as np
import torch

torch.set_default_dtype(torch.float64)


def _fp_fh(c, A, B):
    return A * (torch.log(c) - torch.log(1.0 - c)) + B * (1.0 - 2.0 * c)


def _fpp_fh(c, A, B):
    return A * (1.0 / c + 1.0 / (1.0 - c)) - 2.0 * B


class NeuralCHEnergy(torch.nn.Module):
    r"""FH bulk energy + a gauge-anchored MLP correction to f'(c).

    Parameters
    ----------
    A, B : float
        Flory-Huggins entropic scale and interaction (chi).  Learnable.
    hidden : tuple[int]
        MLP hidden widths (tanh activations).  R^1 -> R^1.
    c_lo, c_hi : float
        Composition domain over which the {1, c} gauge modes are projected out
        (should bracket the compositions the march actually visits).
    corr_scale : float
        Fixed output scale on the raw correction (keeps the beyond-FH term a
        perturbation at init; the network can still grow it through training).
    n_quad : int
        Gauss-Legendre nodes for the L2 gauge projection over [c_lo, c_hi].
    seed : int
        Weight-init seed (kept explicit — Math.random is unavailable in some
        harnesses and reproducibility is the point).
    """

    def __init__(self, A=1.0, B=2.5, hidden=(16, 16), c_lo=0.05, c_hi=0.95,
                 corr_scale=0.1, n_quad=64, seed=0):
        super().__init__()
        self.A = torch.nn.Parameter(torch.tensor(float(A)))
        self.B = torch.nn.Parameter(torch.tensor(float(B)))
        self.corr_scale = float(corr_scale)
        self.c_lo, self.c_hi = float(c_lo), float(c_hi)
        self.c_mid = 0.5 * (c_hi + c_lo)
        self.c_half = 0.5 * (c_hi - c_lo)

        # explicit MLP weights so the input-derivative g'(c) is closed form.
        g = torch.Generator().manual_seed(int(seed))
        dims = [1, *hidden, 1]
        self.W = torch.nn.ParameterList()
        self.b = torch.nn.ParameterList()
        for din, dout in zip(dims[:-1], dims[1:]):
            # modest init; last layer small so the correction starts tiny
            scale = (1.0 / np.sqrt(din)) * (0.1 if dout == 1 else 1.0)
            self.W.append(torch.nn.Parameter(
                torch.randn(dout, din, generator=g) * scale))
            self.b.append(torch.nn.Parameter(torch.zeros(dout)))

        # Gauss-Legendre nodes/weights on [c_lo, c_hi] for the gauge projection.
        x, w = np.polynomial.legendre.leggauss(int(n_quad))
        cq = self.c_mid + self.c_half * x
        wq = self.c_half * w
        self.register_buffer("cq", torch.tensor(cq))
        self.register_buffer("wq", torch.tensor(wq))

    # -- raw MLP correction and its analytic input-derivative --------------
    def _net(self, c):
        """Return (g(c), dg/dc) on the raw composition tensor c (any shape).
        Normalises the input to [-1, 1] for conditioning; the chain rule
        carries the 1/c_half factor into the derivative."""
        u = (c - self.c_mid) / self.c_half            # normalised input
        a = u.reshape(-1, 1)                           # [P, 1]
        da = torch.ones_like(a) / self.c_half          # d a / d c  (pre-net)
        nlayers = len(self.W)
        for li in range(nlayers):
            z = a @ self.W[li].T + self.b[li]          # [P, dout]
            dz = da @ self.W[li].T                      # [P, dout]
            if li < nlayers - 1:
                t = torch.tanh(z)
                a = t
                da = (1.0 - t * t) * dz                 # tanh'
            else:
                a = z                                   # linear head
                da = dz
        g = self.corr_scale * a.reshape(c.shape)
        dg = self.corr_scale * da.reshape(c.shape)
        return g, dg

    # -- L2 projection of the raw correction onto {1, c} over the domain ----
    def _gauge_coeffs(self):
        """(a0, a1) = argmin || g - a0 - a1 c ||^2 over L2([c_lo,c_hi]).
        Differentiable in theta (g depends on the weights)."""
        g, _ = self._net(self.cq)
        w, c = self.wq, self.cq
        # Gram matrix of {1, c} and moments <phi_i, g>
        s0 = w.sum()
        s1 = (w * c).sum()
        s2 = (w * c * c).sum()
        G = torch.stack([torch.stack([s0, s1]),
                         torch.stack([s1, s2])])
        m = torch.stack([(w * g).sum(), (w * c * g).sum()])
        a = torch.linalg.solve(G, m)
        return a[0], a[1]

    # -- gauge-anchored correction to f'(c) and its derivative -------------
    def corr_fp(self, c):
        a0, a1 = self._gauge_coeffs()
        g, _ = self._net(c)
        return g - a0 - a1 * c

    def corr_fpp(self, c):
        _, dg = self._net(c)
        _, a1 = self._gauge_coeffs()
        return dg - a1

    # -- energy interface consumed by the twin -----------------------------
    def fp(self, c):
        return _fp_fh(c, self.A, self.B) + self.corr_fp(c)

    def fpp(self, c):
        return _fpp_fh(c, self.A, self.B) + self.corr_fpp(c)

    # -- diagnostics --------------------------------------------------------
    def gauge_residual(self):
        """<corr, 1> and <corr, c> over the domain — both must be ~0 by
        construction (the gauge-anchoring property test asserts this)."""
        c, w = self.cq, self.wq
        with torch.no_grad():
            r = self.corr_fp(c)
            return float((w * r).sum()), float((w * c * r).sum())

    def corr_on(self, c_np):
        """The learned beyond-FH correction f'_corr(c) as a numpy array on a
        given composition grid (for plotting / truth comparison)."""
        with torch.no_grad():
            return self.corr_fp(torch.tensor(np.asarray(c_np, np.float64))
                                 ).cpu().numpy()


def _legendre(u, k):
    """Shifted-Legendre P_k(u) and derivative dP_k/du on u in [-1,1].
    k>=2 is the beyond-quadratic, gauge-anchored basis; k in {0,1} are the
    {1, c} gauge modes themselves (P0 constant = T0, P1 linear = T1) and are
    available only to DEMONSTRATE the alias they create (an un-anchored fit).
    Closed form keeps the correction's f''(c) analytic."""
    if k == 0:
        return torch.ones_like(u), torch.zeros_like(u)
    if k == 1:
        return u, torch.ones_like(u)
    if k == 2:
        return 0.5 * (3.0 * u * u - 1.0), 3.0 * u
    if k == 3:
        return 0.5 * (5.0 * u ** 3 - 3.0 * u), 0.5 * (15.0 * u * u - 3.0)
    if k == 4:
        return (35.0 * u ** 4 - 30.0 * u * u + 3.0) / 8.0, \
               (140.0 * u ** 3 - 60.0 * u) / 8.0
    raise ValueError(f"basis degree {k} not in 2..4")


class BasisCorrEnergy(torch.nn.Module):
    r"""FH bulk energy + a low-dimensional gauge-anchored beyond-FH correction
    f'_corr(c) = sum_k gamma_k P_k((c-mid)/half), on shifted-Legendre degrees
    k>=2.  Because P_k for k>=2 are L2-orthogonal to {1, c} on [c_lo, c_hi] by
    construction, the correction is gauge-anchored EXACTLY (no projection) —
    P0/P1 (the unidentifiable T0/T1 modes) are simply not in the basis.

    This is the interpretable sibling of NeuralCHEnergy: a handful of
    coefficients whose Jacobian-Gramian conditioning is a clean identifiability
    number (it reproduces the recorded M4 'T2..T4 alias the FH span over one
    trajectory' finding), while NeuralCHEnergy is the general functional head.
    Same .fp/.fpp contract for adjoint/torch_twin.CHTwin."""

    def __init__(self, A=1.0, B=2.5, degrees=(2, 3, 4), coeffs=None,
                 c_lo=0.05, c_hi=0.95):
        super().__init__()
        self.A = torch.nn.Parameter(torch.tensor(float(A)))
        self.B = torch.nn.Parameter(torch.tensor(float(B)))
        self.degrees = tuple(int(k) for k in degrees)
        self.c_mid = 0.5 * (c_hi + c_lo)
        self.c_half = 0.5 * (c_hi - c_lo)
        g0 = (torch.zeros(len(self.degrees)) if coeffs is None
              else torch.tensor([float(x) for x in coeffs]))
        self.gamma = torch.nn.Parameter(g0)

    def corr_fp(self, c):
        u = (c - self.c_mid) / self.c_half
        out = torch.zeros_like(c)
        for i, k in enumerate(self.degrees):
            pk, _ = _legendre(u, k)
            out = out + self.gamma[i] * pk
        return out

    def corr_fpp(self, c):
        u = (c - self.c_mid) / self.c_half
        out = torch.zeros_like(c)
        for i, k in enumerate(self.degrees):
            _, dpk = _legendre(u, k)
            out = out + self.gamma[i] * dpk / self.c_half
        return out

    def fp(self, c):
        return _fp_fh(c, self.A, self.B) + self.corr_fp(c)

    def fpp(self, c):
        return _fpp_fh(c, self.A, self.B) + self.corr_fpp(c)

    def corr_on(self, c_np):
        with torch.no_grad():
            return self.corr_fp(torch.tensor(np.asarray(c_np, np.float64))
                                 ).cpu().numpy()
