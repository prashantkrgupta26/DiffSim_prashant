r"""M5 S0/S1: (M, K)-generic coupled multi-Cahn-Hilliard x multi-Allen-Cahn
brick for phase separation + crystallization of organic blends
("OrgElMorph"; formulation contract docs/theory/crystallization_formulation_p1.md,
milestone ledger docs/dev/specs/2026-07-10-m5-orgelmorph.md).

FIELDS AND LAYOUT.  n = M + 1 species with volume fractions phi
(sum = 1; the LAST species, index M, is the eliminated solvent:
phi_s = 1 - sum_i phi_i).  Species 0..M-1 each carry a conserved
(phi_i, mu_i) CH pair; the FIRST K retained species (0..K-1, K <= M)
are crystallizable and additionally carry a non-conserved crystallinity
psi_k in [0, 1] and an orientation theta_k.  Node-major dof layout:

    [(phi_0, mu_0), ..., (phi_{M-1}, mu_{M-1}),
     (psi_0, theta_0), ..., (psi_{K-1}, theta_{K-1})]   = 2M + 2K dofs.

(M, K) are COMPILE-TIME constants of the kernel factory (the nbf/nqp
pattern); species loops run over small compile-time ranges and the chi
matrices ride in as small device arrays.

CHI CONVENTION.  All four chi matrices are (M+1) x (M+1) in the FULL
species indexing INCLUDING the eliminated solvent (index M).  Diagonals
unused.  chi_ac[i, j] couples AMORPHOUS i with CRYSTALLINE j;
chi_ca[i, j] couples CRYSTALLINE i with AMORPHOUS j.  Consistency of the
pair energy requires chi_aa/chi_cc symmetric and chi_ca = chi_ac.T —
asserted at construction (2512.16370 Eq. 26 writes the ordered pair sum
i < j; the derivative bookkeeping below assumes the symmetrized reading).

FREE ENERGY (volume density, kT/v0 units).  Two compile-time bulk modes:

"p1" — THE RATIFIED FORM (P1 memo Sec 2 = Siber-Ronsin-Harting
2512.16370 Eq. 26/33):

  f = SUM_k (phi_k^2/N_k) [ psi_k (1-psi_k) dsig_k + psi_k^2 drive_k ]
    + SUM_i (phi_i/N_i) ln phi_i
    + SUM_{i<j} phi_i phi_j chi_eff(i, j)
    + SUM_i (kap_i/2)|grad phi_i|^2 + SUM_k (eps_k^2/2)|grad psi_k|^2
    + b SUM_i 1/phi_i                                (simplex reg., wodo)
  chi_eff(i,j) = (1-psi_i)(1-psi_j) chi_aa + (1-psi_i) psi_j chi_ac
               + psi_i (1-psi_j) chi_ca + psi_i psi_j chi_cc
  drive_k = dh_k (1 - T/Tm_k)                        (literal P1 form)

  SIGN RULING (recorded for Baskar review): with the memo's literal
  "(1 - T/Tm)" factor, a crystallizing species below its melting point
  needs dh_k < 0 (the crystallization enthalpy, released) for psi -> 1
  to LOWER f.  The companion dynamics paper (2310.11844 Eq. 3) writes
  the same driving as +L (T/Tm - 1) with L > 0 the latent heat of
  fusion — the two conventions agree under dh = -L.  We implement the
  memo literally and document; gates use dh < 0 in p1 mode.
  NOTE the p1 psi-bulk psi(1-psi) dsig is a BARRIER, not a double well:
  the crystalline state is the BOUNDARY minimum psi = 1, held by the
  projected [0, 1] iterates (2512.16370 pp. 8-10 give no other bounds
  treatment; projection is our recorded choice).

"r14" — the 2310.11844 polynomial family (their Eqs. 3 + 7), needed for
the QUANTITATIVE S1c replication because its psi-polynomials differ
from the p1 form:

  f_cr  = SUM_k phi_k [ q(psi_k) dsig_k + p(psi_k) drive_k ],
          q(psi) = psi^2 (1-psi)^2,  p(psi) = psi^2 (3 - 2 psi),
          drive_k = dh_k (T/Tm_k - 1)   (dh_k = L latent heat > 0;
          NEGATIVE below Tm — their Turnbull form, Eq. 3),
          dsig_k = W-bar (barrier height, energy-density units)
  chi_eff(i,j) = chi_aa[i,j] + psi_i^2 chi_ca[i,j] + psi_j^2 chi_ac[i,j]
               + psi_i^2 psi_j^2 chi_cc[i,j]
  (their Eq. 7 phi(1-phi)(chi_aa + chi_ca psi^2) is the K=1, M=1 case;
  in r14 mode the chi_ac/ca/cc inputs are DELTA-chi corrections in the
  2512.16370 Eq. 29 sense, with the paper's psi^2 interpolation.)
  Note the LINEAR phi_k prefactor of f_cr (their Eq. 3 phi*rho[...]),
  vs phi^2/N in p1.

DYNAMICS (P1 Sec 3 = 2310.11844 Sec 2.2).

CH (each retained phi_i):  dphi_i/dt = div( SUM_j Lam_ij grad mu_j ),
mu_i = df/dphi_i - df/dphi_s - kap_i lap phi_i (exchange potential;
solvent eliminated).  Mobility closures (NAMED, per the house rule):
  mob="const":    constant SPD Onsager matrix Lam (gate variant; the
                  S0 parity case).
  mob="fastmode": their Eq. 15 fast-mode scalar (M = 1 ONLY, asserted):
                  Lam(phi) = (1-phi)^2 phi N_1 D1(phi)
                           + phi^2 (1-phi) N_2 D2(phi),
                  log-mean self-diffusion interpolation (their Eq. 16,
                  asymmetric exponent convention as printed):
                  D1(phi) = D1_lo^(1-phi) D1_hi^phi,
                  D2(phi) = D2_lo^phi     D2_hi^(1-phi).
                  Picard-frozen in the Jacobian (wodo var_mob lesson:
                  same converged solution, the Appendix-A heuristic
                  absorbs the odd extra iteration).
  mob="fastmode_n" | "slowmode_n" (S2): the MULTICOMPONENT Onsager
                  matrices of the anchor framework (Ronsin-Harting 2022,
                  2204.11628 Eqs. 11-12), any M, over the FULL species
                  set (eliminated solvent included) with omega_i
                  = N_i phi_i D_i(phi, psi):
                  fast:  Lam_ii = (1-phi_i)^2 w_i + phi_i^2 SUM_{k!=i} w_k
                         Lam_ij = -(1-phi_i) phi_j w_i
                                  - (1-phi_j) phi_i w_j
                                  + phi_i phi_j SUM_{k!=i,j} w_k
                  slow:  Lam_ii = w_i (1 - w_i/SUM w),
                         Lam_ij = -w_i w_j / SUM w
                  (the M x M block over retained species enters the
                  exchange-potential fluxes; fastmode_n reduces EXACTLY
                  to "fastmode" at M = 1 — parity-gated).  Self-diffusion
                  is the VIGNES law (their Eq. 14 product):
                  D_i = f_drop PROD_j Dslf[i, j]^phi_j, with Dslf[i, j]
                  the self-diffusion of i in pure j ((M+1)^2 input
                  D_self), and f_drop the LIQUID-SOLID MOBILITY DROP
                  (their Eq. 13-14):
                    log f(x; d, c, w) = (1/2) log(d) (1 + tanh(w(x-c))),
                    x = psi_tot = 1 - PROD_k (1 - psi_k),
                  parameters ls_drop = (d, c, w) (their PenVal/PenCentr/
                  PenSlop; d = 1 disables, the default).  UNLIKE the
                  M = 1 "fastmode" (Picard-frozen), the _n modes carry
                  the EXACT dLam/dphi_j and dLam/dpsi_k blocks in the
                  Jacobian (div(dLam .. grad mu) terms) — MEASURED
                  NECESSITY: the 16390 deep-quench SD at the Vignes
                  3-decade mobility contrast reduces frozen-Lam Newton
                  to a 13-43-iteration linear tail with dt-ladder
                  collapse to 1e-3 s (2026-07-10 calibration), i.e. the
                  wodo var_mob lesson does NOT transfer to
                  composition-singular mobilities.  dLam/dpsi_k uses
                  the homogeneity Lam ~ f_drop:
                  dLam_ij/dpsi_k = Lam_ij dln(f)/dpsi_k.
                  CHC noise_phi with a matrix mobility needs a flux-space
                  factorization we do not carry — asserted noise_phi = 0
                  in the _n modes.

CRYSTALLINE NOISE DAMPING (S2, anchor Eq. 18's f(phi_k) interpolation):
noise_damp = (d, c, w) multiplies the FDT psi-noise per GP by
f(psi_k; d, c, w) with the SAME Eq. 13 interpolation evaluated at the
last committed psi_k (frozen per attempt, consistent with the noise
freezing) — fluctuations damp inside crystalline domains (their
PenValFluctAC/PenCentrFluctAC/PenSlopFluctAC).  None disables.

Stochastic AC (each psi_k):  dpsi_k/dt = -L_k [ df/dpsi_k
- div(eps_k^2 grad psi_k) ] + xi_k, with L_k the LUMPED mobility
prefactor (their (v0 N_1/RT) M(phi) with M = M0 constant — the
reference-case closure; composition-dependent M(phi) is the S2+ item).
xi_k = FDT Gaussian, load-only, frozen per attempt, seedable:
per-GP std = noise_psi * sqrt(2 L_k / (dt * wJ_gp)) — the wJ factor is
the white-noise quadrature normalization (Var Int N_a xi
= 2 noise_psi^2 L_k Int N_a^2 / dt exactly), so noise_psi^2 = kB T in
the nondimensional energy-density x volume units.  This is the
dimensionally consistent version of the wodo CHC knob (which absorbed
the normalization into the amplitude); their Sec 2.2.1 std
sqrt((2 v0/Na) N1 M(phi)) maps onto it.  Optional CHC noise for phi
(wodo pattern, conserved flux, same wJ normalization) via noise_phi.

ORIENTATION theta (P1 ruling R1).  2310.11844 Eq. 5 gives only a
Kronecker-delta energy p(psi)(alpha/2) delta_{grad theta} and NO
evolution equation (verified: theta appears only on their p. 3; the
kinetics defer to Ronsin & Harting, Adv. Theory Simul. 2022, 2200286).
Per the contract we implement the STANDARD KOBAYASHI-GIGA REGULARIZED
form and document the choice:

  f_ori,k = p(psi_k) [ (alpha_k/2) |grad theta_k|_d
                       + (beta_k/2) |grad theta_k|^2 ],
  |g|_d = sqrt(|g|^2 + kg_delta^2)      (the KG smoothing of |g|),
  p(psi) = psi^2 (3 - 2 psi)            (their Eq. 5 gate; p'(0) = 0
                                         so amorphous regions feel no
                                         orientation force),

  (p(psi_k) + p_floor) dtheta_k/dt =
      L_th,k div[ (p(psi_k) + p_floor)
                  ( (alpha_k/2)/|grad theta_k|_d + beta_k ) grad theta_k ]

p_floor > 0 keeps the theta row nonsingular where psi = 0 (there the
equation reduces to p_floor theta_t = 0: theta FROZEN in the amorphous
phase up to the tiny p_floor diffusion).  beta_k = 0 reproduces the
ratified alpha/2 |grad theta| term alone (KG-regularized); beta_k > 0
is the standard KWC quadratic augmentation, exposed for stiff cases.
The psi equation picks up the consistent coupling p'(psi_k) f_ori-bracket
— this is what makes impinging crystals with different theta STOP
(grain boundary energy) instead of merging.  theta is treated as a
plain scalar (no 2 pi wrap; grain-periodic bookkeeping recorded as a
follow-up — gates use well-separated theta values).

JACOBIAN APPROXIMATIONS (documented, Picard-class; converged solutions
are unaffected — fixed-point of the exact residual):
  (i)  the 1/|grad theta|_d factor is frozen per iterate (the exact
       KWC tensor I/|g| - g g^T/|g|^3 is indefinite-prone);
  (ii) the psi-row theta-column block is dropped (the p'(psi) Eori
       term's theta derivative);
  (iii) fastmode Lam(phi) is frozen (wodo pattern).
The psi-column of the theta row and all phi/psi cross blocks are exact.

TIME STEPPING.  BDF1 + Newton with the house safeguards: trust clamp
(+ OPTIONAL ||r||_2 backtracking line search, line_search=True — the S2
globalization; measured on the 16390 SD purification: plain Newton
line-searches to 3-4 iterations/step at dt = 2e-2 where the unglobalized
iteration rejected down to dt ~ 1e-4)
(a phi/psi increment beyond 2 units scales the WHOLE update), projected
iterates (phi clipped to [1e-3, 1-1e-3] then simplex-rescaled so
sum phi <= 1 - 1e-3 preserving ratios; psi clipped to [0, 1]; theta
free), divergence detection.  march() = the wodo Appendix-A ladder
(iters < 20 => dt *= 1.25 capped at dt_max; no convergence/divergence
=> dt *= 0.25, retry).  Temperature T is a parameter with a T_fn(t)
schedule hook (annealing protocols); drive_k is refreshed per attempt.
Linear solve: splu (default) | cudss (wodo DirectSolver pattern).
blockch_pairs integration NOT wired (the AC blocks need the per-psi
scalar-block extension of the "pairs" meta — recorded follow-up).
"""
import numpy as np
import scipy.sparse as sp
import warp as wp

from ..assembly.operators import _kernel_cache
from .ternary_ch import _rlog, _rinv
from .wodo_film import _binv2, _binv3


# ---------------------------------------------------------------------
# numpy mirrors (tests/MMS/diagnostics; identical regularizations)
# ---------------------------------------------------------------------
def np_rlog(x, eps=1e-4):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x < eps, np.log(eps) + (x - eps) / eps,
                    np.log(np.maximum(x, eps)))


def np_rinv(x, eps=1e-4):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x < eps, 1.0 / eps, 1.0 / np.maximum(x, eps))


def np_ls_interp(x, d, c, w):
    """Anchor Eq. 13 liquid-solid interpolation: log f = (1/2) log(d)
    (1 + tanh(w (x - c))).  f -> 1 for x << c (liquid), f -> ~d for
    x >> c (crystal).  Used for the mobility drop (kernel mirror) and
    the crystalline FDT-noise damping (host side)."""
    x = np.asarray(x, dtype=np.float64)
    return d ** (0.5 * (1.0 + np.tanh(w * (x - c))))


def _np_chi_eff(pars, i, j, psi_of):
    """Pair value chi_eff(i, j) at the mode of pars (arrays broadcast)."""
    caa = pars["chi_aa"][i, j]
    cac = pars["chi_ac"][i, j]
    cca = pars["chi_ca"][i, j]
    ccc = pars["chi_cc"][i, j]
    si, sj = psi_of(i), psi_of(j)
    if pars.get("bulk", "p1") == "r14":
        return caa + si ** 2 * cca + sj ** 2 * cac + si ** 2 * sj ** 2 * ccc
    return ((1 - si) * (1 - sj) * caa + (1 - si) * sj * cac
            + si * (1 - sj) * cca + si * sj * ccc)


def _np_dchi_first(pars, a, b, psi_of):
    """d chi_eff(a, b) / d psi_a."""
    caa = pars["chi_aa"][a, b]
    cac = pars["chi_ac"][a, b]
    cca = pars["chi_ca"][a, b]
    ccc = pars["chi_cc"][a, b]
    sa, sb = psi_of(a), psi_of(b)
    if pars.get("bulk", "p1") == "r14":
        return 2 * sa * (cca + sb ** 2 * ccc)
    return (1 - sb) * (cca - caa) + sb * (ccc - cac)


def np_potentials(phis, psis, pars):
    """Bulk exchange potentials mu_i and psi driving forces df/dpsi_k
    (chi + crystal-bulk + entropy + b-reg; NO gradient terms, NO
    orientation term — MMS adds those analytically).  phis: list of M
    arrays; psis: list of K arrays; pars: dict with chi_aa/ac/ca/cc
    ((M+1)^2), Ninv (M+1), dsig, drive (K), breg, bulk ("p1"|"r14").
    Returns (mu list[M], dfdpsi list[K])."""
    M = len(phis)
    K = len(psis)
    ps = 1.0 - sum(phis)
    phi_of = lambda l: phis[l] if l < M else ps
    psi_of = lambda l: psis[l] if l < K else 0.0
    Ninv = pars["Ninv"]
    breg = pars.get("breg", 0.0)
    r14 = pars.get("bulk", "p1") == "r14"

    def W(k):
        s = psis[k]
        if r14:
            return (s ** 2 * (1 - s) ** 2 * pars["dsig"][k]
                    + s ** 2 * (3 - 2 * s) * pars["drive"][k])
        return s * (1 - s) * pars["dsig"][k] + s ** 2 * pars["drive"][k]

    def Wp(k):
        s = psis[k]
        if r14:
            return (2 * s * (1 - s) * (1 - 2 * s) * pars["dsig"][k]
                    + 6 * s * (1 - s) * pars["drive"][k])
        return (1 - 2 * s) * pars["dsig"][k] + 2 * s * pars["drive"][k]

    def S(m):
        return sum(phi_of(l) * _np_chi_eff(pars, m, l, psi_of)
                   for l in range(M + 1) if l != m)

    mus = []
    for i in range(M):
        mu = (Ninv[i] * (np_rlog(phis[i]) + 1.0)
              - Ninv[M] * (np_rlog(ps) + 1.0) + S(i) - S(M))
        if i < K:
            if r14:
                mu = mu + W(i)
            else:
                mu = mu + 2.0 * phis[i] * Ninv[i] * W(i)
        if breg:
            b2 = lambda x: 1.0 / np.maximum(x, 1e-3) ** 2
            mu = mu + breg * (b2(ps) - b2(phis[i]))
        mus.append(mu)
    dfs = []
    for k in range(K):
        if r14:
            bulk = phis[k] * Wp(k)
        else:
            bulk = phis[k] ** 2 * Ninv[k] * Wp(k)
        chi = phis[k] * sum(phi_of(l) * _np_dchi_first(pars, k, l, psi_of)
                            for l in range(M + 1) if l != k)
        dfs.append(bulk + chi)
    return mus, dfs


# ---------------------------------------------------------------------
# kernel factory
# ---------------------------------------------------------------------
def make_mpf_newton(nbf: int, nqp: int, dim: int, M: int, K: int,
                    bulk: str = "p1", mob: str = "const",
                    theta: str = "kwc"):
    """Monolithic Newton kernel for the 2M+2K node-major system.
    (M, K, bulk, mob, theta) compile-time; see module docstring.
    theta="frozen" replaces the KWC theta rows by the EXACT bookkeeping
    identity theta = theta_old (mass-matrix row) — the anchor's marker
    semantics, and the well-posed form of the alpha = beta = 0 limit:
    the degenerate KWC row (diag ~ p_floor sigma NN ~ 1e-9 with
    roundoff coupling entries) makes the direct solve return garbage
    theta increments (MEASURED 2026-07-11: |dx_theta| 4.9e13 at
    |r_theta| 1e-19, dt-INDEPENDENT — the divergence guard then
    underflows the ladder; whether the garbage crosses the 1e6 guard
    is an assembly-atomics coin flip, the house FP-fate lesson)."""
    key = ("mpf_newton", nbf, nqp, dim, M, K, bulk, mob, theta)
    if key in _kernel_cache:
        return _kernel_cache[key]
    assert bulk in ("p1", "r14"), bulk
    assert mob in ("const", "fastmode", "fastmode_n", "slowmode_n"), mob
    if mob == "fastmode":
        assert M == 1, "fastmode Onsager closure: M = 1 only " \
            "(fastmode_n is the generic-M mode)"
    assert theta in ("kwc", "frozen"), theta
    TH_FROZEN = theta == "frozen"
    dim_pow = float(dim)
    Kp = max(K, 1)
    ndof = 2 * M + 2 * K
    R14 = bulk == "r14"
    FASTMODE = mob == "fastmode"
    SLOWN = mob == "slowmode_n"
    MATMOB = mob in ("fastmode_n", "slowmode_n")
    n_sp = M + 1
    VecSp = wp.types.vector(length=n_sp, dtype=wp.float64)
    VecM = wp.types.vector(length=M, dtype=wp.float64)
    VecKp = wp.types.vector(length=Kp, dtype=wp.float64)
    MatSp = wp.types.matrix(shape=(n_sp, n_sp), dtype=wp.float64)
    MatKSp = wp.types.matrix(shape=(Kp, n_sp), dtype=wp.float64)
    MatMM = wp.types.matrix(shape=(M, M), dtype=wp.float64)
    MatSpM = wp.types.matrix(shape=(n_sp, M), dtype=wp.float64)
    MatMD = wp.types.matrix(shape=(M, M * M), dtype=wp.float64)
    MatMK = wp.types.matrix(shape=(M, Kp), dtype=wp.float64)
    MatKK = wp.types.matrix(shape=(Kp, Kp), dtype=wp.float64)
    MatKM = wp.types.matrix(shape=(Kp, M), dtype=wp.float64)

    @wp.kernel(module="unique", enable_backward=False,
               module_options={"max_unroll": 0})
    def mpf_k(conn: wp.array2d(dtype=wp.int32),
              h: wp.array(dtype=wp.float64),
              Ntab: wp.array2d(dtype=wp.float64),
              dNtab: wp.array3d(dtype=wp.float64),
              wtab: wp.array(dtype=wp.float64),
              vals: wp.array2d(dtype=wp.float64),    # [ngp, ndof]
              grads: wp.array3d(dtype=wp.float64),   # [ngp, ndof, dim]
              hist: wp.array2d(dtype=wp.float64),    # [ngp, ndof] (mu rows 0)
              src: wp.array2d(dtype=wp.float64),     # [ngp, ndof] MMS
              qpsi: wp.array2d(dtype=wp.float64),    # [ngp, Kp] FDT noise
              qphi: wp.array3d(dtype=wp.float64),    # [ngp, M, dim] CHC
              chi_aa: wp.array2d(dtype=wp.float64),  # [(M+1), (M+1)]
              chi_ac: wp.array2d(dtype=wp.float64),
              chi_ca: wp.array2d(dtype=wp.float64),
              chi_cc: wp.array2d(dtype=wp.float64),
              Ninv: wp.array(dtype=wp.float64),      # [M+1]
              Ons: wp.array2d(dtype=wp.float64),     # [M, M] (const mob)
              dlo: wp.array(dtype=wp.float64),       # [M+1] fastmode D(own->0)
              dhi: wp.array(dtype=wp.float64),       # [M+1] fastmode D(own->1)
              Dslf: wp.array2d(dtype=wp.float64),    # [(M+1),(M+1)] Vignes
              dsl: wp.float64, csl: wp.float64,      # Eq. 13 drop d, c
              wsl: wp.float64,                       # Eq. 13 drop w
              kap: wp.array(dtype=wp.float64),       # [M]
              dsig: wp.array(dtype=wp.float64),      # [Kp]
              drive: wp.array(dtype=wp.float64),     # [Kp] (host: T-law)
              eps2: wp.array(dtype=wp.float64),      # [Kp]
              Lpsi: wp.array(dtype=wp.float64),      # [Kp]
              alpha: wp.array(dtype=wp.float64),     # [Kp] KWC |g| coeff
              beta: wp.array(dtype=wp.float64),      # [Kp] KWC quad coeff
              Lth: wp.array(dtype=wp.float64),       # [Kp] theta mobility
              sigma: wp.float64, breg: wp.float64,
              kgd: wp.float64, pfloor: wp.float64,
              Ae: wp.array3d(dtype=wp.float64),
              be: wp.array2d(dtype=wp.float64)):
        e = wp.tid()
        he = h[e]
        jac = wp.pow(he * wp.float64(0.5), wp.float64(dim_pow))
        dscale = wp.float64(2.0) / he
        for q in range(nqp):
            dJxW = wtab[q] * jac
            gp = e * nqp + q
            # ---- unpack fields -----------------------------------------
            phiv = VecSp()
            ssum = wp.float64(0.0)
            for i in range(M):
                phiv[i] = vals[gp, 2 * i]
                ssum += vals[gp, 2 * i]
            phiv[M] = wp.float64(1.0) - ssum
            psiv = VecKp()
            for k in range(Kp):
                psiv[k] = wp.float64(0.0)
            for k in range(K):
                psiv[k] = vals[gp, 2 * M + 2 * k]
            # ---- chi_eff pair matrix + first-index psi derivative ------
            chiE = MatSp()
            for a1 in range(n_sp):
                chiE[a1, a1] = wp.float64(0.0)
            for a1 in range(n_sp):
                for b1 in range(n_sp):
                    if b1 > a1:
                        sa = wp.float64(0.0)
                        sb = wp.float64(0.0)
                        if a1 < K:
                            sa = psiv[a1]
                        if b1 < K:
                            sb = psiv[b1]
                        if wp.static(R14):
                            v = chi_aa[a1, b1] \
                                + sa * sa * chi_ca[a1, b1] \
                                + sb * sb * chi_ac[a1, b1] \
                                + sa * sa * sb * sb * chi_cc[a1, b1]
                        else:
                            v = (wp.float64(1.0) - sa) \
                                * (wp.float64(1.0) - sb) * chi_aa[a1, b1] \
                                + (wp.float64(1.0) - sa) * sb \
                                * chi_ac[a1, b1] \
                                + sa * (wp.float64(1.0) - sb) \
                                * chi_ca[a1, b1] \
                                + sa * sb * chi_cc[a1, b1]
                        chiE[a1, b1] = v
                        chiE[b1, a1] = v
            # D1[k, l] = d chi_eff(k, l) / d psi_k (k crystallizable)
            D1 = MatKSp()
            for k in range(Kp):
                for l in range(n_sp):
                    D1[k, l] = wp.float64(0.0)
            for k in range(K):
                for l in range(n_sp):
                    if l != k:
                        sl = wp.float64(0.0)
                        if l < K:
                            sl = psiv[l]
                        if wp.static(R14):
                            D1[k, l] = wp.float64(2.0) * psiv[k] \
                                * (chi_ca[k, l] + sl * sl * chi_cc[k, l])
                        else:
                            D1[k, l] = (wp.float64(1.0) - sl) \
                                * (chi_ca[k, l] - chi_aa[k, l]) \
                                + sl * (chi_cc[k, l] - chi_ac[k, l])
            # ---- crystal bulk W(psi), W', W'' --------------------------
            Wv = VecKp()
            Wpv = VecKp()
            Wppv = VecKp()
            for k in range(Kp):
                Wv[k] = wp.float64(0.0)
                Wpv[k] = wp.float64(0.0)
                Wppv[k] = wp.float64(0.0)
            for k in range(K):
                s_ = psiv[k]
                if wp.static(R14):
                    om = wp.float64(1.0) - s_
                    Wv[k] = s_ * s_ * om * om * dsig[k] \
                        + s_ * s_ * (wp.float64(3.0)
                                     - wp.float64(2.0) * s_) * drive[k]
                    Wpv[k] = wp.float64(2.0) * s_ * om \
                        * (wp.float64(1.0) - wp.float64(2.0) * s_) \
                        * dsig[k] + wp.float64(6.0) * s_ * om * drive[k]
                    Wppv[k] = wp.float64(2.0) \
                        * (wp.float64(1.0) - wp.float64(6.0) * s_
                           + wp.float64(6.0) * s_ * s_) * dsig[k] \
                        + (wp.float64(6.0) - wp.float64(12.0) * s_) \
                        * drive[k]
                else:
                    Wv[k] = s_ * (wp.float64(1.0) - s_) * dsig[k] \
                        + s_ * s_ * drive[k]
                    Wpv[k] = (wp.float64(1.0) - wp.float64(2.0) * s_) \
                        * dsig[k] + wp.float64(2.0) * s_ * drive[k]
                    Wppv[k] = wp.float64(-2.0) * dsig[k] \
                        + wp.float64(2.0) * drive[k]
            # ---- orientation (KG-regularized KWC) ----------------------
            Sd = VecKp()      # |grad theta|_delta
            Eori = VecKp()    # (a/2)|g|_d + (b/2)|g|^2
            porv = VecKp()    # p(psi)
            porp = VecKp()    # p'(psi)
            ceff = VecKp()    # L_th ((a/2)/|g|_d + b)   (KG-Picard)
            for k in range(Kp):
                Sd[k] = wp.float64(1.0)
                Eori[k] = wp.float64(0.0)
                porv[k] = wp.float64(0.0)
                porp[k] = wp.float64(0.0)
                ceff[k] = wp.float64(0.0)
            for k in range(K):
                g2 = wp.float64(0.0)
                for dd in range(dim):
                    gtd = grads[gp, 2 * M + 2 * k + 1, dd]
                    g2 += gtd * gtd
                Sd[k] = wp.sqrt(g2 + kgd * kgd)
                Eori[k] = wp.float64(0.5) * alpha[k] * Sd[k] \
                    + wp.float64(0.5) * beta[k] * g2
                s_ = psiv[k]
                porv[k] = s_ * s_ * (wp.float64(3.0)
                                     - wp.float64(2.0) * s_)
                porp[k] = wp.float64(6.0) * s_ * (wp.float64(1.0) - s_)
                ceff[k] = Lth[k] * (wp.float64(0.5) * alpha[k] / Sd[k]
                                    + beta[k])
            # ---- exchange potentials + derivatives ---------------------
            # S_m = SUM_{l != m} phi_l chi_eff(m, l)
            Ssp = VecSp()
            for m in range(n_sp):
                acc = wp.float64(0.0)
                for l in range(n_sp):
                    if l != m:
                        acc += phiv[l] * chiE[m, l]
                Ssp[m] = acc
            mub = VecM()
            dmudphi = MatMM()
            dmudpsi = MatMK()
            for i in range(M):
                for k in range(Kp):
                    dmudpsi[i, k] = wp.float64(0.0)
            for i in range(M):
                v = Ninv[i] * (_rlog(phiv[i]) + wp.float64(1.0)) \
                    - Ninv[M] * (_rlog(phiv[M]) + wp.float64(1.0)) \
                    + Ssp[i] - Ssp[M] \
                    + breg * (_binv2(phiv[M]) - _binv2(phiv[i]))
                if i < K:
                    if wp.static(R14):
                        v += Wv[i]
                    else:
                        v += wp.float64(2.0) * phiv[i] * Ninv[i] * Wv[i]
                mub[i] = v
                for j in range(M):
                    dv = Ninv[M] * _rinv(phiv[M]) - chiE[i, M] \
                        - chiE[M, j] \
                        + wp.float64(2.0) * breg * _binv3(phiv[M])
                    if j == i:
                        dv += Ninv[i] * _rinv(phiv[i]) \
                            + wp.float64(2.0) * breg * _binv3(phiv[i])
                        if wp.static(not R14):
                            if i < K:      # r14 bulk is linear in phi
                                dv += wp.float64(2.0) * Ninv[i] * Wv[i]
                    else:
                        dv += chiE[i, j]
                    dmudphi[i, j] = dv
                for k in range(K):
                    # d mu_i / d psi_k = dS_i/dpsi_k - dS_s/dpsi_k
                    dv = wp.float64(0.0)
                    if k == i:
                        for l in range(n_sp):
                            if l != i:
                                dv += phiv[l] * D1[i, l]
                        if wp.static(R14):
                            dv += Wpv[i]
                        else:
                            dv += wp.float64(2.0) * phiv[i] * Ninv[i] \
                                * Wpv[i]
                    else:
                        # d chi_eff(i, k)/d psi_k = D1[k, i] (symmetry)
                        dv = phiv[k] * D1[k, i]
                    dv -= phiv[k] * D1[k, M]
                    dmudpsi[i, k] = dv
            # ---- psi driving force + second derivatives ----------------
            Fpsi = VecKp()
            d2pp = MatKK()
            d2pf = MatKM()
            for k in range(Kp):
                Fpsi[k] = wp.float64(0.0)
                for l in range(Kp):
                    d2pp[k, l] = wp.float64(0.0)
                for j in range(M):
                    d2pf[k, j] = wp.float64(0.0)
            for k in range(K):
                chis = wp.float64(0.0)
                for l in range(n_sp):
                    if l != k:
                        chis += phiv[l] * D1[k, l]
                if wp.static(R14):
                    bulkF = phiv[k] * Wpv[k]
                    bulk2 = phiv[k] * Wppv[k]
                    dbdf = Wpv[k]          # d(bulk dF/dpsi)/dphi_k
                else:
                    bulkF = phiv[k] * phiv[k] * Ninv[k] * Wpv[k]
                    bulk2 = phiv[k] * phiv[k] * Ninv[k] * Wppv[k]
                    dbdf = wp.float64(2.0) * phiv[k] * Ninv[k] * Wpv[k]
                Fpsi[k] = bulkF + phiv[k] * chis + porp[k] * Eori[k]
                d2pp[k, k] = bulk2 \
                    + (wp.float64(6.0) - wp.float64(12.0) * psiv[k]) \
                    * Eori[k]
                if wp.static(R14):
                    # r14 chi curvature: dD1[k,l]/dpsi_k = D1/psi_k
                    # (D1 = 2 psi_k (chi_ca + s_l^2 chi_cc)); was a
                    # frozen quasi-Newton block through S1 (measured FD
                    # gap 4.7e-4 rel) — exact since S2
                    for l in range(n_sp):
                        if l != k:
                            sl2 = wp.float64(0.0)
                            if l < K:
                                sl2 = psiv[l] * psiv[l]
                            d2pp[k, k] += phiv[k] * phiv[l] \
                                * wp.float64(2.0) \
                                * (chi_ca[k, l] + sl2 * chi_cc[k, l])
                for l in range(K):
                    if l != k:
                        if wp.static(R14):
                            c2 = wp.float64(4.0) * psiv[k] * psiv[l] \
                                * chi_cc[k, l]
                        else:
                            c2 = chi_aa[k, l] - chi_ac[k, l] \
                                - chi_ca[k, l] + chi_cc[k, l]
                        d2pp[k, l] = phiv[k] * phiv[l] * c2
                for j in range(M):
                    if j == k:
                        d2pf[k, j] = dbdf + chis - phiv[k] * D1[k, M]
                    else:
                        d2pf[k, j] = phiv[k] * (D1[k, j] - D1[k, M])
            # ---- mobility ----------------------------------------------
            lam = wp.float64(0.0)
            if wp.static(FASTMODE):
                pc = wp.min(wp.max(phiv[0], wp.float64(1e-6)),
                            wp.float64(1.0) - wp.float64(1e-6))
                omp = wp.float64(1.0) - pc
                d1s = wp.pow(dlo[0], omp) * wp.pow(dhi[0], pc)
                d2s = wp.pow(dlo[1], pc) * wp.pow(dhi[1], omp)
                lam = omp * omp * pc / Ninv[0] * d1s \
                    + pc * pc * omp / Ninv[1] * d2s
            lamM = MatMM()
            dLamT = MatMD()      # dLamT[i, jp*M + j] = d Lam_{i,jp}/d phi_j
            dlnf = VecKp()       # d ln(f_drop) / d psi_k
            for i0 in range(M):
                for j0 in range(M):
                    lamM[i0, j0] = wp.float64(0.0)
                for j0 in range(M * M):
                    dLamT[i0, j0] = wp.float64(0.0)
            for k0 in range(Kp):
                dlnf[k0] = wp.float64(0.0)
            if wp.static(MATMOB):
                # liquid-solid drop factor (Eq. 13-14; dsl = 1 disables)
                pt = wp.float64(1.0)
                for k0 in range(K):
                    pt *= wp.float64(1.0) - wp.min(
                        wp.max(psiv[k0], wp.float64(0.0)), wp.float64(1.0))
                tnh = wp.tanh(wsl * (wp.float64(1.0) - pt - csl))
                fdrop = wp.pow(dsl, wp.float64(0.5)
                               * (wp.float64(1.0) + tnh))
                # d ln f/d psi_k = 0.5 ln(d) w sech^2 PROD_{l!=k}(1-psi_l)
                dfac = wp.float64(0.5) * wp.log(dsl) * wsl \
                    * (wp.float64(1.0) - tnh * tnh)
                for k0 in range(K):
                    prd = wp.float64(1.0)
                    for l0 in range(K):
                        if l0 != k0:
                            prd *= wp.float64(1.0) - wp.min(
                                wp.max(psiv[l0], wp.float64(0.0)),
                                wp.float64(1.0))
                    dlnf[k0] = dfac * prd
                # omega_i = N_i phi_i D_i, Vignes D_i (Eq. 14), and
                # d omega_i/d phi_j (j retained; phi_M = 1 - sum)
                omv = VecSp()
                dom = MatSpM()
                somv = wp.float64(0.0)
                for i0 in range(n_sp):
                    dsi = fdrop
                    for j0 in range(n_sp):
                        pj = wp.min(wp.max(phiv[j0], wp.float64(0.0)),
                                    wp.float64(1.0))
                        dsi *= wp.pow(Dslf[i0, j0], pj)
                    pc0 = wp.min(wp.max(phiv[i0], wp.float64(1e-6)),
                                 wp.float64(1.0) - wp.float64(1e-6))
                    omv[i0] = pc0 / Ninv[i0] * dsi
                    somv += omv[i0]
                    for j0 in range(M):
                        fac = wp.float64(0.0)
                        if i0 == j0:
                            fac = wp.float64(1.0)
                        if i0 == M:
                            fac = wp.float64(-1.0)
                        dom[i0, j0] = fac * omv[i0] / pc0 \
                            + omv[i0] * wp.log(Dslf[i0, j0]
                                               / Dslf[i0, M])
                dS = VecM()
                for j0 in range(M):
                    acc0 = wp.float64(0.0)
                    for i0 in range(n_sp):
                        acc0 += dom[i0, j0]
                    dS[j0] = acc0
                for i0 in range(M):
                    if wp.static(SLOWN):
                        lamM[i0, i0] = omv[i0] \
                            * (wp.float64(1.0) - omv[i0] / somv)
                        for j0 in range(M):
                            dLamT[i0, i0 * M + j0] = dom[i0, j0] \
                                * (wp.float64(1.0)
                                   - wp.float64(2.0) * omv[i0] / somv) \
                                + omv[i0] * omv[i0] / (somv * somv) \
                                * dS[j0]
                    else:
                        opi = wp.float64(1.0) - phiv[i0]
                        lamM[i0, i0] = opi * opi * omv[i0] \
                            + phiv[i0] * phiv[i0] * (somv - omv[i0])
                        for j0 in range(M):
                            dij = wp.float64(0.0)
                            if j0 == i0:
                                dij = wp.float64(1.0)
                            dLamT[i0, i0 * M + j0] = \
                                wp.float64(-2.0) * opi * dij * omv[i0] \
                                + opi * opi * dom[i0, j0] \
                                + wp.float64(2.0) * phiv[i0] * dij \
                                * (somv - omv[i0]) \
                                + phiv[i0] * phiv[i0] \
                                * (dS[j0] - dom[i0, j0])
                    for l0 in range(M):
                        if l0 != i0:
                            if wp.static(SLOWN):
                                lamM[i0, l0] = -omv[i0] * omv[l0] / somv
                                for j0 in range(M):
                                    dLamT[i0, l0 * M + j0] = \
                                        -(dom[i0, j0] * omv[l0]
                                          + omv[i0] * dom[l0, j0]) \
                                        / somv \
                                        + omv[i0] * omv[l0] \
                                        / (somv * somv) * dS[j0]
                            else:
                                opi = wp.float64(1.0) - phiv[i0]
                                opl = wp.float64(1.0) - phiv[l0]
                                rest = somv - omv[i0] - omv[l0]
                                lamM[i0, l0] = -opi * phiv[l0] * omv[i0] \
                                    - opl * phiv[i0] * omv[l0] \
                                    + phiv[i0] * phiv[l0] * rest
                                for j0 in range(M):
                                    dij = wp.float64(0.0)
                                    if j0 == i0:
                                        dij = wp.float64(1.0)
                                    dlj = wp.float64(0.0)
                                    if j0 == l0:
                                        dlj = wp.float64(1.0)
                                    dLamT[i0, l0 * M + j0] = \
                                        dij * phiv[l0] * omv[i0] \
                                        - opi * dlj * omv[i0] \
                                        - opi * phiv[l0] * dom[i0, j0] \
                                        + dlj * phiv[i0] * omv[l0] \
                                        - opl * dij * omv[l0] \
                                        - opl * phiv[i0] * dom[l0, j0] \
                                        + (dij * phiv[l0]
                                           + dlj * phiv[i0]) * rest \
                                        + phiv[i0] * phiv[l0] \
                                        * (dS[j0] - dom[i0, j0]
                                           - dom[l0, j0])
            # ---- assemble ----------------------------------------------
            for a in range(nbf):
                Na = Ntab[q, a]
                gmu = VecM()     # grad Na . grad mu_j
                gphi = VecM()    # grad Na . grad phi_i
                gq = VecM()      # grad Na . qphi_i (CHC noise)
                for i in range(M):
                    gmu[i] = wp.float64(0.0)
                    gphi[i] = wp.float64(0.0)
                    gq[i] = wp.float64(0.0)
                gpsi = VecKp()
                gth = VecKp()
                for k in range(Kp):
                    gpsi[k] = wp.float64(0.0)
                    gth[k] = wp.float64(0.0)
                for dd in range(dim):
                    gNa = dNtab[q, a, dd] * dscale
                    for i in range(M):
                        gmu[i] += gNa * grads[gp, 2 * i + 1, dd]
                        gphi[i] += gNa * grads[gp, 2 * i, dd]
                        gq[i] += gNa * qphi[gp, i, dd]
                    for k in range(K):
                        gpsi[k] += gNa * grads[gp, 2 * M + 2 * k, dd]
                        gth[k] += gNa * grads[gp, 2 * M + 2 * k + 1, dd]
                # residual rows (be = -r)
                for i in range(M):
                    tr = wp.float64(0.0)
                    sm = wp.float64(0.0)
                    if wp.static(FASTMODE):
                        tr = lam * gmu[i]
                        sm = wp.sqrt(lam)
                    else:
                        if wp.static(MATMOB):
                            for j in range(M):
                                tr += lamM[i, j] * gmu[j]
                            sm = wp.sqrt(wp.max(lamM[i, i],
                                                wp.float64(0.0)))
                        else:
                            for j in range(M):
                                tr += Ons[i, j] * gmu[j]
                            sm = wp.sqrt(wp.max(Ons[i, i],
                                                wp.float64(0.0)))
                    r_p = (Na * (sigma * vals[gp, 2 * i]
                                 - hist[gp, 2 * i] - src[gp, 2 * i])
                           + tr + sm * gq[i]) * dJxW
                    r_m = (Na * (vals[gp, 2 * i + 1] - mub[i]
                                 - src[gp, 2 * i + 1])) * dJxW \
                        - kap[i] * gphi[i] * dJxW
                    wp.atomic_add(be, e, ndof * a + 2 * i, -r_p)
                    wp.atomic_add(be, e, ndof * a + 2 * i + 1, -r_m)
                for k in range(K):
                    rp = 2 * M + 2 * k
                    r_s = (Na * (sigma * vals[gp, rp] - hist[gp, rp]
                                 + Lpsi[k] * Fpsi[k] + qpsi[gp, k]
                                 - src[gp, rp])
                           + Lpsi[k] * eps2[k] * gpsi[k]) * dJxW
                    pk = porv[k] + pfloor
                    if wp.static(TH_FROZEN):
                        r_t = (Na * (sigma * vals[gp, rp + 1]
                                     - hist[gp, rp + 1]
                                     - src[gp, rp + 1])) * dJxW
                    else:
                        r_t = (Na * (pk * (sigma * vals[gp, rp + 1]
                                           - hist[gp, rp + 1])
                                     - src[gp, rp + 1])
                               + pk * ceff[k] * gth[k]) * dJxW
                    wp.atomic_add(be, e, ndof * a + rp, -r_s)
                    wp.atomic_add(be, e, ndof * a + rp + 1, -r_t)
                # Jacobian blocks
                for b in range(nbf):
                    Nb = Ntab[q, b]
                    lap = wp.float64(0.0)
                    for dd in range(dim):
                        lap += dNtab[q, a, dd] * dNtab[q, b, dd] \
                            * dscale * dscale
                    NN = Na * Nb * dJxW
                    lapw = lap * dJxW
                    for i in range(M):
                        ra = ndof * a + 2 * i
                        # phi_i row: time + transport
                        wp.atomic_add(Ae, e, ra, ndof * b + 2 * i,
                                      sigma * NN)
                        if wp.static(FASTMODE):
                            wp.atomic_add(Ae, e, ra, ndof * b + 1,
                                          lam * lapw)
                        else:
                            if wp.static(MATMOB):
                                for j in range(M):
                                    wp.atomic_add(Ae, e, ra,
                                                  ndof * b + 2 * j + 1,
                                                  lamM[i, j] * lapw)
                                # exact mobility-derivative columns:
                                # d/dphi_j int grad(Na).Lam grad(mu)
                                trm = wp.float64(0.0)
                                for j in range(M):
                                    trm += lamM[i, j] * gmu[j]
                                    dv2 = wp.float64(0.0)
                                    for jp in range(M):
                                        dv2 += dLamT[i, jp * M + j] \
                                            * gmu[jp]
                                    wp.atomic_add(Ae, e, ra,
                                                  ndof * b + 2 * j,
                                                  dv2 * Nb * dJxW)
                                for k in range(K):
                                    wp.atomic_add(
                                        Ae, e, ra,
                                        ndof * b + 2 * M + 2 * k,
                                        dlnf[k] * trm * Nb * dJxW)
                            else:
                                for j in range(M):
                                    wp.atomic_add(Ae, e, ra,
                                                  ndof * b + 2 * j + 1,
                                                  Ons[i, j] * lapw)
                        # mu_i row
                        rm = ra + 1
                        wp.atomic_add(Ae, e, rm, ndof * b + 2 * i + 1,
                                      NN)
                        for j in range(M):
                            v = -dmudphi[i, j] * NN
                            if j == i:
                                v -= kap[i] * lapw
                            wp.atomic_add(Ae, e, rm, ndof * b + 2 * j, v)
                        for k in range(K):
                            wp.atomic_add(Ae, e, rm,
                                          ndof * b + 2 * M + 2 * k,
                                          -dmudpsi[i, k] * NN)
                    for k in range(K):
                        rp = 2 * M + 2 * k
                        rs = ndof * a + rp
                        wp.atomic_add(Ae, e, rs, ndof * b + rp,
                                      sigma * NN
                                      + Lpsi[k] * (d2pp[k, k] * NN
                                                   + eps2[k] * lapw))
                        for l in range(K):
                            if l != k:
                                wp.atomic_add(Ae, e, rs,
                                              ndof * b + 2 * M + 2 * l,
                                              Lpsi[k] * d2pp[k, l] * NN)
                        for j in range(M):
                            wp.atomic_add(Ae, e, rs, ndof * b + 2 * j,
                                          Lpsi[k] * d2pf[k, j] * NN)
                        # theta row: d/dtheta + d/dpsi (exact in the
                        # frozen-|g| sense; see docstring)
                        rt = rs + 1
                        pk = porv[k] + pfloor
                        if wp.static(TH_FROZEN):
                            wp.atomic_add(Ae, e, rt, ndof * b + rp + 1,
                                          sigma * NN)
                        else:
                            wp.atomic_add(Ae, e, rt, ndof * b + rp + 1,
                                          pk * sigma * NN
                                          + pk * ceff[k] * lapw)
                            wp.atomic_add(
                                Ae, e, rt, ndof * b + rp,
                                porp[k] * Nb
                                * (Na * (sigma * vals[gp, rp + 1]
                                         - hist[gp, rp + 1])
                                   + ceff[k] * gth[k]) * dJxW)

    _kernel_cache[key] = mpf_k
    return mpf_k


# ---------------------------------------------------------------------
# stepper
# ---------------------------------------------------------------------
class MultiPhaseStepper:
    """(M, K)-generic coupled CH x AC stepper; BDF1 + Newton with the
    house safeguards.  See module docstring for the formulation."""

    def __init__(self, dm, M, K, chi_aa, chi_ac=None, chi_ca=None,
                 chi_cc=None, N=None, onsager=None, kappa=None,
                 dsig=None, dh=None, Tm=None, eps2=None, L_psi=None,
                 alpha_th=None, beta_th=None, L_th=None,
                 T=1.0, T_fn=None, dt=1e-3, bulk="p1", mob="const",
                 D_lo=None, D_hi=None, D_self=None, ls_drop=None,
                 newton_tol=1e-9, newton_max=50, linsolver="splu",
                 noise_psi=0.0, noise_phi=0.0, noise_seed=0,
                 noise_damp=None,
                 b_reg=0.0, kg_delta=1e-3, p_floor=1e-6,
                 dirichlet=None, g_fns=None, src_fns=None, guards=True,
                 clip_psi=True, line_search=False):
        from ..physics.poisson import gauss_points
        self.dm = dm
        self.M, self.K = int(M), int(K)
        assert 1 <= self.M and 0 <= self.K <= self.M
        self.ndof = 2 * self.M + 2 * self.K
        self.Kp = max(self.K, 1)
        n_sp = self.M + 1
        sym = lambda A: np.asarray(A, np.float64).reshape(n_sp, n_sp)
        self.chi_aa = sym(chi_aa)
        z = np.zeros((n_sp, n_sp))
        self.chi_ac = sym(chi_ac) if chi_ac is not None else z.copy()
        self.chi_ca = sym(chi_ca) if chi_ca is not None else \
            self.chi_ac.T.copy()
        self.chi_cc = sym(chi_cc) if chi_cc is not None else z.copy()
        assert np.allclose(self.chi_aa, self.chi_aa.T), "chi_aa symmetric"
        assert np.allclose(self.chi_cc, self.chi_cc.T), "chi_cc symmetric"
        assert np.allclose(self.chi_ca, self.chi_ac.T), \
            "convention: chi_ca = chi_ac.T (chi_ac[i,j] = amorphous i / " \
            "crystalline j)"
        Nv = np.ones(n_sp) if N is None else np.asarray(N, np.float64)
        assert len(Nv) == n_sp
        self.Ninv = 1.0 / Nv
        self.onsager = (np.eye(self.M) if onsager is None
                        else np.asarray(onsager, np.float64
                                        ).reshape(self.M, self.M))
        self.kap = (np.full(self.M, 1e-3) if kappa is None
                    else np.asarray(kappa, np.float64).reshape(self.M))
        pad = lambda v, d: (np.full(self.Kp, d) if v is None else
                            np.concatenate([np.asarray(v, np.float64
                                                       ).reshape(self.K),
                                            np.zeros(self.Kp - self.K)]))
        self.dsig = pad(dsig, 1.0)
        self.dh = pad(dh, 0.0)
        self.Tm = pad(Tm, 1.0)
        self.Tm[self.Tm == 0.0] = 1.0
        self.eps2 = pad(eps2, 1e-4)
        self.L_psi = pad(L_psi, 1.0)
        self.alpha_th = pad(alpha_th, 0.0)
        self.beta_th = pad(beta_th, 0.0)
        self.L_th = pad(L_th, 1.0)
        # KWC coefficients all zero => the theta rows are pure
        # bookkeeping: compile the EXACT frozen form (factory docstring;
        # the degenerate-KWC garbage-dx finding, 2026-07-11)
        self.theta_mode = "frozen" if (self.alpha_th == 0.0).all() \
            and (self.beta_th == 0.0).all() else "kwc"
        self.T, self.T_fn = float(T), T_fn
        self.bulk, self.mob = bulk, mob
        assert bulk in ("p1", "r14")
        assert mob in ("const", "fastmode", "fastmode_n", "slowmode_n")
        self.D_lo = (np.ones(n_sp) if D_lo is None
                     else np.asarray(D_lo, np.float64).reshape(n_sp))
        self.D_hi = (np.ones(n_sp) if D_hi is None
                     else np.asarray(D_hi, np.float64).reshape(n_sp))
        self.D_self = (np.ones((n_sp, n_sp)) if D_self is None
                       else np.asarray(D_self, np.float64
                                       ).reshape(n_sp, n_sp))
        # anchor Eq. 13 drop (d, c, w); d = 1 disables exactly (1^x = 1)
        self.ls_drop = (1.0, 0.5, 1.0) if ls_drop is None \
            else tuple(float(v) for v in ls_drop)
        assert len(self.ls_drop) == 3 and self.ls_drop[0] > 0.0
        self.noise_damp = None if noise_damp is None \
            else tuple(float(v) for v in noise_damp)
        if mob in ("fastmode_n", "slowmode_n"):
            assert noise_phi == 0.0, \
                "CHC noise with a matrix Onsager mobility needs a " \
                "flux-space factorization (not carried; docstring)"
        self.dt = float(dt)
        self.newton_tol, self.newton_max = newton_tol, newton_max
        self.linsolver = linsolver
        self._cudss = None
        self.noise_psi = float(noise_psi)
        self.noise_phi = float(noise_phi)
        self._nrng = np.random.default_rng(noise_seed)
        self.b_reg = float(b_reg)
        self.kg_delta = float(kg_delta)
        self.p_floor = float(p_floor)
        self.dirichlet = dirichlet
        self.g_fns = g_fns
        self.src_fns = src_fns
        # guards=False disables the trust clamp + projection (plain
        # Newton — the S0 parity gate matches the reference stepper's
        # unguarded iteration; production keeps them on)
        self.guards = bool(guards)
        # line_search=True globalizes Newton with ||r||_2 backtracking
        # (S2; _attempt docstring) — default OFF (S0/S1 behavior exact)
        self.line_search = bool(line_search)
        # clip_psi=False skips the [0, 1] psi projection: the r14
        # double well q(psi) is SELF-RESTORING outside [0, 1], and the
        # 2310.11844 replication needs unrectified FDT noise statistics
        # around psi = 0 (a hard clip biases the nucleation rate).  The
        # p1 barrier form REQUIRES the clip (boundary minima).
        self.clip_psi = bool(clip_psi)
        assert clip_psi or bulk == "r14", \
            "clip_psi=False only valid for the self-restoring r14 well"
        self.n_reject = 0
        self.mesh, self.cons = dm.mesh, dm.constraints
        self.Tc = self.cons.T.tocsr()
        self.Tn = sp.kron(self.Tc, sp.identity(self.ndof, format="csr"),
                          format="csr").tocsr()
        self.free_coords = self.mesh.node_coords[self.cons.free_nodes]
        self.xq = gauss_points(self.mesh, dm.tables_by_p)
        self.nfree = self.Tc.shape[1]
        self.t = 0.0
        # quadrature weight x jacobian per GP (noise normalization)
        self._wJ = {}
        for pv, b in dm.bins.items():
            hh = self.mesh.tree.h()[self.mesh.bins[pv]]
            ne = len(self.mesh.conn_of[pv])
            self._wJ[pv] = (np.tile(dm.tables_by_p[pv].w, ne)
                            * np.repeat((hh / 2.0) ** dm.dim, b["nqp"]))
        d = dm.device
        arr = lambda a_: wp.array(np.ascontiguousarray(a_),
                                  dtype=wp.float64, device=d)
        self._par = {k: arr(v) for k, v in dict(
            chi_aa=self.chi_aa, chi_ac=self.chi_ac, chi_ca=self.chi_ca,
            chi_cc=self.chi_cc, Ninv=self.Ninv, Ons=self.onsager,
            dlo=self.D_lo, dhi=self.D_hi, Dslf=self.D_self,
            kap=self.kap, dsig=self.dsig,
            eps2=self.eps2, Lpsi=self.L_psi, alpha=self.alpha_th,
            beta=self.beta_th, Lth=self.L_th).items()}

    # -- initial state ---------------------------------------------------
    def set_initial(self, phi_fns, psi_fns=None, theta_fns=None):
        """phi_fns: list of M callables x -> phi_i; psi_fns/theta_fns:
        lists of K callables (default 0).  mu seeded 0 (the ternary
        brick's convention — S0 parity requires it)."""
        xc = self.free_coords
        self.x = np.zeros(self.nfree * self.ndof)
        for i, fn in enumerate(phi_fns):
            self.x[2 * i::self.ndof] = fn(xc)
        for k in range(self.K):
            if psi_fns is not None and psi_fns[k] is not None:
                self.x[2 * self.M + 2 * k::self.ndof] = psi_fns[k](xc)
            if theta_fns is not None and theta_fns[k] is not None:
                self.x[2 * self.M + 2 * k + 1::self.ndof] = \
                    theta_fns[k](xc)
        self.hist = self.x.copy()
        self.t = 0.0

    def _gp(self, vec):
        full = np.asarray(self.Tc @ vec)
        v, g = {}, {}
        for pv, b in self.dm.bins.items():
            tb = self.dm.tables_by_p[pv]
            conn = self.mesh.conn_of[pv]
            vals = full[conn]
            v[pv] = np.einsum("qa,ea->eq", tb.N, vals).reshape(-1)
            hh = self.mesh.tree.h()[self.mesh.bins[pv]]
            g[pv] = np.stack(
                [(np.einsum("qa,ea->eq", tb.dN[:, :, d], vals)
                  * (2.0 / hh)[:, None]).reshape(-1)
                 for d in range(self.dm.dim)], axis=1)
        return v, g

    def _pack_fields(self, x):
        """[ngp, ndof] values and [ngp, ndof, dim] gradients per bin."""
        vs, gs = [], []
        for f in range(self.ndof):
            v, g = self._gp(x[f::self.ndof])
            vs.append(v)
            gs.append(g)
        vals, grads = {}, {}
        for pv in vs[0]:
            vals[pv] = np.stack([vs[f][pv] for f in range(self.ndof)],
                                axis=1)
            grads[pv] = np.stack([gs[f][pv] for f in range(self.ndof)],
                                 axis=1)
        return vals, grads

    def _project(self, x):
        """Projected iterates: phi simplex-aware clip, psi in [0, 1]."""
        lo, hi = 1e-3, 1.0 - 1e-3
        phis = [x[2 * i::self.ndof] for i in range(self.M)]
        for p in phis:
            np.clip(p, lo, None, out=p)
        s = sum(phis)
        bad = s > hi
        if np.any(bad):
            scale = hi / s[bad]
            for p in phis:
                p[bad] *= scale
        if self.clip_psi:
            for k in range(self.K):
                np.clip(x[2 * self.M + 2 * k::self.ndof], 0.0, 1.0,
                        out=x[2 * self.M + 2 * k::self.ndof])
        return x

    def _drive(self, t_new):
        T = self.T_fn(t_new) if self.T_fn is not None else self.T
        if self.bulk == "r14":
            return self.dh * (T / self.Tm - 1.0)
        return self.dh * (1.0 - T / self.Tm)

    def _solve(self, A, r):
        if self.linsolver == "cudss":
            from nvmath.sparse.advanced import (DirectSolver,
                                                DirectSolverOptions)
            b = np.ascontiguousarray(r, np.float64)
            try:
                # the scipy T^T K T triple product PRUNES exact zeros, so
                # the CSR pattern GROWS when a field leaves its uniform
                # initial state (measured: 884736 -> 1179648 nnz when
                # psi-noise switches on the coupling blocks at iterate 1)
                # — reset_operands demands a fixed pattern; rebuild the
                # plan on any nnz change instead of failing the attempt.
                if self._cudss is not None and \
                        getattr(self, "_cudss_nnz", -1) != A.nnz:
                    # EXPLICITLY release the old plan: leaving it to the
                    # GC finalizer double-frees device buffers under
                    # nnz-flapping noise runs (measured 2026-07-10:
                    # intermittent CUDA 700 + glibc free() corruption in
                    # the S2c calibration marches)
                    try:
                        self._cudss.free()
                    except Exception:
                        pass
                    self._cudss = None
                if self._cudss is None:
                    # PLAIN options (no libcudss_mtlayer_gomp): the mt
                    # planning layer leaks one gomp thread team per
                    # solver call (MEASURED 2026-07-10: 18.6k native
                    # threads after ~20k solves -> libgomp 'Thread
                    # creation failed' killed three marches).  The 2-D
                    # S2 systems plan in 65-340 ms single-threaded —
                    # the M3-bunny mt-planning speedup is a 3-D
                    # large-matrix concern, not ours.
                    self._cudss = DirectSolver(
                        A, b, options=DirectSolverOptions(blocking=True))
                    self._cudss.plan()
                    self._cudss_nnz = A.nnz
                else:
                    self._cudss.reset_operands(a=A, b=b)
                self._cudss.factorize()
                return np.asarray(self._cudss.solve())
            except Exception:
                try:
                    self._cudss.free()
                except Exception:
                    pass
                self._cudss = None
                return np.full(A.shape[0], np.nan)
        from scipy.sparse.linalg import splu
        return splu(A.tocsc()).solve(r)

    # -- one implicit BDF1 solve at frozen dt; does NOT commit -----------
    def _attempt(self, dt):
        d = self.dm.device
        nd = self.ndof
        sigma = 1.0 / dt
        t_new = self.t + dt
        drive = np.concatenate([self._drive(t_new)[:self.K],
                                np.zeros(self.Kp - self.K)]) \
            if self.K else np.zeros(self.Kp)
        arr = lambda a_: wp.array(np.ascontiguousarray(a_),
                                  dtype=wp.float64, device=d)
        drive_d = arr(drive)
        hv, _ = self._pack_fields(self.hist)
        hist_gp = {pv: sigma * hv[pv] for pv in hv}
        for pv in hist_gp:          # mu rows carry no time derivative
            for i in range(self.M):
                hist_gp[pv][:, 2 * i + 1] = 0.0
        # FDT noise (frozen per attempt, wJ-normalized; docstring)
        qpsi_gp, qphi_gp = {}, {}
        for pv, b in self.dm.bins.items():
            ngp = len(self.mesh.conn_of[pv]) * b["nqp"]
            wJ = self._wJ[pv]
            qpsi = np.zeros((ngp, self.Kp))
            if self.noise_psi > 0.0 and self.K:
                for k in range(self.K):
                    qpsi[:, k] = (self.noise_psi
                                  * np.sqrt(2.0 * self.L_psi[k]
                                            / (dt * wJ))
                                  * self._nrng.standard_normal(ngp))
                    if self.noise_damp is not None:
                        dda, ddc, ddw = self.noise_damp
                        qpsi[:, k] *= np_ls_interp(
                            hv[pv][:, 2 * self.M + 2 * k], dda, ddc, ddw)
            qphi = np.zeros((ngp, self.M, self.dm.dim))
            if self.noise_phi > 0.0:
                qphi = (self.noise_phi
                        * np.sqrt(2.0 / (dt * wJ))[:, None, None]
                        * self._nrng.standard_normal(
                            (ngp, self.M, self.dm.dim)))
            qpsi_gp[pv], qphi_gp[pv] = qpsi, qphi
        # MMS sources at t_new
        src_gp = {}
        for pv, b in self.dm.bins.items():
            ngp = len(self.mesh.conn_of[pv]) * b["nqp"]
            s = np.zeros((ngp, nd))
            if self.src_fns is not None:
                for f, fn in enumerate(self.src_fns):
                    if fn is not None:
                        s[:, f] = fn(self.xq[pv], t_new)
            src_gp[pv] = s
        def assemble(x):
            vals, grads = self._pack_fields(x)
            rows, cols, valsK = [], [], []
            F_full = np.zeros(self.dm.n_nodes * nd)
            for pv, b in self.dm.bins.items():
                conn = self.mesh.conn_of[pv].astype(np.int64)
                ne, nbf = conn.shape
                nqp = b["nqp"]
                Ae = wp.zeros((ne, nd * nbf, nd * nbf),
                              dtype=wp.float64, device=d)
                be = wp.zeros((ne, nd * nbf), dtype=wp.float64,
                              device=d)
                kk = make_mpf_newton(nbf, nqp, self.dm.dim, self.M,
                                     self.K, self.bulk, self.mob,
                                     self.theta_mode)
                p = self._par
                wp.launch(kk, dim=ne, inputs=[
                    b["conn"], b["h"], b["N"], b["dN"], b["w"],
                    arr(vals[pv]), arr(grads[pv]), arr(hist_gp[pv]),
                    arr(src_gp[pv]), arr(qpsi_gp[pv]), arr(qphi_gp[pv]),
                    p["chi_aa"], p["chi_ac"], p["chi_ca"], p["chi_cc"],
                    p["Ninv"], p["Ons"], p["dlo"], p["dhi"], p["Dslf"],
                    wp.float64(self.ls_drop[0]),
                    wp.float64(self.ls_drop[1]),
                    wp.float64(self.ls_drop[2]), p["kap"],
                    p["dsig"], drive_d, p["eps2"], p["Lpsi"],
                    p["alpha"], p["beta"], p["Lth"],
                    wp.float64(sigma), wp.float64(self.b_reg),
                    wp.float64(self.kg_delta), wp.float64(self.p_floor),
                    Ae, be], device=d)
                Aeh, beh = Ae.numpy(), be.numpy()
                gdof = (conn[:, :, None] * nd
                        + np.arange(nd)[None, None, :]
                        ).reshape(ne, nd * nbf)
                rows.append(np.repeat(gdof, nd * nbf, axis=1).ravel())
                cols.append(np.tile(gdof, (1, nd * nbf)).ravel())
                valsK.append(Aeh.ravel())
                np.add.at(F_full, gdof.ravel(), beh.ravel())
            Kmat = sp.coo_matrix(
                (np.concatenate(valsK),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(self.dm.n_nodes * nd,) * 2).tocsr()
            A = (self.Tn.T @ Kmat @ self.Tn).tocsr()
            r = np.asarray(self.Tn.T @ F_full)
            if self.dirichlet is not None:
                A = A.tolil()
                for f, gfn in enumerate(self.g_fns):
                    if gfn is None:
                        continue
                    gv = gfn(self.free_coords[self.dirichlet], t_new)
                    for k2, i in enumerate(self.dirichlet):
                        rr = i * nd + f
                        A.rows[rr] = [int(rr)]
                        A.data[rr] = [1.0]
                        r[rr] = gv[k2] - x[rr]
                A = A.tocsr()
            return A, r

        x = self.x.copy()
        A, r = assemble(x)
        for it in range(self.newton_max):
            dx = self._solve(A, r)
            if not np.isfinite(dx).all():
                return None, it + 1, False               # diverged
            if self.line_search:
                # globalized path: NEVER abort on a finite oversized
                # direction — near-degenerate blocks (an inert
                # crystallinity/orientation row whose residual is
                # roundoff-zero) yield garbage-SCALED directions
                # (measured 2026-07-11: |dx| = 4.9e13 at |r| = 1e-19,
                # dt-INDEPENDENT; the hard 1e6 guard then underflowed
                # the ladder).  Rescale to the trust size and let the
                # backtracking judge; a zero-residual state then
                # correctly CONVERGES on the applied-increment
                # criterion instead of dying.
                mx = np.abs(dx).max()
                if mx > 2.0:
                    dx = dx * (2.0 / mx)
            elif np.abs(dx).max() > 1e6:
                return None, it + 1, False               # diverged
            if self.guards:
                # trust clamp on the physical (phi, psi) increments
                inc = max(np.abs(dx[2 * i::nd]).max()
                          for i in range(self.M))
                for k in range(self.K):
                    inc = max(inc,
                              np.abs(dx[2 * self.M + 2 * k::nd]).max())
                if inc > 2.0:
                    dx = dx * (2.0 / inc)
            if self.line_search:
                # backtracking on ||r||_2 (globalized Newton; the
                # deep-quench SD purification lesson — plain Newton
                # diverges at dt > ~2e-4 while the damped direction
                # converges at the physical dt): accept the first
                # fraction that reduces the residual norm; the assembly
                # at the accepted iterate is REUSED for the next solve,
                # so an accepted full step costs exactly one assembly
                # (the undamped path's cost).
                rn0 = float(np.linalg.norm(r))
                best, best_rn = None, np.inf
                for s in (1.0, 0.5, 0.25, 0.125, 0.0625):
                    xt = x + s * dx
                    if self.guards:
                        xt = self._project(xt)
                    At, rt = assemble(xt)
                    rnt = float(np.linalg.norm(rt))
                    if rnt < best_rn:
                        best, best_rn = (xt, At, rt), rnt
                    if rnt < rn0 * (1.0 - 1e-4):
                        break
                xn, A, r = best
                conv = np.abs(xn - x).max()
                x = xn
            elif self.guards:
                xn = self._project(x + dx)
                # projected-Newton convergence: the APPLIED increment.
                # A p1-mode boundary well pins psi at the [0, 1] clip
                # with a nonzero raw dx forever (measured: dt_underflow
                # on the seeded-disc gate); stationarity of the
                # projected iterate is the correct criterion there.
                conv = np.abs(xn - x).max()
                x = xn
                A, r = assemble(x)
            else:
                x = x + dx
                conv = np.abs(dx).max()
                A, r = assemble(x)
            if conv < self.newton_tol:
                return x, it + 1, True
        return x, self.newton_max, False                 # no convergence

    def step(self):
        """One FIXED-dt BDF1 step (gates/MMS/parity path — no ladder)."""
        x, iters, ok = self._attempt(self.dt)
        assert ok, f"Newton failed at fixed dt (iters={iters})"
        self.x = x
        self.hist = x.copy()
        self.t += self.dt
        return x

    def march(self, t_end, max_steps=100000, dt_min=1e-12, dt_max=None,
              callback=None, grow_iters=20):
        """Appendix-A ladder (wodo pattern, no evaporation): reject
        (no convergence/divergence) => dt *= 0.25 retry; accept with
        iters < grow_iters => dt *= 1.25 (capped).  grow_iters default
        20 (the wodo constant); KWC grain-boundary states carry the
        KG-Picard LINEAR Newton tail (measured 0.5-0.9 contraction),
        where 20 starves dt growth — raise it there.  Returns stop
        reason."""
        reason = "max_steps"
        for _ in range(max_steps):
            if self.t >= t_end - 1e-14:
                reason = "t_end"
                break
            dt_eff = min(self.dt, t_end - self.t)
            x_new, iters, ok = self._attempt(dt_eff)
            if not ok:
                self.n_reject += 1
                self.dt = dt_eff * 0.25
                if self.dt < dt_min:
                    reason = "dt_underflow"
                    break
                continue
            self.x = x_new
            self.hist = x_new.copy()
            self.t += dt_eff
            if iters < grow_iters:
                self.dt = dt_eff * 1.25
                if dt_max is not None:
                    self.dt = min(self.dt, dt_max)
            else:
                self.dt = dt_eff
            if callback is not None:
                callback(self, dt_eff, iters)
        return reason

    # convenient views ---------------------------------------------------
    def phi(self, i):
        return self.x[2 * i::self.ndof]

    def mu(self, i):
        return self.x[2 * i + 1::self.ndof]

    def psi(self, k):
        return self.x[2 * self.M + 2 * k::self.ndof]

    def theta(self, k):
        return self.x[2 * self.M + 2 * k + 1::self.ndof]


# ---------------------------------------------------------------------
# grain identification (analysis-side, numpy)
# ---------------------------------------------------------------------
def grain_labels(coords, psi, theta, psi_th=0.5, theta_tol=0.3,
                 periodic=False):
    """Theta-watershed crystal labeling on a UNIFORM 2-D node grid.
    Nodes with psi > psi_th are crystalline; two neighboring (4-conn)
    crystalline nodes join the same grain iff |theta_a - theta_b|
    < theta_tol (theta plateaus label individual crystals; boundaries
    show as theta jumps and/or psi dips).  Returns (labels, sizes):
    labels int array over nodes (-1 = amorphous, 0..G-1 grains),
    sizes = node counts per grain, descending."""
    coords = np.asarray(coords)
    psi = np.asarray(psi)
    theta = np.asarray(theta)
    xs = np.unique(np.round(coords[:, 0], 12))
    ys = np.unique(np.round(coords[:, 1], 12))
    nx, ny = len(xs), len(ys)
    assert nx * ny == len(coords), "grain_labels: uniform grid only"
    ix = np.searchsorted(xs, np.round(coords[:, 0], 12))
    iy = np.searchsorted(ys, np.round(coords[:, 1], 12))
    node_of = -np.ones((nx, ny), np.int64)
    node_of[ix, iy] = np.arange(len(coords))
    parent = np.arange(len(coords))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    cry = psi > psi_th
    pairs = []
    g = node_of
    pairs.append((g[:-1, :].ravel(), g[1:, :].ravel()))
    pairs.append((g[:, :-1].ravel(), g[:, 1:].ravel()))
    if periodic:
        pairs.append((g[-1, :].ravel(), g[0, :].ravel()))
        pairs.append((g[:, -1].ravel(), g[:, 0].ravel()))
    for aa, bb in pairs:
        m = (aa >= 0) & (bb >= 0)
        aa, bb = aa[m], bb[m]
        ok = (cry[aa] & cry[bb]
              & (np.abs(theta[aa] - theta[bb]) < theta_tol))
        for a, b in zip(aa[ok], bb[ok]):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    labels = -np.ones(len(coords), np.int64)
    roots = {}
    for n in np.where(cry)[0]:
        r = find(n)
        if r not in roots:
            roots[r] = len(roots)
        labels[n] = roots[r]
    sizes = np.bincount(labels[labels >= 0], minlength=len(roots)) \
        if roots else np.zeros(0, np.int64)
    order = np.argsort(sizes)[::-1]
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    labels[labels >= 0] = remap[labels[labels >= 0]]
    return labels, np.sort(sizes)[::-1]
