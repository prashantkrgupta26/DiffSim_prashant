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

M5 (a)-PACK REALISM EXTENSIONS (Track A of flow_film_formulation_p2;
dev ledger docs/dev/2026-07-12-m5-apack.md):

A1 — TEMPERATURE AS A FIELD (T_mode="field").  Nodal T(x, t) solved as
a SEGREGATED linear step per attempt (TemperatureField below): the
annealing stage has no flow, so rho_cp dT/dt = div(k_th grad T) + s
with Dirichlet substrate/ambient BCs (annealing protocols become BCs).
Lie (first-order) operator split per BDF1 step: T advances first on
[t, t+dt] from the committed field, then the crystallization step
reads T^{n+1} at quadrature points — splitting error O(dt), the SAME
order as BDF1 itself, so no order degradation (recorded choice).  In
field mode the kernel computes the crystal driving PER GP:
  p1:  drive_k(x) = dh_k (1 - T(x)/Tm_k)
  r14: drive_k(x) = dh_k (T(x)/Tm_k - 1)
(the host passes dh in the drive slot + Tm and the GP-interpolated T).
T_mode="scalar" (default) keeps the EXISTING host-side scalar drive
path bit-identically (the field inputs are dummy, compile-time dead).
chi(T) is NOT parameterized in this brick (chi matrices are constant
inputs); when a chi(T) law lands it must read the same per-GP Tq array
(plumbing exists).  FDT noise amplitude stays the global calibration
knob (anchor Sec 2.2.1); spatially-varying kBT(x) noise is a recorded
frontier.  D(T) HOOK (D_T=(Ea, T_ref), either T mode): multiplies the
CH mobility by the Arrhenius factor exp(-Ea (1/T - 1/T_ref)) at each
GP (fastmode lam, the _n-mode Vignes D_i, const-mob Ons) — mobility
freeze-out across a T gradient.  T-independent AC prefactor M_psi(T)
(WLF) is a recorded follow-up.

A2 — SUBSTRATE SURFACE ENERGY (wall_g/wall_h, wall_face=(axis, side)).
Wall free energy F_w = Int_{Gamma_w} f_w(phi) dS,
f_w = SUM_i g_i phi_i + h_i phi_i^2 over RETAINED species (the
eliminated solvent carries zero wall energy by convention; a solvent
preference is the exchange shift g_i -> g_i - g_s).  Variation adds
the NATURAL boundary term to the mu_i equation: the total-variation
boundary condition is kap_i dphi_i/dn = -f_w'(phi_i) on Gamma_w, so
the mu_i residual gains  - Int_w N_a (g_i + 2 h_i phi_i) dS
(consistent basis-generic face mass, host-assembled — the wodo
top-flux pattern;
Jacobian block -2 h_i Mw on (mu_i row, phi_i col)).  Default face
(1, 0) = the y = 0 substrate edge; face-generic by construction.
g_i < 0 attracts species i to the wall (f_w decreases with phi_i).

A3 — ANISOTROPIC CRYSTAL GROWTH (delta_a, m_a, a_reg; 2-D).  The psi
gradient energy becomes  f_grad,k = (1/2) a_k(alpha)^2 |grad psi_k|^2
with  a_k = eps_k (1 + delta_a,k cos(m_a,k (alpha - theta_k))),
alpha = angle(grad psi_k), theta_k the EXISTING orientation field
(frozen-marker mode REQUIRED — asserted; the KWC back-torque
d f_grad/d theta on a DYNAMIC theta is a recorded extension).  The
standard anisotropic Allen-Cahn variational flux (2-D):
  q_k = a^2 grad psi_k + a a' R grad psi_k,   R g = (-g_y, g_x),
  a' = d a/d alpha = -eps delta m sin(m (alpha - theta))
(the d(eps^2)/d alpha rotation contribution).  |grad psi| -> 0 guard:
delta_eff = delta_a g2/(g2 + a_reg^2) (smooth blend; atan2 itself is
finite).  Coefficients (a^2, a a') are PICARD-FROZEN per iterate (the
house KG pattern): residual exact, Jacobian quasi-Newton.  delta_a = 0
compiles the ISOTROPIC kernel (same factory key) — exact regression.

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
schedule hook (annealing protocols); drive_k is refreshed per attempt
— OR a nodal field (T_mode="field", A1 above): the segregated
TemperatureField advances first within each attempt and commits only
with the accepted step.

S3a — FILM MODE (film=dict(...); the Landau-mapped moving frame of
wodo_film brought into the (M, K) factory; formulation contract P1
Sec 4 stage S3: psi/theta ride the frame EXACTLY as phi — mapped
gradients + frame advection on every field; the top-flux term is
phi-only, solvent evaporates AMORPHOUS).  The computational strip
[0, Xcomp] x [0, Ycomp] represents a physical domain
(lat_scale Xcomp) x h_curr(t); vertical = LAST axis (vax = dim-1),
theta_map = xi_y / Ycomp, physical z = h_curr theta_map.  Mapped
gradient factors (applied to test AND trial/field gradients — 1/h^2
on vertical grad.grad blocks): mlat = 1/lat_scale,
mvert = Ycomp/h_curr.  The wodo v1.1 generalized-metric convention is
kept verbatim: the kernel works in RAW computational coordinates, the
advection coefficient is K xi_y (1/h) on the raw xi_y-derivative
(Ycomp cancels), and volume/boundary integrals carry computational
measure (the mu rows are the physical rows divided by the constant-
in-space ratio lat_scale h/Ycomp — same solutions; boundary naturals
pick up Ycomp/h, see below).  WEAK FORM ADDITIONS (test v; per
retained species i, crystallizable k; term-to-code labels inline):

  R_phi_i += Int v (K xi_y/h) d(phi_i)/dxi_y dV        (frame advection)
           - ((K - k_e_i)/h) Ycomp Int_top v phi_i dS  (enrichment flux)
  R_psi_k += Int v (K xi_y/h) d(psi_k)/dxi_y dV        (psi advects as phi;
                                                        NO top flux — the
                                                        S3 contract)
  R_th_k  += Int v (p+pf) (K xi_y/h) d(th_k)/dxi_y dV  (KWC; frozen mode
                                                        drops the (p+pf))

with K = SUM_i k_e_i avg(phi_i^top) >= 0 frozen at t_n per attempt
(k_e a FULL-species vector; scalar input = the eliminated solvent's
rate, the exact wodo Bi semantics — S3 configs translate 1:1; per-
RETAINED-species k_e_i > 0 is the S4a hook, carried but not gated:
its avg-vs-pointwise flux closure is recorded there), and
h_curr -= dt K on ACCEPTED steps (explicit, wodo contract).  The
enrichment-flux/advection sign pairing conserves the physical solute
content h Int phi_i dtheta EXACTLY per BDF1 step (wodo SIGN NOTE;
partition of unity + the implicit-advection/explicit-h cancellation);
the psi analogue h Int psi_k dtheta is conserved exactly while
psi^top = 0 (pure advection).  Under BDF2 the pairing is no longer
telescoping-exact: the content drift is the BDF2 global error on
P' = (K/h) P (O(dt^2), measured in the S3 gates — a discretization-
order drift, not a leak).  march() gains the evaporation dt-cap
dt <= dh_cap/K and h_min/phis_stop criteria.  The A2 wall energy
stays on the SUBSTRATE face (assert: not the moving face) and its
natural term picks up the mapped boundary measure factor Ycomp/h.
FDT/CHC noise normalization stays in COMPUTATIONAL measure (the
amplitude is the anchor's calibration knob — recorded).  A1 T-field
in film mode: TemperatureField splits the stiffness into lateral +
vertical blocks scaled per attempt by mlat^2/mvert^2, adds the frame
advection rho_cp (K/h) Int N_a xi_y dN_b/dxi_y, and the EVAPORATIVE
COOLING natural load  - L_vap K (Ycomp/h) Int_top N_a dS  (latent
heat sink at the receding surface; L_vap = 0 default OFF).  film
dict keys: k_e (scalar | (M+1)-vector), h0, lat_scale, dh_cap, K_fn
(manufactured-frame override for MMS: K(t) callable, replaces the
k_e closure).

TIME SCHEME (A4b).  tstep="bdf1" (default, existing behavior
bit-identically) | "bdf2": VARIABLE-STEP BDF2 with the standard
variable coefficients — for step ratio r = dt_n/dt_{n-1},

  [ a x^{n+1} - b x^n + c x^{n-1} ] / dt_n = f(x^{n+1}, t^{n+1}),
  a = (1+2r)/(1+r),   b = 1+r,   c = r^2/(1+r)   (a = b - c),

first step bootstrapped with BDF1 (no x^{n-1}).  BDF2 rides the SAME
kernel: the kernel's time term is sigma*x - hist per GP, so the host
passes sigma = a/dt_n and hist = (b x^n - c x^{n-1})/dt_n — no kernel
change, and the frozen-theta bookkeeping row remains the EXACT
identity (equal histories give theta^{n+1} = theta^n since a = b - c).
LADDER CONSISTENCY: rejected attempts never mutate the history, and
the coefficients are recomputed from the ACTUAL (dt_n, dt_{n-1}) per
attempt, so dt rescaling on rejects is automatically consistent.
SCOPE LIMIT (recorded): BDF2 is DETERMINISTIC-ONLY (asserted noise
off) — the FDT-noise weak order under BDF2 is out of scope.
Linear solve: splu (default) | cudss (wodo DirectSolver pattern) |
blockch / blockch_dev (B-track): the G4 per-pair two-factor Schur
preconditioner extended to the full (M, K) system — M CH pairs get
W1/W2 (pair m from _mob_ref, the committed-mean mirror of the kernel
mobility; kappa per pair), K AC (psi, theta) blocks get extracted
diagonal-block solves with the theta<-psi torque carried lower-
triangularly (linsolve "pairs"/"ac" meta; _blockch_meta).  blockch_dev
= device inners.  With assembly="device" both route to
blockch_pairs_device (zero-copy on the slot-map CSR).  Solver failure
(two-factor AND exact-Schur escalation) signals divergence to the
reject ladder, matching cudss.

DEVICE ASSEMBLY (D-track).  assembly="host" (default) keeps the scipy
COO -> CSR -> T^T K T finalization bit-identically; assembly="device"
replaces it with the wodo v1.2 slot-map pattern (DeviceNSAssembler,
ndof = 2M + 2K generic, nbf/nqp from the factory tabulation — p = 1
and p = 2 ride the same machinery): the CSR pattern is built ONCE per
mesh from the element graph, each Newton iterate evaluates the GP
fields ON DEVICE (gp_multifield — the _pack_fields mirror), launches
the SAME mpf kernel in element batches (Ae transient capped ~2 GB) and
atomic-scatters the blocks straight into the device CSR values; the
A2 wall / S3a top-flux natural face terms keep host-computed O(surface)
values slot-added per iterate (the wodo documented hybrid), Dirichlet
rows are applied in-kernel (strong-row plan), and the solve consumes
the buffers zero-copy (cudss: plan-once + refactorize on the FIXED
pattern — nnz never flaps, curing the S2 plan-flapping; plain options,
explicit .free()).  Identity constraints REQUIRED (all uniform meshes,
periodic included — asserted); adapted meshes stay on the host path.
See _init_device_assembly for the weak-form -> scatter identity map.
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
                    theta: str = "kwc", tfield: bool = False,
                    dth: bool = False, aniso: bool = False,
                    film: bool = False):
    """Monolithic Newton kernel for the 2M+2K node-major system.
    (M, K, bulk, mob, theta, tfield, dth, aniso) compile-time; see
    module docstring.
    theta="frozen" replaces the KWC theta rows by the EXACT bookkeeping
    identity theta = theta_old (mass-matrix row) — the anchor's marker
    semantics, and the well-posed form of the alpha = beta = 0 limit:
    the degenerate KWC row (diag ~ p_floor sigma NN ~ 1e-9 with
    roundoff coupling entries) makes the direct solve return garbage
    theta increments (MEASURED 2026-07-11: |dx_theta| 4.9e13 at
    |r_theta| 1e-19, dt-INDEPENDENT — the divergence guard then
    underflows the ladder; whether the garbage crosses the 1e6 guard
    is an assembly-atomics coin flip, the house FP-fate lesson).
    tfield=True reads the per-GP temperature Tq and computes the
    crystal driving in-kernel (A1); dth=True applies the Arrhenius
    D(T) mobility factor per GP (A1 hook); aniso=True compiles the
    anisotropic psi gradient flux (A3; dim = 2 + frozen theta only).
    film=True compiles the Landau-mapped moving-frame terms (S3a,
    module docstring): mapped gradient factors mlat/mvert on test AND
    field gradients, and the frame advection kadv xiy madv d/dxi_y on
    the phi/psi/theta rows (vertical = dim-1).
    All flags default False = the pre-A-pack kernel bit-identically."""
    key = ("mpf_newton", nbf, nqp, dim, M, K, bulk, mob, theta,
           tfield, dth, aniso, film)
    if key in _kernel_cache:
        return _kernel_cache[key]
    assert bulk in ("p1", "r14"), bulk
    assert mob in ("const", "fastmode", "fastmode_n", "slowmode_n"), mob
    if mob == "fastmode":
        assert M == 1, "fastmode Onsager closure: M = 1 only " \
            "(fastmode_n is the generic-M mode)"
    assert theta in ("kwc", "frozen"), theta
    if aniso:
        assert dim == 2, "anisotropic growth: 2-D only (A3 contract)"
        assert theta == "frozen", \
            "anisotropy requires marker (frozen) theta: the KWC " \
            "back-torque d f_grad/d theta is a recorded extension"
    TH_FROZEN = theta == "frozen"
    TFIELD = bool(tfield)
    DTH = bool(dth)
    ANISO = bool(aniso)
    FILM = bool(film)
    vax = dim - 1               # film vertical = LAST axis (wodo)
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
              Tmv: wp.array(dtype=wp.float64),       # [Kp] Tm (tfield law)
              Tq: wp.array(dtype=wp.float64),        # [ngp] T at GPs
              ea: wp.float64, tref: wp.float64,      # D(T) Arrhenius
              da: wp.array(dtype=wp.float64),        # [Kp] aniso delta_a
              ma: wp.array(dtype=wp.float64),        # [Kp] aniso m-fold
              areg: wp.float64,                      # aniso |g| guard
              xiy: wp.array(dtype=wp.float64),       # [ngp] film xi_y
              mlat: wp.float64, mvert: wp.float64,   # film metric (S3a)
              madv: wp.float64, kadv: wp.float64,    # film 1/h, K
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
            # S3a film frame: advection coefficient on the RAW xi_y
            # derivative, K * xi_y * (1/h) (wodo pattern: the Ycomp
            # factors cancel; module docstring S3a)
            adv = wp.float64(0.0)
            if wp.static(FILM):
                adv = kadv * xiy[gp] * madv
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
                # A1 field mode: the crystal driving is a FIELD quantity
                # — drive slot carries dh_k, the T law applies per GP
                # (scalar mode: drive[k] is host-precomputed, unchanged)
                drv = drive[k]
                if wp.static(TFIELD):
                    if wp.static(R14):
                        drv = drive[k] * (Tq[gp] / Tmv[k]
                                          - wp.float64(1.0))
                    else:
                        drv = drive[k] * (wp.float64(1.0)
                                          - Tq[gp] / Tmv[k])
                if wp.static(R14):
                    om = wp.float64(1.0) - s_
                    Wv[k] = s_ * s_ * om * om * dsig[k] \
                        + s_ * s_ * (wp.float64(3.0)
                                     - wp.float64(2.0) * s_) * drv
                    Wpv[k] = wp.float64(2.0) * s_ * om \
                        * (wp.float64(1.0) - wp.float64(2.0) * s_) \
                        * dsig[k] + wp.float64(6.0) * s_ * om * drv
                    Wppv[k] = wp.float64(2.0) \
                        * (wp.float64(1.0) - wp.float64(6.0) * s_
                           + wp.float64(6.0) * s_ * s_) * dsig[k] \
                        + (wp.float64(6.0) - wp.float64(12.0) * s_) \
                        * drv
                else:
                    Wv[k] = s_ * (wp.float64(1.0) - s_) * dsig[k] \
                        + s_ * s_ * drv
                    Wpv[k] = (wp.float64(1.0) - wp.float64(2.0) * s_) \
                        * dsig[k] + wp.float64(2.0) * s_ * drv
                    Wppv[k] = wp.float64(-2.0) * dsig[k] \
                        + wp.float64(2.0) * drv
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
                    if wp.static(FILM):
                        # mapped |grad~ theta| (S3a): 1/h on vertical
                        if dd == vax:
                            gtd = gtd * mvert
                        else:
                            gtd = gtd * mlat
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
            # ---- anisotropic psi gradient coefficients (A3) -------------
            # f_grad,k = (1/2) a_k(alpha)^2 |grad psi_k|^2 with
            # a_k = eps_k (1 + d cos(m (alpha - theta_k))), alpha =
            # angle(grad psi_k); variational flux (weak: Int grad(N_a)
            # . q_k):  q_k = a^2 grad psi + (a a') R grad psi,
            # R g = (-g_y, g_x),  a a' = -eps2 (1 + d c) d m s.
            # |grad psi| -> 0 guard: d_eff = d g2/(g2 + areg^2).
            # PICARD-FROZEN coefficients (module docstring A3).
            A2k = VecKp()     # a(alpha)^2            (isotropic: eps2)
            AAk = VecKp()     # a a'                  (isotropic: 0)
            Rgx = VecKp()     # (R grad psi)_x = -psi_y
            Rgy = VecKp()     # (R grad psi)_y = +psi_x
            for k in range(Kp):
                A2k[k] = eps2[k]
                AAk[k] = wp.float64(0.0)
                Rgx[k] = wp.float64(0.0)
                Rgy[k] = wp.float64(0.0)
            if wp.static(ANISO):
                for k in range(K):
                    gpx = grads[gp, 2 * M + 2 * k, 0]
                    gpy = grads[gp, 2 * M + 2 * k, 1]
                    if wp.static(FILM):
                        # anisotropy acts on the PHYSICAL gradient
                        # (S3a: mapped components; dim=2, vax=1)
                        gpx = gpx * mlat
                        gpy = gpy * mvert
                    g2p = gpx * gpx + gpy * gpy
                    deff = da[k] * g2p / (g2p + areg * areg)
                    ang = ma[k] * (wp.atan2(gpy, gpx)
                                   - vals[gp, 2 * M + 2 * k + 1])
                    fac = wp.float64(1.0) + deff * wp.cos(ang)
                    A2k[k] = eps2[k] * fac * fac
                    AAk[k] = -eps2[k] * fac * deff * ma[k] \
                        * wp.sin(ang)
                    Rgx[k] = -gpy
                    Rgy[k] = gpx
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
            # A1 D(T) hook: Arrhenius factor mT = exp(-Ea (1/T - 1/Tref))
            # multiplies the CH mobility (per GP; T from the field or
            # the scalar schedule — the host fills Tq either way when
            # the hook is on).  mT is phi/psi-INDEPENDENT, so all the
            # exact dLam/dphi, dLam/dpsi blocks below scale through it
            # unchanged (homogeneity).
            mT = wp.float64(1.0)
            if wp.static(DTH):
                mT = wp.exp(ea * (wp.float64(1.0) / tref
                                  - wp.float64(1.0) / Tq[gp]))
            lam = wp.float64(0.0)
            if wp.static(FASTMODE):
                pc = wp.min(wp.max(phiv[0], wp.float64(1e-6)),
                            wp.float64(1.0) - wp.float64(1e-6))
                omp = wp.float64(1.0) - pc
                d1s = wp.pow(dlo[0], omp) * wp.pow(dhi[0], pc)
                d2s = wp.pow(dlo[1], pc) * wp.pow(dhi[1], omp)
                lam = omp * omp * pc / Ninv[0] * d1s \
                    + pc * pc * omp / Ninv[1] * d2s
                if wp.static(DTH):
                    lam *= mT
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
                    if wp.static(DTH):
                        dsi = fdrop * mT      # Vignes D_i x Arrhenius
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
                grot = VecKp()   # grad Na . R grad psi_k (aniso)
                for k in range(Kp):
                    grot[k] = wp.float64(0.0)
                for dd in range(dim):
                    gNa = dNtab[q, a, dd] * dscale
                    if wp.static(FILM):
                        # mapped gradient grad~ (S3a): the factor ms
                        # rides on BOTH the test and the field gradient
                        # (1/h^2 on vertical grad.grad blocks); the
                        # stochastic flux q is a physical flux — test
                        # side only (the wodo convention)
                        ms = mlat
                        if dd == vax:
                            ms = mvert
                        gNa = gNa * ms
                        for i in range(M):
                            gmu[i] += gNa * (grads[gp, 2 * i + 1, dd]
                                             * ms)
                            gphi[i] += gNa * (grads[gp, 2 * i, dd] * ms)
                            gq[i] += gNa * qphi[gp, i, dd]
                        for k in range(K):
                            gpsi[k] += gNa * (grads[gp, 2 * M + 2 * k,
                                                    dd] * ms)
                            gth[k] += gNa * (grads[gp,
                                                   2 * M + 2 * k + 1,
                                                   dd] * ms)
                    else:
                        for i in range(M):
                            gmu[i] += gNa * grads[gp, 2 * i + 1, dd]
                            gphi[i] += gNa * grads[gp, 2 * i, dd]
                            gq[i] += gNa * qphi[gp, i, dd]
                        for k in range(K):
                            gpsi[k] += gNa * grads[gp, 2 * M + 2 * k, dd]
                            gth[k] += gNa * grads[gp,
                                                  2 * M + 2 * k + 1, dd]
                if wp.static(ANISO):
                    for k in range(K):
                        if wp.static(FILM):
                            # mapped test gradient . R grad~ psi (Rg is
                            # built from mapped components above)
                            grot[k] = (dNtab[q, a, 0] * mlat * Rgx[k]
                                       + dNtab[q, a, 1] * mvert
                                       * Rgy[k]) * dscale
                        else:
                            grot[k] = (dNtab[q, a, 0] * Rgx[k]
                                       + dNtab[q, a, 1] * Rgy[k]) \
                                * dscale
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
                            if wp.static(DTH):
                                tr *= mT
                                sm *= wp.sqrt(mT)
                    r_p = (Na * (sigma * vals[gp, 2 * i]
                                 - hist[gp, 2 * i] - src[gp, 2 * i])
                           + tr + sm * gq[i]) * dJxW
                    if wp.static(FILM):
                        # frame advection Int v (K xi_y/h) dphi/dxi_y
                        # (S3a; raw xi_y-derivative, coefficient adv)
                        r_p += Na * adv * grads[gp, 2 * i, vax] * dJxW
                    r_m = (Na * (vals[gp, 2 * i + 1] - mub[i]
                                 - src[gp, 2 * i + 1])) * dJxW \
                        - kap[i] * gphi[i] * dJxW
                    wp.atomic_add(be, e, ndof * a + 2 * i, -r_p)
                    wp.atomic_add(be, e, ndof * a + 2 * i + 1, -r_m)
                for k in range(K):
                    rp = 2 * M + 2 * k
                    # gradient flux: isotropic eps2 grad psi, or the
                    # anisotropic q_k = a^2 grad psi + a a' R grad psi
                    # (isotropic branch keeps the ORIGINAL association
                    # Lpsi*eps2*gpsi — bit-identical S0/S1 arithmetic)
                    flx = Lpsi[k] * eps2[k] * gpsi[k]
                    if wp.static(ANISO):
                        flx = Lpsi[k] * (A2k[k] * gpsi[k]
                                         + AAk[k] * grot[k])
                    r_s = (Na * (sigma * vals[gp, rp] - hist[gp, rp]
                                 + Lpsi[k] * Fpsi[k] + qpsi[gp, k]
                                 - src[gp, rp])
                           + flx) * dJxW
                    if wp.static(FILM):
                        # psi advects EXACTLY as phi (S3 contract);
                        # no top flux (crystallinity does not evaporate)
                        r_s += Na * adv * grads[gp, rp, vax] * dJxW
                    pk = porv[k] + pfloor
                    if wp.static(TH_FROZEN):
                        r_t = (Na * (sigma * vals[gp, rp + 1]
                                     - hist[gp, rp + 1]
                                     - src[gp, rp + 1])) * dJxW
                        if wp.static(FILM):
                            # frozen theta in the film: the bookkeeping
                            # row becomes MARKER ADVECTION (theta rides
                            # the frame; k_e = 0 restores the identity)
                            r_t += Na * adv * grads[gp, rp + 1, vax] \
                                * dJxW
                    else:
                        r_t = (Na * (pk * (sigma * vals[gp, rp + 1]
                                           - hist[gp, rp + 1])
                                     - src[gp, rp + 1])
                               + pk * ceff[k] * gth[k]) * dJxW
                        if wp.static(FILM):
                            # KWC film: (p+pf)(theta_t + adv theta_y)
                            r_t += Na * pk * adv \
                                * grads[gp, rp + 1, vax] * dJxW
                    wp.atomic_add(be, e, ndof * a + rp, -r_s)
                    wp.atomic_add(be, e, ndof * a + rp + 1, -r_t)
                # Jacobian blocks
                for b in range(nbf):
                    Nb = Ntab[q, b]
                    lap = wp.float64(0.0)
                    if wp.static(FILM):
                        # mapped grad~.grad~ (S3a): ms^2 per direction
                        for dd in range(dim):
                            ms = mlat
                            if dd == vax:
                                ms = mvert
                            lap += dNtab[q, a, dd] * dNtab[q, b, dd] \
                                * dscale * dscale * ms * ms
                    else:
                        for dd in range(dim):
                            lap += dNtab[q, a, dd] * dNtab[q, b, dd] \
                                * dscale * dscale
                    NN = Na * Nb * dJxW
                    lapw = lap * dJxW
                    advw = wp.float64(0.0)
                    if wp.static(FILM):
                        # d/dx of the frame-advection term (S3a)
                        advw = Na * adv * dNtab[q, b, vax] * dscale \
                            * dJxW
                    for i in range(M):
                        ra = ndof * a + 2 * i
                        # phi_i row: time + transport
                        wp.atomic_add(Ae, e, ra, ndof * b + 2 * i,
                                      sigma * NN)
                        if wp.static(FILM):
                            wp.atomic_add(Ae, e, ra, ndof * b + 2 * i,
                                          advw)
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
                                    ov = Ons[i, j] * lapw
                                    if wp.static(DTH):
                                        ov *= mT
                                    wp.atomic_add(Ae, e, ra,
                                                  ndof * b + 2 * j + 1,
                                                  ov)
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
                        # gradient-flux Jacobian: d/dpsi_b of
                        # Int grad(Na).q_k with FROZEN (a^2, a a'):
                        # a^2 grad(Na).grad(Nb) + a a' grad(Na).R grad(Nb)
                        gj = eps2[k] * lapw
                        if wp.static(ANISO):
                            rotw = (dNtab[q, a, 1] * dNtab[q, b, 0]
                                    - dNtab[q, a, 0] * dNtab[q, b, 1]) \
                                * dscale * dscale * dJxW
                            if wp.static(FILM):
                                # grad~Na . R grad~Nb = mlat mvert x
                                # the unmapped rotation pairing (S3a)
                                rotw = rotw * (mlat * mvert)
                            gj = A2k[k] * lapw + AAk[k] * rotw
                        wp.atomic_add(Ae, e, rs, ndof * b + rp,
                                      sigma * NN
                                      + Lpsi[k] * (d2pp[k, k] * NN
                                                   + gj))
                        if wp.static(FILM):
                            wp.atomic_add(Ae, e, rs, ndof * b + rp,
                                          advw)
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
                            if wp.static(FILM):
                                wp.atomic_add(Ae, e, rt,
                                              ndof * b + rp + 1, advw)
                        else:
                            wp.atomic_add(Ae, e, rt, ndof * b + rp + 1,
                                          pk * sigma * NN
                                          + pk * ceff[k] * lapw)
                            if wp.static(FILM):
                                wp.atomic_add(Ae, e, rt,
                                              ndof * b + rp + 1,
                                              pk * advw)
                                # psi-col of the (p+pf) adv theta_y term
                                wp.atomic_add(
                                    Ae, e, rt, ndof * b + rp,
                                    porp[k] * Nb * Na * adv
                                    * grads[gp, rp + 1, vax] * dJxW)
                            wp.atomic_add(
                                Ae, e, rt, ndof * b + rp,
                                porp[k] * Nb
                                * (Na * (sigma * vals[gp, rp + 1]
                                         - hist[gp, rp + 1])
                                   + ceff[k] * gth[k]) * dJxW)

    _kernel_cache[key] = mpf_k
    return mpf_k


# ---------------------------------------------------------------------
# generic boundary-face consistent mass (A2/A4a builder, factored for
# reuse: wall energy, film top flux, evaporative-cooling load)
# ---------------------------------------------------------------------
def _face_mass(dm, face):
    """Boundary-face node lists + CONSISTENT face mass matrices:
    tensor product of the 1-D consistent edge mass over the in-face
    dims.  BASIS-GENERIC (A4a): the 1-D edge mass is built from the
    tabulated 1-D Lagrange basis by quadrature (exact at the (p+1)-
    point Gauss rule for the degree-2p integrand), so p = 1 reproduces
    le [[1/3, 1/6], [1/6, 1/3]] exactly and p = 2 the Simpson-
    consistent le/30 [[4, 2, -1], [2, 16, 2], [-1, 2, 4]].
    face = (axis, side): side 0 = the min face, 1 = the max face.
    Returns (faces [nf, nfn] int64, fmass [nf, nfn, nfn])."""
    from ..mesh.nodes import _local_offsets
    from ..mesh.basis import gauss_1d, lagrange_1d
    mesh = dm.mesh
    vax, side = int(face[0]), int(face[1])
    assert 0 <= vax < dm.dim, face
    assert not mesh.tree.periodic[vax], "face on a periodic axis"
    coords = mesh.node_coords
    target = (coords[:, vax].min() if side == 0
              else coords[:, vax].max())
    tol = 1e-12
    faces, fmass = [], []
    for pv, conn in mesh.conn_of.items():
        # 1-D consistent edge mass on the UNIT interval:
        # m1[a, b] = Int_0^1 N_a N_b (reference [-1, 1] halved)
        pts, wts = gauss_1d(int(pv))
        Nq = np.array([lagrange_1d(int(pv), x)[0] for x in pts])
        m1 = 0.5 * np.einsum("q,qa,qb->ab", wts, Nq, Nq)
        offs = _local_offsets(int(pv), dm.dim)
        loc = np.where(offs[:, vax] == (0 if side == 0 else pv))[0]
        of = np.delete(offs[loc], vax, axis=1)      # in-face offsets
        Mu = np.ones((len(loc), len(loc)))
        for dd in range(dm.dim - 1):
            Mu *= m1[of[:, None, dd], of[None, :, dd]]
        nn = conn[:, loc]                           # [ne, nfn]
        on_w = np.all(np.abs(coords[nn, vax] - target) < tol, axis=1)
        h_el = mesh.tree.h()[mesh.bins[pv]]
        for e in np.where(on_w)[0]:
            faces.append(nn[e])
            fmass.append(h_el[e] ** (dm.dim - 1) * Mu)
    assert faces, "no elements on the requested face"
    return np.asarray(faces, np.int64), np.asarray(fmass, np.float64)


# ---------------------------------------------------------------------
# A1 — segregated temperature field (annealing stage: no flow)
# ---------------------------------------------------------------------
class TemperatureField:
    r"""Linear scalar diffusion solve for the nodal temperature field
    (A1; module docstring).  Strong form on the annealing stage:

        rho_cp dT/dt = div(k_th grad T) + s(x, t),

    BDF1 weak form (test function N_a, integrate the divergence by
    parts; natural BCs are INSULATED faces, Dirichlet faces by row
    replacement):

        Int N_a rho_cp (T^{n+1} - T^n)/dt dV
      + Int grad(N_a) . k_th grad(T^{n+1}) dV
      - Int N_a s(x, t^{n+1}) dV                       = 0.

    Term-to-code map: the mass matrix M (Int N_a N_b, element-exact
    (h/2)^dim scaling of the reference einsum) carries the time term;
    the stiffness Kst (Int grad N_a . grad N_b, extra (2/h)^2 metric)
    carries the conduction term; _load carries the consistent source
    Int N_a s via the per-bin GP tables.  System per attempt:

        (rho_cp/dt M + k_th Kst) T^{n+1} = rho_cp/dt M T^n + load,

    LINEAR — one factorization per dt value (cached; the Appendix-A
    ladder re-tries at scaled dt, so the cache keys on dt).  Dirichlet:
    row-replaced identity rows, rhs = g(x, t^{n+1}) (substrate/ambient
    annealing protocols are exactly these g functions).  Assembly is
    host scipy in FULL node space, then the constraint triple product
    (hanging nodes ride the same Tc as the multiphase system).
    attempt() does NOT commit — the caller owns T^n (the stepper
    commits on accepted steps only, matching the reject ladder).

    FILM MODE (film=True; S3a, module docstring): the strip is the
    Landau-mapped frame — the stiffness splits into lateral + vertical
    blocks recombined per attempt as mlat^2 Klat + mvert^2 Kvert
    (mlat = 1/lat_scale, mvert = Ycomp/h), the frame advection adds
    rho_cp (K/h) Adv with Adv = Int N_a xi_y dN_b/dxi_y (raw xi_y,
    wodo convention), and the EVAPORATIVE-COOLING natural load
    k_th dT/dn = -L_vap J_evap at the receding surface enters as
    rhs -= L_vap K (Ycomp/h) Int_top N_a dS (L_vap = 0 default OFF).
    The factorization cache keys on (dt, h, K) — h moves every step,
    so film marches refactorize per accepted-dt change (the scalar
    T system is small next to the multiphase solve)."""

    def __init__(self, dm, rho_cp=1.0, k_th=1.0, src_fn=None,
                 dirichlet=None, g_fn=None, film=False, lat_scale=1.0,
                 L_vap=0.0):
        from .poisson import gauss_points
        self.dm = dm
        self.rho_cp, self.k_th = float(rho_cp), float(k_th)
        self.src_fn = src_fn
        self.dirichlet = None if dirichlet is None \
            else np.asarray(dirichlet, np.int64)
        self.g_fn = g_fn
        self.film = bool(film)
        self.lat_scale = float(lat_scale)
        self.L_vap = float(L_vap)
        mesh, cons = dm.mesh, dm.constraints
        self.Tc = cons.T.tocsr()
        self.free_coords = mesh.node_coords[cons.free_nodes]
        self.xq = gauss_points(mesh, dm.tables_by_p)
        vax = dm.dim - 1
        rows, cols, mv, kv, kvv, av = [], [], [], [], [], []
        self._wJ = {}
        for pv, eids in mesh.bins.items():
            tb = dm.tables_by_p[pv]
            conn = mesh.conn_of[pv].astype(np.int64)
            hh = mesh.tree.h()[eids]
            ne, nbf = conn.shape
            # reference element matrices (exact for the affine map)
            Mref = np.einsum("q,qa,qb->ab", tb.w, tb.N, tb.N)
            Kref = np.einsum("q,qad,qbd->ab", tb.w, tb.dN, tb.dN)
            jac = (hh / 2.0) ** dm.dim
            ksc = jac * (2.0 / hh) ** 2
            rows.append(np.repeat(conn, nbf, axis=1).ravel())
            cols.append(np.tile(conn, (1, nbf)).ravel())
            mv.append((jac[:, None, None] * Mref[None]).ravel())
            kv.append((ksc[:, None, None] * Kref[None]).ravel())
            self._wJ[pv] = np.tile(tb.w, ne) * np.repeat(jac, tb.nqp)
            # film split (S3a): vertical-only stiffness + advection
            Kvref = np.einsum("q,qa,qb->ab", tb.w, tb.dN[:, :, vax],
                              tb.dN[:, :, vax])
            kvv.append((ksc[:, None, None] * Kvref[None]).ravel())
            if self.film:
                xv = self.xq[pv][:, vax].reshape(ne, tb.nqp)
                Aev = np.einsum("q,qa,qb,eq->eab", tb.w, tb.N,
                                tb.dN[:, :, vax], xv) \
                    * (jac * (2.0 / hh))[:, None, None]
                av.append(Aev.ravel())
        n = len(mesh.node_coords)
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        Mfull = sp.coo_matrix((np.concatenate(mv), (r, c)),
                              shape=(n, n)).tocsr()
        Kfull = sp.coo_matrix((np.concatenate(kv), (r, c)),
                              shape=(n, n)).tocsr()
        self.Mfree = (self.Tc.T @ Mfull @ self.Tc).tocsr()
        self.Kfree = (self.Tc.T @ Kfull @ self.Tc).tocsr()
        if self.film:
            Kvfull = sp.coo_matrix((np.concatenate(kvv), (r, c)),
                                   shape=(n, n)).tocsr()
            self.Kv_free = (self.Tc.T @ Kvfull @ self.Tc).tocsr()
            self.Klat_free = (self.Kfree - self.Kv_free).tocsr()
            Afull = sp.coo_matrix((np.concatenate(av), (r, c)),
                                  shape=(n, n)).tocsr()
            self.Adv_free = (self.Tc.T @ Afull @ self.Tc).tocsr()
            self.y_comp = float(mesh.node_coords[:, vax].max())
            # top-face lumped load Int_top N_a dS (consistent face
            # mass @ 1) for the evaporative-cooling natural term
            faces, fM = _face_mass(dm, (vax, 1))
            Fe = np.zeros(n)
            np.add.at(Fe, faces.ravel(), fM.sum(axis=2).ravel())
            self._evap_free = np.asarray(self.Tc.T @ Fe)
        self._lu, self._lu_dt = None, None

    def _load(self, t):
        F = np.zeros(self.Tc.shape[0])
        if self.src_fn is not None:
            for pv, conn in self.dm.mesh.conn_of.items():
                tb = self.dm.tables_by_p[pv]
                s = self.src_fn(self.xq[pv], t) * self._wJ[pv]
                Fa = s.reshape(len(conn), tb.nqp) @ tb.N
                np.add.at(F, np.asarray(conn, np.int64).ravel(),
                          Fa.ravel())
        return np.asarray(self.Tc.T @ F)

    def attempt(self, T_old, dt, t_new, h=None, K_evap=0.0):
        """One BDF1 solve [t_new - dt, t_new] from T_old (free vector);
        returns T_new WITHOUT committing (linear: always 'converged').
        Film mode passes the frozen frame state (h, K_evap)."""
        from scipy.sparse.linalg import splu
        a = self.rho_cp / dt
        rhs = a * (self.Mfree @ T_old) + self._load(t_new)
        key = (dt, h, K_evap) if self.film else dt
        if self._lu is None or self._lu_dt != key:
            if self.film:
                assert h is not None and h > 0.0, h
                mlat = 1.0 / self.lat_scale
                mvert = self.y_comp / h
                A = (a * self.Mfree
                     + self.k_th * (mlat * mlat * self.Klat_free
                                    + mvert * mvert * self.Kv_free)
                     + self.rho_cp * (K_evap / h)
                     * self.Adv_free).tolil()
            else:
                A = (a * self.Mfree + self.k_th * self.Kfree).tolil()
            if self.dirichlet is not None:
                for i in self.dirichlet:
                    A.rows[i] = [int(i)]
                    A.data[i] = [1.0]
            self._lu = splu(A.tocsr().tocsc())
            self._lu_dt = key
        if self.film and self.L_vap != 0.0 and K_evap > 0.0:
            # evaporative cooling s_evap = -L_vap J_evap at the top
            # face (natural term; mapped boundary measure Ycomp/h)
            rhs -= (self.L_vap * K_evap * self.y_comp / h) \
                * self._evap_free
        if self.dirichlet is not None:
            rhs[self.dirichlet] = self.g_fn(
                self.free_coords[self.dirichlet], t_new)
        return self._lu.solve(rhs)


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
                 clip_psi=True, line_search=False,
                 T_mode="scalar", T_field=None, D_T=None,
                 wall_g=None, wall_h=None, wall_face=(1, 0),
                 delta_a=None, m_a=None, a_reg=1e-8, tstep="bdf1",
                 film=None, assembly="host", block_sparse=False):
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
        # A1 — temperature mode + D(T) hook (module docstring)
        assert T_mode in ("scalar", "field"), T_mode
        self.T_mode = T_mode
        if T_mode == "field":
            assert T_fn is None, \
                "field mode replaces the T_fn schedule (annealing " \
                "protocols become Dirichlet BCs)"
        self.D_T = None if D_T is None \
            else (float(D_T[0]), float(D_T[1]))
        if self.D_T is not None:
            assert self.D_T[1] > 0.0, "D_T = (Ea, T_ref), T_ref > 0"
        # A3 — anisotropic crystal growth (module docstring)
        self.delta_a = pad(delta_a, 0.0)
        self.m_a = pad(m_a, 4.0)
        self.a_reg = float(a_reg)
        self.aniso = bool((self.delta_a != 0.0).any())
        if self.aniso:
            assert dm.dim == 2, "anisotropy: 2-D only (A3 contract)"
            assert self.theta_mode == "frozen", \
                "anisotropy requires marker (frozen) theta " \
                "(alpha_th = beta_th = 0); the KWC back-torque is a " \
                "recorded extension"
        # A2 — substrate surface energy (module docstring)
        self.wall_g = (np.zeros(self.M) if wall_g is None
                       else np.asarray(wall_g, np.float64
                                       ).reshape(self.M))
        self.wall_h = (np.zeros(self.M) if wall_h is None
                       else np.asarray(wall_h, np.float64
                                       ).reshape(self.M))
        self.wall_on = bool((self.wall_g != 0.0).any()
                            or (self.wall_h != 0.0).any())
        self.wall_face = (int(wall_face[0]), int(wall_face[1]))
        # S3a — film mode (Landau-mapped moving frame; module docstring)
        self.film_on = film is not None
        if self.film_on:
            f = dict(film)
            ke = f.pop("k_e", 0.0)
            self.h_curr = float(f.pop("h0", 1.0))
            self.lat_scale = float(f.pop("lat_scale", 1.0))
            self.dh_cap = float(f.pop("dh_cap", 0.004))
            self.K_fn = f.pop("K_fn", None)
            assert not f, f"unknown film keys: {sorted(f)}"
            if np.ndim(ke) == 0:
                # scalar k_e = the eliminated solvent's rate (the
                # exact wodo Bi semantics — configs translate 1:1)
                kev = np.zeros(n_sp)
                kev[self.M] = float(ke)
            else:
                kev = np.asarray(ke, np.float64).reshape(n_sp)
            assert (kev >= 0.0).all(), kev
            self.k_e = kev
            self.vax = dm.dim - 1       # vertical = LAST axis (wodo)
            assert not dm.mesh.tree.periodic[self.vax], \
                "film: the vertical axis must be non-periodic"
            if self.wall_on:
                assert self.wall_face != (self.vax, 1), \
                    "A2 wall energy on the MOVING (top) face is " \
                    "unsupported (substrate face only in film mode)"
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
        # A4b time scheme (module docstring): bdf1 default; bdf2 =
        # variable-step, deterministic-only (noise weak order out of
        # scope — recorded), BDF1 bootstrap on the first step
        assert tstep in ("bdf1", "bdf2"), tstep
        self.tstep = tstep
        if tstep == "bdf2":
            assert noise_psi == 0.0 and noise_phi == 0.0, \
                "BDF2 is deterministic-only (FDT-noise weak order " \
                "under BDF2 is a recorded scope limit)"
        self.hist2 = None       # x^{n-1} (None => BDF1 bootstrap)
        self.dt_prev = None     # dt_{n-1} of the last ACCEPTED step
        self.newton_tol, self.newton_max = newton_tol, newton_max
        assert linsolver in ("splu", "cudss", "blockch", "blockch_dev"), \
            linsolver
        self.linsolver = linsolver
        self._cudss = None
        # B-track blockch: per-mesh symbolic setup + iteration records
        # (('blockch_iters', key) contract) live here
        self._solver_cache = {}
        self._sigma = None      # stashed per attempt (blockch meta)
        self._m_ref = None      # representative pair mobilities
        # D-track device-side assembly (module docstring DEVICE
        # ASSEMBLY): "host" = the existing scipy COO -> CSR -> T^T K T
        # path bit-identically; "device" = slot-map CSR scatter
        # (DeviceNSAssembler, wodo v1.2 pattern) with zero-copy solver
        # handoff.  Constructor flag => bit-class-parity testable.
        assert assembly in ("host", "device"), assembly
        self.assembly = assembly
        # B5: block-masked device pattern = kron(G, blockmask) with the
        # compile-time (M, K, mob, theta) live-block mask — drops the
        # structural-zero couplings (cuDSS fill fix; int32-nnz and
        # value-memory headroom for the capacity ladder).  Opt-in: the
        # superset pattern stays the gated default.
        self.block_sparse = bool(block_sparse)
        assert not (block_sparse and assembly == "host"), \
            "block_sparse is a device-assembly pattern option"
        self._asm = None            # DeviceNSAssembler (lazy, per mesh)
        self._cudss_dev = None      # device-CSR DirectSolver plan
        self._n_dev_plans = 0       # nnz-stability regression counter
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
            beta=self.beta_th, Lth=self.L_th, Tm=self.Tm,
            da=self.delta_a, ma=self.m_a).items()}
        # dummy per-GP T array (compile-time dead unless tfield/dth)
        self._tq_dummy = wp.array(np.zeros(1), dtype=wp.float64,
                                  device=d)
        # S3a film topology: top faces (basis-generic consistent face
        # mass), top-node row (K closure), xi_y at GPs (advection)
        if self.film_on:
            coords = self.mesh.node_coords
            self.y_comp = float(coords[:, self.vax].max())
            tol = 1e-12
            self.top_nodes = np.where(
                coords[:, self.vax] > self.y_comp - tol)[0]
            self.top_faces, self.top_face_M = _face_mass(
                dm, (self.vax, 1))
            self._xiy_wp = {
                pv: wp.array(np.ascontiguousarray(
                    self.xq[pv][:, self.vax]), dtype=wp.float64,
                    device=d)
                for pv in self.xq}
            self._K_pend = 0.0
        # A1 field mode: segregated TemperatureField + nodal state
        if self.T_mode == "field":
            tf = dict(T_field or {})
            T0 = tf.pop("T0", self.T)
            self._Tdiff = TemperatureField(
                dm, rho_cp=tf.pop("rho_cp", 1.0),
                k_th=tf.pop("k_th", 1.0),
                src_fn=tf.pop("src", None),
                dirichlet=tf.pop("dirichlet", None),
                g_fn=tf.pop("g", None),
                film=self.film_on,
                lat_scale=(self.lat_scale if self.film_on else 1.0),
                L_vap=tf.pop("L_vap", 0.0))
            assert not tf, f"unknown T_field keys: {sorted(tf)}"
            self.T_nodes = (np.full(self.nfree, float(T0))
                            if np.isscalar(T0)
                            else np.asarray(T0(self.free_coords),
                                            np.float64))
            self._T_pend = self.T_nodes
        # A2 wall face mass (built only when the wall energy is on)
        if self.wall_on:
            self._build_wall_faces()

    # -- A2 wall-face topology (wodo _build_top_faces pattern) -----------
    def _build_wall_faces(self):
        """Substrate-face node lists + CONSISTENT face mass matrices
        for the wall free-energy natural term (basis-generic builder
        _face_mass, A4a).  wall_face = (axis, side): side 0 = the min
        face (y = 0 substrate default), 1 = the max face (face-generic
        per the A2 contract)."""
        self.wall_faces, self.wall_face_M = _face_mass(self.dm,
                                                       self.wall_face)

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
        self.hist2 = None       # restart the BDF2 bootstrap
        self.dt_prev = None
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

    def _film_K(self):
        """S3a evaporation velocity K = SUM_i k_e_i avg(phi_i^top)
        >= 0, frozen at t_n from the COMMITTED state (wodo pattern;
        the eliminated solvent's avg closes the simplex).  A K_fn
        manufactured-frame override (MMS) replaces the closure."""
        if self.K_fn is not None:
            return float(self.K_fn(self.t))
        tv = [np.asarray(self.Tc @ self.hist[2 * i::self.ndof])
              [self.top_nodes] for i in range(self.M)]
        avg = [float(np.mean(v)) for v in tv]
        avg.append(float(np.mean(1.0 - sum(tv))))
        return max(sum(self.k_e[i] * max(a, 0.0)
                       for i, a in enumerate(avg)), 0.0)

    def _solve(self, A, r):
        if self.linsolver in ("blockch", "blockch_dev"):
            # B1: per-pair two-factor Schur + AC diagonal blocks
            # (linsolve "pairs"/"ac" meta).  Non-convergence of BOTH
            # the two-factor form and the exact-Schur escalation
            # signals divergence to the Appendix-A reject ladder,
            # matching the cudss contract (nan return).
            from ..solvers.linsolve import solve_linear
            meta = self._blockch_meta()
            if self.linsolver == "blockch_dev":
                meta["inners"] = "device"
            self._solver_cache[("blockch_meta", "mpf")] = meta
            try:
                return solve_linear(A, r, solver="blockch", tol=1e-10,
                                    cache=self._solver_cache,
                                    cache_key="mpf",
                                    device=self.dm.device)
            except RuntimeError:
                return np.full(A.shape[0], np.nan)
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

    # -- per-attempt frozen inputs (shared host/device assembly) ---------
    def _attempt_ctx(self, dt):
        """Everything FROZEN over one implicit solve: BDF coefficients
        and combined history, crystal driving, film frame state, per-GP
        temperature, FDT/CHC noise draws, MMS sources.  Code motion
        from the head of _attempt (host path bit-identical: same
        computations, same RNG draw order); the device-assembly path
        consumes the same ctx so both paths freeze the same physics."""
        d = self.dm.device
        nd = self.ndof
        sigma = 1.0 / dt
        t_new = self.t + dt
        if self.T_mode == "field":
            # A1: the drive slot carries dh_k; the kernel applies the
            # T law per GP (drive is refreshed through Tq instead)
            drive = self.dh.copy()
            drive[self.K:] = 0.0
        else:
            drive = np.concatenate([self._drive(t_new)[:self.K],
                                    np.zeros(self.Kp - self.K)]) \
                if self.K else np.zeros(self.Kp)
        arr = lambda a_: wp.array(np.ascontiguousarray(a_),
                                  dtype=wp.float64, device=d)
        drive_d = arr(drive)
        # S3a film frame state, FROZEN per attempt (wodo pattern):
        # K from the committed state at t_n, metric from h_curr
        if self.film_on:
            K_tot = self._film_K()
            self._K_pend = K_tot
            minv = 1.0 / self.h_curr
            mlat = 1.0 / self.lat_scale
            mvert = self.y_comp * minv
        else:
            K_tot, minv, mlat, mvert = 0.0, 0.0, 1.0, 1.0
        # A1: per-GP temperature (field mode: segregated linear T
        # advance FIRST — Lie split, O(dt); committed only on accept.
        # Scalar mode with the D(T) hook: uniform T(t_new) at GPs.)
        tq_gp = None
        if self.T_mode == "field":
            self._T_pend = self._Tdiff.attempt(
                self.T_nodes, dt, t_new,
                h=(self.h_curr if self.film_on else None),
                K_evap=K_tot)
            tq_gp, _ = self._gp(self._T_pend)
        elif self.D_T is not None:
            Tsc = self.T_fn(t_new) if self.T_fn is not None else self.T
            tq_gp = {pv: np.full(len(self.mesh.conn_of[pv]) * b["nqp"],
                                 Tsc)
                     for pv, b in self.dm.bins.items()}
        dtea, dtref = self.D_T if self.D_T is not None else (0.0, 1.0)
        hv, _ = self._pack_fields(self.hist)
        if self.tstep == "bdf2" and self.hist2 is not None:
            # A4b variable-step BDF2 (module docstring): sigma = a/dt,
            # hist = (b x^n - c x^{n-1})/dt — same kernel time term.
            # Coefficients from the ACTUAL (dt, dt_prev) per attempt:
            # ladder rejects rescale dt without touching the history.
            rr = dt / self.dt_prev
            sigma = (1.0 + 2.0 * rr) / (1.0 + rr) / dt
            bv = 1.0 + rr
            cv = rr * rr / (1.0 + rr)
            hv2, _ = self._pack_fields(self.hist2)
            hist_gp = {pv: (bv * hv[pv] - cv * hv2[pv]) / dt
                       for pv in hv}
        else:                       # BDF1 (default + BDF2 bootstrap)
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
        # blockch meta scalars, FROZEN per attempt: sigma (the only
        # per-attempt scalar of the two-factor recipe) and the
        # representative pair mobilities at the committed mean
        # composition (wodo precedent: constant reference M11/M22 under
        # var_mob — the scalar only weights the W factors)
        self._sigma = sigma
        if self.linsolver in ("blockch", "blockch_dev"):
            self._m_ref = self._mob_ref(t_new)
        return dict(sigma=sigma, t_new=t_new, drive_d=drive_d,
                    K_tot=K_tot, minv=minv, mlat=mlat, mvert=mvert,
                    tq_gp=tq_gp, dtea=dtea, dtref=dtref,
                    hist_gp=hist_gp, qpsi_gp=qpsi_gp, qphi_gp=qphi_gp,
                    src_gp=src_gp)

    # -- blockch (B-track): representative scalars + pair/AC meta --------
    def _mob_ref(self, t_new):
        """Representative diagonal mobility per CH pair, evaluated at
        the COMMITTED state's mean composition (numpy mirror of the
        kernel's lam/lamM diagonal; the Eq. 13 drop and the D(T)
        Arrhenius factor included).  A scalar per pair is all the
        two-factor recipe needs (it weights W1/W2; the true variable
        mobility rides in through the extracted Acm/Amc blocks) — the
        wodo film uses its constant reference M11/M22 the same way
        under var_mob."""
        nd = self.ndof
        ph = np.array([float(np.mean(self.hist[2 * i::nd]))
                       for i in range(self.M)])
        ps_solv = 1.0 - ph.sum()
        phis = np.concatenate([ph, [ps_solv]])
        psis = np.array([float(np.mean(
            self.hist[2 * self.M + 2 * k::nd]))
            for k in range(self.K)])
        mT = 1.0
        if self.D_T is not None:
            if self.T_mode == "field":
                Tv = float(np.mean(self.T_nodes))
            else:
                Tv = float(self.T_fn(t_new)) if self.T_fn is not None \
                    else self.T
            ea, tref = self.D_T
            mT = float(np.exp(ea * (1.0 / tref - 1.0 / Tv)))
        if self.mob == "const":
            m = np.array([self.onsager[i, i] for i in range(self.M)])
            return np.maximum(m * mT, 1e-14)
        if self.mob == "fastmode":            # M = 1 closure
            pc = float(np.clip(phis[0], 1e-6, 1.0 - 1e-6))
            omp = 1.0 - pc
            d1s = self.D_lo[0] ** omp * self.D_hi[0] ** pc
            d2s = self.D_lo[1] ** pc * self.D_hi[1] ** omp
            lam = (omp * omp * pc / self.Ninv[0] * d1s
                   + pc * pc * omp / self.Ninv[1] * d2s)
            return np.maximum(np.array([lam * mT]), 1e-14)
        # fastmode_n / slowmode_n: Vignes omega + Eq. 13 drop (kernel
        # mirror at the mean composition)
        dsl, csl, wsl = self.ls_drop
        pt = float(np.prod(1.0 - np.clip(psis, 0.0, 1.0))) \
            if self.K else 1.0
        fdrop = dsl ** (0.5 * (1.0 + np.tanh(wsl * (1.0 - pt - csl))))
        n_sp = self.M + 1
        omv = np.empty(n_sp)
        for i in range(n_sp):
            dsi = fdrop * mT
            for j in range(n_sp):
                dsi *= self.D_self[i, j] ** float(
                    np.clip(phis[j], 0.0, 1.0))
            pc0 = float(np.clip(phis[i], 1e-6, 1.0 - 1e-6))
            omv[i] = pc0 / self.Ninv[i] * dsi
        somv = omv.sum()
        m = np.empty(self.M)
        for i in range(self.M):
            if self.mob == "slowmode_n":
                m[i] = omv[i] * (1.0 - omv[i] / somv)
            else:
                opi = 1.0 - phis[i]
                m[i] = (opi * opi * omv[i]
                        + phis[i] * phis[i] * (somv - omv[i]))
        return np.maximum(m, 1e-14)

    def _block_mask(self):
        """Compile-time dof-pair block sparsity of the mpf kernel (the
        kron(G, blockmask) pattern, B5): exactly the (ca, cb) blocks
        the kernel/face terms ever WRITE.  Mirrors make_mpf_newton's
        Jacobian scatter (phi row: time/adv diag + mobility mu cols +
        MATMOB dLam phi cols + dlnf psi cols; FASTMODE: mu_0 only;
        const: mu cols only.  mu row: mass diag + dmudphi + dmudpsi.
        psi row: diag + d2pp psi cols + d2pf phi cols.  theta row:
        diag (+ psi col under KWC)).  Face terms (wall (mu_i, phi_i),
        top flux (phi_i, phi_i)) and Dirichlet diagonals are inside.
        Exactness masked-vs-unmasked is gated."""
        nd = self.ndof
        M, K = self.M, self.K
        m = np.zeros((nd, nd), dtype=bool)
        matmob = self.mob in ("fastmode_n", "slowmode_n")
        for i in range(M):
            rp, rm = 2 * i, 2 * i + 1
            m[rp, rp] = True                     # time + advection
            if self.mob == "fastmode":
                m[rp, 1] = True                  # lam lap on mu_0
            elif matmob:
                for j in range(M):
                    m[rp, 2 * j] = m[rp, 2 * j + 1] = True
                for k in range(K):
                    m[rp, 2 * M + 2 * k] = True  # dlnf drop cols
            else:                                # const Onsager
                for j in range(M):
                    m[rp, 2 * j + 1] = True
            m[rm, rm] = True                     # mass
            for j in range(M):
                m[rm, 2 * j] = True              # dmudphi (+kappa lap)
            for k in range(K):
                m[rm, 2 * M + 2 * k] = True      # dmudpsi
        for k in range(K):
            rs, rt = 2 * M + 2 * k, 2 * M + 2 * k + 1
            for l in range(K):
                m[rs, 2 * M + 2 * l] = True      # d2pp
            for j in range(M):
                m[rs, 2 * j] = True              # d2pf
            m[rs, rs] = True
            m[rt, rt] = True
            if self.theta_mode == "kwc":
                m[rt, rs] = True                 # p'(psi) torque col
        return m

    def _blockch_meta(self):
        """The (M, K) blockch meta (linsolve _blockch_pairs contract):
        M CH pairs {'off', 'm', 'kappa'} + K AC blocks {'off'} on the
        2M+2K node-major layout.  sigma/m_ref are attempt-frozen by
        _attempt_ctx."""
        assert self._sigma is not None and self._m_ref is not None, \
            "blockch meta requested before _attempt_ctx froze sigma/m"
        return {"sigma": self._sigma, "ndof": self.ndof,
                "pairs": [{"off": 2 * i, "m": float(self._m_ref[i]),
                           "kappa": float(self.kap[i])}
                          for i in range(self.M)],
                "ac": [{"off": 2 * self.M + 2 * k}
                       for k in range(self.K)]}

    # -- host-path assembly at one Newton iterate ------------------------
    def _assemble_host(self, x, ctx):
        """The ORIGINAL host assembly (code motion from the _attempt
        closure — bit-identical computations): device element kernel
        -> host Ae/be pull -> scipy COO -> CSR -> constraint triple
        product -> Dirichlet row surgery.  Returns (A, r) free-space."""
        d = self.dm.device
        nd = self.ndof
        arr = lambda a_: wp.array(np.ascontiguousarray(a_),
                                  dtype=wp.float64, device=d)
        sigma, t_new = ctx["sigma"], ctx["t_new"]
        drive_d, K_tot = ctx["drive_d"], ctx["K_tot"]
        minv, mlat, mvert = ctx["minv"], ctx["mlat"], ctx["mvert"]
        tq_gp, dtea, dtref = ctx["tq_gp"], ctx["dtea"], ctx["dtref"]
        hist_gp, src_gp = ctx["hist_gp"], ctx["src_gp"]
        qpsi_gp, qphi_gp = ctx["qpsi_gp"], ctx["qphi_gp"]
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
                                 self.theta_mode,
                                 self.T_mode == "field",
                                 self.D_T is not None, self.aniso,
                                 self.film_on)
            p = self._par
            tq_d = (arr(tq_gp[pv]) if tq_gp is not None
                    else self._tq_dummy)
            xiy_d = (self._xiy_wp[pv] if self.film_on
                     else self._tq_dummy)
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
                p["Tm"], tq_d, wp.float64(dtea), wp.float64(dtref),
                p["da"], p["ma"], wp.float64(self.a_reg),
                xiy_d, wp.float64(mlat), wp.float64(mvert),
                wp.float64(minv), wp.float64(K_tot),
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
        if self.wall_on:
            # A2 WALL FREE ENERGY natural term (module docstring):
            # r_mu_i(a) += - Int_w N_a (g_i + 2 h_i phi_i) dS
            # (the kap dphi/dn = -f_w' variational BC), so
            # F (= -r) += + Mw @ (g_i + 2 h_i phi_i)|_face and the
            # Jacobian gains d r/d phi_i = -2 h_i Mw on the
            # (mu_i row, phi_i col) block.  Consistent P1 face
            # mass; assembled in FULL node space (constraints ride
            # the Tn triple product below).
            nfn = self.wall_faces.shape[1]
            # S3a: the wall natural term picks up the mapped
            # boundary measure factor Ycomp/h in film mode (the
            # computational mu rows are the physical ones divided
            # by lat_scale h/Ycomp — module docstring S3a);
            # wf = 1.0 outside film mode (bitwise identity)
            wf = (self.y_comp / self.h_curr) if self.film_on \
                else 1.0
            for i in range(self.M):
                gi, hi = self.wall_g[i], self.wall_h[i]
                if gi == 0.0 and hi == 0.0:
                    continue
                fv = np.asarray(self.Tc @ x[2 * i::nd])
                gd = nd * self.wall_faces + (2 * i + 1)
                fw = wf * (gi + 2.0 * hi * fv[self.wall_faces])
                np.add.at(F_full, gd.ravel(),
                          np.einsum("fab,fb->fa",
                                    self.wall_face_M, fw).ravel())
                if hi != 0.0:
                    cd = nd * self.wall_faces + 2 * i
                    rows.append(np.repeat(gd, nfn, axis=1).ravel())
                    cols.append(np.tile(cd, (1, nfn)).ravel())
                    valsK.append((-2.0 * hi * wf)
                                 * self.wall_face_M.ravel())
        if self.film_on and K_tot > 0.0:
            # S3a TOP-SURFACE ENRICHMENT FLUX — phi rows ONLY
            # (solvent evaporates AMORPHOUS; psi/theta carry no
            # flux — the S3 contract).  Weak term per retained i:
            # R_phi_i -= coef_i Int_top N_a phi_i dS, coef_i =
            # (K - k_e_i)(1/h) Ycomp (wodo v1.1 metric; k_e_i = 0
            # for nonvolatile species gives the exact wodo pair
            # that conserves h Int phi_i dtheta per step).
            # F (= -R) += +coef_i Mf phi_i; Jacobian -coef_i Mf on
            # the (phi_i row, phi_i col) face block.
            nfn = self.top_faces.shape[1]
            for i in range(self.M):
                coef = (K_tot - self.k_e[i]) * minv * self.y_comp
                if coef == 0.0:
                    continue
                fv = np.asarray(self.Tc @ x[2 * i::nd])
                gd = nd * self.top_faces + 2 * i
                Mf = coef * self.top_face_M
                np.add.at(F_full, gd.ravel(),
                          np.einsum("fab,fb->fa", Mf,
                                    fv[self.top_faces]).ravel())
                rows.append(np.repeat(gd, nfn, axis=1).ravel())
                cols.append(np.tile(gd, (1, nfn)).ravel())
                valsK.append(-Mf.ravel())
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

    # -- one implicit BDF1 solve at frozen dt; does NOT commit -----------
    def _attempt(self, dt):
        if self.assembly == "device":
            return self._attempt_device(dt)
        nd = self.ndof
        ctx = self._attempt_ctx(dt)

        def assemble(x):
            return self._assemble_host(x, ctx)

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

    # ==================================================================
    # DEVICE-SIDE ASSEMBLY (D-track; the wodo_film v1.2 model)
    # ==================================================================
    def _init_device_assembly(self):
        """Once per mesh: slot-map CSR pattern (DeviceNSAssembler,
        ndof = 2M + 2K), face-term slot/dof arrays, strong-row plan,
        and per-bin device GP-field buffers.

        WEAK-FORM -> SCATTER MAP (the assembly identity): the global
        Jacobian is A[ga, gb] = SUM_e Ae[e, la, lb] over all element
        local pairs whose global dofs coincide (ga = conn[e, a] * ndof
        + ca).  The host path realizes the sum by scipy COO -> CSR
        dedup + T^T K T; the device path precomputes ONCE per mesh
        epoch the CSR value slot of every (e, la, lb) pair and
        realizes the SAME sum as atomicAdd(vals[slot], Ae) — summation
        ORDER is the only difference (FP-chaos class; parity gates on
        canonicalized matrices / observables per the house rules).

        CONSTRAINTS: every multiphase production mesh (uniform 2-D/
        3-D, p = 1/2, periodic included — build_mesh bakes periodic
        seams into the connectivity) has IDENTITY constraints,
        asserted here, so the host triple products are numeric no-ops
        that merely PRUNE exact zeros (the measured S2 cuDSS
        plan-flapping source).  The device pattern is the FIXED
        element-graph superset: nnz never changes across noise steps
        — the cuDSS plan becomes nnz-stable (bonus fix, gated).
        Adapted/hanging-node meshes: assembly="host" (the constraint-
        aware weighted scatter exists in DeviceNSAssembler but the
        multiphase face terms and the node-graph pattern are not
        wired through constraint weights — recorded frontier).

        FACE TERMS (A2 wall energy, S3a top flux): natural-BC
        enrichments of interior equations; every (row, col) pair
        lives inside a boundary element block, hence inside the
        element pattern.  Their O(surface) values are computed on
        host (tiny closed-form face-mass products — the wodo v1.2
        documented hybrid) and folded in by slot scatter-add."""
        from ..assembly.device_assembly import DeviceNSAssembler
        n = self.Tc.shape[0]
        assert self.Tc.shape[0] == self.Tc.shape[1] and \
            (self.Tc - sp.identity(n, format="csr")).nnz == 0, (
                "assembly='device' requires identity constraints "
                "(uniform meshes; periodicity rides the connectivity)."
                "  Adapted meshes with hanging nodes: assembly='host'.")
        nd = self.ndof
        d = self.dm.device
        self._asm = DeviceNSAssembler(
            self.dm, ndof=nd,
            node_pattern=getattr(self, "_node_pattern", None),
            blockmask=self._block_mask() if self.block_sparse
            else None)
        asm = self._asm
        # persistent per-bin GP-field buffers (device _pack_fields
        # mirror) + zero buffers for compile-dead/inactive inputs
        self._vals_dev, self._grads_dev = {}, {}
        self._zsrc_d, self._zqpsi_d, self._zqphi_d = {}, {}, {}
        for pv, b in self.dm.bins.items():
            ngp = len(self.mesh.conn_of[pv]) * b["nqp"]
            self._vals_dev[pv] = wp.zeros((ngp, nd), dtype=wp.float64,
                                          device=d)
            self._grads_dev[pv] = wp.zeros((ngp, nd, self.dm.dim),
                                           dtype=wp.float64, device=d)
            if self.src_fns is None:
                self._zsrc_d[pv] = wp.zeros((ngp, nd),
                                            dtype=wp.float64, device=d)
            if not (self.noise_psi > 0.0 and self.K):
                self._zqpsi_d[pv] = wp.zeros((ngp, self.Kp),
                                             dtype=wp.float64, device=d)
            if not self.noise_phi > 0.0:
                self._zqphi_d[pv] = wp.zeros(
                    (ngp, self.M, self.dm.dim), dtype=wp.float64,
                    device=d)
        # A2 wall slots: Jacobian (mu_i row, phi_i col) face blocks for
        # species with h_i != 0; rhs rows for any active species
        # (host-order concatenation — values built per attempt)
        if self.wall_on:
            nfn = self.wall_faces.shape[1]
            jr, jc, jbase, rr_ = [], [], [], []
            for i in range(self.M):
                gi, hi = self.wall_g[i], self.wall_h[i]
                if gi == 0.0 and hi == 0.0:
                    continue
                gd = nd * self.wall_faces + (2 * i + 1)
                rr_.append(gd.ravel())
                if hi != 0.0:
                    cd = nd * self.wall_faces + 2 * i
                    jr.append(np.repeat(gd, nfn, axis=1).ravel())
                    jc.append(np.tile(cd, (1, nfn)).ravel())
                    jbase.append(-2.0 * hi * self.wall_face_M.ravel())
            self._wall_rhs_dof_d = wp.array(
                np.concatenate(rr_).astype(np.int32), dtype=wp.int32,
                device=d)
            if jr:
                slots = asm.csr_slots(np.concatenate(jr),
                                      np.concatenate(jc))
                self._wall_slots_d = wp.array(slots.astype(np.int32),
                                              dtype=wp.int32, device=d)
                self._wall_jbase = np.concatenate(jbase)  # x wf/attempt
            else:
                self._wall_slots_d = None
        # S3a top-flux slots (all retained species — per-attempt coefs
        # may vanish per species; zero adds are numeric no-ops)
        if self.film_on:
            nfn = self.top_faces.shape[1]
            fr = [np.repeat(nd * self.top_faces + 2 * i, nfn,
                            axis=1).ravel() for i in range(self.M)]
            fc = [np.tile(nd * self.top_faces + 2 * i,
                          (1, nfn)).ravel() for i in range(self.M)]
            slots = asm.csr_slots(np.concatenate(fr),
                                  np.concatenate(fc))
            self._flux_slots_d = wp.array(slots.astype(np.int32),
                                          dtype=wp.int32, device=d)
            self._flux_gdof_d = wp.array(np.concatenate(
                [(nd * self.top_faces + 2 * i).ravel()
                 for i in range(self.M)]).astype(np.int32),
                dtype=wp.int32, device=d)
        # Dirichlet strong rows (host nesting order: field f outer,
        # node inner — b_vals per iterate must match)
        if self.dirichlet is not None:
            rows = [i * nd + f for f, gfn in enumerate(self.g_fns)
                    if gfn is not None for i in self.dirichlet]
            asm.set_strong_rows(np.asarray(rows, np.int64))
        # element-batch buffers (Ae transient capped ~2 GB, G5 rung c)
        self._dev_bufs = {}

    def _dev_inputs(self, ctx):
        """Per-attempt device GP inputs (hist/src/noise/T), uploaded
        ONCE per ctx and cached on ctx['_dev'] (iterate-independent;
        zero buffers reused when an input is inactive)."""
        dev = ctx.get("_dev")
        if dev is not None:
            return dev
        d = self.dm.device
        arr = lambda a_: wp.array(np.ascontiguousarray(a_),
                                  dtype=wp.float64, device=d)
        tq_gp = ctx["tq_gp"]
        dev = dict(
            hist={pv: arr(v) for pv, v in ctx["hist_gp"].items()},
            src=({pv: arr(v) for pv, v in ctx["src_gp"].items()}
                 if self.src_fns is not None else self._zsrc_d),
            qpsi=({pv: arr(v) for pv, v in ctx["qpsi_gp"].items()}
                  if (self.noise_psi > 0.0 and self.K)
                  else self._zqpsi_d),
            qphi=({pv: arr(v) for pv, v in ctx["qphi_gp"].items()}
                  if self.noise_phi > 0.0 else self._zqphi_d),
            tq=(None if tq_gp is None
                else {pv: arr(v) for pv, v in tq_gp.items()}))
        ctx["_dev"] = dev
        return dev

    def _gp_eval_dev(self, x, ctx):
        """Device GP-field eval of x into _vals_dev/_grads_dev (the
        _pack_fields mirror).  Per-bin buffers are independent, so the
        result is order-invariant — shared by the CSR-scatter assembly
        (_assemble_device) and the matrix-free apply (apply_Jv)."""
        from ..assembly.gp_field import make_gp_multifield
        asm = self._asm
        d = self.dm.device
        nd = self.ndof
        X_d = wp.array(np.ascontiguousarray(
            x.reshape(self.dm.n_nodes, nd)), dtype=wp.float64, device=d)
        for k_bin, (pv, b, ne, nbf, _gd) in enumerate(asm._bins):
            gpk = make_gp_multifield(nbf, b["nqp"], self.dm.dim, nd)
            wp.launch(gpk, dim=ne,
                      inputs=[b["conn"], b["h"], b["N"], b["dN"], X_d,
                              self._vals_dev[pv], self._grads_dev[pv]],
                      device=d)

    def _fill_element_batches(self, ctx):
        """Generator over element-Jacobian fill: launch make_mpf_newton
        into the batch buffers (Ae [<=nb_cap, nl, nl], be) for every
        element batch and YIELD (k_bin, e0, nb, Ae, be).  The GP fields
        (_gp_eval_dev) must be current.  THE single source of the
        element-block fill — consumed by _assemble_device (scatter into
        the global CSR) AND by apply_Jv (matrix-free element matvec),
        so the two paths cannot drift."""
        asm = self._asm
        d = self.dm.device
        nd = self.ndof
        sigma = ctx["sigma"]
        drive_d, K_tot = ctx["drive_d"], ctx["K_tot"]
        minv, mlat, mvert = ctx["minv"], ctx["mlat"], ctx["mvert"]
        dtea, dtref = ctx["dtea"], ctx["dtref"]
        dev = self._dev_inputs(ctx)
        p = self._par
        for k_bin, (pv, b, ne, nbf, _gd) in enumerate(asm._bins):
            nqp = b["nqp"]
            nl = nd * nbf
            if k_bin not in self._dev_bufs:
                nb_cap = min(ne, max(1, (2 << 30) // (nl * nl * 8)))
                self._dev_bufs[k_bin] = (
                    nb_cap,
                    wp.zeros((nb_cap, nl, nl), dtype=wp.float64,
                             device=d),
                    wp.zeros((nb_cap, nl), dtype=wp.float64, device=d))
            nb_cap, Ae, be = self._dev_bufs[k_bin]
            kk = make_mpf_newton(nbf, nqp, self.dm.dim, self.M,
                                 self.K, self.bulk, self.mob,
                                 self.theta_mode,
                                 self.T_mode == "field",
                                 self.D_T is not None, self.aniso,
                                 self.film_on)
            tq_full = dev["tq"][pv] if dev["tq"] is not None else None
            xiy_full = self._xiy_wp[pv] if self.film_on else None
            for e0 in range(0, ne, nb_cap):
                nb = min(nb_cap, ne - e0)
                s0, s1 = e0 * nqp, (e0 + nb) * nqp
                Ae.zero_()
                be.zero_()
                wp.launch(kk, dim=nb, inputs=[
                    b["conn"], b["h"][e0:e0 + nb], b["N"], b["dN"],
                    b["w"],
                    self._vals_dev[pv][s0:s1],
                    self._grads_dev[pv][s0:s1],
                    dev["hist"][pv][s0:s1], dev["src"][pv][s0:s1],
                    dev["qpsi"][pv][s0:s1], dev["qphi"][pv][s0:s1],
                    p["chi_aa"], p["chi_ac"], p["chi_ca"], p["chi_cc"],
                    p["Ninv"], p["Ons"], p["dlo"], p["dhi"], p["Dslf"],
                    wp.float64(self.ls_drop[0]),
                    wp.float64(self.ls_drop[1]),
                    wp.float64(self.ls_drop[2]), p["kap"],
                    p["dsig"], drive_d, p["eps2"], p["Lpsi"],
                    p["alpha"], p["beta"], p["Lth"],
                    p["Tm"],
                    (tq_full[s0:s1] if tq_full is not None
                     else self._tq_dummy),
                    wp.float64(dtea), wp.float64(dtref),
                    p["da"], p["ma"], wp.float64(self.a_reg),
                    (xiy_full[s0:s1] if xiy_full is not None
                     else self._tq_dummy),
                    wp.float64(mlat), wp.float64(mvert),
                    wp.float64(minv), wp.float64(K_tot),
                    wp.float64(sigma), wp.float64(self.b_reg),
                    wp.float64(self.kg_delta),
                    wp.float64(self.p_floor),
                    Ae, be], device=d)
                yield k_bin, e0, nb, Ae, be

    def _assemble_device(self, x, ctx):
        """Device-path assembly at one Newton iterate: device GP-field
        eval (gp_multifield, the _pack_fields mirror), BATCHED element
        launches scattered through the slot maps into the device CSR
        values, host-valued O(surface) face terms slot-added, strong
        rows applied in-kernel.  Fills asm.vals_d / asm.F_d in place
        (free dofs == full dofs: identity constraints asserted).

        GP eval + element-block fill are factored into _gp_eval_dev /
        _fill_element_batches (the SAME blocks apply_Jv consumes); this
        method scatters them into the CSR.  Bit-identical to the former
        inline loop: GP writes per-bin-independent buffers, so hoisting
        the eval only reorders launches."""
        asm = self._asm
        d = self.dm.device
        nd = self.ndof
        arr = lambda a_: wp.array(np.ascontiguousarray(a_),
                                  dtype=wp.float64, device=d)
        t_new = ctx["t_new"]
        K_tot, minv = ctx["K_tot"], ctx["minv"]
        self._gp_eval_dev(x, ctx)
        asm.zero_fill()
        for k_bin, e0, nb, Ae, be in self._fill_element_batches(ctx):
            asm.scatter_batch(k_bin, e0, Ae, be, nb)
        # A2 wall face terms (host-valued; same weak terms as the
        # host path — module docstring A2)
        if self.wall_on:
            wf = (self.y_comp / self.h_curr) if self.film_on else 1.0
            if self._wall_slots_d is not None:
                asm.add_matrix_values(self._wall_slots_d,
                                      arr(wf * self._wall_jbase))
            loads = []
            for i in range(self.M):
                gi, hi = self.wall_g[i], self.wall_h[i]
                if gi == 0.0 and hi == 0.0:
                    continue
                fv = x[2 * i::nd]      # identity constraints: Tc@x = x
                fw = wf * (gi + 2.0 * hi * fv[self.wall_faces])
                loads.append(np.einsum("fab,fb->fa", self.wall_face_M,
                                       fw).ravel())
            asm.add_rhs_values(self._wall_rhs_dof_d,
                               arr(np.concatenate(loads)))
        # S3a top-surface enrichment flux (host-valued; docstring S3a)
        if self.film_on and K_tot > 0.0:
            jv, loads = [], []
            for i in range(self.M):
                coef = (K_tot - self.k_e[i]) * minv * self.y_comp
                Mf = coef * self.top_face_M
                jv.append(-Mf.ravel())
                fv = x[2 * i::nd]
                loads.append(np.einsum("fab,fb->fa", Mf,
                                       fv[self.top_faces]).ravel())
            asm.add_matrix_values(self._flux_slots_d,
                                  arr(np.concatenate(jv)))
            asm.add_rhs_values(self._flux_gdof_d,
                               arr(np.concatenate(loads)))
        # Dirichlet strong rows LAST (host order): identity row,
        # rhs = g - x (Newton-increment form)
        if self.dirichlet is not None:
            bv = []
            for f, gfn in enumerate(self.g_fns):
                if gfn is None:
                    continue
                gv = gfn(self.free_coords[self.dirichlet], t_new)
                for k2, i in enumerate(self.dirichlet):
                    bv.append(gv[k2] - x[i * nd + f])
            asm.apply_strong_rows(np.asarray(bv, np.float64))

    # == M5 matrix-free OUTER (2026-07-14-matrixfree-outer) =============
    # The monolithic Jacobian J is NEVER stored: J @ v is applied
    # batch-wise through the SAME device element blocks the CSR scatter
    # uses (fill Ae per batch -> Ae @ v_e -> scatter into y), plus the
    # O(surface) face-term + strong-row corrections.  Enables the
    # 256x256x128 (M=3, K=2) system whose stored CSR (~148 GB) fits on
    # NO single card; the blockch W-factor inners stay stored (node-
    # pattern sized).  See docs/dev/2026-07-14-matrixfree-outer.md.
    def _matfree_setup(self, x, ctx):
        """Freeze the linearization point for the matrix-free J @ v:
        GP-eval x ONCE (Ae is x-only, identical across the outer Krylov
        matvecs — recomputed per apply since the full element-block set
        is too large to store, e.g. ~430 GB at 256^3), and precompute
        the x-independent face-term Jacobian correction (host CSR, tiny)
        + strong-row indices."""
        assert self._asm.node_mode, (
            "matrix-free apply: node-graph pattern (large 3-D) only")
        assert self.dirichlet is None or True   # handled in apply_Jv
        self._gp_eval_dev(x, ctx)
        self._mf_ctx = ctx
        self._mf_face = self._build_face_jac(x, ctx)
        if self.dirichlet is not None:
            nd = self.ndof
            self._mf_strong = np.array(
                [i * nd + f for f, gfn in enumerate(self.g_fns)
                 if gfn is not None for i in self.dirichlet], np.int64)
        else:
            self._mf_strong = None

    def _build_face_jac(self, x, ctx):
        """The face-term Jacobian entries (A2 wall (mu_i,phi_i) block +
        S3a top-flux (phi_i,phi_i) block) as a host CSR over the full
        (== free) dof space.  These entries are x-INDEPENDENT within an
        attempt (geometric face-mass * frozen coefficients), so the CSR
        is built once per outer solve and applied on the host as a small
        correction to the device element matvec.  Returns None when no
        face terms are active."""
        nd = self.ndof
        K_tot, minv = ctx["K_tot"], ctx["minv"]
        N = self.nfree * nd
        rows, cols, vals = [], [], []
        if self.wall_on:
            nfn = self.wall_faces.shape[1]
            wf = (self.y_comp / self.h_curr) if self.film_on else 1.0
            for i in range(self.M):
                gi, hi = self.wall_g[i], self.wall_h[i]
                if hi == 0.0:
                    continue
                gd = nd * self.wall_faces + (2 * i + 1)
                cd = nd * self.wall_faces + 2 * i
                rows.append(np.repeat(gd, nfn, axis=1).ravel())
                cols.append(np.tile(cd, (1, nfn)).ravel())
                vals.append((-2.0 * hi * wf) * self.wall_face_M.ravel())
        if self.film_on and K_tot > 0.0:
            nfn = self.top_faces.shape[1]
            for i in range(self.M):
                coef = (K_tot - self.k_e[i]) * minv * self.y_comp
                if coef == 0.0:
                    continue
                gd = nd * self.top_faces + 2 * i
                rows.append(np.repeat(gd, nfn, axis=1).ravel())
                cols.append(np.tile(gd, (1, nfn)).ravel())
                vals.append((-coef * self.top_face_M).ravel())
        if not rows:
            return None
        return sp.coo_matrix(
            (np.concatenate(vals),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(N, N)).tocsr()

    def apply_Jv(self, v):
        """Matrix-free J @ v at the frozen _matfree_setup point.  Host
        v -> device; per element batch: refill Ae (make_mpf_newton) and
        Ae @ v_e scatter-added into y (never assembles the global CSR);
        then the host face-term correction and strong-row identity.
        Returns host y.  Cost basis = one element-fill pass per matvec
        (the outer-FGMRES J.v; see the dev note's measured table)."""
        asm = self._asm
        d = self.dm.device
        v_d = wp.array(np.ascontiguousarray(v), dtype=wp.float64,
                       device=d)
        y_d = wp.zeros(asm.Nfull, dtype=wp.float64, device=d)
        for k_bin, e0, nb, Ae, be in self._fill_element_batches(
                self._mf_ctx):
            asm.apply_batch_matvec(k_bin, e0, Ae, nb, v_d, y_d)
        y = y_d.numpy()
        if self._mf_face is not None:
            y += self._mf_face @ v
        if self._mf_strong is not None:
            y[self._mf_strong] = v[self._mf_strong]
        return y

    def _solve_dev(self):
        """Linear solve on the device-resident CSR.  cudss: zero-copy
        torch CSR (dlpack over asm.vals_d) with STABLE operands — plan
        ONCE (the pattern is fixed by the mesh graph, so nnz NEVER
        flaps, unlike the host T^T K T which prunes exact zeros — the
        S2 plan-flapping finding), refactorize per iterate; PLAIN
        DirectSolverOptions (no mt layer — the gomp thread-leak
        finding, replication ledger Sec 9); explicit .free() when a
        plan is dropped (the discarded-plan double-free finding).
        splu: host pull on the fixed pattern (the parity-gate
        solver)."""
        asm = self._asm
        if self.linsolver in ("blockch", "blockch_dev"):
            # B2: device-resident blockch — per-pair/AC-block values
            # built by fill/gather kernels on the assembler's slot-map
            # CSR (no host matrix ever exists); same divergence
            # contract as the host branch.
            from ..solvers.linsolve import blockch_pairs_device
            asm.device_operator()      # ensures the _op_idx upload
            try:
                return blockch_pairs_device(
                    asm.indptr, asm.indices, asm.vals_d,
                    asm.F_d.numpy(), self._blockch_meta(), tol=1e-10,
                    device=self.dm.device, cache=self._solver_cache,
                    cache_key="mpf_dev", idx_dev=asm._op_idx)
            except RuntimeError:
                return np.full(asm.Nfull, np.nan)
        if self.linsolver == "cudss":
            import torch
            from nvmath.sparse.advanced import (DirectSolver,
                                                DirectSolverOptions)
            try:
                if self._cudss_dev is None:
                    A_t, F_t = asm.device_csr()
                    self._F_view = F_t     # zero-copy over asm.F_d
                    self._b_t = torch.empty_like(F_t)
                    self._b_t.copy_(self._F_view)
                    self._cudss_dev = DirectSolver(
                        A_t, self._b_t,
                        options=DirectSolverOptions(blocking=True))
                    self._cudss_dev.plan()
                    self._n_dev_plans += 1
                else:
                    self._b_t.copy_(self._F_view)
                self._cudss_dev.factorize()
                return np.asarray(self._cudss_dev.solve().cpu())
            except Exception:
                try:
                    self._cudss_dev.free()
                except Exception:
                    pass
                self._cudss_dev = None
                return np.full(asm.Nfull, np.nan)
        assert self.linsolver == "splu", (
            "assembly='device': linsolver in ('cudss', 'splu', "
            "'blockch', 'blockch_dev')")
        from scipy.sparse.linalg import splu
        A = sp.csr_matrix((asm.vals_d.numpy(), asm.indices,
                           asm.indptr), shape=(asm.Nfull,) * 2)
        return splu(A.tocsc()).solve(asm.F_d.numpy())

    def _attempt_device(self, dt):
        """One implicit solve with DEVICE-BOUND assembly (wodo v1.2
        model): the same ctx-frozen physics and the same Newton
        safeguards as the host _attempt; assembly fills the device
        CSR in place, the solve consumes it zero-copy.  Line-search
        difference (documented): the device buffers hold the LAST
        evaluated trial, so when the accepted trial is not the last
        one it is re-assembled (same accepted point; one extra
        assembly on the rare non-monotone backtrack)."""
        if self._asm is None:
            self._init_device_assembly()
        ctx = self._attempt_ctx(dt)
        asm = self._asm
        ls = self.line_search

        def assemble(x):
            self._assemble_device(x, ctx)
            return asm.F_d.numpy() if ls else None

        x = self.x.copy()
        r = assemble(x)
        for it in range(self.newton_max):
            dx = self._solve_dev()
            if not np.isfinite(dx).all():
                return None, it + 1, False               # diverged
            if ls:
                # rescale finite oversized directions (host semantics;
                # the degenerate-row garbage-direction finding)
                mx = np.abs(dx).max()
                if mx > 2.0:
                    dx = dx * (2.0 / mx)
            elif np.abs(dx).max() > 1e6:
                return None, it + 1, False               # diverged
            if self.guards:
                inc = max(np.abs(dx[2 * i::self.ndof]).max()
                          for i in range(self.M))
                for k in range(self.K):
                    inc = max(inc, np.abs(
                        dx[2 * self.M + 2 * k::self.ndof]).max())
                if inc > 2.0:
                    dx = dx * (2.0 / inc)
            if ls:
                rn0 = float(np.linalg.norm(r))
                best_x, best_rn, last_best = None, np.inf, False
                for s in (1.0, 0.5, 0.25, 0.125, 0.0625):
                    xt = x + s * dx
                    if self.guards:
                        xt = self._project(xt)
                    rt = assemble(xt)
                    rnt = float(np.linalg.norm(rt))
                    if rnt < best_rn:
                        best_x, best_rn, r = xt, rnt, rt
                        last_best = True
                    else:
                        last_best = False
                    if rnt < rn0 * (1.0 - 1e-4):
                        break
                if not last_best:
                    r = assemble(best_x)   # buffers -> accepted point
                conv = np.abs(best_x - x).max()
                x = best_x
            elif self.guards:
                xn = self._project(x + dx)
                conv = np.abs(xn - x).max()
                x = xn
                r = assemble(x)
            else:
                x = x + dx
                conv = np.abs(dx).max()
                r = assemble(x)
            if conv < self.newton_tol:
                return x, it + 1, True
        return x, self.newton_max, False                 # no convergence

    def _debug_assemble(self, dt):
        """Parity instrumentation (D1 gates): ONE assembly at the
        current committed iterate self.x with attempt-frozen inputs;
        returns (A, r) on the free dofs for the ACTIVE assembly mode
        (host: pruned scipy CSR; device: CSR over the pulled device
        values — the fixed superset pattern, canonicalize before
        comparing).  Consumes the same RNG draws as one attempt —
        seed-align the steppers before calling."""
        ctx = self._attempt_ctx(dt)
        if self.assembly == "device":
            if self._asm is None:
                self._init_device_assembly()
            self._assemble_device(self.x.copy(), ctx)
            asm = self._asm
            A = sp.csr_matrix((asm.vals_d.numpy(), asm.indices,
                               asm.indptr), shape=(asm.Nfull,) * 2)
            return A, asm.F_d.numpy().copy()
        return self._assemble_host(self.x.copy(), ctx)

    def step(self):
        """One FIXED-dt BDF1 step (gates/MMS/parity path — no ladder)."""
        x, iters, ok = self._attempt(self.dt)
        assert ok, f"Newton failed at fixed dt (iters={iters})"
        self.x = x
        if self.tstep == "bdf2":       # shift the two-level history
            self.hist2 = self.hist
            self.dt_prev = self.dt
        self.hist = x.copy()
        self.t += self.dt
        if self.T_mode == "field":
            self.T_nodes = self._T_pend    # commit the segregated T
        if self.film_on:
            # explicit h-update with the SAME frozen K the attempt
            # used (wodo contract; exact BDF1 content pairing)
            self.h_curr -= self.dt * self._K_pend
        return x

    def march(self, t_end, max_steps=100000, dt_min=1e-12, dt_max=None,
              callback=None, grow_iters=20, h_min=None,
              phis_stop=None):
        """Appendix-A ladder (wodo pattern): reject (no convergence/
        divergence) => dt *= 0.25 retry; accept with iters <
        grow_iters => dt *= 1.25 (capped).  grow_iters default 20 (the
        wodo constant); KWC grain-boundary states carry the KG-Picard
        LINEAR Newton tail (measured 0.5-0.9 contraction), where 20
        starves dt growth — raise it there.  FILM MODE (S3a): the
        evaporation dt-cap dt <= dh_cap/K bounds the per-step height
        decrement (the h-update is explicit), and the optional film
        stop criteria h_min (physical height floor) / phis_stop (avg
        solvent fraction = dryness) apply.  Returns stop reason."""
        reason = "max_steps"
        for _ in range(max_steps):
            if self.t >= t_end - 1e-14:
                reason = "t_end"
                break
            if self.film_on:
                if h_min is not None and self.h_curr <= h_min:
                    reason = "h_min"
                    break
                if phis_stop is not None:
                    phis = 1.0 - sum(
                        np.asarray(self.Tc @ self.x[2 * i::self.ndof])
                        for i in range(self.M))
                    if float(np.mean(phis)) <= phis_stop:
                        reason = "phis_stop"
                        break
            dt_eff = min(self.dt, t_end - self.t)
            if self.film_on:
                # evaporation dt-cap (wodo dh_cap): bound dh per step
                Kt = self._film_K()
                if Kt > 0.0:
                    dt_eff = min(dt_eff, self.dh_cap / Kt)
            x_new, iters, ok = self._attempt(dt_eff)
            if not ok:
                self.n_reject += 1
                self.dt = dt_eff * 0.25
                if self.dt < dt_min:
                    reason = "dt_underflow"
                    break
                continue
            self.x = x_new
            if self.tstep == "bdf2":   # shift the two-level history
                self.hist2 = self.hist
                self.dt_prev = dt_eff
            self.hist = x_new.copy()
            self.t += dt_eff
            if self.T_mode == "field":
                self.T_nodes = self._T_pend    # commit segregated T
            if self.film_on:
                self.h_curr -= dt_eff * self._K_pend
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
