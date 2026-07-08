# P0 — Ternary Phase-Field with Evaporation and Crystallization
## The M4 formulation memo (student-facing; the system everything else builds on)

*Drafted 2026-07-08 (night). Sources: Raghu_Learning_Free_energy.pdf
(coupled c-eta model), the group FEM guide (canonical AC/CH weak forms),
Wodo & Ganapathysubramanian JCP 2011 (adaptive implicit CH) and CMS 2012
(solvent-based OSC fabrication — the replication target), both in
MyPapers/LearningFreeEnergy/.*

## 1. Fields and the simplex reduction

Volume fractions: phi_1 (donor/polymer), phi_2 (acceptor/small
molecule), phi_s (solvent), with phi_1 + phi_2 + phi_s = 1. EVOLVE
(phi_1, phi_2); eliminate phi_s = 1 - phi_1 - phi_2. Crystallinities:
eta_1, eta_2 (non-conserved; either may be disabled). Mixed CH form
carries exchange chemical potentials (mu_1, mu_2). Monolithic unknowns:
(phi_1, mu_1, phi_2, mu_2 [, eta_1, eta_2]) — 4 to 6 fields.

## 2. Free energy (the learnable object)

F[phi, eta] = Int_Omega [ f_mix(phi_1, phi_2)
    + sum_i phi_i f_cr,i(eta_i) + f_couple(phi, eta)
    + W_12 eta_1^2 eta_2^2                        (crystal exclusion)
    + sum_i (kappa_i/2)|grad phi_i|^2 + kappa_12 grad phi_1 . grad phi_2
    + sum_i (kappa_eta,i/2)|grad eta_i|^2 ] dV

Baseline f_mix = ternary Flory-Huggins:
    f_mix = (phi_1/N_1) ln phi_1 + (phi_2/N_2) ln phi_2
          + phi_s ln phi_s + chi_12 phi_1 phi_2
          + chi_1s phi_1 phi_s + chi_2s phi_2 phi_s
(k_B T / v_site = 1 after nondimensionalization). The LEARNED f_mix
replaces this with a basis expansion / Lipschitz-MLP surface over the
simplex (M4 track e); FH remains the reference misfit ladder rung.
Crystallization wells f_cr,i: tilted double wells per Raghu's Eq. 3;
coupling f_couple = sum_i chi_c,i phi_i(1-phi_i) eta_i^2.

GAUGE NOTE (identifiability): F is observable only up to terms affine
in each conserved phi_i (mu shifts by constants leave the dynamics
invariant). Anchor the learned f_mix at reference compositions; same
for f_cr,i up to constants.

## 3. Dynamics

Conserved (coupled CH, Onsager):
    d(phi_i)/dt = div( sum_j M_ij(phi) grad mu_j ),     i,j in {1,2}
    mu_j = df/d(phi_j) - kappa_j lap phi_j - kappa_12 lap phi_(3-j)
(exchange potentials: derivatives at eliminated phi_s; the ln phi_s
terms contribute -ln phi_s - 1 - chi_is(...) pieces with OPPOSITE sign
— derive carefully, gate by MMS.) Mobility matrix SPD, degenerate:
    M_ii = M0_i phi_i (1 - phi_i),  M_12 = -M0_12 phi_1 phi_2  (>= 0
    eigenvalues enforced; M_ij are per-GP FIELDS — the closure
    interface — so learned/composition-dependent mobilities are free).
Non-conserved (AC):
    d(eta_i)/dt = -L_i [ phi_i f_cr,i'(eta_i) + df_couple/d(eta_i)
                         + 2 W_12 eta_i eta_(3-i)^2
                         - kappa_eta,i lap eta_i ]

## 4. Evaporation (staged fidelity)

E0 (validation only): prescribed mean-solvent drift (bulk sink).
E1 (THE M4 WORKHORSE, Wodo-CMS-2012 class): fixed domain, evaporative
   flux BC on the top surface: - sum_j M_ij grad mu_j . n = k_e,i *
   (phi_s|surf - phi_s,amb) delta_is-weighted (solvent leaves; solutes
   no-flux). Captures the through-thickness solvent gradient -> vertical
   stratification. Robin/Neumann face term — existing machinery.
E2 (stretch; M3 synergy): receding top surface as an SBM moving
   boundary (film thinning h(t)); the level-set surface + epoch/transfer
   machinery from M3 rungs 1-2. h(t) is also the cheapest operando
   observable (pins k_e — identifiability).

## 5. Discretization (per the group guide + Wodo JCP 2011)

Weak mixed form, equal-order elements (p1 AND p2 — the lapN/VMS-complete
infrastructure exists); backward Euler and BDF2, fully implicit
monolithic Newton (block Jacobian per Suresh's derivation pattern;
d(mu)/d(phi) includes f''_mix — at learned f, the network's derivative
enters HERE: keep f' as the primitive so Newton needs f'' = one
autograd/basis-derivative). TEMPORAL ADAPTIVITY: BDF1/BDF2 with local
truncation-error control (Wodo JCP 2011's adaptive implicit scheme = the
reference design; dt grows through coarsening, shrinks at nucleation
events). SPATIAL ADAPTIVITY: octree refinement tracking |grad phi|
(interface bands) — refine/coarsen per epoch, state transfer via the M3
rung-1 P operator (adjoint-exact by construction).

## 6. Gates (the M4 ritual)

MMS orders (p1: 2; p2: 3) per field and coupled; MASS conservation to
1e-12/step (CH); ENERGY DECAY monotone (no-evaporation cases);
shrinking-circle (AC) + spinodal-decomposition (CH) benchmarks per the
guide; Wodo-CMS-2012 replication (track c): 1D/2D solvent-evaporation
morphology evolution — match the paper's regimes (surface-directed
spinodal waves, stratification vs blend ratio); everything DEVICE-RUN
(A100 class: M1d stack — device assembly + cuDSS + mtlayer).

## 7. Nondimensionalization

Length: film thickness h0. Time: h0^2/(M0 kT/v). Energy: kT/v_site.
Groups: Cn_i = sqrt(kappa_i)/h0 (Cahn numbers), evaporation Biot
Bi = k_e h0/(M0 kT/v), mobility ratios, chi's. The Wodo papers' regime
map is in these groups — replicate in THEIR nondimensionalization for
the track-(c) comparison, document the mapping.
