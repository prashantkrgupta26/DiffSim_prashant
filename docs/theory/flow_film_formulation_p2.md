# P2 — Realism Pack + Diffuse Film-Air Interface + Flow: Formulation

For Baskar's ratification (2026-07-11). Extends P1 (crystallization).
Anchors: Baskar's AM governing-equations document (MyPapers/AM/compass_
artifact_*.md — the CAC/rheology recommendations are adopted from it),
Ronsin-Harting 2204.11628 (vapor-field evaporation, Hertz-Knudsen),
Gu-Diao-Ganapathysubramanian-Toney-Bao Org. Electron. 2017 (MyPapers/
OSC/1-s2.0-S1566119916304608: solution shearing of isoindigo/PC61BM —
THE validation campaign: domain size + crystallinity vs T and shear
speed, GIWAXS RDoC, 2pi/q_peak; Jsc-vs-domain-size = the (c) hook),
Suresh neural-constitutive paper (MyPapers/AM/ — the v3 closure form).

## Track A — the realism pack (M5 additions, build after S2 lands)

A1. TEMPERATURE AS A FIELD. VMS scalar transport (M2 brick) coupled in:
    rho cp (T_t + u.grad T) = div(k grad T) + s_evap, with the
    evaporative-cooling sink s_evap = -L_vap J_evap at the interface
    (A3/B-coupled) and substrate/ambient BCs (stage temperature = the
    Gu-2017 knob). All chi(T), D(T) (WLF/Arrhenius), dh(1-T/Tm) terms
    already parameterized — they just read the field. Annealing
    protocols become boundary conditions instead of global schedules.
A2. SUBSTRATE SURFACE ENERGY. Wall free energy f_w(phi)|_substrate =
    sum_i g_i phi_i (+ optional quadratic) -> natural BC entering the
    mu equations at the substrate face; sets effective contact/wetting
    preferences (vertical stratification control; Bergermann-style
    patterned g_i(x) supported by construction).
A3. ANISOTROPIC CRYSTAL GROWTH. eps_k^2 -> eps_k^2(theta_grain, n_hat):
    eps(1 + delta_a cos(m(angle(grad psi) - theta)))^2 in 2-D (m-fold
    anisotropy; needles/fibrils at large delta_a). Uses the EXISTING
    theta field; adds the standard anisotropic AC terms (d(eps^2)/d
    angle contributions). Gates: single-seed aspect-ratio vs delta_a;
    kinetics unchanged at delta_a=0 (regression).

## Track B — diffuse film-air interface + flow (the M6 centerpiece)

B1. INTERFACE: conservative Allen-Cahn (CAC, Joshi-Jaiman/Mirjalili
    class) for phi_vap — second-order, interface-thickness preserving,
    no curvature shrinkage of thin films; Lagrange-multiplier global
    mass correction. Film thickness h(x,y,t) = the phi_vap=1/2 level
    set: HETEROGENEOUS by construction (replaces the Landau frame's
    flat h). The Landau-frame film stepper REMAINS the fast path for
    flat-film campaigns (measured, validated); CAC mode is the
    heterogeneous/flow path. CH/AGG variant retained for
    high-density-ratio consistency studies (per the AM doc).
B2. EVAPORATION: interface-localized Hertz-Knudsen sink
    J_i = a_i(T) (p_sat,i(T)/P0 - phi_i^inf) |grad phi_vap| per
    volatile species — per-species volatility = solvent blends (S4a
    synergy); T-coupled through p_sat(T) and the A1 cooling sink.
B3. FLOW: variable-(rho, eta) incompressible NS-VMS (creeping regime;
    the M1b/M2 stack), quasi-incompressible with AGG flux where CH is
    used; surface tension in potential form mu grad phi (Jacqmin);
    MARANGONI: sigma(T, phi_s) tangential interfacial force — the
    driver of thickness heterogeneity (Benard-Marangoni, edge/ring
    flows). Blade/shear driving: moving-wall BC (solution shearing =
    Couette-like gap flow, the Gu-2017 geometry).
B4. CONSTITUTIVE HIERARCHY (the non-Newtonian ruling):
    v1 WORKHORSE — generalized Newtonian, three factors:
        eta(phi_s, T, gdot) = CarreauYasuda(gdot; eta0(phi_s,T), lam, n)
        eta0 = eta_ref * exp(k_c (1-phi_s)) * WLF(T)   [flow arrest =
        6-9 decades through drying; the rheological twin of mobility
        freeze-out]
    v2 — viscoelastic log-conformation (Giesekus/PTT) where Wi ~ O(1)
        (high-MW under blade shear; conformational alignment +
        shear-induced nucleation via chain extension, Gu-2017 p4).
    v3 — frame-invariant neural closure (Suresh form): invariant
        inputs (gdot, phi_s, T, state), tensor-basis outputs,
        thermo-consistency constraints; v1 as the prior. SHARED with
        the AM program (one rheology module, two applications).
B5. ALIGNMENT: v1 = advection of all phase fields (Ronsin advective
    forms) + A3 anisotropy -> shear-emergent alignment; v2 = Jeffery-
    class theta-vorticity rotation if emergent under-predicts.
B6. COUPLING TO CRYSTALLIZATION: shear-induced nucleation enters
    naturally through v2's conformation tensor (trace -> effective
    undercooling shift) — recorded as the B-v2/M6-learnable frontier;
    v1 captures the evaporation-coarsening race that Gu-2017's model
    identifies as dominant for THEIR system (spinodal-dictated domains).

## Validation campaign (Gu 2017 — Baskar's own data)

Config: isoindigo-TT/PC61BM 10:15 mg/mL in chlorobenzene, 30 um gap,
stage T + shear speed sweeps. Gates: (i) domain size vs T at fixed
speed (their factor-2 tuning; lower T -> smaller domains); (ii) RDoC
trend vs T (lower T -> HIGHER crystallinity — the nontrivial one:
evaporation-coarsening race + crystallization kinetics); (iii) domain
size vs shear speed; observables = S(q) ring (2pi/q_peak) + grain
RDoC analogue (crystalline fraction). Quantitative where their figures
state numbers; the honest-verdict discipline throughout. Their
chi/N/D/p_sat for isoindigo-TT: partially unknown — parameter
estimation via the M6 learning machinery is the recorded fallback
(fit chi/D to the T-sweep, predict the speed sweep = a genuine
validation, not a fit).

## Sequencing

After S2: A-pack (agent, ~1-2 days) -> S3 (film + crystallization,
uses A1/A2) -> S4 closes M5. Track B = M6's build centerpiece
alongside learning rung 2 (B4-v3 IS a learning rung); B validation =
the Gu-2017 campaign. Track C (structure-property) planning proceeds
from MyPapers/OSC/StructureProperty + the excitonic_drift_diffusion
code (tarball pending).

## Rulings requested

RB1. CAC-vs-Landau-frame coexistence as designed (two film modes)? 
RB2. Constitutive v1 (generalized Newtonian) as the S/M6 entry, v2
     viscoelastic deferred until the Gu speed-sweep demands it?
RB3. Gu-2017 as Track B's headline validation campaign (with
     parameter-estimation fallback for unknown isoindigo params)?
