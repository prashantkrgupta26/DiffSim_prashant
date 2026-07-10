# Finding: film front end + Negi-2018 2-D validation (honest verdict)

Date: 2026-07-10. Deliverable: `diffsim.film` (the PI-facing
parametrized front end + four-layer RunLog), 13 named configs,
3-D cluster kits. Gates: `tests/test_film_frontend.py` (tier3).
All runs below: RTX 6000 Ada (cuda:1), cudss device-bound, full config
resolution 125x250 (the paper-scale 4 nm mesh over 500x1000 nm).

## What ran

The four named configs `negi2018_{6000,3000,1500,500}rpm.yaml`
(N = (87, 5, 1), chi = (1.0, 0.1, 0.9), 1:2 PDPP5T:PC71BM by volume at
90% solvent, Bi = 0.0073/0.0051/0.0033/0.0019, eps^2 = 1e-10 J/m
-> kappa = 3.235e-6 via kappa = eps^2/((RT/V_s) h0^2) at h0 = 1 um,
V_s = 80.7 cm^3/mol, T = 300 K; D_p = 1e-3, D_f = 5e-3, CHC noise 1e-3,
b_reg 1e-3), plus two time-scale diagnostics of the 6000 rpm case at
Bi x100 = 0.73 and Bi x1000 = 7.3.

## Measured table

| case | steps | wall | mass drift | phi range | onset vert (theta) | onset lat (theta) | L_c final |
|---|---|---|---|---|---|---|---|
| 6000 rpm | 304 | 104 s | 2.0e-15 | [0.033, 0.925] | 1.00 (top) | 0.00 | 0.0185 |
| 3000 rpm | 285 | 88 s | 2.2e-15 | [0.030, 0.932] | 1.00 (top) | 0.00 | 0.0183 |
| 1500 rpm | 263 | 66 s | 2.2e-15 | [0.032, 0.930] | 1.00 (top) | none | 0.0188 |
| 500 rpm | 283 | 80 s | 2.2e-15 | [0.033, 0.926] | 1.00 (top) | none | 0.0186 |
| diag Bi=0.73 | 448 | 276 s | 2.0e-15 | [-0.002, 0.943] | 1.00 (h=0.35) | 0.46 (h=0.25) | 0.0202 |
| diag Bi=7.3 | 301 | 214 s | 1.5e-15 | [0.009, 0.978] | 1.00 (h=0.89) | 0.22 (h<0.3) | 0.0368 |

(L_c = detrended first-moment structure-factor wavelength of phi_f,
units of h0 = 1000 nm; onset detectors at the house amplitude 0.1 —
film/analysis.py docstring.)

## Verdict vs the paper's expectations

REPRODUCED (asserted in the gate):
- All four cases march to dryness (phi_s < 0.05, ~300 accepted steps)
  with ZERO autopsies and solute-content drift at the 2e-15 class
  (asserted < 1e-10, 4+ decades headroom).
- Separation initiates AT THE TOP: the vertical onset (first order-one
  row-mean phi_f deviation) fires at theta = 1.0 in every case — the
  less-soluble fullerene (chi_fs = 0.9 >> chi_ps = 0.1) enriches at
  the solvent-lean free surface, the fig7 selectivity mechanism.
- Final films are near-pure PC71BM(top)/PDPP5T(bottom) bilayers
  (row-mean phi_f 0.93 top vs 0.05 bottom).

REPLICATION GAPS (printed by the gate, not asserted):
- The rpm ladder does NOT differentiate: Bi = 0.0019..0.0073 under the
  D_s-based mapping is quasi-static (evaporation Peclet K h / D ~
  1e-3), so all four rpm give the SAME equilibrium bilayer; final
  lateral structure is noise-scale (max lateral std 0.001-0.007) and
  L_c is flat (0.0183-0.0188), not monotone in rpm. The paper's
  500-rpm bulk-onset / larger-domain contrast and 6000-rpm lateral
  surface-directed morphology are absent.

DIAGNOSIS (measured, the nondimensionalization is the suspect):
- Scaling ONLY Bi restores the paper's phenomenology. At Bi x100 the
  6000 rpm case develops a genuine surface-directed gradient (row-mean
  phi_f 0.40 top vs 0.18 bottom at h = 0.3, BEFORE lateral onset),
  lateral onset mid-film, and arrested lateral domains that survive to
  dryness (max lateral std 0.15). At Bi x1000 the surface layer forms
  at h = 0.89 and the final L_c doubles. Interpretation: the paper's
  phenomenology needs an evaporation/diffusion ratio ~2-3 orders
  larger than our mapping produces. The Bi VALUES themselves are
  plausible (t_final = 204 at 6000 rpm is 0.2 s physical at
  D_s = 1e-9 m^2/s — a real spin-coating time), so the PRIME suspect
  is the local diffusivity closure, not k_e: our v2 mobility is the
  LINEAR mix D = phi_s + D_p phi_p + D_f phi_f, which sits at 0.90 D_s
  for the 90%-solvent film — a free-volume (Vrentas-class) D(phi) for
  a real polymer solution is already 1-2 orders below neat D_s at 10%
  solids, exactly the x100 the diagnostic needed. Pin the paper's
  D(phi) model (and its D_s reference) before the 3-D campaign burns
  GPU-days; the front end exposes the ratio knobs (D_p, D_f) but not
  yet a nonlinear D(phi_s) shape — recorded as the follow-up if the
  paper's closure turns out non-linear.

Evidence: `benchmarks/data/negi2018/` (gitignored runtime data; the
RunLogs carry full provenance). Preflight note: the physical eps^2
under-resolves forming interfaces at the 4 nm mesh (xi_early ~ 1.5
elements < the validated 2.5 floor) — the negi configs run with
`preflight: warn` and the excursion monitor stayed clean
(phi within [-0.002, 0.978] across all six runs).
