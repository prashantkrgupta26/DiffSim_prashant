# Projection Validation Ladder — benchmark references

Reference values the rung drivers (Tasks 2–7) anchor against. **The same-mesh
monolithic oracle is the PRIMARY, box-free bar** (projection Cd must equal
monolithic Cd on the identical carved mesh within R0 tolerance). Literature is
the SECONDARY absolute anchor; because the ladder uses a coarse carved octree on
a confined unit-cube channel (not the benchmark's exact domain/refinement),
literature agreement is expected only to benchmark tolerance and some numbers
below are flagged **(to confirm)** — approximate literature is acceptable when
flagged, per the spec. The fixture that produces these meshes is
`tests/ladder_fixtures.py`.

## Fixture geometry (as built)

- Domain: octree unit cube `[0,1]^dim` = the channel (physical coords).
- Obstacle: `Box(center, half)`, side `D = 2*half`. Aligned `half = k/2^level`
  (e.g. `half=0.25` at level 4) ⇒ **exact body-fitted carve**: `|d|_max = 0`,
  `geo.corr ≡ 1.0` (SBM Nitsche ≡ standard Nitsche). A sub-cell center
  `offset` ⇒ **genuine SBM shift**: `0 < |d|_max < h` (verified: level-4
  `offset=0.03` ⇒ `|d|_max ≈ 0.0325`, `h = 0.0625`).
- Blockage as built with the default center `(0.5, 0.5[, 0.5])` and
  `half = 0.25`: `β = D / L_y = 0.5 / 1.0 = 1/2`. This is a HIGH blockage vs the
  Breuer β = 1/8 channel — the confinement raises Cd and shifts St, so the
  literature values below are directional anchors and the **same-mesh oracle is
  decisive**. If a driver needs the low-β regime it must place a thinner box
  (`half = k/2^level` small) in a longer channel; the fixture supports any
  aligned `half` and center. Recorded here so the blockage is explicit, not
  implied.
- Inflow: strong uniform `U_IN = 1` at `x = 0`. Outflow: free at `x = 1` (Taly
  physical-pressure node set). Lateral faces: no-slip walls (y in 2-D; y,z in
  3-D). Re via `nu = U_IN * D / Re` (D = obstacle side).

## Rung 0 — lid-driven cavity (Ghia, Ghia & Shin 1982)

Unit square, moving lid `U = 1` at `y = 1`, no-slip on the other three walls,
Re = 100 and 400. Pass = projection centerline profiles match the monolithic on
the same mesh AND track Ghia within a few % at the tabulated points.

Reference: **Ghia, U., Ghia, K.N., Shin, C.T. (1982).** "High-Re Solutions for
Incompressible Flow Using the Navier–Stokes Equations and a Multigrid Method."
*J. Comput. Phys.* 48, 387–411. Tables I & II: `u` along the vertical
centerline `x = 0.5` and `v` along the horizontal centerline `y = 0.5`.

`u(y)` on the vertical centerline `x = 0.5` (Ghia Table I):

> **Corrected 2026-07-23 (Task 2 / rung 0).** The earlier version of this table
> paired the upper *u*-values with the WRONG y-stations — it listed
> `0.9531 / 0.8516 / 0.7344` where Ghia Table I has the four closely-spaced
> stations `0.9688 / 0.9609 / 0.9531` (two stations, `0.9688` and `0.9609`,
> had been dropped), and the Re=400 column below `y ≈ 0.6` was mis-shifted.
> The values below are the canonical Ghia (1982) Table I (cross-checked against
> the long-standing `tests/test_cavity.py` Re=100 table and the rung-0 driver's
> monolithic same-mesh oracle). The rung-0 driver `tests/ladder_rung0_cavity.py`
> carries the same corrected values.

| y | Re=100 | Re=400 |
|---|---|---|
| 1.0000 | 1.00000 | 1.00000 |
| 0.9766 | 0.84123 | 0.75837 |
| 0.9688 | 0.78871 | 0.68439 |
| 0.9609 | 0.73722 | 0.61756 |
| 0.9531 | 0.68717 | 0.55892 |
| 0.8516 | 0.23151 | 0.29093 |
| 0.7344 | 0.00332 | 0.16256 |
| 0.6172 | −0.13641 | 0.02135 |
| 0.5000 | −0.20581 | −0.11477 |
| 0.4531 | −0.21090 | −0.17119 |
| 0.2813 | −0.15662 | −0.32726 |
| 0.1719 | −0.10150 | −0.24299 |
| 0.1016 | −0.06434 | −0.14612 |
| 0.0703 | −0.04775 | −0.10338 |
| 0.0625 | −0.04192 | −0.09266 |
| 0.0547 | −0.03717 | −0.08186 |
| 0.0000 | 0.00000 | 0.00000 |

`v(x)` on the horizontal centerline `y = 0.5` (Ghia Table II):

| x | Re=100 | Re=400 |
|---|---|---|
| 1.0000 | 0.00000 | 0.00000 |
| 0.9688 | −0.05906 | −0.12146 |
| 0.9609 | −0.07391 | −0.15663 |
| 0.9531 | −0.08864 | −0.19254 |
| 0.9453 | −0.10313 | −0.22847 |
| 0.9063 | −0.16914 | −0.23827 |
| 0.8594 | −0.22445 | −0.44993 |
| 0.8047 | −0.24533 | −0.38598 |
| 0.5000 | 0.05454 | 0.05186 |
| 0.2344 | 0.17527 | 0.30174 |
| 0.2266 | 0.17507 | 0.30203 |
| 0.1563 | 0.16077 | 0.28124 |
| 0.0938 | 0.12317 | 0.22965 |
| 0.0781 | 0.10890 | 0.20920 |
| 0.0703 | 0.10091 | 0.19713 |
| 0.0625 | 0.09233 | 0.18360 |
| 0.0000 | 0.00000 | 0.00000 |

(Values transcribed from the Ghia et al. 1982 tables. Primary-vortex-center
locations there are an additional cross-check: Re=100 ≈ (0.6172, 0.7344);
Re=400 ≈ (0.5547, 0.6055).)

## Rungs A/B/C — square cylinder in a channel (Breuer et al. 2000; Okajima 1982)

Reference: **Breuer, M., Bernsdorf, J., Zeiser, T., Durst, F. (2000).**
"Accurate computations of the laminar flow past a square cylinder based on two
different methods: lattice-Boltzmann and finite-volume." *Int. J. Heat and
Fluid Flow* 21, 186–196. Confined square cylinder, **blockage β = 1/8**,
Re (based on `U_max` and side `D`) from 0.5 to 300; also
**Okajima, A. (1982).** "Strouhal numbers of rectangular cylinders."
*J. Fluid Mech.* 123, 379–398 (unconfined St(Re) for the square section).

Pinned anchors (β = 1/8 channel; use as directional literature checks — the
same-mesh monolithic oracle is decisive since the fixture's default blockage is
higher):

- **Shedding onset `Re_crit`:** the steady→unsteady (Hopf) transition for the
  β = 1/8 confined square cylinder is at **`Re_crit ≈ 60`** (Re on `U_max`, `D`).
  **(to confirm — exact value depends on β and the onset criterion; commonly
  cited in the 55–65 band for confined β≈1/8.)** The steady rung Re must sit
  clearly BELOW this (use Re ≈ 30–40).
- **Steady Cd at Re ≈ 30–40:** the confined steady square-cylinder drag is of
  order **`Cd ≈ 1.7–2.0`** at Re ≈ 30–40 (β = 1/8), rising as Re falls.
  **(to confirm — exact tabulated value.)** The steady primary check is
  projection Cd == monolithic Cd on the same mesh (box-free).
- **Re = 100 (shedding):** mean **`Cd ≈ 1.4–1.5`** and Strouhal
  **`St ≈ 0.13–0.15`** for the β = 1/8 confined square cylinder (Breuer et al.
  2000). **(to confirm — exact digits.)** Okajima's unconfined square section
  gives `St ≈ 0.13` near Re = 100 as an independent anchor. For Re = 100 the
  driver reports mean Cd over ≥ 1 shedding period + St from the lift signal.
- **Domain / blockage:** Breuer uses β = D/H = 1/8 with long up/downstream
  extents. The fixture default is a compact unit-cube channel at higher β
  (β = 1/2 at half = 0.25); the driver may thin the box / lengthen the channel
  to approach β = 1/8 if a low-blockage literature comparison is wanted. The
  blockage as built is stated in the fixture section above.

## Rungs A′/C′ — cube in a channel (3-D)

Re = 40 (steady) primary; monolithic same-mesh is the oracle, literature
secondary. A confined-cube-in-channel steady-Cd reference (e.g. Saha 2004 /
Klotz et al. on the wall-mounted or channel-confined cube) provides an
order-of-magnitude literature anchor. **(to confirm — the confined-cube steady
Cd is sensitive to blockage and wall proximity; no single canonical
β = 1/8 unit-cube-channel table is pinned here. The R0 same-mesh oracle
(projection == monolithic, already cuDSS==splu validated) is the primary bar,
mirroring the sphere de-risk.)**

References (3-D, secondary): Saha, A.K. (2004), "Three-dimensional numerical
simulations of the transition of flow past a cube," *Phys. Fluids* 16, 1630;
Klotz, L. et al., laminar flow past a confined cube (for St / wake structure).

## Anti-vacuity guards (asserted in `tests/test_ladder_fixtures.py`)

- Aligned Box (offset=0) ⇒ `dmax == 0` and `geo.corr ≈ 1.0` (body-fitted).
- Offset Box ⇒ `0 < dmax < h` (genuine shift; rung C is load-bearing).
- Obstacle mask non-empty and its nodes lie on the Box faces (aligned).
- Fluid cell count == full uniform cells − carved Box cells.
- Cavity lid mask == the top boundary row (`2^level + 1` nodes), no obstacle.

## Sources

- [Ghia, Ghia & Shin (1982), J. Comput. Phys. 48, 387–411](https://doi.org/10.1016/0021-9991(82)90058-4)
- [Breuer, Bernsdorf, Zeiser & Durst (2000), Int. J. Heat Fluid Flow 21, 186–196](https://www.sciencedirect.com/science/article/abs/pii/S0142727X99000818)
- [Okajima (1982), J. Fluid Mech. 123, 379–398](https://doi.org/10.1017/S0022112082003115)
- [Saha (2004), Phys. Fluids 16, 1630](https://doi.org/10.1063/1.1688324)
