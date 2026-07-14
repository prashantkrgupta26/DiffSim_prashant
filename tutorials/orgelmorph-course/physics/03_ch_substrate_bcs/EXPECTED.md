# P3 — expected results (self-check)

Running the harness at `--mode reference` (64×64, χ = 2.2, quench to
*t* = 0.4, fixed seed) reproduces the following. The wall free energy is
the bounded quadratic well *f*_w = *g* φ + *h* φ², minimized at the
preferred surface composition φ* = −*g*/(2*h*). All content and energy
numbers are **quadrature** integrals (`∫ φ dV`, `∫ f dV`, `∫ f_w dS`),
not nodal averages.

| case | (*g*, *h*) | φ* | substrate φ | enrichment Δφ |
|---|---|---|---|---|
| neutral (no-flux) | (0, 0) | — | ≈ 0.49 | ≈ 0.00 |
| attracting | (−1.5, 1.0) | 0.75 | ≈ 0.75 | ≈ +0.25 |
| repelling / depleting | (−0.5, 1.0) | 0.25 | ≈ 0.25 | ≈ −0.25 |
| opposing walls | sub (−1.5,1) / air (−0.5,1) | — | ≈ 0.75 | air φ ≈ 0.25 |
| confined lateral | (−1.5, 1.0) | 0.75 | ≈ 0.74 | ≈ +0.24 |
| demixing + wetting (χ=2.7) | (−1.5, 1.0) | 0.75 | ≈ 0.75 | ≈ +0.25 |

**Boundary-layer scaling** (stable sub-spinodal bulk): the enrichment
decay length δ ~ κ^s with **s ≈ 0.5** across four κ values.

**What must be true regardless of hardware:**

- **The quadrature mass ∫ φ dV is conserved to machine precision**
  (drift < 10⁻¹³) in *every* case — the wall energy is a natural
  condition on μ, not a mass flux, so it re-arranges material without a
  reservoir. (Measuring mass as a *nodal* mean against the nominal 0.5
  gives a spurious ~10⁻³ "drift" that is a boundary-weighting artifact,
  **not** a leak — use quadrature.)
- **The projection never fires** for these interior-φ* wells
  (`projected_dofs = 0`): φ stays strictly inside (0, 1), so nothing
  clips.
- **The substrate composition tracks φ\*** = −*g*/(2*h*): attracting
  enriches (φ ≈ 0.75), depleting lowers it (φ ≈ 0.25), neutral is flat.
- **The wall energy F_wall < 0 and falls as the wall wets** (most
  strongly in the demixing case); the total budget is
  F = F_bulk + F_grad + F_wall, each by quadrature.
- **δ ~ √κ.** Lower κ ⇒ thinner substrate boundary layer.

**Honest failure mode.** A preference that drives φ *outside* (0, 1) —
e.g. a linear wall *f*_w = *g* φ with no interior minimum, or an
initial condition that violates the bounds — makes the box projection
fire (`projected_dofs > 0`) and inject mass; the resulting drift is a
projection artifact, not a leak, and the FH log-barrier also collapses
the time step. Keep φ* strictly interior and conservation is
unconditional.

Third-significant-figure differences from a different card are normal;
the qualitative story (attract enriches, deplete lowers, neutral flat,
mass conserved by quadrature, δ ~ √κ) must hold.
