# Changelog — sextic Cahn–Hilliard free-energy term (C8 worked reference)

The changelog entry a real DiffSim contribution ships with. It names the term,
its parameter, the equations, and the tests that cover it, so a reviewer can see
the scope and a future contributor can find the guarantees.

## Added

- **Sixth-order Landau bulk free-energy term** in the transparent mixed
  Cahn–Hilliard reference (`sextic_ch.py`):

      f(c)  = 1/4 (c^2 - 1)^2 + (beta/6) c^6      [energy density, energy/volume]
      f'(c) = c^3 - c        + beta c^5           (chemical-potential residual)
      f''(c)= 3c^2 - 1        + 5 beta c^4          (mu-c Jacobian block)

  A higher-order stabiliser that steepens the double well and bounds deep
  quenches. `beta = 0` recovers the base model exactly.

- **Config field** `beta: Field(float, default=0.5, min=0.0)` — validated and
  recorded in `config.resolved.yaml`.

- **Analytic Jacobian contribution** `5 beta c^4` in the `J_mu_c` block only;
  every other block is unchanged from the base model.

## Verified (tests in `test_sextic_ch.py`)

- `test_sextic_term_derivatives` — f', f'' of the new term vs finite difference.
- `test_jacobian_matches_fd[0.0, 0.5]` — whole-system analytic tangent vs a
  residual finite difference (rel error ~1e-10, FD-limited).
- `test_limiting_case_beta_zero` — beta→0 reduces to the base double well.
- `test_mms_converges` — steady manufactured solution, observed order ~2.01.
- `test_mass_conserved` — no-flux march conserves INT c dV (drift ~1e-15).
- `test_energy_decreases` — the Ginzburg–Landau energy is non-increasing.

## Notes

- Units: `beta` dimensionless (energy-scaled); `M` [length^2/time]; `kappa`
  [energy*length^2/volume] sets the interface width ell ~ sqrt(kappa/W).
- The reference is intentionally transparent NumPy (like C0's `scalar_reaction`)
  so every stage is inspectable. Porting the term to the production Warp kernel
  (`src/diffsim/physics/cahn_hilliard.py: make_ch_newton`) is left as the closing
  exercise; the same test battery re-runs against a device implementation.
