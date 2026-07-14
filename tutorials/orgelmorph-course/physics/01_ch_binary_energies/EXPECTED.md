# P1 — expected results (self-check)

Running `python run.py` (defaults: 64×64, 250 steps, seed fixed in the
tutorial) should reproduce the following to a few significant figures.
Small differences from a different card or BLAS are normal; the
*qualitative* checks (energy decreasing, mass conserved, two phases)
must hold exactly.

| quantity | polynomial | Flory–Huggins ($A=1,\ B=2.5$) |
|---|---|---|
| total $F$: start → end | 0.256 → 0.0494 | −0.0619 → −0.101 |
| interfacial energy: peak → end | 0.0749 → 0.024 | 0.00741 → 0.00146 |
| phase values $[c_{\min}, c_{\max}]$ | [−1.04, 0.99] | [0.14, 0.86] |
| mass drift $\lvert\Delta m\rvert$ | 4.1×10⁻¹⁶ | < 10⁻¹⁴ (machine) |
| $F$ monotone decreasing after step 1 | yes | yes |

**What must be true regardless of hardware:**

- **Total free energy decreases** monotonically after the first step
  (the one-step start-up is the `mu_init="consistent"` transient,
  explained in the course document). $F$ is a Lyapunov functional — a
  rising total $F$ means a bug.
- **Mass is conserved** to solver tolerance (drift ≈ machine epsilon).
  Conserved dynamics cannot create or destroy material.
- **Two phases form.** Polynomial saturates near $c=\pm1$;
  Flory–Huggins stays strictly inside $(0,1)$ (the logarithm forbids
  overshoot) and separates into two composition-rich phases.
- **Interfacial energy is non-monotone**: it rises while interfaces
  form, then falls as domains coarsen and total interface area shrinks.

If your total energy rises, or mass drifts by more than ~10⁻¹⁰, or the
field does not separate, re-read the walkthrough — something in the
setup differs from the tutorial.
