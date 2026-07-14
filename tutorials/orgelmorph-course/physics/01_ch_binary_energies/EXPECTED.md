# P1 — expected results (self-check)

The automated gate is **`baseline.yaml`** (tolerance-based scientific
invariants, not bit-identity), checked by the harness:

```
python run_harness.py --config configs/p1.yaml --mode reference \
    --output outputs/p1 --overwrite            # runs + checks baseline.yaml
python gen_figures.py --run-dir outputs/p1     # figures + numbers/p1.tex
```

Reference mode (64×64, 250 steps, fixed seed) reproduces the following to
a few significant figures. Different cards/BLAS differ in the last digits
— the tolerances in `baseline.yaml` absorb that; the *qualitative* checks
must hold.

| quantity | polynomial | Flory–Huggins ($A=1,\ B=2.5$) |
|---|---|---|
| total $F$: start → end | 0.256 → 0.0494 | −0.0619 → −0.101 |
| interfacial energy: peak → end | 0.0749 → 0.024 | 0.00741 → 0.00146 |
| interface width $\ell=\sqrt{\kappa/W}$ (cells) | 0.0316 (2.0) | 0.0224 (1.4) |
| $\lambda^\star$: predicted / probe-measured | 0.199 / 0.182 | 0.199 / 0.182 |
| coarsening $n$ (area length, pre-saturation) | 0.30 ± 0.02 | 0.20 ± 0.02 |
| phase values $[c_{\min}, c_{\max}]$ (converged) | [−1.04, 0.99] | [0.14, 0.86] |
| mass drift $\lvert\Delta m\rvert$ | 4.1×10⁻¹⁶ | < 10⁻¹⁴ (machine) |
| FH projected dofs (transient iterates) | 0 | 2308 |
| $F$ monotone decreasing after step 1 | yes | yes |

**What must be true regardless of hardware:**

- **Total free energy decreases** monotonically after the first step
  (the `mu_init="consistent"` start-up transient). The largest positive
  stepwise increment is ~0 to round-off — the scheme is *discretely*
  energy-stable, a stronger statement than "this run looked monotone".
- **Mass is conserved** to solver tolerance. For Flory–Huggins the box
  projection fires on *transient* Newton iterates (≈2308 dof-clips) but
  the *converged* per-step states stay strictly interior, so mass is
  still conserved (|Δm|≈0). On a coarser mesh the accepted state itself
  gets clipped and mass drifts (~2×10⁻⁴ in quick mode) — reported, not
  hidden.
- **The interface is $\ell\sim\sqrt{\kappa/W}$, not $\sqrt{\kappa}$** —
  ~2 cells here (FH ~1.4 cells is borderline: a deliberate resolution
  lesson).
- **Linear stability holds**: the small-Δt probe's measured growth
  spectrum peaks at the predicted fastest mode $k^\star=\sqrt{-f''/2\kappa}$
  ($\lambda^\star$ within one FFT shell of the measurement).
- **Coarsening**: the interfacial-area length $L=A/P$ grows as $t^{n}$
  with $n$ of order the Lifshitz–Slyozov 1/3 on the pre-saturation
  window; the peak (quantization-limited) and first-moment (tail-biased)
  definitions give different slopes — the length definition matters.

If your total energy rises, or mass drifts by more than ~10⁻¹⁰ where the
projection did not fire, or the field does not separate, re-read the
walkthrough — something differs from the tutorial.
