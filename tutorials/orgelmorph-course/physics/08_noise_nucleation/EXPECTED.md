# P8 — expected results (self-check)

Running `python run.py` (defaults: 64×64, p1 bulk, undercooled melt
T = 0.5·Tm, ψ starts at 0 with NO seed, fixed noise seed, BDF1) sweeps
the FDT noise amplitude and reports the final crystalline fraction, grain
count, and induction time.

Measured sweep (32×32, level 5, fixed noise seed):

| noise_psi | 0 | 0.03 | 0.06 | 0.12 |
|---|---|---|---|---|
| final crystalline fraction X | 0.000 | (nucleates) | (more) | 0.516 |

Zero noise → X = 0.000 (amorphous); the onset (first amplitude with
X > 0.02) is at noise_psi = 0.03; the largest amplitude reaches
X = 0.516 in 25 distinct grains.

**What must be true regardless of hardware:**

- **Zero noise stays amorphous.** With `noise_psi = 0` the melt does not
  nucleate (X ≈ 0): ψ = 0 is metastable behind the nucleation barrier,
  and without fluctuations nothing crosses it. A nonzero X at zero noise
  means a bug (a spurious seed or an unstable, not metastable, state).
- **Larger noise nucleates more.** The final crystalline fraction and the
  number of grains increase with the noise amplitude — the
  fluctuation–dissipation amplitude *is* kᵦT, and the nucleation rate
  scales like exp(−ΔF\*/kᵦT), so a modest amplitude increase produces a
  large increase in nucleation by a fixed time (a threshold-like onset).
- **The noise is the physical seed.** Unlike Chapter 1's numerical RNG
  seed, here changing `noise_seed` changes *which* grains form where, but
  not the statistics (grain count, X) — a different seed is a different,
  equally valid sample of the same fluctuating system.
- **BDF1 only.** The brick asserts noise off under BDF2; this concept
  runs BDF1.

If your sweep shows nucleation at zero noise, or no nucleation at the
largest amplitude, re-check the undercooling and the amplitudes — the
run should span "amorphous" to "several grains".
