# P10 — expected results (reference mode, RTX 6000 Ada)

Ground truth is captured on an RTX 6000 Ada (CUDA 12.9 / driver 13.2, cuDSS).
We assert **scientific invariants within documented tolerances**, never bitwise
GPU identity. The exact numbers below are what `gen_figures.py` writes into
`../../latex/numbers/p10.tex`; the tolerance gate lives in `baseline.yaml`.

## What must be true regardless of hardware

1. **All rungs actually dry.** Every ladder rung exits `phis_stop` (not `t_end`)
   with φ_s monotonically decreasing. A truncated run is never reported as dried.
2. **Fullerene conservation is machine precision.** The per-solute moving-frame
   balance for the fullerene (N=5) is `< 1e-9` relative — the hard, resolution-
   robust gate.
3. **The demixing DEGREE is set by dryness, not rate.** Phase contrast rises
   monotonically wet → dry on every rung, and its spread across the whole
   spin-speed ladder at matched dryness is small (a few %). This survives mesh +
   time refinement (contrast changes by only a few % under both) and is a
   monotone function of χ_pf across its order-of-magnitude band.
4. **Morphology collapses onto the Biot number.** The matched-state wavelength
   spread across the ladder is modest.

## What is NOT robust (documented honestly)

- The **"faster spin = finer wavelength"** ordering. At a single seed with only
  a few lateral domains it is weak, noisy, and moves under mesh refinement. The
  chapter states this as the capstone's central honest verdict.
- The **polymer** per-solute conservation. With a large polymer N=87 the polymer
  forms a sharp demixed surface layer that the 32×32 reference mesh only partly
  resolves, so the polymer balance is ~1e-4 relative (looser than the
  fullerene's machine precision) and tightens with mesh refinement. It is
  reported and tracked, not hidden — and gated loosely (`< 1e-2`).

## Reproduced trend, in one line

Higher spin speed → faster drying → a slightly finer *or comparable* frozen
morphology, but the reproducible physics is that **dryness**, not spin speed,
sets how strongly the blend demixes.
