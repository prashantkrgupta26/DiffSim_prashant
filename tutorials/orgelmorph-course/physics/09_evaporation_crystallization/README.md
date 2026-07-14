# Physics P9 — Evaporation-conditioned embryo growth (the hero concept)

The full arc, at tutorial scale: a **wet** ternary film **dries**, the
concentrating blend **phase-separates**, and an **implanted** crystal
embryo **grows** into a **crystalline film** — how a real solution-cast
organic solar cell forms.

**Naming (Phase-1 correction):** the default runs **implant** a
supercritical embryo — they do **not** spontaneously nucleate — so this is
*evaporation-conditioned embryo growth*, not "nucleation". Genuine
FDT-noise nucleation (P8) is the advanced `make_stepper(noise_psi=...)`
mode.

The mechanism: the r14 crystal-contact penalty χ_ca gives a **solubility**
φ\* = 1 − |drive|/χ_ca in the local small-molecule fraction (**derived
from the free energy**, validated by an embryo composition sweep). Below
φ\* an embryo dissolves; above it, it grows. Drying raises the local φ_f,
so the same embryo **dissolves when wet** and **grows mid-drying**.

**Read** the course document, Chapter *"Evaporation-conditioned embryo
growth"* (start with `arc.py`).

**Run:**
```bash
python run.py                 # solubility + controls + arc + checks, 32x32
python run.py --level 6       # finer mesh (slower)
```

`run.py` derives φ\*, validates it by the composition sweep, runs the
sub/supercritical embryo control and the wet-vs-dry arc, writes
`outputs/results.json`, and checks it against `baseline.yaml`; compare with
[`EXPECTED.md`](EXPECTED.md). Figures + numbers via `python gen_figures.py`.

| file | role |
|------|------|
| `arc.py` | core: `run_arc`, `solubility_threshold`, `embryo_composition_sweep`, `run_static_embryo`, `TERMINATION` |
| `run.py` | driver; solubility + controls + arc + checks |
| `gen_figures.py` | regenerates `p9_*.png` + `numbers/p9.tex` |
| `baseline.yaml` | tolerance-based scientific-invariant checks |
| `EXPECTED.md` | reference numbers |

Uses `diffsim.physics.multiphase.MultiPhaseStepper` (M=2, K=1) in film mode
with r14 crystallization. A march that stops at the time horizon is
reported as `time_horizon`, **not** a "drying time" (see the `TERMINATION`
enum). Tutorial simplification (recorded in `arc.py`): constant Onsager
mobility with cuDSS instead of the production Vignes mobility of
`tests/test_multiphase_s3.py`.

## Learning objectives

By the end you can:

- Derive the solubility threshold `φ* = 1 − |drive|/χ_ca` from the
  homogeneous free-energy comparison at ψ=1 vs ψ=0, and explain why it is
  an *upper bound* (a supercritical embryo self-enriches its local
  composition, the P7 crystal-bulk channel) rather than an exact
  crossover.
- Validate the derived `φ*` against an independent embryo-composition
  sweep (no drying) and correctly interpret why the measured crossover
  sits below `φ*`.
- Explain why the default runs **implant** a supercritical embryo rather
  than nucleate spontaneously, and connect this naming choice to P8's
  genuine FDT-noise nucleation as the advanced alternative.
- Run and interpret the wet-vs-dry arc: the same embryo dissolves when
  implanted wet and grows when implanted mid-drying, driven by the
  drying-conditioned local composition crossing `φ*` — not by time or
  evaporation per se.
- Distinguish a sub-critical from a supercritical embryo at the same
  super-solubility composition (the Gibbs–Thomson radius floor) and read
  the honest `TERMINATION` status of a march rather than mislabeling a
  `time_horizon` stop as a "drying time."

## Prerequisites

- **Concepts:** solvent evaporation and drying kinetics (P5); the
  Cahn–Hilliard + Allen–Cahn coupling and χ-driven demixing (P7); the
  Allen–Cahn crystallization model and driving-force sign (P6); the
  Flory–Huggins / ternary χ framework (P4).
- **Chapters:** Chapter 00 (environment green). **P5** (evaporation) for
  the drying film mechanics this chapter implants an embryo into; **P7**
  (coupled CH+AC) for the crystal-bulk self-enrichment channel that makes
  `φ*` an upper bound rather than exact; **P6** for the crystallization
  model and Gibbs–Thomson critical radius; **P4** for the χ/ternary-blend
  language `φ*` is expressed in.

## Expected cost

- **Device:** any CUDA GPU, 2-D, well under 1 GB of device memory — an
  8 GB card is ample. **Solver:** the multiphase brick with cuDSS (a
  documented `splu` fallback if cuDSS is unavailable). **Tutorial
  simplification:** constant Onsager mobility with cuDSS, not the
  production Vignes composition-singular mobility — a labelled
  simplification, not the production path.
- **Quick mode:** comparable to **P1's measured ≈ 75 s wall** on an RTX
  6000 Ada (mostly Warp kernel compile + Python start-up); target < 2 min.
  First run of a session pays the one-time compile cost.
- **Reference mode** (level 5 = 32×32; solubility derivation + composition
  sweep + sub/supercritical control + wet-vs-dry arc): a few minutes;
  target 5–30 min.
- **Research mode** (finer mesh, `--level 6`): tens of minutes.

## Required deliverable

Submit the eight-item report of `../../ASSESSMENT.md`, specialised:

1. The `config.resolved.yaml` + `metadata.json` from a reference run.
2. The harness `baseline.yaml` check green (or a documented deviation).
3. The two figures (`p9_arc`, `p9_wet_vs_dry`) regenerated from your
   saved run.
4. **Headline:** the solubility `φ* = 1 − |drive|/χ_ca` **derived** from
   the free energy, **validated** by the embryo composition sweep
   (crossover bracketed below `φ*`, with the self-enrichment explanation
   for the gap), and demonstrated by the wet-dissolves/dry-grows arc,
   with the sub/supercritical embryo-radius control at fixed
   super-solubility composition.
5. **Verification:** the embryo composition sweep's grow/dissolve
   crossover bracket, reported alongside the derived `φ*` and the
   direction (below, not above) of the gap.
6. **Failure:** implant a sub-critical embryo (small `r0`) at a
   super-solubility composition and show it redissolves despite
   favorable composition — diagnose via the Gibbs–Thomson radius, not a
   composition error.
7. **Exploration:** answer one "Explore on your own" question with a
   plot (the implant-time sweep locating the wet/dry crossover in `φ_s`
   is recommended).
8. **Research bridge:** one paragraph on what the derived/validated `φ*`
   and the wet-dissolves/dry-grows arc predict for solvent/blend-ratio
   choices in a real solution-cast organic solar cell.
