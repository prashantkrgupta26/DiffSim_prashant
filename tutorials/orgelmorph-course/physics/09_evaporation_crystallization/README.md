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
