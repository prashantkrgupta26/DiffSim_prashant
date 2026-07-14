# P10 — instructor notes

## What this capstone is for

It is the integrative physics capstone: the student demonstrates the *whole*
scientific-computing workflow on ONE real material — load with provenance,
reproduce a published trend, converge it, propagate uncertainty, and deliver a
maintained, tolerance-checked reference case with an honest verdict. It pairs
with C9 (reproducible campaigns) and C10 (validation/uncertainty).

## The pedagogical core

The point is NOT to get a pretty "faster = finer" curve. The point is that a
*real* reference case, run honestly at the resolution actually affordable,
separates the **robust invariant** (demixing degree set by dryness and χ) from
the **scale-limited claim** (the wavelength ordering), and says so. Students who
report a clean monotone "faster = finer" trend without seed statistics have
missed the lesson — push them on the error bars.

## Cost tiers

- `quick` (~<2 min): 16×16, two-rung smoke. CI micro tier; not gated (level 4
  under-resolves the demixed surface layer so conservation is loose).
- `reference` (~a few–20 min): 32×32, the 4-rung ladder + one level-6 mesh rung
  + two time steps + two χ_pf values (9 films). The gated EXPECTED numbers.
- `research`: 64×64 ladder + a 128×128 convergence rung + a wider χ band.

The dominant cost is the number of Newton steps along each drying trajectory
(not mesh size at these small meshes), so the fastest and slowest rungs are both
more expensive than the middle — the drying-rate window is deliberately kept
moderate (Bi ∈ [1.5, 3.0]).

## Common issues

- **A rung exits `t_end` not `phis_stop`.** Its drying rate is too low for the
  horizon; raise `k_e_slow` or `t_end`. The harness reports the exit reason so
  this is never silent.
- **Polymer conservation looks "bad" (~1e-4).** Expected: large polymer N sharpens
  the surface layer. It is the fullerene balance that must hit machine precision.
- **Switching to P3HT_PCBM gives near-zero contrast.** That is the documented
  honest negative, not a bug — weak χ_pf, no demixing at tutorial scale.

## Grading

See `grading_rubric.md`. The single most important criterion is the honesty of
the robustness verdict.
