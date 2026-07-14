# Instructor companion — P5 Evaporation (the drying film)

*Teaching notes for the instructor. Student-facing material is in
`README.md` and the course document chapter `p5.tex`. Grading is in
`grading_rubric.md`; hints and selected solutions in `solutions/`.*

## What this chapter is really teaching

The moving-boundary mechanics are a vehicle for one methodological habit
this course keeps returning to: **a fair comparison holds the right thing
fixed**. Three ideas do the teaching:

1. **Matched state, not matched time.** A sweep over evaporation rate that
   reads morphology at a fixed time horizon is comparing apples dried to
   different degrees. The fix — interpolate every run's own `φ_s(t)`
   trajectory to a common dryness — is the chapter's single most
   transferable idea, and it recurs anywhere a "rate" and a "clock" can be
   confused (annealing schedules, quench rates, anywhere a process
   variable also sets how far along a system is).
2. **A dimensionless group is a claim to be tested, not assumed.**
   `Bi = k_e·h0/D_s` is proposed as *the* controlling group; the chapter
   does not just cite it, it sweeps `(k_e, D_s)` independently and checks
   that runs sharing a `Bi` land on the same matched-dryness wavelength.
   That the pair doesn't matter, only the ratio, is the thing to verify,
   not assert.
3. **Conservation is a resolution claim, not a free theorem.** Only the
   solvent leaves the frame, so each solute's content `h(t)·∫φᵢ` should be
   exactly constant — and it is, to machine precision, but *only* once the
   demixed interfaces and the surface-enrichment layer are resolved
   (level 5–6). At level 4 the same conservation law leaks ~10%. This
   reuses P1's lesson (a continuous invariant is only inherited by the
   discretization you can afford) in a new, moving-boundary setting.

## Common student misconceptions & typical incorrect conclusions

- **"The fast film's finer morphology proves faster evaporation makes
  finer domains."** This is the cardinal error the chapter is built to
  correct. A fast run compared at a fixed time horizon has simply dried
  further than a slow run at the same time — the apparent "rate effect"
  is a dryness difference in disguise. Push students to check: was this
  comparison made at matched time or matched `φ_s`? If the former, the
  conclusion is unsupported.
- **"Once I've matched the dryness, evaporation rate has no effect at
  all."** Not quite — the chapter's own reading is that rate is a *weak,
  secondary* knob: at matched dryness the faster (higher-Bi) film is if
  anything *slightly coarser* (less time to develop lateral structure),
  the opposite of the naive story, not a null result.
- **"Bi collapses everything, so I don't need to check the regime."**
  The collapse is demonstrated over the sweep's actual range,
  `Bi ∈ [0.75, 6.0]`, straddling the `Bi = 1` crossover — it is not shown
  to hold arbitrarily far into either the deeply drying-limited or
  deeply diffusion-limited limit. A student who extrapolates the
  collapse outside the tested range is asserting, not verifying.
- **"The wavelength-vs-Bi law is sharp, like a power law I can fit."**
  With only ~2 lateral domains across the box the wavelength is
  statistically noisy (single seed). The chapter explicitly says: trust
  the *collapse* (agreement across different `(k_e, D_s)` at the same
  Bi), not a sharp size-versus-Bi functional form.
- **"A wavelength of 0.6 cells means the domain is smaller than a
  cell."** Reported wavelengths must be a real length: a fraction of the
  box width, and in cells `>1` (`λ/Δx = n_x/k̄`). A bare "cells" number
  below one is a units bug (the old code's mistake, per the chapter),
  not a physically resolved small domain.
- **"Solute conservation is automatic because the model conserves
  it."** True in the continuous PDE, false on an under-resolved mesh:
  level 4 leaks ~10% of a solute. Conservation here is inherited from
  resolving the demixed interfaces and the surface-enrichment layer, not
  from the model alone.

## Expected runtime ranges (RTX 6000 Ada; scale by card)

| mode | mesh / runs | wall | notes |
|---|---|---|---|
| quick | level 5 (32×32) / 3 runs | target < 2 min | comparable to P1's measured ≈75 s for a similarly-sized 2-D mesh |
| reference | level 6 (64×64) / 7-run `(k_e, D_s)` sweep | target 5–30 min | the `EXPECTED.md` numbers |
| research | level 7 (128×128) / finer regime grid (4 `k_e` rows, 3×3 regime grid) | longer than reference | wider Bi coverage |

First run of a session pays a one-time Warp kernel-compile cost;
subsequent runs in the same environment are faster.

## Common CUDA / solver errors students hit

- **Forcing `--solver splu` on a research-mode run and it crawling.**
  `splu` is the documented small-run fallback; cuDSS is the intended path
  once the mesh/sweep grows. A student who always forces `splu` on a
  research-size sweep will see wall time balloon — check the resolved
  config's `solver` field.
- **`out of memory` at level 7+.** 2-D P5 meshes are modest, so OOM at
  research level on a small card usually means an unintentionally large
  `level` or an over-wide sweep grid, not a genuine memory requirement —
  check `level` and the sweep lists in the resolved config.
- **Conservation "blowing up" and the student blaming the solver.** If
  per-solute drift is ~10⁻² instead of ~10⁻¹⁵, the first suspect is
  `level` (did they run at level 4?), not the linear solver or step size.
- **A run reporting `t_end` and being called "dried".** Only
  `phis_stop` is an honest "dried to the cutoff" exit. A `t_end`-truncated
  run reached the time horizon without drying past `phis_stop` — check
  `exit_reason` in `results.json` before trusting any dryness-dependent
  number from that run.
- **`h_min` or `dt_underflow` exits.** These indicate the film ran out of
  physical height or the adaptive step collapsed before reaching the
  cutoff — a real failure mode, not noise; the harness records the exit
  reason precisely so this is caught, not silently swallowed.

## Discussion prompts

- If a colleague hands you a plot of "morphology vs evaporation rate" at
  a fixed drying *time*, what is the first question you ask before
  trusting it? What single change would you require before believing any
  conclusion drawn from it?
- The Biot number mixes a rate (`k_e`) and a mobility (`D_s`) into one
  ratio. What does it mean, physically, for two very different real
  systems (a fast, volatile solvent with a mobile blend vs a slow solvent
  with an immobile blend) to share the same `Bi` and dry to the "same"
  morphology?
- Conservation of solute content is exact in the continuous model but
  resolution-limited in the discretization (leaks ~10% at level 4, machine
  precision at level 5–6). What general lesson does this carry for
  trusting *any* conservation law reported from a simulation you haven't
  personally resolution-checked?
