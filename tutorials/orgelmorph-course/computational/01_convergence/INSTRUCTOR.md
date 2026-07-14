# Instructor companion — C1 Convergence: basis order and time integration

*Teaching notes for the instructor. Student-facing material is in
`README.md` and the course document chapter `c1.tex`. Grading is in
`grading_rubric.md`; hints and selected solutions in `solutions/`.*

## What this chapter is really teaching

The computational track's central habit is measure-the-claim applied to
the discretization itself, not the physics. Four ideas do the teaching:

1. **A convergence order is meaningless unless exactly one error source
   is active.** Spatial, temporal, and algebraic errors are entangled by
   default; the chapter's whole design (steady MMS for spatial,
   fixed-mesh self-convergence for temporal, a Newton-tolerance sweep for
   algebraic) exists to isolate one at a time.
2. **A tight residual does not certify a discretization.** Newton
   converging to `1e-12` says nothing about whether the weak form,
   source, or BC are correct — only that the *algebraic* system was
   solved accurately. This has to be *shown* (the invariance sweep), not
   assumed.
3. **A self-convergence reference must itself be verified before it is
   "truth."** Halving the reference `dt` and checking the order is
   stable, plus a Richardson bound on the reference's own residual error,
   is what separates a real verification from a circular one.
4. **A wrong-but-plausible order is the dangerous failure mode, not a
   crash.** The four deliberate failures are chosen so that at least two
   of them (wrong BC, wrong source) produce a clean-looking run with a
   *finite* printed order — no NaN, no exception — that a rushed read
   would wave through.

## Common student misconceptions & typical incorrect conclusions

- **"I tightened Newton's tolerance to `1e-12`, so my discretization
  error is correct."** No — a tight residual only proves the *algebraic*
  system was solved accurately; it says nothing about whether the weak
  form, source term, or boundary condition are right. This chapter's
  algebraic-control check exists precisely to separate "solved
  accurately" from "solving the right problem": the discretization error
  must be shown *invariant* as the tolerance tightens (`EXPECTED.md` §2:
  the error moves by only ≈3e-13 of itself from `1e-4` to `1e-12`) — that
  invariance is the evidence, not the tight tolerance itself.
- **Not separating spatial, temporal, and algebraic error.** A student
  who runs a "spatial" sweep with a small `dt` and few Newton iterations
  is measuring some mixture of all three, and the resulting order is not
  diagnostic of anything. Push them to name, for their own study, which
  of the three errors is switched off and how (steady MMS kills the time
  term identically; self-convergence on a fixed mesh cancels the spatial
  term; the Newton sweep isolates algebraic).
- **A wrong-but-nonzero (or near-zero) observed order is MORE dangerous
  than an outright crash, because it looks like a pass.** The
  wrong-source failure (`EXPECTED.md` §4: measured order ≈0.00) still
  *runs to completion* and prints a number — the scheme converges, just
  to the wrong steady state, so the error does not shrink with `h` at
  all. A crash or `NaN` stops a pipeline immediately and demands
  attention; a finite, wrong order slips through any check that only
  asks "did it print a number." This is the single most transferable
  point in the chapter.
- **Reading the wrong-BC and wrong-source failures as "the mesh is too
  coarse."** Both have flat, non-shrinking error across all sampled
  levels (`EXPECTED.md` order ≈0.00 for both) — refining the mesh further
  will not fix either, because the bug is not resolution, it's a mismatch
  between the imposed BC/source and the manufactured field. Contrast with
  the under-resolved-feature failure (order 1.18), which *does* recover
  with refinement (see Explore Q4) because that one genuinely is a
  resolution problem.
- **Trusting a self-convergence reference without checking it.** A
  reference `dt` that is not "fine enough" makes the whole temporal study
  circular — the under-resolved-reference failure (`EXPECTED.md` §4:
  order ≈ −0.01) is exactly this: the sampled errors saturate at the
  reference's own error rather than reflecting the scheme's true order.
  The fix the chapter teaches is to verify the reference itself first
  (order-stability across two reference `dt`'s, Richardson bound).

## Expected runtime ranges (RTX 6000 Ada; scale by card)

| run | what it does | wall | notes |
|---|---|---|---|
| `python run.py` | spatial (8 levels total) + algebraic (3 Newton tols) + temporal (8 `dt` total) + reference verification + 4 deliberate failures | **~1-2 min** (this chapter's own README figure) | comparable to physics P1's quick mode, measured ≈75 s wall (mostly Warp kernel compile + Python start-up) |
| `python gen_figures.py` | same core, re-run to render `c1_spatial.png`, `c1_temporal.png`, `c1_diagnostics.png` + `numbers/c1.tex` | comparable to `run.py` | no additional CUDA work beyond re-running the studies |

First run of a session pays a one-time Warp kernel-compile cost, as in
every other chapter on this brick; subsequent runs in the same
environment are faster. Do not expect finer per-study timing breakdowns
than the figure above — this chapter's driver does not report per-phase
wall time.

## Common CUDA / solver errors students hit

- **Forcing a different linear solver onto the mixed `(c, mu)` saddle.**
  The saddle is indefinite, same as physics P1 — `splu` is used here for
  the same reason (see P1's `INSTRUCTOR.md` / Chapter C4 for the full
  story). If a student edits `convergence.py` to swap in a solver that
  assumes definiteness or does no pivoting, expect divergence or a wrong
  Newton increment — a good teachable moment, not a bug to silently work
  around.
- **No CUDA device available.** `build_dm` defaults to `device="cuda:0"`;
  a machine without a visible CUDA GPU will fail at mesh construction.
  Check `--device` on `run.py`/`gen_figures.py` before assuming the code
  is broken.
- **Newton failing to reach a tight `newton_tol` within `newton_max`
  iterations.** The algebraic-control sweep pushes the tolerance down to
  `1e-12`; on a modified problem (different `kappa`, coarser/finer mesh)
  Newton may stall before reaching it. Check `st.last_newton["dx_inf"]`
  — a stalled Newton inflates the reported `L2` error and can masquerade
  as a discretization defect if not caught.
- **Accidentally reintroducing a time-error contamination into the
  "spatial" study.** `_solve_steady` uses a very large `dt=1e3` and only
  `nsteps=3` specifically so the discrete steady state is reached without
  a spurious small-`dt` transient. A student who lowers `dt` while
  exploring will silently measure a mixture of spatial and temporal
  error and get a puzzling, non-`p+1` order — walk them back to the
  chapter's "three errors, not one" framing.

## Discussion prompts

- Why must the spatial manufactured field be *steady* (`c_t^\ast = 0`)
  rather than time-dependent? What would a naive transient MMS actually
  be measuring if you ran it with a moderate `dt`?
- The `p=1` and `p=2` spatial sweeps use different level ranges (`3–6`
  vs. `2–5`). Is comparing their raw errors at a shared level a fair
  comparison of the two discretizations, or does it need to be
  normalized by degrees of freedom (see Explore Q5)?
- The wrong-source failure "converges" (Newton finds a solution, the
  run doesn't crash) but to the wrong steady state, at order ≈0. What
  would this look like in a research code where nobody was watching for
  an order — how would you *notice* it at all?
- Self-convergence needs its reference verified before it is trusted.
  What is the general principle here that recurs whenever you compare
  against "the truth" rather than a closed-form exact solution?
