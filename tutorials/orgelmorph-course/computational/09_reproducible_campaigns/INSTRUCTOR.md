# C9 — Instructor notes

## The one idea
A campaign is defined by its *bookkeeping*, not its physics. The physics here
is deliberately trivial and already verified in P1; the chapter is entirely
about the workflow that turns runs into a result you can defend: immutable
config, deterministic ids, restart, and aggregation that cannot hide a failure.

## Where students go wrong
- **Silently dropping NaNs.** The most common real-world bug: `np.nanmean` over
  an ensemble quietly averages the survivors. Point at the tightened-gate demo
  — the deep-quench cell is *numerically finite* but *physically unreliable*
  (mass drift `3e-2`); a `nanmean` would bless it. Our gate marks it failed.
- **Confusing exploratory and confirmatory.** A 3-point scan with n=2 seeds is a
  hypothesis generator. Its CIs are wide and not a measurement. The confirmatory
  replicate exists precisely so the headline number is pre-registered, not
  cherry-picked from the scan.
- **Non-idempotent restart.** If run ids aren't deterministic, a restart
  re-runs everything or collides. Show `runs/<run_id>/status.json` and the
  `[skip]/[retry]` log.

## Talking points
- Why `campaign_complete` ignores *injected* faults but not *real* ones: the
  injected fault is an expected negative control (it proves surfacing works);
  a real failure means the science is incomplete.
- Why the bootstrap CI (`diagnostics.stochastic.bootstrap_ci`) over a normal
  ±1.96·sem for small n: it makes no Gaussian assumption and degrades
  gracefully at n=2.
- The SLURM array: `--only k` runs one cell; a dependent aggregate job re-reads
  every `status.json` including crashed tasks. Map this to the real
  `cluster/wodo_campaign/` kits.

## Grading
See `grading_rubric.md`. The deliverable is a student's own 2-parameter
manifest with a working restart and a surfaced injected fault.
