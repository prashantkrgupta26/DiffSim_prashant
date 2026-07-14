# C9 — Reproducible campaigns

**Why it matters.** A single verified run is a data point, not a result. Real
morphology science is a *campaign*: the same frozen model swept over
parameters and seeds, aggregated with confidence intervals, and reported
without quietly discarding the runs that crashed. The difference between a
campaign and a pile of scripts is bookkeeping — immutable configs, deterministic
run ids, checkpoint/restart, and an aggregation step that *surfaces* failures
instead of averaging around them.

**Learning objectives.** After this chapter you can:
- freeze a campaign to an immutable resolved manifest (`campaign.resolved.yaml`)
  the way the Phase-0 harness freezes a single run to `config.resolved.yaml`;
- expand a parameter sweep × seed ensemble into a deterministic run grid;
- checkpoint and restart a campaign idempotently (skip completed, retry failed);
- aggregate a seed ensemble into mean + sd + bootstrap 95% CI using the shared
  `diffsim.diagnostics.stochastic`;
- enforce the **no-silent-drop** rule: catch a crashed run, mark a
  tolerance-violating run failed, and *surface both* in the summary;
- separate **exploratory** scans from **confirmatory** replicates;
- scale the identical manifest to a SLURM job array.

**Prerequisites.** Physics P1 (the binary Cahn–Hilliard spinodal brick that is
the campaign engine here), Computational C1 (verification — a campaign is only
meaningful over a *converged* model), the Phase-0 harness (`common/`), and
`diffsim.diagnostics.stochastic`.

**Expected cost.** Any CUDA GPU, 2-D, `<1 GB` device memory. The whole
9-run demo campaign (8 real runs + 1 injected fault) finishes in well under a
minute on an RTX 6000 Ada (mostly Warp kernel compile on the first run;
subsequent cells reuse the cache). A restart with everything completed returns
in a couple of seconds.

## The model, and why we sweep χ (not k_e)

The engine is the fast 2-D binary Cahn–Hilliard spinodal march from Physics P1
(`physics/01_ch_binary_energies/spinodal.py`), with the Flory–Huggins log free
energy

```
f(c) = A[c ln c + (1-c) ln(1-c)] + B·c(1-c)
```

The swept parameter is the **enthalpic Flory–Huggins parameter χ ≡ B** (the
code's `fh_B`), with the entropic scale `fh_A = 0.15` fixed so the whole χ band
is spinodally unstable. Two reasons to sweep χ rather than the evaporation rate
`k_e` the spec suggests: (1) χ is the one *production* material parameter in
`materials.yaml` that carries a real measured uncertainty (P3HT:PCBM
`chi_pf = 0.86`, order-of-magnitude `0.5`), which Chapter **C10** then
propagates — the two chapters share one physical control; (2) the CH spinodal
march is ~10× faster than the drying-film `k_e` driver, so the campaign runs
locally on a shared GPU. The `k_e` campaign is the natural cluster-scale
variant; `job_array.sbatch` shows how the same manifest maps to a SLURM array.

The morphology observable is `L_area`, the interfacial-area domain length
(box-fraction units) — the robust, monotone coarsening measure from
P1's diagnostics.

## Files

| file | role |
|------|------|
| `manifest.yaml` | the campaign's canonical, re-runnable description |
| `campaign.py` | the runner: resolve → enumerate → run → aggregate |
| `gen_figures.py` | runs the campaign, renders `c9_*.png`, writes `numbers/c9.tex` |
| `job_array.sbatch` | the SLURM job-array sketch (mirrors `cluster/wodo_campaign/`) |
| `baseline.yaml` | tolerance gate on the aggregate (invariants, not bit-identity) |

## Run it

```bash
cd computational/09_reproducible_campaigns
PYTHONPATH=<repo>/src python campaign.py \
    --manifest manifest.yaml --output outputs/demo --overwrite
# restart (idempotent): skips completed, retries the injected fault
PYTHONPATH=<repo>/src python campaign.py \
    --manifest manifest.yaml --output outputs/demo
```

Each cell writes `outputs/demo/runs/<run_id>/status.json`; the campaign writes
`campaign.resolved.yaml`, `metadata.json`, and `summary.json`. Regenerate the
figures + numbers with `gen_figures.py`.

## The no-silent-drop rule (the point of the chapter)

Two failure classes are demonstrated, both **surfaced** in `summary.json`:

1. **A crashed run** — the manifest injects a cell with a mistyped solver
   (`solver: nope`). It raises `ConfigError`; the runner catches it, records the
   cell `failed`, and the campaign *continues*.
2. **A tolerance-violating run** — under a tightened mass-drift gate the deepest
   quench (χ = 1.36) exceeds it. It "completes" numerically but is marked
   `failed` (validation), flipping `campaign_complete → false` and the process
   exit code to `3` — a CI gate that refuses to bless an incomplete campaign.

An aggregation that dropped these would report a clean mean over the survivors
and hide the fact that a corner of parameter space is unreliable. Ours reports
the mean *and* the failures, with how many seeds actually contributed at each
point.

See `EXPECTED.md` for the measured self-check numbers, `INSTRUCTOR.md` for
teaching notes, and `grading_rubric.md` for the deliverable rubric.
