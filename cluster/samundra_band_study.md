# Band-Study Runbook (for Samundra) — P2-P1 Neumann SBM Sphere, Levels 4–8

*Companion to `docs/p2_band_study_for_students.md` (the guided walkthrough)
and `docs/p2_band_problem_statement.md` (the self-contained research
brief — read §4–5 first). This runbook is the operational side: exactly
what to run on Nova, what it costs, and what the numbers decide.*

## IMPORTANT UPDATE (2026-07-06 night): the dip is RESOLVED

The L4-L6 non-monotonicity was a benchmark DEGENERACY: r=0.25 aligns the
sphere's tangent planes with mesh faces at every level (findings 4b(h)).
The corrected canonical radius is r=0.27 (now the driver default), whose
ladder is MONOTONE: 1.95e-3 / 5.68e-4 / 1.91e-4 (orders 1.78, 1.57).
Your L7 run is now the ASYMPTOTIC CONFIRMATION (expect order -> 2), not
the dip verdict. The verdict table below is updated accordingly.

## The question you are answering (superseded framing kept for context)

The 3-D band errors are **non-monotone**: L4 1.99e-2 → L5 5.70e-4 →
L6 1.38e-3. Our hypothesis: an error-cancellation **dip at L5** (the
dominant error component crosses zero during the preasymptotic-to-
asymptotic transition), so L6 sits on the true curve and **L6→L7 should
show order ≈ 2**. L7 has been blocked by the 48 GB cuDSS ceiling — the
A100/H200 remove that. Your run decides the hypothesis.

## How to run

```bash
cd DiffSim && bash cluster/bootstrap.sh
# individual levels (each appends to band_study_results.txt):
.venv-nova/bin/python benchmarks/band_study.py 3 4 5     # ~minutes
.venv-nova/bin/python benchmarks/band_study.py 3 6       # ~20-30 min
.venv-nova/bin/python benchmarks/band_study.py 3 7       # hours; needs
                                                          # >= 80 GB card
.venv-nova/bin/python benchmarks/band_study.py 3 8       # H200 ONLY,
                                                          # stretch: may
                                                          # ALLOC-fail —
                                                          # that is a
                                                          # RESULT, record
                                                          # the message
# or as part of the arrival campaign (band-study stage):
.venv-nova/bin/python cluster/campaign.py --card h200 --hours 48
```

Also rerun **2-D L5–L8** once (fast, ~minutes) to confirm the locked
table on the new machine: errors 6.62e-4 / 1.75e-4 / 4.53e-5 / 1.09e-5.

## What to look at

Each line reports `band(3) Lk: err=... dofs=... band_elems=...`.
Compute consecutive orders: order = log2(err_k / err_{k+1}).

| Outcome at L6→L7 | Verdict |
|---|---|
| order ≈ 2 (1.7–2.3) | **Asymptotic second order confirmed** in 3-D at the corrected radius. Write it up. |
| order ≈ 1 | The 3-D band has a REAL problem the 2-D study missed — check band thickness vs the adjacency rules (problem statement §2, §4.1) at 3-D; escalate. |
| non-monotone again | Sample a **radius sweep** at fixed L6 (r ∈ 0.15…0.35) — the cheap dip-probe (problem statement Q5); the error-vs-resolution curve shape will say. |

Also record: wall time per stage (the constraints build should be
minutes — if it is hours, something regressed), peak GPU memory
(`nvidia-smi --query-gpu=memory.used -l 60` in a side terminal), and the
solver used per level (printed; cuDSS expected).

## Known sharp edges

- L7/L8 memory: the *solve* is the wall, not the build. If cuDSS
  ALLOC-fails at L7 even on the H200, record the exact message + DOF
  count — that number recalibrates the capacity table.
- Host RAM: L8 3-D assembly wants ~100+ GB host — Nova nodes have 512 GB,
  fine, but don't run two levels concurrently.
- The λ=1 keep-all, node-band(3), exterior-sphere configuration is
  LOCKED — don't vary it in this run; variations are the follow-up
  study (problem statement Q1/Q4).
