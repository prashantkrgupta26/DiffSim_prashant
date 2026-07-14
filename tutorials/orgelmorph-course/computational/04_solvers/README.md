# Computational C4 — The solver ecosystem

Every implicit step ends in a linear solve of the **indefinite (c,μ)
Cahn–Hilliard saddle**, and the solver choice is a **correctness** decision
before a speed one. This tutorial teaches it with a *marched* test, not a
one-shot residual.

- **The headline (measured):** **cuDSS DIVERGES on the polynomial CH
  saddle.** Marching the real 20-step spinodal, `poly + cuDSS` blows the
  field up to ~[−552, 542] while `poly + splu` stays [−1.03, 1.01]. cuDSS
  does no partial pivoting, and the poly well is **indefinite** (f″=3c²−1 < 0)
  in the spinodal band. cuDSS survives Flory–Huggins (f″ ≥ 4A > 0), but that
  is energy-specific.
- **The trap:** a residual captured at *one* Newton iterate is deceptively
  tiny for cuDSS (~5e-13) even though the march diverges — a small residual
  ≠ a small error. Measure the **marched solution** (c.min/max).
- **The recipe:** **splu** (pivoted, safe on the saddle) at small scale;
  **blockch / blockch_dev** (splits the saddle into SPD sub-solves) at
  scale; **not cuDSS on the raw CH block**. The `--solver auto` default of
  splu for CH is correct.
- **At scale, memory too:** cuDSS ceilings at ~811k dofs on 48 GB;
  blockch_dev is 8.4× cheaper at slab64 and marches on; matrix-free reaches
  6.4M dofs on one card.

Also: a **measured decision-support** function (`recommend_solver`) that
returns a *rationale* (never picks raw-CH cuDSS); **versioned provenance**;
cold+warm CUDA-synced median timings; and a **supported capture API**
(`capture_system=True`) instead of a `solve_linear` monkeypatch.

**Read** the course document, Computational Chapter *"The solver
ecosystem"* (start with `solvers.py`).

**Run:**
```bash
python run.py                 # march divergence + 2-D timing + decision support
```
`run.py` prints the divergence table (cuDSS blows up on poly), the
residual trap, the timing, and `recommend_solver`; compare with
[`EXPECTED.md`](EXPECTED.md).

**Regenerate the document's figures + numbers** (optional):
```bash
python gen_figures.py         # writes ../../latex/figures/c4_*.png + numbers/c4.tex
```

| file | role |
|------|------|
| `solvers.py` | the core: `march_divergence`, `solver_correctness` (the trap), `benchmark_solvers_2d`, `recommend_solver`, `benchmark_provenance`, `CITED_3D` — read this first |
| `run.py` | the driver you run; prints divergence + trap + timing + decision support |
| `gen_figures.py` | regenerates the divergence/crossover/cited-3D figures and `numbers/c4.tex` |
| `EXPECTED.md` | reference numbers your run should reproduce (divergence exact, ms card-dependent) |

The **2-D numbers are live** (the real Cahn–Hilliard Jacobian, captured
from the production brick). The **3-D numbers are cited** from
`docs/dev/2026-07-13-m5-device-assembly.md` (D3 table) and
`docs/dev/2026-07-13-blockch-mpf.md` (B2–B5) — they take minutes per step
to reproduce, so we do not re-run them here.

## Learning objectives

By the end you can:

- Explain why the mixed `(c, μ)` Cahn–Hilliard Jacobian is a saddle matrix
  that becomes genuinely **indefinite** once the bulk curvature
  `f″=3c²−1` goes negative in the spinodal band `|c|<1/√3`, and why an
  unpivoted GPU direct solver cannot be trusted there.
- Reproduce the headline divergence: marching the real 20-step polynomial
  spinodal (level 5, `dt=0.02`) with cuDSS blows the field from the
  physical band out to **[−552, 542]** (a ~500× blow-up), while `splu`
  stays bounded at **[−1.03, 1.01]** — and state that Flory–Huggins
  survives cuDSS (`[0.001, 0.999]`) only because its entropic curvature
  stays positive, which is energy-specific, not a general green light.
- Recognize the **residual trap**: a single captured Newton iterate's
  residual is deceptively tiny for cuDSS (~5e-13, next to `splu`'s ~1e-12
  and `blockch`'s ~4e-10) even though the 20-step march with cuDSS
  diverges — and explain, in terms of conditioning, why a small residual
  does not certify a small solution error.
- Describe how `blockch` / `blockch_dev` avoids ever forming the
  indefinite monolithic factorization by splitting the saddle into SPD
  sub-solves (mass, `W1`, `W2` blocks) inside an FGMRES iteration, and why
  that is the "at-scale" recipe while `splu` is the "small-scale" one.
- Read `recommend_solver`'s rationale-producing decision support and
  reproduce its regime table (2-D small `<2e4` dofs → `splu`; 2-D large
  `<5e5` → `splu`/`blockch`; 3-D small `<5e5` → masked cuDSS/`blockch_dev`;
  3-D large up to `~1.5e6` → `blockch_dev`; extreme → matrix-free), and
  explain why it never recommends cuDSS on the raw CH block.
- Distinguish the **live** 2-D numbers (measured on the real captured CH
  Jacobian, e.g. cuDSS 2–6× faster than `splu` at 33k dofs) from the
  **cited** 3-D numbers (dev-note tables, e.g. the ~811k-dof/48 GB cuDSS
  ceiling and `blockch_dev`'s 8.4× speedup at slab64) that this chapter
  reports but does not re-run.

## Prerequisites

- **Concepts:** the mixed `(c, μ)` Cahn–Hilliard formulation (Physics P1)
  — this chapter reuses the same production Jacobian and asks a *solver*
  question about it, not a physics one; basic linear-algebra notions of
  definiteness/indefiniteness, pivoting, and residual vs. error.
- **Chapters:** Chapter 00 (`00_setup_and_smoke_test`, environment green);
  Computational C1 (this track); Physics P1 (Binary Cahn–Hilliard), for
  the `(c, μ)` model this chapter solves.

## Expected cost

- **Device:** any CUDA GPU. The live parts of this chapter are 2-D (the
  largest live benchmark is ~33k dofs, per `EXPECTED.md` item 3) — any
  8 GB+ card is ample. The 3-D numbers are cited from the dev notes
  (`docs/dev/2026-07-13-m5-device-assembly.md`,
  `docs/dev/2026-07-13-blockch-mpf.md`), not re-run here, because they
  take minutes per step.
- **Solver:** this chapter compares `splu` / cuDSS / `blockch` /
  matrix-free head-on. Headline: cuDSS **diverges** on the raw polynomial
  CH saddle; `splu` is the safe small-scale choice, `blockch`/
  `blockch_dev` the safe at-scale choice.
- **Quick/live run:** comparable to Physics P1's quick mode (**≈75 s**
  wall measured on an RTX 6000 Ada, mostly Warp kernel compile + Python
  start-up) — `python run.py` here marches the 20-step spinodal for both
  energies/solvers, times the 2-D benchmark, and prints the decision
  support; expect a similar order of magnitude. No separate per-chapter
  second count is measured in this chapter's own files.
- First run of a session pays the same one-time Warp kernel-compile cost
  noted in P1.
- Any 3-D-scale timing you quote (the `CITED_3D` table: the 811k-dof
  cuDSS ceiling, `blockch_dev`'s 8.4× slab64 speedup, masked cuDSS's
  17.7→3.70 s/call, matrix-free's 6.39M dofs) is **cited** from the dev
  notes, never re-run in this chapter — do not fabricate a live number
  for it.

## Required deliverable

Submit the eight-item report of `../../ASSESSMENT.md`, specialised as
follows, with the headline explicitly called out:

1. The `config.resolved.yaml` + `metadata.json` from a reference run of
   `python run.py`.
2. The harness's own PASS/FAIL self-check block green (every line
   `[PASS]`, `ALL CHECKS: PASS`) — or a documented deviation.
3. The three figures (`c4_divergence`, `c4_crossover`, `c4_cited3d`)
   regenerated via `gen_figures.py`, plus `numbers/c4.tex`.
4. **Headline:** the divergence table with exact numbers — polynomial +
   cuDSS marches the field to **[−552, 542]** (~500× blow-up) while
   polynomial + `splu` stays at **[−1.03, 1.01]**; Flory–Huggins + cuDSS
   stays `[0.001, 0.999]` (energy-specific, not a green light) while
   Flory–Huggins + `splu` stays `[0.06, 0.94]`. State the mechanism:
   `f″=3c²−1<0` in the spinodal band makes the saddle indefinite, and
   cuDSS's unpivoted factorization is wrong there.
5. **Verification / the residual trap:** report the one-iterate residuals
   (`splu` ~1e-12, cuDSS ~5e-13, `blockch` ~4e-10) alongside the marched
   divergence, and state explicitly that the tiny cuDSS residual does
   **not** predict the diverging march — a small residual is not a small
   error on an indefinite, unpivoted system.
6. **Failure diagnosis:** force `--solver cudss` on the polynomial saddle
   (or read `march_divergence`'s `diverged=True` rows) and diagnose it
   from the diagnostics: which number (`c.min`/`c.max` escaping the
   physical band) shows the failure, and why (no partial pivoting on an
   indefinite block).
7. **Exploration:** answer one "Explore on your own" question from
   `c4.tex` with evidence (e.g. Q1: `max|c|` vs step count for cuDSS).
8. **Research bridge / decision support:** report `recommend_solver`'s
   rationale for at least the small-2-D, large-2-D, and one 3-D regime,
   and state why it never recommends cuDSS on the raw CH block.
