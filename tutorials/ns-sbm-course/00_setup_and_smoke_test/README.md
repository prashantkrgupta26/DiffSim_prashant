# 00 — Setup & smoke test

**Run this chapter first.** It confirms your machine can run the course and
teaches the workflow every later chapter uses (a YAML config, a provenance
`metadata.json`, a `results.json` checked against a tolerance baseline).

## 1. Install DiffSim

The course lives in the DiffSim repo. Install variants (extras defined in
the repo-root `pyproject.toml`):

| command | what you get | when |
|---|---|---|
| `pip install -e .` | core: `warp-lang`, `numpy`, `scipy`, `torch` | minimum to run the 2-D chapters (00–04) |
| `pip install -e ".[cudss]"` | core **+ cuDSS** GPU direct solver | the 3-D chapter (05) at scale; faster saddle solves |
| `pip install -e ".[dev]"` | core **+** `pytest`, `ruff` | to run the test suite |

**AMGX** (algebraic multigrid — the SPD-PPE scaling path in Chapter 06) has
no PyPI package: build [NVIDIA/AMGX](https://github.com/NVIDIA/AMGX), then
`pip install pyamgx --no-build-isolation`. It is optional — the doctor
reports it as "not built" and the current chapters fall back to `splu`.

## 2. Run the doctor

```bash
python doctor.py
```

It probes the whole stack (python, diffsim commit, Warp, torch+CUDA, GPU +
memory, FP64, cuDSS/nvmath, AMGX, and the **NS steppers**), then
**compiles+runs a tiny Warp kernel** and **assembles+solves one monolithic
Navier–Stokes step** (a tiny lid-driven cavity — the real
`LinearizedMonolithicStepper`), and ends in exactly one of:

- `READY: full CUDA` — you can run every chapter as written.
- `READY: reduced (no cuDSS)` — CUDA works but cuDSS is missing; the 2-D
  chapters run on `splu`, the 3-D chapter is slower.
- `NOT READY: <fix>` — do what the line says, then re-run.

## 3. Run the smoke test

```bash
python run.py --config configs/smoke_cpu.yaml --output outputs/smoke --device cpu
# with a GPU:
python run.py --config configs/smoke_cuda.yaml --output outputs/smoke --device cuda:0
```

This marches a tiny 8×8 lid-driven cavity through **both** NS engines
(monolithic saddle **and** pressure-projection split) through the standard
harness and checks the result against `baseline.yaml`. Compare with
[`EXPECTED.md`](EXPECTED.md); it must print `baseline check PASSED`.

| file | role |
|------|------|
| `doctor.py` | environment probe → `READY` / `NOT READY` verdict |
| `ns_smoke.py` | the core: `build_cavity`, `march_monolithic`, `march_projection` — read this first |
| `run.py` | the driver you run; harness + self-check table |
| `EXPECTED.md` | reference numbers your run should reproduce |
| `baseline.yaml` | the tolerance baseline the harness checks |

The smoke calls the **production steppers** directly — no toy solver.

## Learning objectives

By the end of this chapter you can:

- Confirm your device toolchain runs a real NS assemble + saddle solve, not
  just imports.
- Explain the standard `outputs/<run>/` layout and what each file records.
- Run *both* engines from one interface and read their `||div u||` and
  `|u|max` sentinels.
- State why the pointwise `||div u||` is finite/bounded (not zero) for the
  equal-order VMS scheme — the single most common misreading in the course.

## Prerequisites

- **Concepts:** finite elements + VMS stabilization in general; Python +
  NumPy. No CFD-of-this-codebase knowledge assumed.
- **Chapters:** none — this is the gate.

## Expected cost

- **Device:** any CPU runs the 2-D smoke in seconds; a CUDA GPU is needed
  for Chapter 05 and the scaling path.
- **Quick / reference:** the smoke is ~seconds of solve plus a one-time
  Warp kernel compile (~30–60 s the first time in a session).

## Required deliverable

Submit the eight-item report of `../ASSESSMENT.md`, specialised:

1. The `config.resolved.yaml` + `metadata.json` from your smoke run.
2. The `baseline.yaml` check green.
3. The printed self-check table (both engines' `|u|max` and `||div u||`).
4. **Headline:** the two engines' `||div u||` values, and a sentence on why
   they differ yet are both finite/bounded.
5. **Verification:** `all_finite == true` and `|u|max ≈ 1` (the lid speed).
6. **Failure:** force `--solver cudss` and report what happens on the
   indefinite saddle (or, if no cuDSS, force a diverging `dt` and show the
   `||div u||` blow-up the baseline catches).
7. **Exploration:** raise `nsteps` and watch `max|proj−mono|` shrink toward
   the Chapter 01–02 agreement.
8. **Research bridge:** one sentence on why the course keeps *two* engines.
