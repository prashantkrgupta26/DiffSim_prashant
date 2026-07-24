# 01 — The monolithic VMS engine (lid-driven cavity)

The oracle. One coupled $(u,p)$ saddle solve per step, equal-order
$P_1/P_1$ stabilized by residual-based VMS (SUPG + PSPG + grad-div). This
chapter marches the classic **lid-driven cavity** and compares the $x=0.5$
centerline $u(y)$ against Ghia, Ghia & Shin (1982), Table I.

**Read** the course document, Chapter 1 (*The monolithic VMS engine*).
Start with `cavity.py` — the importable core it walks through.

**Run:**
```bash
python run.py                                       # level 4, Re=100, both engines
python run.py --config configs/ldc.yaml --mode reference --output outputs/ldc
python run.py --config configs/ldc.yaml --mode research  # level 5, tighter to Ghia
```

`run.py` prints the centerline self-check table; compare with
[`EXPECTED.md`](EXPECTED.md). It also runs the projection leg so the
same-mesh **faithfulness** comparison (the course's central check) is
visible from step one.

| file | role |
|------|------|
| `cavity.py` | the core: `build_cavity`, `march_monolithic`, `march_projection`, `run_cavity` — read this first (shared with Chapter 02) |
| `run.py` | the driver you run; harness + centerline self-check table |
| `gen_figures.py` | regenerates the centerline figure + `numbers/c1.tex` |
| `EXPECTED.md` | reference numbers your run should reproduce |
| `baseline.yaml` | the tolerance baseline the harness checks |

The chapter uses the production brick
`src/diffsim/steppers/linearized.py` (`LinearizedMonolithicStepper`)
directly — no toy solver.

## Learning objectives

By the end of this chapter you can:

- Derive the equal-order $P_1/P_1$ VMS weak form: SUPG on the momentum
  test, **PSPG** on the continuity test (the term that makes equal-order
  work), grad-div/LSIC, and the $\tau_M,\tau_C$ metric form.
- Read the saddle block structure $\begin{bmatrix}F&G\\D&C\end{bmatrix}$
  off the element kernel `lin_ns_Ae`, and say what the nonzero PSPG
  C-block does.
- March the monolithic engine to steady and compare against a benchmark
  (Ghia), separating coarse-mesh discretization error from the
  same-mesh split-vs-oracle difference.
- Explain why the pointwise $\|\nabla\!\cdot u\|$ is finite/bounded, not
  zero, and why it is the wrong steady-state gate.

## Prerequisites

- **Concepts:** finite elements, VMS/SUPG–PSPG stabilization in general,
  the inf–sup (LBB) condition. Python + NumPy.
- **Chapters:** Chapter 00 (environment green).

## Expected cost

- **Device:** any CPU runs level 4 in a minute or two (`splu`); a GPU is
  optional here. The indefinite saddle uses `splu` (host direct) — cuDSS is
  **not** used at this size.
- **Quick** (`--mode quick`, 100 steps): under a minute of solve.
- **Reference** (level 4, 200 steps, the EXPECTED numbers): a couple of
  minutes.
- **Research** (level 5, 400 steps): several minutes; tighter to Ghia.

## Required deliverable

Submit the eight-item report of `../ASSESSMENT.md`, specialised:

1. The `config.resolved.yaml` + `metadata.json` from a reference run.
2. The `baseline.yaml` check green.
3. The centerline figure (`gen_figures.py`) — proj / mono / Ghia overlay.
4. **Headline:** `max|mono − Ghia|` and `max|proj − mono|`, each with the
   station where the max occurs.
5. **Verification:** the faithfulness check — the projection tracks the
   same-mesh monolithic (`d_proj_mono`) — stated against the coarse-mesh
   `d_mono_ghia`.
6. **Failure:** force `--solver cudss` on the saddle and diagnose the
   symptom (indefinite operator, no pivoting).
7. **Exploration:** refine to level 5 and show `d_mono_ghia` shrink toward
   Ghia.
8. **Research bridge:** one paragraph on why the oracle is the *reference*
   but not the *scalable* engine (setting up Chapter 02).
