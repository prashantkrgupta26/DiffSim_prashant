# Computational C4 — The solver ecosystem

Every implicit step ends in a linear solve of the **indefinite (c,μ)
Cahn–Hilliard saddle**, and there is no single best solver. This tutorial
teaches how to *choose* — scoring **correctness (residual) alongside
speed**, because a fast solver that returns a wrong answer is not fast.

- **The headline (Phase-0 finding):** a general-purpose GPU direct solver
  (cuDSS) does **not** pivot the indefinite saddle, so on the raw CH
  Jacobian it can return a **large-residual (wrong)** solution. The tutorial
  measures this. The correct recipe is **splu** (small, pivoted, exact) and
  **blockch / blockch_dev** (at scale, saddle-aware) — *not* cuDSS on the
  raw CH block. cuDSS is excellent on the well-conditioned sub-blocks and on
  block-masked 3-D film patterns.
- **Direct (splu)** — pivoted LU; exact, no tuning; cheap in 2-D.
- **Block preconditioner (blockch / blockch_dev)** — FGMRES around a
  CH-specific block factorization; **8.4×** cuDSS at slab64 and marches at
  811k dofs where masked cuDSS ceilings.
- **Matrix-free** — never stores the matrix; the survivor beyond int32/card
  memory.

Also: a **measured decision-support** function (`recommend_solver`) that
returns a *rationale*, not just a name, weighing dofs/nnz/block-structure/
precision/memory/reuse/tol/dt/hardware; **versioned benchmark provenance**;
cold+warm CUDA-synced median timings; and a **supported capture API**
(`capture_system=True`) instead of a `solve_linear` monkeypatch.

**Read** the course document, Computational Chapter *"The solver
ecosystem"* (start with `solvers.py`).

**Run:**
```bash
python run.py                 # correctness + 2-D benchmark + decision support
```
`run.py` prints the correctness table (cuDSS's large residual), the live
benchmark, and `recommend_solver`; compare with [`EXPECTED.md`](EXPECTED.md).

**Regenerate the document's figures + numbers** (optional):
```bash
python gen_figures.py         # writes ../../latex/figures/c4_*.png + numbers/c4.tex
```

| file | role |
|------|------|
| `solvers.py` | the core: `solver_correctness`, `benchmark_solvers_2d`, `recommend_solver`, `capture_ch_system`, `benchmark_provenance`, `CITED_3D` — read this first |
| `run.py` | the driver you run; prints correctness + benchmark + decision support |
| `gen_figures.py` | regenerates the correctness/crossover/cited-3D figures and `numbers/c4.tex` |
| `EXPECTED.md` | reference numbers your run should reproduce (residuals exact-ish, ms card-dependent) |

The **2-D numbers are live** (the real Cahn–Hilliard Jacobian, captured
from the production brick). The **3-D numbers are cited** from
`docs/dev/2026-07-13-m5-device-assembly.md` (D3 table) and
`docs/dev/2026-07-13-blockch-mpf.md` (B2–B5) — they take minutes per step
to reproduce, so we do not re-run them here.
