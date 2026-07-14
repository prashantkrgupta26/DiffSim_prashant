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
