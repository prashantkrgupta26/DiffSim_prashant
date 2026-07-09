# Track P — Performance — the profiling thread

Part of the [DiffSim curriculum](../README.md). Every chapter is a runnable,
self-contained script: *Learning outcome → Background → documented code →
Expected results (measured numbers you must reproduce) → Explore.*

| ch | script | outcome |
|---|---|---|
| P1 | `P1_cost_model_and_scaling.py` | per-stage timing vs level, measured exponents, the memory model |
| P2 | `P2_solver_showdown.py` | direct vs host-Krylov vs fused device Krylov; counting synchronizations |

Run a chapter from the repo root, e.g.:

```bash
python tutorials/P_performance/P1_cost_model_and_scaling.py
```

Related solver project page: [`docs/projects/performance.md`](../../docs/projects/performance.md).
