# Computational C0 — From weak form to CUDA

The foundation of the Computational track. One small **scalar nonlinear
reaction–diffusion** problem,

```
-∇²u + α u³ = f    on (0,1)²,   u = g on ∂Ω,
```

walked through the *entire* DiffSim pipeline — strong form → integration by
parts → element residual + Jacobian → quadrature → local-to-global → CSR
assembly → constraints → Newton → linear solve → device update — with **each
stage mapped to the exact source file and function** it uses in production.

**Read** the course document, Chapter *"From weak form to CUDA"* (Chapter
`ch:c0`), then open `scalar_reaction.py` — it implements the whole path in
transparent NumPy/SciPy, one function per stage, each annotated with its
`src/diffsim/...` counterpart.

## Hands-on (the required exercise): add a coefficient's analytic Jacobian

The starter `scalar_reaction_starter.py` has the reaction term in the
**residual** but not in the **Jacobian**. A finite-difference derivative check
fails against it and passes once you add the analytic term `3·α·u²`:

```bash
# RED — the starter's Jacobian omits the reaction term
RD_MODULE=scalar_reaction_starter pytest -q test_reaction_jacobian.py
#   -> FAILED: analytic Jacobian disagrees with finite difference (~4.9e-3)

# add   Ke = prob._Ke + reaction_jacobian_elem(prob, gp_value(prob, u))
# to jacobian()  (the completed reference is scalar_reaction.py)

# GREEN
pytest -q test_reaction_jacobian.py
#   -> 3 passed  (jacobian-vs-FD ~7.9e-9)
```

`test_reaction_jacobian.py` also cross-checks the transparent stiffness against
the **production Warp-kernel** assembly (`assemble_brick_csr` with
`PoissonBrick`) — they match to `0.0` — and verifies second-order MMS
convergence.

## Verify the whole pipeline (harness-driven)

```bash
python run.py --config configs/c0.yaml --device cpu --solver splu \
    --mode reference --output outputs/c0 --overwrite
python gen_figures.py --run-dir outputs/c0
```

`run.py` runs the manufactured-solution convergence study (p=1 and p=2), the
Newton-convergence trace, the analytic-Jacobian check, and the production
cross-check, prints a PASS/FAIL gate table, writes `results.json`, and checks
it against `baseline.yaml`. Compare with [`EXPECTED.md`](EXPECTED.md).

| file | role |
|------|------|
| `scalar_reaction.py` | the transparent pipeline core — read this first |
| `scalar_reaction_starter.py` | the hands-on "before" state (reaction Jacobian missing) |
| `test_reaction_jacobian.py` | the red→green gate + production cross-check + MMS |
| `run.py` | harness driver: MMS verification, prints the gate table |
| `gen_figures.py` | regenerates the pipeline diagram, the convergence figure, `numbers/c0.tex` |
| `baseline.yaml` | tolerance baseline the harness checks |
| `EXPECTED.md` | reference numbers your run should reproduce |

The tutorial reuses the production mesh, basis tables, Gauss points and linear
solver directly (`build_uniform`, `build_mesh`, `basis_tables`, `DeviceMesh`,
`gauss_points`, `solve_linear`); the transparent assembly is a readable stand-in
for the Warp kernels, proven equal to them by the cross-check test.
