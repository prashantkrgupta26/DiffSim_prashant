# The DiffSim curriculum

A progressive course in GPU-native, differentiable finite elements — from a
2-D manufactured solution to Navier–Stokes and shape optimization. We assume
you know FEM basics (weak forms, shape functions, quadrature); everything
else is built here, one concept per chapter.

**Every chapter is a runnable, self-contained script** with a fixed
structure: *Learning outcome → Background → documented code → Expected
results (measured numbers you must reproduce) → Explore* (open questions
worth a lab-notebook entry). The scripts are the source of truth; the
website is generated from them (`python tutorials/build_site.py`, then
`mkdocs serve` at the repo root).

## Tracks

**A. Foundations (steady diffusion)**
| ch | script | outcome |
|---|---|---|
| A1 | `A_foundations/A1_mms_convergence.py` | the MMS discipline; p1→order 2, p2→order 3 |
| A2 | `A_foundations/A2_boundary_conditions.py` | strong Dirichlet vs natural Neumann; what "do-nothing" means |
| A3 | `A_foundations/A3_shifted_boundary.py` | immersed geometry: surrogate boundaries + the Taylor shift |
| A4 | `A_foundations/A4_mixed_elements.py` | mixed p1/p2 meshes, the minimum rule, where p2 pays |
| A5 | `A_foundations/A5_three_dimensions.py` | the same physics at k=3; what changes (cost!) and what doesn't |
| A6 | `A_foundations/A6_complex_geometry.py` | carving domains: channel → sphere → STL |

**B. Nonlinear**
| B1 | `B_nonlinear/B1_bratu_newton.py` | Newton's method on Bratu; quadratic convergence; MMS orders survive nonlinearity |
| B2 | `B_nonlinear/B2_bratu_3d.py` | 3-D Bratu + continuation in lambda toward the fold |

**C. Time**
| C1 | `C_time/C1_heat_bdf.py` | transient heat; BDF1 vs BDF2 measured orders 1 and 2 |
| C2 | `C_time/C2_advection_diffusion.py` | when Galerkin fails: SUPG stabilization, p1/p2 × BDF1/BDF2 |

**D. Flow**
| D1 | `D_flow/D1_ns_mms.py` | incompressible NS: saddle points, PSPG, steady MMS orders |
| D2 | `D_flow/D2_lid_driven_cavity.py` | validation against Ghia et al. (1982) |
| D3 | `D_flow/D3_cylinder.py` | immersed obstacles in flow + force extraction |

**E. Differentiable simulation**
| E0a | `E_differentiable/E0a_thinking_differentiable.py` | three paths to a gradient (FD / tangent-linear / adjoint), naked `wp.Tape` toy, Poisson dJ/dκ verified to three-way agreement at the 1e-6/1e-9 class (adjoint-vs-tape at 1e-9; vs FD at 1e-6) |
| E0b | `E_differentiable/E0b_anatomy_of_a_taped_brick.py` | open a real DiffSim kernel, do-not-differentiate list, compute-once patterns, dot-product test, probe-misfit drop >100× + the underdetermination lesson (5 probes ≠ 64 unknowns) |
| E0c | `E_differentiable/E0c_recipe_new_pde.py` | the adjoint-readiness checklist applied end-to-end: transient heat chain, store-vs-recompute, choosing J, mini Allen-Cahn phase-field gradient |
| E1 | `E_differentiable/E1_shape_optimization.py` | the adjoint loop: recover a hidden shape from probe data |

**F. Phase field (microstructure evolution; the M4 track)**
| F1 | `F_phasefield/F1_allen_cahn.py` | free energies, non-conserved gradient flow; the shrinking-circle law measured at 0.4% |
| F2 | `F_phasefield/F2_cahn_hilliard.py` | mixed (c, mu) form; conservation as a structural property (1.9e-15); spinodal decomposition |
| F3 | `F_phasefield/F3_adaptivity.py` | interface-band refinement, transfer operators, conservation by nestedness (drift exactly zero) |

The F track has a self-contained LaTeX course document
(`docs/course/m4_phasefield_tutorial.tex`) covering the theory,
the discretization as implemented, adaptivity, and the road to learned
free energies.

**P. Performance (the profiling thread)**
| P1 | `P_performance/P1_cost_model_and_scaling.py` | the group's cost-model exercise on this stack: per-stage timing vs level, measured exponents, the memory model |
| P2 | `P_performance/P2_solver_showdown.py` | direct vs host-Krylov vs fused device Krylov; counting synchronizations |

Additionally, **every chapter ends with a "Performance corner"** — one
measurement task in the spirit of `FEM_computational_cost.pdf`: time a stage,
predict its exponent before you measure, explain any disagreement. Keep a
ledger; the habit is the point.

## Suggested paths

- *Application user* (wants to run physics): A1 → A3 → A6 → C1 → D2 → D3.
- *Methods student*: all of A, then B, C, D in order; P1 after A5, P2 after D2.
- *Differentiable-simulation student*: E0a → E0b → E0c → E1, then backfill.

## Where this is going

The same brick/stepper architecture you learn here is how the coupled stacks
will be built: NS-HT (flow + heat), CH-NS (Cahn–Hilliard two-phase flow),
NS-PNP (electrokinetics), CH-NS-PNP. Each will arrive as a new track with
the same chapter contract. If you can finish track D, you will be able to
read — and extend — those solvers.
