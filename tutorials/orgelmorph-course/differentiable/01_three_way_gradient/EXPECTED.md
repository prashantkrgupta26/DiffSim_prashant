# D1 — expected results (self-check)

Running `python run.py` (defaults: 9×9 nodes, 3 steps, BDF1, Flory–Huggins
with $A=1,\ B=2.5$, interior initial data) reproduces the three-way
gradient check below. The gradients are physics — they match the merged
adjoint gate `tests/test_phasefield_adjoint.py::test_g1` to every digit.

| parameter | adjoint | autograd twin | finite diff | adj/twin | adj/fd |
|---|---|---|---|---|---|
| $M$ (mobility)     | $+3.076615\mathrm{e}{-1}$ | $+3.076615\mathrm{e}{-1}$ | $+3.076615\mathrm{e}{-1}$ | $1.8\mathrm{e}{-16}$ | $3.3\mathrm{e}{-10}$ |
| $\kappa$ (gradient)| $-7.356749\mathrm{e}{+0}$ | $-7.356749\mathrm{e}{+0}$ | $-7.356749\mathrm{e}{+0}$ | $1.2\mathrm{e}{-16}$ | $1.5\mathrm{e}{-9}$ |
| $\chi=B$ (Flory)   | $+8.195832\mathrm{e}{-1}$ | $+8.195832\mathrm{e}{-1}$ | $+8.195832\mathrm{e}{-1}$ | $1.3\mathrm{e}{-16}$ | $4.1\mathrm{e}{-11}$ |
| $A$ (entropic)     | $-1.667729\mathrm{e}{+0}$ | $-1.667729\mathrm{e}{+0}$ | $-1.667729\mathrm{e}{+0}$ | $0$ | $3.6\mathrm{e}{-11}$ |

- objective $J = \tfrac12\lVert c_N - \tfrac12\rVert^2 = 3.19\mathrm{e}{-1}$
- worst adj-vs-twin agreement $= 1.8\mathrm{e}{-16}$
- worst adj-vs-FD agreement $= 1.5\mathrm{e}{-9}$

**What must be true regardless of hardware:**

- **The three methods agree.** Adjoint and autograd twin agree to
  ~$10^{-16}$ (they encode the *same* discrete math, differentiated by two
  independent engines — hand calculus vs. `torch.autograd`). Adjoint and
  central finite differences agree to ~$10^{-9}$ (the FD truncation +
  roundoff floor). A disagreement means a bug in the gradient — the whole
  point of a three-way check.
- **The signs are physical.** $dJ/d\chi>0$: raising the interaction pushes
  the field *away* from the uniform $c=\tfrac12$, so $J=\tfrac12\lVert
  c-\tfrac12\rVert^2$ grows. $dJ/dM>0$: a faster mobility gets there
  sooner in the same wall-clock of steps.
- **The FD-step valley** (see `d1_fd_valley.png`): the error of
  finite differences is a V-shaped function of the step $\epsilon$
  (truncation on the right, roundoff on the left); its floor is far above
  the adjoint, which is exact and needs no step to tune.

If any pair disagrees by more than the tolerances (`adj/twin < 1e-10`,
`adj/fd < 1e-6`), the self-check aborts — re-read the walkthrough.
