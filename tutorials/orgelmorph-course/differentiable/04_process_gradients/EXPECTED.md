# D4 — expected results (self-check)

Running `python run.py` (defaults: 9×9 nodes, coupled Cahn–Hilliard ×
Allen–Cahn crystallization, 4 steps, BDF2, schedule
$T=[0.5,0.6,0.7,0.55]$, $b_T=0.8$, $T_{\rm ref}=0.5$) reproduces the
schedule gradient below.

- objective $J$ (final $\phi,\psi$ vs target) $= 1.647\mathrm{e}{+0}$
- crystallinity $\langle\psi\rangle$: $0.304 \to 0.308 \to 0.311 \to 0.315$

| step $n$ | $T_n$ | $dJ/dT_n$ (adjoint) | $dJ/dT_n$ (finite diff) | adj/fd |
|---|---|---|---|---|
| 0 | 0.500 | $+3.826368\mathrm{e}{-1}$ | $+3.826368\mathrm{e}{-1}$ | $2.2\mathrm{e}{-10}$ |
| 1 | 0.600 | $+2.997910\mathrm{e}{-1}$ | $+2.997910\mathrm{e}{-1}$ | $6.8\mathrm{e}{-10}$ |
| 2 | 0.700 | $+3.328497\mathrm{e}{-1}$ | $+3.328497\mathrm{e}{-1}$ | $3.3\mathrm{e}{-10}$ |
| 3 | 0.550 | $+2.822188\mathrm{e}{-1}$ | $+2.822188\mathrm{e}{-1}$ | $4.6\mathrm{e}{-10}$ |

worst adjoint-vs-FD agreement over the schedule: $6.8\mathrm{e}{-10}$.

**What must be true regardless of hardware:**

- **The whole time series matches finite differences** (`adj/fd < 1e-6`;
  measured $\sim10^{-10}$). This is the FD-verified gate — and it is *N*
  gradients checked, not one.
- **One reverse sweep returns all $N$ gradients.** Finite differences
  would need $2N$ extra forward solves; the adjoint needs one backward
  solve regardless of schedule length. That is why process design (long
  schedules) is tractable at all.
- **The gradient reflects two channels.** Temperature enters each step
  through the crystallization drive $\Delta h\,(T/T_m-1)$ *and* the Flory
  interaction $B(T)=B_0+b_T(T-T_{\rm ref})$; the adjoint's
  `temperature_gradient` sums both, which is why the FD check (which sees
  both automatically) agrees.

Signs and magnitudes depend on the objective's target; the invariant is
the step-by-step agreement with finite differences. If any step
disagrees by more than $10^{-6}$, re-read the walkthrough.
