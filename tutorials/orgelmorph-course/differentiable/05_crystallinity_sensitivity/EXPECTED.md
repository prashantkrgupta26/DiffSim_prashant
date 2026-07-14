# D5 — expected results (self-check)

Running `python run.py` (defaults: 9×9 nodes, coupled Cahn–Hilliard ×
Allen–Cahn crystallization, 3 steps, BDF1; base $\Delta h=1$, $T_m=2$,
$\Delta\sigma=1$, $\varepsilon^2=0.02$, $L=1$, $T=0.5$; target
$\psi^\star=0.4$) reproduces the crystallinity sensitivities below.

- metric $J=\tfrac12\lVert\psi_N-0.4\rVert^2 = 4.532\mathrm{e}{-1}$
- mean crystallinity $\langle\psi\rangle = 0.3122$

| parameter | adjoint $dJ/dp$ | finite diff | adj/fd |
|---|---|---|---|
| $\Delta h$ (latent heat)     | $-8.325099\mathrm{e}{-2}$ | $-8.325099\mathrm{e}{-2}$ | $3.2\mathrm{e}{-10}$ |
| $T_m$ (melting temperature)  | $-1.387517\mathrm{e}{-2}$ | $-1.387517\mathrm{e}{-2}$ | $9.9\mathrm{e}{-10}$ |
| $\Delta\sigma$ (barrier)     | $+1.639820\mathrm{e}{-2}$ | $+1.639820\mathrm{e}{-2}$ | $9.8\mathrm{e}{-10}$ |
| $\varepsilon^2$ (stiffness)  | $-1.672853\mathrm{e}{-1}$ | $-1.672853\mathrm{e}{-1}$ | $4.7\mathrm{e}{-11}$ |
| $L$ (Allen–Cahn kinetics)    | $-7.191302\mathrm{e}{-2}$ | $-7.191302\mathrm{e}{-2}$ | $4.5\mathrm{e}{-10}$ |

worst adjoint-vs-FD agreement: $9.9\mathrm{e}{-10}$.

**What must be true regardless of hardware:**

- **Every sensitivity matches finite differences** (`adj/fd < 1e-6`;
  measured $\sim10^{-10}$) — the FD-verified gate, over all five
  crystallization parameters, from one reverse sweep.
- **The signs are physical.** With $\langle\psi\rangle\approx0.31$ *below*
  the target $0.4$, any parameter that *raises* crystallinity moves $\psi$
  toward the target and *lowers* $J$: raising the latent heat $\Delta h$
  deepens the Turnbull drive, so $dJ/d\Delta h<0$; raising the barrier
  $\Delta\sigma$ opposes ordering, so $dJ/d\Delta\sigma>0$.
- **$\varepsilon^2$ and $\Delta h$ are the strongest knobs** at this
  setting — the interface stiffness and the driving force dominate the
  crystallinity response.

**On noise.** Nucleation is driven by a *stochastic* term (Physics P8).
The adjoint differentiates the deterministic crystallization *drift*; a
sensitivity to the *noise amplitude* is a pathwise/score-function
derivative of an expectation — a different, documented frontier, not part
of this deterministic gate. The tutorial says so and stops there rather
than fake a stochastic gradient.
