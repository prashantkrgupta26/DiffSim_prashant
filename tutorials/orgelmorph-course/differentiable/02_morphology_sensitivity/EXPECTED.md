# D2 — expected results (self-check)

Running `python run.py` (defaults: 9×9 nodes, 6 steps, BDF1, Flory–Huggins
$A=1$, $\chi=2.5$, $\kappa=0.01$, $dt=0.005$, interior initial data)
reproduces the morphology sensitivity below.

| quantity | value |
|---|---|
| demixing amplitude $J=\tfrac12\lVert c_N-\bar c\rVert^2$ | $3.095\mathrm{e}{-1}$ |
| $dJ/d\chi$ (adjoint) | $+7.432587\mathrm{e}{-1}$ |
| $dJ/d\kappa$ (adjoint) | $-6.690827\mathrm{e}{+0}$ |
| worst adjoint-vs-FD agreement | $1.0\mathrm{e}{-9}$ |

**What must be true regardless of hardware:**

- **The adjoint sensitivities match finite differences** to the FD floor
  (`adj/fd < 1e-6`; measured $\sim10^{-9}$). This is the FD-verified gate.
- **$dJ/d\chi > 0$.** Above the spinodal ($\chi>2A=2$) a larger Flory
  interaction drives the blend *further apart*, so the demixing amplitude
  grows — raising $\chi$ sharpens the morphology.
- **$dJ/d\kappa < 0$.** The gradient penalty $\tfrac{\kappa}{2}|\nabla
  c|^2$ opposes composition variation; a larger $\kappa$ thickens/blurs
  interfaces and *lowers* the demixing amplitude.
- **The gradient is the slope.** In `d2_slope.png`, the adjoint tangents
  $dJ/d\chi$ lie along the sampled response curve $J(\chi)$ — the whole
  meaning of the derivative, seen directly.

If the signs flip, or the adjoint disagrees with FD by more than
$10^{-6}$, re-read the walkthrough — the setup differs from the tutorial
(most likely the rollout ran long enough to push $c$ out of $(0,1)$,
where the Flory–Huggins logarithm is no longer smooth).
