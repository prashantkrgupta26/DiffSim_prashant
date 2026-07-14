# Selected full solutions — C3 verification exercises

*Full worked solutions for the two **verification** exercises only (the
conservative-transfer radius sweep and the constant-vs-variable BDF2
order comparison). Hints for all five exploratory questions are in
`hints.md`. Instructor-only — do not distribute before the deadline.*

---

## V1 — Conservative transfer stays exact at every feature size; naive
injection does not (Q2)

**Claim.** Restricting a fine field of sub-cell droplets to a 2×-coarser
grid, *cell averaging* preserves the mean (hence the total mass) exactly
at every droplet radius, while *nodal injection* loses mass sharply once
the droplet radius drops below the coarse-cell spacing — and its error
shrinks toward zero only once the feature grows large enough to be
resolved by every other node.

**Procedure.**
1. Run `transfer_error_sweep(radii=(0.5, 0.7, 1.0, 1.5, 2.5), n_drops=12,
   seed=3)` (the default in `adaptivity.py`) — each call restricts a
   `64x64` field of 12 random Gaussian "droplets" of the given radius
   (in fine cells) to a `32x32` grid two ways: `inj = c[::2, ::2]`
   (injection) and `avg = c.reshape(32,2,32,2).mean((1,3))` (averaging).
2. Record `inj_err` and `avg_err` (relative change in the field mean,
   i.e. relative mass error) at each radius.
3. Plot both error curves against radius on a log scale (this is exactly
   the third panel of Fig. `c3_transfer`, `figures/c3_transfer.png`).

**Expected result.** At the chapter's reference radius (`0.5` fine
cells, the smallest/hardest case and the one quoted in the main text and
`EXPECTED.md`), injection loses ≈20% of the mass while cell averaging is
exact (`0`, to solver/round-off precision). Sweeping the radius upward,
`avg_err` stays at machine-precision `0` at *every* radius — averaging
integrates each 2×2 block's content regardless of how the droplet sits
relative to the coarse grid, so it cannot miss mass by construction.
`inj_err`, in contrast, falls off sharply as the radius grows past the
coarse-cell spacing: once a droplet spans several coarse cells, sampling
every other fine node no longer risks missing it entirely, and the
injection error shrinks toward zero. The qualitative shape is a sharp
transition, not a smooth power law: injection is fine *only* once the
feature is well-resolved by the coarse grid, and fails exactly in the
sub-cell regime that a real coarsening step in dynamic AMR must handle.

**Common wrong conclusion.** "Injection is acceptable because the
average error over the whole sweep is small." This averages over radii
that were never the point — the ≈20% loss at the smallest (sub-cell)
radius is the regime dynamic AMR actually exercises when it coarsens a
region containing fine sub-cell structure. A restriction operator must
be conservative *at the size that matters*, not on average over sizes
that happen to be easy. The correct comparison is exactness *at every
size* (averaging) versus a size-dependent failure (injection) — which is
exactly why the chapter states this is "the piece a dynamic-AMR
implementation cannot use nodal injection for."

---

## V2 — Variable-coefficient BDF2 keeps order ≈2 under a varying step;
forcing constant coefficients collapses it toward 1 (Q4-adjacent, the
main-text measurement)

**Claim.** On a genuinely varying step-size history, the brick's
variable-coefficient BDF2 (coefficients rebuilt each step from the real
ratio `r = Δt/Δt_prev`) measures observed order ≈2 (order preserved),
while forcing the textbook constant-step coefficients (`r≡1`) onto that
*same* varying history measures observed order ≈1 (order collapses).

**Procedure.**
1. Both marches use the *alternating* `(dt0, dt0/2)` sequence
   (`_var_march` / `_const_march` in `adaptivity.py`), so every step sees
   a fresh, non-trivial ratio (`r=2` then `r=1/2`) — this is deliberate:
   a *fixed* `dt` would make `r≡1` the correct ratio and hide the bug
   entirely (see Q4 in `hints.md`).
2. `_var_march` lets the stepper rebuild `c_0=(1+2r)/(1+r)` and the
   history weights `[1+r, -r^2/(1+r)]` from the real `r` each step (the
   brick's default). `_const_march` keeps the same alternating physical
   step sizes but overwrites `st.dt_prev = st.dt` immediately before each
   `step()` call, forcing the stepper to see `r=1` and therefore use the
   textbook constant-step coefficients `3/2, [2, -1/2]` on a history that
   is not actually uniformly spaced.
3. `bdf2_variable_order` marches both variants at `dts=(8e-3, 4e-3,
   2e-3)` against a fine fixed-step reference (`_fixed_march` at
   `dt=2e-4`) and fits the observed order from successive halvings via
   `_order` (log2 of the error ratio).

**Expected result.** From `EXPECTED.md`: the variable-coefficient
(brick) form measures order **2.01** — order 2 preserved even though
`dt` genuinely varies every step. The tutorial-local
constant-coefficient form, run on the *identical* step sequence, measures
order **0.93** — collapsed toward 1, not toward 0: the scheme does not
become unstable or nonconvergent, it silently downgrades from a
second-order to a first-order method. This is the reproducible,
tutorial-local version of the general fact that BDF2's `3/2, [2,-1/2]`
coefficients are only consistent for a uniform step; feeding them a
step-size history they were not derived for introduces an `O(Δt)`
truncation-error term that a uniform-step analysis assumes away.

**Common wrong conclusion.** "The constant-coefficient run gives order
≈0.93, so BDF2 is broken/unstable here." No — 0.93 is not noise or
breakage, it is the textbook signature of a first-order method (a
correctly implemented BDF1 would also measure order ≈1). The honest
statement is "using constant-step coefficients on a varying step
silently downgrades the method's order by one," which is exactly why the
brick rebuilds the coefficients from the real ratio every step rather
than assuming `r=1`. A second common error is treating this as a purely
academic point: it is not — any adaptive controller that grows or shrinks
`Δt` (§"Temporal adaptivity" in `c3.tex`) exercises exactly this varying-
step regime every time it changes the step size, which is why the LTE
controller and the variable-coefficient BDF2 must be treated as a
matched pair.
