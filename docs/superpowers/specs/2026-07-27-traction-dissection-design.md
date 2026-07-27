# Traction Dissection — Design

**Date:** 2026-07-27. **Approved:** Baskar (brainstorm 2026-07-27).
**Question:** the leak-drag discriminator issued *observable-overestimates*:
`surrogate_traction` (bare σ·n on the shifted faces) reads 5.44 = **2.3×** the
variationally-consistent Nitsche reaction (2.34), while the reaction itself sits
**~30% below** literature 3.36 at r9/α=50, blockage 6.25%. Two puzzles: (a) where
in the Nitsche functional does the force live, and why is bare σ·n so high;
(b) is the reaction's literature deficit a resolution effect, a blockage effect,
or a two-sided-shell formulation systematic?

**Goal (physics-first):** term-by-term decomposition of the consistent reaction
+ a resolution/blockage campaign for the deficit; then a data-driven decision on
the project's canonical force observable.

## Components

### 1. Per-term Nitsche assembly: `sbm_vector_dirichlet_twosided(..., return_terms=False)`

- Default `False`: byte-identical current behavior.
- `True`: additionally return `terms = {"consistency": (A,b), "adjoint": (A,b),
  "penalty": (A,b), "backflow": (A,b)}` — the four constituent blocks of the
  two-sided Nitsche functional (backflow present only when an advecting field is
  supplied; otherwise a zero block with the right shape).
- **Exactness gate (binding):** `Σ_t A_t == A_monolithic` and `Σ_t b_t ==
  b_monolithic` to atol 1e-14 on both a uniform and an adaptive tiny mesh — the
  decomposition must be a PARTITION, not an approximation. If the kernel
  structure makes a clean four-way split impossible (fused terms), the split may
  be coarser (e.g. {consistency+adjoint, penalty, backflow}) — but whatever
  split ships must satisfy the partition gate and be named honestly.
- Implementation freedom: per-term assembly may run the existing kernel with
  term-masking parameters or dedicated small kernels — implementer's choice,
  gated by the partition test.

### 2. Term-split reaction in the march

- `run_flow_past(..., reaction_terms=False)`: when True (requires
  `reaction_sets`), per step also compute `f_x_t = −w₀ᵀ(A_t x − b_t)` per term
  (set 0 suffices — set-independence already established), returning
  `reaction_terms_hist` [nsteps, nterms] with a `reaction_terms_names` list.
- Gate: `Σ_t f_x_t == f_x_total` per step to 1e-12 (partition carried through).
- Also compute, once per run, the **bare-σ·n bridge**: `Cd_surr` vs the
  consistency-term reaction — the direct test of whether bare traction ≈ the
  consistency term (⇒ excess = penalty+adjoint virtual work) or not (⇒ the
  shifted-face integration itself is implicated).

### 3. GPU campaign (probe extension)

Extend `tests/gpu_leakdrag_discriminator.py` (or a sibling probe file — keep the
existing verdict machinery untouched; a new `tests/gpu_traction_dissect.py`
importing the shared pieces is acceptable and preferred for file hygiene):

| leg | config | purpose |
|---|---|---|
| D1 | r9, α=50 (baseline) | term table at the verdict point |
| D2 | r10, α=50 | resolution trend |
| D3 | r11, α=50 | resolution trend (mesh ~2× r10 near plate; budget the wall) |
| D4 | L/32 plate, r10-matched (32 cells/plate), α=50 | blockage axis |

Each: 8000 steps (D3 may need dt check at CFL — same dt=5e-4 is CFL≈0.5 at r11's
h; if unstable or Picard degrades, drop to dt=2.5e-4/16000 steps and record),
time-averaged over t≥1.5: `Cd_rxn` (+per-term), `Cd_surr`, St. Output table +
npz per leg; `TRACTDISSECT-OK` sentinel; per-leg prints flushed (learn from the
buffering saga: print with flush=True after each leg).

## Success criteria / decision rule (recorded with the results)

- **Partition gates pass** (1e-14 assembly, 1e-12 march) — else the
  decomposition is not trustworthy and the campaign halts (BLOCKED, honest).
- **Resolution verdict:** Cd_rxn(r9→r10→r11) monotone toward 3.36 with
  shrinking increments ⇒ deficit = resolution; extrapolated value recorded.
  Flat/oscillating ⇒ resolution exonerated ⇒ formulation under the microscope
  (escalation item, not hidden).
- **Blockage verdict:** direction and magnitude of Cd_rxn(L/32) vs Cd_rxn(L/16)
  recorded against the CV trend.
- **Observable decision:** if the reaction mesh-converges toward literature,
  adopt it as the canonical force observable — follow-up task (stepper method +
  runbook guidance + deprecation note on bare σ·n for validation use). If not,
  the decision escalates to Baskar with the term table in hand.

## Testing

- CPU: partition gates (both levels of the decomposition), default-path parity
  (`return_terms=False`, `reaction_terms=False` bit-for-bit), term-table
  finiteness on the tiny config.
- GPU: the four legs via `gpubox-run.sh`, sequential.

## Out of scope

3-D dissection; the α-scaling redesign (candidate follow-up); CV instrument
improvements; any change to the default solver path.
