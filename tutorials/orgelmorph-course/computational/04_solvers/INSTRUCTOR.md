# Instructor companion — Computational C4 The solver ecosystem

*Teaching notes for the instructor. Student-facing material is in
`README.md` and the course document chapter `c4.tex`. Grading is in
`grading_rubric.md`; hints and selected solutions in `solutions/`.*

## What this chapter is really teaching

The mechanics of `splu` vs. cuDSS vs. `blockch` are a vehicle for the
course's central habit applied to *software*, not just physics: **measure
the claim, do not trust a plausible-looking number**. Three ideas do the
teaching:

1. **Solver choice is a correctness decision before a speed one.** The
   mixed `(c, μ)` Cahn–Hilliard Jacobian is a *saddle* matrix that is
   genuinely **indefinite** once `f″=3c²−1<0` in the spinodal band. cuDSS
   factorizes without partial pivoting; on this matrix that is not a
   performance trade-off, it is a *wrong answer*. Marching the real
   20-step spinodal proves it: `poly + cudss` blows the field up to
   `[−552, 542]` (a ~500× blow-up) while `poly + splu` stays `[−1.03,
   1.01]`. cuDSS is not "always fastest and safe" — on the raw indefinite
   CH block it is **fast and wrong**. This is the central lesson of the
   chapter, and it should survive every other detail a student forgets.
2. **A tiny residual is not evidence of correctness.** `solver_correctness`
   captures one Newton iterate and cuDSS's residual there is ~5e-13 —
   smaller than `splu`'s ~1e-12 and `blockch`'s ~4e-10 — yet the 20-step
   march with cuDSS diverges. A small residual only says the *computed*
   `x̂` nearly satisfies `Ax̂≈b`; it says nothing about `‖x−x̂‖` when `A` is
   ill-conditioned. This is deliberately the trap of the chapter: students
   must never report the one-iterate residual as if it were the
   verification.
3. **Use the supported API, not a monkeypatch.** The one-iterate capture
   goes through `CahnHilliardStepper(..., capture_system=True)` →
   `st.last_system` — a supported flag, not a `solve_linear` monkeypatch.
   `recommend_solver` is the supported decision-support entry point.
   Students who hack around either (e.g. patching internals to force a
   solver, or hand-picking a solver without going through
   `recommend_solver`'s rationale) have skipped the actual lesson, which
   is *why* a choice is correct, not just *that* a run finished.

## Common student misconceptions & typical incorrect conclusions

- **"cuDSS is a GPU direct solver, so it is always fastest and safe."**
  This is the misconception the whole chapter exists to correct. cuDSS
  **diverges** on the indefinite polynomial CH saddle (measured:
  `[−552, 542]`) because it does no partial pivoting. It only survives
  Flory–Huggins because that energy's curvature `f″=A(1/c+1/(1−c))≥4A`
  stays positive — an energy-specific accident, not a property of cuDSS.
  Zero credit if a student's report claims cuDSS is safe on the raw CH
  block.
- **"The residual was tiny, so the solve was correct."** cuDSS's
  one-iterate residual (~5e-13) is smaller than `splu`'s (~1e-12) — and
  cuDSS is the one that diverges. Push students to state *why*: residual
  small ⇏ error small when the matrix is ill-conditioned/indefinite and
  unpivoted. Zero credit if a student's report cites the residual as
  evidence of correctness.
- **"Faster means better."** The 2-D crossover figure shows cuDSS
  overtaking `splu` above ~10⁴ dofs (2–6× at 33k dofs) — but that speed is
  unusable for CH because cuDSS is disqualified on correctness grounds
  first. Students who report the crossover without the disqualification
  have missed the point of the figure.
- **"blockch is just another solver option."** `blockch`/`blockch_dev` is
  qualitatively different: it never forms the monolithic indefinite
  factorization at all — it splits the saddle into SPD sub-solves (mass,
  `W1`, `W2`) inside an FGMRES iteration. That is *why* it is safe at
  scale where a direct factorization (pivoted or not) becomes too
  expensive or, for cuDSS, wrong.
- **"I can just monkeypatch `solve_linear` to grab the system."** The
  supported way to capture a linear system for inspection is
  `capture_system=True` → `st.last_system`. Monkeypatching internals is
  fragile, hides the actual API the production code offers, and is
  explicitly called out as unsupported in `c4.tex`/`solvers.py`.
- **"The 3-D numbers were reproduced this run."** They were not — they
  are cited from `docs/dev/2026-07-13-m5-device-assembly.md` and
  `docs/dev/2026-07-13-blockch-mpf.md` because they take minutes per step.
  A report that presents the 3-D table as freshly measured is
  misrepresenting provenance.

## Expected runtime ranges (RTX 6000 Ada; scale by card)

| mode | what | wall | notes |
|---|---|---|---|
| live 2-D | `march_divergence` (20 steps × 2 energies × 2 solvers) + `solver_correctness` + `benchmark_solvers_2d` (levels 5–7) | comparable to Physics P1's quick mode (**≈75 s**, measured) | mostly Warp compile + Python start-up on first run |
| cited 3-D | `CITED_3D` table (32³, slab64, slab64z32) | **minutes per step** (dev-note measurement, NOT re-run here) | source: `docs/dev/2026-07-13-m5-device-assembly.md` (D3), `docs/dev/2026-07-13-blockch-mpf.md` (B2–B5) |

First run of a session pays a one-time Warp kernel-compile cost, as in
every other chapter; subsequent runs are faster. Do not let students
re-run the cited 3-D cases expecting a quick number — that is precisely
why they are cited, not live.

## Common CUDA / solver errors students hit

- **Forcing `--solver cudss` on the polynomial CH saddle and watching it
  blow up.** This is **expected and teachable**, not a bug report. If a
  student files this as "the code is broken," redirect them to the
  mechanism: `f″<0` in the spinodal band makes the block indefinite,
  cuDSS does no pivoting, so the factorization is wrong and the error
  compounds over the Newton/time iterations until the field escapes the
  physical band.
- **`cuDSS unavailable` (nvmath not installed).** `cudss_available()`
  probes gracefully; a student without `nvmath` sees `note: "cuDSS
  unavailable"` rows rather than a crash. That is correct behavior, not a
  failure to diagnose.
- **`out of memory` at 3-D scale.** Expected once a student tries to
  actually run a 3-D case instead of trusting the cited numbers: cuDSS's
  fill hits the measured ~811k-dof ceiling on a 48 GB card (a >16-minute
  factorization that never returns). This *is* the lesson of the cited
  table, reproduced the hard way.
- **Reporting the one-iterate residual instead of the marched solution
  as the correctness check.** Not a CUDA error, but the single most
  common analysis error in this chapter — see misconceptions above.

## Discussion prompts

- Why does an *indefinite* saddle matrix specifically defeat a solver
  that skips partial pivoting, when the same solver is fine on an SPD
  matrix?
- If cuDSS is fast and wrong on the raw CH block, and `blockch` is safe
  but iterative, what does that say about the general trade space between
  "one fast wrong factorization" and "several small definitely-correct
  factorizations inside an outer loop"?
- `recommend_solver` never recommends cuDSS on the raw CH block, in any
  regime. Is there a regime (bigger card, different physics) where that
  could change, or is the indefiniteness argument regime-independent?
- The one-iterate residual trap generalizes beyond Cahn–Hilliard: name
  another ill-conditioned or indefinite system elsewhere in the course
  where "the residual looked fine" would be exactly the wrong thing to
  trust.
