# Adjoint-Readiness Checklist for Differentiable PDE Simulation

*Normative reference page. Specs, code reviews, and task briefs cite THIS
page. Supersedes scattered "findings-4c" citations for new documents (dev
docs keep their history). The canonical derivation and runnable examples
live in E0a, E0b, and E0c; this page is the terse, citable statement.*

---

## 1. The seven-item checklist

Apply these items to **any new PDE** before writing a line of gradient
code. Each item is developed and demonstrated in the E0 sequence; the
section citations below point at the exact explanatory prose.

### 1. What to tape

Tape only the residual kernels R(u, m) that depend on the parameters m
you differentiate with respect to. Everything algebraic-and-parameter-
dependent is taped; the solver is not (see item 4).

*Derived in E0c §1 (checklist item 1).*

### 2. What to freeze

Freeze mesh construction, octree carving, element classification, and
preflight checks (piecewise-constant in parameters), basis and quadrature
tables (inputs, not tape content), RNG seeds (reparametrize instead), and
any GP-field precomputed outside the tape (the M2 "frozen at GPs"
convention). Differentiate the relation, not the infrastructure.

*Derived in E0b §3 (do-not-differentiate list, items C and D); E0c §1
(checklist item 2).*

### 3. What to store across steps

A transient adjoint sweeps backward and needs the primal state u^n at
each step. Store the whole trajectory (full-store), or store a handful of
checkpoints and recompute forward from the nearest one (recompute). This
is the compute-vs-memory trade-off; REVOLVE / binomial checkpointing is
the production scaling path.

*Derived in E0c §2 (primal-storage inventory) and §3 (compute-vs-memory).*

### 4. Transposed-solve infrastructure

The gradient path through any linear solve is the **transposed solve** at
the converged state — not unrolled LU or Krylov steps. One LU
factorization serves both A and A^T solves (scipy: `lu.solve(rhs,
trans='T')`). For Newton loops, differentiate the converged residual via
the implicit function theorem, not the iteration.

*Derived in E0a §4 (Poisson adjoint derivation, transposed-solve step); E0b §3A
(linear solvers) and §3B (Newton loops); E0c §1 (checklist item 4).*

### 5. The verification ladder

Climb in increasing strength before trusting a gradient in an optimiser:

  (a) **Dot-product test** per operator: `<Av, w> = <v, A^T w>` (E0b §5)  
  (b) **Three-way check** per gradient: adjoint vs tape/complex-step vs
      central FD, rel err ≤ 1e-6 (E0a §4; E0c §5)  
  (c) **Transient-chain check**: full backward sweep vs FD end-to-end
      (E0c §2)

*Derived in E0b §5 (dot-product test); E0a §4, E0c §5 (three-way); E0c §2
(transient-chain).*

### 6. Nondifferentiability hazards → relaxed forms

`abs`, `min`, `max`, threshold functions, and masks are nonsmooth: their
gradient is jumpy or zero almost everywhere. Replace with relaxed forms
— softabs, logsumexp (smooth max), sigmoid gates — before optimising.

*Derived in E0c §4(i) (nonsmooth J vs logsumexp relaxation).*

### 7. The cost contract

A correct reverse-mode adjoint costs O(1) forward solves: the backward
march should be ≤ ~2.5× the forward march (one transposed solve plus one
gradient contraction per step). Measure it. If backward >> forward, you
are unrolling something you should not be.

*Derived in E0c §3 (the 2.5× contract, measured).*

---

## 2. The do-not-differentiate list

These are **algorithms that implement mathematical relations**. The
relation has a clean adjoint; the algorithm's code graph does not.

| Algorithm | Relation to differentiate | Reference |
|---|---|---|
| Linear solver (LU / Krylov / cuDSS) | A(m) u = b → transpose solve A^T λ = dJ/du, same LU | E0b §3A |
| Newton iteration | Converged residual R(u\*, m) = 0 → implicit function theorem: A^T λ = dJ/du at u\* | E0b §3B |
| Mesh / octree / classification / preflight | Frozen by design (piecewise-constant in parameters); E1's classification-freeze is the worked instance | E0b §3C |
| RNG seeds, basis tabulation | Treat as inputs, not tape content; reparametrize if randomness must flow through the gradient | E0b §3D |

---

## 3. J-design rules

The choice of objective functional J determines whether the inverse
problem is well-posed and whether the gradient is useful.

1. **Smoothness first.** Any `max`/`min`/`abs`/threshold in J produces a
   jumpy gradient. Replace with a smooth surrogate (logsumexp, softabs)
   *before* launching any optimiser. *E0c §4(i).*

2. **Signal-scale check before optimising (H4 protocol).** Measure
   `|dJ/dp|` against the finite-difference noise floor. A gradient buried
   under the floor means J barely responds to p; rescale J or p, or
   choose a more informative observable, before spending optimiser
   iterations. *E0c §4(ii).*

3. **Probe placement matters.** A probe far from the sensitivity support
   sees du/dm ≈ 0 (flat-J / vanishing-gradient geometry). Verify
   `‖dJ/dm‖` for candidate probe layouts before committing. Contrast
   informative vs flat placement should exceed 10×. *E0c §4(iii).*

4. **Identifiability and conditioning.** A correct, large, smooth gradient
   can still leave the inverse problem ill-posed if the parameterisation
   has unidentifiable directions (gauge modes). Diagnosis: the condition
   number of the observation Jacobian, not the gradient alone. Anchoring
   the basis orthogonal to gauge directions removes them by construction
   (the beyond-FH result: un-anchored cond 1.4e16 → anchored cond 1.8e1,
   a 7.6e14× gap). Tikhonov regularisation encodes a prior — state it as
   science. *E0c §4(iv); docs/dev/2026-07-14-beyond-fh-learned-thermo.md.*

5. **Multi-observable J.** A single scalar probe misfit is often
   underdetermined. A richer instrument-space observable — diverse probe
   sets, structure factors, film heights, PSF-convolved images — illuminates
   more directions of parameter space. This is the bridge to SP-1 R3.
   *E0c §4(v).*

---

## Attribution

The checklist and do-not-differentiate list are developed through the E0
tutorial sequence, which is adapted from the mathematical background for
dolfin-adjoint/pyadjoint by Patrick E. Farrell, and from the following
works:

- Farrell, Ham, Funke & Rognes (2013). *Automated Derivation of the Adjoint
  of High-Level Mathematical Programs.* SIAM J. Sci. Comput. 35(4):C369–C393.
- Mitusch, Funke & Dokken (2019). *dolfin-adjoint 2018.1: automated
  adjoints for FEniCS and Firedrake.* JOSS.

DiffSim-native sources: the findings-4c taping rules, the M1a/M2/M4
adjoint stack, the H4 signal-scale protocol, and the beyond-FH gauge
story.
