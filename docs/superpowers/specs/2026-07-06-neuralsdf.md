# NeuralSDF (Provided-INR) Oracle — Implementation Spec (M1c/M2)

Status: DRAFT v2 (revised after GENIE — Karki, Krishnamurthy &
Ganapathysubramanian, MyPapers/2603.29860v1.pdf). Parent:
`2026-07-02-diffsim-design.md` §4. Measured motivation: the L6
shape-optimization demo (reclassification-jump finding).

## N0. Program boundaries (directive, 2026-07-06)

**DiffSim does not train INRs.** A trained INR is a PROVIDED input
(SIREN-family checkpoint with a linear last layer, C² activations).
All development and testing uses an ANALYTICAL PROXY with the same
mathematical structure; real checkpoints drop in unchanged.

## N1. The GENIE structure is the design contract

For linear-last-layer INRs, ψ(x) = h_φ(x)ᵀθ_L: with the feature
extractor h_φ frozen, any last-layer edit Δθ changes the field by
Δψ(x) = h_φ(x)ᵀΔθ — the geometry is AFFINE in the edit parameters.
GENIE's results adopted here as load-bearing:

1. **Design space = Gram deformation modes.** G = E_μ[h hᵀ] over a
   THICK-BAND sampling distribution (thin bands give unstable modes —
   the paper's Fig. 4a finding becomes our sampling rule). Top-k
   eigenvectors v_k define the design variables α ∈ R^k (k ~ 10–40):
   θ_L(α) = θ_L⁰ + Σ α_k v_k.
2. **Well-posedness boundary**: edits are well-posed iff the target
   deformation lies in the mode span — optimization in α-space stays
   inside it BY CONSTRUCTION. (The optimizer never asks the INR for a
   shape it cannot represent.)
3. **Editing does not preserve eikonality**: post-edit ψ is a level-set
   function, not an SDF. Our framework already assumes non-eikonal
   (Newton+IFT projector, S4.1); the admissibility audit (ĉ₀, Newton
   masks) is the per-epoch safety contract and RAISES on violation.

## N2. Oracle classes

```
class ProvidedINROracle(SDFOracle):     # wraps a checkpoint
    near_eikonal = False
    psi = frozen h_phi + editable linear head theta_L(alpha)
    params = [alpha]                     # torch, the ONLY design vars
    modes: (v_k) from thick-band Gram eigendecomposition (cached)

class AnalyticINRProxy(SDFOracle):      # the TEST double
    psi(x; alpha) = psi0(x) + sum_k alpha_k b_k(x)
    psi0: analytic SDF (sphere/cylinder from geometry.csg)
    b_k:  fixed analytic C-inf features (Gaussian bumps on a shell /
          low-order spherical harmonics x radial envelope) — the same
          affine structure, everything closed-form
```

Both implement classify = sign(ψ) and distance_vector via the existing
generic Newton+IFT projector (`geometry/project.py`; GridSDF already
exercises the non-eikonal path). `distance_torch` free (torch
end-to-end). No new projection or adjoint code.

## N3. Mode extraction (ProvidedINROracle only)

`extract_modes(inr, band_pts, k)` — one batched forward for H on
thick-band ∪ ambient samples (paper's sampling rule), eigh(G), cache
(v_k, spectrum). Diagnostics locked with the oracle: spectrum decay,
mode-stability check vs a second independent sample draw (the paper's
reproducibility criterion made a test).

## N4. Gradients and gates

θ-chain: α enters through step-4 geometry data d(α), n(α), corr(α)
only (S4.3); the m1a/m1c face-cotangent → torch pipeline is unchanged.
Because ψ is affine in α, dψ/dα_k = b_k(x) (proxy) or h(x)ᵀv_k (INR) —
smooth, closed-form, cheap.

Gates (all against the PROXY; measure-then-lock):
1. **Admissibility gate**: proxy at α=0 and random small α — ĉ₀,
   Newton masks, ε̂∞ report locked.
2. **Forward equivalence**: SBM Poisson MMS through the proxy at α=0
   matches the analytic-oracle solution to machine-level (same zero
   level set); perturbed-α forward runs are regression-locked.
3. **AD gate (4c ritual)**: d(probe-QoI)/dα three-way (adjoint vs
   FD-on-α vs torch dense twin at L4), all k modes, β on/off.
4. **NS composition**: Re=20 cylinder smoke with the proxy
   (psi0 = cylinder); Cd within the locked baseline tolerance.
5. **Drag gradient**: d(Cd)/dα vs FD — the hero-demo ingredient.
6. **INR drop-in** (when a checkpoint is provided): gates 1-2-4 rerun
   verbatim against the real INR; no code changes expected.

## N5. Shape optimization in mode space

Design variables α (tens). The L6 reclassification-jump finding still
applies; mode-space mitigations, in order: (i) GENIE modes are GLOBAL
smooth deformations — smaller interface displacement per unit
objective change than rigid translation (measure on the L6 benchmark);
(ii) epoch trust region with objective-accepted steps + h-continuation
(the band-study feature-resolution rule applied to optimization);
(iii) the N6 relaxations if (i)+(ii) measure insufficient.

## N6. Deferred research fork (unchanged from v1)

(a) λ-relaxed retained-set weights (assembly + conditioning audit
required) vs (b) grid-dither objective smoothing (no formulation
change, N× forwards). Decide AFTER gates 1-5, on the L6 benchmark:
first relaxation to turn the measured Cd-ascent into descent wins.

## N7. Order of work

1. AnalyticINRProxy + gates 1-2 (no training anywhere).
2. AD gate 3 (α-space three-way).
3. Gates 4-5 (compose with existing NS/drag machinery).
4. ProvidedINROracle wrapper + mode extraction (ready for the first
   real checkpoint; gate 6 awaits one).
5. N5 optimizer on the L6 benchmark; measure; N6 decision with data.
